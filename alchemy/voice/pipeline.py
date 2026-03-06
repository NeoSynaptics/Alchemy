"""Voice pipeline — state machine orchestrator.

IDLE → LISTENING → RECORDING → PROCESSING → SPEAKING → IDLE (loop)

Runs as a background asyncio task. Feeds transcribed speech into
SmartRouter, streams response to TTS.

Supports two modes:
- Single-GPU: VRAM swaps between Whisper/Qwen3/Fish Speech (buffered LLM).
- Dual-GPU / no manager: all models resident, direct streaming.
"""

from __future__ import annotations

import asyncio
import logging
from enum import Enum
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from alchemy.voice.models.schemas import ChatRequest
from alchemy.voice.router.router import SmartRouter
from alchemy.voice.audio import AudioStream
from alchemy.voice.listener import SpeechListener
from alchemy.voice.stt import WhisperSTT
from alchemy.voice.tts import FishSpeechTTS, PiperTTS
from alchemy.voice.wake_word import WakeWordDetector

if TYPE_CHECKING:
    from alchemy.voice.fish_speech import FishSpeechProcess
    from alchemy.voice.vram_manager import VRAMManager

logger = logging.getLogger(__name__)


class PipelineState(str, Enum):
    IDLE = "idle"
    LISTENING = "listening"
    RECORDING = "recording"
    PROCESSING = "processing"
    SPEAKING = "speaking"


class VoicePipeline:
    """Orchestrates the full voice loop: wake → record → transcribe → route → speak."""

    def __init__(
        self,
        router: SmartRouter,
        wake_word: WakeWordDetector,
        listener: SpeechListener,
        stt: WhisperSTT,
        tts: PiperTTS | FishSpeechTTS,
        audio_stream: AudioStream,
        vram: VRAMManager | None = None,
        fish_process: FishSpeechProcess | None = None,
    ) -> None:
        self._router = router
        self._wake_word = wake_word
        self._listener = listener
        self._stt = stt
        self._tts = tts
        self._audio = audio_stream
        self._vram = vram
        self._fish_process = fish_process

        self._state = PipelineState.IDLE
        self._task: asyncio.Task | None = None
        self._conversation_id: UUID = uuid4()

    @property
    def state(self) -> PipelineState:
        return self._state

    @property
    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    @property
    def conversation_id(self) -> UUID:
        return self._conversation_id

    async def start(self) -> None:
        """Start the voice pipeline as a background task."""
        if self.is_running:
            logger.warning("Voice pipeline already running")
            return

        self._conversation_id = uuid4()
        await self._audio.start()
        self._task = asyncio.create_task(self._loop(), name="voice-pipeline")
        logger.info("Voice pipeline started (conversation=%s)", self._conversation_id)

    async def stop(self) -> None:
        """Gracefully stop the voice pipeline."""
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        await self._audio.stop()
        self._state = PipelineState.IDLE
        self._task = None
        logger.info("Voice pipeline stopped")

    async def _loop(self) -> None:
        """Main pipeline loop. Runs until cancelled."""
        try:
            while True:
                await self._cycle()
        except asyncio.CancelledError:
            logger.info("Voice pipeline cancelled")
            raise
        except Exception:
            logger.exception("Voice pipeline error")
            self._state = PipelineState.IDLE

    async def _cycle(self) -> None:
        """One full cycle: listen → record → transcribe → route → speak."""
        # 1. Wait for wake word (CPU, no VRAM needed)
        self._state = PipelineState.LISTENING
        await self._wake_word.listen(self._audio)

        # 2. Record speech (CPU, no VRAM needed)
        self._state = PipelineState.RECORDING
        audio_bytes = await self._listener.record(self._audio)

        if len(audio_bytes) < 1600:  # Less than 0.1s of audio — ignore
            logger.debug("Recording too short, ignoring")
            return

        # 3. Transcribe — may need VRAM swap
        self._state = PipelineState.PROCESSING

        if self._vram and self._vram.is_active:
            await self._vram.ensure_stt(self._stt)

        text = await self._stt.transcribe(audio_bytes)

        if self._vram and self._vram.is_active:
            await self._vram.release_stt(self._stt)

        if not text.strip():
            logger.debug("Empty transcription, ignoring")
            if self._vram and self._vram.is_active:
                await self._vram.restore_idle()
            return

        logger.info("User said: %s", text)

        # 4. Route through SmartRouter
        request = ChatRequest(
            message=text,
            conversation_id=self._conversation_id,
            source="voice",
        )

        if self._vram and self._vram.is_active:
            # --- Single-GPU path: buffer full LLM response, then swap to TTS ---
            await self._vram.ensure_llm()

            full_response = ""
            async for chunk in self._router.route_stream(request):
                if chunk.content:
                    full_response += chunk.content

            await self._vram.release_llm()
            logger.info("LLM response (%d chars): %s", len(full_response), full_response[:200])

            # 5. Speak — start Fish Speech, play, stop
            self._state = PipelineState.SPEAKING
            if self._fish_process:
                await self._vram.ensure_tts(self._fish_process)

            await self._tts.speak(full_response, self._audio)

            if self._fish_process:
                await self._vram.release_tts(self._fish_process)

            await self._vram.restore_idle()
        else:
            # --- Dual-GPU / no manager: stream directly (original behavior) ---
            self._state = PipelineState.SPEAKING

            async def _response_chunks():
                async for chunk in self._router.route_stream(request):
                    if chunk.content and not chunk.done:
                        yield chunk.content
                    elif chunk.content and chunk.done:
                        yield chunk.content

            await self._tts.speak_streamed(_response_chunks(), self._audio)

        self._wake_word.reset()
