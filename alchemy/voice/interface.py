"""Public interface for the AlchemyVoice subsystem.

This is the ONLY module external code should import from alchemy.voice.
All model details, pipeline internals, and provider logic are hidden behind
this interface. Future GUI settings pages should use VoiceSystem to read/change
configuration without knowing about Whisper, Fish Speech, VRAM swaps, etc.

Usage:
    from alchemy.voice.interface import VoiceSystem, VoiceMode

    voice = VoiceSystem(settings)
    await voice.start()
    voice.set_mode(VoiceMode.CONVERSATION)
    await voice.stop()
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from uuid import UUID

logger = logging.getLogger(__name__)


class VoiceMode(str, Enum):
    """User-facing voice modes. Controls behavior, not models."""
    CONVERSATION = "conversation"   # Normal back-and-forth chat
    COMMAND = "command"             # Short imperative commands, less chatty
    DICTATION = "dictation"         # Transcribe only, no response
    MUTED = "muted"                 # Pipeline paused, no wake word listening


class TTSEngine(str, Enum):
    """Available TTS backends (hidden from GUI — exposed as voice "profiles")."""
    PIPER = "piper"
    FISH_SPEECH = "fish"
    KOKORO = "kokoro"


class VoiceStatus:
    """Snapshot of voice system state — safe to serialize for GUI."""

    def __init__(
        self,
        running: bool,
        mode: VoiceMode,
        pipeline_state: str,
        tts_engine: str,
        wake_word: str,
        conversation_id: str | None = None,
    ):
        self.running = running
        self.mode = mode
        self.pipeline_state = pipeline_state
        self.tts_engine = tts_engine
        self.wake_word = wake_word
        self.conversation_id = conversation_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "running": self.running,
            "mode": self.mode.value,
            "pipeline_state": self.pipeline_state,
            "tts_engine": self.tts_engine,
            "wake_word": self.wake_word,
            "conversation_id": self.conversation_id,
        }


class VoiceSystem:
    """High-level interface to the voice subsystem.

    Hides all model loading, VRAM management, STT/TTS engine details.
    Exposes only what a GUI settings page or API endpoint needs:
    - Start/stop the voice pipeline
    - Change voice mode
    - Query current status
    - Update user-facing settings (wake word, TTS voice, etc.)

    Model internals (Whisper size, Fish Speech params, VRAM swap logic)
    are handled internally and NOT exposed through this interface.
    """

    def __init__(self, settings: Any) -> None:
        self._settings = settings
        self._pipeline = None  # Lazy — built on start()
        self._router = None
        self._mode = VoiceMode.CONVERSATION
        self._tts_engine = TTSEngine(getattr(settings, "tts_engine", "piper"))
        self._started = False

    @property
    def is_running(self) -> bool:
        return self._pipeline is not None and self._pipeline.is_running

    @property
    def mode(self) -> VoiceMode:
        return self._mode

    @property
    def conversation_id(self) -> UUID | None:
        if self._pipeline:
            return self._pipeline.conversation_id
        return None

    def status(self) -> VoiceStatus:
        """Get current voice system status (safe for GUI/API serialization)."""
        pipeline_state = "idle"
        conv_id = None
        if self._pipeline:
            pipeline_state = self._pipeline.state.value
            conv_id = str(self._pipeline.conversation_id)
        return VoiceStatus(
            running=self.is_running,
            mode=self._mode,
            pipeline_state=pipeline_state,
            tts_engine=self._tts_engine.value,
            wake_word=getattr(self._settings, "wake_word", "hey_neo"),
            conversation_id=conv_id,
        )

    def set_mode(self, mode: VoiceMode) -> None:
        """Change voice behavior mode. Takes effect immediately."""
        old = self._mode
        self._mode = mode
        logger.info("Voice mode changed: %s → %s", old.value, mode.value)

    async def start(self, router=None) -> None:
        """Start the voice pipeline.

        Args:
            router: SmartRouter instance (injected by server lifespan).
        """
        if self._started:
            logger.warning("Voice system already started")
            return

        if not getattr(self._settings, "voice_enabled", True):
            logger.info("Voice disabled by settings")
            return

        self._router = router
        try:
            self._pipeline = await self._build_pipeline()
            await self._pipeline.start()
            self._started = True
            logger.info("Voice system started (mode=%s, tts=%s)",
                        self._mode.value, self._tts_engine.value)
        except ImportError:
            logger.warning("Voice dependencies not installed — run: pip install -e '.[voice]'")
        except Exception:
            logger.exception("Failed to start voice system")

    async def stop(self) -> None:
        """Stop the voice pipeline and release resources."""
        if self._pipeline and self._pipeline.is_running:
            await self._pipeline.stop()
        self._started = False
        self._pipeline = None
        logger.info("Voice system stopped")

    async def _build_pipeline(self):
        """Construct the voice pipeline from settings. Internal only."""
        from alchemy.voice.audio import AudioStream
        from alchemy.voice.listener import SpeechListener
        from alchemy.voice.pipeline import VoicePipeline
        from alchemy.voice.stt import WhisperSTT
        from alchemy.voice.wake_word import WakeWordDetector

        s = self._settings

        audio_stream = AudioStream()
        stt = WhisperSTT(
            model_size=getattr(s, "whisper_model", "large-v3"),
            device=getattr(s, "whisper_device", "cuda"),
        )

        # TTS engine selection — hidden from external callers
        fish_process = None
        if self._tts_engine == TTSEngine.FISH_SPEECH:
            from alchemy.voice.fish_speech import FishSpeechProcess
            from alchemy.voice.tts import FishSpeechTTS

            fish_process = FishSpeechProcess(
                port=s.fish_speech_port,
                checkpoint_path=s.fish_speech_checkpoint,
                decoder_path=getattr(s, "fish_speech_decoder_path", "") or None,
                decoder_config=s.fish_speech_decoder_config,
                listen_host=s.fish_speech_host,
                startup_timeout=s.fish_speech_startup_timeout,
                compile=s.fish_speech_compile,
                python_exe=getattr(s, "fish_speech_python_exe", "") or None,
                working_dir=getattr(s, "fish_speech_dir", "") or None,
            )
            tts = FishSpeechTTS(
                fish_process=fish_process,
                sample_rate=s.fish_speech_sample_rate,
                temperature=s.fish_speech_temperature,
                top_p=s.fish_speech_top_p,
                repetition_penalty=s.fish_speech_repetition_penalty,
                max_new_tokens=s.fish_speech_max_new_tokens,
                reference_id=getattr(s, "fish_speech_reference_id", "") or None,
                chunk_length=s.fish_speech_chunk_length,
            )
        elif self._tts_engine == TTSEngine.KOKORO:
            from alchemy.voice.tts import KokoroTTS

            tts = KokoroTTS(
                base_url=f"http://{s.kokoro_host}:{s.kokoro_port}",
                voice=s.kokoro_voice,
            )
        else:
            from alchemy.voice.tts import PiperTTS

            tts = PiperTTS(model=s.piper_model)

        # VRAM manager — internal optimization, never exposed
        vram_mgr = None
        from alchemy.voice.vram_manager import GPUMode, VRAMManager

        gpu_mode = GPUMode(getattr(s, "gpu_mode", "single"))
        if gpu_mode == GPUMode.SINGLE:
            vram_mgr = VRAMManager(
                mode=gpu_mode,
                ollama_host=s.ollama_host,
                gpu_model=s.gpu_model,
                keep_alive=s.gpu_model_keep_alive,
            )
            await vram_mgr.start()
        else:
            if fish_process:
                logger.info("Dual-GPU: pre-starting Fish Speech...")
                await fish_process.start()
            logger.info("Dual-GPU: pre-loading Whisper...")
            await stt.load()

        return VoicePipeline(
            router=self._router,
            wake_word=WakeWordDetector(
                model_name=getattr(s, "wake_word", "hey_neo"),
                threshold=getattr(s, "voice_wake_threshold", 0.5),
            ),
            listener=SpeechListener(
                vad_aggressiveness=getattr(s, "voice_vad_aggressiveness", 2),
                silence_ms=getattr(s, "voice_silence_ms", 800),
            ),
            stt=stt,
            tts=tts,
            audio_stream=audio_stream,
            vram=vram_mgr,
            fish_process=fish_process,
        )
