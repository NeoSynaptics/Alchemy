"""Text-to-speech — Piper TTS, Fish Speech 1.5, Kokoro-82M.

PiperTTS: CPU-only (~50MB), robotic but zero-GPU.
FishSpeechTTS: GPU via subprocess HTTP server, near-human quality.
KokoroTTS: Kokoro-82M via Kokoro-FastAPI, ultra-fast (~0.3s/sentence), ~200MB VRAM.

All share the same interface: load, speak, speak_streamed, _synthesize, is_loaded.
"""

from __future__ import annotations

import asyncio
import io
import logging
import re
import wave
from typing import TYPE_CHECKING, AsyncIterator

import numpy as np

from alchemy.voice.audio import AudioStream

if TYPE_CHECKING:
    from alchemy.voice.fish_speech import FishSpeechProcess

logger = logging.getLogger(__name__)

_SENTENCE_END = re.compile(r"[.!?\n]")


class PiperTTS:
    """Synthesizes text to speech using Piper (CPU, local)."""

    def __init__(self, model: str = "en_US-lessac-medium") -> None:
        self._model_name = model
        self._voice = None
        self._audio_stream: AudioStream | None = None

    async def load(self, audio_stream: AudioStream | None = None) -> None:
        """Load the Piper voice model."""
        from pathlib import Path

        from piper import PiperVoice

        self._audio_stream = audio_stream

        # Resolve model path — accept bare name or full path
        model_path = self._model_name
        if not Path(model_path).exists():
            # Look in project models directory
            project_models = Path(__file__).resolve().parents[2] / "models" / "piper"
            candidate = project_models / f"{self._model_name}.onnx"
            if candidate.exists():
                model_path = str(candidate)
            else:
                raise FileNotFoundError(
                    f"Piper model not found: {self._model_name}. "
                    f"Expected at {candidate}. Download with: "
                    f"python -c \"from piper.download_voices import download_voice; "
                    f"from pathlib import Path; download_voice('{self._model_name}', Path('{project_models}'))\""
                )

        loop = asyncio.get_event_loop()
        self._voice = await loop.run_in_executor(
            None,
            lambda: PiperVoice.load(model_path, use_cuda=False),
        )
        logger.info("Piper TTS loaded: %s (rate=%d)", model_path, self._voice.config.sample_rate)

    async def speak(self, text: str, audio_stream: AudioStream | None = None) -> None:
        """Synthesize full text and play it."""
        if not self._voice:
            await self.load()

        stream = audio_stream or self._audio_stream
        audio = await self._synthesize(text)
        if audio is not None and stream:
            stream.play_audio(audio, sample_rate=self._voice.config.sample_rate)

    async def speak_streamed(
        self,
        chunks: AsyncIterator[str],
        audio_stream: AudioStream | None = None,
        flush_timeout: float = 0.5,
    ) -> None:
        """Buffer streaming text into sentences, synthesize and play each.

        Buffers text until a sentence boundary (. ! ? newline), then
        synthesizes and plays that sentence while continuing to buffer.
        If no new chunk arrives within flush_timeout seconds, any buffered
        text is spoken immediately to avoid perceived latency.

        Uses an asyncio.Queue so the generator runs independently and
        timeouts don't cancel the underlying streaming HTTP request.
        """
        if not self._voice:
            await self.load()

        stream = audio_stream or self._audio_stream
        buffer = ""

        # Decouple the generator from timeout-based reads via a queue.
        _DONE = object()
        queue: asyncio.Queue = asyncio.Queue()

        async def _fill_queue():
            try:
                async for text in chunks:
                    await queue.put(text)
            except Exception:
                logger.exception("Error reading response chunks")
            finally:
                await queue.put(_DONE)

        fill_task = asyncio.create_task(_fill_queue())

        try:
            while True:
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=flush_timeout)
                except asyncio.TimeoutError:
                    if buffer.strip():
                        audio = await self._synthesize(buffer.strip())
                        if audio is not None and stream:
                            stream.play_audio(audio, sample_rate=self._voice.config.sample_rate)
                        buffer = ""
                    continue

                if item is _DONE:
                    break

                buffer += item

                # Check for sentence boundaries
                while True:
                    match = _SENTENCE_END.search(buffer)
                    if not match:
                        break

                    end_idx = match.end()
                    sentence = buffer[:end_idx].strip()
                    buffer = buffer[end_idx:]

                    if sentence:
                        audio = await self._synthesize(sentence)
                        if audio is not None and stream:
                            stream.play_audio(audio, sample_rate=self._voice.config.sample_rate)

            # Flush remaining buffer
            remaining = buffer.strip()
            if remaining:
                audio = await self._synthesize(remaining)
                if audio is not None and stream:
                    stream.play_audio(audio, sample_rate=self._voice.config.sample_rate)
        finally:
            if not fill_task.done():
                fill_task.cancel()
                try:
                    await fill_task
                except asyncio.CancelledError:
                    pass

    async def _synthesize(self, text: str) -> np.ndarray | None:
        """Convert text to audio array via Piper."""
        if not self._voice or not text.strip():
            return None

        loop = asyncio.get_event_loop()

        def _do_synth():
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, "wb") as wav_out:
                self._voice.synthesize_wav(text, wav_out)
            wav_buffer.seek(0)
            with wave.open(wav_buffer, "rb") as wav_in:
                frames = wav_in.readframes(wav_in.getnframes())
                return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

        return await loop.run_in_executor(None, _do_synth)

    @property
    def is_loaded(self) -> bool:
        return self._voice is not None


class FishSpeechTTS:
    """Synthesizes text to speech using Fish Speech 1.5 (GPU, subprocess HTTP).

    Same interface as PiperTTS for drop-in replacement. The Fish Speech process
    lifecycle is managed externally (by VRAMManager or kept resident in dual-GPU).
    """

    def __init__(
        self,
        fish_process: FishSpeechProcess,
        sample_rate: int = 44100,
        format: str = "wav",
        temperature: float = 0.8,
        top_p: float = 0.8,
        repetition_penalty: float = 1.1,
        max_new_tokens: int = 1024,
        reference_id: str | None = None,
        chunk_length: int = 200,
    ) -> None:
        self._fish = fish_process
        self._sample_rate = sample_rate
        self._format = format
        self._temperature = temperature
        self._top_p = top_p
        self._repetition_penalty = repetition_penalty
        self._max_new_tokens = max_new_tokens
        self._reference_id = reference_id
        self._chunk_length = chunk_length
        self._client = None  # httpx.AsyncClient, created in load()

    async def load(self, audio_stream: AudioStream | None = None) -> None:
        """Initialize HTTP client for Fish Speech API."""
        if self._client is not None:
            return

        import httpx

        self._client = httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0))
        logger.info("FishSpeechTTS loaded (endpoint=%s)", self._fish.base_url)

    async def speak(self, text: str, audio_stream: AudioStream | None = None) -> None:
        """Synthesize full text and play it."""
        if not self._client:
            await self.load()

        audio = await self._synthesize(text)
        if audio is not None and audio_stream:
            audio_stream.play_audio(audio, sample_rate=self._sample_rate)

    async def speak_streamed(
        self,
        chunks: AsyncIterator[str],
        audio_stream: AudioStream | None = None,
        flush_timeout: float = 0.5,
    ) -> None:
        """Buffer streaming text into sentences, synthesize and play each.

        Identical sentence-buffering pattern to PiperTTS: asyncio.Queue +
        regex boundaries + timeout-based flushing.
        """
        if not self._client:
            await self.load()

        buffer = ""
        _DONE = object()
        queue: asyncio.Queue = asyncio.Queue()

        async def _fill_queue():
            try:
                async for text in chunks:
                    await queue.put(text)
            except Exception:
                logger.exception("Error reading response chunks")
            finally:
                await queue.put(_DONE)

        fill_task = asyncio.create_task(_fill_queue())

        try:
            while True:
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=flush_timeout)
                except asyncio.TimeoutError:
                    if buffer.strip():
                        audio = await self._synthesize(buffer.strip())
                        if audio is not None and audio_stream:
                            audio_stream.play_audio(audio, sample_rate=self._sample_rate)
                        buffer = ""
                    continue

                if item is _DONE:
                    break

                buffer += item

                while True:
                    match = _SENTENCE_END.search(buffer)
                    if not match:
                        break

                    end_idx = match.end()
                    sentence = buffer[:end_idx].strip()
                    buffer = buffer[end_idx:]

                    if sentence:
                        audio = await self._synthesize(sentence)
                        if audio is not None and audio_stream:
                            audio_stream.play_audio(audio, sample_rate=self._sample_rate)

            remaining = buffer.strip()
            if remaining:
                audio = await self._synthesize(remaining)
                if audio is not None and audio_stream:
                    audio_stream.play_audio(audio, sample_rate=self._sample_rate)
        finally:
            if not fill_task.done():
                fill_task.cancel()
                try:
                    await fill_task
                except asyncio.CancelledError:
                    pass

    async def _synthesize(self, text: str) -> np.ndarray | None:
        """Send text to Fish Speech HTTP API, receive WAV, return float32 array."""
        if not text.strip() or not self._client:
            return None

        payload = {
            "text": text,
            "format": self._format,
            "temperature": self._temperature,
            "top_p": self._top_p,
            "repetition_penalty": self._repetition_penalty,
            "max_new_tokens": self._max_new_tokens,
            "chunk_length": self._chunk_length,
            "streaming": False,
        }
        if self._reference_id:
            payload["reference_id"] = self._reference_id

        resp = await self._client.post(
            f"{self._fish.base_url}/v1/tts",
            json=payload,
            headers={"Content-Type": "application/json"},
        )
        resp.raise_for_status()

        wav_buffer = io.BytesIO(resp.content)
        with wave.open(wav_buffer, "rb") as wav_in:
            frames = wav_in.readframes(wav_in.getnframes())
            sample_width = wav_in.getsampwidth()
            if sample_width == 2:
                return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
            elif sample_width == 4:
                return np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
            else:
                raise ValueError(f"Unsupported WAV sample width: {sample_width}")

    @property
    def is_loaded(self) -> bool:
        return self._client is not None

    async def close(self) -> None:
        """Close the httpx client."""
        if self._client:
            await self._client.aclose()
            self._client = None


class KokoroTTS:
    """Synthesizes text to speech using Kokoro-82M via Kokoro-FastAPI.

    Ultra-fast (~0.3s/sentence), OpenAI-compatible API on localhost.
    Same interface as PiperTTS/FishSpeechTTS for drop-in replacement.
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8880",
        voice: str = "af_heart",
        sample_rate: int = 24000,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._voice = voice
        self._sample_rate = sample_rate
        self._client = None  # httpx.AsyncClient

    async def load(self, audio_stream: AudioStream | None = None) -> None:
        """Initialize HTTP client for Kokoro-FastAPI."""
        if self._client is not None:
            return

        import httpx

        self._client = httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=5.0))
        logger.info("KokoroTTS loaded (endpoint=%s, voice=%s)", self._base_url, self._voice)

    async def speak(self, text: str, audio_stream: AudioStream | None = None) -> None:
        """Synthesize full text and play it."""
        if not self._client:
            await self.load()

        audio = await self._synthesize(text)
        if audio is not None and audio_stream:
            audio_stream.play_audio(audio, sample_rate=self._sample_rate)

    async def speak_streamed(
        self,
        chunks: AsyncIterator[str],
        audio_stream: AudioStream | None = None,
        flush_timeout: float = 0.5,
    ) -> None:
        """Buffer streaming text into sentences, synthesize and play each."""
        if not self._client:
            await self.load()

        buffer = ""
        _DONE = object()
        queue: asyncio.Queue = asyncio.Queue()

        async def _fill_queue():
            try:
                async for text in chunks:
                    await queue.put(text)
            except Exception:
                logger.exception("Error reading response chunks")
            finally:
                await queue.put(_DONE)

        fill_task = asyncio.create_task(_fill_queue())

        try:
            while True:
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=flush_timeout)
                except asyncio.TimeoutError:
                    if buffer.strip():
                        audio = await self._synthesize(buffer.strip())
                        if audio is not None and audio_stream:
                            audio_stream.play_audio(audio, sample_rate=self._sample_rate)
                        buffer = ""
                    continue

                if item is _DONE:
                    break

                buffer += item

                while True:
                    match = _SENTENCE_END.search(buffer)
                    if not match:
                        break

                    end_idx = match.end()
                    sentence = buffer[:end_idx].strip()
                    buffer = buffer[end_idx:]

                    if sentence:
                        audio = await self._synthesize(sentence)
                        if audio is not None and audio_stream:
                            audio_stream.play_audio(audio, sample_rate=self._sample_rate)

            remaining = buffer.strip()
            if remaining:
                audio = await self._synthesize(remaining)
                if audio is not None and audio_stream:
                    audio_stream.play_audio(audio, sample_rate=self._sample_rate)
        finally:
            if not fill_task.done():
                fill_task.cancel()
                try:
                    await fill_task
                except asyncio.CancelledError:
                    pass

    async def _synthesize(self, text: str) -> np.ndarray | None:
        """Send text to Kokoro-FastAPI, receive WAV, return float32 array."""
        if not text.strip() or not self._client:
            return None

        payload = {
            "model": "kokoro",
            "input": text,
            "voice": self._voice,
            "response_format": "wav",
        }

        resp = await self._client.post(
            f"{self._base_url}/v1/audio/speech",
            json=payload,
        )
        resp.raise_for_status()

        wav_buffer = io.BytesIO(resp.content)
        with wave.open(wav_buffer, "rb") as wav_in:
            self._sample_rate = wav_in.getframerate()
            frames = wav_in.readframes(wav_in.getnframes())
            sample_width = wav_in.getsampwidth()
            if sample_width == 2:
                return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
            elif sample_width == 4:
                return np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
            else:
                raise ValueError(f"Unsupported WAV sample width: {sample_width}")

    @property
    def is_loaded(self) -> bool:
        return self._client is not None

    async def close(self) -> None:
        """Close the httpx client."""
        if self._client:
            await self._client.aclose()
            self._client = None
