"""Speech-to-text — faster-whisper wrapper.

Lazy-loads the model on first use. Supports CUDA and CPU via settings.
"""

from __future__ import annotations

import asyncio
import logging
import time

import numpy as np

logger = logging.getLogger(__name__)


class WhisperSTT:
    """Transcribes audio bytes to text using faster-whisper."""

    def __init__(
        self,
        model_size: str = "large-v3",
        device: str = "cuda",
        compute_type: str = "float16",
    ) -> None:
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type if device == "cuda" else "int8"
        self._model = None

    async def load(self) -> None:
        """Load the Whisper model (lazy, first call only)."""
        if self._model:
            return

        from faster_whisper import WhisperModel

        loop = asyncio.get_running_loop()
        self._model = await loop.run_in_executor(
            None,
            lambda: WhisperModel(
                self._model_size,
                device=self._device,
                compute_type=self._compute_type,
                local_files_only=True,
            ),
        )
        logger.info(
            "Whisper loaded: %s on %s (%s)",
            self._model_size, self._device, self._compute_type,
        )

    async def transcribe(self, audio_bytes: bytes) -> str:
        """Transcribe 16kHz mono 16-bit PCM audio to text."""
        if not self._model:
            await self.load()

        # Convert PCM bytes to float32 numpy array (Whisper expects float32)
        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        loop = asyncio.get_running_loop()
        t0 = time.monotonic()

        def _do_transcribe():
            segments, info = self._model.transcribe(
                audio,
                beam_size=5,
                language="en",
                vad_filter=True,
            )
            # Consume the generator in the executor thread (does the real work)
            text = " ".join(seg.text.strip() for seg in segments).strip()
            return text, info

        text, info = await loop.run_in_executor(None, _do_transcribe)
        elapsed_ms = (time.monotonic() - t0) * 1000

        logger.info(
            "Transcribed (%.0fms, lang=%s, prob=%.2f): %s",
            elapsed_ms, info.language, info.language_probability, text,
        )
        return text

    async def unload(self) -> None:
        """Unload the Whisper model to free VRAM."""
        if self._model is None:
            return

        model_ref = self._model
        self._model = None

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._free_gpu_memory, model_ref)
        logger.info("Whisper model unloaded, VRAM freed")

    @staticmethod
    def _free_gpu_memory(model):
        import gc

        del model
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    @property
    def is_loaded(self) -> bool:
        return self._model is not None
