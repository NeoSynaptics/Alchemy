"""Wake word detector — openwakeword integration.

Listens continuously on CPU for "hey_neo" (or configured wake word).
Blocks until detected, then returns control to the pipeline.
"""

from __future__ import annotations

import logging

import numpy as np

from alchemy.voice.audio import AudioStream

logger = logging.getLogger(__name__)


class WakeWordDetector:
    """Streams mic audio through openwakeword, blocks until wake word detected."""

    def __init__(self, model_name: str = "hey jarvis", threshold: float = 0.5) -> None:
        self._model_name = model_name
        self._threshold = threshold
        self._oww = None

    async def load(self) -> None:
        """Load the openwakeword model (CPU, ~10MB)."""
        import openwakeword
        from openwakeword.model import Model

        openwakeword.utils.download_models()
        self._oww = Model(wakeword_models=[self._model_name], inference_framework="onnx")
        logger.info("Wake word model loaded: %s (threshold=%.2f)", self._model_name, self._threshold)

    async def listen(self, audio_stream: AudioStream) -> str:
        """Block until wake word detected. Returns the detected wake word name."""
        if not self._oww:
            await self.load()

        self._oww.reset()

        while True:
            chunk = await audio_stream.read_chunk()
            audio_array = np.frombuffer(chunk, dtype=np.int16)

            predictions = self._oww.predict(audio_array)

            for word, score in predictions.items():
                if score >= self._threshold:
                    logger.info("Wake word detected: %s (score=%.3f)", word, score)
                    return word

    def reset(self) -> None:
        """Reset detection state for next listening cycle."""
        if self._oww:
            self._oww.reset()
