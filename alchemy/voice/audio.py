"""Audio I/O — thin wrapper over sounddevice.

Isolates all hardware access so tests can mock this single class.
"""

from __future__ import annotations

import asyncio
import logging

import numpy as np

logger = logging.getLogger(__name__)


class AudioStream:
    """Mic input + speaker output via sounddevice."""

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        chunk_size: int = 512,
        gain: float = 1.0,
    ) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.gain = gain
        self._stream = None

    async def start(self) -> None:
        """Open the mic input stream."""
        import sounddevice as sd

        loop = asyncio.get_event_loop()
        self._queue: asyncio.Queue[bytes] = asyncio.Queue()

        def _callback(indata, frames, time_info, status):
            if status:
                logger.warning("Audio input status: %s", status)
            loop.call_soon_threadsafe(
                self._queue.put_nowait, indata[:, 0].tobytes()
            )

        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="int16",
            blocksize=self.chunk_size,
            callback=_callback,
        )
        self._stream.start()
        logger.info("Audio stream started (rate=%d, chunk=%d)", self.sample_rate, self.chunk_size)

    async def stop(self) -> None:
        """Close the mic input stream."""
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    async def read_chunk(self) -> bytes:
        """Read one chunk of 16-bit PCM audio from the mic."""
        return await self._queue.get()

    def play_audio(self, audio: np.ndarray, sample_rate: int | None = None) -> None:
        """Play audio array to speaker (blocking)."""
        import sounddevice as sd

        rate = sample_rate or self.sample_rate
        sd.play(audio, samplerate=rate)
        sd.wait()

    @staticmethod
    def is_available() -> bool:
        """Check if an audio input device exists."""
        try:
            import sounddevice as sd
            devices = sd.query_devices()
            return len(devices) > 0
        except Exception:
            return False
