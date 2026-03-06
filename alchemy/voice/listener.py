"""Speech listener — VAD-based recording after wake word.

Records audio from mic, uses webrtcvad to detect speech boundaries,
stops when silence exceeds threshold. Returns complete utterance as PCM bytes.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from alchemy.voice.audio import AudioStream

logger = logging.getLogger(__name__)

# webrtcvad requires 10/20/30ms frames at 16kHz
_FRAME_DURATION_MS = 30
_FRAME_SAMPLES = 16000 * _FRAME_DURATION_MS // 1000  # 480 samples


class SpeechListener:
    """Records speech after wake word, using VAD to find end of utterance."""

    def __init__(
        self,
        vad_aggressiveness: int = 2,
        silence_ms: int = 800,
        max_duration_s: float = 30.0,
    ) -> None:
        import webrtcvad

        self._vad = webrtcvad.Vad(vad_aggressiveness)
        self._silence_frames = silence_ms // _FRAME_DURATION_MS
        self._max_frames = int(max_duration_s * 1000 / _FRAME_DURATION_MS)

    async def record(self, audio_stream: AudioStream) -> bytes:
        """Record until speech ends. Returns 16kHz mono 16-bit PCM bytes."""
        frames: list[bytes] = []
        silent_count = 0
        speech_detected = False
        buffer = b""

        for _ in range(self._max_frames):
            chunk = await audio_stream.read_chunk()
            buffer += chunk

            # Process complete VAD frames from buffer
            while len(buffer) >= _FRAME_SAMPLES * 2:  # 2 bytes per int16 sample
                frame = buffer[: _FRAME_SAMPLES * 2]
                buffer = buffer[_FRAME_SAMPLES * 2 :]

                is_speech = self._vad.is_speech(frame, 16000)
                frames.append(frame)

                if is_speech:
                    speech_detected = True
                    silent_count = 0
                else:
                    silent_count += 1

                if speech_detected and silent_count >= self._silence_frames:
                    logger.info(
                        "Speech ended after %d frames (%.1fs)",
                        len(frames),
                        len(frames) * _FRAME_DURATION_MS / 1000,
                    )
                    return b"".join(frames)

        logger.warning("Max recording duration reached (%.0fs)", self._max_frames * _FRAME_DURATION_MS / 1000)
        return b"".join(frames)
