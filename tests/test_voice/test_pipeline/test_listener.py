"""Tests for SpeechListener — mocked VAD and audio."""

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from alchemy.voice.listener import SpeechListener, _FRAME_SAMPLES


def _make_pcm_chunk(samples: int = 512) -> bytes:
    """Create a chunk of silence PCM audio."""
    return np.zeros(samples, dtype=np.int16).tobytes()


class TestSpeechListener:
    def test_init_defaults(self):
        listener = SpeechListener()
        assert listener._silence_frames > 0

    async def test_record_stops_on_silence_after_speech(self):
        listener = SpeechListener(vad_aggressiveness=2, silence_ms=90)

        mock_audio = AsyncMock()
        # Provide enough audio data (each chunk = 512 samples = 1024 bytes)
        # VAD frames are 480 samples = 960 bytes
        chunk = _make_pcm_chunk(960)
        mock_audio.read_chunk = AsyncMock(return_value=chunk)

        with patch.object(listener._vad, "is_speech") as mock_vad:
            # 5 speech frames, then 10 silence frames (enough to trigger stop)
            mock_vad.side_effect = (
                [True] * 5 + [False] * 10
            )

            audio = await listener.record(mock_audio)
            assert len(audio) > 0
            assert isinstance(audio, bytes)

    async def test_record_respects_max_duration(self):
        listener = SpeechListener(
            vad_aggressiveness=2,
            silence_ms=90000,  # Very high — should hit max duration
            max_duration_s=0.1,  # Very short max
        )

        mock_audio = AsyncMock()
        chunk = _make_pcm_chunk(960)
        mock_audio.read_chunk = AsyncMock(return_value=chunk)

        with patch.object(listener._vad, "is_speech", return_value=True):
            audio = await listener.record(mock_audio)
            assert len(audio) > 0

    def test_vad_aggressiveness(self):
        listener = SpeechListener(vad_aggressiveness=3)
        # webrtcvad.Vad(3) should have been created
        assert listener._vad is not None
