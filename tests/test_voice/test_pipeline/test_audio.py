"""Tests for AudioStream — mocked sounddevice."""

import asyncio
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from alchemy.voice.audio import AudioStream


@pytest.fixture
def mock_sd():
    """Mock sounddevice module in sys.modules for lazy imports."""
    mock = MagicMock()
    with patch.dict(sys.modules, {"sounddevice": mock}):
        yield mock


class TestAudioStream:
    def test_defaults(self):
        stream = AudioStream()
        assert stream.sample_rate == 16000
        assert stream.channels == 1
        assert stream.chunk_size == 512

    def test_custom_params(self):
        stream = AudioStream(sample_rate=44100, channels=2, chunk_size=1024)
        assert stream.sample_rate == 44100
        assert stream.channels == 2

    def test_is_available_with_devices(self, mock_sd):
        mock_sd.query_devices.return_value = [{"name": "mic"}]
        assert AudioStream.is_available() is True

    def test_is_available_no_devices(self, mock_sd):
        mock_sd.query_devices.return_value = []
        result = AudioStream.is_available()
        assert result is False

    def test_is_available_on_import_error(self):
        # Setting module to None in sys.modules causes ImportError on import
        with patch.dict(sys.modules, {"sounddevice": None}):
            result = AudioStream.is_available()
            assert result is False

    async def test_read_chunk_from_queue(self):
        stream = AudioStream()
        stream._queue = asyncio.Queue()
        test_data = b"\x00\x01" * 256
        stream._queue.put_nowait(test_data)
        result = await stream.read_chunk()
        assert result == test_data

    def test_play_audio(self, mock_sd):
        stream = AudioStream()
        audio = np.zeros(1000, dtype=np.float32)
        stream.play_audio(audio, sample_rate=22050)
        mock_sd.play.assert_called_once()
        mock_sd.wait.assert_called_once()
