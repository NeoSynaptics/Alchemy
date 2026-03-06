"""Tests for WakeWordDetector — mocked openwakeword."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from alchemy.voice.wake_word import WakeWordDetector


class TestWakeWordDetector:
    def test_init_defaults(self):
        detector = WakeWordDetector()
        assert detector._threshold == 0.5

    def test_custom_threshold(self):
        detector = WakeWordDetector(model_name="test", threshold=0.8)
        assert detector._threshold == 0.8
        assert detector._model_name == "test"

    async def test_listen_detects_wake_word(self):
        detector = WakeWordDetector(threshold=0.5)

        # Mock the openwakeword model
        mock_model = MagicMock()
        mock_model.predict.return_value = {"hey_neo": 0.9}
        mock_model.reset = MagicMock()
        detector._oww = mock_model

        # Mock audio stream
        mock_audio = AsyncMock()
        chunk = np.zeros(512, dtype=np.int16).tobytes()
        mock_audio.read_chunk = AsyncMock(return_value=chunk)

        word = await detector.listen(mock_audio)
        assert word == "hey_neo"

    async def test_listen_waits_for_threshold(self):
        detector = WakeWordDetector(threshold=0.7)

        mock_model = MagicMock()
        # First call below threshold, second above
        mock_model.predict.side_effect = [
            {"hey_neo": 0.3},
            {"hey_neo": 0.5},
            {"hey_neo": 0.85},
        ]
        mock_model.reset = MagicMock()
        detector._oww = mock_model

        mock_audio = AsyncMock()
        chunk = np.zeros(512, dtype=np.int16).tobytes()
        mock_audio.read_chunk = AsyncMock(return_value=chunk)

        word = await detector.listen(mock_audio)
        assert word == "hey_neo"
        assert mock_model.predict.call_count == 3

    def test_reset(self):
        detector = WakeWordDetector()
        mock_model = MagicMock()
        detector._oww = mock_model
        detector.reset()
        mock_model.reset.assert_called_once()

    def test_reset_without_model(self):
        detector = WakeWordDetector()
        detector.reset()  # Should not raise
