"""Tests for WhisperSTT — mocked faster-whisper."""

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from alchemy.voice.stt import WhisperSTT


class TestWhisperSTT:
    def test_init_defaults(self):
        stt = WhisperSTT()
        assert stt._model_size == "large-v3"
        assert stt._device == "cuda"
        assert stt._compute_type == "float16"

    def test_cpu_uses_int8(self):
        stt = WhisperSTT(device="cpu")
        assert stt._compute_type == "int8"

    def test_is_loaded_false_initially(self):
        stt = WhisperSTT()
        assert stt.is_loaded is False

    async def test_transcribe(self):
        stt = WhisperSTT()

        # Mock a segment
        mock_segment = MagicMock()
        mock_segment.text = " Hello world "

        mock_info = MagicMock()
        mock_info.language = "en"
        mock_info.language_probability = 0.99

        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([mock_segment], mock_info)
        stt._model = mock_model

        # Create some PCM audio
        audio_bytes = np.zeros(16000, dtype=np.int16).tobytes()  # 1 second
        text = await stt.transcribe(audio_bytes)

        assert text == "Hello world"
        mock_model.transcribe.assert_called_once()

    async def test_transcribe_multiple_segments(self):
        stt = WhisperSTT()

        seg1 = MagicMock()
        seg1.text = "Hello "
        seg2 = MagicMock()
        seg2.text = " world"

        mock_info = MagicMock()
        mock_info.language = "en"
        mock_info.language_probability = 0.95

        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([seg1, seg2], mock_info)
        stt._model = mock_model

        audio_bytes = np.zeros(16000, dtype=np.int16).tobytes()
        text = await stt.transcribe(audio_bytes)
        assert text == "Hello world"

    async def test_transcribe_empty_audio(self):
        stt = WhisperSTT()

        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([], MagicMock(language="en", language_probability=0.0))
        stt._model = mock_model

        audio_bytes = np.zeros(100, dtype=np.int16).tobytes()
        text = await stt.transcribe(audio_bytes)
        assert text == ""

    async def test_unload(self):
        stt = WhisperSTT()
        stt._model = MagicMock()
        assert stt.is_loaded is True

        with patch.object(WhisperSTT, "_free_gpu_memory"):
            await stt.unload()

        assert stt.is_loaded is False
        assert stt._model is None

    async def test_unload_when_not_loaded(self):
        stt = WhisperSTT()
        assert stt.is_loaded is False

        # Should not raise
        await stt.unload()
        assert stt.is_loaded is False

    async def test_transcribe_after_unload_reloads(self):
        """After unload, transcribe should lazy-reload the model."""
        stt = WhisperSTT()
        stt._model = MagicMock()

        with patch.object(WhisperSTT, "_free_gpu_memory"):
            await stt.unload()

        assert stt.is_loaded is False

        # Now set up a fresh mock for the re-loaded model
        mock_segment = MagicMock()
        mock_segment.text = "Reloaded"
        mock_info = MagicMock(language="en", language_probability=0.99)
        new_model = MagicMock()
        new_model.transcribe.return_value = ([mock_segment], mock_info)

        with patch("alchemy.voice.stt.WhisperSTT.load") as mock_load:
            async def fake_load():
                stt._model = new_model

            mock_load.side_effect = fake_load

            audio_bytes = np.zeros(16000, dtype=np.int16).tobytes()
            text = await stt.transcribe(audio_bytes)
            assert text == "Reloaded"
            mock_load.assert_called_once()
