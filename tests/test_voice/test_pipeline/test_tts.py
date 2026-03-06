"""Tests for PiperTTS — mocked piper voice."""

import io
import wave
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from alchemy.voice.tts import PiperTTS


def _make_wav_bytes(samples: int = 1000, sample_rate: int = 22050) -> bytes:
    """Create a valid WAV file in memory."""
    buf = io.BytesIO()
    audio = np.zeros(samples, dtype=np.int16)
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(audio.tobytes())
    return buf.getvalue()


class TestPiperTTS:
    def test_init(self):
        tts = PiperTTS(model="test-model")
        assert tts._model_name == "test-model"
        assert tts.is_loaded is False

    async def test_speak(self):
        tts = PiperTTS()

        mock_config = MagicMock()
        mock_config.sample_rate = 22050

        mock_voice = MagicMock()
        mock_voice.config = mock_config

        # Mock synthesize to write valid WAV
        def mock_synth(text, wav_file):
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(22050)
            audio = np.zeros(1000, dtype=np.int16)
            wav_file.writeframes(audio.tobytes())

        mock_voice.synthesize_wav = mock_synth
        tts._voice = mock_voice

        mock_audio = MagicMock()
        mock_audio.play_audio = MagicMock()

        await tts.speak("Hello", audio_stream=mock_audio)
        mock_audio.play_audio.assert_called_once()

    async def test_speak_streamed_sentence_buffering(self):
        tts = PiperTTS()

        mock_config = MagicMock()
        mock_config.sample_rate = 22050

        mock_voice = MagicMock()
        mock_voice.config = mock_config

        def mock_synth(text, wav_file):
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(22050)
            audio = np.zeros(500, dtype=np.int16)
            wav_file.writeframes(audio.tobytes())

        mock_voice.synthesize_wav = mock_synth
        tts._voice = mock_voice

        mock_audio = MagicMock()
        mock_audio.play_audio = MagicMock()

        async def chunks():
            yield "Hello world. "
            yield "How are you? "
            yield "I'm fine"

        await tts.speak_streamed(chunks(), audio_stream=mock_audio)

        # Should have played: "Hello world." + "How are you?" + "I'm fine" (flush)
        assert mock_audio.play_audio.call_count == 3

    async def test_speak_streamed_buffers_until_boundary(self):
        tts = PiperTTS()

        mock_config = MagicMock()
        mock_config.sample_rate = 22050

        mock_voice = MagicMock()
        mock_voice.config = mock_config

        def mock_synth(text, wav_file):
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(22050)
            wav_file.writeframes(np.zeros(100, dtype=np.int16).tobytes())

        mock_voice.synthesize_wav = mock_synth
        tts._voice = mock_voice

        mock_audio = MagicMock()
        mock_audio.play_audio = MagicMock()

        async def chunks():
            yield "This is "
            yield "one long "
            yield "sentence."

        await tts.speak_streamed(chunks(), audio_stream=mock_audio)

        # "This is one long sentence." — one sentence boundary + no flush
        assert mock_audio.play_audio.call_count == 1

    async def test_is_loaded(self):
        tts = PiperTTS()
        assert tts.is_loaded is False
        tts._voice = MagicMock()
        assert tts.is_loaded is True
