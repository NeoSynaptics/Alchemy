"""Tests for FishSpeechTTS — mocked HTTP API."""

import io
import wave
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from alchemy.voice.tts import FishSpeechTTS


def _make_wav_bytes(samples: int = 1000, sample_rate: int = 44100) -> bytes:
    """Create a valid WAV file in memory."""
    buf = io.BytesIO()
    audio = np.zeros(samples, dtype=np.int16)
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(audio.tobytes())
    return buf.getvalue()


class TestFishSpeechTTS:
    def test_init(self):
        fish_proc = MagicMock()
        tts = FishSpeechTTS(fish_process=fish_proc)
        assert tts.is_loaded is False
        assert tts._sample_rate == 44100

    def test_init_custom_params(self):
        fish_proc = MagicMock()
        tts = FishSpeechTTS(
            fish_process=fish_proc,
            sample_rate=22050,
            temperature=0.5,
            reference_id="my_voice",
            repetition_penalty=1.2,
        )
        assert tts._sample_rate == 22050
        assert tts._temperature == 0.5
        assert tts._reference_id == "my_voice"
        assert tts._repetition_penalty == 1.2

    async def test_load_creates_client(self):
        fish_proc = MagicMock()
        fish_proc.base_url = "http://localhost:8085"
        tts = FishSpeechTTS(fish_process=fish_proc)

        await tts.load()
        assert tts.is_loaded is True

        # Cleanup
        await tts.close()
        assert tts.is_loaded is False

    async def test_synthesize_returns_audio(self):
        fish_proc = MagicMock()
        fish_proc.base_url = "http://localhost:8085"
        tts = FishSpeechTTS(fish_process=fish_proc)

        wav_bytes = _make_wav_bytes()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = wav_bytes
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        tts._client = mock_client

        audio = await tts._synthesize("Hello world")

        assert audio is not None
        assert isinstance(audio, np.ndarray)
        assert audio.dtype == np.float32
        mock_client.post.assert_called_once()

    async def test_synthesize_empty_text_returns_none(self):
        fish_proc = MagicMock()
        tts = FishSpeechTTS(fish_process=fish_proc)
        tts._client = AsyncMock()

        result = await tts._synthesize("")
        assert result is None

        result = await tts._synthesize("   ")
        assert result is None

    async def test_synthesize_no_client_returns_none(self):
        fish_proc = MagicMock()
        tts = FishSpeechTTS(fish_process=fish_proc)
        # _client is None (not loaded)

        result = await tts._synthesize("Hello")
        assert result is None

    async def test_speak(self):
        fish_proc = MagicMock()
        fish_proc.base_url = "http://localhost:8085"
        tts = FishSpeechTTS(fish_process=fish_proc, sample_rate=44100)

        wav_bytes = _make_wav_bytes()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = wav_bytes
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        tts._client = mock_client

        mock_audio = MagicMock()
        mock_audio.play_audio = MagicMock()

        await tts.speak("Hello", audio_stream=mock_audio)
        mock_audio.play_audio.assert_called_once()

        # Verify sample rate passed correctly
        call_kwargs = mock_audio.play_audio.call_args
        assert call_kwargs[1]["sample_rate"] == 44100

    async def test_speak_streamed_sentence_buffering(self):
        fish_proc = MagicMock()
        fish_proc.base_url = "http://localhost:8085"
        tts = FishSpeechTTS(fish_process=fish_proc, sample_rate=44100)

        wav_bytes = _make_wav_bytes()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = wav_bytes
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        tts._client = mock_client

        mock_audio = MagicMock()
        mock_audio.play_audio = MagicMock()

        async def chunks():
            yield "Hello world. "
            yield "How are you? "
            yield "I'm fine"

        await tts.speak_streamed(chunks(), audio_stream=mock_audio)

        # "Hello world." + "How are you?" + "I'm fine" (flush)
        assert mock_audio.play_audio.call_count == 3

    async def test_speak_streamed_buffers_until_boundary(self):
        fish_proc = MagicMock()
        fish_proc.base_url = "http://localhost:8085"
        tts = FishSpeechTTS(fish_process=fish_proc)

        wav_bytes = _make_wav_bytes()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = wav_bytes
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        tts._client = mock_client

        mock_audio = MagicMock()
        mock_audio.play_audio = MagicMock()

        async def chunks():
            yield "This is "
            yield "one long "
            yield "sentence."

        await tts.speak_streamed(chunks(), audio_stream=mock_audio)

        # One sentence boundary → one play call
        assert mock_audio.play_audio.call_count == 1

    async def test_synthesize_posts_correct_payload(self):
        fish_proc = MagicMock()
        fish_proc.base_url = "http://localhost:8080"
        tts = FishSpeechTTS(
            fish_process=fish_proc,
            reference_id="my_voice",
            repetition_penalty=1.3,
        )

        wav_bytes = _make_wav_bytes()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = wav_bytes
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        tts._client = mock_client

        await tts._synthesize("Test")

        call_kwargs = mock_client.post.call_args
        payload = call_kwargs[1]["json"]
        assert payload["text"] == "Test"
        assert payload["reference_id"] == "my_voice"
        assert payload["repetition_penalty"] == 1.3
        assert payload["streaming"] is False

    async def test_close(self):
        fish_proc = MagicMock()
        tts = FishSpeechTTS(fish_process=fish_proc)

        mock_client = AsyncMock()
        mock_client.aclose = AsyncMock()
        tts._client = mock_client

        await tts.close()
        assert tts._client is None
        mock_client.aclose.assert_called_once()
