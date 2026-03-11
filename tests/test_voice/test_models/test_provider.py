"""Tests for OllamaProvider and AlchemyProvider — mocked HTTP."""

import json
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import httpx
import pytest

from alchemy.voice.models.provider import AlchemyProvider, OllamaProvider
from alchemy.voice.models.schemas import ChatMessage


# ---------------------------------------------------------------------------
# OllamaProvider
# ---------------------------------------------------------------------------


class TestOllamaProvider:
    @pytest.fixture
    async def provider(self):
        p = OllamaProvider(host="http://test-ollama:11434", timeout=5.0)
        await p.start()
        yield p
        await p.close()

    async def test_generate(self, provider):
        mock_response = httpx.Response(
            200,
            json={"message": {"content": "Hello!"}, "done": True},
            request=httpx.Request("POST", "http://test-ollama:11434/api/chat"),
        )
        # Use deterministic monotonic clock so elapsed_ms is always 50.0
        clock = iter([1.0, 1.05]).__next__
        with (
            patch.object(
                provider._clients["http://test-ollama:11434"],
                "post",
                new_callable=AsyncMock,
                return_value=mock_response,
            ),
            patch("alchemy.voice.models.provider.time.monotonic", side_effect=clock),
        ):
            text, ms = await provider.generate(
                "qwen3:14b",
                [ChatMessage(role="user", content="hi")],
            )
            assert text == "Hello!"
            assert ms == pytest.approx(50.0, abs=0.1)

    async def test_generate_with_endpoint_override(self, provider):
        """Test dual-host: per-model endpoint override."""
        mock_response = httpx.Response(
            200,
            json={"message": {"content": "From WSL!"}, "done": True},
        )
        wsl_endpoint = "http://wsl-host:11434"

        # First call creates a new client for the endpoint
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_response):
            # Access internal to verify lazy client creation
            assert wsl_endpoint not in provider._clients
            client = provider._get_client(wsl_endpoint)
            assert wsl_endpoint in provider._clients

    async def test_is_available_true(self, provider):
        mock_response = httpx.Response(200, text="Ollama is running")
        with patch.object(
            provider._clients["http://test-ollama:11434"],
            "get",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            assert await provider.is_available() is True

    async def test_is_available_false_on_error(self, provider):
        with patch.object(
            provider._clients["http://test-ollama:11434"],
            "get",
            new_callable=AsyncMock,
            side_effect=httpx.ConnectError("refused"),
        ):
            assert await provider.is_available() is False

    async def test_ollama_message_format(self, provider):
        messages = [
            ChatMessage(role="system", content="You are Neo"),
            ChatMessage(role="user", content="hi"),
        ]
        formatted = provider._to_ollama_messages(messages)
        assert formatted == [
            {"role": "system", "content": "You are Neo"},
            {"role": "user", "content": "hi"},
        ]


# ---------------------------------------------------------------------------
# AlchemyProvider
# ---------------------------------------------------------------------------


class TestAlchemyProvider:
    async def test_start_creates_client(self):
        provider = AlchemyProvider(base_url="http://test:8000")
        with patch("alchemy.voice.models.provider.AlchemyProvider.start", new_callable=AsyncMock):
            await provider.start()

    async def test_generate_submits_task(self):
        provider = AlchemyProvider(base_url="http://test:8000")

        mock_result = MagicMock()
        mock_result.task_id = uuid4()
        mock_result.status.value = "pending"

        mock_client = AsyncMock()
        mock_client.submit_task = AsyncMock(return_value=mock_result)
        provider._alchemy_client = mock_client

        messages = [ChatMessage(role="user", content="open Chrome")]
        text, ms = await provider.generate("ui-tars:72b", messages)

        assert "Task submitted to Alchemy" in text
        assert str(mock_result.task_id) in text
        mock_client.submit_task.assert_called_once_with(goal="open Chrome")

    async def test_generate_stream_yields_ack(self):
        provider = AlchemyProvider()
        mock_result = MagicMock()
        mock_result.task_id = uuid4()
        mock_result.status.value = "pending"

        mock_client = AsyncMock()
        mock_client.submit_task = AsyncMock(return_value=mock_result)
        provider._alchemy_client = mock_client

        messages = [ChatMessage(role="user", content="test")]
        chunks = []
        async for chunk in provider.generate_stream("ui-tars:72b", messages):
            chunks.append(chunk)

        assert len(chunks) == 1
        assert "Task submitted" in chunks[0]

    async def test_generate_raises_without_start(self):
        provider = AlchemyProvider()
        with pytest.raises(RuntimeError, match="not started"):
            await provider.generate(
                "ui-tars:72b",
                [ChatMessage(role="user", content="test")],
            )

    async def test_is_available_false_without_client(self):
        provider = AlchemyProvider()
        assert await provider.is_available() is False
