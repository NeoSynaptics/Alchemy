"""Tests for new OllamaClient features — retry logic, streaming, options."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from alchemy.models.ollama_client import OllamaClient


@pytest.fixture
async def client():
    c = OllamaClient(host="http://localhost:11434", retry_attempts=3, retry_delay=0.01)
    await c.start()
    yield c
    await c.close()


class TestRetryLogic:
    async def test_chat_retries_on_timeout(self, client):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "message": {"role": "assistant", "content": "ok"},
        }

        call_count = 0

        async def flaky_post(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise httpx.TimeoutException("timeout")
            return mock_resp

        with patch.object(client._client, "post", side_effect=flaky_post):
            result = await client.chat("model", [{"role": "user", "content": "hello"}])

        assert result["message"]["content"] == "ok"
        assert call_count == 3  # 2 failures + 1 success

    async def test_chat_exhausts_retries(self, client):
        with patch.object(
            client._client, "post",
            new_callable=AsyncMock,
            side_effect=httpx.TimeoutException("always timeout"),
        ):
            with pytest.raises(httpx.TimeoutException):
                await client.chat("model", [{"role": "user", "content": "hello"}])

    async def test_chat_no_retry_on_other_errors(self, client):
        """Non-timeout/status errors are not retried."""
        with patch.object(
            client._client, "post",
            new_callable=AsyncMock,
            side_effect=httpx.ConnectError("refused"),
        ):
            with pytest.raises(httpx.ConnectError):
                await client.chat("model", [{"role": "user", "content": "hello"}])


class TestChatOptions:
    async def test_options_passed_to_payload(self, client):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"message": {"content": "ok"}}

        with patch.object(client._client, "post", new_callable=AsyncMock, return_value=mock_resp) as mock_post:
            await client.chat(
                "model",
                [{"role": "user", "content": "hello"}],
                options={"temperature": 0.0, "num_predict": 384},
            )

        payload = mock_post.call_args[1]["json"]
        assert payload["options"]["temperature"] == 0.0
        assert payload["options"]["num_predict"] == 384

    async def test_no_options_omitted(self, client):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"message": {"content": "ok"}}

        with patch.object(client._client, "post", new_callable=AsyncMock, return_value=mock_resp) as mock_post:
            await client.chat("model", [{"role": "user", "content": "hello"}])

        payload = mock_post.call_args[1]["json"]
        assert "options" not in payload


class TestChatStream:
    async def test_stream_accumulates_text(self, client):
        """chat_stream returns the full accumulated text."""
        chunks = [
            '{"message": {"content": "Thought: "}, "done": false}',
            '{"message": {"content": "I see."}, "done": false}',
            '{"message": {"content": "\\nAction: "}, "done": false}',
            '{"message": {"content": "click(start_box=\'(100,200)\')"}, "done": false}',
            '{"message": {"content": ""}, "done": true}',
        ]

        mock_resp = AsyncMock()
        mock_resp.raise_for_status = MagicMock()

        async def mock_aiter_lines():
            for chunk in chunks:
                yield chunk

        mock_resp.aiter_lines = mock_aiter_lines
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        with patch.object(client._client, "stream", return_value=mock_resp):
            result = await client.chat_stream("model", [{"role": "user", "content": "hi"}])

        assert "Thought: I see." in result
        assert "Action: click" in result

    async def test_stream_early_stop(self, client):
        """chat_stream stops reading after finding Action: with closing paren."""
        chunks = [
            '{"message": {"content": "Thought: Click."}, "done": false}',
            '{"message": {"content": "\\nAction: "}, "done": false}',
            '{"message": {"content": "click(start_box=\'(100,200)\')"}, "done": false}',
            # These should NOT be reached if early stop works
            '{"message": {"content": " extra garbage"}, "done": false}',
            '{"message": {"content": ""}, "done": true}',
        ]

        mock_resp = AsyncMock()
        mock_resp.raise_for_status = MagicMock()
        read_count = 0

        async def mock_aiter_lines():
            nonlocal read_count
            for chunk in chunks:
                read_count += 1
                yield chunk

        mock_resp.aiter_lines = mock_aiter_lines
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        with patch.object(client._client, "stream", return_value=mock_resp):
            result = await client.chat_stream(
                "model", [{"role": "user", "content": "hi"}],
                stop_at="\nAction:",
            )

        assert "Action: click" in result
        # Early stop should have kicked in after the closing paren
        assert read_count <= 4  # Should stop before reading all 5 chunks

    async def test_stream_retries_on_timeout(self, client):
        """chat_stream retries on timeout."""
        call_count = 0

        chunks = [
            '{"message": {"content": "Thought: ok."}, "done": false}',
            '{"message": {"content": "\\nAction: wait()"}, "done": false}',
            '{"message": {"content": ""}, "done": true}',
        ]

        mock_resp = AsyncMock()
        mock_resp.raise_for_status = MagicMock()

        async def mock_aiter_lines():
            for chunk in chunks:
                yield chunk

        mock_resp.aiter_lines = mock_aiter_lines
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        def mock_stream(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise httpx.TimeoutException("timeout")
            return mock_resp

        with patch.object(client._client, "stream", side_effect=mock_stream):
            result = await client.chat_stream("model", [{"role": "user", "content": "hi"}])

        assert "Action: wait()" in result
        assert call_count == 2
