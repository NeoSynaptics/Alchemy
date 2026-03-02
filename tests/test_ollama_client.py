"""OllamaClient tests — mock httpx responses."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from alchemy.models.ollama_client import OllamaClient


@pytest.fixture
async def client():
    c = OllamaClient(host="http://localhost:11434")
    await c.start()
    yield c
    await c.close()


class TestChat:
    async def test_chat_text(self, client):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "message": {"role": "assistant", "content": "Thought: I see.\nAction: click(start_box='(100,200)')"},
            "total_duration": 5000000000,
            "eval_count": 20,
        }

        with patch.object(client._client, "post", new_callable=AsyncMock, return_value=mock_resp):
            result = await client.chat("test-model", [{"role": "user", "content": "hello"}])

        assert result["message"]["content"].startswith("Thought:")
        assert result["eval_count"] == 20

    async def test_chat_with_images(self, client):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "message": {"role": "assistant", "content": "Action: click(start_box='(50,50)')"},
        }

        with patch.object(client._client, "post", new_callable=AsyncMock, return_value=mock_resp) as mock_post:
            png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
            await client.chat(
                "test-model",
                [{"role": "user", "content": "click it"}],
                images=[png],
            )

        # Verify images were base64-encoded and attached
        call_payload = mock_post.call_args[1]["json"]
        last_user = [m for m in call_payload["messages"] if m["role"] == "user"][-1]
        assert "images" in last_user
        assert len(last_user["images"]) == 1


class TestListModels:
    async def test_list(self, client):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "models": [
                {"name": "avil/UI-TARS:latest", "size": 3600000000},
                {"name": "qwen2.5-coder:32b", "size": 19000000000},
            ]
        }

        with patch.object(client._client, "get", new_callable=AsyncMock, return_value=mock_resp):
            models = await client.list_models()

        assert len(models) == 2
        assert models[0]["name"] == "avil/UI-TARS:latest"

    async def test_is_model_available_true(self, client):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "models": [{"name": "avil/UI-TARS:latest", "size": 3600000000}]
        }

        with patch.object(client._client, "get", new_callable=AsyncMock, return_value=mock_resp):
            assert await client.is_model_available("avil/UI-TARS") is True

    async def test_is_model_available_false(self, client):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"models": []}

        with patch.object(client._client, "get", new_callable=AsyncMock, return_value=mock_resp):
            assert await client.is_model_available("nonexistent") is False


class TestPing:
    async def test_ping_success(self, client):
        mock_resp = MagicMock()
        mock_resp.status_code = 200

        with patch.object(client._client, "get", new_callable=AsyncMock, return_value=mock_resp):
            assert await client.ping() is True

    async def test_ping_failure(self, client):
        with patch.object(client._client, "get", new_callable=AsyncMock, side_effect=httpx.ConnectError("refused")):
            assert await client.ping() is False


class TestNotStarted:
    async def test_chat_before_start(self):
        c = OllamaClient()
        with pytest.raises(RuntimeError, match="not started"):
            await c.chat("model", [])
