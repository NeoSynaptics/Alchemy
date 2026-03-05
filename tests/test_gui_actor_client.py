"""Tests for GUIActorClient — HTTP client for the GUI-Actor inference server."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from alchemy.adapters.gui_actor import GUIActorClient, GUIActorPrediction


class TestGUIActorPrediction:
    def test_success_prediction(self):
        p = GUIActorPrediction(x=500, y=300, norm_x=0.39, norm_y=0.42, confidence=0.95, success=True)
        assert p.success is True
        assert p.x == 500
        assert p.error is None

    def test_failed_prediction(self):
        p = GUIActorPrediction(x=0, y=0, norm_x=0.0, norm_y=0.0, confidence=0.0, success=False, error="timeout")
        assert p.success is False
        assert p.error == "timeout"


class TestGUIActorClientInit:
    def test_default_config(self):
        client = GUIActorClient()
        assert client._host == "http://localhost:8200"
        assert client._timeout == 30.0

    def test_custom_config(self):
        client = GUIActorClient(host="http://gpu1:9000", timeout=60.0)
        assert client._host == "http://gpu1:9000"
        assert client._timeout == 60.0

    def test_trailing_slash_stripped(self):
        client = GUIActorClient(host="http://localhost:8200/")
        assert client._host == "http://localhost:8200"


class TestGUIActorClientNotStarted:
    @pytest.mark.asyncio
    async def test_predict_before_start_raises(self):
        client = GUIActorClient()
        with pytest.raises(RuntimeError, match="not started"):
            await client.predict(b"fake", "click button")


class TestGUIActorClientPredict:
    @pytest.mark.asyncio
    async def test_successful_predict(self):
        client = GUIActorClient()
        mock_http = AsyncMock(spec=httpx.AsyncClient)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "points": [[0.45, 0.32]],
            "confidences": [0.92],
        }
        mock_http.post = AsyncMock(return_value=mock_response)
        client._client = mock_http

        result = await client.predict(
            image=b"fake-jpeg",
            instruction="Click the search button",
            screen_width=1280,
            screen_height=720,
        )

        assert result.success is True
        assert result.x == 576  # 0.45 * 1280
        assert result.y == 230  # 0.32 * 720
        assert result.norm_x == 0.45
        assert result.confidence == 0.92
        mock_http.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_points_returned(self):
        client = GUIActorClient()
        mock_http = AsyncMock(spec=httpx.AsyncClient)
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"points": [], "confidences": []}
        mock_http.post = AsyncMock(return_value=mock_response)
        client._client = mock_http

        result = await client.predict(b"fake", "click something")

        assert result.success is False
        assert "No points" in result.error

    @pytest.mark.asyncio
    async def test_coordinate_clamping(self):
        client = GUIActorClient()
        mock_http = AsyncMock(spec=httpx.AsyncClient)
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "points": [[1.1, -0.1]],  # Out of bounds
            "confidences": [0.5],
        }
        mock_http.post = AsyncMock(return_value=mock_response)
        client._client = mock_http

        result = await client.predict(b"fake", "click", screen_width=1280, screen_height=720)

        assert result.success is True
        assert result.x == 1280  # Clamped to max
        assert result.y == 0  # Clamped to min

    @pytest.mark.asyncio
    async def test_retry_on_timeout(self):
        client = GUIActorClient(retry_attempts=2, retry_delay=0.01)
        mock_http = AsyncMock(spec=httpx.AsyncClient)

        # First call times out, second succeeds
        good_response = MagicMock()
        good_response.raise_for_status = MagicMock()
        good_response.json.return_value = {"points": [[0.5, 0.5]], "confidences": [0.8]}

        mock_http.post = AsyncMock(
            side_effect=[httpx.TimeoutException("timeout"), good_response]
        )
        client._client = mock_http

        result = await client.predict(b"fake", "click button")

        assert result.success is True
        assert mock_http.post.call_count == 2

    @pytest.mark.asyncio
    async def test_all_retries_exhausted(self):
        client = GUIActorClient(retry_attempts=2, retry_delay=0.01)
        mock_http = AsyncMock(spec=httpx.AsyncClient)
        mock_http.post = AsyncMock(side_effect=httpx.TimeoutException("timeout"))
        client._client = mock_http

        result = await client.predict(b"fake", "click button")

        assert result.success is False
        assert "retries exhausted" in result.error

    @pytest.mark.asyncio
    async def test_payload_format(self):
        """Verify the request payload includes base64 image and instruction."""
        client = GUIActorClient()
        mock_http = AsyncMock(spec=httpx.AsyncClient)
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"points": [[0.5, 0.5]], "confidences": [0.9]}
        mock_http.post = AsyncMock(return_value=mock_response)
        client._client = mock_http

        await client.predict(b"\xff\xd8test", "Click search", topk=3)

        call_kwargs = mock_http.post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert payload["instruction"] == "Click search"
        assert payload["topk"] == 3
        assert len(payload["image"]) > 0  # base64 encoded


class TestGUIActorPing:
    @pytest.mark.asyncio
    async def test_ping_success(self):
        client = GUIActorClient()
        mock_http = AsyncMock(spec=httpx.AsyncClient)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_http.get = AsyncMock(return_value=mock_response)
        client._client = mock_http

        assert await client.ping() is True

    @pytest.mark.asyncio
    async def test_ping_failure(self):
        client = GUIActorClient()
        mock_http = AsyncMock(spec=httpx.AsyncClient)
        mock_http.get = AsyncMock(side_effect=httpx.ConnectError("refused"))
        client._client = mock_http

        assert await client.ping() is False
