"""Tests for /chat and /chat/stream API endpoints."""

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from alchemy.voice.models.conversation import ConversationManager
from alchemy.voice.models.provider import AlchemyProvider, OllamaProvider
from alchemy.voice.models.registry import ModelRegistry
from alchemy.voice.models.schemas import (
    ChatResponse,
    ModelCapability,
    ModelCard,
    ModelLocation,
    RouteDecision,
    RouteIntent,
    SpeedTier,
)
from alchemy.voice.router.router import SmartRouter
from alchemy.server import app


@pytest_asyncio.fixture
async def client():
    """Async client with router pre-wired (no real Ollama/Alchemy)."""
    registry = ModelRegistry()
    registry.register(
        ModelCard(
            name="qwen3:14b",
            capabilities=[ModelCapability.CONVERSATION, ModelCapability.CLASSIFICATION],
            speed_tier=SpeedTier.FAST,
            location=ModelLocation.GPU_LOCAL,
            is_default_for=[ModelCapability.CONVERSATION, ModelCapability.CLASSIFICATION],
        )
    )
    registry.register(
        ModelCard(
            name="ui-tars:72b",
            capabilities=[ModelCapability.VISION],
            speed_tier=SpeedTier.SLOW,
            location=ModelLocation.CPU_REMOTE,
            is_default_for=[ModelCapability.VISION],
        )
    )

    mock_ollama = AsyncMock(spec=OllamaProvider)
    mock_ollama.generate = AsyncMock(
        return_value=("[CONVERSATION]\nHello! How can I help?", 42.0)
    )

    mock_alchemy = AsyncMock(spec=AlchemyProvider)
    task_id = uuid4()
    mock_alchemy.generate = AsyncMock(
        return_value=(f"Task submitted to Alchemy (ID: {task_id}). Status: pending.", 15.0)
    )

    providers = {
        ModelLocation.GPU_LOCAL: mock_ollama,
        ModelLocation.CPU_REMOTE: mock_alchemy,
    }

    app.state.router = SmartRouter(
        registry=registry,
        providers=providers,
        conversation_manager=ConversationManager(),
    )
    app.state.registry = registry
    app.state.providers = providers

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


class TestChatEndpoint:
    async def test_chat_conversation(self, client):
        resp = await client.post("/v1/chat/", json={"message": "What is Python?"})
        assert resp.status_code == 200
        data = resp.json()
        assert "Hello! How can I help?" in data["message"]
        assert data["model_used"] == "qwen3:14b"

    async def test_chat_gui_task(self, client):
        resp = await client.post("/v1/chat/", json={"message": "open Chrome"})
        assert resp.status_code == 200
        data = resp.json()
        assert "Task submitted" in data["message"]
        assert data["model_used"] == "ui-tars:72b"

    async def test_chat_system_command(self, client):
        resp = await client.post("/v1/chat/", json={"message": "pause"})
        assert resp.status_code == 200
        data = resp.json()
        assert "System command" in data["message"]

    async def test_chat_preserves_conversation_id(self, client):
        cid = str(uuid4())
        resp = await client.post(
            "/v1/chat/", json={"message": "hello", "conversation_id": cid}
        )
        assert resp.status_code == 200
        assert resp.json()["conversation_id"] == cid


class TestChatStreamEndpoint:
    async def test_stream_returns_sse(self, client):
        """Stream endpoint should return text/event-stream content type."""
        resp = await client.post("/v1/chat/stream", json={"message": "open Chrome"})
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]

    async def test_stream_contains_data_lines(self, client):
        resp = await client.post("/v1/chat/stream", json={"message": "open Chrome"})
        body = resp.text
        assert "data:" in body
