"""Tests for SmartRouter — full routing flows with mocked providers."""

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from alchemy.voice.models.conversation import ConversationManager
from alchemy.voice.models.provider import AlchemyProvider, OllamaProvider
from alchemy.voice.models.registry import ModelRegistry
from alchemy.voice.models.schemas import (
    ChatMessage,
    ChatRequest,
    ModelCapability,
    ModelCard,
    ModelLocation,
    RouteIntent,
    SpeedTier,
)
from alchemy.voice.router.cascade import ConversationToVisionCascade
from alchemy.voice.router.router import SmartRouter


@pytest.fixture
def registry():
    reg = ModelRegistry()
    reg.register(
        ModelCard(
            name="qwen3:14b",
            capabilities=[ModelCapability.CONVERSATION, ModelCapability.CLASSIFICATION],
            speed_tier=SpeedTier.FAST,
            location=ModelLocation.GPU_LOCAL,
            is_default_for=[ModelCapability.CONVERSATION, ModelCapability.CLASSIFICATION],
        )
    )
    reg.register(
        ModelCard(
            name="ui-tars:72b",
            capabilities=[ModelCapability.VISION],
            speed_tier=SpeedTier.SLOW,
            location=ModelLocation.CPU_REMOTE,
            is_default_for=[ModelCapability.VISION],
        )
    )
    return reg


@pytest.fixture
def mock_ollama():
    provider = AsyncMock(spec=OllamaProvider)
    provider.generate = AsyncMock(return_value=("[CONVERSATION]\nPython is great!", 50.0))
    return provider


@pytest.fixture
def mock_alchemy():
    provider = AsyncMock(spec=AlchemyProvider)
    task_id = uuid4()
    provider.generate = AsyncMock(
        return_value=(f"Task submitted to Alchemy (ID: {task_id}). Status: pending.", 30.0)
    )
    return provider


@pytest.fixture
def router(registry, mock_ollama, mock_alchemy):
    providers = {
        ModelLocation.GPU_LOCAL: mock_ollama,
        ModelLocation.CPU_REMOTE: mock_alchemy,
    }
    return SmartRouter(
        registry=registry,
        providers=providers,
        conversation_manager=ConversationManager(),
        cascades=[ConversationToVisionCascade()],
    )


# ---------------------------------------------------------------------------
# Non-streaming route()
# ---------------------------------------------------------------------------


class TestNonStreamingRoute:
    async def test_conversation_routes_to_14b(self, router, mock_ollama, mock_alchemy):
        request = ChatRequest(message="What is Python?")
        response = await router.route(request)

        assert response.model_used == "qwen3:14b"
        assert "Python is great!" in response.message
        mock_ollama.generate.assert_called_once()
        mock_alchemy.generate.assert_not_called()

    async def test_gui_task_routes_to_alchemy(self, router, mock_ollama, mock_alchemy):
        request = ChatRequest(message="open Chrome and go to gmail.com")
        response = await router.route(request)

        assert response.model_used == "ui-tars:72b"
        assert "Task submitted" in response.message
        mock_alchemy.generate.assert_called_once()

    async def test_system_command_handled_internally(self, router, mock_ollama, mock_alchemy):
        request = ChatRequest(message="pause")
        response = await router.route(request)

        assert response.model_used == "internal"
        assert "System command" in response.message
        mock_ollama.generate.assert_not_called()
        mock_alchemy.generate.assert_not_called()


# ---------------------------------------------------------------------------
# Streaming route_stream()
# ---------------------------------------------------------------------------


class TestStreamingRoute:
    async def test_obvious_gui_skips_14b(self, router, mock_ollama, mock_alchemy):
        """'open Chrome' should go directly to Alchemy without 14B inference."""
        request = ChatRequest(message="open Chrome")
        chunks = []
        async for chunk in router.route_stream(request):
            chunks.append(chunk)

        assert any("Task submitted" in c.content for c in chunks)
        assert any(c.done for c in chunks)
        # 14B should NOT have been called
        mock_ollama.generate_stream.assert_not_called()

    async def test_system_command_stream(self, router, mock_ollama, mock_alchemy):
        request = ChatRequest(message="status")
        chunks = []
        async for chunk in router.route_stream(request):
            chunks.append(chunk)

        assert any("System command" in c.content for c in chunks)
        assert any(c.done for c in chunks)

    async def test_conversation_streams_from_14b(self, router, mock_ollama):
        """Ambiguous query should stream from 14B."""

        async def mock_stream(*args, **kwargs):
            yield "[CONVERSATION]\n"
            yield "Python is a"
            yield " programming language."

        mock_ollama.generate_stream = mock_stream

        request = ChatRequest(message="What is Python?")
        chunks = []
        async for chunk in router.route_stream(request):
            chunks.append(chunk)

        full_text = "".join(c.content for c in chunks)
        assert "Python is a" in full_text
        assert any(c.done for c in chunks)

    async def test_14b_gui_tag_reroutes(self, router, mock_ollama, mock_alchemy):
        """14B responds with [GUI_TASK] → should abort and submit to Alchemy."""

        async def mock_stream(*args, **kwargs):
            yield "[GUI_TASK]\nI'll help you with that."

        mock_ollama.generate_stream = mock_stream

        request = ChatRequest(message="send my hours to work")
        chunks = []
        async for chunk in router.route_stream(request):
            chunks.append(chunk)

        full_text = "".join(c.content for c in chunks)
        assert "Task submitted" in full_text
        mock_alchemy.generate.assert_called_once()

    async def test_no_tag_defaults_to_conversation(self, router, mock_ollama):
        """If 14B doesn't produce a tag after 50 chars, treat as conversation."""

        long_text = "This is a response without any tag prefix that keeps going for a while."

        async def mock_stream(*args, **kwargs):
            yield long_text

        mock_ollama.generate_stream = mock_stream

        request = ChatRequest(message="tell me something")
        chunks = []
        async for chunk in router.route_stream(request):
            chunks.append(chunk)

        full_text = "".join(c.content for c in chunks)
        assert long_text in full_text


# ---------------------------------------------------------------------------
# Cascade escalation
# ---------------------------------------------------------------------------


class TestCascadeEscalation:
    async def test_escalation_on_cant_access(self, registry, mock_alchemy):
        """If 14B says 'I can't access', cascade should reroute to Alchemy."""

        async def mock_stream(*args, **kwargs):
            yield "[CONVERSATION]\n"
            yield "I can't access your email directly. You'll need to open it yourself."

        mock_ollama = AsyncMock(spec=OllamaProvider)
        mock_ollama.generate_stream = mock_stream

        providers = {
            ModelLocation.GPU_LOCAL: mock_ollama,
            ModelLocation.CPU_REMOTE: mock_alchemy,
        }
        router = SmartRouter(
            registry=registry,
            providers=providers,
            conversation_manager=ConversationManager(),
            cascades=[ConversationToVisionCascade()],
        )

        request = ChatRequest(message="check my email for the invoice")
        chunks = []
        async for chunk in router.route_stream(request):
            chunks.append(chunk)

        full_text = "".join(c.content for c in chunks)
        assert "Task submitted" in full_text


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    async def test_no_conversation_model_registered(self, mock_alchemy):
        registry = ModelRegistry()
        providers = {ModelLocation.CPU_REMOTE: mock_alchemy}
        router = SmartRouter(
            registry=registry,
            providers=providers,
            conversation_manager=ConversationManager(),
        )

        request = ChatRequest(message="hello")
        chunks = []
        async for chunk in router.route_stream(request):
            chunks.append(chunk)

        assert any("No conversational model" in c.content for c in chunks)

    async def test_alchemy_unavailable(self, registry, mock_ollama):
        """GUI task with no Alchemy provider should gracefully degrade."""
        providers = {ModelLocation.GPU_LOCAL: mock_ollama}
        router = SmartRouter(
            registry=registry,
            providers=providers,
            conversation_manager=ConversationManager(),
        )

        request = ChatRequest(message="open Chrome")
        chunks = []
        async for chunk in router.route_stream(request):
            chunks.append(chunk)

        full_text = "".join(c.content for c in chunks)
        assert "background agent" in full_text.lower() or "isn't available" in full_text.lower()

    async def test_conversation_id_preserved(self, router, mock_ollama, mock_alchemy):
        cid = uuid4()
        request = ChatRequest(message="open Chrome", conversation_id=cid)
        response = await router.route(request)
        assert response.conversation_id == cid
