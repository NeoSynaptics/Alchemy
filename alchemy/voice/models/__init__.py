"""GPU model management — registry, providers, conversation context.

AlchemyVoice owns the GPU models for fast user interaction:
- 14B conversational model (semantic understanding, NOT coding)
- Small specialized models (~2B) for specific fast tasks
"""

from alchemy.voice.models.conversation import ConversationManager
from alchemy.voice.models.provider import (
    AlchemyProvider,
    GatewayProvider,
    ModelProvider,
    OllamaProvider,
)
from alchemy.voice.models.registry import ModelRegistry, build_default_registry
from alchemy.voice.models.schemas import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ModelCapability,
    ModelCard,
    ModelLocation,
    RouteDecision,
    RouteIntent,
    SpeedTier,
    StreamChunk,
)

__all__ = [
    "ConversationManager",
    "AlchemyProvider",
    "GatewayProvider",
    "ModelProvider",
    "OllamaProvider",
    "ModelRegistry",
    "build_default_registry",
    "ChatMessage",
    "ChatRequest",
    "ChatResponse",
    "ModelCapability",
    "ModelCard",
    "ModelLocation",
    "RouteDecision",
    "RouteIntent",
    "SpeedTier",
    "StreamChunk",
]
