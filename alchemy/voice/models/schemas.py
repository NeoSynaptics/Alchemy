"""Data models for the smart routing layer.

Defines model capabilities, routing intents, chat messages, and streaming chunks.
Separate from alchemy/voice/schemas.py (which holds the voice API contract schemas).
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Model metadata
# ---------------------------------------------------------------------------


class ModelCapability(str, Enum):
    """What a model can do. Used for hard routing gates."""

    CONVERSATION = "conversation"
    VISION = "vision"
    CLASSIFICATION = "classification"
    VOICE_STT = "voice_stt"
    VOICE_TTS = "voice_tts"


class SpeedTier(str, Enum):
    """How fast a model responds. Affects UX decisions."""

    FAST = "fast"  # >20 tok/s — stream directly to user
    MEDIUM = "medium"  # 5-20 tok/s — show spinner, buffer
    SLOW = "slow"  # <5 tok/s — background task, notify on done


class ModelLocation(str, Enum):
    """Where the model runs. Determines which provider handles it."""

    GPU_LOCAL = "gpu_local"  # This machine's GPU via Ollama
    CPU_REMOTE = "cpu_remote"  # Alchemy (CPU-side) via HTTP
    CPU_LOCAL = "cpu_local"  # This machine's CPU (Piper TTS, etc.)


class ModelCard(BaseModel):
    """Registration card for a model in the registry."""

    name: str  # e.g. "qwen3:14b"
    capabilities: list[ModelCapability]
    speed_tier: SpeedTier
    location: ModelLocation
    endpoint: str | None = None  # Override host for this model (dual-host support)
    vram_gb: float = 0.0
    ram_gb: float = 0.0
    keep_alive: str = "30m"
    max_context_tokens: int = 32768
    is_default_for: list[ModelCapability] = []


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


class RouteIntent(str, Enum):
    """Classified intent of a user message."""

    CONVERSATION = "conversation"
    GUI_TASK = "gui_task"
    SYSTEM_COMMAND = "system_command"
    UNCLEAR = "unclear"


class RouteDecision(BaseModel):
    """The router's output: where to send this request."""

    intent: RouteIntent
    target_model: str
    target_location: ModelLocation
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = ""
    should_stream: bool = True
    escalation_possible: bool = False


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    """A single message in a conversation."""

    role: str  # "user", "assistant", "system"
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)


class ChatRequest(BaseModel):
    """Incoming chat request from user/voice/tray."""

    message: str
    conversation_id: UUID = Field(default_factory=uuid4)
    source: str = "api"  # "api", "voice", "tray"
    context: dict | None = None


class ChatResponse(BaseModel):
    """Response back to user."""

    message: str
    conversation_id: UUID
    model_used: str
    route_decision: RouteDecision
    inference_ms: float = 0.0


class StreamChunk(BaseModel):
    """Single chunk in a streaming response."""

    content: str
    done: bool = False
    model_used: str = ""
    inference_ms: float = 0.0
