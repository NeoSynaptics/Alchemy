"""RLHF reaction logging — captures voice responses + user reactions for preference data.

Voice responses and user reactions are logged to NEOSY registro via HTTP,
creating a preference dataset for future DPO/RLHF fine-tuning.
"""

from __future__ import annotations

import asyncio
import logging
from enum import Enum
from uuid import UUID

import httpx
from pydantic import BaseModel, Field

from config.settings import settings

logger = logging.getLogger(__name__)


# ── Schemas ─────────────────────────────────────────────────


class ReactionSentiment(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class VoiceReaction(BaseModel):
    """User reaction to a voice response."""
    conversation_id: str | UUID
    sentiment: ReactionSentiment
    intensity: float = Field(ge=0.0, le=1.0, default=0.5)
    reason: str | None = None  # explicit: "that's funny", "wrong answer"
    source: str = "explicit"   # "explicit" | "implicit_laugh" | "implicit_silence"


class VoiceResponseLog(BaseModel):
    """Logged after each voice turn for RLHF dataset building."""
    conversation_id: str | UUID
    user_query: str
    response_text: str
    model_used: str
    inference_ms: float
    route_intent: str = "conversation"


# ── Logger ──────────────────────────────────────────────────


class ReactionLogger:
    """Async fire-and-forget logger that sends voice turns + reactions to NEOSY."""

    def __init__(self, neosy_url: str | None = None):
        self._neosy_url = neosy_url or getattr(settings, "neosy_url", "http://localhost:8001")
        self._client: httpx.AsyncClient | None = None

    async def start(self):
        self._client = httpx.AsyncClient(base_url=self._neosy_url, timeout=10.0)

    async def stop(self):
        if self._client:
            await self._client.aclose()
            self._client = None

    async def log_voice_response(self, log: VoiceResponseLog) -> None:
        """Log a voice response turn to NEOSY registro.

        Creates a memory record with the conversation context,
        then logs a registro entry with action=voice_response.
        """
        if not self._client:
            return

        try:
            # Ingest the conversation turn as a NEOSY memory
            ingest_resp = await self._client.post("/ingest", json={
                "text": f"[Voice Turn]\nUser: {log.user_query}\nAssistant: {log.response_text}",
                "title": f"Voice conversation ({log.model_used})",
                "source_type": "voice_response",
                "properties": {
                    "conversation_id": str(log.conversation_id),
                    "model_used": log.model_used,
                    "inference_ms": log.inference_ms,
                    "route_intent": log.route_intent,
                },
            })

            if ingest_resp.status_code == 200:
                data = ingest_resp.json()
                memory_id = data.get("memory_id")
                logger.debug("Logged voice turn %s → memory %s", log.conversation_id, memory_id)
            else:
                logger.warning("NEOSY ingest failed: %s", ingest_resp.text)

        except httpx.ConnectError:
            logger.debug("NEOSY not reachable — voice logging skipped")
        except Exception:
            logger.exception("Failed to log voice response")

    async def log_reaction(self, reaction: VoiceReaction) -> None:
        """Log a user reaction to NEOSY registro.

        Stores sentiment + intensity as a registro entry linked to the conversation.
        This creates the preference signal for future DPO/RLHF.
        """
        if not self._client:
            return

        try:
            # Store as a separate memory with reaction metadata
            sentiment_val = {
                ReactionSentiment.POSITIVE: 1.0,
                ReactionSentiment.NEGATIVE: -1.0,
                ReactionSentiment.NEUTRAL: 0.0,
            }[reaction.sentiment]

            await self._client.post("/ingest", json={
                "text": f"[Reaction] {reaction.sentiment.value} (intensity={reaction.intensity})"
                        + (f" — {reaction.reason}" if reaction.reason else ""),
                "title": f"User reaction: {reaction.sentiment.value}",
                "source_type": "user_reaction",
                "properties": {
                    "conversation_id": str(reaction.conversation_id),
                    "sentiment": reaction.sentiment.value,
                    "sentiment_score": sentiment_val * reaction.intensity,
                    "intensity": reaction.intensity,
                    "source": reaction.source,
                    "reason": reaction.reason,
                },
            })

            logger.info(
                "Reaction logged: %s (%.1f) for %s",
                reaction.sentiment.value, reaction.intensity, reaction.conversation_id,
            )

        except httpx.ConnectError:
            logger.debug("NEOSY not reachable — reaction logging skipped")
        except Exception:
            logger.exception("Failed to log reaction")


# ── Singleton ───────────────────────────────────────────────

_logger: ReactionLogger | None = None


def get_reaction_logger() -> ReactionLogger | None:
    return _logger


async def init_reaction_logger() -> ReactionLogger:
    global _logger
    _logger = ReactionLogger()
    await _logger.start()
    return _logger


async def shutdown_reaction_logger() -> None:
    global _logger
    if _logger:
        await _logger.stop()
        _logger = None
