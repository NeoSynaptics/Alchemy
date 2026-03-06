"""Fire-and-forget event reporter — sends events to NEO-RX timeline.

Best-effort: if NEO-RX is down, events are silently dropped.
Same pattern as Alchemy callback forwarding.
"""

from __future__ import annotations

import logging
from uuid import UUID

import httpx

logger = logging.getLogger(__name__)


class EventReporter:
    """Reports AlchemyVoice events (conversations, voice) to NEO-RX timeline."""

    def __init__(
        self,
        neorx_host: str = "http://localhost:8110",
        timeout: float = 3.0,
    ):
        self._host = neorx_host.rstrip("/")
        self._client = httpx.AsyncClient(timeout=timeout)

    async def report_conversation(
        self,
        conversation_id: UUID | str,
        role: str,
        content: str,
        source: str = "alchemy_voice_chat",
    ) -> None:
        """Report a conversation turn to NEO-RX (fire-and-forget)."""
        try:
            await self._client.post(
                f"{self._host}/v1/callbacks/event",
                json={
                    "source": source,
                    "event_type": "conversation_turn",
                    "content": content,
                    "metadata": {
                        "role": role,
                        "conversation_id": str(conversation_id),
                    },
                },
            )
        except Exception:
            pass  # Best-effort, same as Alchemy callbacks

    async def report_voice(
        self,
        transcript: str,
        conversation_id: UUID | str | None = None,
    ) -> None:
        """Report a voice transcript to NEO-RX."""
        try:
            metadata = {}
            if conversation_id:
                metadata["conversation_id"] = str(conversation_id)
            await self._client.post(
                f"{self._host}/v1/callbacks/event",
                json={
                    "source": "alchemy_voice",
                    "event_type": "voice_transcript",
                    "content": transcript,
                    "metadata": metadata,
                },
            )
        except Exception:
            pass

    async def close(self) -> None:
        await self._client.aclose()
