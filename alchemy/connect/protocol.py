"""AlchemyConnect protocol — message envelope for the tunnel.

Every message over the WebSocket uses this envelope. The `agent` field
routes to a registered ConnectAgent (or "system" for tunnel lifecycle).
"""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class AlchemyMessage:
    """Single message envelope for the AlchemyConnect tunnel."""

    agent: str                          # Target agent: "system", "chat", "browser", etc.
    type: str                           # Agent-specific action: "message", "auth", "scrape"
    payload: dict[str, Any] = field(default_factory=dict)
    v: int = 1                          # Protocol version
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    ts: float = field(default_factory=lambda: time.time() * 1000)
    ref: str | None = None              # Reply-to message ID
    seq: int | None = None              # Monotonic sequence for offline queue

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # Strip None optional fields for cleaner wire format
        return {k: v for k, v in d.items() if v is not None}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AlchemyMessage:
        """Parse a dict (from JSON) into an AlchemyMessage.

        Raises ValueError if required fields are missing or invalid.
        """
        if not isinstance(data, dict):
            raise ValueError("Message must be a JSON object")

        agent = data.get("agent")
        msg_type = data.get("type")
        if not agent or not isinstance(agent, str):
            raise ValueError("Missing or invalid 'agent' field")
        if not msg_type or not isinstance(msg_type, str):
            raise ValueError("Missing or invalid 'type' field")

        return cls(
            v=data.get("v", 1),
            id=data.get("id", uuid.uuid4().hex[:12]),
            agent=agent,
            type=msg_type,
            ts=data.get("ts", time.time() * 1000),
            payload=data.get("payload", {}),
            ref=data.get("ref"),
            seq=data.get("seq"),
        )

    def reply(self, type: str, payload: dict[str, Any] | None = None) -> AlchemyMessage:
        """Create a reply message referencing this message."""
        return AlchemyMessage(
            agent=self.agent,
            type=type,
            payload=payload or {},
            ref=self.id,
        )


def system_msg(type: str, payload: dict[str, Any] | None = None) -> AlchemyMessage:
    """Shortcut for creating system agent messages."""
    return AlchemyMessage(agent="system", type=type, payload=payload or {})
