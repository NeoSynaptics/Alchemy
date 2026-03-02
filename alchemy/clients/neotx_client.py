"""Typed HTTP client for Alchemy → NEO-TX callbacks.

Used when Alchemy needs to notify NEO-TX about task updates,
approval requests, or NOTIFY-tier actions.
"""

from __future__ import annotations

import httpx

from alchemy.schemas import (
    ApprovalRequest,
    ApprovalRequestAck,
    NotifyAck,
    NotifyRequest,
    TaskUpdateAck,
    TaskUpdateRequest,
)


class NeoTXClient:
    """Async HTTP client for calling NEO-TX callback endpoints."""

    def __init__(self, base_url: str = "http://localhost:8100", timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    async def request_approval(self, req: ApprovalRequest) -> ApprovalRequestAck:
        """Ask NEO-TX to show an approval dialog to the user."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.base_url}/callbacks/approval",
                json=req.model_dump(mode="json"),
            )
            resp.raise_for_status()
            return ApprovalRequestAck.model_validate(resp.json())

    async def notify(self, req: NotifyRequest) -> NotifyAck:
        """Tell NEO-TX a NOTIFY-tier action was executed."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.base_url}/callbacks/notify",
                json=req.model_dump(mode="json"),
            )
            resp.raise_for_status()
            return NotifyAck.model_validate(resp.json())

    async def task_update(self, req: TaskUpdateRequest) -> TaskUpdateAck:
        """Report task status change to NEO-TX."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.base_url}/callbacks/task-update",
                json=req.model_dump(mode="json"),
            )
            resp.raise_for_status()
            return TaskUpdateAck.model_validate(resp.json())
