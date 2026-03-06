"""Typed HTTP client for voice callback endpoints.

Used when the click agent needs to notify the voice/tray system about
task updates, approval requests, or NOTIFY-tier actions.
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


class VoiceCallbackClient:
    """Async HTTP client for calling voice callback endpoints."""

    def __init__(self, base_url: str = "http://localhost:8100", timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout,
        )

    async def close(self) -> None:
        """Close the underlying HTTP connection pool."""
        await self._client.aclose()

    async def request_approval(self, req: ApprovalRequest) -> ApprovalRequestAck:
        """Show an approval dialog to the user via voice/tray."""
        resp = await self._client.post(
            "/v1/callbacks/approval",
            json=req.model_dump(mode="json"),
        )
        resp.raise_for_status()
        return ApprovalRequestAck.model_validate(resp.json())

    async def notify(self, req: NotifyRequest) -> NotifyAck:
        """Notify that a NOTIFY-tier action was executed."""
        resp = await self._client.post(
            "/v1/callbacks/notify",
            json=req.model_dump(mode="json"),
        )
        resp.raise_for_status()
        return NotifyAck.model_validate(resp.json())

    async def task_update(self, req: TaskUpdateRequest) -> TaskUpdateAck:
        """Report task status change."""
        resp = await self._client.post(
            "/v1/callbacks/task-update",
            json=req.model_dump(mode="json"),
        )
        resp.raise_for_status()
        return TaskUpdateAck.model_validate(resp.json())
