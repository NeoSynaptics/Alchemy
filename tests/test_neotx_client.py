"""NeoTXClient unit tests — mock HTTP responses."""

from unittest.mock import AsyncMock, patch
from uuid import uuid4

import httpx
import pytest

from alchemy.clients.neotx_client import NeoTXClient
from alchemy.schemas import (
    ActionTier,
    ApprovalRequest,
    NotifyRequest,
    TaskStatus,
    TaskUpdateRequest,
    VisionAction,
)


@pytest.fixture
def client():
    return NeoTXClient(base_url="http://test:8100")


def _action() -> VisionAction:
    return VisionAction(
        action="click", x=100, y=200,
        reasoning="Click the button", tier=ActionTier.AUTO,
    )


class TestRequestApproval:
    @pytest.mark.asyncio
    async def test_success(self, client):
        task_id = uuid4()
        mock_resp = httpx.Response(
            200,
            json={"received": True, "task_id": str(task_id)},
            request=httpx.Request("POST", "http://test"),
        )
        client._client.post = AsyncMock(return_value=mock_resp)

        req = ApprovalRequest(
            task_id=task_id, action=_action(),
            screenshot_b64="abc", step=1, goal="test",
        )
        ack = await client.request_approval(req)
        assert ack.received is True
        assert ack.task_id == task_id

    @pytest.mark.asyncio
    async def test_http_error(self, client):
        mock_resp = httpx.Response(
            500, json={},
            request=httpx.Request("POST", "http://test"),
        )
        client._client.post = AsyncMock(return_value=mock_resp)

        req = ApprovalRequest(
            task_id=uuid4(), action=_action(),
            screenshot_b64="abc", step=1, goal="test",
        )
        with pytest.raises(httpx.HTTPStatusError):
            await client.request_approval(req)


class TestNotify:
    @pytest.mark.asyncio
    async def test_success(self, client):
        mock_resp = httpx.Response(
            200, json={"received": True},
            request=httpx.Request("POST", "http://test"),
        )
        client._client.post = AsyncMock(return_value=mock_resp)

        req = NotifyRequest(
            task_id=uuid4(), action=_action(),
            message="Step 1 done", step=1,
        )
        ack = await client.notify(req)
        assert ack.received is True


class TestTaskUpdate:
    @pytest.mark.asyncio
    async def test_success(self, client):
        task_id = uuid4()
        mock_resp = httpx.Response(
            200, json={"received": True},
            request=httpx.Request("POST", "http://test"),
        )
        client._client.post = AsyncMock(return_value=mock_resp)

        req = TaskUpdateRequest(
            task_id=task_id, status=TaskStatus.RUNNING,
            current_step=3, last_action=_action(),
        )
        ack = await client.task_update(req)
        assert ack.received is True


class TestClose:
    @pytest.mark.asyncio
    async def test_close(self, client):
        client._client.aclose = AsyncMock()
        await client.close()
        client._client.aclose.assert_awaited_once()
