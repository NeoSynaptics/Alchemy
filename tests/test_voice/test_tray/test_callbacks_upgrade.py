"""Tests for upgraded callback endpoints — event bus integration."""

import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from alchemy.voice.schemas import ActionTier, VisionAction
from alchemy.server import app
from alchemy.voice.tray.events import TrayEvent, TrayEventBus


@pytest_asyncio.fixture
async def client_with_tray():
    """Client with a mocked tray event bus."""
    bus = TrayEventBus()
    bus.bind_loop(asyncio.get_event_loop())

    mock_alchemy = AsyncMock()
    mock_alchemy.approve_task = AsyncMock()
    mock_alchemy.deny_task = AsyncMock()

    app.state.tray_event_bus = bus
    app.state.alchemy_client = mock_alchemy

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c, bus, mock_alchemy


@pytest_asyncio.fixture
async def client_no_tray():
    """Client with no tray (tray disabled)."""
    mock_alchemy = AsyncMock()
    mock_alchemy.approve_task = AsyncMock()

    app.state.tray_event_bus = None
    app.state.alchemy_client = mock_alchemy

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c, mock_alchemy


class TestApprovalCallback:
    async def test_approval_pushed_to_event_bus(self, client_with_tray):
        client, bus, alchemy = client_with_tray

        task_id = uuid4()

        # Simulate user approving immediately in background
        async def approve_after_delay():
            await asyncio.sleep(0.05)
            msg = bus.poll()
            if msg:
                bus.respond(msg, approved=True)

        asyncio.create_task(approve_after_delay())

        resp = await client.post(
            "/v1/callbacks/approval",
            json={
                "task_id": str(task_id),
                "action": {"action": "click", "x": 100, "y": 200, "reasoning": "test"},
                "screenshot_b64": "",
                "step": 1,
                "timeout_seconds": 5,
                "goal": "Test goal",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["received"] is True

        # Alchemy should have been told to approve
        alchemy.approve_task.assert_called_once()

    async def test_approval_deny(self, client_with_tray):
        client, bus, alchemy = client_with_tray
        task_id = uuid4()

        async def deny_after_delay():
            await asyncio.sleep(0.05)
            msg = bus.poll()
            if msg:
                bus.respond(msg, approved=False)

        asyncio.create_task(deny_after_delay())

        resp = await client.post(
            "/v1/callbacks/approval",
            json={
                "task_id": str(task_id),
                "action": {"action": "click", "x": 100, "y": 200},
                "screenshot_b64": "",
                "step": 1,
                "timeout_seconds": 5,
                "goal": "Test",
            },
        )
        assert resp.status_code == 200
        alchemy.deny_task.assert_called_once()

    async def test_approval_timeout_auto_denies(self, client_with_tray):
        client, bus, alchemy = client_with_tray
        task_id = uuid4()

        # Don't respond — let it timeout (1 second)
        resp = await client.post(
            "/v1/callbacks/approval",
            json={
                "task_id": str(task_id),
                "action": {"action": "click", "x": 100, "y": 200},
                "screenshot_b64": "",
                "step": 1,
                "timeout_seconds": 1,
                "goal": "Timeout test",
            },
        )
        assert resp.status_code == 200
        alchemy.deny_task.assert_called_once()

    async def test_no_tray_auto_approves(self, client_no_tray):
        client, alchemy = client_no_tray
        task_id = uuid4()

        resp = await client.post(
            "/v1/callbacks/approval",
            json={
                "task_id": str(task_id),
                "action": {"action": "click", "x": 100, "y": 200},
                "screenshot_b64": "",
                "step": 1,
                "goal": "Auto-approve test",
            },
        )
        assert resp.status_code == 200
        alchemy.approve_task.assert_called_once()


class TestNotifyCallback:
    async def test_notify_pushed_to_bus(self, client_with_tray):
        client, bus, _ = client_with_tray
        task_id = uuid4()

        resp = await client.post(
            "/v1/callbacks/notify",
            json={
                "task_id": str(task_id),
                "action": {"action": "type", "text": "hello"},
                "message": "Typing text",
                "step": 2,
            },
        )
        assert resp.status_code == 200
        assert resp.json()["received"] is True

        msg = bus.poll()
        assert msg is not None
        assert msg.event == TrayEvent.NOTIFICATION

    async def test_notify_no_tray_still_acks(self, client_no_tray):
        client, _ = client_no_tray

        resp = await client.post(
            "/v1/callbacks/notify",
            json={
                "task_id": str(uuid4()),
                "action": {"action": "type"},
                "message": "test",
                "step": 1,
            },
        )
        assert resp.status_code == 200


class TestTaskUpdateCallback:
    async def test_update_pushed_to_bus(self, client_with_tray):
        client, bus, _ = client_with_tray

        resp = await client.post(
            "/v1/callbacks/task-update",
            json={
                "task_id": str(uuid4()),
                "status": "running",
                "current_step": 3,
                "message": "Clicking button",
            },
        )
        assert resp.status_code == 200

        msg = bus.poll()
        assert msg is not None
        assert msg.event == TrayEvent.TASK_UPDATE
        assert msg.data["status"] == "running"
