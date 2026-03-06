"""Callback endpoint tests — verify AlchemyVoice accepts Alchemy callbacks."""

import pytest
from httpx import ASGITransport, AsyncClient

from alchemy.server import app


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.mark.asyncio
async def test_approval_callback(client):
    resp = await client.post("/v1/callbacks/approval", json={
        "task_id": "00000000-0000-0000-0000-000000000001",
        "action": {
            "action": "click", "x": 340, "y": 200,
            "reasoning": "Click Send button", "tier": "approve",
        },
        "screenshot_b64": "iVBORw0KGgo=",
        "step": 3,
        "timeout_seconds": 60,
        "goal": "send email with hours",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["received"] is True
    assert data["task_id"] == "00000000-0000-0000-0000-000000000001"


@pytest.mark.asyncio
async def test_notify_callback(client):
    resp = await client.post("/v1/callbacks/notify", json={
        "task_id": "00000000-0000-0000-0000-000000000001",
        "action": {
            "action": "click", "x": 50, "y": 50,
            "reasoning": "Opening Firefox", "tier": "notify",
        },
        "message": "Opening Firefox",
        "step": 1,
    })
    assert resp.status_code == 200
    assert resp.json()["received"] is True


@pytest.mark.asyncio
async def test_task_update_callback(client):
    resp = await client.post("/v1/callbacks/task-update", json={
        "task_id": "00000000-0000-0000-0000-000000000001",
        "status": "completed",
        "current_step": 5,
        "message": "Task finished successfully",
    })
    assert resp.status_code == 200
    assert resp.json()["received"] is True


@pytest.mark.asyncio
async def test_health(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
