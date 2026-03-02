"""Vision endpoint tests — verify stub endpoints return correct types."""

import pytest
from httpx import ASGITransport, AsyncClient

from alchemy.server import app


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.mark.asyncio
async def test_create_task(client):
    resp = await client.post("/vision/task", json={
        "goal": "send email with hours",
        "callback_url": "http://localhost:8100",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "task_id" in data
    assert data["status"] == "pending"
    assert "created_at" in data


@pytest.mark.asyncio
async def test_analyze(client):
    resp = await client.post("/vision/analyze", json={
        "screenshot_b64": "iVBORw0KGgo=",
        "goal": "find the search box",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["action"]["action"] == "click"
    assert data["model"] == "ui-tars:72b"
    assert data["inference_ms"] > 0


@pytest.mark.asyncio
async def test_task_status(client):
    # Create task first
    create = await client.post("/vision/task", json={"goal": "test"})
    task_id = create.json()["task_id"]

    resp = await client.get(f"/vision/task/{task_id}/status")
    assert resp.status_code == 200
    assert resp.json()["task_id"] == task_id
    assert resp.json()["status"] == "pending"


@pytest.mark.asyncio
async def test_task_status_not_found(client):
    resp = await client.get("/vision/task/00000000-0000-0000-0000-000000000001/status")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_approve_task(client):
    create = await client.post("/vision/task", json={"goal": "test approval"})
    task_id = create.json()["task_id"]

    resp = await client.post(f"/vision/task/{task_id}/approve", json={
        "decided_by": "user", "reason": "looks good",
    })
    assert resp.status_code == 200
    assert resp.json()["decision"] == "approved"
    assert resp.json()["status"] == "running"


@pytest.mark.asyncio
async def test_deny_task(client):
    create = await client.post("/vision/task", json={"goal": "test denial"})
    task_id = create.json()["task_id"]

    resp = await client.post(f"/vision/task/{task_id}/deny", json={
        "decided_by": "user", "reason": "wrong recipient",
    })
    assert resp.status_code == 200
    assert resp.json()["decision"] == "denied"
    assert resp.json()["status"] == "denied"
