"""Vision endpoint tests — with mock Ollama + controller."""

from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from alchemy.agent.task_manager import TaskManager
from alchemy.models.ollama_client import OllamaClient
from alchemy.schemas import ShadowStatus
from alchemy.server import app
from alchemy.shadow.controller import ShadowDesktopController


@pytest.fixture
async def client():
    # Inject mock dependencies into app state
    mock_ollama = AsyncMock(spec=OllamaClient)
    mock_ollama.chat = AsyncMock(return_value={
        "message": {"role": "assistant", "content": "Thought: Click.\nAction: click(start_box='(500,500)')"},
        "total_duration": 2500000000,
        "eval_count": 15,
    })
    mock_ollama.chat_stream = AsyncMock(
        return_value="Thought: Click.\nAction: click(start_box='(500,500)')"
    )
    mock_ollama.ping = AsyncMock(return_value=True)
    mock_ollama.list_models = AsyncMock(return_value=[])

    mock_controller = AsyncMock(spec=ShadowDesktopController)
    mock_controller.screenshot = AsyncMock(return_value=b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
    mock_controller.execute = AsyncMock(return_value="")

    task_manager = TaskManager()

    app.state.ollama_client = mock_ollama
    app.state.shadow_controller = mock_controller
    app.state.task_manager = task_manager

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c

    # Cleanup
    app.state.ollama_client = None
    app.state.shadow_controller = None
    app.state.task_manager = None


@pytest.mark.asyncio
async def test_create_task(client):
    resp = await client.post("/v1/vision/task", json={
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
    resp = await client.post("/v1/vision/analyze", json={
        "screenshot_b64": "iVBORw0KGgo=",
        "goal": "find the search box",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["action"]["action"] == "click"
    assert data["inference_ms"] > 0


@pytest.mark.asyncio
async def test_task_status(client):
    create = await client.post("/v1/vision/task", json={"goal": "test"})
    task_id = create.json()["task_id"]

    resp = await client.get(f"/v1/vision/task/{task_id}/status")
    assert resp.status_code == 200
    assert resp.json()["task_id"] == task_id


@pytest.mark.asyncio
async def test_task_status_not_found(client):
    resp = await client.get("/v1/vision/task/00000000-0000-0000-0000-000000000001/status")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_approve_task(client):
    create = await client.post("/v1/vision/task", json={"goal": "test approval"})
    task_id = create.json()["task_id"]

    resp = await client.post(f"/v1/vision/task/{task_id}/approve", json={
        "decided_by": "user", "reason": "looks good",
    })
    assert resp.status_code == 200
    assert resp.json()["decision"] == "approved"


@pytest.mark.asyncio
async def test_deny_task(client):
    create = await client.post("/v1/vision/task", json={"goal": "test denial"})
    task_id = create.json()["task_id"]

    resp = await client.post(f"/v1/vision/task/{task_id}/deny", json={
        "decided_by": "user", "reason": "wrong recipient",
    })
    assert resp.status_code == 200
    assert resp.json()["decision"] == "denied"
