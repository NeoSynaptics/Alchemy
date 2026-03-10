"""Desktop API endpoint tests — /v1/desktop/* routes."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from alchemy.desktop.agent import DesktopTaskResult, DesktopTaskStatus
from alchemy.server import app


def _mock_desktop_agent(mode: str = "shadow"):
    """Create a mock desktop agent with controller."""
    mock_controller = MagicMock()
    mock_controller.mode = mode
    mock_controller.summon = MagicMock()
    mock_controller.dismiss = MagicMock()
    mock_controller.screenshot = AsyncMock(return_value=b"\x89PNG")

    agent = MagicMock()
    agent._controller = mock_controller
    agent.run = AsyncMock(return_value=DesktopTaskResult(
        status=DesktopTaskStatus.COMPLETED,
        steps=[],
        total_ms=100.0,
        error=None,
    ))
    return agent


@pytest_asyncio.fixture
async def client():
    app.state.contract_reports = {}
    app.state.desktop_agent = _mock_desktop_agent()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c

    # Clean up in-memory task store
    from alchemy.api.desktop_api import _tasks
    _tasks.clear()
    app.state.contract_reports = {}
    app.state.desktop_agent = None


@pytest_asyncio.fixture
async def client_no_agent():
    """Client without desktop agent → 503."""
    app.state.contract_reports = {}
    app.state.desktop_agent = None

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c

    app.state.contract_reports = {}


# --- POST /v1/desktop/task ---


@pytest.mark.asyncio
async def test_submit_task_returns_pending(client):
    resp = await client.post("/v1/desktop/task", json={"goal": "open notepad"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "pending"
    assert "task_id" in data
    assert "created_at" in data


@pytest.mark.asyncio
async def test_submit_task_with_ghost_mode(client):
    resp = await client.post("/v1/desktop/task", json={
        "goal": "click button",
        "mode": "ghost",
    })
    assert resp.status_code == 200
    app.state.desktop_agent._controller.summon.assert_called_once()


@pytest.mark.asyncio
async def test_submit_task_with_shadow_mode(client):
    resp = await client.post("/v1/desktop/task", json={
        "goal": "click button",
        "mode": "shadow",
    })
    assert resp.status_code == 200
    app.state.desktop_agent._controller.dismiss.assert_called_once()


@pytest.mark.asyncio
async def test_submit_task_custom_max_steps(client):
    resp = await client.post("/v1/desktop/task", json={
        "goal": "do stuff",
        "max_steps": 5,
    })
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_submit_task_missing_goal(client):
    resp = await client.post("/v1/desktop/task", json={})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_submit_task_no_agent_returns_503(client_no_agent):
    resp = await client_no_agent.post("/v1/desktop/task", json={"goal": "test"})
    assert resp.status_code == 503
    assert "not available" in resp.json()["detail"]


# --- GET /v1/desktop/task/{task_id} ---


@pytest.mark.asyncio
async def test_get_task_status(client):
    # Submit first
    resp = await client.post("/v1/desktop/task", json={"goal": "test"})
    task_id = resp.json()["task_id"]

    # Let background task complete
    await asyncio.sleep(0.05)

    # Poll status
    resp = await client.get(f"/v1/desktop/task/{task_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["task_id"] == task_id
    assert data["status"] in ("pending", "running", "completed", "failed")


@pytest.mark.asyncio
async def test_get_task_status_not_found(client):
    resp = await client.get("/v1/desktop/task/nonexistent-id")
    assert resp.status_code == 404


# --- POST /v1/desktop/summon ---


@pytest.mark.asyncio
async def test_summon(client):
    resp = await client.post("/v1/desktop/summon")
    assert resp.status_code == 200
    data = resp.json()
    assert data["mode"] == "ghost"
    app.state.desktop_agent._controller.summon.assert_called_once()


@pytest.mark.asyncio
async def test_summon_no_agent(client_no_agent):
    resp = await client_no_agent.post("/v1/desktop/summon")
    assert resp.status_code == 503


# --- POST /v1/desktop/dismiss ---


@pytest.mark.asyncio
async def test_dismiss(client):
    resp = await client.post("/v1/desktop/dismiss")
    assert resp.status_code == 200
    data = resp.json()
    assert data["mode"] == "shadow"
    app.state.desktop_agent._controller.dismiss.assert_called_once()


@pytest.mark.asyncio
async def test_dismiss_no_agent(client_no_agent):
    resp = await client_no_agent.post("/v1/desktop/dismiss")
    assert resp.status_code == 503


# --- GET /v1/desktop/mode ---


@pytest.mark.asyncio
async def test_get_mode(client):
    resp = await client.get("/v1/desktop/mode")
    assert resp.status_code == 200
    assert resp.json()["mode"] == "shadow"


@pytest.mark.asyncio
async def test_get_mode_no_agent(client_no_agent):
    resp = await client_no_agent.get("/v1/desktop/mode")
    assert resp.status_code == 503


# --- Contract guard ---


@pytest.mark.asyncio
async def test_contract_unsatisfied():
    mock_report = MagicMock()
    mock_report.satisfied = False
    mock_report.missing = ["vision"]
    app.state.contract_reports = {"desktop": mock_report}
    app.state.desktop_agent = _mock_desktop_agent()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.post("/v1/desktop/task", json={"goal": "test"})
        assert resp.status_code == 503
        assert resp.json()["detail"]["error"] == "model_contract_unsatisfied"

        resp = await c.post("/v1/desktop/summon")
        assert resp.status_code == 503

        resp = await c.get("/v1/desktop/mode")
        assert resp.status_code == 503

    app.state.contract_reports = {}
    app.state.desktop_agent = None
