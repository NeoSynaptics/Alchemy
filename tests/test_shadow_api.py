"""Shadow desktop endpoint tests — with mock controller."""

from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient

from alchemy.schemas import ShadowHealthResponse, ShadowStartResponse, ShadowStatus, ShadowStopResponse
from alchemy.server import app
from alchemy.shadow.controller import ShadowDesktopController


@pytest.fixture
async def client():
    # Inject a mock controller into app state
    mock_controller = AsyncMock(spec=ShadowDesktopController)
    mock_controller.start.return_value = ShadowStartResponse(
        status=ShadowStatus.RUNNING, display=":99",
        vnc_url="localhost:5900",
        novnc_url="http://localhost:6080/vnc.html?autoconnect=true",
    )
    mock_controller.stop.return_value = ShadowStopResponse(status=ShadowStatus.STOPPED)
    mock_controller.health.return_value = ShadowHealthResponse(
        status=ShadowStatus.STOPPED,
        xvfb_running=False, fluxbox_running=False,
        vnc_running=False, novnc_running=False,
    )
    mock_controller.screenshot.return_value = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100

    app.state.shadow_controller = mock_controller

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c

    # Cleanup
    app.state.shadow_controller = None


@pytest.mark.asyncio
async def test_shadow_start(client):
    resp = await client.post("/v1/shadow/start", json={
        "resolution": "1920x1080x24", "display_num": 99,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "running"
    assert data["display"] == ":99"
    assert "novnc_url" in data


@pytest.mark.asyncio
async def test_shadow_start_defaults(client):
    resp = await client.post("/v1/shadow/start")
    assert resp.status_code == 200
    assert resp.json()["status"] == "running"


@pytest.mark.asyncio
async def test_shadow_stop(client):
    resp = await client.post("/v1/shadow/stop")
    assert resp.status_code == 200
    assert resp.json()["status"] == "stopped"


@pytest.mark.asyncio
async def test_shadow_health(client):
    resp = await client.get("/v1/shadow/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "xvfb_running" in data
    assert "fluxbox_running" in data
    assert "vnc_running" in data
    assert "novnc_running" in data


@pytest.mark.asyncio
async def test_screenshot_returns_png(client):
    resp = await client.get("/v1/shadow/screenshot")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"
    assert resp.content[:4] == b"\x89PNG"
