"""Shadow desktop endpoint tests."""

import pytest
from httpx import ASGITransport, AsyncClient

from alchemy.server import app


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.mark.asyncio
async def test_shadow_start(client):
    resp = await client.post("/shadow/start", json={
        "resolution": "1920x1080x24", "display_num": 99,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "running"
    assert data["display"] == ":99"
    assert "novnc_url" in data


@pytest.mark.asyncio
async def test_shadow_start_defaults(client):
    resp = await client.post("/shadow/start")
    assert resp.status_code == 200
    assert resp.json()["status"] == "running"


@pytest.mark.asyncio
async def test_shadow_stop(client):
    resp = await client.post("/shadow/stop")
    assert resp.status_code == 200
    assert resp.json()["status"] == "stopped"


@pytest.mark.asyncio
async def test_shadow_health(client):
    resp = await client.get("/shadow/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "xvfb_running" in data
    assert "fluxbox_running" in data
    assert "vnc_running" in data
    assert "novnc_running" in data


@pytest.mark.asyncio
async def test_screenshot_returns_png(client):
    resp = await client.get("/shadow/screenshot")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"
    assert resp.content[:4] == b"\x89PNG"
