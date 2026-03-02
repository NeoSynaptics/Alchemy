"""Models endpoint tests."""

import pytest
from httpx import ASGITransport, AsyncClient

from alchemy.server import app


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.mark.asyncio
async def test_get_models(client):
    resp = await client.get("/models")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["models"]) >= 1
    assert data["models"][0]["name"] == "ui-tars:72b"
    assert data["total_ram_gb"] > 0
    assert data["available_ram_gb"] > 0
