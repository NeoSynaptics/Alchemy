"""Models endpoint tests — with mock Ollama."""

from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient

from alchemy.models.ollama_client import OllamaClient
from alchemy.server import app


@pytest.fixture
async def client():
    mock_ollama = AsyncMock(spec=OllamaClient)
    mock_ollama.list_models = AsyncMock(return_value=[
        {"name": "avil/UI-TARS:latest", "size": 3600000000},
        {"name": "qwen2.5-coder:32b", "size": 19000000000},
    ])
    app.state.ollama_client = mock_ollama

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c

    app.state.ollama_client = None


@pytest.mark.asyncio
async def test_get_models(client):
    resp = await client.get("/models")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["models"]) == 2
    assert data["models"][0]["name"] == "avil/UI-TARS:latest"
    assert data["total_ram_gb"] > 0
    assert data["available_ram_gb"] > 0
