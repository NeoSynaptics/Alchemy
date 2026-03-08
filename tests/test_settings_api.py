"""Tests for GET/PATCH /v1/settings API."""

import pytest
from httpx import ASGITransport, AsyncClient


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def client():
    # Import app lazily to avoid heavy startup (lifespan disabled)
    from alchemy.api.settings_api import router
    from fastapi import FastAPI

    test_app = FastAPI()
    test_app.include_router(router, prefix="/v1")

    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


class TestGetSettings:
    async def test_returns_nested_groups(self, client):
        resp = await client.get("/v1/settings")
        assert resp.status_code == 200
        data = resp.json()
        assert "ollama" in data
        assert "pw" in data
        assert "voice" in data
        assert "agents" in data
        # Flat compat fields should NOT appear at top level
        assert "ollama_host" not in data
        assert "pw_enabled" not in data

    async def test_ollama_has_expected_fields(self, client):
        resp = await client.get("/v1/settings")
        ollama = resp.json()["ollama"]
        assert "host" in ollama
        assert "temperature" in ollama

    async def test_voice_has_expected_fields(self, client):
        resp = await client.get("/v1/settings")
        voice = resp.json()["voice"]
        assert "tts_engine" in voice
        assert "enabled" in voice


class TestPatchSettings:
    async def test_patch_single_field(self, client):
        resp = await client.patch("/v1/settings", json={
            "pw": {"temperature": 0.99},
        })
        assert resp.status_code == 200
        assert "pw.temperature" in resp.json()["updated"]

        # Verify the change stuck
        get_resp = await client.get("/v1/settings")
        assert get_resp.json()["pw"]["temperature"] == 0.99

    async def test_patch_multiple_groups(self, client):
        resp = await client.patch("/v1/settings", json={
            "ollama": {"temperature": 0.5},
            "desktop": {"max_steps": 99},
        })
        assert resp.status_code == 200
        updated = resp.json()["updated"]
        assert "ollama.temperature" in updated
        assert "desktop.max_steps" in updated

    async def test_patch_unknown_group_returns_422(self, client):
        resp = await client.patch("/v1/settings", json={
            "nonexistent": {"foo": "bar"},
        })
        assert resp.status_code == 422

    async def test_patch_invalid_value_returns_422(self, client):
        resp = await client.patch("/v1/settings", json={
            "pw": {"temperature": "not-a-number"},
        })
        assert resp.status_code == 422

    async def test_patch_preserves_other_fields(self, client):
        # Get current model value
        before = (await client.get("/v1/settings")).json()["pw"]["model"]

        # Patch only temperature
        await client.patch("/v1/settings", json={
            "pw": {"temperature": 0.42},
        })

        # Model should be unchanged
        after = (await client.get("/v1/settings")).json()["pw"]
        assert after["model"] == before
        assert after["temperature"] == 0.42
