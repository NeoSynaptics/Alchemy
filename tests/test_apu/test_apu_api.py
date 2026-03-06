"""GPU API endpoint tests — mocked orchestrator."""

from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient

from alchemy.apu.monitor import GPUInfo, HardwareSnapshot, RAMInfo
from alchemy.apu.orchestrator import LoadResult, StackOrchestrator, StackStatus
from alchemy.apu.registry import (
    ModelBackend,
    ModelCard,
    ModelLocation,
    ModelRegistry,
    ModelTier,
)
from alchemy.server import app


def _mock_card() -> ModelCard:
    return ModelCard(
        name="qwen3:14b",
        display_name="Qwen3 14B",
        backend=ModelBackend.OLLAMA,
        vram_mb=9216,
        ram_mb=9216,
        disk_mb=9216,
        preferred_gpu=1,
        default_tier=ModelTier.RESIDENT,
        current_tier=ModelTier.RESIDENT,
        current_location=ModelLocation.GPU_1,
        capabilities=["conversation", "agent"],
    )


def _mock_status() -> StackStatus:
    return StackStatus(
        snapshot=HardwareSnapshot(
            gpus=[
                GPUInfo(index=0, name="RTX 4070", total_vram_mb=12288,
                        used_vram_mb=4000, free_vram_mb=8288,
                        temperature_c=50, utilization_pct=20),
                GPUInfo(index=1, name="RTX 5060 Ti", total_vram_mb=16384,
                        used_vram_mb=13000, free_vram_mb=3384,
                        temperature_c=45, utilization_pct=60),
            ],
            ram=RAMInfo(total_mb=131072, used_mb=40000, free_mb=91072, available_mb=91072),
        ),
        models=[_mock_card()],
        mode="auto",
    )


@pytest.fixture
async def client():
    mock_orch = AsyncMock(spec=StackOrchestrator)
    mock_orch.status = AsyncMock(return_value=_mock_status())
    mock_orch.gpu_status = AsyncMock(return_value=_mock_status().snapshot.gpus)
    mock_orch.ram_status = AsyncMock(return_value=_mock_status().snapshot.ram)

    mock_registry = ModelRegistry()
    mock_registry.register(_mock_card())
    mock_orch._registry = mock_registry

    app.state.orchestrator = mock_orch

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c

    app.state.orchestrator = None


@pytest.mark.asyncio
async def test_get_status(client):
    resp = await client.get("/v1/apu/status")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["gpus"]) == 2
    assert data["gpus"][0]["name"] == "RTX 4070"
    assert data["gpus"][1]["name"] == "RTX 5060 Ti"
    assert len(data["models"]) == 1
    assert data["models"][0]["name"] == "qwen3:14b"
    assert data["mode"] == "auto"
    assert data["ram"]["total_mb"] == 131072


@pytest.mark.asyncio
async def test_get_gpus(client):
    resp = await client.get("/v1/apu/gpus")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 2
    assert data[0]["total_vram_mb"] == 12288
    assert data[1]["total_vram_mb"] == 16384


@pytest.mark.asyncio
async def test_get_ram(client):
    resp = await client.get("/v1/apu/ram")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_mb"] == 131072
    assert data["available_mb"] == 91072


@pytest.mark.asyncio
async def test_get_models(client):
    resp = await client.get("/v1/apu/models")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["name"] == "qwen3:14b"
    assert data[0]["current_tier"] == "resident"
    assert data[0]["current_location"] == "gpu_1"


@pytest.mark.asyncio
async def test_get_model_by_name(client):
    resp = await client.get("/v1/apu/models/qwen3:14b")
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "qwen3:14b"
    assert data["vram_mb"] == 9216


@pytest.mark.asyncio
async def test_get_model_not_found(client):
    resp = await client.get("/v1/apu/models/nonexistent")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_promote_model(client):
    client_fixture = client
    mock_orch = app.state.orchestrator
    mock_orch.promote = AsyncMock(return_value=LoadResult(
        success=True, location=ModelLocation.GPU_1
    ))

    resp = await client_fixture.post("/v1/apu/models/qwen3:14b/promote")
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data["location"] == "gpu_1"


@pytest.mark.asyncio
async def test_demote_model(client):
    mock_orch = app.state.orchestrator
    mock_orch.demote = AsyncMock(return_value=True)

    resp = await client.post("/v1/apu/models/qwen3:14b/demote")
    assert resp.status_code == 200
    assert resp.json()["success"] is True


@pytest.mark.asyncio
async def test_load_model(client):
    mock_orch = app.state.orchestrator
    mock_orch.ensure_loaded = AsyncMock(return_value=LoadResult(
        success=True, location=ModelLocation.GPU_0, evicted=["old-model"]
    ))

    resp = await client.post("/v1/apu/models/qwen3:14b/load")
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data["evicted"] == ["old-model"]


@pytest.mark.asyncio
async def test_rebalance(client):
    mock_orch = app.state.orchestrator
    mock_orch.rebalance = AsyncMock(return_value=["Moved X: GPU 0 → GPU 1"])

    resp = await client.post("/v1/apu/rebalance")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["actions"]) == 1


@pytest.mark.asyncio
async def test_app_activate(client):
    mock_orch = app.state.orchestrator
    mock_orch.app_activate = AsyncMock(return_value=LoadResult(success=True))

    resp = await client.post("/v1/apu/app/spotify/activate",
                             json={"models": ["music-model"]})
    assert resp.status_code == 200
    assert resp.json()["success"] is True


@pytest.mark.asyncio
async def test_app_deactivate(client):
    mock_orch = app.state.orchestrator
    mock_orch.app_deactivate = AsyncMock(return_value=["music-model"])

    resp = await client.post("/v1/apu/app/spotify/deactivate")
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert "music-model" in data["demoted"]
