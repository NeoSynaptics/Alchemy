"""Stack Orchestrator tests — mock monitor + registry + backends."""

from unittest.mock import AsyncMock, patch

import pytest

from alchemy.apu.monitor import GPUInfo, GPUMonitor, HardwareSnapshot, RAMInfo
from alchemy.apu.orchestrator import StackOrchestrator
from alchemy.apu.registry import (
    ModelBackend,
    ModelCard,
    ModelLocation,
    ModelRegistry,
    ModelTier,
)


def _ram() -> RAMInfo:
    return RAMInfo(total_mb=131072, used_mb=40000, free_mb=91072, available_mb=91072)


def _dual_gpus(gpu0_used: int = 0, gpu1_used: int = 0) -> list[GPUInfo]:
    return [
        GPUInfo(index=0, name="RTX 4070", total_vram_mb=12288,
                used_vram_mb=gpu0_used, free_vram_mb=12288 - gpu0_used,
                temperature_c=50, utilization_pct=20),
        GPUInfo(index=1, name="RTX 5060 Ti", total_vram_mb=16384,
                used_vram_mb=gpu1_used, free_vram_mb=16384 - gpu1_used,
                temperature_c=45, utilization_pct=15),
    ]


def _card(name: str, vram: int = 4096, tier: ModelTier = ModelTier.WARM,
          gpu: int | None = None, backend: ModelBackend = ModelBackend.OLLAMA,
          location: ModelLocation = ModelLocation.DISK) -> ModelCard:
    return ModelCard(
        name=name, display_name=name, backend=backend,
        vram_mb=vram, ram_mb=vram, disk_mb=vram,
        preferred_gpu=gpu, default_tier=tier, current_tier=tier,
        current_location=location, capabilities=[],
    )


async def _make_orchestrator(
    gpus: list[GPUInfo] | None = None,
    cards: list[ModelCard] | None = None,
    auto_start: bool = False,
) -> StackOrchestrator:
    """Create orchestrator with mock monitor and optional cards."""
    monitor = GPUMonitor(mock_gpus=gpus or _dual_gpus())
    await monitor.start()

    registry = ModelRegistry()
    for card in (cards or []):
        registry.register(card)

    orch = StackOrchestrator(
        monitor=monitor, registry=registry,
        ollama_host="http://localhost:11434",
    )
    # Patch backend calls to avoid real HTTP
    orch._ollama_load = AsyncMock(return_value=True)
    orch._ollama_unload = AsyncMock(return_value=True)

    if auto_start:
        orch._started = True

    return orch


@pytest.mark.asyncio
async def test_load_model_to_preferred_gpu():
    """Model loads to its preferred GPU when space is available."""
    card = _card("qwen3:14b", vram=9216, gpu=1)
    orch = await _make_orchestrator(cards=[card], auto_start=True)

    result = await orch.load_model("qwen3:14b")
    assert result.success
    assert result.location == ModelLocation.GPU_1
    assert card.current_location == ModelLocation.GPU_1


@pytest.mark.asyncio
async def test_load_model_auto_select_gpu():
    """Without preferred GPU, picks GPU with most free VRAM."""
    card = _card("small-model", vram=2048, gpu=None)
    orch = await _make_orchestrator(cards=[card], auto_start=True)

    result = await orch.load_model("small-model")
    assert result.success
    # GPU 1 has more free VRAM (16384 vs 12288)
    assert result.location == ModelLocation.GPU_1


@pytest.mark.asyncio
async def test_load_model_already_on_gpu():
    """Loading a model already on GPU just touches it."""
    card = _card("loaded", location=ModelLocation.GPU_0, tier=ModelTier.AGENT)
    orch = await _make_orchestrator(cards=[card], auto_start=True)

    result = await orch.load_model("loaded")
    assert result.success
    assert result.location == ModelLocation.GPU_0
    assert card.last_used is not None


@pytest.mark.asyncio
async def test_load_model_not_in_registry():
    """Loading unknown model fails."""
    orch = await _make_orchestrator(auto_start=True)
    result = await orch.load_model("nonexistent")
    assert not result.success
    assert "not in registry" in result.error


@pytest.mark.asyncio
async def test_unload_model():
    """Unloading moves model to DISK/COLD."""
    card = _card("model", location=ModelLocation.GPU_0, tier=ModelTier.AGENT)
    orch = await _make_orchestrator(cards=[card], auto_start=True)

    ok = await orch.unload_model("model")
    assert ok
    assert card.current_location == ModelLocation.DISK
    assert card.current_tier == ModelTier.COLD


@pytest.mark.asyncio
async def test_unload_allows_resident():
    """No model is immune — even RESIDENT can be unloaded."""
    card = _card("voice", tier=ModelTier.RESIDENT, location=ModelLocation.GPU_0)
    orch = await _make_orchestrator(cards=[card], auto_start=True)

    ok = await orch.unload_model("voice")
    assert ok
    assert card.current_location == ModelLocation.DISK
    assert card.current_tier == ModelTier.COLD


@pytest.mark.asyncio
async def test_demote_to_warm():
    """Demoting moves model from GPU to CPU RAM (WARM)."""
    card = _card("model", location=ModelLocation.GPU_0, tier=ModelTier.AGENT)
    orch = await _make_orchestrator(cards=[card], auto_start=True)

    ok = await orch.demote("model")
    assert ok
    assert card.current_location == ModelLocation.CPU_RAM
    assert card.current_tier == ModelTier.WARM


@pytest.mark.asyncio
async def test_demote_allows_resident():
    """No model is immune — even RESIDENT can be demoted to RAM."""
    card = _card("core", tier=ModelTier.RESIDENT, location=ModelLocation.GPU_1)
    orch = await _make_orchestrator(cards=[card], auto_start=True)

    ok = await orch.demote("core")
    assert ok
    assert card.current_location == ModelLocation.CPU_RAM
    assert card.current_tier == ModelTier.WARM


@pytest.mark.asyncio
async def test_ensure_loaded_already_on_gpu():
    """ensure_loaded is a no-op for models already on GPU."""
    card = _card("loaded", location=ModelLocation.GPU_0, tier=ModelTier.AGENT)
    orch = await _make_orchestrator(cards=[card], auto_start=True)

    result = await orch.ensure_loaded("loaded")
    assert result.success
    assert result.location == ModelLocation.GPU_0


@pytest.mark.asyncio
async def test_ensure_loaded_promotes_from_ram():
    """ensure_loaded promotes WARM model to VRAM."""
    card = _card("warm-model", vram=2048, tier=ModelTier.WARM, location=ModelLocation.CPU_RAM)
    orch = await _make_orchestrator(cards=[card], auto_start=True)

    result = await orch.ensure_loaded("warm-model")
    assert result.success
    assert card.current_location.is_gpu


@pytest.mark.asyncio
async def test_ensure_loaded_evicts_lower_priority():
    """ensure_loaded evicts lower-priority models to make room."""
    # GPU 0 has 12288 MB. Fill it with an AGENT model.
    occupant = _card("occupant", vram=10000, tier=ModelTier.AGENT,
                     location=ModelLocation.GPU_0)
    # New model needs space — it's USER_ACTIVE (higher priority than AGENT)
    newcomer = _card("newcomer", vram=8000, tier=ModelTier.USER_ACTIVE, gpu=0)

    orch = await _make_orchestrator(cards=[occupant, newcomer], auto_start=True)
    result = await orch.load_model("newcomer", gpu=0)

    assert result.success
    assert "occupant" in result.evicted
    assert occupant.current_location == ModelLocation.CPU_RAM
    assert occupant.current_tier == ModelTier.WARM


@pytest.mark.asyncio
async def test_ensure_loaded_can_evict_resident():
    """No model is immune — RESIDENT can be evicted to make room."""
    # GPU 0 full with resident model
    resident = _card("voice", vram=11000, tier=ModelTier.RESIDENT,
                     location=ModelLocation.GPU_0)
    newcomer = _card("newcomer", vram=8000, tier=ModelTier.AGENT, gpu=0)

    orch = await _make_orchestrator(cards=[resident, newcomer], auto_start=True)
    result = await orch.load_model("newcomer", gpu=0)

    assert result.success
    # Resident evicted to RAM (warm), not disk
    assert resident.current_location == ModelLocation.CPU_RAM
    assert resident.current_tier == ModelTier.WARM


@pytest.mark.asyncio
async def test_ensure_loaded_tries_other_gpu():
    """If model exceeds preferred GPU's total capacity, falls back to the other."""
    # GPU 0 = 12288MB total. Model prefers GPU 0 but needs 13000MB — doesn't fit.
    # GPU 1 = 16384MB total — fits fine.
    newcomer = _card("bigmodel", vram=13000, tier=ModelTier.AGENT, gpu=0)

    orch = await _make_orchestrator(cards=[newcomer], auto_start=True)
    result = await orch.ensure_loaded("bigmodel")

    assert result.success
    assert result.location == ModelLocation.GPU_1  # Fell back to GPU 1


@pytest.mark.asyncio
async def test_app_activate():
    """App activation marks models as USER_ACTIVE and loads them."""
    card = _card("app-model", vram=2048, tier=ModelTier.WARM, location=ModelLocation.CPU_RAM)
    orch = await _make_orchestrator(cards=[card], auto_start=True)

    result = await orch.app_activate("spotify", ["app-model"])
    assert result.success
    assert card.current_tier == ModelTier.USER_ACTIVE
    assert card.owner_app == "spotify"
    assert card.current_location.is_gpu


@pytest.mark.asyncio
async def test_app_deactivate():
    """App deactivation demotes models back to WARM."""
    card = _card("app-model", vram=2048, tier=ModelTier.USER_ACTIVE,
                 location=ModelLocation.GPU_0)
    card.owner_app = "spotify"
    orch = await _make_orchestrator(cards=[card], auto_start=True)
    orch._app_models["spotify"] = ["app-model"]

    demoted = await orch.app_deactivate("spotify")
    assert "app-model" in demoted
    assert card.current_tier == ModelTier.WARM
    assert card.current_location == ModelLocation.CPU_RAM


@pytest.mark.asyncio
async def test_app_deactivate_demotes_all_models():
    """Deactivating an app demotes ALL models, including former residents."""
    card = _card("voice", tier=ModelTier.RESIDENT, location=ModelLocation.GPU_0)
    card.owner_app = "core"
    orch = await _make_orchestrator(cards=[card], auto_start=True)
    orch._app_models["core"] = ["voice"]

    demoted = await orch.app_deactivate("core")
    assert "voice" in demoted
    assert card.current_location == ModelLocation.CPU_RAM
    assert card.current_tier == ModelTier.WARM


@pytest.mark.asyncio
async def test_user_idle_demotes_user_active():
    """user_idle demotes all P1 USER_ACTIVE models to WARM."""
    card = _card("active", tier=ModelTier.USER_ACTIVE, location=ModelLocation.GPU_0)
    orch = await _make_orchestrator(cards=[card], auto_start=True)

    demoted = await orch.user_idle()
    assert "active" in demoted
    assert card.current_tier == ModelTier.WARM


@pytest.mark.asyncio
async def test_status():
    """Status returns full snapshot."""
    card = _card("model", location=ModelLocation.GPU_0, tier=ModelTier.AGENT)
    orch = await _make_orchestrator(cards=[card], auto_start=True)

    status = await orch.status()
    assert len(status.snapshot.gpus) == 2
    assert len(status.models) == 1
    assert status.mode == "auto"


@pytest.mark.asyncio
async def test_promote_alias():
    """promote() is an alias for load_model()."""
    card = _card("model", vram=2048)
    orch = await _make_orchestrator(cards=[card], auto_start=True)

    result = await orch.promote("model")
    assert result.success
    assert card.current_location.is_gpu
