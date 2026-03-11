"""Tests for APU shutdown/wake lifecycle.

Scaffold: shutdown/wake methods not yet implemented on StackOrchestrator.
These tests define the expected contract — implement, then remove xfail.
"""

import pytest

from alchemy.apu.monitor import GPUInfo, GPUMonitor, RAMInfo
from alchemy.apu.orchestrator import StackOrchestrator
from alchemy.apu.registry import (
    ModelBackend,
    ModelCard,
    ModelLocation,
    ModelRegistry,
    ModelTier,
)
from unittest.mock import AsyncMock


def _ram() -> RAMInfo:
    return RAMInfo(total_mb=131072, used_mb=40000, free_mb=91072, available_mb=91072)


def _dual_gpus() -> list[GPUInfo]:
    return [
        GPUInfo(index=0, name="RTX 4070", total_vram_mb=12288,
                used_vram_mb=0, free_vram_mb=12288,
                temperature_c=50, utilization_pct=20),
        GPUInfo(index=1, name="RTX 5060 Ti", total_vram_mb=16384,
                used_vram_mb=0, free_vram_mb=16384,
                temperature_c=45, utilization_pct=15),
    ]


def _card(name: str, vram: int = 4096, tier: ModelTier = ModelTier.WARM,
          gpu: int | None = None, location: ModelLocation = ModelLocation.DISK) -> ModelCard:
    return ModelCard(
        name=name, display_name=name, backend=ModelBackend.OLLAMA,
        vram_mb=vram, ram_mb=vram, disk_mb=vram,
        preferred_gpu=gpu, default_tier=tier, current_tier=tier,
        current_location=location, capabilities=[],
    )


async def _make_orchestrator(cards: list[ModelCard] | None = None) -> StackOrchestrator:
    monitor = GPUMonitor(mock_gpus=_dual_gpus())
    await monitor.start()
    registry = ModelRegistry()
    for card in (cards or []):
        registry.register(card)
    orch = StackOrchestrator(
        monitor=monitor, registry=registry,
        ollama_host="http://localhost:11434",
    )
    orch._ollama_load = AsyncMock(return_value=True)
    orch._ollama_unload = AsyncMock(return_value=True)
    orch._started = True
    return orch


@pytest.mark.asyncio
@pytest.mark.xfail(reason="shutdown/wake not yet implemented")
async def test_shutdown_unloads_all_gpu_models():
    """Shutdown unloads every model from GPU."""
    m1 = _card("model-a", location=ModelLocation.GPU_0, tier=ModelTier.AGENT)
    m2 = _card("model-b", location=ModelLocation.GPU_1, tier=ModelTier.RESIDENT)
    m3 = _card("model-c", location=ModelLocation.CPU_RAM, tier=ModelTier.WARM)
    orch = await _make_orchestrator(cards=[m1, m2, m3])

    unloaded = await orch.shutdown()

    assert "model-a" in unloaded
    assert "model-b" in unloaded
    assert "model-c" not in unloaded  # Was in RAM, not GPU
    assert m1.current_location == ModelLocation.DISK
    assert m2.current_location == ModelLocation.DISK
    assert m3.current_location == ModelLocation.CPU_RAM  # Unchanged


@pytest.mark.asyncio
@pytest.mark.xfail(reason="shutdown/wake not yet implemented")
async def test_shutdown_sets_flag():
    """Shutdown sets is_shutdown flag."""
    orch = await _make_orchestrator()
    assert not orch.is_shutdown

    await orch.shutdown()
    assert orch.is_shutdown


@pytest.mark.asyncio
@pytest.mark.xfail(reason="shutdown/wake not yet implemented")
async def test_load_blocked_during_shutdown():
    """Model loading is rejected while APU is shut down."""
    card = _card("model", vram=2048)
    orch = await _make_orchestrator(cards=[card])

    await orch.shutdown()
    result = await orch.load_model("model")

    assert not result.success
    assert "shut down" in result.error


@pytest.mark.asyncio
@pytest.mark.xfail(reason="shutdown/wake not yet implemented")
async def test_wake_re_enables_loading():
    """Wake clears shutdown flag and allows loading again."""
    card = _card("model", vram=2048)
    orch = await _make_orchestrator(cards=[card])

    await orch.shutdown()
    assert orch.is_shutdown

    orch.wake()
    assert not orch.is_shutdown

    result = await orch.load_model("model")
    assert result.success


@pytest.mark.asyncio
@pytest.mark.xfail(reason="shutdown/wake not yet implemented")
async def test_restore_frozen_blocked_during_shutdown():
    """Frozen baseline restore is blocked during shutdown."""
    orch = await _make_orchestrator()
    await orch.shutdown()

    actions = await orch.restore_frozen_baseline()
    assert any("shut down" in a.lower() for a in actions)


@pytest.mark.asyncio
@pytest.mark.xfail(reason="shutdown/wake not yet implemented")
async def test_shutdown_empty_gpu():
    """Shutdown with no GPU models returns empty list."""
    card = _card("ram-model", location=ModelLocation.CPU_RAM)
    orch = await _make_orchestrator(cards=[card])

    unloaded = await orch.shutdown()
    assert unloaded == []
    assert orch.is_shutdown
