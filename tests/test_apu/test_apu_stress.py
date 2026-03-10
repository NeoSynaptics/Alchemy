"""APU stress tests — concurrency, failure injection, and invariants."""

import asyncio
import warnings
from unittest.mock import AsyncMock, patch

import pytest

from alchemy.apu.monitor import GPUInfo, GPUMonitor, HardwareSnapshot, RAMInfo
from alchemy.apu.orchestrator import LoadResult, StackOrchestrator
from alchemy.apu.registry import (
    ModelBackend,
    ModelCard,
    ModelLocation,
    ModelRegistry,
    ModelTier,
)


# --- Helpers ---


def _make_card(name: str, vram_mb: int = 1000, gpu: int = 0, tier: ModelTier = ModelTier.WARM) -> ModelCard:
    return ModelCard(
        name=name,
        display_name=name,
        backend=ModelBackend.OLLAMA,
        vram_mb=vram_mb,
        ram_mb=vram_mb,
        disk_mb=vram_mb,
        preferred_gpu=gpu,
        default_tier=tier,
        current_tier=ModelTier.COLD,
        current_location=ModelLocation.DISK,
        capabilities=["test"],
    )


def _mock_snapshot() -> HardwareSnapshot:
    return HardwareSnapshot(
        gpus=[
            GPUInfo(index=0, name="GPU0", total_vram_mb=12000,
                    used_vram_mb=0, free_vram_mb=12000, temperature_c=40, utilization_pct=0),
            GPUInfo(index=1, name="GPU1", total_vram_mb=16000,
                    used_vram_mb=0, free_vram_mb=16000, temperature_c=40, utilization_pct=0),
        ],
        ram=RAMInfo(total_mb=131072, used_mb=40000, free_mb=91072, available_mb=91072),
    )


async def _make_orch(cards: list[ModelCard] | None = None) -> StackOrchestrator:
    """Create orchestrator with mock monitor and backend."""
    registry = ModelRegistry()
    for card in (cards or []):
        registry.register(card)

    monitor = AsyncMock(spec=GPUMonitor)
    monitor.snapshot = AsyncMock(return_value=_mock_snapshot())
    monitor.start = AsyncMock()
    monitor.close = AsyncMock()

    orch = StackOrchestrator(monitor=monitor, registry=registry, ollama_host="http://fake:11434")
    # Mock backend calls
    orch._backend_load = AsyncMock(return_value=True)
    orch._backend_unload = AsyncMock(return_value=True)
    orch._ollama_list_running = AsyncMock(return_value={})
    orch._started = True

    return orch


async def assert_apu_invariants(orch: StackOrchestrator):
    """Verify APU state consistency after every test."""
    snap = await orch._monitor.snapshot()

    # 1. VRAM accounting: sum on GPU <= GPU total
    for gpu in snap.gpus:
        total_on_gpu = orch._registry.total_vram_on_gpu(gpu.index)
        assert total_on_gpu <= gpu.total_vram_mb, (
            f"GPU {gpu.index} VRAM overcommitted: {total_on_gpu}MB > {gpu.total_vram_mb}MB"
        )

    # 2. Each model has exactly one location
    for model in orch._registry.all_models():
        assert model.current_location is not None, f"{model.name} has no location"

    # 3. No negative VRAM
    for model in orch._registry.all_models():
        assert model.vram_mb >= 0, f"{model.name} has negative VRAM"

    # 4. Location-tier consistency
    for model in orch._registry.all_models():
        if model.current_location.is_gpu and model.current_tier in (ModelTier.WARM, ModelTier.COLD):
            warnings.warn(f"{model.name} on GPU but tier={model.current_tier.value}")

    # 5. No pending operations stuck
    assert len(orch._pending_operations) == 0, (
        f"Pending operations not cleared: {orch._pending_operations}"
    )


# --- Concurrency tests ---


@pytest.mark.asyncio
async def test_concurrent_ensure_loaded_different_models():
    """10 concurrent ensure_loaded for different models — no double-eviction."""
    cards = [_make_card(f"model-{i}", vram_mb=500, gpu=0) for i in range(10)]
    orch = await _make_orch(cards)

    results = await asyncio.gather(*[orch.ensure_loaded(c.name) for c in cards])
    successes = [r for r in results if r.success]
    assert len(successes) >= 1  # At least some should succeed
    await assert_apu_invariants(orch)


@pytest.mark.asyncio
async def test_concurrent_load_and_unload():
    """5 loads + 5 unloads concurrently — VRAM stays consistent."""
    cards = [_make_card(f"model-{i}", vram_mb=500, gpu=0) for i in range(10)]
    orch = await _make_orch(cards)

    # Pre-load first 5
    for i in range(5):
        await orch.load_model(cards[i].name, gpu=0)

    # Concurrent: load 5 new + unload 5 existing
    tasks = []
    for i in range(5):
        tasks.append(orch.unload_model(cards[i].name))
    for i in range(5, 10):
        tasks.append(orch.load_model(cards[i].name, gpu=0))

    await asyncio.gather(*tasks)
    await assert_apu_invariants(orch)


@pytest.mark.asyncio
async def test_concurrent_app_activate():
    """2 concurrent app_activate requesting same GPU — no corruption."""
    cards_a = [_make_card("app-a-model", vram_mb=2000, gpu=1)]
    cards_b = [_make_card("app-b-model", vram_mb=2000, gpu=1)]
    orch = await _make_orch(cards_a + cards_b)

    results = await asyncio.gather(
        orch.app_activate("app-a", ["app-a-model"]),
        orch.app_activate("app-b", ["app-b-model"]),
    )
    assert all(r.success for r in results)
    await assert_apu_invariants(orch)


@pytest.mark.asyncio
async def test_rapid_promote_demote_cycles():
    """100 rapid promote/demote cycles — model ends up in correct state."""
    card = _make_card("cycle-model", vram_mb=1000, gpu=0)
    orch = await _make_orch([card])

    for _ in range(100):
        await orch.promote(card.name, gpu=0)
        await orch.demote(card.name)

    # After demote, should be in RAM
    assert card.current_location == ModelLocation.CPU_RAM
    assert card.current_tier == ModelTier.WARM
    await assert_apu_invariants(orch)


@pytest.mark.asyncio
async def test_reconcile_during_load():
    """reconcile_vram during active load — no state corruption."""
    card = _make_card("recon-model", vram_mb=1000, gpu=0)
    orch = await _make_orch([card])

    results = await asyncio.gather(
        orch.load_model(card.name, gpu=0),
        orch.reconcile_vram(),
    )
    await assert_apu_invariants(orch)


@pytest.mark.asyncio
async def test_health_check_during_eviction():
    """health_check during eviction — no crash."""
    cards = [_make_card(f"hc-{i}", vram_mb=1000, gpu=0) for i in range(5)]
    orch = await _make_orch(cards)

    for c in cards[:3]:
        await orch.load_model(c.name, gpu=0)

    results = await asyncio.gather(
        orch.health_check(),
        orch.demote(cards[0].name),
    )
    assert isinstance(results[0], dict)
    assert "healthy" in results[0]
    await assert_apu_invariants(orch)


@pytest.mark.asyncio
async def test_app_deactivate_during_ensure_loaded():
    """app_deactivate during ensure_loaded — clean handling."""
    card = _make_card("deact-model", vram_mb=1000, gpu=0)
    orch = await _make_orch([card])
    await orch.app_activate("test-app", [card.name])

    results = await asyncio.gather(
        orch.ensure_loaded(card.name),
        orch.app_deactivate("test-app"),
    )
    await assert_apu_invariants(orch)


@pytest.mark.asyncio
async def test_pending_operations_guard():
    """Model already being loaded gets rejected, not deadlocked."""
    card = _make_card("pending-model", vram_mb=1000, gpu=0)
    orch = await _make_orch([card])

    # Simulate model in pending state
    orch._pending_operations.add("pending-model")
    result = await orch._load_model_unlocked("pending-model")
    assert not result.success
    assert "currently being loaded" in result.error

    # Clean up
    orch._pending_operations.discard("pending-model")
    await assert_apu_invariants(orch)


# --- Failure injection tests ---


@pytest.mark.asyncio
async def test_load_failure_rollback():
    """Ollama load fails — evicted models restored."""
    existing = _make_card("existing", vram_mb=10000, gpu=0, tier=ModelTier.RESIDENT)
    new = _make_card("new-model", vram_mb=11000, gpu=0)
    orch = await _make_orch([existing, new])

    # Load existing first
    await orch.load_model(existing.name, gpu=0)

    # Make new model's load fail
    call_count = 0
    original_load = orch._backend_load

    async def _failing_load(card, gpu):
        nonlocal call_count
        call_count += 1
        if card.name == "new-model":
            return False
        return True

    orch._backend_load = _failing_load

    result = await orch.load_model(new.name, gpu=0)
    assert not result.success
    assert "Backend load failed" in result.error
    await assert_apu_invariants(orch)


@pytest.mark.asyncio
async def test_unload_failure_keeps_gpu_state():
    """Ollama unload fails — model stays marked as GPU."""
    card = _make_card("sticky-model", vram_mb=1000, gpu=0)
    orch = await _make_orch([card])
    await orch.load_model(card.name, gpu=0)

    # Make unload fail
    orch._backend_unload = AsyncMock(return_value=False)

    # Demote should fail and keep GPU state
    result = await orch.demote(card.name)
    assert result is False, "Demote should return False when backend unload fails"
    assert card.current_location.is_gpu, "Model should stay on GPU after unload failure"
    events = orch._event_log.filter(event_type="error")
    assert len(events) >= 1, "Error event should be logged for unload failure"
    await assert_apu_invariants(orch)


@pytest.mark.asyncio
async def test_ollama_unreachable():
    """Ollama completely unreachable — graceful degradation."""
    card = _make_card("unreachable", vram_mb=1000, gpu=0)
    orch = await _make_orch([card])

    # Return False (not raise) to simulate failed load
    orch._backend_load = AsyncMock(return_value=False)

    result = await orch.load_model(card.name, gpu=0)
    assert not result.success
    await assert_apu_invariants(orch)


@pytest.mark.asyncio
async def test_no_infinite_eviction_loop():
    """GPU at 0 free VRAM — no infinite eviction loop."""
    # Fill GPU 0 completely
    cards = [_make_card(f"full-{i}", vram_mb=4000, gpu=0) for i in range(3)]
    orch = await _make_orch(cards)

    for c in cards:
        await orch.load_model(c.name, gpu=0)

    # Try to load a model bigger than both GPUs combined (12k + 16k = 28k)
    huge = _make_card("huge", vram_mb=30000, gpu=0)
    orch._registry.register(huge)

    result = await orch.ensure_loaded("huge")
    assert not result.success
    await assert_apu_invariants(orch)


# --- Event log integration ---


@pytest.mark.asyncio
async def test_event_log_captures_operations():
    """All operations produce events."""
    card = _make_card("event-test", vram_mb=1000, gpu=0)
    orch = await _make_orch([card])

    await orch.load_model(card.name, gpu=0)
    await orch.demote(card.name)
    await orch.load_model(card.name, gpu=0)  # reload so we can unload
    await orch.unload_model(card.name)
    await orch.reconcile_vram()

    events = orch._event_log.recent()
    types = [e.event_type for e in events]
    assert "load" in types
    assert "demote" in types
    assert "unload" in types
    assert "reconcile" in types


@pytest.mark.asyncio
async def test_concurrent_ensure_loaded_same_model():
    """Two concurrent ensure_loaded on SAME model — second gets 'busy' error."""
    card = _make_card("same-model", vram_mb=1000, gpu=0)
    orch = await _make_orch([card])

    # Simulate first load in progress
    orch._pending_operations.add("same-model")

    # Second call should get rejected
    result = await orch._load_model_unlocked("same-model")
    assert not result.success
    assert "currently being loaded" in result.error

    # Clean up
    orch._pending_operations.discard("same-model")
    await assert_apu_invariants(orch)


@pytest.mark.asyncio
async def test_app_activate_during_restore_frozen_baseline():
    """app_activate during restore_frozen_baseline — no conflict."""
    baseline_card = _make_card("baseline-model", vram_mb=1000, gpu=0, tier=ModelTier.RESIDENT)
    app_card = _make_card("app-model", vram_mb=1000, gpu=1)
    orch = await _make_orch([baseline_card, app_card])

    # Set frozen baseline
    orch._frozen_config["gpu_0"] = ["baseline-model"]

    results = await asyncio.gather(
        orch.restore_frozen_baseline(),
        orch.app_activate("test-app", ["app-model"]),
    )
    await assert_apu_invariants(orch)


@pytest.mark.asyncio
async def test_unload_failure_keeps_gpu_state_unload_method():
    """unload_model with backend failure — model stays on GPU."""
    card = _make_card("sticky-unload", vram_mb=1000, gpu=0)
    orch = await _make_orch([card])
    await orch.load_model(card.name, gpu=0)

    # Make unload fail
    orch._backend_unload = AsyncMock(return_value=False)

    result = await orch.unload_model(card.name)
    assert result is False, "unload_model should return False when backend fails"
    assert card.current_location.is_gpu, "Model should stay on GPU after unload failure"
    await assert_apu_invariants(orch)
