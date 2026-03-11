"""APU diagnostics tests — event log and invariant checker."""

import pytest

from alchemy.apu.event_log import APUEventLog, VALID_EVENT_TYPES


# --- Event log ---


def test_event_log_records_and_retrieves():
    log = APUEventLog()
    log.record("load", model_name="test-model", gpu_index=0, success=True)
    assert len(log) == 1
    events = log.recent()
    assert events[0].event_type == "load"
    assert events[0].model_name == "test-model"


def test_event_log_ring_buffer_limit():
    log = APUEventLog(max_events=5)
    for i in range(10):
        log.record("load", model_name=f"model-{i}")
    assert len(log) == 5
    # Most recent should be model-9
    events = log.recent()
    assert events[0].model_name == "model-9"
    assert events[-1].model_name == "model-5"


def test_event_log_filter_by_type():
    log = APUEventLog()
    log.record("load", model_name="a")
    log.record("unload", model_name="b")
    log.record("load", model_name="c")

    loads = log.filter(event_type="load")
    assert len(loads) == 2
    assert all(e.event_type == "load" for e in loads)


def test_event_log_filter_by_model():
    log = APUEventLog()
    log.record("load", model_name="alpha")
    log.record("load", model_name="beta")
    log.record("unload", model_name="alpha")

    alpha_events = log.filter(model_name="alpha")
    assert len(alpha_events) == 2


def test_event_log_filter_by_app():
    log = APUEventLog()
    log.record("app_activate", app_name="voice")
    log.record("app_activate", app_name="click")

    voice = log.filter(app_name="voice")
    assert len(voice) == 1


def test_event_log_filter_errors_only():
    log = APUEventLog()
    log.record("load", success=True)
    log.record("load", success=False, error="failed")
    log.record("drift", success=True)  # successful drift excluded from errors_only
    log.record("drift", success=False, error="drift detected")  # failed drift included

    errors = log.filter(errors_only=True)
    assert len(errors) == 2  # the failed load + the failed drift


def test_event_log_to_dict():
    log = APUEventLog()
    event = log.record(
        "load", model_name="test", gpu_index=1,
        vram_before_mb=5000, vram_after_mb=6000, vram_expected_mb=1000,
        duration_ms=150.0, success=True,
    )
    d = event.to_dict()
    assert d["event_type"] == "load"
    assert d["model_name"] == "test"
    assert d["gpu_index"] == 1
    assert d["vram_before_mb"] == 5000
    assert d["vram_after_mb"] == 6000
    assert d["duration_ms"] == 150.0
    assert "timestamp" in d


def test_event_log_slow_detection():
    log = APUEventLog()
    # Expected ~100ms (1GB * 100ms/GB), actual 500ms → slow
    event = log.record(
        "load", model_name="slow",
        vram_expected_mb=1024, duration_ms=500.0,
    )
    assert event.slow is True

    # Expected ~100ms, actual 50ms → not slow
    event2 = log.record(
        "load", model_name="fast",
        vram_expected_mb=1024, duration_ms=50.0,
    )
    assert event2.slow is False


def test_event_log_clear():
    log = APUEventLog()
    log.record("load")
    log.record("unload")
    assert len(log) == 2
    log.clear()
    assert len(log) == 0


def test_event_log_limit_param():
    log = APUEventLog()
    for i in range(20):
        log.record("load", model_name=f"m-{i}")

    events = log.recent(limit=5)
    assert len(events) == 5

    filtered = log.filter(event_type="load", limit=3)
    assert len(filtered) == 3


def test_valid_event_types_comprehensive():
    """All expected event types are in the valid set."""
    expected = {"load", "unload", "evict", "promote", "demote",
                "drift", "error", "reconcile", "app_activate",
                "app_deactivate", "health_check", "invariant_violation"}
    assert expected == VALID_EVENT_TYPES


# --- Invariants ---


@pytest.mark.asyncio
async def test_invariants_clean_state():
    from unittest.mock import AsyncMock
    from alchemy.apu.invariants import check_invariants
    from alchemy.apu.monitor import GPUInfo, HardwareSnapshot, RAMInfo
    from alchemy.apu.registry import ModelCard, ModelBackend, ModelLocation, ModelRegistry, ModelTier

    registry = ModelRegistry()
    card = ModelCard(
        name="test", backend=ModelBackend.OLLAMA, vram_mb=1000,
        current_location=ModelLocation.GPU_0, current_tier=ModelTier.RESIDENT,
    )
    registry.register(card)

    monitor = AsyncMock()
    monitor.snapshot = AsyncMock(return_value=HardwareSnapshot(
        gpus=[
            GPUInfo(index=0, name="GPU0", total_vram_mb=12000,
                    used_vram_mb=1000, free_vram_mb=11000, temperature_c=40, utilization_pct=10),
            GPUInfo(index=1, name="GPU1", total_vram_mb=16000,
                    used_vram_mb=0, free_vram_mb=16000, temperature_c=40, utilization_pct=0),
        ],
        ram=RAMInfo(total_mb=131072, used_mb=40000, free_mb=91072, available_mb=91072),
    ))

    violations = await check_invariants(registry, monitor)
    assert violations == []


@pytest.mark.asyncio
async def test_invariants_detect_overcommit():
    from unittest.mock import AsyncMock
    from alchemy.apu.invariants import check_invariants
    from alchemy.apu.monitor import GPUInfo, HardwareSnapshot, RAMInfo
    from alchemy.apu.registry import ModelCard, ModelBackend, ModelLocation, ModelRegistry, ModelTier

    registry = ModelRegistry()
    # Register a model that claims 15000MB on a 12000MB GPU
    card = ModelCard(
        name="huge", backend=ModelBackend.OLLAMA, vram_mb=15000,
        current_location=ModelLocation.GPU_0, current_tier=ModelTier.RESIDENT,
    )
    registry.register(card)

    monitor = AsyncMock()
    monitor.snapshot = AsyncMock(return_value=HardwareSnapshot(
        gpus=[
            GPUInfo(index=0, name="GPU0", total_vram_mb=12000,
                    used_vram_mb=12000, free_vram_mb=0, temperature_c=40, utilization_pct=100),
        ],
        ram=RAMInfo(total_mb=131072, used_mb=40000, free_mb=91072, available_mb=91072),
    ))

    violations = await check_invariants(registry, monitor)
    assert len(violations) >= 1
    assert any("overcommitted" in v for v in violations)


@pytest.mark.asyncio
async def test_invariants_detect_tier_mismatch():
    from unittest.mock import AsyncMock
    from alchemy.apu.invariants import check_invariants
    from alchemy.apu.monitor import GPUInfo, HardwareSnapshot, RAMInfo
    from alchemy.apu.registry import ModelCard, ModelBackend, ModelLocation, ModelRegistry, ModelTier

    registry = ModelRegistry()
    # Model on GPU but tier=COLD (inconsistent)
    card = ModelCard(
        name="confused", backend=ModelBackend.OLLAMA, vram_mb=1000,
        current_location=ModelLocation.GPU_0, current_tier=ModelTier.COLD,
    )
    registry.register(card)

    monitor = AsyncMock()
    monitor.snapshot = AsyncMock(return_value=HardwareSnapshot(
        gpus=[
            GPUInfo(index=0, name="GPU0", total_vram_mb=12000,
                    used_vram_mb=1000, free_vram_mb=11000, temperature_c=40, utilization_pct=10),
        ],
        ram=RAMInfo(total_mb=131072, used_mb=40000, free_mb=91072, available_mb=91072),
    ))

    violations = await check_invariants(registry, monitor)
    assert any("tier=cold" in v for v in violations)


@pytest.mark.asyncio
async def test_invariants_detect_vram_drift():
    """Flag when tracked VRAM vs actual VRAM differs by >500MB."""
    from unittest.mock import AsyncMock
    from alchemy.apu.invariants import check_invariants
    from alchemy.apu.monitor import GPUInfo, HardwareSnapshot, RAMInfo
    from alchemy.apu.registry import ModelCard, ModelBackend, ModelLocation, ModelRegistry, ModelTier

    registry = ModelRegistry()
    # Registry says 1000MB on GPU 0
    card = ModelCard(
        name="drifted", backend=ModelBackend.OLLAMA, vram_mb=1000,
        current_location=ModelLocation.GPU_0, current_tier=ModelTier.RESIDENT,
    )
    registry.register(card)

    # But actual GPU reports 2000MB used — drift of 1000MB > 500MB threshold
    monitor = AsyncMock()
    monitor.snapshot = AsyncMock(return_value=HardwareSnapshot(
        gpus=[
            GPUInfo(index=0, name="GPU0", total_vram_mb=12000,
                    used_vram_mb=2000, free_vram_mb=10000, temperature_c=40, utilization_pct=10),
            GPUInfo(index=1, name="GPU1", total_vram_mb=16000,
                    used_vram_mb=0, free_vram_mb=16000, temperature_c=40, utilization_pct=0),
        ],
        ram=RAMInfo(total_mb=131072, used_mb=40000, free_mb=91072, available_mb=91072),
    ))

    violations = await check_invariants(registry, monitor)
    assert any("drift" in v.lower() for v in violations)


@pytest.mark.asyncio
async def test_invariants_detect_resident_not_on_gpu():
    """RESIDENT tier model not on GPU should be flagged."""
    from unittest.mock import AsyncMock
    from alchemy.apu.invariants import check_invariants
    from alchemy.apu.monitor import GPUInfo, HardwareSnapshot, RAMInfo
    from alchemy.apu.registry import ModelCard, ModelBackend, ModelLocation, ModelRegistry, ModelTier

    registry = ModelRegistry()
    # RESIDENT model stuck in RAM — invariant violation
    card = ModelCard(
        name="lost-resident", backend=ModelBackend.OLLAMA, vram_mb=1000,
        current_location=ModelLocation.CPU_RAM, current_tier=ModelTier.RESIDENT,
    )
    registry.register(card)

    monitor = AsyncMock()
    monitor.snapshot = AsyncMock(return_value=HardwareSnapshot(
        gpus=[
            GPUInfo(index=0, name="GPU0", total_vram_mb=12000,
                    used_vram_mb=0, free_vram_mb=12000, temperature_c=40, utilization_pct=0),
        ],
        ram=RAMInfo(total_mb=131072, used_mb=40000, free_mb=91072, available_mb=91072),
    ))

    violations = await check_invariants(registry, monitor)
    assert any("RESIDENT" in v for v in violations)
