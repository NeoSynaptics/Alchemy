"""APU Runtime Invariant Checker — validates state consistency.

Can be called periodically or after every APU operation.
Violations are logged and optionally recorded as events.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from alchemy.apu.registry import ModelLocation, ModelTier

if TYPE_CHECKING:
    from alchemy.apu.event_log import APUEventLog
    from alchemy.apu.monitor import GPUMonitor
    from alchemy.apu.registry import ModelRegistry

logger = logging.getLogger(__name__)

# VRAM drift threshold: flag if tracked vs actual differs by more than this.
# Set to 500MB to allow for timing races between snapshot and check.
_VRAM_DRIFT_THRESHOLD_MB = 500


async def check_invariants(
    registry: ModelRegistry,
    monitor: GPUMonitor,
    event_log: APUEventLog | None = None,
) -> list[str]:
    """Check all APU invariants. Returns list of violations (empty = healthy)."""
    violations: list[str] = []

    all_models = registry.all_models()

    # 1. VRAM accounting: sum of model cards on GPU <= GPU total
    snap = None
    try:
        snap = await monitor.snapshot()
        for gpu in snap.gpus:
            # APU only supports GPU 0-1; flag unexpected indices
            if gpu.index > 1:
                violations.append(
                    f"GPU index {gpu.index} found but APU only supports GPU 0-1"
                )
                continue
            total_on_gpu = registry.total_vram_on_gpu(gpu.index)
            if total_on_gpu > gpu.total_vram_mb:
                violations.append(
                    f"GPU {gpu.index} VRAM overcommitted: "
                    f"registry={total_on_gpu}MB > total={gpu.total_vram_mb}MB"
                )
    except Exception as e:
        violations.append(f"Cannot check GPU VRAM: {e}")

    # 2. No negative VRAM values
    for model in all_models:
        if model.vram_mb < 0:
            violations.append(f"{model.name} has negative VRAM: {model.vram_mb}MB")

    # 3. Location-tier consistency
    for model in all_models:
        if model.current_location.is_gpu and model.current_tier in (ModelTier.WARM, ModelTier.COLD):
            violations.append(
                f"{model.name} on GPU but tier={model.current_tier.value}"
            )
        if model.current_location == ModelLocation.CPU_RAM:
            if model.current_tier == ModelTier.COLD:
                violations.append(f"{model.name} in RAM but tier=cold")
            elif model.current_tier == ModelTier.RESIDENT:
                violations.append(
                    f"{model.name} in RAM but tier=resident (must be on GPU)"
                )

    # 4. RESIDENT tier model NOT on GPU → violation
    for model in all_models:
        if model.current_tier == ModelTier.RESIDENT and not model.current_location.is_gpu:
            violations.append(
                f"{model.name} is RESIDENT but location={model.current_location.value} (should be on GPU)"
            )

    # 5. VRAM drift: tracked total vs actual used — flag drift > threshold
    if snap is not None:
        for gpu in snap.gpus:
            if gpu.index > 1:
                continue  # Already flagged in check #1
            tracked = registry.total_vram_on_gpu(gpu.index)
            actual = gpu.used_vram_mb
            drift = abs(tracked - actual)
            if drift > _VRAM_DRIFT_THRESHOLD_MB:
                violations.append(
                    f"GPU {gpu.index} VRAM drift: tracked={tracked}MB, "
                    f"actual={actual}MB, drift={drift}MB (threshold={_VRAM_DRIFT_THRESHOLD_MB}MB)"
                )

    # Log violations
    for v in violations:
        logger.warning("APU invariant violation: %s", v)

    # Record as events if event_log available
    if event_log and violations:
        for v in violations:
            try:
                event_log.record(
                    "invariant_violation",
                    success=False,
                    error=v,
                )
            except Exception as e:
                logger.error("Failed to record invariant violation in event log: %s", e)

    return violations
