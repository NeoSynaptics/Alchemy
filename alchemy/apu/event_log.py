"""APU Event Logger — structured ring buffer for GPU orchestration diagnostics.

Captures every state-changing operation with VRAM before/after, timing,
and flexible details. Queryable via API for debugging.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

VALID_EVENT_TYPES = frozenset({
    "load", "unload", "evict", "promote", "demote",
    "drift", "error", "reconcile", "app_activate",
    "app_deactivate", "health_check", "invariant_violation",
})

# Rough estimate: 100ms per GB for Ollama loads
_MS_PER_GB = 100.0


@dataclass
class APUEvent:
    """A single APU operation record."""

    timestamp: datetime
    event_type: str
    model_name: str | None = None
    gpu_index: int | None = None
    app_name: str | None = None
    vram_before_mb: int = 0
    vram_after_mb: int = 0
    vram_expected_mb: int = 0
    duration_ms: float = 0.0
    success: bool = True
    error: str | None = None
    details: dict[str, Any] = field(default_factory=dict)
    expected_duration_ms: float = 0.0
    slow: bool = False  # True if actual > 2x expected

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type,
            "model_name": self.model_name,
            "gpu_index": self.gpu_index,
            "app_name": self.app_name,
            "vram_before_mb": self.vram_before_mb,
            "vram_after_mb": self.vram_after_mb,
            "vram_expected_mb": self.vram_expected_mb,
            "duration_ms": round(self.duration_ms, 1),
            "success": self.success,
            "error": self.error,
            "details": self.details,
            "expected_duration_ms": round(self.expected_duration_ms, 1),
            "slow": self.slow,
        }


class APUEventLog:
    """Ring buffer of APU events with filtering."""

    def __init__(self, max_events: int = 500) -> None:
        self._events: deque[APUEvent] = deque(maxlen=max_events)

    def record(
        self,
        event_type: str,
        *,
        model_name: str | None = None,
        gpu_index: int | None = None,
        app_name: str | None = None,
        vram_before_mb: int = 0,
        vram_after_mb: int = 0,
        vram_expected_mb: int = 0,
        duration_ms: float = 0.0,
        success: bool = True,
        error: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> APUEvent:
        """Record an event and return it."""
        # Estimate expected duration based on model VRAM size
        expected_ms = (vram_expected_mb / 1024.0) * _MS_PER_GB if vram_expected_mb > 0 else 0.0
        slow = duration_ms > 2 * expected_ms if expected_ms > 0 else False

        event = APUEvent(
            timestamp=datetime.now(timezone.utc),
            event_type=event_type,
            model_name=model_name,
            gpu_index=gpu_index,
            app_name=app_name,
            vram_before_mb=vram_before_mb,
            vram_after_mb=vram_after_mb,
            vram_expected_mb=vram_expected_mb,
            duration_ms=duration_ms,
            success=success,
            error=error,
            details=details or {},
            expected_duration_ms=expected_ms,
            slow=slow,
        )
        self._events.append(event)

        if not success or error:
            logger.warning("APU event [%s] %s: %s", event_type, model_name or "", error or "failed")
        elif slow:
            logger.warning("APU slow [%s] %s: %.0fms (expected %.0fms)",
                           event_type, model_name or "", duration_ms, expected_ms)

        return event

    def recent(self, limit: int = 100) -> list[APUEvent]:
        """Return most recent events (newest first)."""
        events = list(self._events)
        events.reverse()
        return events[:limit]

    def filter(
        self,
        event_type: str | None = None,
        model_name: str | None = None,
        app_name: str | None = None,
        errors_only: bool = False,
        limit: int = 100,
    ) -> list[APUEvent]:
        """Filter events by criteria."""
        result = []
        for event in reversed(self._events):
            if event_type and event.event_type != event_type:
                continue
            if model_name and event.model_name != model_name:
                continue
            if app_name and event.app_name != app_name:
                continue
            if errors_only and event.success and event.event_type != "drift":
                continue
            result.append(event)
            if len(result) >= limit:
                break
        return result

    def clear(self) -> None:
        self._events.clear()

    def __len__(self) -> int:
        return len(self._events)
