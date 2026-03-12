"""Inference metrics — per-call records for APU gateway diagnostics."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone


@dataclass
class InferenceRecord:
    """One inference call through the gateway."""

    caller: str
    model: str
    priority: int
    method: str  # "chat" | "chat_think" | "chat_stream" | "generate" | "embed"
    elapsed_ms: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    success: bool = True
    error: str | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat()
        d["elapsed_ms"] = round(self.elapsed_ms, 1)
        return d


class InferenceMetrics:
    """Ring buffer of inference records."""

    def __init__(self, max_records: int = 500) -> None:
        self._records: deque[InferenceRecord] = deque(maxlen=max_records)

    def record(self, rec: InferenceRecord) -> None:
        self._records.append(rec)

    def recent(self, limit: int = 50) -> list[dict]:
        records = list(self._records)
        records.reverse()
        return [r.to_dict() for r in records[:limit]]

    def by_caller(self, caller: str, limit: int = 50) -> list[dict]:
        result = []
        for r in reversed(self._records):
            if r.caller == caller:
                result.append(r.to_dict())
                if len(result) >= limit:
                    break
        return result

    def by_model(self, model: str, limit: int = 50) -> list[dict]:
        result = []
        for r in reversed(self._records):
            if r.model == model:
                result.append(r.to_dict())
                if len(result) >= limit:
                    break
        return result

    def __len__(self) -> int:
        return len(self._records)
