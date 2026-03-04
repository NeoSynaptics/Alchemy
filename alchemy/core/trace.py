"""Agent trace — immutable record of an agent run for replay and debugging.

Every step records the snapshot hash, LLM I/O, parsed action, and timing.
Traces can be serialized to JSON for post-mortem analysis.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field


@dataclass
class TraceEntry:
    """Single step in an agent trace."""

    step: int
    timestamp_ms: float
    snapshot_hash: str  # SHA256 of accessibility tree text
    llm_input_hash: str  # SHA256 of prompt sent to model
    llm_output: str  # Raw model response
    parsed_action: str  # "click @e5" or "type @e3 'hello'"
    success: bool
    inference_ms: float = 0.0
    execution_ms: float = 0.0
    escalated: bool = False
    error: str | None = None


@dataclass
class AgentTrace:
    """Immutable record of a complete agent run."""

    task: str = ""
    started_at: float = 0.0
    entries: list[TraceEntry] = field(default_factory=list)

    def record(self, entry: TraceEntry) -> None:
        """Append a step to the trace."""
        self.entries.append(entry)

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict."""
        return {
            "task": self.task,
            "started_at": self.started_at,
            "total_steps": len(self.entries),
            "entries": [
                {
                    "step": e.step,
                    "timestamp_ms": e.timestamp_ms,
                    "snapshot_hash": e.snapshot_hash,
                    "llm_input_hash": e.llm_input_hash,
                    "llm_output": e.llm_output,
                    "parsed_action": e.parsed_action,
                    "success": e.success,
                    "inference_ms": e.inference_ms,
                    "execution_ms": e.execution_ms,
                    "escalated": e.escalated,
                    "error": e.error,
                }
                for e in self.entries
            ],
        }

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: dict) -> AgentTrace:
        """Deserialize from a dict."""
        trace = cls(task=data.get("task", ""), started_at=data.get("started_at", 0.0))
        for e in data.get("entries", []):
            trace.entries.append(TraceEntry(**e))
        return trace

    @classmethod
    def from_json(cls, raw: str) -> AgentTrace:
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(raw))


def hash_text(text: str) -> str:
    """SHA256 hash of text — for deterministic snapshot/prompt comparison."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def make_trace_entry(
    step: int,
    snapshot_text: str,
    prompt: str,
    llm_output: str,
    action_str: str,
    success: bool,
    inference_ms: float = 0.0,
    execution_ms: float = 0.0,
    escalated: bool = False,
    error: str | None = None,
) -> TraceEntry:
    """Convenience factory for creating trace entries."""
    return TraceEntry(
        step=step,
        timestamp_ms=time.monotonic() * 1000,
        snapshot_hash=hash_text(snapshot_text),
        llm_input_hash=hash_text(prompt),
        llm_output=llm_output,
        parsed_action=action_str,
        success=success,
        inference_ms=inference_ms,
        execution_ms=execution_ms,
        escalated=escalated,
        error=error,
    )
