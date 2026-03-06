"""Task decomposer — breaks complex goals into sub-tasks.

Uses the 14B model to decompose ambiguous multi-step goals into concrete
sub-tasks. Simple goals pass through without decomposition.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from uuid import UUID, uuid4

logger = logging.getLogger(__name__)


class SubTaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class SubTask:
    """A single step in a decomposed plan."""

    id: UUID = field(default_factory=uuid4)
    description: str = ""
    status: SubTaskStatus = SubTaskStatus.PENDING
    result: str = ""
    error: str = ""
    order: int = 0

    def complete(self, result: str = "") -> None:
        self.status = SubTaskStatus.COMPLETED
        self.result = result

    def fail(self, error: str = "") -> None:
        self.status = SubTaskStatus.FAILED
        self.error = error

    def skip(self, reason: str = "") -> None:
        self.status = SubTaskStatus.SKIPPED
        self.result = reason


@dataclass
class TaskPlan:
    """A decomposed plan with ordered sub-tasks."""

    id: UUID = field(default_factory=uuid4)
    goal: str = ""
    sub_tasks: list[SubTask] = field(default_factory=list)
    is_simple: bool = False  # True if no decomposition needed

    @property
    def current_task(self) -> SubTask | None:
        """The first pending or running sub-task."""
        for task in self.sub_tasks:
            if task.status in (SubTaskStatus.PENDING, SubTaskStatus.RUNNING):
                return task
        return None

    @property
    def is_complete(self) -> bool:
        return all(
            t.status in (SubTaskStatus.COMPLETED, SubTaskStatus.SKIPPED)
            for t in self.sub_tasks
        )

    @property
    def has_failed(self) -> bool:
        return any(t.status == SubTaskStatus.FAILED for t in self.sub_tasks)

    @property
    def progress(self) -> tuple[int, int]:
        """(completed_count, total_count)."""
        done = sum(
            1 for t in self.sub_tasks
            if t.status in (SubTaskStatus.COMPLETED, SubTaskStatus.SKIPPED)
        )
        return done, len(self.sub_tasks)

    def summary(self) -> str:
        """Human-readable progress summary."""
        done, total = self.progress
        lines = [f"Plan: {self.goal} ({done}/{total} steps)"]
        for t in self.sub_tasks:
            marker = {
                SubTaskStatus.PENDING: "  ",
                SubTaskStatus.RUNNING: ">>",
                SubTaskStatus.COMPLETED: "OK",
                SubTaskStatus.FAILED: "XX",
                SubTaskStatus.SKIPPED: "--",
            }.get(t.status, "  ")
            lines.append(f"  [{marker}] {t.order}. {t.description}")
        return "\n".join(lines)


# -- Complexity heuristics --

_MULTI_STEP_SIGNALS = [
    r"\b(?:and then|after that|next|finally|first|second|third)\b",
    r"\b(?:steps?|phase|stage)\b",
    r"[;,]\s*(?:then|and)\s",
]
_MULTI_STEP_COMPILED = [re.compile(p, re.IGNORECASE) for p in _MULTI_STEP_SIGNALS]

_SIMPLE_PATTERNS = [
    r"^(?:open|close|click|type|scroll|search)\s",
    r"^(?:what|how|when|where|who|why)\s",
    r"^(?:tell me|show me|explain)\s",
]
_SIMPLE_COMPILED = [re.compile(p, re.IGNORECASE) for p in _SIMPLE_PATTERNS]


def is_complex_goal(goal: str) -> bool:
    """Heuristic: does this goal need decomposition?

    Multi-step instructions with sequencing words always get decomposed.
    Simple questions and single-action commands pass through.
    """
    # Multi-step signals → always decompose (check first, overrides simple)
    for pattern in _MULTI_STEP_COMPILED:
        if pattern.search(goal):
            return True

    # Simple patterns → no decomposition
    for pattern in _SIMPLE_COMPILED:
        if pattern.search(goal):
            return False

    # Long goals are more likely complex
    word_count = len(goal.split())
    if word_count > 20:
        return True

    return False


def parse_plan_from_text(text: str, goal: str) -> TaskPlan:
    """Parse a numbered plan from 14B model output into a TaskPlan.

    Expected format from the model:
    1. First step
    2. Second step
    3. Third step
    """
    plan = TaskPlan(goal=goal)

    # Match numbered lines (1. Step, 2. Step, etc.)
    numbered = re.findall(r"^\s*(\d+)[.)]\s*(.+)$", text, re.MULTILINE)

    if numbered:
        for i, (num, desc) in enumerate(numbered):
            plan.sub_tasks.append(
                SubTask(description=desc.strip(), order=i + 1)
            )
    else:
        # Fallback: split on newlines
        lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
        for i, line in enumerate(lines):
            # Strip leading bullets/dashes
            cleaned = re.sub(r"^[-*]+\s*", "", line)
            if cleaned:
                plan.sub_tasks.append(
                    SubTask(description=cleaned, order=i + 1)
                )

    if not plan.sub_tasks:
        # No decomposition possible — treat as single task
        plan.is_simple = True
        plan.sub_tasks.append(SubTask(description=goal, order=1))

    return plan
