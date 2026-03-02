"""In-memory task state management with asyncio.Event signaling.

Tracks vision agent tasks, their status, and provides approval/deny
signaling between the API endpoints and running agent loops.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import UUID

from alchemy.schemas import TaskStatus, TaskStatusResponse, VisionAction

logger = logging.getLogger(__name__)


@dataclass
class TaskState:
    """Internal state for a running vision task."""
    task_id: UUID
    goal: str
    status: TaskStatus = TaskStatus.PENDING
    current_step: int = 0
    last_action: VisionAction | None = None
    error: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    # Approval signaling
    approval_event: asyncio.Event = field(default_factory=asyncio.Event)
    approved: bool | None = None
    # Cancellation
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)


class TaskManager:
    """Thread-safe in-memory task store with asyncio.Event signaling."""

    def __init__(self):
        self._tasks: dict[UUID, TaskState] = {}
        self._agent_tasks: dict[UUID, asyncio.Task] = {}

    def create_task(self, task_id: UUID, goal: str) -> TaskState:
        """Register a new task."""
        state = TaskState(task_id=task_id, goal=goal)
        self._tasks[task_id] = state
        return state

    def get_task(self, task_id: UUID) -> TaskState | None:
        """Look up a task by ID."""
        return self._tasks.get(task_id)

    def update_task(self, task_id: UUID, **kwargs):
        """Update task fields."""
        state = self._tasks.get(task_id)
        if not state:
            return
        for key, value in kwargs.items():
            if hasattr(state, key):
                setattr(state, key, value)
        state.updated_at = datetime.now(timezone.utc)

    def approve(self, task_id: UUID):
        """Signal approval for a waiting task."""
        state = self._tasks.get(task_id)
        if state:
            state.approved = True
            state.approval_event.set()
            logger.info("Task %s approved", task_id)

    def deny(self, task_id: UUID):
        """Signal denial for a waiting task."""
        state = self._tasks.get(task_id)
        if state:
            state.approved = False
            state.approval_event.set()
            state.cancel_event.set()
            logger.info("Task %s denied", task_id)

    def cancel(self, task_id: UUID):
        """Cancel a running task."""
        state = self._tasks.get(task_id)
        if state:
            state.cancel_event.set()
        agent_task = self._agent_tasks.get(task_id)
        if agent_task and not agent_task.done():
            agent_task.cancel()
            logger.info("Task %s cancelled", task_id)

    def register_agent_task(self, task_id: UUID, task: asyncio.Task):
        """Track the running asyncio.Task for cancellation."""
        self._agent_tasks[task_id] = task

    def to_status_response(self, task_id: UUID) -> TaskStatusResponse | None:
        """Convert TaskState to API response model."""
        state = self._tasks.get(task_id)
        if not state:
            return None
        return TaskStatusResponse(
            task_id=state.task_id,
            status=state.status,
            current_step=state.current_step,
            last_action=state.last_action,
            error=state.error,
            created_at=state.created_at,
            updated_at=state.updated_at,
        )
