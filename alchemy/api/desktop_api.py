"""Desktop automation API — shadow agent for native Windows apps.

POST /v1/desktop/task           — Submit a desktop automation task
GET  /v1/desktop/task/{id}      — Check task status and steps
POST /v1/desktop/summon         — Switch to ghost mode (orange cursor visible)
POST /v1/desktop/dismiss        — Switch back to shadow mode (invisible)
GET  /v1/desktop/mode           — Get current mode

Default mode is "shadow" — the agent works invisibly in the background.
When summoned, the orange AI cursor appears so the user can see what it does.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from alchemy.api.contract_guard import require_contract
from alchemy.desktop.agent import DesktopStep, DesktopTaskStatus

logger = logging.getLogger(__name__)
router = APIRouter(tags=["desktop"], dependencies=[Depends(require_contract("desktop"))])

# In-memory task store with max size cap to prevent unbounded growth
_MAX_TASKS = 1000
_tasks: dict[str, dict] = {}


def _evict_old_tasks() -> None:
    """Remove oldest completed tasks when store exceeds max size."""
    if len(_tasks) <= _MAX_TASKS:
        return
    completed = [
        (tid, t) for tid, t in _tasks.items()
        if t.get("status") in ("completed", "failed")
    ]
    completed.sort(key=lambda x: x[1].get("created_at", ""))
    for tid, _ in completed[: len(_tasks) - _MAX_TASKS]:
        del _tasks[tid]


# --- Request / Response schemas ---

class DesktopTaskRequest(BaseModel):
    goal: str
    max_steps: int = 20
    mode: str | None = None  # "shadow" or "ghost" — overrides default for this task


class DesktopStepInfo(BaseModel):
    step: int
    action_type: str
    x: int | None = None
    y: int | None = None
    text: str | None = None
    thought: str = ""
    inference_ms: float = 0.0
    execution_ms: float = 0.0
    success: bool = True
    error: str | None = None


class DesktopTaskResponse(BaseModel):
    task_id: str
    status: str = "pending"
    created_at: datetime


class DesktopTaskStatusResponse(BaseModel):
    task_id: str
    status: str
    current_step: int = 0
    steps: list[DesktopStepInfo] = Field(default_factory=list)
    total_ms: float = 0.0
    error: str | None = None


@router.post("/desktop/task", response_model=DesktopTaskResponse)
async def submit_desktop_task(req: DesktopTaskRequest, request: Request):
    """Submit a desktop automation task.

    Default mode is "shadow" (invisible). Pass mode="ghost" to show the
    orange AI cursor. When the task finishes, mode reverts to what it was before.
    """
    desktop_agent = getattr(request.app.state, "desktop_agent", None)

    if desktop_agent is None:
        raise HTTPException(status_code=503, detail="Desktop agent not available")

    # Handle per-task mode override
    controller = desktop_agent._controller
    if req.mode and req.mode in ("shadow", "ghost"):
        if req.mode == "ghost":
            controller.summon()
        else:
            controller.dismiss()

    task_id = str(uuid4())
    now = datetime.now(timezone.utc)

    _evict_old_tasks()
    _tasks[task_id] = {
        "task_id": task_id,
        "status": "running",
        "created_at": now,
        "steps": [],
        "error": None,
        "total_ms": 0.0,
    }

    asyncio.create_task(_run_desktop_task(task_id, req, desktop_agent, req.mode))

    return DesktopTaskResponse(task_id=task_id, status="pending", created_at=now)


@router.post("/desktop/summon")
async def summon_desktop_agent(request: Request):
    """Switch to ghost mode — the orange AI cursor becomes visible."""
    desktop_agent = getattr(request.app.state, "desktop_agent", None)
    if desktop_agent is None:
        raise HTTPException(status_code=503, detail="Desktop agent not available")
    desktop_agent._controller.summon()
    return {"mode": "ghost", "message": "Desktop agent summoned — orange cursor visible"}


@router.post("/desktop/dismiss")
async def dismiss_desktop_agent(request: Request):
    """Switch back to shadow mode — completely invisible."""
    desktop_agent = getattr(request.app.state, "desktop_agent", None)
    if desktop_agent is None:
        raise HTTPException(status_code=503, detail="Desktop agent not available")
    desktop_agent._controller.dismiss()
    return {"mode": "shadow", "message": "Desktop agent dismissed — invisible"}


@router.get("/desktop/mode")
async def get_desktop_mode(request: Request):
    """Get the current desktop agent mode."""
    desktop_agent = getattr(request.app.state, "desktop_agent", None)
    if desktop_agent is None:
        raise HTTPException(status_code=503, detail="Desktop agent not available")
    return {"mode": desktop_agent._controller.mode}


@router.get("/desktop/task/{task_id}", response_model=DesktopTaskStatusResponse)
async def get_desktop_task_status(task_id: str):
    """Get the status of a desktop automation task."""
    task = _tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    return DesktopTaskStatusResponse(
        task_id=task["task_id"],
        status=task["status"],
        current_step=len(task["steps"]),
        steps=task["steps"],
        total_ms=task["total_ms"],
        error=task["error"],
    )


async def _run_desktop_task(task_id: str, req: DesktopTaskRequest, desktop_agent, mode: str | None):
    """Background task runner for desktop automation."""
    # Revert mode after task if it was overridden
    prev_mode = desktop_agent._controller.mode
    try:
        desktop_agent._max_steps = req.max_steps
        result = await desktop_agent.run(req.goal)

        _tasks[task_id]["status"] = result.status.value
        _tasks[task_id]["total_ms"] = result.total_ms
        _tasks[task_id]["error"] = result.error
        _tasks[task_id]["steps"] = [
            DesktopStepInfo(
                step=s.step,
                action_type=s.action_type,
                x=s.x,
                y=s.y,
                text=s.text,
                thought=s.thought,
                inference_ms=s.inference_ms,
                execution_ms=s.execution_ms,
                success=s.success,
                error=s.error,
            )
            for s in result.steps
        ]

        logger.info(
            "Desktop task %s finished: %s (%d steps, %.0fms)",
            task_id, result.status.value, len(result.steps), result.total_ms,
        )

    except Exception as e:
        logger.error("Desktop task %s failed: %s", task_id, e)
        _tasks[task_id]["status"] = "failed"
        _tasks[task_id]["error"] = str(e)
    finally:
        # Revert mode if it was overridden for this task
        if mode and prev_mode != desktop_agent._controller.mode:
            if prev_mode == "shadow":
                desktop_agent._controller.dismiss()
            else:
                desktop_agent._controller.summon()
