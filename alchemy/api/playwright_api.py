"""Playwright agent API — submit tasks, check status.

POST /v1/playwright/task   — Submit a GUI task
GET  /v1/playwright/task/{id}/status — Check task status
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request

from alchemy.schemas import (
    PlaywrightStepInfo,
    PlaywrightTaskRequest,
    PlaywrightTaskResponse,
    PlaywrightTaskStatusResponse,
    TaskStatus,
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["playwright"])

# In-memory task store (same pattern as vision API)
_tasks: dict[str, dict] = {}


@router.post("/playwright/task", response_model=PlaywrightTaskResponse)
async def submit_task(req: PlaywrightTaskRequest, request: Request):
    """Submit a GUI automation task to the Playwright agent."""
    pw_agent = getattr(request.app.state, "pw_agent", None)
    browser_mgr = getattr(request.app.state, "browser_manager", None)

    if not pw_agent or not browser_mgr:
        raise HTTPException(status_code=503, detail="Playwright agent not available")

    task_id = str(uuid4())
    now = datetime.now(timezone.utc)

    _tasks[task_id] = {
        "task_id": task_id,
        "status": TaskStatus.RUNNING,
        "created_at": now,
        "steps": [],
        "error": None,
        "total_ms": 0.0,
    }

    # Run task in background
    asyncio.create_task(_run_task(task_id, req, pw_agent, browser_mgr))

    return PlaywrightTaskResponse(task_id=task_id, status=TaskStatus.PENDING, created_at=now)


@router.get("/playwright/task/{task_id}/status", response_model=PlaywrightTaskStatusResponse)
async def get_task_status(task_id: str):
    """Get the status of a Playwright agent task."""
    task = _tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    return PlaywrightTaskStatusResponse(
        task_id=task["task_id"],
        status=task["status"],
        current_step=len(task["steps"]),
        steps=task["steps"],
        total_ms=task["total_ms"],
        error=task["error"],
    )


async def _run_task(task_id: str, req: PlaywrightTaskRequest, pw_agent, browser_mgr):
    """Background task runner."""
    try:
        # Get or create page
        if req.cdp_endpoint:
            page = await browser_mgr.connect_cdp(req.cdp_endpoint)
        else:
            page = await browser_mgr.new_page(req.url)

        # Run the agent loop
        result = await pw_agent.run_task(req.goal, page)

        # Store results
        _tasks[task_id]["status"] = (
            TaskStatus.COMPLETED if result.status.value == "completed"
            else TaskStatus.WAITING_APPROVAL if result.status.value == "waiting_approval"
            else TaskStatus.FAILED
        )
        _tasks[task_id]["total_ms"] = result.total_ms
        _tasks[task_id]["error"] = result.error
        _tasks[task_id]["steps"] = [
            PlaywrightStepInfo(
                step=s.step,
                action_type=s.action.type,
                ref=s.action.ref,
                text=s.action.text,
                thought=s.action.thought,
                success=s.success,
                error=s.error,
                inference_ms=s.inference_ms,
                execution_ms=s.execution_ms,
                escalated=s.escalated,
            )
            for s in result.steps
        ]
        _tasks[task_id]["escalation_count"] = result.escalation_count

        logger.info("Task %s finished: %s (%d steps, %d escalations, %.0fms)",
                    task_id, result.status.value, result.total_steps,
                    result.escalation_count, result.total_ms)

    except Exception as e:
        logger.error("Task %s failed: %s", task_id, e)
        _tasks[task_id]["status"] = TaskStatus.FAILED
        _tasks[task_id]["error"] = str(e)
