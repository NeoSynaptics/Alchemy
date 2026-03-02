"""Vision API — task submission, analysis, and approval flow.

NEO-TX calls these endpoints to delegate GUI work to Alchemy.
All responses are stubs (Phase 0) — correct types, mock data.
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID, uuid4

from fastapi import APIRouter, HTTPException

from alchemy.schemas import (
    ActionTier,
    ApprovalDecision,
    ApprovalDecisionResponse,
    TaskStatus,
    TaskStatusResponse,
    VisionAction,
    VisionAnalyzeRequest,
    VisionAnalyzeResponse,
    VisionTaskRequest,
    VisionTaskResponse,
)

router = APIRouter(prefix="/vision", tags=["vision"])

# In-memory task store (Phase 0 only — replaced by real agent in Phase 1+)
_tasks: dict[UUID, TaskStatusResponse] = {}


@router.post("/task", response_model=VisionTaskResponse)
async def create_task(req: VisionTaskRequest) -> VisionTaskResponse:
    """Submit a GUI task for the vision agent to execute."""
    task_id = uuid4()
    now = datetime.now(timezone.utc)
    _tasks[task_id] = TaskStatusResponse(
        task_id=task_id,
        status=TaskStatus.PENDING,
        current_step=0,
        created_at=now,
        updated_at=now,
    )
    return VisionTaskResponse(task_id=task_id, status=TaskStatus.PENDING, created_at=now)


@router.post("/analyze", response_model=VisionAnalyzeResponse)
async def analyze(req: VisionAnalyzeRequest) -> VisionAnalyzeResponse:
    """Analyze a single screenshot and return the next action."""
    return VisionAnalyzeResponse(
        action=VisionAction(
            action="click",
            x=100,
            y=200,
            reasoning="Mock: would click target element",
            tier=ActionTier.AUTO,
        ),
        model="ui-tars:72b",
        inference_ms=2500.0,
    )


@router.get("/task/{task_id}/status", response_model=TaskStatusResponse)
async def task_status(task_id: UUID) -> TaskStatusResponse:
    """Poll the current status of a vision task."""
    if task_id not in _tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    return _tasks[task_id]


@router.post("/task/{task_id}/approve", response_model=ApprovalDecisionResponse)
async def approve_task(task_id: UUID, decision: ApprovalDecision) -> ApprovalDecisionResponse:
    """User approves an APPROVE-tier action. Agent resumes."""
    if task_id not in _tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    _tasks[task_id].status = TaskStatus.RUNNING
    _tasks[task_id].updated_at = datetime.now(timezone.utc)
    return ApprovalDecisionResponse(
        task_id=task_id, decision="approved", status=TaskStatus.RUNNING
    )


@router.post("/task/{task_id}/deny", response_model=ApprovalDecisionResponse)
async def deny_task(task_id: UUID, decision: ApprovalDecision) -> ApprovalDecisionResponse:
    """User denies an APPROVE-tier action. Task is aborted."""
    if task_id not in _tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    _tasks[task_id].status = TaskStatus.DENIED
    _tasks[task_id].updated_at = datetime.now(timezone.utc)
    return ApprovalDecisionResponse(
        task_id=task_id, decision="denied", status=TaskStatus.DENIED
    )
