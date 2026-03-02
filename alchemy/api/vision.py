"""Vision API — task submission, analysis, and approval flow.

NEO-TX calls these endpoints to delegate GUI work to Alchemy.
Phase 2: real UI-TARS inference via Ollama + agent loop.
"""

from __future__ import annotations

import asyncio
import base64
import logging
from datetime import datetime, timezone
from uuid import UUID, uuid4

from fastapi import APIRouter, HTTPException, Request

from alchemy.agent.task_manager import TaskManager
from alchemy.agent.vision_agent import VisionAgent
from alchemy.clients.neotx_client import NeoTXClient
from alchemy.models.ollama_client import OllamaClient
from alchemy.schemas import (
    ApprovalDecision,
    ApprovalDecisionResponse,
    TaskStatus,
    TaskStatusResponse,
    VisionAnalyzeRequest,
    VisionAnalyzeResponse,
    VisionTaskRequest,
    VisionTaskResponse,
)
from alchemy.shadow.controller import ShadowDesktopController
from config.settings import settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/vision", tags=["vision"])


def _get_deps(request: Request):
    """Extract shared dependencies from app state."""
    return (
        getattr(request.app.state, "ollama_client", None),
        getattr(request.app.state, "shadow_controller", None),
        getattr(request.app.state, "task_manager", None),
    )


@router.post("/task", response_model=VisionTaskResponse)
async def create_task(req: VisionTaskRequest, request: Request) -> VisionTaskResponse:
    """Submit a GUI task for the vision agent to execute."""
    ollama, controller, task_manager = _get_deps(request)

    if not ollama or not controller:
        raise HTTPException(status_code=503, detail="Ollama or shadow desktop not available")
    if not task_manager:
        raise HTTPException(status_code=503, detail="Task manager not initialized")

    task_id = uuid4()
    now = datetime.now(timezone.utc)
    task_manager.create_task(task_id, req.goal)

    # Parse screen resolution from settings
    parts = settings.resolution.split("x")
    width, height = int(parts[0]), int(parts[1])

    neotx = NeoTXClient(base_url=req.callback_url)
    agent = VisionAgent(
        ollama=ollama,
        controller=controller,
        neotx=neotx,
        task_manager=task_manager,
        model=settings.ollama_cpu_model,
        max_steps=settings.agent_max_steps,
        timeout=settings.agent_timeout,
        screenshot_interval=settings.agent_screenshot_interval,
        approval_timeout=settings.agent_approval_timeout,
        history_window=settings.agent_history_window,
        screen_width=width,
        screen_height=height,
    )

    # Start agent loop as background task
    async_task = asyncio.create_task(agent.run_task(task_id, req.goal))
    task_manager.register_agent_task(task_id, async_task)

    return VisionTaskResponse(task_id=task_id, status=TaskStatus.PENDING, created_at=now)


@router.post("/analyze", response_model=VisionAnalyzeResponse)
async def analyze(req: VisionAnalyzeRequest, request: Request) -> VisionAnalyzeResponse:
    """Analyze a single screenshot and return the next action."""
    ollama, controller, _ = _get_deps(request)

    if not ollama:
        raise HTTPException(status_code=503, detail="Ollama not available")

    # Parse screen resolution
    parts = settings.resolution.split("x")
    width, height = int(parts[0]), int(parts[1])

    neotx = NeoTXClient()  # dummy — not used for single analysis
    task_manager = TaskManager()  # dummy — not used for single analysis
    agent = VisionAgent(
        ollama=ollama,
        controller=controller,
        neotx=neotx,
        task_manager=task_manager,
        model=settings.ollama_cpu_model,
        screen_width=width,
        screen_height=height,
    )

    screenshot = base64.b64decode(req.screenshot_b64)
    return await agent.analyze_single(screenshot, req.goal)


@router.get("/task/{task_id}/status", response_model=TaskStatusResponse)
async def task_status(task_id: UUID, request: Request) -> TaskStatusResponse:
    """Poll the current status of a vision task."""
    _, _, task_manager = _get_deps(request)
    if not task_manager:
        raise HTTPException(status_code=503, detail="Task manager not initialized")

    resp = task_manager.to_status_response(task_id)
    if not resp:
        raise HTTPException(status_code=404, detail="Task not found")
    return resp


@router.post("/task/{task_id}/approve", response_model=ApprovalDecisionResponse)
async def approve_task(
    task_id: UUID, decision: ApprovalDecision, request: Request,
) -> ApprovalDecisionResponse:
    """User approves an APPROVE-tier action. Agent resumes."""
    _, _, task_manager = _get_deps(request)
    if not task_manager:
        raise HTTPException(status_code=503, detail="Task manager not initialized")

    state = task_manager.get_task(task_id)
    if not state:
        raise HTTPException(status_code=404, detail="Task not found")

    task_manager.approve(task_id)
    return ApprovalDecisionResponse(
        task_id=task_id, decision="approved", status=TaskStatus.RUNNING,
    )


@router.post("/task/{task_id}/deny", response_model=ApprovalDecisionResponse)
async def deny_task(
    task_id: UUID, decision: ApprovalDecision, request: Request,
) -> ApprovalDecisionResponse:
    """User denies an APPROVE-tier action. Task is aborted."""
    _, _, task_manager = _get_deps(request)
    if not task_manager:
        raise HTTPException(status_code=503, detail="Task manager not initialized")

    state = task_manager.get_task(task_id)
    if not state:
        raise HTTPException(status_code=404, detail="Task not found")

    task_manager.deny(task_id)
    return ApprovalDecisionResponse(
        task_id=task_id, decision="denied", status=TaskStatus.DENIED,
    )
