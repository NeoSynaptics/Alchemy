"""Vision API — task submission, analysis, and approval flow.

AlchemyVoice calls these endpoints to delegate GUI work to Alchemy.
Phase 4: optimized inference (streaming, dual-model, adaptive timeouts).
"""

from __future__ import annotations

import asyncio
import base64
import logging
from datetime import datetime, timezone
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, Request

from alchemy.api.contract_guard import require_contract

from alchemy.click.task_manager import TaskManager
from alchemy.click.vision_agent import VisionAgent
from alchemy.clients.voice_callback import VoiceCallbackClient
from alchemy.models.ollama_client import OllamaClient
from alchemy.router.context_builder import ContextBuilder
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
from config.settings import settings

logger = logging.getLogger(__name__)
router = APIRouter(
    prefix="/vision",
    tags=["vision"],
    dependencies=[Depends(require_contract("click"))],
)


def _get_deps(request: Request):
    """Extract shared dependencies from app state."""
    return (
        getattr(request.app.state, "ollama_client", None),
        getattr(request.app.state, "task_manager", None),
    )


def _get_context_builder(request: Request) -> ContextBuilder | None:
    """Build a ContextBuilder from cached environment, if router is enabled."""
    env = getattr(request.app.state, "environment", None)
    if not env or not settings.router_enabled:
        return None
    return ContextBuilder(
        env,
        category_hints=settings.router_category_hints,
        recovery_nudges=settings.router_recovery_nudges,
        completion_criteria=settings.router_completion_criteria,
    )


def _make_omniparser():
    """Create OmniParser if enabled in settings."""
    if not settings.click_omniparser_enabled:
        return None
    from alchemy.click.flow.omniparser import OmniParser
    return OmniParser(
        confidence_threshold=settings.click_omniparser_confidence,
        device=settings.click_omniparser_device,
        model_path=settings.click_omniparser_model_path or None,
    )


def _make_agent(
    ollama: OllamaClient,
    task_manager: TaskManager,
    voice_cb: VoiceCallbackClient,
    context_builder: ContextBuilder | None,
) -> VisionAgent:
    """Create a VisionAgent with all current settings wired in."""
    width = settings.desktop_screenshot_width or 1920
    height = settings.desktop_screenshot_height or 1080

    return VisionAgent(
        ollama=ollama,
        voice_cb=voice_cb,
        task_manager=task_manager,
        model=settings.ollama_cpu_model,
        max_steps=settings.click_max_steps,
        timeout=settings.click_timeout,
        screenshot_interval=settings.click_screenshot_interval,
        approval_timeout=settings.click_approval_timeout,
        history_window=settings.click_history_window,
        screen_width=width,
        screen_height=height,
        context_builder=context_builder,
        use_streaming=settings.click_use_streaming,
        temperature=settings.ollama_temperature,
        max_tokens=settings.ollama_max_tokens,
        omniparser=_make_omniparser(),
    )


@router.post("/task", response_model=VisionTaskResponse)
async def create_task(req: VisionTaskRequest, request: Request) -> VisionTaskResponse:
    """Submit a GUI task for the vision agent to execute."""
    ollama, task_manager = _get_deps(request)

    if not ollama:
        raise HTTPException(status_code=503, detail="Ollama not available")
    if not task_manager:
        raise HTTPException(status_code=503, detail="Task manager not initialized")

    task_id = uuid4()
    now = datetime.now(timezone.utc)
    task_manager.create_task(task_id, req.goal)

    voice_cb = VoiceCallbackClient(base_url=req.callback_url)
    agent = _make_agent(ollama, task_manager, voice_cb, _get_context_builder(request))

    async def _run_and_close():
        try:
            return await agent.run_task(task_id, req.goal)
        finally:
            await voice_cb.close()

    async_task = asyncio.create_task(_run_and_close())
    task_manager.register_agent_task(task_id, async_task)

    return VisionTaskResponse(task_id=task_id, status=TaskStatus.PENDING, created_at=now)


@router.post("/analyze", response_model=VisionAnalyzeResponse)
async def analyze(req: VisionAnalyzeRequest, request: Request) -> VisionAnalyzeResponse:
    """Analyze a single screenshot and return the next action."""
    ollama, _ = _get_deps(request)

    if not ollama:
        raise HTTPException(status_code=503, detail="Ollama not available")

    voice_cb = VoiceCallbackClient()
    task_manager = TaskManager()
    agent = _make_agent(ollama, task_manager, voice_cb, _get_context_builder(request))

    try:
        screenshot = base64.b64decode(req.screenshot_b64)
        return await agent.analyze_single(screenshot, req.goal)
    finally:
        await voice_cb.close()


@router.get("/task/{task_id}/status", response_model=TaskStatusResponse)
async def task_status(task_id: UUID, request: Request) -> TaskStatusResponse:
    """Poll the current status of a vision task."""
    _, task_manager = _get_deps(request)
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
    _, task_manager = _get_deps(request)
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
    _, task_manager = _get_deps(request)
    if not task_manager:
        raise HTTPException(status_code=503, detail="Task manager not initialized")

    state = task_manager.get_task(task_id)
    if not state:
        raise HTTPException(status_code=404, detail="Task not found")

    task_manager.deny(task_id)
    return ApprovalDecisionResponse(
        task_id=task_id, decision="denied", status=TaskStatus.DENIED,
    )
