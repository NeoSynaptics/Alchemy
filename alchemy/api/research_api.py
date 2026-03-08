"""Research API — submit queries, check status.

POST /v1/research           — Submit a research query (semantic or direct)
GET  /v1/research/{id}/status — Check research task status + results
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Request

from alchemy.api.contract_guard import require_contract
from alchemy.schemas import (
    ResearchMode,
    ResearchResult,
    ResearchSource,
    ResearchTaskRequest,
    ResearchTaskResponse,
    ResearchTaskStatusResponse,
    TaskStatus,
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["research"], dependencies=[Depends(require_contract("research"))])

# In-memory task store with max size cap to prevent unbounded growth
_MAX_TASKS = 1000
_tasks: dict[str, dict] = {}


def _evict_old_tasks() -> None:
    """Remove oldest completed tasks when store exceeds max size."""
    if len(_tasks) <= _MAX_TASKS:
        return
    completed = [
        (tid, t) for tid, t in _tasks.items()
        if t.get("status") in ("completed", "failed", TaskStatus.COMPLETED, TaskStatus.FAILED)
    ]
    completed.sort(key=lambda x: str(x[1].get("created_at", "")))
    for tid, _ in completed[: len(_tasks) - _MAX_TASKS]:
        del _tasks[tid]


@router.post("/research", response_model=ResearchTaskResponse)
async def submit_research(req: ResearchTaskRequest, request: Request):
    """Submit a research query for background processing."""
    from alchemy.research.collector import PageCollector
    from alchemy.research.engine import ResearchEngine, ResearchProgress
    from alchemy.research.searcher import SearchProvider
    from alchemy.research.synthesizer import Synthesizer
    from config.settings import settings

    if not settings.research_enabled:
        raise HTTPException(status_code=503, detail="Research module disabled")

    ollama = getattr(request.app.state, "ollama_client", None)
    if not ollama:
        raise HTTPException(status_code=503, detail="Ollama client not available")

    browser_mgr = getattr(request.app.state, "browser_manager", None)  # May be None

    # Validate direct mode
    if req.mode == ResearchMode.DIRECT and not req.urls:
        raise HTTPException(status_code=400, detail="Direct mode requires at least one URL")

    task_id = str(uuid4())
    now = datetime.now(timezone.utc)

    # Build pipeline components
    synthesizer = Synthesizer(
        ollama_client=ollama,
        model=settings.research_model,
        think=settings.research_think,
        temperature=settings.research_temperature,
        max_tokens=settings.research_max_tokens,
    )
    searcher = SearchProvider(max_results_per_query=5)
    collector = PageCollector(
        timeout=settings.research_fetch_timeout,
        browser_manager=browser_mgr,
        max_pages=settings.research_max_pages,
    )
    engine = ResearchEngine(
        synthesizer=synthesizer,
        searcher=searcher,
        collector=collector,
        max_queries=settings.research_max_queries,
        top_k=settings.research_top_k,
    )

    progress = ResearchProgress()

    _evict_old_tasks()
    _tasks[task_id] = {
        "task_id": task_id,
        "status": TaskStatus.RUNNING,
        "created_at": now,
        "progress": progress,
    }

    # Run in background
    asyncio.create_task(_run_research(task_id, req, engine, progress))

    return ResearchTaskResponse(task_id=task_id, status=TaskStatus.PENDING, created_at=now)


@router.get("/research/{task_id}/status", response_model=ResearchTaskStatusResponse)
async def get_research_status(task_id: str):
    """Get the status and results of a research task."""
    task = _tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Research task not found")

    progress = task["progress"]

    # Build result if completed
    result = None
    if progress.result:
        result = ResearchResult(
            answer=progress.result.answer,
            sources=[
                ResearchSource(title=s["title"], excerpt=s["excerpt"])
                for s in progress.result.sources
            ],
            pages_fetched=progress.pages_fetched,
            pages_used=progress.pages_used,
        )

    return ResearchTaskStatusResponse(
        task_id=task["task_id"],
        status=task["status"],
        result=result,
        queries_generated=progress.queries_generated,
        pages_fetched=progress.pages_fetched,
        pipeline_stage=progress.stage.value,
        total_ms=progress.total_ms,
        error=progress.error,
    )


async def _run_research(task_id: str, req: ResearchTaskRequest, engine, progress):
    """Background task runner for research pipeline."""
    try:
        if req.mode == ResearchMode.DIRECT:
            await engine.run_direct(req.query, req.urls, progress)
        else:
            await engine.run_semantic(req.query, progress)

        _tasks[task_id]["status"] = (
            TaskStatus.COMPLETED
            if progress.stage.value == "completed"
            else TaskStatus.FAILED
        )

        logger.info(
            "Research task %s finished: %s (%.0fms)",
            task_id,
            progress.stage.value,
            progress.total_ms,
        )

    except Exception as e:
        logger.error("Research task %s failed: %s", task_id, e)
        _tasks[task_id]["status"] = TaskStatus.FAILED
        progress.error = str(e)
