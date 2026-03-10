"""AlchemyMemory FastAPI routes.

All memory endpoints live under /v1/memory. Mounted in server.py.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request, UploadFile, File, Query
from fastapi.responses import FileResponse, StreamingResponse

from alchemy.memory.api.schemas import (
    ActivityResponse,
    BucketResponse,
    ContextPackResponse,
    HealthResponse,
    IngestRequest,
    MemorySearchRequest,
    SearchTaskResponse,
    TagRequest,
    TimelineEventResponse,
    TimelineQueryRequest,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/memory", tags=["memory"])

# In-memory search task store (ephemeral)
_search_tasks: dict[str, dict[str, Any]] = {}


def _get_memory(request: Request):
    mem = getattr(request.app.state, "memory_system", None)
    if mem is None:
        raise HTTPException(503, "AlchemyMemory not initialized")
    return mem


@router.get("/health")
async def health(request: Request) -> HealthResponse:
    mem = _get_memory(request)
    data = {
        "status": "running" if mem.capture.is_running() else "stopped",
        "timeline": mem.timeline.stats(),
        "vectors": {"count": mem.vectors.count()},
        "stm": mem.stm.stats(),
        "activity": mem.classifier.current_activity,
        "storage_path": mem.settings.storage_path,
    }
    return HealthResponse(**data)


@router.post("/search")
async def search(request: Request, body: MemorySearchRequest) -> SearchTaskResponse:
    """Fire a unified search across STM + LTM + internet. Returns a task_id."""
    mem = _get_memory(request)
    task_id = str(uuid.uuid4())
    _search_tasks[task_id] = {"status": "running", "results": None, "error": None}

    asyncio.create_task(_run_search(task_id, mem, body))
    return SearchTaskResponse(task_id=task_id)


async def _run_search(task_id: str, mem, body: MemorySearchRequest) -> None:
    try:
        start_ts = None
        end_ts = None
        if body.time_range_hours:
            end_ts = time.time()
            start_ts = end_ts - (body.time_range_hours * 3600)

        # STM context
        stm_events = mem.stm.recent(window_minutes=60, limit=mem.settings.max_stm_results)
        stm_results = [
            {"type": "stm", "summary": e.summary, "app_name": e.app_name, "ts": e.ts}
            for e in stm_events if body.query.lower() in e.summary.lower()
        ]

        # LTM semantic search
        ltm_results = await mem.searcher.search(
            query=body.query,
            start_ts=start_ts,
            end_ts=end_ts,
            limit=body.max_ltm_results,
            semantic=True,
        )

        # Internet search (optional, uses research module if available)
        internet_results: list[dict] = []
        if body.include_internet:
            try:
                from alchemy.research.engine import ResearchEngine
                engine = ResearchEngine.__new__(ResearchEngine)
                # Internet search is best-effort
            except ImportError:
                pass

        combined = {
            "stm": stm_results,
            "ltm": ltm_results,
            "internet": internet_results,
        }
        _search_tasks[task_id] = {"status": "done", "results": combined, "error": None}
    except Exception as e:
        logger.exception("Search task %s failed", task_id)
        _search_tasks[task_id] = {"status": "error", "results": None, "error": str(e)}


@router.get("/search/{task_id}/status")
async def search_status(request: Request, task_id: str) -> dict:
    task = _search_tasks.get(task_id)
    if task is None:
        raise HTTPException(404, "Search task not found")
    return task


@router.post("/timeline/query")
async def timeline_query(
    request: Request, body: TimelineQueryRequest
) -> list[TimelineEventResponse]:
    mem = _get_memory(request)
    results = await mem.searcher.search(
        query=body.query,
        start_ts=body.start_ts,
        end_ts=body.end_ts,
        event_types=body.event_types or None,
        app_names=body.app_names or None,
        limit=body.limit,
        semantic=body.semantic,
    )
    return [
        TimelineEventResponse(
            id=r["id"],
            ts=r["ts"],
            event_type=r["event_type"],
            source=r.get("source", ""),
            summary=r["summary"],
            app_name=r["app_name"],
            screenshot_url=f"/v1/memory/screenshot/{r['id']}" if r.get("screenshot_path") else None,
            score=r.get("score", 0.0),
            meta=r.get("meta", {}),
        )
        for r in results
    ]


@router.get("/timeline/recent")
async def timeline_recent(
    request: Request, limit: int = 20
) -> list[TimelineEventResponse]:
    mem = _get_memory(request)
    events = mem.timeline.recent(limit=limit)
    return [
        TimelineEventResponse(
            id=e.id,
            ts=e.ts,
            event_type=e.event_type,
            source=e.source,
            summary=e.summary,
            app_name=e.app_name,
            screenshot_url=f"/v1/memory/screenshot/{e.id}" if e.screenshot_path else None,
            meta=e.meta,
        )
        for e in events
    ]


@router.get("/screenshot/{event_id}")
async def screenshot(request: Request, event_id: int):
    mem = _get_memory(request)
    event = mem.timeline.get(event_id)
    if event is None or event.screenshot_path is None:
        raise HTTPException(404, "Screenshot not found")

    path = Path(event.screenshot_path)
    if not path.exists():
        raise HTTPException(404, "Screenshot file missing from disk")

    return FileResponse(path, media_type="image/jpeg")


@router.get("/stm/context")
async def stm_context(request: Request) -> ContextPackResponse:
    mem = _get_memory(request)
    pack = mem.context_packer.build()
    return ContextPackResponse(**pack)


@router.get("/stm/activity")
async def stm_activity(request: Request) -> ActivityResponse:
    mem = _get_memory(request)
    return ActivityResponse(
        activity=mem.classifier.current_activity,
        last_classified_at=mem.classifier.last_classified_at,
    )


@router.post("/timeline/ingest")
async def timeline_ingest(request: Request, body: IngestRequest) -> dict:
    mem = _get_memory(request)
    event_id = await mem.capture.ingest_event(
        event_type=body.event_type,
        summary=body.summary,
        source=body.source,
        app_name=body.app_name,
        raw_text=body.raw_text,
        meta=body.meta,
    )
    return {"event_id": event_id, "status": "ingested"}


@router.delete("/stm/flush")
async def stm_flush(request: Request) -> dict:
    """Debug: manually purge expired STM events."""
    mem = _get_memory(request)
    deleted = mem.stm.purge_expired()
    return {"deleted": deleted}


@router.get("/timeline/buckets")
async def timeline_buckets(
    request: Request,
    start_ts: float = Query(...),
    end_ts: float = Query(...),
    bucket_seconds: int = Query(86400),
    event_types: str | None = Query(None),
) -> list[BucketResponse]:
    """Aggregated event counts in time buckets for zoom-out views."""
    mem = _get_memory(request)
    types_list = event_types.split(",") if event_types else None
    raw = mem.timeline.buckets(start_ts, end_ts, bucket_seconds, types_list)
    return [BucketResponse(**b) for b in raw]


@router.post("/timeline/photo/upload")
async def photo_upload(request: Request, file: UploadFile = File(...)) -> dict:
    """Upload a photo into the timeline. VLM summarizes and embeds it."""
    mem = _get_memory(request)
    content = await file.read()
    if not content:
        raise HTTPException(400, "Empty file")

    # Save to screenshots dir
    now = time.time()
    from datetime import datetime
    dt = datetime.fromtimestamp(now)
    photo_dir = Path(mem.settings.storage_path) / "screenshots" / dt.strftime("%Y/%m/%d")
    photo_dir.mkdir(parents=True, exist_ok=True)
    photo_path = photo_dir / f"{int(now)}_photo.jpg"
    photo_path.write_bytes(content)

    # Summarize via VLM
    summary = ""
    try:
        summary = await mem._summarizer.summarize(content)
    except Exception:
        logger.warning("Photo VLM summary failed, storing without caption")

    # Insert into timeline with screenshot path
    event_id = mem.timeline.insert(
        event_type="photo",
        summary=summary,
        source="upload",
        screenshot_path=str(photo_path),
        meta={"original_filename": file.filename or ""},
        ts=now,
    )

    # Embed asynchronously
    try:
        text = summary or (file.filename or "uploaded photo")
        embedding = await mem.embedder.embed(text)
        chroma_id = str(event_id)
        mem.vectors.upsert(
            event_id=event_id, embedding=embedding,
            document=text, ts=now, event_type="photo",
            app_name="", has_screenshot=True,
        )
        mem.timeline.update_chroma_id(event_id, chroma_id)
    except Exception:
        logger.warning("Photo embedding failed for event %d", event_id)
    return {"event_id": event_id, "summary": summary, "status": "uploaded"}


@router.post("/timeline/tag")
async def timeline_tag(request: Request, body: TagRequest) -> dict:
    """Batch-tag timeline events. Updates meta_json with tags."""
    mem = _get_memory(request)
    updated = mem.timeline.batch_update_tags(body.event_ids, body.tags)
    return {"updated": updated, "tags": body.tags}


# ── Phone Import ─────────────────────────────────────────────


@router.get("/import/detect")
async def import_detect_devices(request: Request) -> dict:
    """Detect connected phones with DCIM folders."""
    mem = _get_memory(request)
    devices = await asyncio.get_event_loop().run_in_executor(
        None, mem.importer.detect_devices
    )
    return {
        "devices": [
            {"name": d.name, "device_id": d.device_id, "photos": d.total_photos}
            for d in devices
        ]
    }


@router.post("/import/start")
async def import_start(
    request: Request,
    device_index: int = Query(0),
) -> dict:
    """Start importing photos from a connected phone.

    Phase 1 (fast): Copy photos, extract EXIF dates, insert into timeline.
    Phase 2 (background): VLM classifies photos newest-first.
    """
    mem = _get_memory(request)
    progress = await mem.importer.start_import(device_index)

    # Auto-start VLM worker after import begins
    if mem.settings.vlm_auto_start:
        # Schedule VLM worker to start after a delay (let Phase 1 get ahead)
        async def _start_vlm_delayed():
            await asyncio.sleep(10)
            mem.vlm_worker.start()
        asyncio.create_task(_start_vlm_delayed())

    return progress.to_dict()


@router.get("/import/progress")
async def import_progress(request: Request) -> dict:
    """Get current import progress (Phase 1 + Phase 2)."""
    mem = _get_memory(request)
    return {
        "import": mem.importer.progress.to_dict(),
        "vlm_gpu": mem.vlm_worker.progress.to_dict(),
        "vlm_cpu": mem.vlm_worker_cpu.progress.to_dict(),
    }


@router.post("/import/vlm/start")
async def vlm_worker_start(request: Request) -> dict:
    """Start VLM classification workers. Pauses screenshot capture to avoid GPU contention."""
    mem = _get_memory(request)
    # Pause screenshot capture so VLM worker gets full GPU
    await mem.capture.stop()
    mem.vlm_worker.start()
    workers = ["gpu"]
    # CPU worker opt-in (only useful with a second Ollama instance)
    if request.query_params.get("cpu") == "true":
        mem.vlm_worker_cpu.start()
        workers.append("cpu")
    return {
        "status": "started",
        "pending": mem.vlm_worker.progress.total_pending,
        "workers": workers,
        "capture_paused": True,
    }


@router.post("/import/vlm/stop")
async def vlm_worker_stop(request: Request) -> dict:
    """Stop VLM workers and resume screenshot capture."""
    mem = _get_memory(request)
    mem.vlm_worker.stop()
    mem.vlm_worker_cpu.stop()
    # Resume screenshot capture
    if mem.capture._controller is not None:
        await mem.capture.start()
    return {
        "status": "stopped",
        "gpu_processed": mem.vlm_worker.progress.processed,
        "cpu_processed": mem.vlm_worker_cpu.progress.processed,
    }
