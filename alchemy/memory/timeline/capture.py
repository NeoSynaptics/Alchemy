"""ScreenshotCapture — background asyncio task that periodically captures
the desktop, summarizes with VLM, embeds with nomic, and writes to
TimelineStore + VectorStore.

Two decoupled tasks:
  _capture_loop  — timer-driven, takes screenshots at configurable intervals
  _summarize_loop — queue-driven, VLM + embed + store (slower, runs sequentially)

Backpressure: Queue(maxsize=10) drops frames when VLM is saturated.
Idle detection: win32api GetLastInputInfo() — no extra dependency.
"""

from __future__ import annotations

import asyncio
import ctypes
import ctypes.wintypes
import io
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from alchemy.memory.cache.store import STMStore
    from alchemy.memory.timeline.embedder import EmbeddingClient
    from alchemy.memory.timeline.store import TimelineStore
    from alchemy.memory.timeline.summarizer import ScreenshotSummarizer
    from alchemy.memory.timeline.vectordb import VectorStore

logger = logging.getLogger(__name__)

# Windows LASTINPUTINFO structure
class _LASTINPUTINFO(ctypes.Structure):
    _fields_ = [("cbSize", ctypes.wintypes.UINT), ("dwTime", ctypes.wintypes.DWORD)]


def _get_idle_seconds() -> float:
    """Return seconds since the last user input (keyboard or mouse)."""
    try:
        info = _LASTINPUTINFO()
        info.cbSize = ctypes.sizeof(_LASTINPUTINFO)
        ctypes.windll.user32.GetLastInputInfo(ctypes.byref(info))
        tick_now = ctypes.windll.kernel32.GetTickCount()
        return (tick_now - info.dwTime) / 1000.0
    except Exception:
        return 0.0


def _get_foreground_app() -> str:
    """Return the title of the currently focused window, or empty string."""
    try:
        hwnd = ctypes.windll.user32.GetForegroundWindow()
        length = ctypes.windll.user32.GetWindowTextLengthW(hwnd) + 1
        buf = ctypes.create_unicode_buffer(length)
        ctypes.windll.user32.GetWindowTextW(hwnd, buf, length)
        return buf.value or ""
    except Exception:
        return ""


class ScreenshotCapture:
    """Background screenshot capture + summarize pipeline."""

    def __init__(
        self,
        controller,           # DesktopController (or any obj with async screenshot() -> bytes)
        summarizer: ScreenshotSummarizer,
        embedder: EmbeddingClient,
        store: TimelineStore,
        vector_store: VectorStore,
        stm_store: STMStore,
        storage_path: Path,
        screenshot_quality: int,
        interval_active: int,
        interval_idle: int,
        idle_threshold: int,
    ) -> None:
        self._controller = controller
        self._summarizer = summarizer
        self._embedder = embedder
        self._store = store
        self._vectors = vector_store
        self._stm = stm_store
        self._storage_path = storage_path
        self._screenshot_quality = screenshot_quality
        self._interval_active = interval_active
        self._interval_idle = interval_idle
        self._idle_threshold = idle_threshold

        self._queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=10)
        self._running = False
        self._capture_task: asyncio.Task | None = None
        self._summarize_task: asyncio.Task | None = None

    async def start(self) -> None:
        self._running = True
        self._capture_task = asyncio.create_task(
            self._capture_loop(), name="memory:capture"
        )
        self._summarize_task = asyncio.create_task(
            self._summarize_loop(), name="memory:summarize"
        )
        logger.info("ScreenshotCapture started (active=%ds, idle=%ds)",
                    self._interval_active, self._interval_idle)

    async def stop(self) -> None:
        self._running = False
        for task in (self._capture_task, self._summarize_task):
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        logger.info("ScreenshotCapture stopped")

    def is_running(self) -> bool:
        return self._running and bool(
            self._capture_task and not self._capture_task.done()
        )

    async def _capture_loop(self) -> None:
        """Timer-driven capture. Non-blocking — just enqueues."""
        while self._running:
            idle_secs = await asyncio.to_thread(_get_idle_seconds)
            is_idle = idle_secs >= self._idle_threshold
            interval = self._interval_idle if is_idle else self._interval_active

            await asyncio.sleep(interval)
            if not self._running:
                break

            try:
                img_bytes = await self._controller.screenshot()
                if self._queue.full():
                    logger.debug("Capture queue full — dropping frame (VLM busy)")
                    continue
                await self._queue.put(img_bytes)
            except Exception:
                logger.warning("Screenshot failed", exc_info=True)

    async def _summarize_loop(self) -> None:
        """Queue-driven: VLM caption → embed → store. Runs one at a time."""
        while self._running:
            try:
                img_bytes = await asyncio.wait_for(self._queue.get(), timeout=5.0)
            except asyncio.TimeoutError:
                continue

            try:
                await self._process_screenshot(img_bytes)
            except Exception:
                logger.warning("Screenshot processing failed", exc_info=True)

    async def _process_screenshot(self, img_bytes: bytes) -> None:
        ts = time.time()
        app_name = await asyncio.to_thread(_get_foreground_app)

        # 1. Caption via VLM
        summary = await self._summarizer.summarize(img_bytes)
        if not summary:
            return  # Don't store empty summaries

        # 2. Save JPEG to disk
        screenshot_path = self._save_screenshot(img_bytes, ts)

        # 3. Insert into timeline (no chroma_id yet)
        event_id = self._store.insert(
            event_type="screenshot",
            summary=summary,
            source="desktop",
            app_name=app_name,
            screenshot_path=str(screenshot_path) if screenshot_path else None,
            ts=ts,
        )

        # 4. Embed summary text
        try:
            embedding = await self._embedder.embed(summary)
            self._vectors.upsert(
                event_id=event_id,
                embedding=embedding,
                document=summary,
                ts=ts,
                event_type="screenshot",
                app_name=app_name,
                has_screenshot=screenshot_path is not None,
            )
            self._store.update_chroma_id(event_id, str(event_id))
        except Exception:
            logger.warning("Embedding/vector store failed for event %d", event_id, exc_info=True)

        # 5. Mirror to STM
        try:
            from config.settings import settings
            ttl = settings.memory.cache_ttl_days * 86400
            self._stm.insert(
                event_type="screenshot",
                summary=summary,
                app_name=app_name,
                ttl_seconds=ttl,
            )
        except Exception:
            logger.debug("STM mirror failed (non-critical)", exc_info=True)

        logger.debug("Captured: [%s] %s", app_name, summary[:60])

    def _save_screenshot(self, img_bytes: bytes, ts: float) -> Path | None:
        """Save screenshot to D:/AlchemyMemory/screenshots/YYYY/MM/DD/{event_id}.jpg"""
        try:
            from PIL import Image

            dt = datetime.fromtimestamp(ts)
            date_dir = (
                self._storage_path
                / "screenshots"
                / str(dt.year)
                / f"{dt.month:02d}"
                / f"{dt.day:02d}"
            )
            date_dir.mkdir(parents=True, exist_ok=True)
            filename = date_dir / f"{int(ts * 1000)}.jpg"

            img = Image.open(io.BytesIO(img_bytes))
            img.save(filename, "JPEG", quality=self._screenshot_quality)
            return filename
        except Exception:
            logger.warning("Failed to save screenshot to disk", exc_info=True)
            return None

    async def ingest_event(
        self,
        event_type: str,
        summary: str,
        source: str = "",
        app_name: str = "",
        raw_text: str = "",
        meta: dict | None = None,
    ) -> int:
        """Manual event ingest — used by voice/click modules via API.

        Embeds and stores without a screenshot.
        """
        ts = time.time()
        event_id = self._store.insert(
            event_type=event_type,
            summary=summary,
            source=source,
            raw_text=raw_text,
            app_name=app_name,
            meta=meta,
            ts=ts,
        )

        try:
            embedding = await self._embedder.embed(summary or raw_text)
            self._vectors.upsert(
                event_id=event_id,
                embedding=embedding,
                document=(summary or raw_text)[:2000],
                ts=ts,
                event_type=event_type,
                app_name=app_name,
                has_screenshot=False,
            )
            self._store.update_chroma_id(event_id, str(event_id))
        except Exception:
            logger.warning("Embedding failed for manual ingest event %d", event_id, exc_info=True)

        return event_id
