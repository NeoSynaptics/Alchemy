"""AlchemyMemory — two-layer persistent memory + AI-native search UI.

Public surface:
  MemorySystem   — facade that owns all sub-components
  TimelineStore  — append-only long-term event log
  STMStore       — short-term cache with TTL

Usage in server.py:
    from alchemy.memory import MemorySystem
    memory = MemorySystem(ollama=ollama, orchestrator=orchestrator,
                          controller=desktop_ctrl, settings=settings.memory)
    await memory.start()
    app.state.memory_system = memory
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from alchemy.memory.cache.apu_signal import APUSignal
from alchemy.memory.cache.classifier import ActivityClassifier
from alchemy.memory.cache.context import ContextPacker
from alchemy.memory.cache.store import STMStore
from alchemy.memory.ingest.photo_importer import PhotoImporter
from alchemy.memory.ingest.vlm_worker import VLMWorker
from alchemy.memory.timeline.capture import ScreenshotCapture
from alchemy.memory.timeline.embedder import EmbeddingClient
from alchemy.memory.timeline.search import TimelineSearcher
from alchemy.memory.timeline.store import TimelineStore
from alchemy.memory.timeline.summarizer import ScreenshotSummarizer
from alchemy.memory.timeline.vectordb import VectorStore

if TYPE_CHECKING:
    from config.settings import MemorySettings

logger = logging.getLogger(__name__)

__all__ = ["MemorySystem", "TimelineStore", "STMStore"]


class MemorySystem:
    """Facade that owns all AlchemyMemory sub-components.

    Injected into app.state.memory_system by server.py.
    """

    def __init__(
        self,
        ollama,
        orchestrator,
        controller,          # DesktopController — may be None (graceful)
        settings: MemorySettings,
    ) -> None:
        storage = Path(settings.storage_path)

        # Timeline (LTM)
        self._timeline = TimelineStore(storage / settings.ltm_db)
        # sqlite-vec uses the same DB file as timeline
        self._vectors = VectorStore(str(storage / settings.ltm_db))
        self._embedder = EmbeddingClient(ollama, settings.embedder_model)
        self._summarizer = ScreenshotSummarizer(ollama, settings.summarizer_model)
        self._searcher = TimelineSearcher(self._timeline, self._vectors, self._embedder)

        # Cache (STM)
        self._stm = STMStore(storage / settings.stm_db, settings.stm_purge_interval_seconds)
        self._context_packer = ContextPacker(self._stm)
        self._classifier_enabled = settings.classifier_enabled
        self._classifier = (
            ActivityClassifier(ollama, self._stm, settings.classifier_model)
            if self._classifier_enabled else None
        )
        self._apu_signal = APUSignal(orchestrator)

        # Capture loop
        self._capture = ScreenshotCapture(
            controller=controller,
            summarizer=self._summarizer,
            embedder=self._embedder,
            store=self._timeline,
            vector_store=self._vectors,
            stm_store=self._stm,
            storage_path=storage,
            screenshot_quality=settings.screenshot_quality,
            interval_active=settings.screenshot_interval_active,
            interval_idle=settings.screenshot_interval_idle,
            idle_threshold=settings.idle_threshold_seconds,
        )

        # Phone import
        self._importer = PhotoImporter(
            timeline=self._timeline,
            storage_path=storage,
            screenshot_quality=settings.screenshot_quality,
        )
        # GPU worker — newest-first, no delay
        self._vlm_worker = VLMWorker(
            timeline=self._timeline,
            vectors=self._vectors,
            summarizer=self._summarizer,
            embedder=self._embedder,
            batch_size=settings.vlm_worker_batch_size,
            delay_between=0.0,
            order="DESC",
            use_cpu=False,
            worker_name="gpu",
        )
        # CPU worker — oldest-first, processes from the other end
        self._vlm_worker_cpu = VLMWorker(
            timeline=self._timeline,
            vectors=self._vectors,
            summarizer=self._summarizer,
            embedder=self._embedder,
            batch_size=settings.vlm_worker_batch_size,
            delay_between=0.0,
            order="ASC",
            use_cpu=True,
            worker_name="cpu",
        )

        self._settings = settings

    async def start(self) -> None:
        """Initialize storage and start all background tasks."""
        # Init storage
        self._timeline.init()
        self._stm.init()
        self._vectors.init()

        # Rebuild FTS5 index (catches any rows inserted before triggers existed)
        try:
            self._timeline.rebuild_fts()
        except Exception:
            logger.debug("FTS rebuild skipped (may be first run)")

        # Start background tasks
        self._stm.start_purge_loop()
        if self._classifier:
            self._classifier.on_activity_change(self._context_packer.set_activity)
            self._classifier.start()
            self._apu_signal.start(self._classifier)
        else:
            logger.info("ActivityClassifier disabled (memory.classifier_enabled=False)")

        if self._capture._controller is not None:
            await self._capture.start()
        else:
            logger.warning("MemorySystem: no DesktopController — screenshot capture disabled")

        logger.info("AlchemyMemory started (storage=%s)", self._settings.storage_path)

    async def stop(self) -> None:
        """Graceful shutdown."""
        self._vlm_worker.stop()
        self._vlm_worker_cpu.stop()
        await self._capture.stop()
        if self._classifier:
            self._classifier.stop()
            self._apu_signal.stop()
        self._stm.stop_purge_loop()
        logger.info("AlchemyMemory stopped")

    # --- Public accessors for API layer ---

    @property
    def timeline(self) -> TimelineStore:
        return self._timeline

    @property
    def vectors(self) -> VectorStore:
        return self._vectors

    @property
    def stm(self) -> STMStore:
        return self._stm

    @property
    def searcher(self) -> TimelineSearcher:
        return self._searcher

    @property
    def context_packer(self) -> ContextPacker:
        return self._context_packer

    @property
    def classifier(self) -> ActivityClassifier | None:
        return self._classifier

    @property
    def capture(self) -> ScreenshotCapture:
        return self._capture

    @property
    def importer(self) -> PhotoImporter:
        return self._importer

    @property
    def vlm_worker(self) -> VLMWorker:
        return self._vlm_worker

    @property
    def vlm_worker_cpu(self) -> VLMWorker:
        return self._vlm_worker_cpu

    @property
    def embedder(self) -> EmbeddingClient:
        return self._embedder

    @property
    def settings(self) -> MemorySettings:
        return self._settings
