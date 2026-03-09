"""VLMWorker — background worker that classifies untagged photos.

Processes photos newest-first using Qwen2.5-VL 7B. Updates the
timeline event summary and embeds into ChromaDB for semantic search.

Runs as an asyncio task. Can be started/stopped independently.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from alchemy.memory.timeline.embedder import EmbeddingClient
    from alchemy.memory.timeline.store import TimelineStore
    from alchemy.memory.timeline.summarizer import ScreenshotSummarizer
    from alchemy.memory.timeline.vectordb import VectorStore

logger = logging.getLogger(__name__)

# Photo-specific VLM prompt (different from desktop screenshot prompt)
_PHOTO_PROMPT = (
    "You are a photo cataloger. Describe this photo in ONE detailed sentence. "
    "Include: main subject(s), setting/location type, activity, time of day if visible, "
    "notable objects. Be specific about people count, animals, food, landmarks. "
    "No preamble, no punctuation at the end"
)


@dataclass
class VLMProgress:
    """Progress of the VLM classification worker."""
    status: str = "idle"      # idle | running | paused | done
    total_pending: int = 0
    processed: int = 0
    errors: int = 0
    current_file: str = ""
    rate: float = 0.0         # photos per minute

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "total_pending": self.total_pending,
            "processed": self.processed,
            "errors": self.errors,
            "current_file": self.current_file,
            "rate_per_min": round(self.rate, 1),
        }


class VLMWorker:
    """Background worker that VLM-classifies pending photos newest-first."""

    def __init__(
        self,
        timeline: TimelineStore,
        vectors: VectorStore,
        summarizer: ScreenshotSummarizer,
        embedder: EmbeddingClient,
        batch_size: int = 50,
        delay_between: float = 0.5,
    ) -> None:
        self._timeline = timeline
        self._vectors = vectors
        self._summarizer = summarizer
        self._embedder = embedder
        self._batch_size = batch_size
        self._delay = delay_between
        self._task: asyncio.Task | None = None
        self._running = False
        self._progress = VLMProgress()

    @property
    def progress(self) -> VLMProgress:
        return self._progress

    def start(self) -> None:
        """Start the background VLM classification loop."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info("VLM worker started")

    def stop(self) -> None:
        """Stop the background worker."""
        self._running = False
        if self._task:
            self._task.cancel()
            self._task = None
        self._progress.status = "idle"
        logger.info("VLM worker stopped")

    def pause(self) -> None:
        self._running = False
        self._progress.status = "paused"

    def resume(self) -> None:
        if self._progress.status == "paused":
            self.start()

    async def _run_loop(self) -> None:
        """Main loop: fetch pending photos, process newest-first."""
        self._progress.status = "running"
        start_time = time.time()

        try:
            while self._running:
                # Fetch batch of unclassified photos (newest first)
                pending = self._get_pending_photos(self._batch_size)
                self._progress.total_pending = self._count_pending()

                if not pending:
                    self._progress.status = "done"
                    logger.info(
                        "VLM worker done: %d processed, %d errors",
                        self._progress.processed, self._progress.errors,
                    )
                    break

                for event_id, screenshot_path, meta_json in pending:
                    if not self._running:
                        break

                    self._progress.current_file = Path(screenshot_path).name

                    try:
                        await self._process_photo(event_id, screenshot_path, meta_json)
                        self._progress.processed += 1
                    except Exception:
                        logger.warning(
                            "VLM failed for event %d (%s)",
                            event_id, screenshot_path, exc_info=True,
                        )
                        self._progress.errors += 1
                        # Mark as failed so we don't retry forever
                        self._mark_vlm_status(event_id, meta_json, "failed")

                    # Update rate
                    elapsed = time.time() - start_time
                    if elapsed > 0:
                        self._progress.rate = (self._progress.processed / elapsed) * 60

                    # Throttle to avoid GPU saturation
                    await asyncio.sleep(self._delay)

        except asyncio.CancelledError:
            logger.info("VLM worker cancelled")
        except Exception:
            logger.error("VLM worker crashed", exc_info=True)
            self._progress.status = "idle"

    async def _process_photo(
        self, event_id: int, screenshot_path: str, meta_json: str
    ) -> None:
        """Summarize a single photo with VLM, embed, and update timeline."""
        photo_path = Path(screenshot_path)
        if not photo_path.exists():
            logger.warning("Photo file missing: %s", screenshot_path)
            self._mark_vlm_status(event_id, meta_json, "file_missing")
            return

        # Read image bytes
        image_bytes = await asyncio.to_thread(photo_path.read_bytes)

        # VLM summarize (uses photo-specific prompt)
        summary = await self._summarize_photo(image_bytes)

        if not summary:
            self._mark_vlm_status(event_id, meta_json, "vlm_empty")
            return

        # Get the actual event timestamp from timeline
        event = self._timeline.get(event_id)
        event_ts = event.ts if event else time.time()

        # Update timeline event with summary
        with sqlite3.connect(self._timeline._db_path) as conn:
            conn.execute(
                "UPDATE timeline_events SET summary = ? WHERE id = ?",
                (summary, event_id),
            )
            conn.commit()

        # Embed and upsert to vector DB
        try:
            # Include GPS/meta context in embedding text for richer search
            meta = json.loads(meta_json) if meta_json else {}
            embed_text = summary
            if meta.get("gps"):
                embed_text += f" [GPS: {meta['gps']['lat']:.4f}, {meta['gps']['lon']:.4f}]"
            if meta.get("camera"):
                embed_text += f" [Camera: {meta['camera']}]"

            embedding = await self._embedder.embed(embed_text)
            self._vectors.upsert(
                event_id=event_id,
                embedding=embedding,
                document=embed_text,
                ts=event_ts,
                event_type="photo",
                app_name="",
                has_screenshot=True,
            )
            # Update chroma_id
            self._timeline.update_chroma_id(event_id, str(event_id))
        except Exception:
            logger.warning("Embedding failed for event %d", event_id, exc_info=True)

        # Mark VLM status as done
        self._mark_vlm_status(event_id, meta_json, "done")

        logger.debug("VLM classified event %d: %s", event_id, summary[:80])

    async def _summarize_photo(self, image_bytes: bytes) -> str:
        """Run VLM on a photo with the photo-specific prompt."""
        try:
            result = await self._summarizer._ollama.chat(
                model=self._summarizer._model,
                messages=[
                    {"role": "system", "content": _PHOTO_PROMPT},
                    {"role": "user", "content": "Describe this photo."},
                ],
                images=[image_bytes],
                options={"num_ctx": 8192, "temperature": 0.0, "num_predict": 200},
            )
            return result.get("message", {}).get("content", "").strip()
        except Exception:
            logger.warning("Photo VLM summarization failed", exc_info=True)
            return ""

    def _get_pending_photos(self, limit: int) -> list[tuple]:
        """Fetch unclassified photos from timeline, newest first."""
        with sqlite3.connect(self._timeline._db_path) as conn:
            return conn.execute(
                """SELECT id, screenshot_path, meta_json
                   FROM timeline_events
                   WHERE event_type = 'photo'
                     AND (summary = '' OR summary IS NULL)
                     AND screenshot_path IS NOT NULL
                     AND meta_json NOT LIKE '%"vlm_status": "failed"%'
                     AND meta_json NOT LIKE '%"vlm_status": "file_missing"%'
                   ORDER BY ts DESC
                   LIMIT ?""",
                (limit,),
            ).fetchall()

    def _count_pending(self) -> int:
        """Count total unclassified photos."""
        with sqlite3.connect(self._timeline._db_path) as conn:
            row = conn.execute(
                """SELECT COUNT(*) FROM timeline_events
                   WHERE event_type = 'photo'
                     AND (summary = '' OR summary IS NULL)
                     AND screenshot_path IS NOT NULL
                     AND meta_json NOT LIKE '%"vlm_status": "failed"%'
                     AND meta_json NOT LIKE '%"vlm_status": "file_missing"%'""",
            ).fetchone()
            return row[0] if row else 0

    def _mark_vlm_status(self, event_id: int, meta_json: str, status: str) -> None:
        """Update vlm_status in the event's meta_json."""
        try:
            meta = json.loads(meta_json) if meta_json else {}
            meta["vlm_status"] = status
            with sqlite3.connect(self._timeline._db_path) as conn:
                conn.execute(
                    "UPDATE timeline_events SET meta_json = ? WHERE id = ?",
                    (json.dumps(meta), event_id),
                )
                conn.commit()
        except Exception:
            pass
