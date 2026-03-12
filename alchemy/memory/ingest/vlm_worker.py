"""VLMWorker — background worker that classifies untagged photos.

Processes photos newest-first using Qwen2.5-VL 7B. Updates the
timeline event summary and embeds into sqlite-vec for semantic search.

Supports dual-worker mode: GPU worker (newest-first) + CPU worker
(oldest-first) process the queue from both ends simultaneously.

Runs as an asyncio task. Can be started/stopped independently.
"""

from __future__ import annotations

import asyncio
import io
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

# Max dimension for VLM input — photos are resized to fit within this box.
# Original files on disk are NOT modified. Only the bytes sent to VLM are smaller.
_VLM_MAX_DIM = 640

# Photo-specific VLM prompt — outputs JSON with summary + structured tags
_PHOTO_PROMPT = (
    "You are a photo cataloger. Respond ONLY with valid JSON, no markdown.\n"
    '{"summary": "one detailed sentence describing the photo",'
    '"tags": {"subject": "main subject (dog, person, food, landscape, car, building, etc)",'
    '"scene": "setting (forest, beach, indoor, street, park, kitchen, office, etc)",'
    '"activity": "what is happening (playing, eating, walking, posing, resting, etc)",'
    '"time_of_day": "daytime, night, sunset, sunrise, or overcast",'
    '"people_count": 0,'
    '"mood": "playful, calm, dramatic, cozy, bright, dark, etc"}}\n'
    "Be specific. Use lowercase single words for tags. For subject use the most specific "
    "noun (golden retriever -> dog, tabby -> cat, pizza -> food)."
)


def _resize_for_vlm(image_bytes: bytes, max_dim: int = _VLM_MAX_DIM) -> bytes:
    """Resize image so longest side <= max_dim. Returns JPEG bytes.

    Only the bytes sent to VLM are resized — originals on disk are untouched.
    """
    from PIL import Image

    img = Image.open(io.BytesIO(image_bytes))
    w, h = img.size

    # Skip resize if already small enough
    if max(w, h) <= max_dim:
        return image_bytes

    # Calculate new dimensions preserving aspect ratio
    if w > h:
        new_w = max_dim
        new_h = int(h * max_dim / w)
    else:
        new_h = max_dim
        new_w = int(w * max_dim / h)

    img = img.resize((new_w, new_h), Image.LANCZOS)

    # Convert to RGB if needed (e.g. RGBA PNGs)
    if img.mode != "RGB":
        img = img.convert("RGB")

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


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
        delay_between: float = 0.0,
        order: str = "DESC",
        use_cpu: bool = False,
        worker_name: str = "gpu",
    ) -> None:
        self._timeline = timeline
        self._vectors = vectors
        self._summarizer = summarizer
        self._embedder = embedder
        self._batch_size = batch_size
        self._delay = delay_between
        self._order = order          # DESC = newest-first, ASC = oldest-first
        self._use_cpu = use_cpu      # Force CPU-only inference
        self._worker_name = worker_name
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
        logger.info("VLM worker [%s] started (cpu=%s, order=%s)",
                     self._worker_name, self._use_cpu, self._order)

    def stop(self) -> None:
        """Stop the background worker."""
        self._running = False
        if self._task:
            self._task.cancel()
            self._task = None
        self._progress.status = "idle"
        logger.info("VLM worker [%s] stopped", self._worker_name)

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
                # Fetch batch of unclassified photos
                pending = self._get_pending_photos(self._batch_size)
                self._progress.total_pending = self._count_pending()

                if not pending:
                    self._progress.status = "done"
                    logger.info(
                        "VLM worker [%s] done: %d processed, %d errors",
                        self._worker_name,
                        self._progress.processed, self._progress.errors,
                    )
                    break

                for event_id, screenshot_path, meta_json in pending:
                    if not self._running:
                        break

                    self._progress.current_file = Path(screenshot_path).name

                    try:
                        await asyncio.wait_for(
                            self._process_photo(event_id, screenshot_path, meta_json),
                            timeout=15.0,
                        )
                        self._progress.processed += 1
                    except asyncio.TimeoutError:
                        logger.warning(
                            "VLM [%s] timed out for event %d (%s) — skipping",
                            self._worker_name, event_id, screenshot_path,
                        )
                        self._progress.errors += 1
                        self._mark_vlm_status(event_id, meta_json, "timeout")
                    except Exception:
                        logger.warning(
                            "VLM [%s] failed for event %d (%s)",
                            self._worker_name,
                            event_id, screenshot_path, exc_info=True,
                        )
                        self._progress.errors += 1
                        self._mark_vlm_status(event_id, meta_json, "failed")

                    # Update rate
                    elapsed = time.time() - start_time
                    if elapsed > 0:
                        self._progress.rate = (self._progress.processed / elapsed) * 60

                    # Minimal yield to event loop (no throttle delay)
                    if self._delay > 0:
                        await asyncio.sleep(self._delay)
                    else:
                        await asyncio.sleep(0)

        except asyncio.CancelledError:
            logger.info("VLM worker [%s] cancelled", self._worker_name)
        except Exception:
            logger.error("VLM worker [%s] crashed", self._worker_name, exc_info=True)
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

        # Resize for VLM (originals on disk untouched)
        try:
            image_bytes = await asyncio.to_thread(_resize_for_vlm, image_bytes)
        except Exception:
            logger.debug("Resize failed for %s, using original", screenshot_path)

        # VLM summarize — returns JSON with summary + tags
        vlm_result = await self._summarize_photo(image_bytes)

        if not vlm_result:
            self._mark_vlm_status(event_id, meta_json, "vlm_empty")
            return

        # Parse JSON response from VLM
        summary, tags = self._parse_vlm_response(vlm_result)

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

        # Store structured tags for token-first search
        if tags:
            try:
                self._timeline.insert_tags(event_id, tags, ts=event_ts)
            except Exception:
                logger.warning("Tag insert failed for event %d", event_id, exc_info=True)

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

        logger.debug("VLM [%s] classified event %d: %s",
                      self._worker_name, event_id, summary[:80])

    @staticmethod
    def _parse_vlm_response(raw: str) -> tuple[str, dict]:
        """Parse VLM JSON response into (summary, tags). Falls back gracefully."""
        # Try JSON parse first
        try:
            # Strip markdown fences if present
            text = raw.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1] if "\n" in text else text[3:]
                if text.endswith("```"):
                    text = text[:-3]
                text = text.strip()
                if text.startswith("json"):
                    text = text[4:].strip()

            data = json.loads(text)
            summary = data.get("summary", "").strip()
            tags = data.get("tags", {})
            # Normalize tag values to lowercase
            for k, v in tags.items():
                if isinstance(v, str):
                    tags[k] = v.lower()
            return summary, tags
        except (json.JSONDecodeError, AttributeError):
            # VLM didn't return valid JSON — use raw text as summary, no tags
            return raw.strip(), {}

    async def _summarize_photo(self, image_bytes: bytes) -> str:
        """Run VLM on a photo via the shared inference client (gateway or OllamaClient)."""
        try:
            options: dict = {"temperature": 0.0, "num_predict": 150}
            if self._use_cpu:
                options["num_gpu"] = 0

            result = await self._summarizer._ollama.chat(
                model=self._summarizer._model,
                messages=[
                    {"role": "system", "content": _PHOTO_PROMPT},
                    {"role": "user", "content": "Describe this photo."},
                ],
                images=[image_bytes],
                options=options,
            )
            return result.get("message", {}).get("content", "").strip()
        except Exception:
            logger.warning("Photo VLM [%s] summarization failed",
                           self._worker_name, exc_info=True)
            return ""

    def _get_pending_photos(self, limit: int) -> list[tuple]:
        """Fetch unclassified photos from timeline."""
        with sqlite3.connect(self._timeline._db_path) as conn:
            return conn.execute(
                f"""SELECT id, screenshot_path, meta_json
                   FROM timeline_events
                   WHERE event_type = 'photo'
                     AND (summary = '' OR summary IS NULL)
                     AND screenshot_path IS NOT NULL
                     AND meta_json NOT LIKE '%"vlm_status": "failed"%'
                     AND meta_json NOT LIKE '%"vlm_status": "file_missing"%'
                     AND meta_json NOT LIKE '%"vlm_status": "timeout"%'
                   ORDER BY ts {self._order}
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
                     AND meta_json NOT LIKE '%"vlm_status": "file_missing"%'
                     AND meta_json NOT LIKE '%"vlm_status": "timeout"%'""",
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
