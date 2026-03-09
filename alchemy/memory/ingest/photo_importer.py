"""PhotoImporter — orchestrates the two-phase phone photo import pipeline.

Phase 1 (fast, no LLM):
    Scan device → Copy to staging → Extract EXIF → Insert timeline at date_taken

Phase 2 (async, background):
    VLM worker picks unclassified photos newest-first → summarize → embed

This module handles Phase 1. Phase 2 is handled by VLMWorker.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from alchemy.memory.ingest.exif_parser import extract_metadata
from alchemy.memory.ingest.phone_scanner import PhoneScanner, PhotoFile

if TYPE_CHECKING:
    from alchemy.memory.timeline.store import TimelineStore

logger = logging.getLogger(__name__)


@dataclass
class ImportProgress:
    """Live progress of an import job."""
    job_id: str = ""
    phase: str = "idle"           # idle | scanning | copying | inserting | done | error
    device_name: str = ""
    total_photos: int = 0
    copied: int = 0
    inserted: int = 0
    skipped: int = 0              # Already imported
    errors: int = 0
    started_at: float = 0.0
    finished_at: float = 0.0
    error_message: str = ""

    @property
    def is_running(self) -> bool:
        return self.phase in ("scanning", "copying", "inserting")

    @property
    def elapsed(self) -> float:
        end = self.finished_at or time.time()
        return end - self.started_at if self.started_at else 0.0

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "phase": self.phase,
            "device_name": self.device_name,
            "total_photos": self.total_photos,
            "copied": self.copied,
            "inserted": self.inserted,
            "skipped": self.skipped,
            "errors": self.errors,
            "elapsed_seconds": round(self.elapsed, 1),
        }


class PhotoImporter:
    """Orchestrates Phase 1: scan → copy → EXIF → insert timeline."""

    def __init__(
        self,
        timeline: TimelineStore,
        storage_path: Path,
        screenshot_quality: int = 70,
    ) -> None:
        self._timeline = timeline
        self._storage_path = storage_path
        self._staging_dir = storage_path / "import_staging"
        self._photos_dir = storage_path / "photos"
        self._scanner = PhoneScanner(self._staging_dir)
        self._quality = screenshot_quality
        self._progress = ImportProgress()
        self._import_task: asyncio.Task | None = None
        self._wia_dates: dict[str, float] = {}  # filename → WIA timestamp

    @property
    def progress(self) -> ImportProgress:
        return self._progress

    def detect_devices(self):
        """Synchronous device detection."""
        return self._scanner.detect_devices()

    async def start_import(self, device_index: int = 0) -> ImportProgress:
        """Start importing photos from a detected device.

        Non-blocking — runs in background via asyncio.to_thread.
        """
        if self._progress.is_running:
            return self._progress

        self._progress = ImportProgress(
            job_id=f"import_{int(time.time())}",
            started_at=time.time(),
        )

        self._import_task = asyncio.create_task(
            self._run_import(device_index)
        )
        return self._progress

    async def _run_import(self, device_index: int) -> None:
        """Full import pipeline (runs as async task)."""
        try:
            # Phase 1a: Detect device
            self._progress.phase = "scanning"
            devices = await asyncio.to_thread(self._scanner.detect_devices)
            if not devices:
                self._progress.phase = "error"
                self._progress.error_message = "No phone detected. Connect iPhone via USB."
                self._progress.finished_at = time.time()
                return

            if device_index >= len(devices):
                device_index = 0
            device = devices[device_index]
            self._progress.device_name = device.name

            # Phase 1b: List photos via WIA (1-based index)
            wia_index = device_index + 1
            photos = await asyncio.to_thread(
                self._scanner.list_photos, wia_index
            )
            self._progress.total_photos = len(photos)

            if not photos:
                self._progress.phase = "done"
                self._progress.finished_at = time.time()
                return

            # Store WIA timestamps for date_taken (more reliable than EXIF)
            self._wia_dates = {p.filename: p.date_taken for p in photos if p.date_taken}

            # Phase 1c: Copy to staging via WIA
            self._progress.phase = "copying"
            await self._copy_photos(wia_index, photos)

            # Phase 1d: EXIF parse + insert into timeline
            self._progress.phase = "inserting"
            await self._insert_photos()

            self._progress.phase = "done"
            self._progress.finished_at = time.time()
            logger.info(
                "Import complete: %d inserted, %d skipped, %d errors (%.1fs)",
                self._progress.inserted, self._progress.skipped,
                self._progress.errors, self._progress.elapsed,
            )

        except Exception as e:
            logger.error("Import failed: %s", e, exc_info=True)
            self._progress.phase = "error"
            self._progress.error_message = str(e)
            self._progress.finished_at = time.time()

    async def _copy_photos(self, wia_index: int, photos: list[PhotoFile]) -> None:
        """Copy photos from device to staging directory via WIA transfer."""
        def _transfer():
            self._scanner.bulk_transfer(
                photos, self._staging_dir,
                device_index=wia_index,
                on_progress=lambda done, total: setattr(
                    self._progress, 'copied', done
                ),
            )
        await asyncio.to_thread(_transfer)
        self._progress.copied = len(list(self._staging_dir.glob("*")))

    async def _insert_photos(self) -> None:
        """Parse EXIF for all staged photos and insert into timeline."""
        image_exts = {".jpg", ".jpeg", ".png", ".heic", ".heif"}
        staged_files = sorted(
            [f for f in self._staging_dir.iterdir()
             if f.is_file() and f.suffix.lower() in image_exts],
            key=lambda f: f.stat().st_mtime,
        )

        for i, photo_path in enumerate(staged_files):
            try:
                # Check if already imported (by filename in meta)
                if self._is_already_imported(photo_path.name):
                    self._progress.skipped += 1
                    continue

                # Extract EXIF
                meta = await asyncio.to_thread(extract_metadata, photo_path)

                # Determine timestamp: WIA date > EXIF date > file mtime
                # WIA timestamp is stored per photo during listing
                wia_ts = self._wia_dates.get(photo_path.name, 0.0)
                ts = wia_ts or meta.date_taken or photo_path.stat().st_mtime

                # Move to permanent storage organized by date
                dt = datetime.fromtimestamp(ts)
                dest_dir = self._photos_dir / dt.strftime("%Y/%m/%d")
                dest_dir.mkdir(parents=True, exist_ok=True)
                dest = dest_dir / photo_path.name

                if not dest.exists():
                    shutil.move(str(photo_path), str(dest))
                else:
                    # Same name exists — add timestamp suffix
                    stem = photo_path.stem
                    suffix = photo_path.suffix
                    dest = dest_dir / f"{stem}_{int(ts)}{suffix}"
                    shutil.move(str(photo_path), str(dest))

                # Build meta dict
                event_meta = meta.to_meta_dict()
                event_meta["original_filename"] = photo_path.name
                event_meta["import_source"] = "phone"
                event_meta["vlm_status"] = "pending"  # Phase 2 will update this

                # Insert into timeline — summary empty until VLM processes it
                self._timeline.insert(
                    event_type="photo",
                    summary="",  # Will be filled by VLM worker
                    source="phone_import",
                    screenshot_path=str(dest),
                    meta=event_meta,
                    ts=ts,
                )
                self._progress.inserted += 1

            except Exception:
                logger.warning("Failed to import %s", photo_path.name, exc_info=True)
                self._progress.errors += 1

    def _is_already_imported(self, filename: str) -> bool:
        """Check if a photo with this filename was already imported."""
        import sqlite3
        try:
            with sqlite3.connect(self._timeline._db_path) as conn:
                row = conn.execute(
                    "SELECT 1 FROM timeline_events WHERE event_type = 'photo' "
                    "AND meta_json LIKE ? LIMIT 1",
                    (f'%"{filename}"%',),
                ).fetchone()
                return row is not None
        except Exception:
            return False
