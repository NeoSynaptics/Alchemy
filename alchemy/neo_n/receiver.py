"""NEO-N file receiver — saves uploaded files + JSON sidecars to staging folder.

Responsibilities:
  - Accept multipart file uploads from paired devices
  - Save file + metadata sidecar to inbox folder
  - Move processed files to processed/ subfolder
  - Enforce file size limits and blocked extensions
  - Rate limit per device

Does NOT do classification, embedding, or AI. That's BaratzaMemory's job.
"""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

BLOCKED_EXTENSIONS = {".exe", ".bat", ".sh", ".ps1", ".cmd", ".com", ".scr", ".msi", ".vbs", ".wsf"}
DEFAULT_INBOX = Path.home() / "neosy_inbox" / "neo_n"
DEFAULT_MAX_FILE_SIZE_MB = 500
DEFAULT_RATE_LIMIT_PER_HOUR = 100


@dataclass
class UploadResult:
    """Result of a file upload attempt."""
    success: bool
    filename: str = ""
    size_bytes: int = 0
    error: str = ""


@dataclass
class DeviceRateTracker:
    """Track upload timestamps per device for rate limiting."""
    uploads: list[float] = field(default_factory=list)

    def check_and_record(self, limit: int) -> bool:
        """Return True if under rate limit, and record the upload."""
        now = time.monotonic()
        cutoff = now - 3600  # 1 hour window
        self.uploads = [t for t in self.uploads if t > cutoff]
        if len(self.uploads) >= limit:
            return False
        self.uploads.append(now)
        return True


class FileReceiver:
    """Receives files from paired devices and saves them to the staging folder.

    Usage:
        receiver = FileReceiver(inbox_path=Path("~/neosy_inbox/neo_n"))
        result = await receiver.save(
            file_data=b"...",
            original_filename="IMG_4521.jpg",
            content_type="image/jpeg",
            device_id="d_a1b2c3",
            device_name="iPhone 15",
            metadata={"tags": ["vacation"]},
        )
    """

    def __init__(
        self,
        inbox_path: Path = DEFAULT_INBOX,
        max_file_size_mb: int = DEFAULT_MAX_FILE_SIZE_MB,
        rate_limit_per_hour: int = DEFAULT_RATE_LIMIT_PER_HOUR,
    ) -> None:
        self._inbox = Path(inbox_path).expanduser()
        self._processed = self._inbox / "processed"
        self._max_bytes = max_file_size_mb * 1024 * 1024
        self._rate_limit = rate_limit_per_hour
        self._rate_trackers: dict[str, DeviceRateTracker] = defaultdict(DeviceRateTracker)

        # Ensure directories exist
        self._inbox.mkdir(parents=True, exist_ok=True)
        self._processed.mkdir(parents=True, exist_ok=True)

    async def save(
        self,
        file_data: bytes,
        original_filename: str,
        content_type: str,
        device_id: str,
        device_name: str = "",
        metadata: dict | None = None,
    ) -> UploadResult:
        """Save an uploaded file + JSON sidecar to the inbox.

        Returns UploadResult with success status.
        """
        # Validate extension
        ext = Path(original_filename).suffix.lower()
        if ext in BLOCKED_EXTENSIONS:
            logger.warning("Blocked upload: %s (extension %s)", original_filename, ext)
            return UploadResult(success=False, error=f"File type '{ext}' is not allowed")

        # Validate size
        if len(file_data) > self._max_bytes:
            size_mb = len(file_data) / (1024 * 1024)
            return UploadResult(
                success=False,
                error=f"File too large: {size_mb:.1f}MB (max {self._max_bytes // (1024 * 1024)}MB)",
            )

        # Rate limit
        if not self._rate_trackers[device_id].check_and_record(self._rate_limit):
            return UploadResult(
                success=False,
                error=f"Rate limit exceeded: {self._rate_limit} uploads per hour",
            )

        # Build filenames
        now = datetime.now(timezone.utc)
        timestamp = now.strftime("%Y-%m-%d_%H%M%S")
        safe_device = device_name.replace(" ", "_").lower()[:20] or device_id[:10]
        safe_name = Path(original_filename).stem[:50]
        stored_name = f"{timestamp}_{safe_device}_{safe_name}{ext}"
        sidecar_name = f"{timestamp}_{safe_device}_{safe_name}.json"

        file_path = self._inbox / stored_name
        sidecar_path = self._inbox / sidecar_name

        # Write file
        file_path.write_bytes(file_data)

        # Write sidecar
        sidecar = {
            "source_device": device_name,
            "device_id": device_id,
            "original_filename": original_filename,
            "content_type": content_type,
            "size_bytes": len(file_data),
            "uploaded_at": now.isoformat(),
            "stored_as": stored_name,
        }
        if metadata:
            sidecar["user_tags"] = metadata.get("tags", [])
            sidecar["user_notes"] = metadata.get("notes", "")
            sidecar["source_app"] = metadata.get("source_app", "")

        sidecar_path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")

        logger.info("NEO-N received: %s (%d bytes) from %s", stored_name, len(file_data), device_name)

        return UploadResult(
            success=True,
            filename=stored_name,
            size_bytes=len(file_data),
        )

    def mark_processed(self, filename: str) -> bool:
        """Move a file + its sidecar to the processed/ subfolder."""
        src = self._inbox / filename
        if not src.exists():
            return False

        dst = self._processed / filename
        src.rename(dst)

        # Also move sidecar if exists
        sidecar_name = src.stem + ".json"
        sidecar_src = self._inbox / sidecar_name
        if sidecar_src.exists():
            sidecar_src.rename(self._processed / sidecar_name)

        return True

    def list_pending(self) -> list[dict]:
        """List files in inbox that haven't been processed yet."""
        results = []
        for f in sorted(self._inbox.iterdir()):
            if f.is_file() and f.suffix != ".json" and f.name != ".gitkeep":
                sidecar_path = self._inbox / (f.stem + ".json")
                sidecar = {}
                if sidecar_path.exists():
                    try:
                        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
                    except Exception:
                        pass
                results.append({
                    "filename": f.name,
                    "size_bytes": f.stat().st_size,
                    "uploaded_at": sidecar.get("uploaded_at", ""),
                    "source_device": sidecar.get("source_device", ""),
                    "original_filename": sidecar.get("original_filename", f.name),
                })
        return results

    def stats(self) -> dict:
        """Return inbox statistics."""
        pending = [f for f in self._inbox.iterdir()
                   if f.is_file() and f.suffix != ".json" and f.name != ".gitkeep"]
        processed = [f for f in self._processed.iterdir()
                     if f.is_file() and f.suffix != ".json"]
        total_pending_bytes = sum(f.stat().st_size for f in pending)
        return {
            "inbox_path": str(self._inbox),
            "pending_count": len(pending),
            "pending_bytes": total_pending_bytes,
            "processed_count": len(processed),
        }

    @property
    def inbox_path(self) -> Path:
        return self._inbox
