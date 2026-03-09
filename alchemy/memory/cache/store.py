"""STMStore — short-term memory SQLite store with TTL auto-expiry.

Events are automatically purged after `expires_at`. A background
purge loop runs every N seconds.

Two tables:
  stm_events      — recent activity, TTL-bounded (max 3-4 days)
  stm_preferences — graduated preferences, never expire
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class STMEvent:
    id: int
    ts: float
    event_type: str
    summary: str
    app_name: str
    weight: float
    expires_at: float
    meta: dict = field(default_factory=dict)


class STMStore:
    """Short-term memory: TTL-bounded SQLite event store."""

    def __init__(self, db_path: Path, purge_interval: int = 60) -> None:
        self._db_path = db_path
        self._purge_interval = purge_interval
        self._purge_task: asyncio.Task | None = None

    def init(self) -> None:
        """Create tables and indexes."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS stm_events (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts          REAL    NOT NULL,
                    event_type  TEXT    NOT NULL,
                    summary     TEXT    NOT NULL DEFAULT '',
                    app_name    TEXT    NOT NULL DEFAULT '',
                    weight      REAL    NOT NULL DEFAULT 1.0,
                    expires_at  REAL    NOT NULL,
                    meta_json   TEXT    NOT NULL DEFAULT '{}'
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_stm_expires "
                "ON stm_events(expires_at)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_stm_ts "
                "ON stm_events(ts DESC)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_stm_app "
                "ON stm_events(app_name, ts DESC)"
            )
            conn.execute("""
                CREATE TABLE IF NOT EXISTS stm_preferences (
                    key         TEXT PRIMARY KEY,
                    value       TEXT NOT NULL,
                    confidence  REAL NOT NULL DEFAULT 0.5,
                    updated_at  REAL NOT NULL
                )
            """)
            conn.commit()
        logger.info("STMStore ready: %s", self._db_path)

    def insert(
        self,
        event_type: str,
        summary: str,
        app_name: str = "",
        ttl_seconds: float = 86400 * 4,
        weight: float = 1.0,
        meta: dict[str, Any] | None = None,
        ts: float | None = None,
    ) -> int:
        now = ts or time.time()
        expires_at = now + ttl_seconds
        meta_json = json.dumps(meta or {})
        with sqlite3.connect(self._db_path) as conn:
            cursor = conn.execute(
                """INSERT INTO stm_events
                   (ts, event_type, summary, app_name, weight, expires_at, meta_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (now, event_type, summary, app_name, weight, expires_at, meta_json),
            )
            conn.commit()
            return cursor.lastrowid  # type: ignore[return-value]

    def recent(self, window_minutes: int = 60, limit: int = 20) -> list[STMEvent]:
        """Return events from the last N minutes, most recent first."""
        cutoff = time.time() - (window_minutes * 60)
        now = time.time()
        with sqlite3.connect(self._db_path) as conn:
            rows = conn.execute(
                "SELECT id, ts, event_type, summary, app_name, weight, expires_at, meta_json "
                "FROM stm_events "
                "WHERE ts >= ? AND expires_at > ? "
                "ORDER BY ts DESC LIMIT ?",
                (cutoff, now, limit),
            ).fetchall()
        return [self._row_to_event(r) for r in rows]

    def active_apps(self, last_hours: int = 1) -> list[str]:
        """Return distinct app names seen in the last N hours."""
        cutoff = time.time() - (last_hours * 3600)
        with sqlite3.connect(self._db_path) as conn:
            rows = conn.execute(
                "SELECT DISTINCT app_name FROM stm_events "
                "WHERE ts >= ? AND app_name != '' ORDER BY ts DESC",
                (cutoff,),
            ).fetchall()
        return [r[0] for r in rows]

    def set_preference(self, key: str, value: str, confidence: float = 0.8) -> None:
        now = time.time()
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """INSERT INTO stm_preferences (key, value, confidence, updated_at)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(key) DO UPDATE SET
                     value=excluded.value,
                     confidence=excluded.confidence,
                     updated_at=excluded.updated_at""",
                (key, value, confidence, now),
            )
            conn.commit()

    def get_preferences(self) -> dict[str, str]:
        with sqlite3.connect(self._db_path) as conn:
            rows = conn.execute(
                "SELECT key, value FROM stm_preferences ORDER BY confidence DESC"
            ).fetchall()
        return {r[0]: r[1] for r in rows}

    def purge_expired(self) -> int:
        """Delete expired events. Returns count of deleted rows."""
        now = time.time()
        with sqlite3.connect(self._db_path) as conn:
            cursor = conn.execute(
                "DELETE FROM stm_events WHERE expires_at < ?", (now,)
            )
            conn.commit()
            return cursor.rowcount

    def start_purge_loop(self) -> None:
        """Start background asyncio purge task."""
        self._purge_task = asyncio.create_task(
            self._purge_loop(), name="memory:stm_purge"
        )

    def stop_purge_loop(self) -> None:
        if self._purge_task and not self._purge_task.done():
            self._purge_task.cancel()

    async def _purge_loop(self) -> None:
        while True:
            await asyncio.sleep(self._purge_interval)
            try:
                deleted = self.purge_expired()
                if deleted:
                    logger.debug("STM purged %d expired events", deleted)
            except Exception:
                logger.warning("STM purge failed", exc_info=True)

    def stats(self) -> dict[str, Any]:
        now = time.time()
        with sqlite3.connect(self._db_path) as conn:
            total = conn.execute(
                "SELECT COUNT(*) FROM stm_events WHERE expires_at > ?", (now,)
            ).fetchone()[0]
            prefs = conn.execute(
                "SELECT COUNT(*) FROM stm_preferences"
            ).fetchone()[0]
        return {"active_events": total, "preferences": prefs}

    @staticmethod
    def _row_to_event(row: tuple) -> STMEvent:
        return STMEvent(
            id=row[0],
            ts=row[1],
            event_type=row[2],
            summary=row[3],
            app_name=row[4],
            weight=row[5],
            expires_at=row[6],
            meta=json.loads(row[7]),
        )
