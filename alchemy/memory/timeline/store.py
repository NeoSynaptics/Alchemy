"""TimelineStore — SQLite persistence for long-term memory events.

Events are NEVER deleted. Time is the primary axis. Every captured
moment (screenshot, voice, action, search) becomes a row here.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class TimelineEvent:
    id: int
    ts: float
    event_type: str          # 'screenshot' | 'voice' | 'action' | 'search' | 'app_switch'
    source: str              # 'desktop' | 'voice' | 'click' | 'research'
    summary: str
    raw_text: str
    app_name: str
    screenshot_path: str | None
    chroma_id: str | None
    meta: dict = field(default_factory=dict)


class TimelineStore:
    """Long-term memory: append-only SQLite event log."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path

    def init(self) -> None:
        """Create tables and indexes. Safe to call multiple times."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS timeline_events (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts              REAL    NOT NULL,
                    event_type      TEXT    NOT NULL,
                    source          TEXT    NOT NULL DEFAULT '',
                    summary         TEXT    NOT NULL DEFAULT '',
                    raw_text        TEXT    NOT NULL DEFAULT '',
                    app_name        TEXT    NOT NULL DEFAULT '',
                    screenshot_path TEXT    DEFAULT NULL,
                    chroma_id       TEXT    DEFAULT NULL,
                    meta_json       TEXT    NOT NULL DEFAULT '{}'
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_timeline_ts "
                "ON timeline_events(ts DESC)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_timeline_type "
                "ON timeline_events(event_type, ts DESC)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_timeline_app "
                "ON timeline_events(app_name, ts DESC)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_timeline_chroma "
                "ON timeline_events(chroma_id)"
            )
            conn.commit()
        logger.info("TimelineStore ready: %s", self._db_path)

    def insert(
        self,
        event_type: str,
        summary: str,
        source: str = "",
        raw_text: str = "",
        app_name: str = "",
        screenshot_path: str | None = None,
        chroma_id: str | None = None,
        meta: dict[str, Any] | None = None,
        ts: float | None = None,
    ) -> int:
        """Insert a new event. Returns the new row ID."""
        now = ts or time.time()
        meta_json = json.dumps(meta or {})
        with sqlite3.connect(self._db_path) as conn:
            cursor = conn.execute(
                """INSERT INTO timeline_events
                   (ts, event_type, source, summary, raw_text, app_name,
                    screenshot_path, chroma_id, meta_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (now, event_type, source, summary, raw_text, app_name,
                 screenshot_path, chroma_id, meta_json),
            )
            conn.commit()
            return cursor.lastrowid  # type: ignore[return-value]

    def update_chroma_id(self, event_id: int, chroma_id: str) -> None:
        """Set the ChromaDB ID after embedding is stored."""
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                "UPDATE timeline_events SET chroma_id = ? WHERE id = ?",
                (chroma_id, event_id),
            )
            conn.commit()

    def get(self, event_id: int) -> TimelineEvent | None:
        """Fetch a single event by ID."""
        with sqlite3.connect(self._db_path) as conn:
            row = conn.execute(
                "SELECT id, ts, event_type, source, summary, raw_text, "
                "app_name, screenshot_path, chroma_id, meta_json "
                "FROM timeline_events WHERE id = ?",
                (event_id,),
            ).fetchone()
        return self._row_to_event(row) if row else None

    def recent(
        self,
        limit: int = 20,
        event_types: list[str] | None = None,
        app_names: list[str] | None = None,
    ) -> list[TimelineEvent]:
        """Fetch the most recent events, optionally filtered."""
        clauses: list[str] = []
        params: list[Any] = []

        if event_types:
            placeholders = ",".join("?" * len(event_types))
            clauses.append(f"event_type IN ({placeholders})")
            params.extend(event_types)

        if app_names:
            placeholders = ",".join("?" * len(app_names))
            clauses.append(f"app_name IN ({placeholders})")
            params.extend(app_names)

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)

        with sqlite3.connect(self._db_path) as conn:
            rows = conn.execute(
                f"SELECT id, ts, event_type, source, summary, raw_text, "
                f"app_name, screenshot_path, chroma_id, meta_json "
                f"FROM timeline_events {where} ORDER BY ts DESC LIMIT ?",
                params,
            ).fetchall()
        return [self._row_to_event(r) for r in rows]

    def query_time_range(
        self,
        start_ts: float,
        end_ts: float,
        event_types: list[str] | None = None,
        limit: int = 50,
    ) -> list[TimelineEvent]:
        """Fetch events within a time window."""
        clauses = ["ts >= ?", "ts <= ?"]
        params: list[Any] = [start_ts, end_ts]

        if event_types:
            placeholders = ",".join("?" * len(event_types))
            clauses.append(f"event_type IN ({placeholders})")
            params.extend(event_types)

        params.append(limit)
        where = "WHERE " + " AND ".join(clauses)

        with sqlite3.connect(self._db_path) as conn:
            rows = conn.execute(
                f"SELECT id, ts, event_type, source, summary, raw_text, "
                f"app_name, screenshot_path, chroma_id, meta_json "
                f"FROM timeline_events {where} ORDER BY ts DESC LIMIT ?",
                params,
            ).fetchall()
        return [self._row_to_event(r) for r in rows]

    def count(self) -> int:
        """Total number of events in the timeline."""
        with sqlite3.connect(self._db_path) as conn:
            row = conn.execute("SELECT COUNT(*) FROM timeline_events").fetchone()
        return row[0] if row else 0

    def stats(self) -> dict[str, Any]:
        """Quick storage stats for health endpoint."""
        with sqlite3.connect(self._db_path) as conn:
            total = conn.execute("SELECT COUNT(*) FROM timeline_events").fetchone()[0]
            oldest = conn.execute(
                "SELECT MIN(ts) FROM timeline_events"
            ).fetchone()[0]
            by_type = conn.execute(
                "SELECT event_type, COUNT(*) FROM timeline_events GROUP BY event_type"
            ).fetchall()
        return {
            "total_events": total,
            "oldest_ts": oldest,
            "by_type": {row[0]: row[1] for row in by_type},
        }

    def buckets(
        self,
        start_ts: float,
        end_ts: float,
        bucket_seconds: int = 86400,
        event_types: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Aggregate event counts into time buckets for zoom-out views."""
        clauses = ["ts >= ?", "ts <= ?"]
        params: list[Any] = [start_ts, end_ts]

        if event_types:
            placeholders = ",".join("?" * len(event_types))
            clauses.append(f"event_type IN ({placeholders})")
            params.extend(event_types)

        where = "WHERE " + " AND ".join(clauses)

        with sqlite3.connect(self._db_path) as conn:
            rows = conn.execute(
                f"SELECT CAST(ts / ? AS INT) * ? AS bucket_ts, "
                f"event_type, COUNT(*) "
                f"FROM timeline_events {where} "
                f"GROUP BY bucket_ts, event_type "
                f"ORDER BY bucket_ts",
                [bucket_seconds, bucket_seconds] + params,
            ).fetchall()

        # Collapse rows into {bucket_ts: {count, types}} dicts
        buckets_map: dict[float, dict[str, Any]] = {}
        for bucket_ts, event_type, count in rows:
            if bucket_ts not in buckets_map:
                buckets_map[bucket_ts] = {"bucket_ts": float(bucket_ts), "count": 0, "types": {}}
            buckets_map[bucket_ts]["count"] += count
            buckets_map[bucket_ts]["types"][event_type] = count

        return list(buckets_map.values())

    def batch_update_tags(
        self, event_ids: list[int], tags: list[str]
    ) -> int:
        """Add tags to meta_json for a batch of events. Returns count updated."""
        updated = 0
        with sqlite3.connect(self._db_path) as conn:
            for eid in event_ids:
                row = conn.execute(
                    "SELECT meta_json FROM timeline_events WHERE id = ?", (eid,)
                ).fetchone()
                if not row:
                    continue
                meta = json.loads(row[0])
                existing = meta.get("tags", [])
                merged = list(set(existing + tags))
                meta["tags"] = merged
                conn.execute(
                    "UPDATE timeline_events SET meta_json = ? WHERE id = ?",
                    (json.dumps(meta), eid),
                )
                updated += 1
            conn.commit()
        return updated

    @staticmethod
    def _row_to_event(row: tuple) -> TimelineEvent:
        return TimelineEvent(
            id=row[0],
            ts=row[1],
            event_type=row[2],
            source=row[3],
            summary=row[4],
            raw_text=row[5],
            app_name=row[6],
            screenshot_path=row[7],
            chroma_id=row[8],
            meta=json.loads(row[9]),
        )
