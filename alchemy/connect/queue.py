"""Offline message queue — stores messages for disconnected devices.

When the server wants to push a message but the device is offline,
the message goes into this SQLite-backed queue. On reconnect, the
queue drains in sequence order.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from pathlib import Path

from alchemy.connect.protocol import AlchemyMessage

logger = logging.getLogger(__name__)


class OfflineQueue:
    """SQLite-backed offline message queue per device."""

    def __init__(self, data_dir: str | Path, max_per_device: int = 200) -> None:
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = self._data_dir / "queue.db"
        self._max = max_per_device
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS queue (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    device_id TEXT NOT NULL,
                    message TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_queue_device "
                "ON queue(device_id, id)"
            )
            conn.commit()

    def enqueue(self, device_id: str, msg: AlchemyMessage) -> None:
        """Queue a message for an offline device."""
        now = time.time()
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                "INSERT INTO queue (device_id, message, created_at) VALUES (?, ?, ?)",
                (device_id, json.dumps(msg.to_dict()), now),
            )
            # Trim oldest if over limit
            count = conn.execute(
                "SELECT COUNT(*) FROM queue WHERE device_id = ?",
                (device_id,),
            ).fetchone()[0]
            if count > self._max:
                excess = count - self._max
                conn.execute(
                    "DELETE FROM queue WHERE id IN ("
                    "  SELECT id FROM queue WHERE device_id = ? ORDER BY id ASC LIMIT ?"
                    ")",
                    (device_id, excess),
                )
            conn.commit()

    def drain(self, device_id: str) -> list[AlchemyMessage]:
        """Drain all queued messages for a device (FIFO order). Deletes them."""
        with sqlite3.connect(self._db_path) as conn:
            rows = conn.execute(
                "SELECT id, message FROM queue WHERE device_id = ? ORDER BY id ASC",
                (device_id,),
            ).fetchall()

            if rows:
                ids = [r[0] for r in rows]
                placeholders = ",".join("?" * len(ids))
                conn.execute(
                    f"DELETE FROM queue WHERE id IN ({placeholders})", ids,
                )
                conn.commit()

        messages = []
        for _, msg_json in rows:
            try:
                messages.append(AlchemyMessage.from_dict(json.loads(msg_json)))
            except (json.JSONDecodeError, ValueError):
                logger.warning("Skipping malformed queued message")
        return messages

    def count(self, device_id: str) -> int:
        """Count queued messages for a device."""
        with sqlite3.connect(self._db_path) as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM queue WHERE device_id = ?",
                (device_id,),
            ).fetchone()
        return row[0] if row else 0

    def clear(self, device_id: str) -> int:
        """Clear all queued messages for a device. Returns count deleted."""
        with sqlite3.connect(self._db_path) as conn:
            cursor = conn.execute(
                "DELETE FROM queue WHERE device_id = ?", (device_id,),
            )
            conn.commit()
            return cursor.rowcount
