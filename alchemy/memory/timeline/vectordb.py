"""VectorStore — sqlite-vec based semantic search for the timeline.

Replaces ChromaDB. Embeddings live in the same SQLite database as the
timeline data. Used as a fallback for fuzzy queries that can't be
resolved by tag lookup or FTS5. Operates on pre-filtered subsets, so
brute-force search is fast (<10ms on hundreds of items).
"""

from __future__ import annotations

import logging
import sqlite3
import struct
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _serialize_f32(vec: list[float]) -> bytes:
    """Pack a float list into little-endian f32 bytes for sqlite-vec."""
    return struct.pack(f"<{len(vec)}f", *vec)


class VectorStore:
    """SQLite-vec vector store — embeddings in the timeline .db file."""

    def __init__(self, db_path: str, embedding_dim: int = 768) -> None:
        self._db_path = db_path
        self._dim = embedding_dim

    def init(self) -> None:
        """Create the vec0 virtual table if it doesn't exist."""
        import sqlite_vec

        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self._db_path)
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)

        conn.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_embeddings
            USING vec0(
                event_id INTEGER PRIMARY KEY,
                embedding float[{self._dim}]
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS vec_meta (
                event_id     INTEGER PRIMARY KEY,
                ts           REAL NOT NULL DEFAULT 0,
                event_type   TEXT NOT NULL DEFAULT '',
                app_name     TEXT NOT NULL DEFAULT '',
                document     TEXT NOT NULL DEFAULT ''
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_vm_type ON vec_meta(event_type)"
        )
        conn.commit()
        conn.close()

        count = self.count()
        logger.info("VectorStore (sqlite-vec) ready: %s (%d vectors)", self._db_path, count)

    def _connect(self) -> sqlite3.Connection:
        """Get a connection with sqlite-vec loaded."""
        import sqlite_vec

        conn = sqlite3.connect(self._db_path)
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        return conn

    def upsert(
        self,
        event_id: int,
        embedding: list[float],
        document: str,
        ts: float,
        event_type: str,
        app_name: str,
        has_screenshot: bool = False,
    ) -> None:
        """Add or update a vector + metadata."""
        conn = self._connect()
        try:
            conn.execute("DELETE FROM vec_embeddings WHERE event_id = ?", (event_id,))
            conn.execute(
                "INSERT INTO vec_embeddings(event_id, embedding) VALUES (?, ?)",
                (event_id, _serialize_f32(embedding)),
            )
            conn.execute(
                "INSERT OR REPLACE INTO vec_meta(event_id, ts, event_type, app_name, document) "
                "VALUES (?, ?, ?, ?, ?)",
                (event_id, ts, event_type, app_name, document),
            )
            conn.commit()
        finally:
            conn.close()

    def search(
        self,
        query_embedding: list[float],
        n_results: int = 10,
        start_ts: float | None = None,
        end_ts: float | None = None,
        event_types: list[str] | None = None,
        event_ids: list[int] | None = None,
    ) -> list[dict[str, Any]]:
        """Semantic search via sqlite-vec KNN.

        If event_ids is provided, only searches within those IDs (pre-filtered subset).
        """
        conn = self._connect()
        try:
            if event_ids is not None and len(event_ids) == 0:
                return []

            query_bytes = _serialize_f32(query_embedding)

            # sqlite-vec KNN query
            rows = conn.execute(
                "SELECT v.event_id, v.distance, m.document, m.ts, m.event_type, m.app_name "
                "FROM vec_embeddings v "
                "JOIN vec_meta m ON m.event_id = v.event_id "
                "WHERE v.embedding MATCH ? AND k = ? "
                "ORDER BY v.distance",
                (query_bytes, min(n_results * 3, max(n_results, self.count()))),
            ).fetchall()

            results = []
            for row in rows:
                eid, dist, doc, ts_val, etype, app = row
                if event_ids is not None and eid not in set(event_ids):
                    continue
                if start_ts is not None and ts_val < start_ts:
                    continue
                if end_ts is not None and ts_val > end_ts:
                    continue
                if event_types and etype not in event_types:
                    continue
                results.append({
                    "event_id": eid,
                    "distance": dist,
                    "document": doc,
                    "metadata": {
                        "event_id": eid,
                        "ts": ts_val,
                        "event_type": etype,
                        "app_name": app,
                    },
                })
                if len(results) >= n_results:
                    break
            return results
        finally:
            conn.close()

    def count(self) -> int:
        """Total number of vectors stored."""
        try:
            conn = sqlite3.connect(self._db_path)
            row = conn.execute("SELECT COUNT(*) FROM vec_meta").fetchone()
            conn.close()
            return row[0] if row else 0
        except Exception:
            return 0
