"""TimelineSearcher — unified search over the long-term timeline.

Combines semantic vector search (ChromaDB) with optional time-range
filtering in SQLite. Joins both result sets and deduplicates by event_id.
"""

from __future__ import annotations

import logging
from typing import Any

from alchemy.memory.timeline.embedder import EmbeddingClient
from alchemy.memory.timeline.store import TimelineEvent, TimelineStore
from alchemy.memory.timeline.vectordb import VectorStore

logger = logging.getLogger(__name__)


class TimelineSearcher:
    """Search the long-term timeline by semantic similarity and/or time."""

    def __init__(
        self,
        store: TimelineStore,
        vector_store: VectorStore,
        embedder: EmbeddingClient,
    ) -> None:
        self._store = store
        self._vectors = vector_store
        self._embedder = embedder

    async def search(
        self,
        query: str | None = None,
        start_ts: float | None = None,
        end_ts: float | None = None,
        event_types: list[str] | None = None,
        app_names: list[str] | None = None,
        limit: int = 20,
        semantic: bool = True,
    ) -> list[dict[str, Any]]:
        """Search the timeline.

        If query is given and semantic=True: embed query → vector search.
        If time range given: also run SQLite time-range query.
        Results are merged and deduped by event_id, sorted by ts DESC.
        """
        event_ids: list[int] = []
        semantic_scores: dict[int, float] = {}

        # Semantic search
        if query and semantic and self._vectors.count() > 0:
            try:
                embedding = await self._embedder.embed(query)
                hits = self._vectors.search(
                    query_embedding=embedding,
                    n_results=limit,
                    start_ts=start_ts,
                    end_ts=end_ts,
                    event_types=event_types,
                )
                for hit in hits:
                    eid = hit["event_id"]
                    event_ids.append(eid)
                    # distance is cosine distance (lower = better) → convert to score
                    semantic_scores[eid] = 1.0 - hit["distance"]
            except Exception:
                logger.warning("Semantic search failed", exc_info=True)

        # Time-range search (always run if range given)
        if start_ts is not None or end_ts is not None:
            s = start_ts or 0.0
            e = end_ts or float("inf")
            time_events = self._store.query_time_range(
                start_ts=s,
                end_ts=e,
                event_types=event_types,
                limit=limit,
            )
            for ev in time_events:
                if ev.id not in semantic_scores:
                    event_ids.append(ev.id)

        # If no filters at all, just return recent
        if not event_ids and not query:
            events = self._store.recent(
                limit=limit,
                event_types=event_types,
                app_names=app_names,
            )
            return [self._event_to_dict(ev, 0.0) for ev in events]

        # Fetch full event objects for all collected IDs
        seen: set[int] = set()
        results: list[dict[str, Any]] = []
        for eid in event_ids:
            if eid in seen:
                continue
            seen.add(eid)
            ev = self._store.get(eid)
            if ev is None:
                continue
            if app_names and ev.app_name not in app_names:
                continue
            results.append(self._event_to_dict(ev, semantic_scores.get(eid, 0.0)))

        # Sort: semantic hits first (by score), then by recency
        results.sort(key=lambda r: (-r["score"], -r["ts"]))
        return results[:limit]

    @staticmethod
    def _event_to_dict(ev: TimelineEvent, score: float) -> dict[str, Any]:
        return {
            "id": ev.id,
            "ts": ev.ts,
            "event_type": ev.event_type,
            "source": ev.source,
            "summary": ev.summary,
            "app_name": ev.app_name,
            "screenshot_path": ev.screenshot_path,
            "score": score,
            "meta": ev.meta,
        }
