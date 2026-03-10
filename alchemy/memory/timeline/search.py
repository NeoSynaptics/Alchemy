"""TimelineSearcher — cascading search over the long-term timeline.

Search cascade:
  1. Tag lookup (structured, <1ms) — handles "dog", "beach", "selfie"
  2. FTS5 full-text (keyword, <5ms) — handles multi-word queries
  3. sqlite-vec semantic (vector, <50ms) — handles fuzzy/poetic queries

Each layer narrows the search space for the next. Vector search only
runs on pre-filtered subsets, keeping RAM near zero at any scale.
"""

from __future__ import annotations

import logging
from typing import Any

from alchemy.memory.timeline.embedder import EmbeddingClient
from alchemy.memory.timeline.store import TimelineEvent, TimelineStore
from alchemy.memory.timeline.vectordb import VectorStore

logger = logging.getLogger(__name__)

# Common query words → tag column mappings
_TAG_KEYWORDS: dict[str, str] = {
    # subjects
    "dog": "subject", "dogs": "subject", "cat": "subject", "cats": "subject",
    "person": "subject", "people": "subject", "selfie": "subject",
    "food": "subject", "car": "subject", "animal": "subject",
    "flower": "subject", "bird": "subject", "tree": "subject",
    "landscape": "subject", "building": "subject", "pet": "subject",
    # scenes
    "beach": "scene", "forest": "scene", "park": "scene",
    "indoor": "scene", "outdoor": "scene", "street": "scene",
    "kitchen": "scene", "garden": "scene", "mountain": "scene",
    "city": "scene", "home": "scene", "office": "scene",
    # activities
    "playing": "activity", "eating": "activity", "walking": "activity",
    "running": "activity", "cooking": "activity", "sleeping": "activity",
    "swimming": "activity", "hiking": "activity",
    # time
    "sunset": "time_of_day", "sunrise": "time_of_day",
    "night": "time_of_day", "daytime": "time_of_day",
}


def _parse_query_to_tags(query: str) -> dict[str, str]:
    """Map query words to tag column filters."""
    tags: dict[str, str] = {}
    for word in query.lower().split():
        col = _TAG_KEYWORDS.get(word)
        if col and col not in tags:
            tags[col] = word
    return tags


class TimelineSearcher:
    """Cascading search: tags → FTS5 → sqlite-vec."""

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
        """Search the timeline using cascading strategy."""

        # No query → just return recent
        if not query:
            if start_ts is not None or end_ts is not None:
                events = self._store.query_time_range(
                    start_ts=start_ts or 0.0,
                    end_ts=end_ts or float("inf"),
                    event_types=event_types,
                    limit=limit,
                )
            else:
                events = self._store.recent(
                    limit=limit,
                    event_types=event_types,
                    app_names=app_names,
                )
            return [self._event_to_dict(ev, 0.0) for ev in events]

        # ── Layer 1: Tag lookup ──
        tag_filters = _parse_query_to_tags(query)
        tag_results: list[TimelineEvent] = []
        if tag_filters:
            tag_results = self._store.search_by_tags(
                subject=tag_filters.get("subject"),
                scene=tag_filters.get("scene"),
                activity=tag_filters.get("activity"),
                limit=limit,
            )
            if len(tag_results) >= 3:
                logger.debug("Tag search for '%s' returned %d results", query, len(tag_results))
                results = [self._event_to_dict(ev, 1.0) for ev in tag_results]
                return self._apply_filters(results, app_names, limit)

        # ── Layer 2: FTS5 full-text ──
        try:
            fts_results = self._store.fts_search(query, limit=limit)
            if fts_results:
                logger.debug("FTS search for '%s' returned %d results", query, len(fts_results))
                # Merge with any tag results
                seen = {ev.id for ev in tag_results}
                combined = [self._event_to_dict(ev, 1.0) for ev in tag_results]
                for ev in fts_results:
                    if ev.id not in seen:
                        combined.append(self._event_to_dict(ev, 0.8))
                        seen.add(ev.id)
                if combined:
                    return self._apply_filters(combined, app_names, limit)
        except Exception:
            logger.debug("FTS search failed for '%s'", query, exc_info=True)

        # ── Layer 3: sqlite-vec semantic search (fallback) ──
        if semantic and self._vectors.count() > 0:
            try:
                embedding = await self._embedder.embed(query)
                hits = self._vectors.search(
                    query_embedding=embedding,
                    n_results=limit,
                    start_ts=start_ts,
                    end_ts=end_ts,
                    event_types=event_types,
                )
                seen = {ev.id for ev in tag_results}
                results = [self._event_to_dict(ev, 1.0) for ev in tag_results]
                for hit in hits:
                    eid = hit["event_id"]
                    if eid in seen:
                        continue
                    seen.add(eid)
                    ev = self._store.get(eid)
                    if ev is None:
                        continue
                    score = 1.0 - hit["distance"]
                    results.append(self._event_to_dict(ev, score))
                results.sort(key=lambda r: (-r["score"], -r["ts"]))
                return self._apply_filters(results, app_names, limit)
            except Exception:
                logger.warning("Semantic search failed", exc_info=True)

        # Nothing found — return tag results if any, else empty
        return [self._event_to_dict(ev, 0.5) for ev in tag_results][:limit]

    def _apply_filters(
        self, results: list[dict], app_names: list[str] | None, limit: int
    ) -> list[dict]:
        if app_names:
            results = [r for r in results if r["app_name"] in app_names]
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
