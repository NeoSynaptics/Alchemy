"""VectorStore — ChromaDB wrapper for semantic search over the timeline.

Each document corresponds to one timeline_events row. The embedding
is generated externally (nomic-embed-text via Ollama) and passed in —
we never use ChromaDB's built-in embedding functions.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class VectorStore:
    """Persistent ChromaDB collection for timeline semantic search."""

    def __init__(self, chroma_path: str, collection_name: str) -> None:
        self._chroma_path = chroma_path
        self._collection_name = collection_name
        self._client = None
        self._collection = None

    def init(self) -> None:
        """Connect to ChromaDB and get/create the collection."""
        import chromadb

        Path(self._chroma_path).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=self._chroma_path)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine", "hnsw:M": 16},
            embedding_function=None,  # embeddings provided externally
        )
        logger.info(
            "VectorStore ready: %s (collection=%s, docs=%d)",
            self._chroma_path,
            self._collection_name,
            self._collection.count(),
        )

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
        """Add or update a document in the collection."""
        if self._collection is None:
            raise RuntimeError("VectorStore not initialized — call init() first")
        self._collection.upsert(
            ids=[str(event_id)],
            embeddings=[embedding],
            documents=[document],
            metadatas=[{
                "event_id": event_id,
                "ts": ts,
                "event_type": event_type,
                "app_name": app_name,
                "has_screenshot": has_screenshot,
            }],
        )

    def search(
        self,
        query_embedding: list[float],
        n_results: int = 10,
        start_ts: float | None = None,
        end_ts: float | None = None,
        event_types: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Semantic search. Returns list of dicts with event_id, distance, document, metadata."""
        if self._collection is None:
            raise RuntimeError("VectorStore not initialized — call init() first")

        where: dict[str, Any] | None = None
        filters: list[dict] = []

        if start_ts is not None and end_ts is not None:
            filters.append({"ts": {"$gte": start_ts, "$lte": end_ts}})
        elif start_ts is not None:
            filters.append({"ts": {"$gte": start_ts}})
        elif end_ts is not None:
            filters.append({"ts": {"$lte": end_ts}})

        if event_types:
            filters.append({"event_type": {"$in": event_types}})

        if len(filters) == 1:
            where = filters[0]
        elif len(filters) > 1:
            where = {"$and": filters}

        kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": min(n_results, max(1, self._collection.count())),
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        results = self._collection.query(**kwargs)

        out: list[dict[str, Any]] = []
        if not results["ids"] or not results["ids"][0]:
            return out

        for i, doc_id in enumerate(results["ids"][0]):
            out.append({
                "event_id": int(doc_id),
                "distance": results["distances"][0][i],  # type: ignore[index]
                "document": results["documents"][0][i],   # type: ignore[index]
                "metadata": results["metadatas"][0][i],   # type: ignore[index]
            })
        return out

    def count(self) -> int:
        if self._collection is None:
            return 0
        return self._collection.count()
