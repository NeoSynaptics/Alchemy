"""HTTP client that queries NEO-RX for relevant knowledge.

Graceful degradation: if NEO-RX is down, returns empty list.
AlchemyVoice works fine without knowledge — it's an enhancement layer.
"""

from __future__ import annotations

import logging

import httpx

logger = logging.getLogger(__name__)


class KnowledgeRetriever:
    """Queries NEO-RX /v1/knowledge/search for relevant context."""

    def __init__(
        self,
        neorx_host: str = "http://localhost:8110",
        timeout: float = 5.0,
        max_docs: int = 3,
    ):
        self._host = neorx_host.rstrip("/")
        self._max_docs = max_docs
        self._client = httpx.AsyncClient(timeout=timeout)

    async def retrieve(self, query: str, limit: int | None = None) -> list[str]:
        """Query NEO-RX knowledge search, return doc contents.

        Returns empty list if NEO-RX is down (graceful degradation).
        """
        limit = limit or self._max_docs
        try:
            resp = await self._client.get(
                f"{self._host}/v1/knowledge/search",
                params={"q": query, "limit": limit},
            )
            resp.raise_for_status()
            results = resp.json()
            contents = [r["content"] for r in results if r.get("content")]
            if contents:
                logger.debug("Retrieved %d knowledge docs for: %s", len(contents), query[:50])
            return contents
        except httpx.ConnectError:
            # NEO-RX not running — silent degradation
            return []
        except Exception:
            logger.debug("Knowledge retrieval failed (non-critical)")
            return []

    async def close(self) -> None:
        await self._client.aclose()
