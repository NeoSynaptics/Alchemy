"""DuckDuckGo search provider — async, no API key required."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A single search result from DuckDuckGo."""

    url: str
    title: str
    snippet: str


class SearchProvider:
    """Async DuckDuckGo search wrapper.

    Usage:
        provider = SearchProvider(max_results_per_query=5)
        results = await provider.search_many(["query1", "query2", "query3"])
    """

    def __init__(self, max_results_per_query: int = 5):
        self._max_results = max_results_per_query

    async def search(self, query: str) -> list[SearchResult]:
        """Run a single search query. Returns list of SearchResult."""
        try:
            raw = await asyncio.to_thread(self._search_sync, query)
            return [
                SearchResult(
                    url=r.get("href", ""),
                    title=r.get("title", ""),
                    snippet=r.get("body", ""),
                )
                for r in raw
                if r.get("href")
            ]
        except Exception as e:
            logger.warning("Search failed for '%s': %s", query, e)
            return []

    def _search_sync(self, query: str) -> list[dict]:
        """Synchronous DDG search — called via to_thread."""
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS

        with DDGS() as ddgs:
            return list(ddgs.text(query, region="wt-wt", safesearch="moderate", max_results=self._max_results))

    async def search_many(self, queries: list[str]) -> list[SearchResult]:
        """Run multiple queries in parallel, dedup by URL.

        Args:
            queries: List of search query strings.

        Returns:
            Deduplicated list of SearchResult objects.
        """
        tasks = [self.search(q) for q in queries]
        results_nested = await asyncio.gather(*tasks, return_exceptions=True)

        seen_urls: set[str] = set()
        deduped: list[SearchResult] = []

        for batch in results_nested:
            if isinstance(batch, Exception):
                logger.warning("Search batch failed: %s", batch)
                continue
            for r in batch:
                if r.url not in seen_urls:
                    seen_urls.add(r.url)
                    deduped.append(r)

        logger.info("Search: %d queries -> %d unique URLs", len(queries), len(deduped))
        return deduped
