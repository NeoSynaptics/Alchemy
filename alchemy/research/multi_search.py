"""Multi-engine search with Reciprocal Rank Fusion (RRF).

Fires Google CSE, Bing, and DuckDuckGo in parallel. Merges results
by URL using RRF scoring — results found by multiple engines rank higher.

Engines degrade gracefully: missing API keys = engine skipped.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from dataclasses import dataclass, field

import httpx

from alchemy.research.searcher import SearchProvider, SearchResult

logger = logging.getLogger(__name__)


@dataclass
class RankedResult:
    """A search result with RRF fusion metadata."""

    url: str
    title: str
    snippet: str
    rrf_score: float = 0.0
    engines: list[str] = field(default_factory=list)
    engine_ranks: dict[str, int] = field(default_factory=dict)

    @property
    def domain(self) -> str:
        m = re.match(r"https?://([^/]+)", self.url)
        return m.group(1) if m else self.url

    @property
    def favicon_url(self) -> str:
        return f"https://www.google.com/s2/favicons?domain={self.domain}&sz=32"


# ── Individual Engine Providers ───────────────────────────────────────────────


class GoogleCSEProvider:
    """Google Custom Search Engine — 100 free queries/day."""

    URL = "https://www.googleapis.com/customsearch/v1"

    def __init__(self, api_key: str, cse_id: str):
        self.api_key = api_key
        self.cse_id = cse_id

    async def search(self, query: str, max_results: int = 10) -> list[SearchResult]:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    self.URL,
                    params={
                        "key": self.api_key,
                        "cx": self.cse_id,
                        "q": query,
                        "num": min(max_results, 10),
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                return [
                    SearchResult(
                        url=item.get("link", ""),
                        title=item.get("title", ""),
                        snippet=item.get("snippet", ""),
                    )
                    for item in data.get("items", [])
                    if item.get("link")
                ]
        except Exception as e:
            logger.warning("Google CSE search failed: %s", e)
            return []


class BingSearchProvider:
    """Bing Web Search API v7 — 1000 free queries/month."""

    URL = "https://api.bing.microsoft.com/v7.0/search"

    def __init__(self, api_key: str):
        self.api_key = api_key

    async def search(self, query: str, max_results: int = 10) -> list[SearchResult]:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    self.URL,
                    params={"q": query, "count": min(max_results, 50)},
                    headers={"Ocp-Apim-Subscription-Key": self.api_key},
                )
                resp.raise_for_status()
                data = resp.json()
                return [
                    SearchResult(
                        url=item.get("url", ""),
                        title=item.get("name", ""),
                        snippet=item.get("snippet", ""),
                    )
                    for item in data.get("webPages", {}).get("value", [])
                    if item.get("url")
                ]
        except Exception as e:
            logger.warning("Bing search failed: %s", e)
            return []


class DDGProvider:
    """DuckDuckGo — free, no API key. Wraps existing SearchProvider."""

    def __init__(self):
        self._provider = SearchProvider(max_results_per_query=10)

    async def search(self, query: str, max_results: int = 10) -> list[SearchResult]:
        self._provider._max_results = max_results
        return await self._provider.search(query)


# ── Multi-Engine Searcher with RRF Fusion ─────────────────────────────────────


class MultiSearcher:
    """Searches multiple engines in parallel, fuses with RRF.

    Usage:
        searcher = MultiSearcher()   # auto-detects available API keys
        results = await searcher.search("query")
    """

    def __init__(
        self,
        google_api_key: str | None = None,
        google_cse_id: str | None = None,
        bing_api_key: str | None = None,
        rrf_k: int = 60,
        enable_google: bool = True,
        enable_bing: bool = True,
        enable_ddg: bool = True,
    ):
        self.rrf_k = rrf_k
        self._engines: dict[str, object] = {}

        # Google CSE
        gkey = google_api_key or os.environ.get("GOOGLE_CSE_API_KEY")
        gcse = google_cse_id or os.environ.get("GOOGLE_CSE_ID")
        if enable_google and gkey and gcse:
            self._engines["google"] = GoogleCSEProvider(gkey, gcse)

        # Bing
        bkey = bing_api_key or os.environ.get("BING_SEARCH_KEY")
        if enable_bing and bkey:
            self._engines["bing"] = BingSearchProvider(bkey)

        # DuckDuckGo (always available)
        if enable_ddg:
            self._engines["duckduckgo"] = DDGProvider()

    @property
    def available_engines(self) -> list[str]:
        return list(self._engines.keys())

    async def search(self, query: str, max_results: int = 10) -> list[RankedResult]:
        """Search all engines in parallel, return RRF-fused results."""
        if not self._engines:
            logger.error("No search engines available")
            return []

        # Fire all engines in parallel
        tasks = {
            name: asyncio.create_task(engine.search(query, max_results))
            for name, engine in self._engines.items()
        }

        engine_results: dict[str, list[SearchResult]] = {}
        for name, task in tasks.items():
            try:
                engine_results[name] = await task
                logger.info(
                    "Engine %s returned %d results",
                    name, len(engine_results[name]),
                )
            except Exception as e:
                logger.warning("Engine %s failed: %s", name, e)
                engine_results[name] = []

        return self._fuse_rrf(engine_results, max_results)

    def _fuse_rrf(
        self,
        engine_results: dict[str, list[SearchResult]],
        max_results: int,
    ) -> list[RankedResult]:
        """Reciprocal Rank Fusion: score = sum(1 / (k + rank)) across engines."""
        k = self.rrf_k
        url_data: dict[str, RankedResult] = {}

        for engine_name, results in engine_results.items():
            for rank, r in enumerate(results, start=1):
                url = _normalize_url(r.url)
                if url not in url_data:
                    url_data[url] = RankedResult(
                        url=r.url,
                        title=r.title,
                        snippet=r.snippet,
                    )
                entry = url_data[url]
                entry.rrf_score += 1.0 / (k + rank)
                entry.engines.append(engine_name)
                entry.engine_ranks[engine_name] = rank

                # Prefer longer title/snippet (Google often has better ones)
                if len(r.title) > len(entry.title):
                    entry.title = r.title
                if len(r.snippet) > len(entry.snippet):
                    entry.snippet = r.snippet

        ranked = sorted(url_data.values(), key=lambda r: r.rrf_score, reverse=True)
        return ranked[:max_results]


def _normalize_url(url: str) -> str:
    """Normalize URL for dedup: strip trailing slash, force https."""
    url = url.rstrip("/")
    url = re.sub(r"^http://", "https://", url)
    return url.lower()
