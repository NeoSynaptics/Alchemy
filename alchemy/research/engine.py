"""ResearchEngine — orchestrates the full research pipeline."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum

from alchemy.research.collector import PageCollector
from alchemy.research.searcher import SearchProvider
from alchemy.research.synthesizer import Synthesizer, SynthesisResult

logger = logging.getLogger(__name__)


class PipelineStage(str, Enum):
    PENDING = "pending"
    DECOMPOSING = "decomposing"
    SEARCHING = "searching"
    FETCHING = "fetching"
    SCORING = "scoring"
    SYNTHESIZING = "synthesizing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ResearchProgress:
    """Mutable state tracked during pipeline execution."""

    stage: PipelineStage = PipelineStage.PENDING
    queries_generated: int = 0
    pages_fetched: int = 0
    pages_used: int = 0
    total_ms: float = 0.0
    error: str | None = None
    result: SynthesisResult | None = None


class ResearchEngine:
    """Orchestrates the full AlchemyBrowser pipeline.

    Receives dependencies from the caller (API layer), does NOT
    create its own OllamaClient or BrowserManager instances.

    Args:
        synthesizer: Synthesizer instance for LLM calls.
        searcher: SearchProvider for DuckDuckGo searches.
        collector: PageCollector for page fetching + extraction.
        max_queries: Max sub-queries from decomposition.
        top_k: Top K pages to use for synthesis.
    """

    def __init__(
        self,
        synthesizer: Synthesizer,
        searcher: SearchProvider,
        collector: PageCollector,
        max_queries: int = 10,
        top_k: int = 5,
    ):
        self._synthesizer = synthesizer
        self._searcher = searcher
        self._collector = collector
        self._max_queries = max_queries
        self._top_k = top_k

    async def run_semantic(self, query: str, progress: ResearchProgress) -> ResearchProgress:
        """Run the full semantic research pipeline.

        Args:
            query: User's natural language question.
            progress: Mutable progress object (updated in-place for polling).

        Returns:
            The same progress object, updated with results.
        """
        start = time.monotonic()

        try:
            # 1. Decompose query
            progress.stage = PipelineStage.DECOMPOSING
            sub_queries = await self._synthesizer.decompose_query(query, self._max_queries)
            progress.queries_generated = len(sub_queries)

            if not sub_queries:
                sub_queries = [query]

            # 2. Search
            progress.stage = PipelineStage.SEARCHING
            search_results = await self._searcher.search_many(sub_queries)

            if not search_results:
                progress.stage = PipelineStage.FAILED
                progress.error = "No search results found"
                progress.total_ms = (time.monotonic() - start) * 1000
                return progress

            # 3. Fetch pages
            progress.stage = PipelineStage.FETCHING
            urls = [r.url for r in search_results]
            titles = {r.url: r.title for r in search_results}
            pages = await self._collector.collect(urls, titles)
            progress.pages_fetched = sum(1 for p in pages if p.fetch_ok)

            if not any(p.fetch_ok and p.text for p in pages):
                progress.stage = PipelineStage.FAILED
                progress.error = "All page fetches failed or returned empty content"
                progress.total_ms = (time.monotonic() - start) * 1000
                return progress

            # 4. Score relevance
            progress.stage = PipelineStage.SCORING
            top_pages = self._synthesizer.score_relevance(query, pages, self._top_k)
            progress.pages_used = len(top_pages)

            # 5. Synthesize
            progress.stage = PipelineStage.SYNTHESIZING
            synthesis = await self._synthesizer.synthesize(query, top_pages)

            progress.stage = PipelineStage.COMPLETED
            progress.result = synthesis
            progress.total_ms = (time.monotonic() - start) * 1000

            logger.info(
                "Research complete: %d queries, %d fetched, %d used, %.0fms",
                progress.queries_generated,
                progress.pages_fetched,
                progress.pages_used,
                progress.total_ms,
            )

        except Exception as e:
            logger.error("Research pipeline failed: %s", e, exc_info=True)
            progress.stage = PipelineStage.FAILED
            progress.error = str(e)
            progress.total_ms = (time.monotonic() - start) * 1000

        return progress

    async def run_direct(
        self, query: str, urls: list[str], progress: ResearchProgress
    ) -> ResearchProgress:
        """Run the direct-read pipeline (skip search, user provides URLs).

        Args:
            query: Synthesis prompt / question to answer from the pages.
            urls: List of specific URLs to read.
            progress: Mutable progress object.

        Returns:
            The same progress object, updated with results.
        """
        start = time.monotonic()

        try:
            # 1. Fetch pages
            progress.stage = PipelineStage.FETCHING
            pages = await self._collector.collect(urls)
            progress.pages_fetched = sum(1 for p in pages if p.fetch_ok)

            if not any(p.fetch_ok and p.text for p in pages):
                progress.stage = PipelineStage.FAILED
                progress.error = "All page fetches failed or returned empty content"
                progress.total_ms = (time.monotonic() - start) * 1000
                return progress

            # 2. Score + synthesize
            progress.stage = PipelineStage.SCORING
            top_pages = self._synthesizer.score_relevance(query, pages, self._top_k)
            progress.pages_used = len(top_pages)

            progress.stage = PipelineStage.SYNTHESIZING
            synthesis = await self._synthesizer.synthesize(query, top_pages)

            progress.stage = PipelineStage.COMPLETED
            progress.result = synthesis
            progress.total_ms = (time.monotonic() - start) * 1000

            logger.info(
                "Direct read complete: %d fetched, %d used, %.0fms",
                progress.pages_fetched,
                progress.pages_used,
                progress.total_ms,
            )

        except Exception as e:
            logger.error("Direct read pipeline failed: %s", e, exc_info=True)
            progress.stage = PipelineStage.FAILED
            progress.error = str(e)
            progress.total_ms = (time.monotonic() - start) * 1000

        return progress
