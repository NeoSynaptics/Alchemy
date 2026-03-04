"""Tests for ResearchEngine — pipeline orchestration."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from alchemy.research.collector import PageContent
from alchemy.research.engine import PipelineStage, ResearchEngine, ResearchProgress
from alchemy.research.searcher import SearchResult
from alchemy.research.synthesizer import SynthesisResult


def _make_engine(
    decompose_result=None,
    search_result=None,
    collect_result=None,
    score_result=None,
    synthesize_result=None,
):
    """Build a ResearchEngine with fully mocked dependencies."""
    default_page = PageContent(url="https://a.com", title="A", text="content", word_count=100)

    synth = AsyncMock()
    synth.decompose_query = AsyncMock(
        return_value=decompose_result if decompose_result is not None else ["q1", "q2"]
    )
    synth.score_relevance = MagicMock(
        return_value=score_result if score_result is not None else [default_page]
    )
    synth.synthesize = AsyncMock(
        return_value=synthesize_result
        if synthesize_result is not None
        else SynthesisResult(
            answer="The answer",
            sources=[{"title": "A", "excerpt": "key point"}],
            inference_ms=100.0,
        )
    )

    searcher = AsyncMock()
    searcher.search_many = AsyncMock(
        return_value=search_result
        if search_result is not None
        else [SearchResult(url="https://a.com", title="A", snippet="snip")]
    )

    collector = AsyncMock()
    collector.collect = AsyncMock(
        return_value=collect_result if collect_result is not None else [default_page]
    )

    engine = ResearchEngine(
        synthesizer=synth,
        searcher=searcher,
        collector=collector,
        max_queries=10,
        top_k=5,
    )
    return engine, synth, searcher, collector


class TestRunSemantic:
    async def test_full_pipeline_success(self):
        engine, synth, searcher, collector = _make_engine()
        progress = ResearchProgress()

        result = await engine.run_semantic("test question", progress)

        assert result.stage == PipelineStage.COMPLETED
        assert result.result is not None
        assert result.result.answer == "The answer"
        assert result.queries_generated == 2
        assert result.pages_fetched == 1
        assert result.pages_used == 1
        assert result.total_ms > 0

        synth.decompose_query.assert_called_once()
        searcher.search_many.assert_called_once()
        collector.collect.assert_called_once()
        synth.score_relevance.assert_called_once()
        synth.synthesize.assert_called_once()

    async def test_empty_decomposition_uses_original_query(self):
        engine, synth, searcher, _ = _make_engine(decompose_result=[])
        progress = ResearchProgress()

        await engine.run_semantic("original question", progress)

        # Should use original query as search input
        searcher.search_many.assert_called_once_with(["original question"])

    async def test_no_search_results_fails(self):
        engine, _, _, _ = _make_engine(search_result=[])
        progress = ResearchProgress()

        result = await engine.run_semantic("test", progress)

        assert result.stage == PipelineStage.FAILED
        assert "No search results" in result.error

    async def test_all_fetches_fail(self):
        failed_pages = [
            PageContent(url="https://a.com", title="A", text="", fetch_ok=False, error="failed")
        ]
        engine, _, _, _ = _make_engine(collect_result=failed_pages)
        progress = ResearchProgress()

        result = await engine.run_semantic("test", progress)

        assert result.stage == PipelineStage.FAILED
        assert "failed or returned empty" in result.error

    async def test_pipeline_exception_caught(self):
        engine, synth, _, _ = _make_engine()
        synth.decompose_query = AsyncMock(side_effect=Exception("unexpected crash"))
        progress = ResearchProgress()

        result = await engine.run_semantic("test", progress)

        assert result.stage == PipelineStage.FAILED
        assert "unexpected crash" in result.error
        assert result.total_ms > 0

    async def test_progress_stages_updated(self):
        """Verify progress flows through all expected stages."""
        engine, _, _, _ = _make_engine()
        progress = ResearchProgress()

        result = await engine.run_semantic("test", progress)

        # Final stage should be COMPLETED
        assert result.stage == PipelineStage.COMPLETED
        # All pipeline outputs should be populated
        assert result.queries_generated > 0
        assert result.pages_fetched > 0
        assert result.pages_used > 0
        assert result.result is not None


class TestRunDirect:
    async def test_direct_skips_search(self):
        engine, synth, searcher, collector = _make_engine()
        progress = ResearchProgress()

        await engine.run_direct("summarize this", ["https://a.com"], progress)

        searcher.search_many.assert_not_called()
        synth.decompose_query.assert_not_called()
        collector.collect.assert_called_once_with(["https://a.com"])

    async def test_direct_success(self):
        engine, _, _, _ = _make_engine()
        progress = ResearchProgress()

        result = await engine.run_direct("question", ["https://a.com"], progress)

        assert result.stage == PipelineStage.COMPLETED
        assert result.result.answer == "The answer"

    async def test_direct_all_fail(self):
        failed_pages = [
            PageContent(url="https://a.com", title="A", text="", fetch_ok=False, error="failed")
        ]
        engine, _, _, _ = _make_engine(collect_result=failed_pages)
        progress = ResearchProgress()

        result = await engine.run_direct("test", ["https://a.com"], progress)

        assert result.stage == PipelineStage.FAILED

    async def test_direct_exception_caught(self):
        engine, _, _, collector = _make_engine()
        collector.collect = AsyncMock(side_effect=Exception("network error"))
        progress = ResearchProgress()

        result = await engine.run_direct("test", ["https://a.com"], progress)

        assert result.stage == PipelineStage.FAILED
        assert "network error" in result.error
