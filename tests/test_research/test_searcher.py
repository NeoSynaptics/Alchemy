"""Tests for SearchProvider — DuckDuckGo search wrapper."""

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from alchemy.research.searcher import SearchProvider, SearchResult


class TestSearchProvider:
    def test_init_default_max_results(self):
        provider = SearchProvider()
        assert provider._max_results == 5

    def test_init_custom_max_results(self):
        provider = SearchProvider(max_results_per_query=10)
        assert provider._max_results == 10

    async def test_search_single_query(self):
        provider = SearchProvider()
        mock_results = [
            {"href": "https://example.com/1", "title": "Result 1", "body": "Snippet 1"},
            {"href": "https://example.com/2", "title": "Result 2", "body": "Snippet 2"},
        ]

        with patch.object(provider, "_search_sync", return_value=mock_results):
            results = await provider.search("test query")

        assert len(results) == 2
        assert results[0].url == "https://example.com/1"
        assert results[0].title == "Result 1"
        assert results[0].snippet == "Snippet 1"

    async def test_search_empty_results(self):
        provider = SearchProvider()

        with patch.object(provider, "_search_sync", return_value=[]):
            results = await provider.search("empty query")

        assert results == []

    async def test_search_exception_returns_empty(self):
        provider = SearchProvider()

        with patch.object(provider, "_search_sync", side_effect=Exception("network error")):
            results = await provider.search("failing query")

        assert results == []

    async def test_search_skips_empty_href(self):
        provider = SearchProvider()
        mock_results = [
            {"href": "https://example.com/1", "title": "Good", "body": "ok"},
            {"href": "", "title": "No URL", "body": "skip"},
            {"title": "Missing href", "body": "skip"},
        ]

        with patch.object(provider, "_search_sync", return_value=mock_results):
            results = await provider.search("filter test")

        assert len(results) == 1
        assert results[0].url == "https://example.com/1"

    async def test_search_many_parallel(self):
        provider = SearchProvider()

        async def mock_search(query):
            return [SearchResult(url=f"https://example.com/{query}", title=query, snippet="")]

        provider.search = mock_search
        results = await provider.search_many(["q1", "q2", "q3"])

        assert len(results) == 3
        urls = {r.url for r in results}
        assert "https://example.com/q1" in urls
        assert "https://example.com/q2" in urls
        assert "https://example.com/q3" in urls

    async def test_search_many_dedup(self):
        provider = SearchProvider()

        call_count = 0

        async def mock_search(query):
            nonlocal call_count
            call_count += 1
            return [SearchResult(url="https://example.com/same", title=query, snippet="")]

        provider.search = mock_search
        results = await provider.search_many(["q1", "q2"])

        assert call_count == 2
        assert len(results) == 1
        assert results[0].url == "https://example.com/same"

    async def test_search_many_partial_failure(self):
        provider = SearchProvider()

        async def mock_search(query):
            if query == "fail":
                raise Exception("network error")
            return [SearchResult(url=f"https://example.com/{query}", title=query, snippet="")]

        provider.search = mock_search
        results = await provider.search_many(["ok1", "fail", "ok2"])

        assert len(results) == 2
        urls = {r.url for r in results}
        assert "https://example.com/ok1" in urls
        assert "https://example.com/ok2" in urls

    def test_search_sync(self):
        """Verify _search_sync calls DDGS correctly."""
        provider = SearchProvider(max_results_per_query=3)

        mock_ddgs_instance = MagicMock()
        mock_ddgs_instance.text = MagicMock(return_value=[
            {"href": "https://example.com", "title": "Test", "body": "Body"},
        ])
        mock_ddgs_instance.__enter__ = MagicMock(return_value=mock_ddgs_instance)
        mock_ddgs_instance.__exit__ = MagicMock(return_value=False)

        mock_ddgs_cls = MagicMock(return_value=mock_ddgs_instance)

        # Patch both possible import paths (ddgs and duckduckgo_search)
        with patch.dict("sys.modules", {"ddgs": MagicMock(DDGS=mock_ddgs_cls)}):
            results = provider._search_sync("test query")

        assert len(results) == 1
        mock_ddgs_instance.text.assert_called_once_with(
            "test query", region="wt-wt", safesearch="moderate", max_results=3
        )
