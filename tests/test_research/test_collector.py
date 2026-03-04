"""Tests for PageCollector — fetch + extract."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from alchemy.research.collector import PageCollector, PageContent


class TestPageCollector:
    def test_init_defaults(self):
        collector = PageCollector()
        assert collector._timeout == 15.0
        assert collector._max_pages == 8
        assert collector._browser_mgr is None

    def test_init_custom(self):
        mock_browser = MagicMock()
        collector = PageCollector(timeout=10.0, browser_manager=mock_browser, max_pages=5)
        assert collector._timeout == 10.0
        assert collector._max_pages == 5
        assert collector._browser_mgr is mock_browser


class TestFetchHttpx:
    async def test_success(self):
        collector = PageCollector()
        mock_resp = httpx.Response(200, text="<html><body>Hello</body></html>",
                                   request=httpx.Request("GET", "https://example.com"))

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("alchemy.research.collector.httpx.AsyncClient", return_value=mock_client):
            result = await collector._fetch_httpx("https://example.com")

        assert result == "<html><body>Hello</body></html>"

    async def test_blocked_403_returns_none(self):
        collector = PageCollector()
        mock_resp = httpx.Response(403, text="Forbidden",
                                   request=httpx.Request("GET", "https://example.com"))

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("alchemy.research.collector.httpx.AsyncClient", return_value=mock_client):
            result = await collector._fetch_httpx("https://example.com")

        assert result is None

    async def test_exception_returns_none(self):
        collector = PageCollector()
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=httpx.ConnectError("connection refused"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("alchemy.research.collector.httpx.AsyncClient", return_value=mock_client):
            result = await collector._fetch_httpx("https://example.com")

        assert result is None


class TestFetchPlaywright:
    async def test_success(self):
        mock_page = AsyncMock()
        mock_page.content = AsyncMock(return_value="<html>Playwright</html>")
        mock_page.close = AsyncMock()

        mock_browser = AsyncMock()
        mock_browser.new_page = AsyncMock(return_value=mock_page)

        collector = PageCollector(browser_manager=mock_browser)
        result = await collector._fetch_playwright("https://example.com")

        assert result == "<html>Playwright</html>"
        mock_page.close.assert_called_once()

    async def test_exception_returns_none(self):
        mock_browser = AsyncMock()
        mock_browser.new_page = AsyncMock(side_effect=Exception("browser error"))

        collector = PageCollector(browser_manager=mock_browser)
        result = await collector._fetch_playwright("https://example.com")

        assert result is None

    async def test_closes_page_on_error(self):
        mock_page = AsyncMock()
        mock_page.content = AsyncMock(side_effect=Exception("content error"))
        mock_page.close = AsyncMock()

        mock_browser = AsyncMock()
        mock_browser.new_page = AsyncMock(return_value=mock_page)

        collector = PageCollector(browser_manager=mock_browser)
        result = await collector._fetch_playwright("https://example.com")

        assert result is None
        mock_page.close.assert_called_once()


class TestExtract:
    def test_extract_returns_text(self):
        with patch("trafilatura.extract", return_value="Clean text content"):
            result = PageCollector._extract("<html><body>Clean text content</body></html>")
        assert result == "Clean text content"

    def test_extract_empty_returns_empty_string(self):
        with patch("trafilatura.extract", return_value=None):
            result = PageCollector._extract("<html></html>")
        assert result == ""


class TestCollect:
    async def test_parallel_fetch(self):
        collector = PageCollector(max_pages=10)

        async def mock_fetch_one(url, title):
            return PageContent(url=url, title=title or url, text="content", word_count=1)

        collector._fetch_one = mock_fetch_one
        results = await collector.collect(
            ["https://a.com", "https://b.com", "https://c.com"],
            {"https://a.com": "Page A"},
        )

        assert len(results) == 3
        assert results[0].title == "Page A"
        assert results[1].title == "https://b.com"

    async def test_caps_at_max_pages(self):
        collector = PageCollector(max_pages=2)

        call_count = 0

        async def mock_fetch_one(url, title):
            nonlocal call_count
            call_count += 1
            return PageContent(url=url, title=url, text="content", word_count=1)

        collector._fetch_one = mock_fetch_one
        results = await collector.collect(
            ["https://a.com", "https://b.com", "https://c.com", "https://d.com"]
        )

        assert len(results) == 2
        assert call_count == 2

    async def test_exception_in_one_url(self):
        collector = PageCollector(max_pages=10)

        async def mock_fetch_one(url, title):
            if "bad" in url:
                raise Exception("fetch error")
            return PageContent(url=url, title=url, text="content", word_count=1)

        collector._fetch_one = mock_fetch_one
        results = await collector.collect(["https://good.com", "https://bad.com"])

        assert len(results) == 2
        assert results[0].fetch_ok is True
        assert results[1].fetch_ok is False
        assert "fetch error" in results[1].error

    async def test_titles_passed_through(self):
        collector = PageCollector()

        async def mock_fetch_one(url, title):
            return PageContent(url=url, title=title, text="content", word_count=1)

        collector._fetch_one = mock_fetch_one
        results = await collector.collect(
            ["https://example.com"],
            {"https://example.com": "My Title"},
        )

        assert results[0].title == "My Title"


class TestFetchOne:
    async def test_httpx_success_path(self):
        collector = PageCollector()

        collector._fetch_httpx = AsyncMock(return_value="<html>content</html>")
        with patch("trafilatura.extract", return_value="clean content"):
            result = await collector._fetch_one("https://example.com", "Test Page")

        assert result.fetch_ok is True
        assert result.text == "clean content"
        assert result.title == "Test Page"

    async def test_httpx_blocked_falls_back_to_playwright(self):
        mock_browser = AsyncMock()
        collector = PageCollector(browser_manager=mock_browser)

        collector._fetch_httpx = AsyncMock(return_value=None)
        collector._fetch_playwright = AsyncMock(return_value="<html>pw content</html>")
        with patch("trafilatura.extract", return_value="pw clean"):
            result = await collector._fetch_one("https://example.com", "PW Page")

        collector._fetch_playwright.assert_called_once_with("https://example.com")
        assert result.fetch_ok is True
        assert result.text == "pw clean"

    async def test_no_browser_manager_no_fallback(self):
        collector = PageCollector(browser_manager=None)
        collector._fetch_httpx = AsyncMock(return_value=None)

        result = await collector._fetch_one("https://example.com", "")

        assert result.fetch_ok is False
        assert result.error == "fetch failed"

    async def test_extraction_empty_returns_error(self):
        collector = PageCollector()
        collector._fetch_httpx = AsyncMock(return_value="<html></html>")

        with patch("trafilatura.extract", return_value=None):
            result = await collector._fetch_one("https://example.com", "")

        assert result.fetch_ok is False
        assert result.error == "extraction empty"
