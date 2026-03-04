"""Tests for Playwright browser manager."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from alchemy.playwright.browser import BrowserManager


class TestBrowserManager:
    def test_initial_state(self):
        mgr = BrowserManager()
        assert not mgr.is_running

    def test_headless_default(self):
        mgr = BrowserManager()
        assert mgr._headless is True

    def test_headless_override(self):
        mgr = BrowserManager(headless=False)
        assert mgr._headless is False

    async def test_new_page_not_started(self):
        mgr = BrowserManager()
        with pytest.raises(RuntimeError, match="not started"):
            await mgr.new_page()

    async def test_connect_cdp_not_started(self):
        mgr = BrowserManager()
        with pytest.raises(RuntimeError, match="not started"):
            await mgr.connect_cdp("http://localhost:9222")

    async def test_start_and_close(self):
        """Test start/close lifecycle with injected mocks (no real Playwright needed)."""
        mock_pw = AsyncMock()
        mock_browser = AsyncMock()

        mgr = BrowserManager()
        mgr._playwright = mock_pw
        mgr._browser = mock_browser

        assert mgr.is_running

        await mgr.close()
        assert not mgr.is_running
        mock_browser.close.assert_called_once()
        mock_pw.stop.assert_called_once()

    async def test_close_when_not_started(self):
        """Close on an unstarted manager should not raise."""
        mgr = BrowserManager()
        await mgr.close()  # Should not raise
        assert not mgr.is_running

    async def test_new_page_with_injected_browser(self):
        """Test new_page with manually injected browser (no playwright import)."""
        mock_context = AsyncMock()
        mock_page = AsyncMock()
        mock_context.new_page = AsyncMock(return_value=mock_page)

        mock_browser = AsyncMock()
        mock_browser.new_context = AsyncMock(return_value=mock_context)

        mgr = BrowserManager()
        mgr._browser = mock_browser  # Inject directly

        page = await mgr.new_page("https://example.com")
        assert page == mock_page
        mock_page.goto.assert_called_once()

    async def test_new_page_no_url(self):
        """Test new_page without URL doesn't navigate."""
        mock_context = AsyncMock()
        mock_page = AsyncMock()
        mock_context.new_page = AsyncMock(return_value=mock_page)

        mock_browser = AsyncMock()
        mock_browser.new_context = AsyncMock(return_value=mock_context)

        mgr = BrowserManager()
        mgr._browser = mock_browser

        page = await mgr.new_page()
        assert page == mock_page
        mock_page.goto.assert_not_called()

    async def test_connect_cdp_with_injected_playwright(self):
        """Test CDP connection with injected playwright instance."""
        mock_cdp_browser = AsyncMock()
        mock_context = AsyncMock()
        mock_page = AsyncMock()
        mock_cdp_browser.contexts = [mock_context]
        mock_context.pages = [mock_page]

        mock_pw = AsyncMock()
        mock_pw.chromium.connect_over_cdp = AsyncMock(return_value=mock_cdp_browser)

        mgr = BrowserManager()
        mgr._playwright = mock_pw
        mgr._browser = AsyncMock()  # Mark as started

        page = await mgr.connect_cdp("http://localhost:9222")
        assert page == mock_page
        mock_pw.chromium.connect_over_cdp.assert_called_once_with("http://localhost:9222")
