"""Playwright browser lifecycle — launch headless Chromium or connect to Electron via CDP."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class BrowserManager:
    """Manages headless Chromium and Electron CDP connections.

    Usage:
        mgr = BrowserManager()
        await mgr.start()
        page = await mgr.new_page("https://example.com")
        # ... use page ...
        await mgr.close()
    """

    def __init__(self, headless: bool = True):
        self._headless = headless
        self._playwright = None
        self._browser = None
        self._cdp_browsers: dict[str, object] = {}  # endpoint → browser

    async def start(self):
        """Launch Playwright and headless Chromium."""
        from playwright.async_api import async_playwright

        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(headless=self._headless)
        logger.info("Chromium launched (headless=%s)", self._headless)

    async def new_page(self, url: str | None = None):
        """Create a new browser page, optionally navigating to a URL.

        Returns:
            Playwright Page object.
        """
        if not self._browser:
            raise RuntimeError("BrowserManager not started — call start() first")

        context = await self._browser.new_context()
        page = await context.new_page()

        if url:
            await page.goto(url, wait_until="domcontentloaded")
            logger.info("Navigated to %s", url)

        return page

    async def connect_cdp(self, endpoint: str):
        """Connect to an Electron app or remote browser via Chrome DevTools Protocol.

        Args:
            endpoint: CDP endpoint URL, e.g. "http://localhost:9222"

        Returns:
            Playwright Page object for the first available page.
        """
        if not self._playwright:
            raise RuntimeError("BrowserManager not started — call start() first")

        browser = await self._playwright.chromium.connect_over_cdp(endpoint)
        self._cdp_browsers[endpoint] = browser

        # Get the first context and page
        contexts = browser.contexts
        if contexts and contexts[0].pages:
            page = contexts[0].pages[0]
        else:
            ctx = contexts[0] if contexts else await browser.new_context()
            page = await ctx.new_page()

        logger.info("Connected via CDP to %s", endpoint)
        return page

    async def close(self):
        """Clean up all browser instances."""
        for endpoint, browser in self._cdp_browsers.items():
            try:
                await browser.close()
                logger.debug("Closed CDP connection to %s", endpoint)
            except Exception:
                pass
        self._cdp_browsers.clear()

        if self._browser:
            await self._browser.close()
            self._browser = None

        if self._playwright:
            await self._playwright.stop()
            self._playwright = None

        logger.info("BrowserManager closed")

    @property
    def is_running(self) -> bool:
        return self._browser is not None
