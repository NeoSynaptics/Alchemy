"""Page fetcher + content extractor — httpx primary, Playwright fallback."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)

_BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36"
)

_BLOCKED_STATUS = {403, 429, 503}


@dataclass
class PageContent:
    """Extracted content from a single page."""

    url: str
    title: str
    text: str  # Clean extracted text (trafilatura)
    fetch_ok: bool = True
    error: str | None = None
    word_count: int = 0


class PageCollector:
    """Fetches pages and extracts clean text.

    Primary: httpx with browser User-Agent.
    Fallback: BrowserManager (Playwright) for JS-rendered / blocked pages.

    Args:
        timeout: HTTP request timeout in seconds.
        browser_manager: Optional BrowserManager for fallback rendering.
        max_pages: Maximum pages to fetch (caps the input list).
    """

    def __init__(
        self,
        timeout: float = 15.0,
        browser_manager=None,
        max_pages: int = 8,
    ):
        self._timeout = timeout
        self._browser_mgr = browser_manager
        self._max_pages = max_pages

    async def collect(
        self, urls: list[str], titles: dict[str, str] | None = None
    ) -> list[PageContent]:
        """Fetch and extract content from multiple URLs in parallel.

        Args:
            urls: List of URLs to fetch.
            titles: Optional URL->title mapping from search results.

        Returns:
            List of PageContent objects (one per URL, in same order).
        """
        titles = titles or {}
        capped = urls[: self._max_pages]

        tasks = [self._fetch_one(url, titles.get(url, "")) for url in capped]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        pages: list[PageContent] = []
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                pages.append(
                    PageContent(
                        url=capped[i],
                        title=titles.get(capped[i], ""),
                        text="",
                        fetch_ok=False,
                        error=str(r),
                    )
                )
            else:
                pages.append(r)

        ok = sum(1 for p in pages if p.fetch_ok and p.text)
        logger.info("Collector: %d/%d pages fetched successfully", ok, len(capped))
        return pages

    async def _fetch_one(self, url: str, title: str) -> PageContent:
        """Fetch a single page, extract text. Falls back to Playwright if blocked."""
        html = await self._fetch_httpx(url)

        if html is None and self._browser_mgr:
            logger.info("Falling back to Playwright for %s", url)
            html = await self._fetch_playwright(url)

        if not html:
            return PageContent(url=url, title=title, text="", fetch_ok=False, error="fetch failed")

        text = self._extract(html)
        if not text:
            return PageContent(
                url=url, title=title, text="", fetch_ok=False, error="extraction empty"
            )

        extracted_title = self._extract_title(html) if not title else title
        word_count = len(text.split())

        return PageContent(
            url=url,
            title=extracted_title or title or url,
            text=text,
            fetch_ok=True,
            word_count=word_count,
        )

    async def _fetch_httpx(self, url: str) -> str | None:
        """Fetch page HTML via httpx."""
        try:
            async with httpx.AsyncClient(
                timeout=self._timeout,
                follow_redirects=True,
                headers={"User-Agent": _BROWSER_UA},
            ) as client:
                resp = await client.get(url)
                if resp.status_code in _BLOCKED_STATUS:
                    logger.debug("httpx blocked (%d) for %s", resp.status_code, url)
                    return None
                resp.raise_for_status()
                return resp.text
        except Exception as e:
            logger.debug("httpx fetch failed for %s: %s", url, e)
            return None

    async def _fetch_playwright(self, url: str) -> str | None:
        """Fallback: render page via Playwright headless browser."""
        page = None
        try:
            page = await self._browser_mgr.new_page(url)
            html = await page.content()
            return html
        except Exception as e:
            logger.debug("Playwright fetch failed for %s: %s", url, e)
            return None
        finally:
            if page:
                try:
                    await page.close()
                except Exception:
                    pass

    @staticmethod
    def _extract(html: str) -> str:
        """Extract clean text from HTML using trafilatura."""
        import trafilatura

        return trafilatura.extract(html) or ""

    @staticmethod
    def _extract_title(html: str) -> str:
        """Extract page title from HTML."""
        import trafilatura

        metadata = trafilatura.extract_metadata(html)
        if metadata and metadata.title:
            return metadata.title
        return ""
