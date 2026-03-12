# AlchemyAgentInstagramSaves

## What It Does

Automated agent that scrolls through your Instagram Saved posts, downloads all videos and images to a staging folder, and saves metadata. BaratzaMemory batch-ingests from that folder later.

**This agent ONLY downloads and stores. No classification, no embedding, no AI. Just scrape and save.**

## How It Works

1. User provides: Instagram saved posts URL (or agent navigates to it)
2. Agent opens Playwright browser (reuse AlchemyClick's existing browser session)
3. Scrolls through saved posts grid
4. For each post:
   - Click to open post
   - Extract: video URL or image URL, caption text, username, timestamp, post URL
   - Download media to staging folder
   - Write a `.json` sidecar with metadata
   - Close post, move to next
5. Keeps scrolling until no more posts load (infinite scroll end detection)
6. Reports: total downloaded, skipped (already exists), errors

## Staging Folder Structure

```
C:\Users\info\neosy_inbox\instagram\
├── 2026-03-10_post_abc123.mp4
├── 2026-03-10_post_abc123.json
├── 2026-03-10_post_def456.jpg
├── 2026-03-10_post_def456.json
└── ...
```

Each `.json` sidecar:
```json
{
  "source_platform": "instagram",
  "source_url": "https://www.instagram.com/p/abc123/",
  "username": "@someuser",
  "caption": "Original caption text...",
  "timestamp": "2026-03-10T14:30:00Z",
  "media_type": "video",
  "filename": "2026-03-10_post_abc123.mp4",
  "downloaded_at": "2026-03-10T22:00:00Z"
}
```

## File To Create

`alchemy/click/agents/instagram_saves.py`

## Implementation

```python
"""AlchemyAgentInstagramSaves — download all Instagram saved posts."""

import asyncio
import json
import logging
import re
from datetime import datetime
from pathlib import Path

from playwright.async_api import Page, TimeoutError as PlaywrightTimeout

logger = logging.getLogger(__name__)

STAGING_DIR = Path(r"C:\Users\info\neosy_inbox\instagram")
INSTAGRAM_SAVED_URL = "https://www.instagram.com/{username}/saved/"


class InstagramSavesAgent:
    """Scrolls Instagram saved posts, downloads all media to staging folder."""

    def __init__(self, page: Page, username: str = "me"):
        self.page = page
        self.username = username
        self.downloaded = 0
        self.skipped = 0
        self.errors = 0

    async def run(self, max_posts: int | None = None) -> dict:
        """Main entry point. Returns summary of what was downloaded."""
        STAGING_DIR.mkdir(parents=True, exist_ok=True)

        # Navigate to saved posts
        # User should already be logged in via existing Playwright session
        await self.page.goto(INSTAGRAM_SAVED_URL.format(username=self.username))
        await self.page.wait_for_load_state("networkidle")

        # Collect post links by scrolling the grid
        post_urls = await self._collect_post_urls(max_posts)
        logger.info("Found %d saved posts to process", len(post_urls))

        # Process each post
        for i, url in enumerate(post_urls):
            try:
                await self._process_post(url)
                logger.info("Processed %d/%d: %s", i + 1, len(post_urls), url)
            except Exception as e:
                logger.warning("Failed to process %s: %s", url, e)
                self.errors += 1

            # Be polite — don't hammer Instagram
            await asyncio.sleep(1.5)

        return {
            "total_found": len(post_urls),
            "downloaded": self.downloaded,
            "skipped": self.skipped,
            "errors": self.errors,
            "staging_dir": str(STAGING_DIR),
        }

    async def _collect_post_urls(self, max_posts: int | None) -> list[str]:
        """Scroll the saved posts grid and collect all post URLs."""
        urls = set()
        last_count = 0
        stale_rounds = 0

        while True:
            # Get all post links currently visible
            links = await self.page.query_selector_all('a[href*="/p/"]')
            for link in links:
                href = await link.get_attribute("href")
                if href and "/p/" in href:
                    full_url = f"https://www.instagram.com{href}" if href.startswith("/") else href
                    urls.add(full_url)

            if max_posts and len(urls) >= max_posts:
                break

            # Scroll down
            await self.page.evaluate("window.scrollBy(0, window.innerHeight)")
            await asyncio.sleep(2)

            # Check if we got new posts
            if len(urls) == last_count:
                stale_rounds += 1
                if stale_rounds >= 3:
                    break  # No more posts loading
            else:
                stale_rounds = 0
                last_count = len(urls)

        return list(urls)[:max_posts] if max_posts else list(urls)

    async def _process_post(self, post_url: str) -> None:
        """Open a post, extract media and metadata, download to staging."""
        # Extract post ID from URL
        match = re.search(r"/p/([^/]+)", post_url)
        post_id = match.group(1) if match else "unknown"

        # Check if already downloaded
        existing = list(STAGING_DIR.glob(f"*_{post_id}.*"))
        json_exists = any(f.suffix == ".json" for f in existing)
        if json_exists:
            self.skipped += 1
            return

        # Navigate to post
        await self.page.goto(post_url)
        await self.page.wait_for_load_state("networkidle")

        # Detect media type and extract URL
        media_url = None
        media_type = None

        # Try video first
        video_el = await self.page.query_selector("video")
        if video_el:
            media_url = await video_el.get_attribute("src")
            media_type = "video"

        # Fall back to image
        if not media_url:
            # Instagram images are in article > div img
            img_els = await self.page.query_selector_all("article img")
            for img in img_els:
                src = await img.get_attribute("src")
                alt = await img.get_attribute("alt") or ""
                # Skip profile pics and icons (small images)
                if src and "cdninstagram" in src and len(src) > 100:
                    media_url = src
                    media_type = "image"
                    break

        if not media_url:
            logger.warning("No media found for %s", post_url)
            self.errors += 1
            return

        # Extract caption
        caption = ""
        try:
            caption_el = await self.page.query_selector('div[data-testid="post-comment-root"] span')
            if caption_el:
                caption = await caption_el.inner_text()
        except Exception:
            pass

        # Extract username
        username = ""
        try:
            user_el = await self.page.query_selector('a[role="link"][tabindex="0"]')
            if user_el:
                username = await user_el.inner_text()
        except Exception:
            pass

        # Download media
        today = datetime.now().strftime("%Y-%m-%d")
        ext = "mp4" if media_type == "video" else "jpg"
        filename = f"{today}_post_{post_id}.{ext}"
        filepath = STAGING_DIR / filename

        # Use Playwright to download (handles auth cookies)
        response = await self.page.request.get(media_url)
        if response.ok:
            filepath.write_bytes(await response.body())
        else:
            logger.warning("Download failed (%d) for %s", response.status, post_url)
            self.errors += 1
            return

        # Write metadata sidecar
        metadata = {
            "source_platform": "instagram",
            "source_url": post_url,
            "username": username,
            "caption": caption,
            "media_type": media_type,
            "filename": filename,
            "downloaded_at": datetime.utcnow().isoformat() + "Z",
        }
        sidecar = STAGING_DIR / f"{today}_post_{post_id}.json"
        sidecar.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

        self.downloaded += 1


# --- Entry point for AlchemyClick task system ---

async def run_instagram_saves(page: Page, username: str = "me", max_posts: int | None = None) -> dict:
    """AlchemyClick task entry point."""
    agent = InstagramSavesAgent(page, username)
    return await agent.run(max_posts)
```

## How To Run

From Alchemy, this gets triggered via the AlchemyClick task system:
```python
# From server or API
result = await run_instagram_saves(page, username="yourusername", max_posts=50)
```

Or add an API endpoint later:
```
POST /v1/click/agents/instagram-saves
Body: {"username": "yourusername", "max_posts": 50}
```

## After Download — BaratzaMemory Ingest

Once files are in the staging folder, run BaratzaMemory batch ingest (Task 11 in BaratzaMemory TODO):
```
POST http://localhost:8001/ingest/batch
Body: {"source_dir": "C:\\Users\\monic\\neosy_inbox\\instagram", "source_platform": "instagram"}
```

Or manually: point the BaratzaMemory ingest at the folder and it reads each file + its .json sidecar for metadata.

## Important Notes

- User MUST be logged into Instagram in the Playwright browser session
- Instagram may rate-limit or show CAPTCHAs — agent should detect and pause
- Downloads go to staging folder, NOT directly to BaratzaMemory vault
- This agent is fire-and-forget: run it, let it scroll, come back when done
- Carousel posts (multiple images): for v1, just grab the first one. Carousel support can come later.
