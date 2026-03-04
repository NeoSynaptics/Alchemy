"""Playwright-based GUI automation — accessibility tree + ref-based actions.

Tier 1 agent: structured data approach (DOM/a11y tree) instead of vision (screenshots).
Works with Chrome, Electron apps (VS Code, Spotify, Slack, Discord, Notion).
"""

from alchemy.playwright.browser import BrowserManager
from alchemy.playwright.executor import execute_action
from alchemy.playwright.snapshot import SnapshotResult, capture_snapshot

__all__ = ["BrowserManager", "capture_snapshot", "execute_action", "SnapshotResult"]
