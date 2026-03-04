"""Playwright-based GUI automation — accessibility tree + ref-based actions.

Tier 1 agent: structured data approach (DOM/a11y tree) instead of vision (screenshots).
Works with Chrome, Electron apps (VS Code, Spotify, Slack, Discord, Notion).
"""

from alchemy.core import BrowserManager, execute_action, SnapshotResult, capture_snapshot

__all__ = ["BrowserManager", "capture_snapshot", "execute_action", "SnapshotResult"]
