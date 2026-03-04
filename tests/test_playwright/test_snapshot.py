"""Tests for Playwright accessibility tree snapshot formatting.

Tests the aria_snapshot()-based capture (Playwright 1.58+).
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, PropertyMock

from alchemy.playwright.snapshot import (
    SnapshotResult,
    capture_snapshot,
    RefEntry,
    INTERACTIVE_ROLES,
)


# --- Fixtures ---

def _mock_page(aria_text: str | None, *, raise_error: Exception | None = None):
    """Create a mock Playwright page with an aria_snapshot() response."""
    page = AsyncMock()
    locator = AsyncMock()

    if raise_error:
        locator.aria_snapshot = AsyncMock(side_effect=raise_error)
    else:
        locator.aria_snapshot = AsyncMock(return_value=aria_text)

    page.locator = MagicMock(return_value=locator)
    return page


SIMPLE_TREE = """\
- document:
  - heading "Welcome" [level=1]
  - navigation "Main":
    - link "Home"
    - link "About"
  - textbox "Search"
  - button "Submit"\
"""

SPOTIFY_TREE = """\
- document:
  - heading "Good Evening" [level=1]
  - navigation "Main":
    - link "Home"
    - link "Search"
    - link "Your Library"
  - search "Search Spotify":
    - textbox "What do you want to play?"
  - list "Recently Played":
    - listitem "Daily Mix 1"
    - listitem "Discover Weekly"
  - button "Play"
  - slider "Volume" [valuetext=75]\
"""

GOOGLE_CONSENT = """\
- document:
  - dialog "Innan du fortsätter till Google Sök":
    - img "Google"
    - 'button "Språk: Svenska"': sv
    - link "Logga in"
    - heading "Innan du fortsätter till Google" [level=1]
    - button "Avvisa alla"
    - button "Godkänn alla"
    - link "Fler alternativ":
      - /url: https://example.com\
"""


# --- Tests ---

class TestCaptureSnapshot:
    async def test_simple_tree(self):
        page = _mock_page(SIMPLE_TREE)
        result = await capture_snapshot(page)

        assert isinstance(result, SnapshotResult)
        assert result.element_count > 0
        assert "heading" in result.text
        assert "Welcome" in result.text
        assert "button" in result.text
        assert "[ref=e" in result.text

    async def test_refs_assigned_to_interactive_elements(self):
        page = _mock_page(SIMPLE_TREE)
        result = await capture_snapshot(page)

        # link "Home", link "About", textbox "Search", button "Submit" = 4
        assert len(result.ref_map) == 4
        assert result.ref_map["e1"].role == "link"
        assert result.ref_map["e1"].name == "Home"
        assert result.ref_map["e2"].role == "link"
        assert result.ref_map["e2"].name == "About"
        assert result.ref_map["e3"].role == "textbox"
        assert result.ref_map["e3"].name == "Search"
        assert result.ref_map["e4"].role == "button"
        assert result.ref_map["e4"].name == "Submit"

    async def test_spotify_tree(self):
        page = _mock_page(SPOTIFY_TREE)
        result = await capture_snapshot(page)

        # 3 links + textbox + 2 listitems + button + slider = 8
        assert len(result.ref_map) == 8
        assert "Play" in result.text
        assert "Volume" in result.text
        assert "[ref=e" in result.text

    async def test_google_consent_dialog(self):
        page = _mock_page(GOOGLE_CONSENT)
        result = await capture_snapshot(page)

        # button "Språk: Svenska", link "Logga in", button "Avvisa alla",
        # button "Godkänn alla", link "Fler alternativ" = 5
        assert len(result.ref_map) == 5
        assert any(e.name == "Godkänn alla" for e in result.ref_map.values())
        assert any(e.name == "Avvisa alla" for e in result.ref_map.values())

    async def test_url_metadata_stripped(self):
        page = _mock_page(GOOGLE_CONSENT)
        result = await capture_snapshot(page)
        assert "/url:" not in result.text

    async def test_empty_tree(self):
        page = _mock_page(None)
        result = await capture_snapshot(page)

        assert "Empty page" in result.text
        assert len(result.ref_map) == 0

    async def test_empty_string_tree(self):
        page = _mock_page("")
        result = await capture_snapshot(page)

        assert "Empty page" in result.text
        assert len(result.ref_map) == 0

    async def test_snapshot_error(self):
        page = _mock_page(None, raise_error=Exception("boom"))
        result = await capture_snapshot(page)

        assert "Error" in result.text
        assert len(result.ref_map) == 0

    async def test_max_elements_respected(self):
        lines = ["- document:"]
        for i in range(200):
            lines.append(f'  - button "Btn {i}"')
        text = "\n".join(lines)

        page = _mock_page(text)
        result = await capture_snapshot(page, max_elements=50)
        assert result.element_count <= 50

    async def test_duplicate_role_name_indexed(self):
        text = """\
- document:
  - button "Delete"
  - button "Delete"
  - button "Delete"\
"""
        page = _mock_page(text)
        result = await capture_snapshot(page)

        assert len(result.ref_map) == 3
        assert result.ref_map["e1"].index == 0
        assert result.ref_map["e2"].index == 1
        assert result.ref_map["e3"].index == 2

    async def test_ref_injection_format(self):
        """Ref IDs should appear inline: - button "Submit" [ref=e1]"""
        page = _mock_page(SIMPLE_TREE)
        result = await capture_snapshot(page)

        # Check that ref is injected after the name
        assert 'button "Submit" [ref=e' in result.text
        assert 'link "Home" [ref=e' in result.text

    async def test_unicode_control_chars_stripped(self):
        text = '- document:\n  - button "Click\u202ame"'
        page = _mock_page(text)
        result = await capture_snapshot(page)

        assert "\u202a" not in result.text
        assert "Clickme" in result.text

    async def test_non_interactive_roles_no_ref(self):
        text = """\
- document:
  - heading "Title" [level=1]
  - navigation "Main":
    - link "Home"\
"""
        page = _mock_page(text)
        result = await capture_snapshot(page)

        # Only link "Home" gets a ref (heading and navigation are not interactive)
        assert len(result.ref_map) == 1
        assert result.ref_map["e1"].role == "link"
