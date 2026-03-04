"""Tests for Playwright accessibility tree snapshot formatting."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from alchemy.playwright.snapshot import (
    SnapshotResult,
    capture_snapshot,
    _should_assign_ref,
    _format_node,
    RefEntry,
)


# --- Fixtures ---

def _mock_page(tree: dict | None):
    """Create a mock Playwright page with an accessibility snapshot."""
    page = AsyncMock()
    page.accessibility = MagicMock()
    page.accessibility.snapshot = AsyncMock(return_value=tree)
    return page


SIMPLE_TREE = {
    "role": "WebArea",
    "name": "Test Page",
    "children": [
        {"role": "heading", "name": "Welcome", "level": 1},
        {
            "role": "navigation",
            "name": "Main",
            "children": [
                {"role": "link", "name": "Home"},
                {"role": "link", "name": "About"},
            ],
        },
        {"role": "textbox", "name": "Search"},
        {"role": "button", "name": "Submit"},
    ],
}

SPOTIFY_TREE = {
    "role": "WebArea",
    "name": "Spotify",
    "children": [
        {"role": "heading", "name": "Good Evening", "level": 1},
        {
            "role": "navigation",
            "name": "Main",
            "children": [
                {"role": "link", "name": "Home"},
                {"role": "link", "name": "Search"},
                {"role": "link", "name": "Your Library"},
            ],
        },
        {
            "role": "search",
            "name": "Search Spotify",
            "children": [
                {"role": "textbox", "name": "What do you want to play?"},
            ],
        },
        {
            "role": "list",
            "name": "Recently Played",
            "children": [
                {"role": "listitem", "name": "Daily Mix 1"},
                {"role": "listitem", "name": "Discover Weekly"},
            ],
        },
        {"role": "button", "name": "Play"},
        {"role": "slider", "name": "Volume", "valuetext": "75"},
    ],
}


# --- Tests ---

class TestShouldAssignRef:
    def test_interactive_roles_get_ref(self):
        assert _should_assign_ref("button", "Submit") is True
        assert _should_assign_ref("link", "Home") is True
        assert _should_assign_ref("textbox", "Search") is True
        assert _should_assign_ref("checkbox", "Agree") is True
        assert _should_assign_ref("slider", "Volume") is True

    def test_interactive_roles_get_ref_even_without_name(self):
        assert _should_assign_ref("button", "") is True
        assert _should_assign_ref("textbox", "") is True

    def test_container_roles_no_ref(self):
        assert _should_assign_ref("navigation", "Main") is False
        assert _should_assign_ref("list", "Items") is False
        assert _should_assign_ref("WebArea", "Page") is False

    def test_named_display_elements_get_ref(self):
        assert _should_assign_ref("heading", "Title") is True
        assert _should_assign_ref("img", "Logo") is True

    def test_unnamed_display_elements_no_ref(self):
        assert _should_assign_ref("heading", "") is False


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

    async def test_refs_assigned_correctly(self):
        page = _mock_page(SIMPLE_TREE)
        result = await capture_snapshot(page)

        # heading "Welcome", link "Home", link "About", textbox "Search", button "Submit"
        assert len(result.ref_map) == 5
        assert "e1" in result.ref_map
        assert result.ref_map["e1"].role == "heading"
        assert result.ref_map["e1"].name == "Welcome"

    async def test_spotify_tree(self):
        page = _mock_page(SPOTIFY_TREE)
        result = await capture_snapshot(page)

        # heading, 3 links, textbox, 2 listitems, button, slider = 9
        assert len(result.ref_map) == 9
        assert "Play" in result.text
        assert "Volume" in result.text
        assert "[ref=e" in result.text

    async def test_empty_tree(self):
        page = _mock_page(None)
        result = await capture_snapshot(page)

        assert "Empty page" in result.text
        assert len(result.ref_map) == 0

    async def test_snapshot_error(self):
        page = AsyncMock()
        page.accessibility = MagicMock()
        page.accessibility.snapshot = AsyncMock(side_effect=Exception("boom"))
        result = await capture_snapshot(page)

        assert "Error" in result.text
        assert len(result.ref_map) == 0

    async def test_max_elements_respected(self):
        # Create a tree with many elements
        children = [{"role": "button", "name": f"Btn {i}"} for i in range(200)]
        tree = {"role": "WebArea", "name": "Big Page", "children": children}
        page = _mock_page(tree)

        result = await capture_snapshot(page, max_elements=50)
        assert result.element_count <= 51  # +1 for boundary

    async def test_properties_included(self):
        tree = {
            "role": "WebArea",
            "name": "Page",
            "children": [
                {"role": "checkbox", "name": "Agree", "checked": True},
                {"role": "slider", "name": "Vol", "valuetext": "50"},
            ],
        }
        page = _mock_page(tree)
        result = await capture_snapshot(page)

        assert "[checked=True]" in result.text
        assert "[valuetext=50]" in result.text

    async def test_indentation_depth(self):
        tree = {
            "role": "WebArea",
            "name": "Page",
            "children": [
                {
                    "role": "navigation",
                    "name": "Nav",
                    "children": [
                        {"role": "link", "name": "Deep Link"},
                    ],
                },
            ],
        }
        page = _mock_page(tree)
        result = await capture_snapshot(page)

        lines = result.text.strip().split("\n")
        # Nav should be at depth 0, link at depth 1
        assert any(line.startswith("  ") and "Deep Link" in line for line in lines)

    async def test_long_name_truncated(self):
        tree = {
            "role": "WebArea",
            "name": "Page",
            "children": [
                {"role": "heading", "name": "A" * 200},
            ],
        }
        page = _mock_page(tree)
        result = await capture_snapshot(page)

        assert "..." in result.text
        # Should not have the full 200 chars
        assert "A" * 200 not in result.text

    async def test_duplicate_role_name_indexed(self):
        tree = {
            "role": "WebArea",
            "name": "Page",
            "children": [
                {"role": "button", "name": "Delete"},
                {"role": "button", "name": "Delete"},
                {"role": "button", "name": "Delete"},
            ],
        }
        page = _mock_page(tree)
        result = await capture_snapshot(page)

        assert len(result.ref_map) == 3
        assert result.ref_map["e1"].index == 0
        assert result.ref_map["e2"].index == 1
        assert result.ref_map["e3"].index == 2
