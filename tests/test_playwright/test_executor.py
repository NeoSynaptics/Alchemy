"""Tests for Playwright action executor."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from alchemy.playwright.executor import execute_action, ExecutionError, _build_locator
from alchemy.playwright.snapshot import RefEntry


# --- Fixtures ---

def _mock_page():
    """Create a mock Playwright page."""
    page = MagicMock()

    # Mock get_by_role to return a locator
    locator = AsyncMock()
    locator.click = AsyncMock()
    locator.fill = AsyncMock()
    locator.select_option = AsyncMock()
    locator.nth = MagicMock(return_value=locator)

    page.get_by_role = MagicMock(return_value=locator)
    page.mouse = AsyncMock()
    page.mouse.wheel = AsyncMock()
    page.keyboard = AsyncMock()
    page.keyboard.press = AsyncMock()
    page.wait_for_load_state = AsyncMock()

    return page, locator


REF_MAP = {
    "e1": RefEntry(role="heading", name="Welcome"),
    "e2": RefEntry(role="link", name="Home"),
    "e3": RefEntry(role="textbox", name="Search"),
    "e4": RefEntry(role="button", name="Submit"),
    "e5": RefEntry(role="button", name="Delete", index=1),
}


# --- Tests ---

class TestExecuteAction:
    async def test_click(self):
        page, locator = _mock_page()
        result = await execute_action(page, "click", "e4", REF_MAP)

        assert result is True
        page.get_by_role.assert_called_once_with("button", name="Submit")
        locator.click.assert_called_once()

    async def test_type(self):
        page, locator = _mock_page()
        result = await execute_action(page, "type", "e3", REF_MAP, text="hello world")

        assert result is True
        page.get_by_role.assert_called_once_with("textbox", name="Search")
        locator.fill.assert_called_once_with("hello world", timeout=5000)

    async def test_scroll_down(self):
        page, _ = _mock_page()
        result = await execute_action(page, "scroll", None, REF_MAP, direction="down")

        assert result is True
        page.mouse.wheel.assert_called_once_with(0, 300)

    async def test_scroll_up(self):
        page, _ = _mock_page()
        result = await execute_action(page, "scroll", None, REF_MAP, direction="up")

        assert result is True
        page.mouse.wheel.assert_called_once_with(0, -300)

    async def test_key_press(self):
        page, _ = _mock_page()
        result = await execute_action(page, "key", None, REF_MAP, key_name="Enter")

        assert result is True
        page.keyboard.press.assert_called_once_with("Enter")

    async def test_select(self):
        page, locator = _mock_page()
        result = await execute_action(page, "select", "e3", REF_MAP, text="Option A")

        assert result is True
        locator.select_option.assert_called_once_with(label="Option A", timeout=5000)

    async def test_done(self):
        page, _ = _mock_page()
        result = await execute_action(page, "done", None, REF_MAP)
        assert result is True

    async def test_wait(self):
        page, _ = _mock_page()
        result = await execute_action(page, "wait", None, REF_MAP)
        assert result is True

    async def test_click_missing_ref(self):
        page, _ = _mock_page()
        with pytest.raises(ExecutionError, match="requires a ref"):
            await execute_action(page, "click", None, REF_MAP)

    async def test_click_unknown_ref(self):
        page, _ = _mock_page()
        with pytest.raises(ExecutionError, match="Unknown ref"):
            await execute_action(page, "click", "e99", REF_MAP)

    async def test_type_missing_text(self):
        page, _ = _mock_page()
        with pytest.raises(ExecutionError, match="requires text"):
            await execute_action(page, "type", "e3", REF_MAP)

    async def test_key_missing_name(self):
        page, _ = _mock_page()
        with pytest.raises(ExecutionError, match="requires a key_name"):
            await execute_action(page, "key", None, REF_MAP)

    async def test_unknown_action(self):
        page, _ = _mock_page()
        with pytest.raises(ExecutionError, match="Unknown action"):
            await execute_action(page, "dance", "e1", REF_MAP)

    async def test_nth_index_used(self):
        page, locator = _mock_page()
        # e5 has index=1
        await execute_action(page, "click", "e5", REF_MAP)
        locator.nth.assert_called_once_with(1)


class TestBuildLocator:
    def test_simple_locator_uses_nth_zero(self):
        """Even index=0 uses .nth(0) to avoid strict mode violations."""
        page = MagicMock()
        locator = MagicMock()
        nth_locator = MagicMock()
        locator.nth = MagicMock(return_value=nth_locator)
        page.get_by_role = MagicMock(return_value=locator)

        entry = RefEntry(role="button", name="OK", index=0)
        result = _build_locator(page, entry)

        page.get_by_role.assert_called_once_with("button", name="OK")
        locator.nth.assert_called_once_with(0)
        assert result == nth_locator

    def test_nth_locator(self):
        page = MagicMock()
        locator = MagicMock()
        nth_locator = MagicMock()
        locator.nth = MagicMock(return_value=nth_locator)
        page.get_by_role = MagicMock(return_value=locator)

        entry = RefEntry(role="button", name="Delete", index=2)
        result = _build_locator(page, entry)

        locator.nth.assert_called_once_with(2)
        assert result == nth_locator
