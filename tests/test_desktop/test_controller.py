"""Tests for DesktopController — screenshot capture + win32 invisible input."""

from __future__ import annotations

from collections import namedtuple
from unittest.mock import MagicMock, patch

import pytest

from alchemy.desktop.controller import (
    DesktopController,
    ScreenInfo,
    _to_absolute,
    _send_click,
    _send_right_click,
    _send_text,
    _send_hotkey,
    _VK_MAP,
)


# --- Mock pyautogui ---

Size = namedtuple("Size", ["width", "height"])


def _make_mock_pyautogui():
    """Create a mock pyautogui module."""
    mock = MagicMock()
    mock.size.return_value = Size(1920, 1080)
    mock.FAILSAFE = False
    img_mock = MagicMock()
    img_mock.resize.return_value = img_mock
    img_mock.save = _mock_save_jpeg
    mock.screenshot.return_value = img_mock
    return mock


def _mock_save_jpeg(buf, format="JPEG", quality=85):
    """Write fake JPEG bytes to buffer."""
    buf.write(b"\xff\xd8\xff\xe0" + b"\x00" * 100)


class TestDesktopControllerInit:
    def test_default_settings(self):
        ctrl = DesktopController()
        assert ctrl._screenshot_width == 0  # Full res by default
        assert ctrl._screenshot_height == 0
        assert ctrl._screenshot_quality == 85

    def test_custom_settings(self):
        ctrl = DesktopController(
            screenshot_width=800, screenshot_height=600,
            screenshot_quality=50,
        )
        assert ctrl._screenshot_width == 800
        assert ctrl._screenshot_height == 600
        assert ctrl._screenshot_quality == 50

    def test_pyautogui_not_loaded_initially(self):
        ctrl = DesktopController()
        assert ctrl._pyautogui is None

    def test_screen_not_loaded_initially(self):
        ctrl = DesktopController()
        assert ctrl._screen is None


class TestScreenInfo:
    def test_screen_property(self):
        ctrl = DesktopController()
        ctrl._screen = ScreenInfo(width=1920, height=1080)
        assert ctrl.screen.width == 1920
        assert ctrl.screen.height == 1080

    def test_image_width_full_res(self):
        ctrl = DesktopController(screenshot_width=0)
        ctrl._screen = ScreenInfo(width=1920, height=1080)
        assert ctrl.image_width == 1920
        assert ctrl.image_height == 1080

    def test_image_width_resized(self):
        ctrl = DesktopController(screenshot_width=1280, screenshot_height=720)
        ctrl._screen = ScreenInfo(width=1920, height=1080)
        assert ctrl.image_width == 1280
        assert ctrl.image_height == 720


class TestCoordinateConversion:
    def test_to_absolute_center(self):
        abs_x, abs_y = _to_absolute(960, 540, 1920, 1080)
        assert abs_x == 32768  # 960/1920 * 65536
        assert abs_y == 32768  # 540/1080 * 65536

    def test_to_absolute_origin(self):
        abs_x, abs_y = _to_absolute(0, 0, 1920, 1080)
        assert abs_x == 0
        assert abs_y == 0

    def test_to_absolute_bottom_right(self):
        abs_x, abs_y = _to_absolute(1920, 1080, 1920, 1080)
        assert abs_x == 65536
        assert abs_y == 65536


class TestScreenshot:
    @pytest.mark.asyncio
    async def test_screenshot_returns_bytes(self):
        ctrl = DesktopController()
        ctrl._pyautogui = _make_mock_pyautogui()
        ctrl._screen = ScreenInfo(width=1920, height=1080)

        result = await ctrl.screenshot()
        assert isinstance(result, bytes)
        assert result[:2] == b"\xff\xd8"  # JPEG magic bytes

    @pytest.mark.asyncio
    async def test_screenshot_no_resize_when_zero(self):
        ctrl = DesktopController(screenshot_width=0, screenshot_height=0)
        mock_pag = _make_mock_pyautogui()
        ctrl._pyautogui = mock_pag
        ctrl._screen = ScreenInfo(width=1920, height=1080)

        await ctrl.screenshot()
        img = mock_pag.screenshot.return_value
        img.resize.assert_not_called()

    @pytest.mark.asyncio
    async def test_screenshot_resize_when_set(self):
        ctrl = DesktopController(screenshot_width=640, screenshot_height=480)
        mock_pag = _make_mock_pyautogui()
        ctrl._pyautogui = mock_pag
        ctrl._screen = ScreenInfo(width=1920, height=1080)

        await ctrl.screenshot()
        img = mock_pag.screenshot.return_value
        img.resize.assert_called_once_with((640, 480))


class TestWin32Click:
    @pytest.mark.asyncio
    async def test_click_returns_string(self):
        ctrl = DesktopController()
        ctrl._screen = ScreenInfo(width=1920, height=1080)

        with patch("alchemy.desktop.controller._send_click", return_value="clicked (500, 300)") as mock:
            result = await ctrl.click(500, 300)
            mock.assert_called_once_with(500, 300, 1920, 1080)
            assert "clicked" in result

    @pytest.mark.asyncio
    async def test_double_click_returns_string(self):
        ctrl = DesktopController()
        ctrl._screen = ScreenInfo(width=1920, height=1080)

        with patch("alchemy.desktop.controller._send_double_click") as mock:
            result = await ctrl.double_click(500, 300)
            mock.assert_called_once_with(500, 300, 1920, 1080)
            assert "double_clicked" in result

    @pytest.mark.asyncio
    async def test_right_click_returns_string(self):
        ctrl = DesktopController()
        ctrl._screen = ScreenInfo(width=1920, height=1080)

        with patch("alchemy.desktop.controller._send_right_click") as mock:
            result = await ctrl.right_click(500, 300)
            mock.assert_called_once_with(500, 300, 1920, 1080)
            assert "right_clicked" in result


class TestWin32Type:
    @pytest.mark.asyncio
    async def test_type_text_returns_string(self):
        ctrl = DesktopController()
        ctrl._screen = ScreenInfo(width=1920, height=1080)

        with patch("alchemy.desktop.controller._send_text") as mock:
            result = await ctrl.type_text("hello")
            mock.assert_called_once_with("hello")
            assert "typed" in result

    @pytest.mark.asyncio
    async def test_hotkey_returns_string(self):
        ctrl = DesktopController()
        ctrl._screen = ScreenInfo(width=1920, height=1080)

        with patch("alchemy.desktop.controller._send_hotkey") as mock:
            result = await ctrl.hotkey("ctrl", "c")
            mock.assert_called_once_with("ctrl", "c")
            assert "hotkey" in result


class TestWin32Scroll:
    @pytest.mark.asyncio
    async def test_scroll_down(self):
        ctrl = DesktopController()
        ctrl._screen = ScreenInfo(width=1920, height=1080)

        with patch("alchemy.desktop.controller._send_scroll") as mock:
            result = await ctrl.scroll(500, 300, "down", 5)
            mock.assert_called_once_with(500, 300, 1920, 1080, -5)
            assert "scrolled" in result

    @pytest.mark.asyncio
    async def test_scroll_up(self):
        ctrl = DesktopController()
        ctrl._screen = ScreenInfo(width=1920, height=1080)

        with patch("alchemy.desktop.controller._send_scroll") as mock:
            result = await ctrl.scroll(500, 300, "up", 3)
            mock.assert_called_once_with(500, 300, 1920, 1080, 3)
            assert "scrolled" in result


class TestShadowGhostMode:
    def test_default_mode_is_shadow(self):
        ctrl = DesktopController()
        assert ctrl.mode == "shadow"

    def test_explicit_shadow_mode(self):
        ctrl = DesktopController(mode="shadow")
        assert ctrl.mode == "shadow"
        assert ctrl._cursor is None

    def test_explicit_ghost_mode(self):
        ctrl = DesktopController(mode="ghost")
        assert ctrl.mode == "ghost"

    def test_summon_switches_to_ghost(self):
        ctrl = DesktopController(mode="shadow")
        ctrl._screen = ScreenInfo(width=1920, height=1080)
        with patch("alchemy.desktop.controller.DesktopController._start_cursor"):
            ctrl.summon()
        assert ctrl.mode == "ghost"

    def test_dismiss_switches_to_shadow(self):
        ctrl = DesktopController(mode="ghost")
        ctrl._screen = ScreenInfo(width=1920, height=1080)
        ctrl._cursor = MagicMock()
        ctrl.dismiss()
        assert ctrl.mode == "shadow"
        assert ctrl._cursor is None

    def test_summon_when_already_ghost_is_noop(self):
        ctrl = DesktopController(mode="ghost")
        ctrl.summon()  # Should not error
        assert ctrl.mode == "ghost"

    def test_dismiss_when_already_shadow_is_noop(self):
        ctrl = DesktopController(mode="shadow")
        ctrl.dismiss()  # Should not error
        assert ctrl.mode == "shadow"

    def test_shadow_mode_no_cursor_on_click(self):
        """In shadow mode, _cursor_click should be a no-op (cursor is None)."""
        ctrl = DesktopController(mode="shadow")
        ctrl._screen = ScreenInfo(width=1920, height=1080)
        ctrl._cursor_click(500, 300)  # Should not error


class TestVKMap:
    def test_common_keys(self):
        assert _VK_MAP["ctrl"] == 0x11
        assert _VK_MAP["alt"] == 0x12
        assert _VK_MAP["shift"] == 0x10
        assert _VK_MAP["enter"] == 0x0D
        assert _VK_MAP["escape"] == 0x1B
        assert _VK_MAP["tab"] == 0x09

    def test_letter_keys(self):
        assert _VK_MAP["a"] == ord("A")
        assert _VK_MAP["z"] == ord("Z")

    def test_number_keys(self):
        assert _VK_MAP["0"] == ord("0")
        assert _VK_MAP["9"] == ord("9")

    def test_function_keys(self):
        assert _VK_MAP["f1"] == 0x70
        assert _VK_MAP["f12"] == 0x7B
