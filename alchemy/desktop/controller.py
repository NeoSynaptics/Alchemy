"""Windows desktop controller — screenshot capture + invisible mouse/keyboard.

Two modes of operation:
  - Shadow (default): Completely invisible. SendInput clicks + no cursor overlay.
    The agent works silently in the background — user sees nothing.
  - Ghost (summoned): Orange AI cursor overlay appears. SendInput clicks still
    invisible to the real cursor, but the AI cursor shows what the agent is doing.

Input always uses win32api/ctypes SendInput — the user's real mouse is never moved.
"""

from __future__ import annotations

import asyncio
import ctypes
import io
import logging
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# --- Win32 constants for SendInput ---
INPUT_MOUSE = 0
INPUT_KEYBOARD = 1
MOUSEEVENTF_ABSOLUTE = 0x8000
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
MOUSEEVENTF_WHEEL = 0x0800
WHEEL_DELTA = 120
KEYEVENTF_UNICODE = 0x0004
KEYEVENTF_KEYUP = 0x0002


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.c_long),
        ("dy", ctypes.c_long),
        ("mouseData", ctypes.c_ulong),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", ctypes.c_ushort),
        ("wScan", ctypes.c_ushort),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]


class _INPUT_UNION(ctypes.Union):
    _fields_ = [("mi", MOUSEINPUT), ("ki", KEYBDINPUT)]


class INPUT(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong), ("union", _INPUT_UNION)]


@dataclass
class ScreenInfo:
    """Current screen resolution."""
    width: int
    height: int


def _screen_size() -> tuple[int, int]:
    """Get primary screen resolution via win32api."""
    user32 = ctypes.windll.user32
    return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)


def _to_absolute(x: int, y: int, screen_w: int, screen_h: int) -> tuple[int, int]:
    """Convert pixel coords to win32 absolute coords (0-65535 range)."""
    abs_x = int(x * 65536 / screen_w)
    abs_y = int(y * 65536 / screen_h)
    return abs_x, abs_y


class POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


def _save_cursor() -> tuple[int, int]:
    """Save the user's current cursor position."""
    pt = POINT()
    ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
    return pt.x, pt.y


def _restore_cursor(x: int, y: int) -> None:
    """Restore the user's cursor to its saved position."""
    ctypes.windll.user32.SetCursorPos(x, y)


def _send_click(x: int, y: int, screen_w: int, screen_h: int) -> str:
    """Send a left-click at (x, y) — user's cursor stays in place."""
    saved = _save_cursor()
    abs_x, abs_y = _to_absolute(x, y, screen_w, screen_h)

    inputs = (INPUT * 3)()

    # Move to position
    inputs[0].type = INPUT_MOUSE
    inputs[0].union.mi.dx = abs_x
    inputs[0].union.mi.dy = abs_y
    inputs[0].union.mi.dwFlags = MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE

    # Left button down
    inputs[1].type = INPUT_MOUSE
    inputs[1].union.mi.dx = abs_x
    inputs[1].union.mi.dy = abs_y
    inputs[1].union.mi.dwFlags = MOUSEEVENTF_LEFTDOWN | MOUSEEVENTF_ABSOLUTE

    # Left button up
    inputs[2].type = INPUT_MOUSE
    inputs[2].union.mi.dx = abs_x
    inputs[2].union.mi.dy = abs_y
    inputs[2].union.mi.dwFlags = MOUSEEVENTF_LEFTUP | MOUSEEVENTF_ABSOLUTE

    ctypes.windll.user32.SendInput(3, ctypes.pointer(inputs[0]), ctypes.sizeof(INPUT))
    _restore_cursor(*saved)
    return f"clicked ({x}, {y})"


def _send_right_click(x: int, y: int, screen_w: int, screen_h: int) -> None:
    """Send a right-click at (x, y) — user's cursor stays in place."""
    saved = _save_cursor()
    abs_x, abs_y = _to_absolute(x, y, screen_w, screen_h)

    inputs = (INPUT * 3)()
    inputs[0].type = INPUT_MOUSE
    inputs[0].union.mi.dx = abs_x
    inputs[0].union.mi.dy = abs_y
    inputs[0].union.mi.dwFlags = MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE

    inputs[1].type = INPUT_MOUSE
    inputs[1].union.mi.dx = abs_x
    inputs[1].union.mi.dy = abs_y
    inputs[1].union.mi.dwFlags = MOUSEEVENTF_RIGHTDOWN | MOUSEEVENTF_ABSOLUTE

    inputs[2].type = INPUT_MOUSE
    inputs[2].union.mi.dx = abs_x
    inputs[2].union.mi.dy = abs_y
    inputs[2].union.mi.dwFlags = MOUSEEVENTF_RIGHTUP | MOUSEEVENTF_ABSOLUTE

    ctypes.windll.user32.SendInput(3, ctypes.pointer(inputs[0]), ctypes.sizeof(INPUT))
    _restore_cursor(*saved)


def _send_double_click(x: int, y: int, screen_w: int, screen_h: int) -> None:
    """Send a double-click at (x, y)."""
    _send_click(x, y, screen_w, screen_h)
    time.sleep(0.05)
    _send_click(x, y, screen_w, screen_h)


def _send_scroll(x: int, y: int, screen_w: int, screen_h: int, amount: int) -> None:
    """Send scroll wheel events — user's cursor stays in place."""
    saved = _save_cursor()
    abs_x, abs_y = _to_absolute(x, y, screen_w, screen_h)

    inputs = (INPUT * 2)()
    # Move to position first
    inputs[0].type = INPUT_MOUSE
    inputs[0].union.mi.dx = abs_x
    inputs[0].union.mi.dy = abs_y
    inputs[0].union.mi.dwFlags = MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE

    # Scroll
    inputs[1].type = INPUT_MOUSE
    inputs[1].union.mi.dx = abs_x
    inputs[1].union.mi.dy = abs_y
    inputs[1].union.mi.mouseData = ctypes.c_ulong(amount * WHEEL_DELTA & 0xFFFFFFFF)
    inputs[1].union.mi.dwFlags = MOUSEEVENTF_WHEEL | MOUSEEVENTF_ABSOLUTE

    ctypes.windll.user32.SendInput(2, ctypes.pointer(inputs[0]), ctypes.sizeof(INPUT))
    _restore_cursor(*saved)


def _send_text(text: str) -> None:
    """Type text using SendInput with Unicode characters."""
    for char in text:
        code = ord(char)
        inputs = (INPUT * 2)()

        # Key down
        inputs[0].type = INPUT_KEYBOARD
        inputs[0].union.ki.wScan = code
        inputs[0].union.ki.dwFlags = KEYEVENTF_UNICODE

        # Key up
        inputs[1].type = INPUT_KEYBOARD
        inputs[1].union.ki.wScan = code
        inputs[1].union.ki.dwFlags = KEYEVENTF_UNICODE | KEYEVENTF_KEYUP

        ctypes.windll.user32.SendInput(2, ctypes.pointer(inputs[0]), ctypes.sizeof(INPUT))
        time.sleep(0.02)


# VK codes for common modifier keys
_VK_MAP = {
    "ctrl": 0x11, "control": 0x11,
    "alt": 0x12, "menu": 0x12,
    "shift": 0x10,
    "win": 0x5B, "lwin": 0x5B,
    "tab": 0x09, "enter": 0x0D, "return": 0x0D,
    "escape": 0x1B, "esc": 0x1B,
    "backspace": 0x08, "delete": 0x2E,
    "up": 0x26, "down": 0x28, "left": 0x25, "right": 0x27,
    "home": 0x24, "end": 0x23,
    "pageup": 0x21, "pagedown": 0x22,
    "f1": 0x70, "f2": 0x71, "f3": 0x72, "f4": 0x73,
    "f5": 0x74, "f6": 0x75, "f7": 0x76, "f8": 0x77,
    "f9": 0x78, "f10": 0x79, "f11": 0x7A, "f12": 0x7B,
    "space": 0x20,
}

# Single letter/number VK codes
for c in "abcdefghijklmnopqrstuvwxyz":
    _VK_MAP[c] = ord(c.upper())
for c in "0123456789":
    _VK_MAP[c] = ord(c)


def _send_hotkey(*keys: str) -> None:
    """Press a key combination (e.g., ctrl+c)."""
    vk_codes = [_VK_MAP.get(k.lower(), 0) for k in keys]

    # Press all keys down
    for vk in vk_codes:
        if vk == 0:
            continue
        inp = INPUT()
        inp.type = INPUT_KEYBOARD
        inp.union.ki.wVk = vk
        ctypes.windll.user32.SendInput(1, ctypes.pointer(inp), ctypes.sizeof(INPUT))

    # Release all keys (reverse order)
    for vk in reversed(vk_codes):
        if vk == 0:
            continue
        inp = INPUT()
        inp.type = INPUT_KEYBOARD
        inp.union.ki.wVk = vk
        inp.union.ki.dwFlags = KEYEVENTF_KEYUP
        ctypes.windll.user32.SendInput(1, ctypes.pointer(inp), ctypes.sizeof(INPUT))


class DesktopController:
    """Controls the native Windows desktop.

    Screenshots via pyautogui (PIL), input via win32 SendInput (invisible).
    Orange AI cursor overlay appears only in ghost mode.

    Modes:
        shadow: Completely invisible — no cursor overlay. Default.
        ghost:  Orange AI cursor shows what the agent is doing.

    Args:
        screenshot_width: Resize width (0 = no resize, send full resolution).
        screenshot_height: Resize height (0 = no resize).
        screenshot_quality: JPEG quality (0-100).
        mode: "shadow" (invisible) or "ghost" (orange cursor).
    """

    def __init__(
        self,
        screenshot_width: int = 0,
        screenshot_height: int = 0,
        screenshot_quality: int = 85,
        mode: str = "shadow",
    ):
        self._screenshot_width = screenshot_width
        self._screenshot_height = screenshot_height
        self._screenshot_quality = screenshot_quality
        self._mode = mode

        # Lazy import — only when screenshot is needed
        self._pyautogui = None
        self._screen: ScreenInfo | None = None
        self._cursor = None  # AICursor, started only in ghost mode

    @property
    def mode(self) -> str:
        """Current mode: 'shadow' or 'ghost'."""
        return self._mode

    def summon(self) -> None:
        """Switch to ghost mode — show the orange AI cursor."""
        if self._mode == "ghost":
            return
        self._mode = "ghost"
        logger.info("Desktop agent summoned — ghost mode (orange cursor visible)")
        if self._screen is not None and self._cursor is None:
            self._start_cursor()

    def dismiss(self) -> None:
        """Switch back to shadow mode — hide the orange AI cursor."""
        if self._mode == "shadow":
            return
        self._mode = "shadow"
        logger.info("Desktop agent dismissed — shadow mode (invisible)")
        if self._cursor is not None:
            self._cursor.stop()
            self._cursor = None

    def _start_cursor(self) -> None:
        """Start the AI cursor overlay."""
        try:
            from alchemy.desktop.cursor import AICursor
            self._cursor = AICursor()
            self._cursor.start()
        except Exception as e:
            logger.warning("AI cursor overlay failed: %s", e)
            self._cursor = None

    def _ensure_init(self):
        """Lazy-load pyautogui for screenshots + get screen size."""
        if self._screen is None:
            w, h = _screen_size()
            self._screen = ScreenInfo(width=w, height=h)
            logger.info(
                "Desktop controller ready — screen %dx%d, resize %s, mode %s",
                w, h,
                f"{self._screenshot_width}x{self._screenshot_height}"
                if self._screenshot_width > 0 else "off (full res)",
                self._mode,
            )
            # Start AI cursor overlay only in ghost mode
            if self._mode == "ghost":
                self._start_cursor()

    def _ensure_pyautogui(self):
        """Lazy-load pyautogui (only needed for screenshots)."""
        if self._pyautogui is None:
            import pyautogui
            pyautogui.FAILSAFE = False  # We don't move the cursor, no need
            self._pyautogui = pyautogui
        return self._pyautogui

    @property
    def screen(self) -> ScreenInfo:
        """Get screen resolution."""
        self._ensure_init()
        return self._screen  # type: ignore[return-value]

    @property
    def image_width(self) -> int:
        """Width of images sent to the vision model."""
        if self._screenshot_width > 0:
            return self._screenshot_width
        return self.screen.width

    @property
    def image_height(self) -> int:
        """Height of images sent to the vision model."""
        if self._screenshot_height > 0:
            return self._screenshot_height
        return self.screen.height

    async def screenshot(self) -> bytes:
        """Capture the Windows desktop as JPEG bytes."""
        return await asyncio.to_thread(self._screenshot_sync)

    def _screenshot_sync(self) -> bytes:
        """Synchronous screenshot capture + optional resize + JPEG encode."""
        self._ensure_init()
        pag = self._ensure_pyautogui()
        img = pag.screenshot()

        # Optional resize (0 = send full resolution)
        if self._screenshot_width > 0 and self._screenshot_height > 0:
            img = img.resize(
                (self._screenshot_width, self._screenshot_height),
            )

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=self._screenshot_quality)
        return buf.getvalue()

    def _cursor_click(self, x: int, y: int) -> None:
        """Move the AI cursor to position and flash click animation."""
        if self._cursor:
            self._cursor.click_at(x, y)

    def _cursor_move(self, x: int, y: int) -> None:
        """Move the AI cursor to position."""
        if self._cursor:
            self._cursor.move_to(x, y)

    def park_cursor(self) -> None:
        """Glide the AI cursor to its resting position (bottom-right corner)."""
        if self._cursor:
            self._cursor.park()

    async def click(self, x: int, y: int) -> str:
        """Click at screen pixel coordinates. Moves the AI cursor, not yours."""
        self._ensure_init()
        self._cursor_click(x, y)
        return await asyncio.to_thread(
            _send_click, x, y, self._screen.width, self._screen.height,
        )  # type: ignore[union-attr]

    async def double_click(self, x: int, y: int) -> str:
        """Double-click at screen pixel coordinates."""
        self._ensure_init()
        self._cursor_click(x, y)
        await asyncio.to_thread(
            _send_double_click, x, y, self._screen.width, self._screen.height,
        )
        return f"double_clicked ({x}, {y})"

    async def right_click(self, x: int, y: int) -> str:
        """Right-click at screen pixel coordinates."""
        self._ensure_init()
        self._cursor_click(x, y)
        await asyncio.to_thread(
            _send_right_click, x, y, self._screen.width, self._screen.height,
        )
        return f"right_clicked ({x}, {y})"

    async def type_text(self, text: str) -> str:
        """Type text using SendInput (Unicode, works in any app)."""
        await asyncio.to_thread(_send_text, text)
        return f"typed {text!r}"

    async def hotkey(self, *keys: str) -> str:
        """Press a hotkey combination (e.g., 'ctrl', 'c')."""
        await asyncio.to_thread(_send_hotkey, *keys)
        return f"hotkey {'+'.join(keys)}"

    async def scroll(self, x: int, y: int, direction: str = "down", amount: int = 3) -> str:
        """Scroll at position."""
        self._ensure_init()
        self._cursor_move(x, y)
        scroll_amount = -amount if direction == "down" else amount
        await asyncio.to_thread(
            _send_scroll, x, y, self._screen.width, self._screen.height, scroll_amount,
        )
        return f"scrolled {direction} {amount} at ({x}, {y})"

    async def move_to(self, x: int, y: int) -> str:
        """Move mouse to position (this one DOES move the visible cursor)."""
        self._ensure_init()
        abs_x, abs_y = _to_absolute(x, y, self._screen.width, self._screen.height)

        inp = INPUT()
        inp.type = INPUT_MOUSE
        inp.union.mi.dx = abs_x
        inp.union.mi.dy = abs_y
        inp.union.mi.dwFlags = MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE

        await asyncio.to_thread(
            ctypes.windll.user32.SendInput, 1, ctypes.pointer(inp), ctypes.sizeof(INPUT),
        )
        return f"moved to ({x}, {y})"
