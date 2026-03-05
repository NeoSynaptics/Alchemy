"""AI cursor overlay — orange dot that glides independently of the real cursor.

Creates a transparent always-on-top window with an orange circle.
Runs in its own thread so it doesn't block the event loop.

Ghost mode: smooth, graceful movement so the user can follow along.
When idle, parks itself in the bottom-right corner above the clock.

Usage:
    cursor = AICursor()
    cursor.start()             # Spawns overlay thread
    cursor.move_to(500, 300)   # Smooth glide to position
    cursor.click_at(500, 300)  # Glide + flash animation
    cursor.park()              # Glide to resting position (bottom-right)
    cursor.stop()              # Destroy overlay
"""

from __future__ import annotations

import ctypes
import logging
import math
import threading
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Win32 constants
GWL_EXSTYLE = -20
WS_EX_LAYERED = 0x00080000
WS_EX_TRANSPARENT = 0x00000020
WS_EX_TOOLWINDOW = 0x00000080


@dataclass
class CursorConfig:
    """AI cursor appearance settings."""
    size: int = 24  # Diameter in pixels
    color: str = "#FF6600"  # Orange
    click_color: str = "#FFAA00"  # Brighter orange on click
    glide_duration: float = 0.4  # Seconds for smooth glide (ghost mode)
    glide_fps: int = 60  # Animation framerate
    click_flash_ms: int = 200  # Flash duration on click
    park_margin_x: int = 80  # Pixels from right edge when parked
    park_margin_y: int = 50  # Pixels from bottom edge when parked


class AICursor:
    """Orange AI cursor overlay — glides gracefully in ghost mode.

    The overlay window is:
    - Always on top
    - Click-through (transparent to mouse events)
    - No taskbar icon
    """

    def __init__(self, config: CursorConfig | None = None):
        self._config = config or CursorConfig()
        self._thread: threading.Thread | None = None
        self._root = None
        self._canvas = None
        self._dot = None
        self._running = False
        self._x = 0
        self._y = 0
        self._screen_w = 0
        self._screen_h = 0
        self._pending_moves: list[tuple[int, int]] = []
        self._pending_clicks: list[tuple[int, int]] = []
        self._pending_park = False
        self._lock = threading.Lock()

    def start(self) -> None:
        """Start the cursor overlay in a background thread."""
        if self._running:
            return
        self._running = True
        # Get screen size for park position
        user32 = ctypes.windll.user32
        self._screen_w = user32.GetSystemMetrics(0)
        self._screen_h = user32.GetSystemMetrics(1)
        self._thread = threading.Thread(target=self._run_tk, daemon=True)
        self._thread.start()
        # Wait for window to be created
        for _ in range(50):
            if self._root is not None:
                break
            time.sleep(0.02)
        logger.info("AI cursor overlay started (size=%d, color=%s)",
                     self._config.size, self._config.color)

    def stop(self) -> None:
        """Stop and destroy the overlay."""
        self._running = False
        if self._root:
            try:
                self._root.after(0, self._root.destroy)
            except Exception:
                pass

    def move_to(self, x: int, y: int) -> None:
        """Queue a smooth glide to (x, y)."""
        with self._lock:
            self._pending_moves.append((x, y))

    def click_at(self, x: int, y: int) -> None:
        """Queue a glide + click flash at (x, y)."""
        with self._lock:
            self._pending_clicks.append((x, y))

    def park(self) -> None:
        """Glide to resting position (bottom-right corner, above the clock)."""
        with self._lock:
            self._pending_park = True

    def _run_tk(self) -> None:
        """Main tkinter loop (runs in background thread)."""
        import tkinter as tk

        root = tk.Tk()
        root.title("")
        size = self._config.size

        # Start parked in the bottom-right corner
        park_x = self._screen_w - self._config.park_margin_x
        park_y = self._screen_h - self._config.park_margin_y
        self._x = park_x
        self._y = park_y

        root.geometry(f"{size}x{size}+{park_x}+{park_y}")
        root.overrideredirect(True)  # No title bar
        root.attributes("-topmost", True)
        root.attributes("-alpha", 0.85)

        # Make the background transparent (click-through)
        bg_color = "#010101"  # Near-black, used as transparency key
        root.configure(bg=bg_color)
        root.wm_attributes("-transparentcolor", bg_color)

        canvas = tk.Canvas(root, width=size, height=size,
                           bg=bg_color, highlightthickness=0)
        canvas.pack()

        # Draw the orange dot
        pad = 2
        dot = canvas.create_oval(
            pad, pad, size - pad, size - pad,
            fill=self._config.color, outline=self._config.color,
        )

        self._root = root
        self._canvas = canvas
        self._dot = dot

        # Make window click-through via win32
        self._make_click_through(root)

        # Start update loop
        root.after(16, self._update_loop)
        root.mainloop()

    def _make_click_through(self, root) -> None:
        """Make the window click-through using win32 extended styles."""
        try:
            hwnd = ctypes.windll.user32.GetParent(root.winfo_id())
            if hwnd == 0:
                hwnd = root.winfo_id()
            style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            new_style = style | WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOOLWINDOW
            ctypes.windll.user32.SetWindowLongW(hwnd, GWL_EXSTYLE, new_style)
        except Exception as e:
            logger.warning("Could not make cursor click-through: %s", e)

    def _update_loop(self) -> None:
        """Process pending moves and clicks at ~60fps."""
        if not self._running:
            return

        with self._lock:
            moves = self._pending_moves[:]
            self._pending_moves.clear()
            clicks = self._pending_clicks[:]
            self._pending_clicks.clear()
            should_park = self._pending_park
            self._pending_park = False

        # Process moves (smooth glide)
        for x, y in moves:
            self._glide_to(x, y)

        # Process clicks (glide + flash)
        for x, y in clicks:
            self._glide_to(x, y)
            self._flash_click()

        # Park in corner
        if should_park:
            park_x = self._screen_w - self._config.park_margin_x
            park_y = self._screen_h - self._config.park_margin_y
            self._glide_to(park_x, park_y)

        # Schedule next update
        if self._root and self._running:
            self._root.after(16, self._update_loop)

    def _glide_to(self, target_x: int, target_y: int) -> None:
        """Smooth ease-out glide from current position to target."""
        if not self._root:
            return

        start_x = self._x
        start_y = self._y
        dx = target_x - start_x
        dy = target_y - start_y
        distance = math.sqrt(dx * dx + dy * dy)

        # Skip animation for tiny moves
        if distance < 5:
            self._set_position(target_x, target_y)
            return

        # Scale duration by distance (min 0.15s, max glide_duration)
        duration = min(self._config.glide_duration, max(0.15, distance / 3000))
        steps = max(1, int(duration * self._config.glide_fps))
        step_delay = duration / steps

        for i in range(1, steps + 1):
            t = i / steps
            # Ease-out cubic: starts fast, slows down at the end
            ease = 1 - (1 - t) ** 3
            cx = int(start_x + dx * ease)
            cy = int(start_y + dy * ease)
            self._set_position(cx, cy)
            self._root.update()
            time.sleep(step_delay)

        # Ensure exact final position
        self._set_position(target_x, target_y)

    def _set_position(self, x: int, y: int) -> None:
        """Instantly set overlay position (centered on dot)."""
        if not self._root:
            return
        half = self._config.size // 2
        self._root.geometry(f"+{x - half}+{y - half}")
        self._x = x
        self._y = y

    def _flash_click(self) -> None:
        """Briefly change dot color + size to indicate a click."""
        if not self._canvas or not self._dot:
            return
        # Bright flash
        self._canvas.itemconfig(self._dot, fill=self._config.click_color)
        self._root.update()
        time.sleep(self._config.click_flash_ms / 1000)
        # Back to normal
        self._canvas.itemconfig(self._dot, fill=self._config.color)
        if self._root:
            self._root.update()
