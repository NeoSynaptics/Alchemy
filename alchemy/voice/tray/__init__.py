"""System tray widget — PyQt6 tray icon + noVNC viewport + approval dialogs.

Lazy imports to avoid ImportError when PyQt6 is not installed.
"""

from __future__ import annotations

from alchemy.voice.tray.events import TrayEvent, TrayEventBus, TrayMessage

__all__ = ["TrayEvent", "TrayEventBus", "TrayMessage", "TrayManager"]


def __getattr__(name: str):
    if name == "TrayManager":
        from alchemy.voice.tray.app import TrayManager

        return TrayManager
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
