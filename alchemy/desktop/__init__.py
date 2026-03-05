"""Desktop automation module — vision-driven control of native Windows apps.

Uses UI-TARS 7B (or any vision model) to see the screen and win32 SendInput
for invisible clicks/typing. Orange AI cursor overlay runs parallel to the
user's real cursor. Reuses the shared OllamaClient from core.
"""

from alchemy.desktop.agent import DesktopAgent, DesktopStep, DesktopTaskResult, DesktopTaskStatus
from alchemy.desktop.controller import DesktopController, ScreenInfo
from alchemy.desktop.cursor import AICursor, CursorConfig

__all__ = [
    "AICursor",
    "CursorConfig",
    "DesktopAgent",
    "DesktopController",
    "DesktopStep",
    "DesktopTaskResult",
    "DesktopTaskStatus",
    "ScreenInfo",
]
