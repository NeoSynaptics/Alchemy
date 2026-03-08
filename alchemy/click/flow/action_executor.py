"""Execute VisionAction on the native Windows desktop.

Translates structured VisionAction objects into DesktopController calls
(SendInput-based, invisible to the user's real cursor).
"""

from __future__ import annotations

import asyncio
import logging

from alchemy.schemas import VisionAction

logger = logging.getLogger(__name__)


class ActionExecutor:
    """Convert VisionAction to native Windows desktop actions."""

    def __init__(self, controller):
        """Accept any controller with click/type/hotkey/scroll/screenshot methods."""
        self._controller = controller

    async def execute(self, action: VisionAction) -> str:
        """Execute a VisionAction. Returns result string."""
        logger.info("Executing: %s at (%s,%s)", action.action, action.x, action.y)

        match action.action:
            case "click":
                return await self._controller.click(action.x, action.y)
            case "double_click":
                return await self._controller.double_click(action.x, action.y)
            case "right_click":
                return await self._controller.right_click(action.x, action.y)
            case "drag":
                # Proper drag: mousedown at start, move to end, mouseup
                return await self._controller.drag(
                    action.x, action.y, action.end_x, action.end_y
                )
            case "type":
                return await self._controller.type_text(action.text or "")
            case "hotkey":
                keys = (action.text or "").split()
                return await self._controller.hotkey(*keys)
            case "scroll":
                return await self._controller.scroll(
                    action.x or 0, action.y or 0,
                    action.direction or "down", action.amount or 3,
                )
            case "wait":
                await asyncio.sleep(1.0)
                return "waited"
            case "done" | "fail":
                return ""
            case _:
                raise ValueError(f"Unknown action: {action.action}")
