"""Execute VisionAction on the shadow desktop via xdotool.

Translates structured VisionAction objects into xdotool shell commands
and runs them through ShadowDesktopController.execute().
"""

from __future__ import annotations

import asyncio
import logging

from alchemy.schemas import VisionAction
from alchemy.shadow.controller import ShadowDesktopController

logger = logging.getLogger(__name__)


class ActionExecutor:
    """Convert VisionAction to xdotool commands and execute on shadow desktop."""

    def __init__(self, controller: ShadowDesktopController):
        self._controller = controller

    async def execute(self, action: VisionAction) -> str:
        """Execute a VisionAction. Returns command output."""
        logger.info("Executing: %s at (%s,%s)", action.action, action.x, action.y)

        match action.action:
            case "click":
                return await self._click(action.x, action.y)
            case "double_click":
                return await self._double_click(action.x, action.y)
            case "right_click":
                return await self._right_click(action.x, action.y)
            case "drag":
                return await self._drag(action.x, action.y, action.end_x, action.end_y)
            case "type":
                return await self._type_text(action.text or "")
            case "hotkey":
                return await self._hotkey(action.text or "")
            case "scroll":
                return await self._scroll(
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

    async def _click(self, x: int, y: int) -> str:
        return await self._controller.execute(
            f"xdotool mousemove --sync {x} {y} && xdotool click 1"
        )

    async def _double_click(self, x: int, y: int) -> str:
        return await self._controller.execute(
            f"xdotool mousemove --sync {x} {y} && xdotool click --repeat 2 --delay 100 1"
        )

    async def _right_click(self, x: int, y: int) -> str:
        return await self._controller.execute(
            f"xdotool mousemove --sync {x} {y} && xdotool click 3"
        )

    async def _drag(self, x1: int, y1: int, x2: int, y2: int) -> str:
        return await self._controller.execute(
            f"xdotool mousemove --sync {x1} {y1} && "
            f"xdotool mousedown 1 && "
            f"xdotool mousemove --sync {x2} {y2} && "
            f"xdotool mouseup 1"
        )

    async def _type_text(self, text: str) -> str:
        safe = text.replace("'", "'\\''")
        return await self._controller.execute(
            f"xdotool type --clearmodifiers --delay 50 '{safe}'"
        )

    async def _hotkey(self, key_combo: str) -> str:
        return await self._controller.execute(
            f"xdotool key --clearmodifiers {key_combo}"
        )

    async def _scroll(self, x: int, y: int, direction: str, amount: int) -> str:
        button = {"up": 4, "down": 5, "left": 6, "right": 7}.get(direction, 5)
        return await self._controller.execute(
            f"xdotool mousemove --sync {x} {y} && "
            f"xdotool click --repeat {amount} --delay 100 {button}"
        )
