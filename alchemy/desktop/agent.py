"""Desktop automation agent — vision-driven loop for native Windows apps.

Screenshot → Qwen2.5-VL 7B → action → repeat.

PROVEN WORKING (2026-03-05). Do NOT change the coordinate system or prompt
format without re-testing end-to-end on a live desktop.

Key facts (hard-won through debugging):
    - Qwen2.5-VL outputs IMAGE PIXEL coordinates, NOT 0-1000 normalized.
    - Coordinates are scaled: screen_x = point_x / image_width * screen_width
    - Must use SINGLE user message (Ollama VLMs crash with system role + images).
    - Short prompts give better coordinate accuracy than long UI-TARS templates.
    - Screenshot at 1280x720 for speed; scaling to real screen (1920x1080) is automatic.
    - num_ctx=8192 minimum (2048 causes garbage output with images).

Flow:
    1. pyautogui.screenshot() → JPEG bytes (1280x720)
    2. OllamaClient.chat(images=[bytes]) → 'Thought: ... Action: click {"point_2d": [x, y]}'
    3. Parse response → scale image pixel coords → screen pixels
    4. SendInput click(x, y) via ghost cursor
    5. Repeat until done or max_steps
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum

from alchemy.desktop.controller import DesktopController

logger = logging.getLogger(__name__)


# --- Prompt ---

# Minimal prompt for Qwen2.5-VL native grounding.
# Keep it short — long prompts degrade coordinate accuracy at 7B.
_SYSTEM_PROMPT = """\
You are a GUI agent. Look at the screenshot and perform the requested action.
Output format: Thought: <reasoning> Action: <action>
For clicks, output: Action: click {"point_2d": [x, y]}
For typing, output: Action: type "text"
For hotkeys, output: Action: hotkey key1+key2
For scrolling, output: Action: scroll up/down
When done, output: Action: done"""

_USER_PROMPT = """\
{context}
Task: {task}"""


# --- Response Parsing ---

_THOUGHT_RE = re.compile(r"Thought:\s*(.+?)(?=\nAction:|\Z)", re.DOTALL)
# Standard: click(start_box="(X,Y)") — also handles click@(X,Y) format from minicpm-v
_ACTION_RE = re.compile(r"Action:\s*(\w+)\((.*)\).*$", re.MULTILINE)
# Lenient: handles click@(436,15</box>) and click@(463,158)
_ACTION_AT_RE = re.compile(r"Action:\s*(\w+)@\((\d+)\s*,\s*(\d+)", re.MULTILINE)
_COORD_RE = re.compile(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)")
_CONTENT_RE = re.compile(r"""content=['"]((?:[^'"\\]|\\.)*?)['"]""")
_DIRECTION_RE = re.compile(r"""direction=['"]((?:[^'"\\]|\\.)*?)['"]""")
_KEY_RE = re.compile(r"""key=['"]((?:[^'"\\]|\\.)*?)['"]""")


class DesktopTaskStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class DesktopStep:
    """Record of one agent step."""
    step: int
    action_type: str
    x: int | None = None
    y: int | None = None
    text: str | None = None
    thought: str = ""
    inference_ms: float = 0.0
    execution_ms: float = 0.0
    success: bool = True
    error: str | None = None


@dataclass
class DesktopTaskResult:
    """Final result of a desktop automation task."""
    status: DesktopTaskStatus
    steps: list[DesktopStep] = field(default_factory=list)
    total_ms: float = 0.0
    error: str | None = None


class DesktopAgent:
    """Vision-driven desktop automation agent.

    Args:
        ollama_client: Shared OllamaClient for inference.
        controller: DesktopController for screenshots and actions.
        model: Vision model name (e.g., "minicpm-v").
        max_steps: Maximum steps before stopping.
        temperature: LLM temperature.
        max_tokens: Max output tokens per inference.
    """

    def __init__(
        self,
        ollama_client,
        controller: DesktopController,
        model: str = "qwen2.5vl:7b",
        max_steps: int = 20,
        temperature: float = 0.0,
        max_tokens: int = 384,
        num_ctx: int = 8192,
    ):
        self._ollama = ollama_client
        self._controller = controller
        self._model = model
        self._max_steps = max_steps
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._num_ctx = num_ctx

    async def run(self, task: str) -> DesktopTaskResult:
        """Execute a desktop automation task.

        Takes screenshots, asks the vision model what to do, executes actions,
        repeats until the model says finished() or max_steps is reached.
        """
        t0 = time.monotonic()
        steps: list[DesktopStep] = []
        recent_actions: list[str] = []

        for step_num in range(1, self._max_steps + 1):
            logger.info("Desktop step %d/%d — task: %s", step_num, self._max_steps, task[:80])

            # 1. Take screenshot
            try:
                screenshot_bytes = await self._controller.screenshot()
            except Exception as e:
                logger.error("Screenshot failed at step %d: %s", step_num, e)
                self._controller.park_cursor()
                return DesktopTaskResult(
                    status=DesktopTaskStatus.FAILED,
                    steps=steps,
                    total_ms=(time.monotonic() - t0) * 1000,
                    error=f"Screenshot failed: {e}",
                )

            # 2. Build prompt — MUST be single user message (VLMs crash with system+image)
            context_parts = []
            if recent_actions:
                context_parts.append("## Action History")
                for a in recent_actions[-5:]:
                    context_parts.append(f"- {a}")
            context = "\n".join(context_parts)

            user_prompt = _USER_PROMPT.format(
                task=task,
                context=context,
            )

            # Merge system + user into one message (Ollama VLMs crash with system role + images)
            full_prompt = _SYSTEM_PROMPT + "\n\n" + user_prompt

            # 3. Inference — single user message with screenshot attached
            messages = [
                {"role": "user", "content": full_prompt},
            ]
            t_infer = time.monotonic()

            try:
                response = await self._ollama.chat(
                    model=self._model,
                    messages=messages,
                    images=[screenshot_bytes],
                    options={
                        "temperature": self._temperature,
                        "num_predict": self._max_tokens,
                        "num_ctx": self._num_ctx,
                    },
                )
            except Exception as e:
                inference_ms = (time.monotonic() - t_infer) * 1000
                logger.error("Inference failed at step %d: %s", step_num, e)
                steps.append(DesktopStep(
                    step=step_num, action_type="error",
                    inference_ms=inference_ms, success=False, error=str(e),
                ))
                self._controller.park_cursor()
                return DesktopTaskResult(
                    status=DesktopTaskStatus.FAILED,
                    steps=steps,
                    total_ms=(time.monotonic() - t0) * 1000,
                    error=f"Inference failed: {e}",
                )

            inference_ms = (time.monotonic() - t_infer) * 1000
            raw_text = response.get("message", {}).get("content", "")
            logger.info("Step %d response (%.0fms): %s", step_num, inference_ms, raw_text[:200])

            # 4. Parse response
            try:
                action_type, x, y, text, direction, thought = _parse_response(raw_text)
            except ValueError as e:
                logger.warning("Parse failed at step %d: %s", step_num, e)
                steps.append(DesktopStep(
                    step=step_num, action_type="parse_error",
                    thought=raw_text[:200], inference_ms=inference_ms,
                    success=False, error=str(e),
                ))
                continue

            # 5. Check if done
            if action_type in ("done", "finished"):
                steps.append(DesktopStep(
                    step=step_num, action_type="done",
                    thought=thought, inference_ms=inference_ms,
                    text=text,
                ))
                self._controller.park_cursor()
                return DesktopTaskResult(
                    status=DesktopTaskStatus.COMPLETED,
                    steps=steps,
                    total_ms=(time.monotonic() - t0) * 1000,
                )

            # 6. Scale coordinates from image pixels → actual screen pixels
            #    Qwen2.5-VL outputs coords in image pixel space, not 0-1000.
            screen = self._controller.screen
            img_w = self._controller.image_width
            img_h = self._controller.image_height
            scaled_x, scaled_y = None, None
            if x is not None and y is not None:
                scaled_x = round(x / img_w * screen.width)
                scaled_y = round(y / img_h * screen.height)
                scaled_x = min(max(scaled_x, 0), screen.width)
                scaled_y = min(max(scaled_y, 0), screen.height)

            # 7. Execute action
            t_exec = time.monotonic()
            try:
                exec_result = await self._execute_action(
                    action_type, scaled_x, scaled_y, text, direction,
                )
                exec_ms = (time.monotonic() - t_exec) * 1000

                action_sig = f"{action_type}"
                if x is not None and y is not None:
                    action_sig += f" at ({x},{y})"
                if text:
                    action_sig += f" '{text[:30]}'"
                recent_actions.append(action_sig)

                steps.append(DesktopStep(
                    step=step_num, action_type=action_type,
                    x=scaled_x, y=scaled_y, text=text,
                    thought=thought, inference_ms=inference_ms,
                    execution_ms=exec_ms,
                ))

                logger.info("Step %d executed: %s (%.0fms)", step_num, exec_result, exec_ms)

            except Exception as e:
                exec_ms = (time.monotonic() - t_exec) * 1000
                logger.error("Execution failed at step %d: %s", step_num, e)
                steps.append(DesktopStep(
                    step=step_num, action_type=action_type,
                    x=scaled_x, y=scaled_y, text=text,
                    thought=thought, inference_ms=inference_ms,
                    execution_ms=exec_ms, success=False, error=str(e),
                ))

        # Max steps reached
        self._controller.park_cursor()
        return DesktopTaskResult(
            status=DesktopTaskStatus.FAILED,
            steps=steps,
            total_ms=(time.monotonic() - t0) * 1000,
            error=f"Max steps ({self._max_steps}) reached",
        )

    async def _execute_action(
        self,
        action_type: str,
        x: int | None,
        y: int | None,
        text: str | None,
        direction: str | None,
    ) -> str:
        """Dispatch a parsed action to the desktop controller."""
        ctrl = self._controller

        if action_type == "click" and x is not None and y is not None:
            return await ctrl.click(x, y)

        if action_type == "double_click" and x is not None and y is not None:
            return await ctrl.double_click(x, y)

        if action_type == "right_click" and x is not None and y is not None:
            return await ctrl.right_click(x, y)

        if action_type == "type" and text:
            return await ctrl.type_text(text)

        if action_type == "hotkey" and text:
            keys = [k.strip() for k in text.split("+")]
            return await ctrl.hotkey(*keys)

        if action_type == "scroll" and x is not None and y is not None:
            return await ctrl.scroll(x, y, direction or "down")

        if action_type == "wait":
            import asyncio
            await asyncio.sleep(1.0)
            return "waited 1s"

        return f"unknown action: {action_type}"


def _parse_response(raw: str) -> tuple[str, int | None, int | None, str | None, str | None, str]:
    """Parse vision model response.

    Returns:
        (action_type, x, y, text, direction, thought)

    Raises:
        ValueError: If no Action: line found.
    """
    thought = ""
    thought_match = _THOUGHT_RE.search(raw)
    if thought_match:
        thought = thought_match.group(1).strip()

    # Try standard format FIRST: Action: click(start_box="(X,Y)")
    action_match = _ACTION_RE.search(raw)
    if action_match:
        # Standard format found — skip JSON fallback, process below
        pass
    else:
        # Try alternate format: Action: click@(X,Y) — minicpm-v sometimes uses this
        at_match = _ACTION_AT_RE.search(raw)
        if at_match:
            action_type_raw = at_match.group(1)
            at_x = int(at_match.group(2))
            at_y = int(at_match.group(3))
            action_map = {
                "click": "click", "left_double": "double_click",
                "right_single": "right_click", "scroll": "scroll",
                "type": "type", "hotkey": "hotkey",
                "wait": "wait", "finished": "done",
            }
            return action_map.get(action_type_raw, action_type_raw), at_x, at_y, None, None, thought

        # Try Qwen2.5-VL native JSON point format as LAST fallback
        json_result = _parse_qwen_vl_json(raw, thought)
        if json_result:
            return json_result

        raise ValueError(f"No Action: found in: {raw[:200]!r}")

    action_type = action_match.group(1)
    args_str = action_match.group(2)

    # Map action names
    action_map = {
        "click": "click", "left_double": "double_click",
        "right_single": "right_click", "scroll": "scroll",
        "type": "type", "hotkey": "hotkey",
        "wait": "wait", "finished": "done",
    }
    action_type = action_map.get(action_type, action_type)

    # Extract coordinates
    x, y = None, None
    coord_match = _COORD_RE.search(args_str)
    if coord_match:
        x = int(coord_match.group(1))
        y = int(coord_match.group(2))

    # Extract text
    text = None
    content_match = _CONTENT_RE.search(args_str)
    if content_match:
        text = content_match.group(1).replace("\\'", "'").replace('\\"', '"')

    # Extract key (for hotkey)
    if text is None:
        key_match = _KEY_RE.search(args_str)
        if key_match:
            text = key_match.group(1)

    # Extract direction
    direction = None
    dir_match = _DIRECTION_RE.search(args_str)
    if dir_match:
        direction = dir_match.group(1)

    return action_type, x, y, text, direction, thought


# Qwen2.5-VL JSON: {"point_2d": [452, 128], "label": "search button"}
_POINT_JSON_RE = re.compile(r'\{[^{}]*"point_2d"\s*:\s*\[([^\]]+)\][^{}]*\}')


# Parse "Action: click/double_click/etc" before JSON or other content
_ACTION_NAME_RE = re.compile(r"Action:\s*(\w+)")
# Parse "Action: type "text"" or "Action: type 'text'"
_ACTION_TYPE_TEXT_RE = re.compile(r'Action:\s*type\s+["\'](.+?)["\']', re.MULTILINE)
# Parse "Action: hotkey key1+key2"
_ACTION_HOTKEY_RE = re.compile(r"Action:\s*hotkey\s+(.+?)$", re.MULTILINE)
# Parse "Action: scroll up/down"
_ACTION_SCROLL_RE = re.compile(r"Action:\s*scroll\s+(up|down|left|right)", re.MULTILINE)


def _parse_qwen_vl_json(
    raw: str, thought: str,
) -> tuple[str, int | None, int | None, str | None, str | None, str] | None:
    """Try to parse Qwen2.5-VL's native JSON point format.

    Returns (action_type, x, y, text, direction, thought) or None.
    """
    # Check for non-coordinate actions first
    action_name_match = _ACTION_NAME_RE.search(raw)
    action_name = action_name_match.group(1) if action_name_match else None

    # Handle "done" / "wait" (no coordinates needed)
    if action_name in ("done", "finished"):
        text = None
        content_match = _CONTENT_RE.search(raw)
        if content_match:
            text = content_match.group(1).replace("\\'", "'").replace('\\"', '"')
        return "done", None, None, text, None, thought
    if action_name == "wait":
        return "wait", None, None, None, None, thought

    # Handle "type" with text
    type_match = _ACTION_TYPE_TEXT_RE.search(raw)
    if type_match:
        return "type", None, None, type_match.group(1), None, thought

    # Handle "hotkey"
    hotkey_match = _ACTION_HOTKEY_RE.search(raw)
    if hotkey_match:
        return "hotkey", None, None, hotkey_match.group(1).strip(), None, thought

    # Handle "scroll"
    scroll_match = _ACTION_SCROLL_RE.search(raw)
    if scroll_match:
        return "scroll", None, None, None, scroll_match.group(1), thought

    # Try point_2d JSON
    match = _POINT_JSON_RE.search(raw)
    if not match:
        return None

    try:
        json_str = match.group(0)
        data = json.loads(json_str)
        coords = data.get("point_2d", [])
        label = data.get("label", "")

        if len(coords) < 2:
            return None

        x = int(float(coords[0]))
        y = int(float(coords[1]))

        # Determine action type from Action: line
        action_map = {
            "click": "click", "double_click": "double_click",
            "right_click": "right_click", "left_double": "double_click",
            "right_single": "right_click",
        }
        act = action_map.get(action_name, "click") if action_name else "click"
        return act, x, y, None, None, thought or f"[Qwen2.5-VL] {label}"
    except (json.JSONDecodeError, ValueError, TypeError):
        return None
