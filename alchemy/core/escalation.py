"""Tier 1.5 escalation — vision model fallback for stuck Playwright agent.

When Qwen3 14B gets overwhelmed by a complex page (parse failures, action loops,
too many refs), we take a screenshot and ask the vision model (Qwen2.5-VL 7B)
to visually identify the next action. One shot, then hand control back to Qwen3.

Supports multiple vision model output formats:
    - Standard: Thought: ... Action: click(start_box="(X,Y)")
    - Qwen2.5-VL native: {"point_2d": [X, Y]} JSON format
    - minicpm-v: click@(X,Y) format

Flow:
    Qwen3 stuck → screenshot → Qwen2.5-VL 7B → click(x, y) → Qwen3 resumes
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


# --- Stuck Detection ---


class StuckReason(str, Enum):
    """Why the agent is considered stuck."""

    PARSE_FAILURES = "parse_failures"
    ACTION_LOOP = "action_loop"
    COMPLEXITY = "complexity"


@dataclass
class StuckDetector:
    """Tracks agent behavior to detect when it's stuck.

    Args:
        max_parse_failures: Consecutive parse failures before escalating.
        max_repeated_actions: Same action repeated N times = loop.
        complexity_threshold: Ref count above which we preemptively escalate.
    """

    max_parse_failures: int = 3
    max_repeated_actions: int = 3
    complexity_threshold: int = 60

    # Internal state
    _consecutive_parse_failures: int = field(default=0, init=False)
    _recent_actions: list[str] = field(default_factory=list, init=False)

    def record_parse_failure(self) -> None:
        """Record a parse failure (LLM output had no valid Action: line)."""
        self._consecutive_parse_failures += 1

    def record_success(self, action_signature: str) -> None:
        """Record a successful parse + execution.

        Args:
            action_signature: e.g. "click@e5" or "scroll_down"
        """
        self._consecutive_parse_failures = 0
        self._recent_actions.append(action_signature)
        # Keep only recent window
        if len(self._recent_actions) > self.max_repeated_actions + 2:
            self._recent_actions = self._recent_actions[-(self.max_repeated_actions + 2):]

    def check(self, ref_count: int = 0) -> StuckReason | None:
        """Check if the agent is stuck. Returns reason or None."""
        # Too many parse failures in a row
        if self._consecutive_parse_failures >= self.max_parse_failures:
            return StuckReason.PARSE_FAILURES

        # Same action repeated N times (loop detection)
        if len(self._recent_actions) >= self.max_repeated_actions:
            tail = self._recent_actions[-self.max_repeated_actions:]
            if len(set(tail)) == 1:
                return StuckReason.ACTION_LOOP

        # Page complexity exceeds model capability
        if ref_count > self.complexity_threshold:
            return StuckReason.COMPLEXITY

        return None

    def reset(self) -> None:
        """Reset after a successful escalation."""
        self._consecutive_parse_failures = 0
        self._recent_actions.clear()


# --- Vision Escalation Prompt ---

_ESCALATION_PROMPT = """\
You are a GUI automation agent. You look at screenshots and perform actions.

Task: {task}
{context}
The screenshot is {width}x{height} pixels.

IMPORTANT RULES:
- Do NOT repeat the same action. If a click was already performed, do something DIFFERENT next (like type, scroll, or click a different element).
- After clicking a search box or input field, the next action should be type() to enter text.
- Think step by step: what has been done → what needs to happen next.

Action Space:
click(start_box="(X,Y)")  — click an element at pixel position
type(content="text to type")  — type text into the focused element
scroll(start_box="(X,Y)", direction="down or up")  — scroll the page
wait()  — wait for page to load
finished(content="done")  — task is complete

Where X is horizontal pixel (0-{width}) and Y is vertical pixel (0-{height}).

Reply with exactly:
Thought: [what was already done, what you see, what to do next]
Action: [one action]"""


@dataclass
class EscalationResult:
    """Result of a UI-TARS 7B escalation attempt."""

    success: bool
    action_type: str = ""
    x: int | None = None
    y: int | None = None
    text: str | None = None
    direction: str | None = None
    thought: str = ""
    inference_ms: float = 0.0
    error: str | None = None


class VisionEscalation:
    """UI-TARS 7B single-shot visual fallback.

    Takes a Playwright screenshot and asks UI-TARS 7B to identify
    the next action. Returns coordinates for a mouse click (or other action).

    Args:
        ollama_client: OllamaClient for inference.
        model: Vision model name (e.g., UI-TARS 7B GGUF).
        temperature: LLM temperature for the vision model.
        max_tokens: Max output tokens.
        screen_width: Page viewport width (for coordinate mapping).
        screen_height: Page viewport height (for coordinate mapping).
    """

    def __init__(
        self,
        ollama_client,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 384,
        screen_width: int = 1280,
        screen_height: int = 720,
        click_dedup_radius: int = 50,
    ):
        self._ollama = ollama_client
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._screen_width = screen_width
        self._screen_height = screen_height
        self._click_dedup_radius = click_dedup_radius
        self._recent_clicks: list[tuple[int, int]] = []  # (x, y) history

    async def escalate(
        self,
        page,
        task: str,
        recent_actions: list[str] | None = None,
        reason: StuckReason | None = None,
    ) -> EscalationResult:
        """Take a screenshot and ask UI-TARS 7B for the next action.

        Args:
            page: Playwright Page object.
            task: The user's task description.
            recent_actions: Recent action history for context.
            reason: Why escalation was triggered.

        Returns:
            EscalationResult with the suggested action.
        """
        logger.info(
            "Escalating to vision model (%s) — reason: %s",
            self._model, reason or "manual",
        )

        # 1. Take screenshot
        try:
            screenshot_bytes = await page.screenshot(type="jpeg", quality=85)
        except Exception as e:
            logger.error("Screenshot failed during escalation: %s", e)
            return EscalationResult(success=False, error=f"Screenshot failed: {e}")

        # 2. Build prompt with context
        context_parts = []
        if reason:
            context_parts.append(f"Note: The previous agent got stuck ({reason.value}).")
        if recent_actions:
            context_parts.append("Recent actions:")
            for a in recent_actions[-5:]:
                context_parts.append(f"  {a}")

        context = "\n".join(context_parts)
        if context:
            context = f"\n{context}\n"

        prompt = _ESCALATION_PROMPT.format(
            task=task, context=context,
            width=self._screen_width, height=self._screen_height,
        )

        # 3. Call UI-TARS 7B via Ollama
        messages = [{"role": "user", "content": prompt}]

        t0 = time.monotonic()
        try:
            response = await self._ollama.chat(
                model=self._model,
                messages=messages,
                images=[screenshot_bytes],
                options={
                    "temperature": self._temperature,
                    "num_predict": self._max_tokens,
                },
            )
        except Exception as e:
            inference_ms = (time.monotonic() - t0) * 1000
            logger.error("Vision model inference failed: %s", e)
            return EscalationResult(
                success=False, error=f"Inference failed: {e}",
                inference_ms=inference_ms,
            )

        inference_ms = (time.monotonic() - t0) * 1000
        raw_text = response.get("message", {}).get("content", "")

        logger.info("Vision model response (%.0fms): %s", inference_ms, raw_text[:200])

        # 4. Parse UI-TARS response into an action
        try:
            result = _parse_escalation_response(
                raw_text, self._screen_width, self._screen_height,
            )
            result.inference_ms = inference_ms

            # 5. Click-dedup: if model clicks same spot again, auto-type instead
            result = self._apply_click_dedup(result, task)

            return result
        except ValueError as e:
            logger.warning("Could not parse vision model response: %s", e)
            return EscalationResult(
                success=False, error=f"Parse error: {e}",
                inference_ms=inference_ms,
            )


    def _apply_click_dedup(self, result: EscalationResult, task: str) -> EscalationResult:
        """Convert repeated clicks at same location to a type action.

        If the vision model clicks within `click_dedup_radius` pixels of a
        previous click, we assume it clicked an input field. Extract the
        search/type text from the task and return a type action instead.
        """
        if result.action_type != "click" or result.x is None:
            return result

        # Check if this click is near a previous click
        is_repeat = any(
            abs(result.x - px) <= self._click_dedup_radius
            and abs(result.y - py) <= self._click_dedup_radius
            for px, py in self._recent_clicks
        )

        # Track this click
        self._recent_clicks.append((result.x, result.y))
        if len(self._recent_clicks) > 10:
            self._recent_clicks = self._recent_clicks[-10:]

        if is_repeat:
            # Extract text from the task to type
            text = extract_task_text(task)
            if text:
                logger.info(
                    "Click-dedup: repeated click near (%d,%d), converting to type(%r)",
                    result.x, result.y, text,
                )
                return EscalationResult(
                    success=True,
                    action_type="type",
                    text=text,
                    thought=f"[AUTO-TYPE] Repeated click detected, typing task text: {text}",
                    inference_ms=result.inference_ms,
                )
            else:
                logger.warning(
                    "Click-dedup: repeated click but no text extractable from task: %r",
                    task,
                )

        return result

    def reset_clicks(self) -> None:
        """Clear click history (call after successful navigation)."""
        self._recent_clicks.clear()


# --- Response Parsing (minimal, focused on getting one action) ---

# --- Task Text Extraction ---

# Patterns like: "search for X", "type X", "enter X", "search X on", 'search "X"'
_TASK_SEARCH_RE = re.compile(
    r"""(?:search\s+(?:for\s+)?|type\s+|enter\s+|look\s+up\s+|find\s+)"""
    r"""['\"]?(.+?)['\"]?(?=\s+(?:on|in|into)\b|\s*$)""",
    re.IGNORECASE,
)
_TASK_QUOTED_RE = re.compile(r"""['"]([^'"]+)['"]""")


def extract_task_text(task: str) -> str | None:
    """Extract the likely text-to-type from a task description.

    Examples:
        "Search Wikipedia for 'pole vault'" → "pole vault"
        "Search for cats on Google" → "cats"
        "Type hello world" → "hello world"
    """
    # Try quoted text first (most explicit)
    quoted = _TASK_QUOTED_RE.search(task)
    if quoted:
        return quoted.group(1)

    # Try search/type pattern
    search = _TASK_SEARCH_RE.search(task)
    if search:
        text = search.group(1).strip()
        # Remove trailing punctuation
        text = text.rstrip(".,!?")
        if text:
            return text

    return None


_THOUGHT_RE = re.compile(r"Thought:\s*(.+?)(?=\nAction:|\Z)", re.DOTALL)
_ACTION_RE = re.compile(r"Action:\s*(\w+)\((.*)\).*$", re.MULTILINE)
_COORD_RE = re.compile(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)")
# Handle both single and double quotes: content='...' or content="..."
_CONTENT_RE = re.compile(r"""content=['"]((?:[^'"\\]|\\.)*?)['"]""")
_DIRECTION_RE = re.compile(r"""direction=['"]((?:[^'"\\]|\\.)*?)['"]""")


def _parse_escalation_response(
    raw: str, screen_width: int, screen_height: int,
) -> EscalationResult:
    """Parse vision model response into an EscalationResult.

    Tries multiple formats in order:
        1. Standard: Thought: ... Action: click(start_box="(X,Y)")
        2. Qwen2.5-VL native JSON: {"point_2d": [X, Y], "label": "..."}

    Coordinates are treated as raw pixel values (matching the viewport size).

    Raises:
        ValueError: If no parseable action can be found.
    """
    thought = ""
    thought_match = _THOUGHT_RE.search(raw)
    if thought_match:
        thought = thought_match.group(1).strip()

    # --- Try standard Action: format first ---
    action_match = _ACTION_RE.search(raw)
    if action_match:
        return _parse_standard_action(action_match, thought, screen_width, screen_height)

    # --- Fallback: Qwen2.5-VL native JSON point format ---
    json_result = _parse_qwen_vl_json(raw, thought, screen_width, screen_height)
    if json_result:
        return json_result

    raise ValueError(f"No Action: or JSON point found in vision response: {raw[:200]!r}")


def _parse_standard_action(
    action_match: re.Match, thought: str,
    screen_width: int, screen_height: int,
) -> EscalationResult:
    """Parse standard Thought/Action format."""
    action_type = action_match.group(1)
    args_str = action_match.group(2)

    action_map = {
        "click": "click", "left_double": "double_click",
        "scroll": "scroll", "type": "type",
        "wait": "wait", "finished": "done",
    }
    mapped_type = action_map.get(action_type, action_type)

    x, y = None, None
    coord_match = _COORD_RE.search(args_str)
    if coord_match:
        x = int(coord_match.group(1))
        y = int(coord_match.group(2))
        x = min(max(x, 0), screen_width)
        y = min(max(y, 0), screen_height)

    text = None
    content_match = _CONTENT_RE.search(args_str)
    if content_match:
        text = content_match.group(1).replace("\\'", "'").replace('\\"', '"')

    direction = None
    dir_match = _DIRECTION_RE.search(args_str)
    if dir_match:
        direction = dir_match.group(1)

    return EscalationResult(
        success=True, action_type=mapped_type,
        x=x, y=y, text=text, direction=direction, thought=thought,
    )


# Qwen2.5-VL JSON: {"point_2d": [452, 128], "label": "search button"}
_POINT_JSON_RE = re.compile(r'\{[^{}]*"point_2d"\s*:\s*\[([^\]]+)\][^{}]*\}')


def _parse_qwen_vl_json(
    raw: str, thought: str,
    screen_width: int, screen_height: int,
) -> EscalationResult | None:
    """Try to parse Qwen2.5-VL's native JSON point format.

    Qwen2.5-VL outputs pixel coordinates (not normalized) in the actual
    image size. Format: {"point_2d": [X, Y], "label": "description"}

    Returns None if no JSON point found.
    """
    match = _POINT_JSON_RE.search(raw)
    if not match:
        return None

    try:
        # Extract full JSON object for label
        json_str = match.group(0)
        data = json.loads(json_str)
        coords = data.get("point_2d", [])
        label = data.get("label", "")

        if len(coords) < 2:
            return None

        # Qwen2.5-VL uses pixel coords in the image size
        x = int(float(coords[0]))
        y = int(float(coords[1]))
        x = min(max(x, 0), screen_width)
        y = min(max(y, 0), screen_height)

        return EscalationResult(
            success=True,
            action_type="click",
            x=x, y=y,
            thought=thought or f"[Qwen2.5-VL] {label}",
        )
    except (json.JSONDecodeError, ValueError, TypeError):
        return None
