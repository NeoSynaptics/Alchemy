"""Tier 1.5 escalation — UI-TARS 7B visual fallback for stuck Playwright agent.

When Qwen3 14B gets overwhelmed by a complex page (parse failures, action loops,
too many refs), we take a screenshot and ask UI-TARS 7B to visually identify the
next action. One shot, then hand control back to Qwen3.

Flow:
    Qwen3 stuck → screenshot → UI-TARS 7B → click(x, y) → Qwen3 resumes
"""

from __future__ import annotations

import logging
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


# --- UI-TARS 7B Escalation Prompt ---

_ESCALATION_PROMPT = """\
You are a GUI agent. Look at this screenshot and determine the next action.

Task: {task}
{context}
## Action Space

click(start_box='(x1,y1)')
type(content='xxx')
scroll(start_box='(x1,y1)', direction='down or up')
wait()
finished(content='xxx')

## Note
- Coordinates are normalized 0-1000 for both x and y axes.
- Only output ONE action.

Output format:
Thought: ...
Action: ..."""


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
    ):
        self._ollama = ollama_client
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._screen_width = screen_width
        self._screen_height = screen_height

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

        prompt = _ESCALATION_PROMPT.format(task=task, context=context)

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
            return result
        except ValueError as e:
            logger.warning("Could not parse vision model response: %s", e)
            return EscalationResult(
                success=False, error=f"Parse error: {e}",
                inference_ms=inference_ms,
            )


# --- Response Parsing (minimal, focused on getting one action) ---

import re

_THOUGHT_RE = re.compile(r"Thought:\s*(.+?)(?=\nAction:|\Z)", re.DOTALL)
_ACTION_RE = re.compile(r"Action:\s*(\w+)\((.*)\)\s*$", re.MULTILINE)
_COORD_RE = re.compile(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)")
_CONTENT_RE = re.compile(r"content='((?:[^'\\]|\\.)*)'")
_DIRECTION_RE = re.compile(r"direction='([^']*)'")


def _parse_escalation_response(
    raw: str, screen_width: int, screen_height: int,
) -> EscalationResult:
    """Parse UI-TARS response into an EscalationResult.

    Raises:
        ValueError: If no Action: line can be parsed.
    """
    thought = ""
    thought_match = _THOUGHT_RE.search(raw)
    if thought_match:
        thought = thought_match.group(1).strip()

    action_match = _ACTION_RE.search(raw)
    if not action_match:
        raise ValueError(f"No Action: found in vision response: {raw[:200]!r}")

    action_type = action_match.group(1)
    args_str = action_match.group(2)

    # Map UI-TARS action names
    action_map = {
        "click": "click", "left_double": "double_click",
        "scroll": "scroll", "type": "type",
        "wait": "wait", "finished": "done",
    }
    mapped_type = action_map.get(action_type, action_type)

    # Extract coordinates (normalized 0-1000 → pixel)
    x, y = None, None
    coord_match = _COORD_RE.search(args_str)
    if coord_match:
        norm_x, norm_y = int(coord_match.group(1)), int(coord_match.group(2))
        x = round(norm_x / 1000 * screen_width)
        y = round(norm_y / 1000 * screen_height)
        x = min(max(x, 0), screen_width)
        y = min(max(y, 0), screen_height)

    # Extract text content
    text = None
    content_match = _CONTENT_RE.search(args_str)
    if content_match:
        text = content_match.group(1).replace("\\'", "'")

    # Extract direction
    direction = None
    dir_match = _DIRECTION_RE.search(args_str)
    if dir_match:
        direction = dir_match.group(1)

    return EscalationResult(
        success=True,
        action_type=mapped_type,
        x=x,
        y=y,
        text=text,
        direction=direction,
        thought=thought,
    )
