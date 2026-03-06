"""AlchemyFlowAgent — fused screenshot + VLM + action primitive.

The atomic building block: see the screen, decide what to do, do it.
One call = one full vision-action cycle:
    screenshot → Qwen2.5-VL 7B → parse → click/drag/type/scroll

This is the INTERNAL core primitive. VisionAgent uses it for multi-step
task loops. AlchemyAgents (FlowVS, etc.) import it for single-step ops.

Not user-callable. Not exposed via API. Tier 0 core.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Protocol

from alchemy.click.flow.action_executor import ActionExecutor
from alchemy.click.flow.action_parser import (
    CoordMode,
    classify_tier,
    parse_response,
    to_vision_action,
)
from alchemy.schemas import ActionTier, VisionAction, VisionAnalyzeResponse

logger = logging.getLogger(__name__)


class ScreenSource(Protocol):
    """Anything that can produce a screenshot."""

    async def screenshot(self) -> bytes: ...


class ActionSink(Protocol):
    """Anything that can execute a VisionAction."""

    async def execute(self, action: VisionAction) -> str: ...


@dataclass(frozen=True)
class StepResult:
    """Result of a single see-and-act cycle."""

    action: VisionAction
    raw_response: str
    model_used: str
    inference_ms: float
    executed: bool
    execution_output: str = ""


class FlowAgent:
    """Fused vision-action primitive: screenshot → infer → execute.

    Designed to be imported by any internal agent that needs to
    interact with a screen. Stateless per-step — no task loop,
    no history management, no approval gates. Those live in the
    caller (VisionAgent, FlowVS, etc.).
    """

    def __init__(
        self,
        *,
        ollama: Any,
        screen: ScreenSource,
        executor: ActionSink | None = None,
        model: str = "qwen2.5vl:7b",
        screen_width: int = 1920,
        screen_height: int = 1080,
        image_width: int = 1280,
        image_height: int = 720,
        use_streaming: bool = True,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> None:
        self._ollama = ollama
        self._screen = screen
        self._executor = executor
        self._model = model
        self._screen_width = screen_width
        self._screen_height = screen_height
        self._image_width = image_width
        self._image_height = image_height
        self._use_streaming = use_streaming
        self._temperature = temperature
        self._max_tokens = max_tokens

    def _get_options(self) -> dict:
        opts: dict = {"num_ctx": 8192}
        if self._temperature is not None:
            opts["temperature"] = self._temperature
        if self._max_tokens:
            opts["num_predict"] = self._max_tokens
        return opts

    async def see(
        self,
        goal: str,
        *,
        messages: list[dict] | None = None,
        screenshot: bytes | None = None,
    ) -> tuple[VisionAction, str, float]:
        """Screenshot → VLM → parsed action.

        Returns (action, raw_response, inference_ms).
        """
        if screenshot is None:
            screenshot = await self._screen.screenshot()

        start = time.monotonic()

        # VLM inference
        if messages is None:
            messages = [{"role": "user", "content": goal}]

        options = self._get_options()
        raw_text = await self._infer(messages, screenshot, options)
        elapsed_ms = (time.monotonic() - start) * 1000

        parsed = parse_response(raw_text)
        action = to_vision_action(
            parsed, self._screen_width, self._screen_height,
            coord_mode=CoordMode.IMAGE_PIXEL,
            image_width=self._image_width,
            image_height=self._image_height,
        )
        action.tier = classify_tier(action)

        return action, raw_text, elapsed_ms

    async def act(self, action: VisionAction) -> str:
        """Execute an action on the screen. Returns output."""
        if self._executor is None:
            raise RuntimeError("FlowAgent has no executor configured")
        return await self._executor.execute(action)

    async def step(
        self,
        goal: str,
        *,
        messages: list[dict] | None = None,
        screenshot: bytes | None = None,
        execute: bool = True,
    ) -> StepResult:
        """One full cycle: see → act.

        Args:
            goal: What to accomplish.
            messages: Optional conversation history for the VLM.
            screenshot: Optional pre-captured screenshot.
            execute: If False, only infer — don't execute the action.

        Returns:
            StepResult with the action, raw response, timing, and execution output.
        """
        action, raw_text, inference_ms = await self.see(
            goal, messages=messages, screenshot=screenshot,
        )

        executed = False
        output = ""
        if execute and self._executor and action.action not in ("done", "fail"):
            output = await self.act(action)
            executed = True

        return StepResult(
            action=action,
            raw_response=raw_text,
            model_used=self._model,
            inference_ms=inference_ms,
            executed=executed,
            execution_output=output,
        )

    async def _infer(
        self, messages: list[dict], screenshot: bytes, options: dict,
    ) -> str:
        if self._use_streaming:
            return await self._ollama.chat_stream(
                self._model, messages, images=[screenshot],
                options=options, stop_at="\nAction:",
            )
        else:
            response = await self._ollama.chat(
                self._model, messages, images=[screenshot], options=options,
            )
            return response.get("message", {}).get("content", "")
