"""AlchemyFlow Vision Agent — the core screenshot->infer->execute loop.

Uses Qwen2.5-VL 7B with short prompt and native point_2d JSON coordinates.
Screenshot at 1280x720, coords scaled to actual screen resolution.

PROVEN WORKING (2026-03-05). See alchemyflow-vision.md for details.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import time
from uuid import UUID

from alchemy.click.flow.action_executor import ActionExecutor
from alchemy.click.flow.action_parser import (
    CoordMode,
    classify_tier,
    parse_response,
    to_vision_action,
)
from alchemy.click.flow.omniparser import OmniParser
from alchemy.click.task_manager import TaskManager
from alchemy.clients.voice_callback import VoiceCallbackClient
from alchemy.models.ollama_client import OllamaClient
from alchemy.router.categories import TaskCategory, classify_task
from alchemy.router.context_builder import ContextBuilder
from alchemy.router.tier import classify_tier_contextual
from alchemy.schemas import (
    ActionTier,
    ApprovalRequest,
    NotifyRequest,
    TaskStatus,
    TaskUpdateRequest,
    VisionAction,
    VisionAnalyzeResponse,
)
logger = logging.getLogger(__name__)

# Short prompt — PROVEN to give better coordinate accuracy than long UI-TARS templates.
# Qwen2.5-VL outputs image pixel coords natively with this prompt.
# Long prompts degrade spatial reasoning at 7B. Keep it minimal.
_SYSTEM_PROMPT = """\
You are a GUI agent. Look at the screenshot and perform the requested action.
Output format: Thought: <reasoning> Action: <action>
For clicks, output: Action: click {"point_2d": [x, y]}
For typing, output: Action: type "text"
For hotkeys, output: Action: hotkey key1+key2
For scrolling, output: Action: scroll up/down
When done, output: Action: done"""

# Adaptive timeouts per task category — avoids wasting time on simple tasks
# or prematurely timing out complex ones.
_CATEGORY_TIMEOUTS: dict[TaskCategory, float] = {
    TaskCategory.MEDIA: 180.0,
    TaskCategory.WEB: 240.0,
    TaskCategory.FILE: 180.0,
    TaskCategory.COMMUNICATION: 150.0,
    TaskCategory.DEVELOPMENT: 480.0,
    TaskCategory.SYSTEM: 150.0,
    TaskCategory.GENERAL: 300.0,
}

# Simple task categories — these get the fast model first
_SIMPLE_CATEGORIES = {
    TaskCategory.MEDIA,
    TaskCategory.FILE,
    TaskCategory.SYSTEM,
}

# Actions after which we can use a shorter screenshot interval
_FAST_ACTIONS = {"click", "scroll", "hotkey"}
_SLOW_ACTIONS = {"wait", "type"}


class VisionAgent:
    """Visuomotor agent: screenshot -> VLM -> action -> execute, in a loop."""

    def __init__(
        self,
        ollama: OllamaClient,
        voice_cb: VoiceCallbackClient,
        task_manager: TaskManager,
        model: str = "qwen2.5vl:7b",
        max_steps: int = 50,
        timeout: float = 300.0,
        screenshot_interval: float = 1.0,
        approval_timeout: float = 60.0,
        history_window: int = 4,
        screen_width: int = 1920,
        screen_height: int = 1080,
        image_width: int = 1280,
        image_height: int = 720,
        context_builder: ContextBuilder | None = None,
        use_streaming: bool = True,
        temperature: float = 0.0,
        max_tokens: int = 512,
        controller=None,
        omniparser: OmniParser | None = None,
    ):
        self._ollama = ollama
        self._controller = controller
        self._executor = ActionExecutor(controller) if controller else None
        self._voice_cb = voice_cb
        self._task_manager = task_manager
        self._model = model
        self._max_steps = max_steps
        self._timeout = timeout
        self._screenshot_interval = screenshot_interval
        self._approval_timeout = approval_timeout
        self._history_window = history_window
        self._screen_width = screen_width
        self._screen_height = screen_height
        self._image_width = image_width
        self._image_height = image_height
        self._context_builder = context_builder
        self._use_streaming = use_streaming
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._omniparser = omniparser

    def _get_timeout(self, category: TaskCategory | None) -> float:
        """Get adaptive timeout based on task category."""
        if category:
            return _CATEGORY_TIMEOUTS.get(category, self._timeout)
        return self._timeout

    def _get_options(self) -> dict:
        """Build Ollama inference options. num_ctx=8192 is REQUIRED for VLMs."""
        opts: dict = {"num_ctx": 8192}
        if self._temperature is not None:
            opts["temperature"] = self._temperature
        if self._max_tokens:
            opts["num_predict"] = self._max_tokens
        return opts

    def _adaptive_interval(self, action: VisionAction) -> float:
        """Shorter pause after fast actions, longer after waits."""
        if action.action in _SLOW_ACTIONS:
            return max(self._screenshot_interval, 2.0)
        if action.action in _FAST_ACTIONS:
            return max(self._screenshot_interval * 0.5, 0.3)
        return self._screenshot_interval

    async def run_task(self, task_id: UUID, goal: str) -> TaskStatus:
        """Execute a multi-step GUI task. Returns final status."""
        state = self._task_manager.get_task(task_id)
        if not state:
            return TaskStatus.FAILED

        self._task_manager.update_task(task_id, status=TaskStatus.RUNNING)

        category = classify_task(goal) if self._context_builder else None
        timeout = self._get_timeout(category)
        options = self._get_options()

        # Build prompt — single user message (Ollama VLMs crash with system role + images)
        system_text = f"{_SYSTEM_PROMPT}\n\nTask: {goal}"

        messages: list[dict] = []
        deadline = time.monotonic() + timeout
        consecutive_parse_errors = 0

        try:
            for step in range(self._max_steps):
                if time.monotonic() > deadline:
                    logger.warning("Task %s timed out at step %d", task_id, step)
                    self._task_manager.update_task(
                        task_id, status=TaskStatus.FAILED, error="Timeout",
                    )
                    return TaskStatus.FAILED

                if state.cancel_event.is_set():
                    logger.info("Task %s cancelled at step %d", task_id, step)
                    self._task_manager.update_task(task_id, status=TaskStatus.DENIED)
                    return TaskStatus.DENIED

                # Capture screenshot
                try:
                    screenshot = await self._controller.screenshot()
                except RuntimeError as e:
                    logger.error("Screenshot failed at step %d: %s", step, e)
                    self._task_manager.update_task(
                        task_id, status=TaskStatus.FAILED, error=str(e),
                    )
                    return TaskStatus.FAILED

                # --- OmniParser perception (optional) ---
                omni_result = None
                omni_context = ""
                if self._omniparser:
                    try:
                        omni_result = await self._omniparser.parse(screenshot)
                        logger.debug(
                            "OmniParser: %d elements in %.0fms",
                            len(omni_result.elements), omni_result.parse_ms,
                        )
                    except Exception as e:
                        logger.warning("OmniParser failed, skipping: %s", e)

                # --- OmniParser fast-path: skip VLM if element matches goal ---
                if omni_result and omni_result.elements:
                    match = self._omniparser.match_goal(omni_result.elements, goal)
                    if match:
                        logger.info(
                            "OmniParser fast-path: %r at (%d,%d) — skipping VLM",
                            match.label, match.center_x, match.center_y,
                        )
                        action = VisionAction(
                            action="click",
                            x=match.center_x,
                            y=match.center_y,
                            reasoning=f"OmniParser fast-path: matched '{match.label}' (conf={match.confidence:.2f})",
                            tier=ActionTier.AUTO,
                        )
                        if category:
                            action.tier = classify_tier_contextual(action, category, goal)

                        raw_text = f"Thought: OmniParser matched '{match.label}'\nAction: click {{\"point_2d\": [{match.center_x}, {match.center_y}]}}"

                        self._task_manager.update_task(
                            task_id, current_step=step + 1, last_action=action,
                        )
                        await self._safe_task_update(task_id, step + 1, action)

                        if action.action not in ("done", "fail"):
                            await self._executor.execute(action)

                        messages.append({"role": "assistant", "content": raw_text})
                        if len(messages) > self._history_window * 2 + 1:
                            messages = [messages[0]] + messages[-(self._history_window * 2):]
                        await asyncio.sleep(self._adaptive_interval(action))
                        continue

                    # No fast-path match — enrich VLM prompt with element map
                    omni_context = self._omniparser.to_prompt_context(omni_result.elements)

                # Build user message — first step includes system prompt (VLM workaround)
                if step == 0:
                    user_content = system_text
                else:
                    user_content = (
                        f"Step {step + 1}. Here is the current screenshot. "
                        f"Continue working on: {goal}"
                    )

                if omni_context:
                    user_content += f"\n\n{omni_context}"

                messages.append({"role": "user", "content": user_content})

                # Infer
                try:
                    raw_text = await self._infer(
                        self._model, messages, screenshot, options,
                    )
                except Exception as e:
                    logger.error("Ollama inference failed at step %d: %s", step, e)
                    self._task_manager.update_task(
                        task_id, status=TaskStatus.FAILED,
                        error=f"Inference error: {e}",
                    )
                    return TaskStatus.FAILED

                logger.info("Step %d [%s] raw: %s", step, self._model, raw_text[:200])

                # Parse response
                try:
                    parsed = parse_response(raw_text)
                    action = to_vision_action(
                        parsed, self._screen_width, self._screen_height,
                        coord_mode=CoordMode.IMAGE_PIXEL,
                        image_width=self._image_width,
                        image_height=self._image_height,
                    )
                    if category:
                        action.tier = classify_tier_contextual(action, category, goal)
                    else:
                        action.tier = classify_tier(action)
                    consecutive_parse_errors = 0
                except ValueError as e:
                    consecutive_parse_errors += 1
                    logger.warning("Parse error at step %d (%d consecutive): %s", step, consecutive_parse_errors, e)
                    messages.append({"role": "assistant", "content": raw_text})
                    continue

                # --- OmniParser verification: check VLM coords land on a real element ---
                if omni_result and action.x is not None and action.y is not None:
                    verified = self._omniparser.verify_action(
                        omni_result.elements, action.x, action.y,
                    )
                    if verified:
                        logger.debug(
                            "OmniParser verified: VLM click (%d,%d) lands on '%s'",
                            action.x, action.y, verified.label,
                        )
                    else:
                        logger.warning(
                            "OmniParser: VLM click (%d,%d) doesn't match any detected element",
                            action.x, action.y,
                        )

                # Update task state
                self._task_manager.update_task(
                    task_id, current_step=step + 1, last_action=action,
                )

                await self._safe_task_update(task_id, step + 1, action)

                # Handle approval flow
                if action.tier == ActionTier.APPROVE:
                    screenshot_b64 = base64.b64encode(screenshot).decode("ascii")
                    approved = await self._handle_approval(
                        task_id, action, screenshot_b64, step + 1, goal,
                    )
                    if not approved:
                        self._task_manager.update_task(task_id, status=TaskStatus.DENIED)
                        return TaskStatus.DENIED
                elif action.tier == ActionTier.NOTIFY:
                    screenshot_b64 = base64.b64encode(screenshot).decode("ascii")
                    await self._handle_notify(task_id, action, step + 1, screenshot_b64)

                # Check terminal actions
                if action.action == "done":
                    self._task_manager.update_task(task_id, status=TaskStatus.COMPLETED)
                    return TaskStatus.COMPLETED
                if action.action == "fail":
                    self._task_manager.update_task(
                        task_id, status=TaskStatus.FAILED,
                        error=action.reasoning or "Agent reported failure",
                    )
                    return TaskStatus.FAILED

                # Execute action on desktop
                await self._executor.execute(action)

                # Add assistant response to history
                messages.append({"role": "assistant", "content": raw_text})

                # Trim history: keep system (first msg) + recent window.
                # Older steps become text-only summaries to save tokens.
                if len(messages) > self._history_window * 2 + 1:
                    messages = [messages[0]] + messages[-(self._history_window * 2):]

                # Adaptive pause
                await asyncio.sleep(self._adaptive_interval(action))

            # Exhausted max steps
            self._task_manager.update_task(
                task_id, status=TaskStatus.FAILED,
                error=f"Exceeded max steps ({self._max_steps})",
            )
            return TaskStatus.FAILED

        except asyncio.CancelledError:
            logger.info("Task %s was cancelled", task_id)
            self._task_manager.update_task(task_id, status=TaskStatus.DENIED)
            return TaskStatus.DENIED
        except Exception as e:
            logger.exception("Unexpected error in task %s", task_id)
            self._task_manager.update_task(
                task_id, status=TaskStatus.FAILED, error=str(e),
            )
            return TaskStatus.FAILED

    async def analyze_single(
        self, screenshot: bytes, goal: str,
    ) -> VisionAnalyzeResponse:
        """One-shot: analyze a screenshot and return the next action (no execution)."""
        start = time.monotonic()
        omni_context = ""

        # OmniParser perception + fast-path
        if self._omniparser:
            try:
                omni_result = await self._omniparser.parse(screenshot)
                if omni_result.elements:
                    match = self._omniparser.match_goal(omni_result.elements, goal)
                    if match:
                        action = VisionAction(
                            action="click",
                            x=match.center_x,
                            y=match.center_y,
                            reasoning=f"OmniParser fast-path: matched '{match.label}' (conf={match.confidence:.2f})",
                        )
                        action.tier = classify_tier(action)
                        return VisionAnalyzeResponse(
                            action=action,
                            model="omniparser-v2",
                            inference_ms=(time.monotonic() - start) * 1000,
                        )
                    omni_context = self._omniparser.to_prompt_context(omni_result.elements)
            except Exception as e:
                logger.warning("OmniParser failed in analyze_single: %s", e)

        options = self._get_options()

        prompt = f"{_SYSTEM_PROMPT}\n\nTask: {goal}"
        if omni_context:
            prompt += f"\n\n{omni_context}"

        messages = [{"role": "user", "content": prompt}]

        raw_text = await self._infer(self._model, messages, screenshot, options)
        elapsed_ms = (time.monotonic() - start) * 1000

        parsed = parse_response(raw_text)
        action = to_vision_action(
            parsed, self._screen_width, self._screen_height,
            coord_mode=CoordMode.IMAGE_PIXEL,
            image_width=self._image_width,
            image_height=self._image_height,
        )

        category = classify_task(goal) if self._context_builder else None
        if category:
            action.tier = classify_tier_contextual(action, category, goal)
        else:
            action.tier = classify_tier(action)

        return VisionAnalyzeResponse(
            action=action,
            model=self._model,
            inference_ms=elapsed_ms,
        )

    async def _infer(
        self, model: str, messages: list[dict],
        screenshot: bytes, options: dict,
    ) -> str:
        """Run inference — streaming with early stop or buffered."""
        if self._use_streaming:
            return await self._ollama.chat_stream(
                model, messages, images=[screenshot],
                options=options, stop_at="\nAction:",
            )
        else:
            response = await self._ollama.chat(
                model, messages, images=[screenshot], options=options,
            )
            return response.get("message", {}).get("content", "")

    async def _handle_approval(
        self, task_id: UUID, action: VisionAction,
        screenshot_b64: str, step: int, goal: str,
    ) -> bool:
        """Pause agent and request approval from AlchemyVoice."""
        state = self._task_manager.get_task(task_id)
        if not state:
            return False

        self._task_manager.update_task(task_id, status=TaskStatus.WAITING_APPROVAL)
        state.approval_event.clear()
        state.approved = None

        try:
            await self._voice_cb.request_approval(ApprovalRequest(
                task_id=task_id, action=action, screenshot_b64=screenshot_b64,
                step=step, goal=goal, timeout_seconds=int(self._approval_timeout),
            ))
        except Exception as e:
            logger.warning("Failed to send approval request: %s", e)

        try:
            await asyncio.wait_for(
                state.approval_event.wait(), timeout=self._approval_timeout,
            )
        except asyncio.TimeoutError:
            logger.warning("Approval timeout for task %s step %d", task_id, step)
            return False

        self._task_manager.update_task(task_id, status=TaskStatus.RUNNING)
        return state.approved is True

    async def _handle_notify(
        self, task_id: UUID, action: VisionAction,
        step: int, screenshot_b64: str,
    ):
        """Send a notification to AlchemyVoice for NOTIFY-tier actions."""
        try:
            await self._voice_cb.notify(NotifyRequest(
                task_id=task_id, action=action,
                message=f"Step {step}: {action.action} — {action.reasoning}",
                step=step, screenshot_b64=screenshot_b64,
            ))
        except Exception as e:
            logger.warning("Failed to send notification: %s", e)

    async def _safe_task_update(
        self, task_id: UUID, step: int, action: VisionAction,
    ):
        """Send task update to AlchemyVoice (best-effort)."""
        try:
            state = self._task_manager.get_task(task_id)
            if state:
                await self._voice_cb.task_update(TaskUpdateRequest(
                    task_id=task_id, status=state.status,
                    current_step=step, last_action=action,
                ))
        except Exception:
            logger.debug("Failed to send task update for %s step %d", task_id, step)
