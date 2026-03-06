"""AlchemyFlow Vision Agent — the core screenshot->infer->execute loop.

Uses the official UI-TARS COMPUTER_USE prompt template from ByteDance.
Supports dual-model routing (fast 7B on GPU vs full 72B on CPU),
streaming inference with early stop, and adaptive screenshot intervals.
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
    parse_uitars_response,
    smart_resize_dimensions,
    to_vision_action,
)
from alchemy.click.flow.omniparser import OmniParser, ParseResult
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
from alchemy.shadow.controller import ShadowDesktopController

logger = logging.getLogger(__name__)

# Official ByteDance COMPUTER_USE template (from codes/ui_tars/prompt.py).
# The {instruction} and {context} placeholders are filled at runtime.
# Language is set to English; the model was trained with this format.
SYSTEM_PROMPT = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task.

## Output Format
```
Thought: ...
Action: ...
```

## Action Space

click(start_box='(x1,y1)')
left_double(start_box='(x1,y1)')
right_single(start_box='(x1,y1)')
drag(start_box='(x1,y1)', end_box='(x2,y2)')
hotkey(key='ctrl c')
type(content='xxx')
scroll(start_box='(x1,y1)', direction='down or up or right or left')
wait()
finished(content='xxx')

## Note
- Use English in `Thought` part.
- Write a small plan and finally summarize your next action (with its target element) in one sentence in `Thought` part.
- Coordinates are normalized 0-1000 for both x and y axes.
- Only output ONE action per step.

{context}

## User Instruction
{instruction}"""

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
    """Visuomotor agent: screenshot -> UI-TARS -> action -> xdotool, in a loop."""

    def __init__(
        self,
        ollama: OllamaClient,
        controller: ShadowDesktopController,
        voice_cb: VoiceCallbackClient,
        task_manager: TaskManager,
        model: str = "avil/UI-TARS",
        fast_model: str | None = None,
        max_steps: int = 50,
        timeout: float = 300.0,
        screenshot_interval: float = 1.0,
        approval_timeout: float = 60.0,
        history_window: int = 4,
        screen_width: int = 1920,
        screen_height: int = 1080,
        context_builder: ContextBuilder | None = None,
        use_streaming: bool = True,
        model_routing: bool = True,
        temperature: float = 0.0,
        max_tokens: int = 384,
        omniparser: OmniParser | None = None,
    ):
        self._ollama = ollama
        self._executor = ActionExecutor(controller)
        self._controller = controller
        self._voice_cb = voice_cb
        self._task_manager = task_manager
        self._model = model
        self._fast_model = fast_model
        self._max_steps = max_steps
        self._timeout = timeout
        self._screenshot_interval = screenshot_interval
        self._approval_timeout = approval_timeout
        self._history_window = history_window
        self._screen_width = screen_width
        self._screen_height = screen_height
        self._context_builder = context_builder
        self._use_streaming = use_streaming
        self._model_routing = model_routing
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._omniparser = omniparser

    def _select_model(self, category: TaskCategory | None) -> tuple[str, CoordMode]:
        """Pick the best model for this task. Returns (model_name, coord_mode)."""
        if (
            self._model_routing
            and self._fast_model
            and category in _SIMPLE_CATEGORIES
        ):
            return self._fast_model, CoordMode.ABSOLUTE
        return self._model, CoordMode.NORMALIZED

    def _get_timeout(self, category: TaskCategory | None) -> float:
        """Get adaptive timeout based on task category."""
        if category:
            return _CATEGORY_TIMEOUTS.get(category, self._timeout)
        return self._timeout

    def _get_options(self) -> dict:
        """Build Ollama inference options."""
        opts: dict = {}
        if self._temperature is not None:
            opts["temperature"] = self._temperature
        if self._max_tokens:
            opts["num_predict"] = self._max_tokens
        return opts or {}

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

        # Classify task and select model
        category = classify_task(goal) if self._context_builder else None
        active_model, coord_mode = self._select_model(category)
        timeout = self._get_timeout(category)
        options = self._get_options()

        # Compute smart-resize dimensions for absolute coordinate mode
        resized_w, resized_h = 0, 0
        if coord_mode == CoordMode.ABSOLUTE:
            resized_w, resized_h = smart_resize_dimensions(
                self._screen_width, self._screen_height,
            )

        # Build context-enriched system prompt
        context = self._context_builder.build(goal) if self._context_builder else ""
        system_text = SYSTEM_PROMPT.format(instruction=goal, context=context)

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

                # OmniParser perception pass (if enabled)
                omni_context = ""
                omni_parse: ParseResult | None = None
                if self._omniparser is not None:
                    try:
                        omni_parse = await self._omniparser.parse(screenshot)
                        logger.debug(
                            "Step %d OmniParser: %d elements in %.0fms",
                            step, len(omni_parse.elements), omni_parse.parse_ms,
                        )

                        # Fast path: try direct element match
                        matched = self._omniparser.match_goal(
                            omni_parse.elements, goal,
                        )
                        if matched is not None:
                            action = VisionAction(
                                action="click",
                                x=matched.center_x,
                                y=matched.center_y,
                                reasoning=(
                                    f"OmniParser direct: \"{matched.label}\" "
                                    f"({matched.element_type}, "
                                    f"conf={matched.confidence:.2f})"
                                ),
                                tier=ActionTier.AUTO,
                            )
                            raw_text = (
                                f"Thought: OmniParser matched \"{matched.label}\"\n"
                                f"Action: click(start_box="
                                f"'({matched.center_x},{matched.center_y})')"
                            )
                            logger.info(
                                "Step %d OmniParser fast path: %r → (%d,%d)",
                                step, matched.label,
                                matched.center_x, matched.center_y,
                            )
                            # Skip VLM inference, go straight to tier/execution
                            consecutive_parse_errors = 0
                            self._task_manager.update_task(
                                task_id, current_step=step + 1, last_action=action,
                            )
                            await self._safe_task_update(task_id, step + 1, action)

                            if action.action == "done":
                                self._task_manager.update_task(
                                    task_id, status=TaskStatus.COMPLETED,
                                )
                                return TaskStatus.COMPLETED

                            await self._executor.execute(action)
                            messages.append({"role": "assistant", "content": raw_text})
                            if len(messages) > self._history_window * 2 + 1:
                                messages = (
                                    [messages[0]]
                                    + messages[-(self._history_window * 2):]
                                )
                            await asyncio.sleep(self._adaptive_interval(action))
                            continue

                        # No direct match — prepare context for VLM
                        omni_context = self._omniparser.to_prompt_context(
                            omni_parse.elements,
                        )
                    except Exception as e:
                        logger.warning("OmniParser failed at step %d: %s", step, e)

                # Build user message — first step includes system prompt (VLM workaround)
                if step == 0:
                    user_content = system_text
                    if omni_context:
                        user_content = f"{system_text}\n\n{omni_context}"
                else:
                    step_msg = (
                        f"Step {step + 1}. Here is the current screenshot. "
                        f"Continue working on: {goal}"
                    )
                    if omni_context:
                        user_content = f"{step_msg}\n\n{omni_context}"
                    else:
                        user_content = step_msg

                messages.append({"role": "user", "content": user_content})

                # Infer
                try:
                    raw_text = await self._infer(
                        active_model, messages, screenshot, options,
                    )
                except Exception as e:
                    # If fast model fails, escalate to full model
                    if active_model != self._model and self._model:
                        logger.warning(
                            "Fast model failed (%s), escalating to %s",
                            e, self._model,
                        )
                        active_model = self._model
                        coord_mode = CoordMode.NORMALIZED
                        resized_w, resized_h = 0, 0
                        try:
                            raw_text = await self._infer(
                                active_model, messages, screenshot, options,
                            )
                        except Exception as e2:
                            logger.error("Ollama inference failed at step %d: %s", step, e2)
                            self._task_manager.update_task(
                                task_id, status=TaskStatus.FAILED,
                                error=f"Inference error: {e2}",
                            )
                            return TaskStatus.FAILED
                    else:
                        logger.error("Ollama inference failed at step %d: %s", step, e)
                        self._task_manager.update_task(
                            task_id, status=TaskStatus.FAILED,
                            error=f"Inference error: {e}",
                        )
                        return TaskStatus.FAILED

                logger.info("Step %d [%s] raw: %s", step, active_model, raw_text[:200])

                # Parse response
                try:
                    parsed = parse_uitars_response(raw_text)
                    action = to_vision_action(
                        parsed, self._screen_width, self._screen_height,
                        coord_mode=coord_mode,
                        resized_width=resized_w,
                        resized_height=resized_h,
                    )
                    if category:
                        action.tier = classify_tier_contextual(action, category, goal)
                    else:
                        action.tier = classify_tier(action)
                    consecutive_parse_errors = 0

                    # OmniParser post-VLM verification
                    if (
                        omni_parse is not None
                        and self._omniparser is not None
                        and action.x is not None
                        and action.y is not None
                    ):
                        verified = self._omniparser.verify_action(
                            omni_parse.elements, action.x, action.y,
                        )
                        if verified:
                            logger.debug(
                                "Step %d verified: VLM (%d,%d) → %r",
                                step, action.x, action.y, verified.label,
                            )
                        else:
                            logger.debug(
                                "Step %d: VLM (%d,%d) on empty space",
                                step, action.x, action.y,
                            )
                except ValueError as e:
                    consecutive_parse_errors += 1
                    logger.warning("Parse error at step %d (%d consecutive): %s", step, consecutive_parse_errors, e)
                    messages.append({"role": "assistant", "content": raw_text})

                    # After 3 consecutive parse errors, escalate to 72B
                    if (
                        consecutive_parse_errors >= 3
                        and active_model != self._model
                        and self._model
                    ):
                        logger.warning("3 consecutive parse errors — escalating to %s", self._model)
                        active_model = self._model
                        coord_mode = CoordMode.NORMALIZED
                        resized_w, resized_h = 0, 0
                    continue

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

                # Execute action on shadow desktop
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

        # OmniParser fast path for one-shot analysis
        if self._omniparser is not None:
            omni_parse = await self._omniparser.parse(screenshot)
            matched = self._omniparser.match_goal(omni_parse.elements, goal)
            if matched is not None:
                elapsed_ms = (time.monotonic() - start) * 1000
                action = VisionAction(
                    action="click",
                    x=matched.center_x,
                    y=matched.center_y,
                    reasoning=(
                        f"OmniParser direct: \"{matched.label}\" "
                        f"({matched.element_type}, conf={matched.confidence:.2f})"
                    ),
                    tier=ActionTier.AUTO,
                )
                return VisionAnalyzeResponse(
                    action=action,
                    model="omniparser",
                    inference_ms=elapsed_ms,
                )

        category = classify_task(goal) if self._context_builder else None
        active_model, coord_mode = self._select_model(category)
        options = self._get_options()

        resized_w, resized_h = 0, 0
        if coord_mode == CoordMode.ABSOLUTE:
            resized_w, resized_h = smart_resize_dimensions(
                self._screen_width, self._screen_height,
            )

        context = self._context_builder.build(goal) if self._context_builder else ""
        # Inject OmniParser element map if available
        omni_ctx = ""
        if self._omniparser is not None:
            try:
                omni_parse = await self._omniparser.parse(screenshot)
                omni_ctx = self._omniparser.to_prompt_context(omni_parse.elements)
            except Exception:
                pass

        system_text = SYSTEM_PROMPT.format(instruction=goal, context=context)
        if omni_ctx:
            system_text = f"{system_text}\n\n{omni_ctx}"
        messages = [{"role": "user", "content": system_text}]

        raw_text = await self._infer(active_model, messages, screenshot, options)
        elapsed_ms = (time.monotonic() - start) * 1000

        parsed = parse_uitars_response(raw_text)
        action = to_vision_action(
            parsed, self._screen_width, self._screen_height,
            coord_mode=coord_mode,
            resized_width=resized_w,
            resized_height=resized_h,
        )

        if category:
            action.tier = classify_tier_contextual(action, category, goal)
        else:
            action.tier = classify_tier(action)

        return VisionAnalyzeResponse(
            action=action,
            model=active_model,
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
