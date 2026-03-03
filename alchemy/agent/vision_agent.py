"""Vision Agent — the core screenshot→infer→execute loop.

Captures screenshots from the shadow desktop, sends them to UI-TARS via Ollama,
parses the returned actions, and executes them via xdotool. Handles approval flow
by pausing and signaling NEO-TX when APPROVE-tier actions are encountered.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import time
from uuid import UUID

from alchemy.agent.action_executor import ActionExecutor
from alchemy.agent.action_parser import (
    classify_tier,
    parse_uitars_response,
    to_vision_action,
)
from alchemy.agent.task_manager import TaskManager
from alchemy.clients.neotx_client import NeoTXClient
from alchemy.models.ollama_client import OllamaClient
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

SYSTEM_PROMPT = """You are a GUI automation agent. You interact with a desktop computer by looking at screenshots and performing mouse/keyboard actions.

Your task: {goal}

For each step:
1. Look at the screenshot carefully
2. Think about what you need to do next
3. Output your reasoning and action

Output format:
Thought: <your reasoning about what you see and what to do>
Action: <one of the available actions>

Available actions:
- click(start_box='(x,y)') - Left click at position
- left_double(start_box='(x,y)') - Double click at position
- right_single(start_box='(x,y)') - Right click at position
- drag(start_box='(x1,y1)', end_box='(x2,y2)') - Drag from start to end
- type(content='text') - Type text at current cursor position
- hotkey(key='ctrl+c') - Press keyboard shortcut
- scroll(start_box='(x,y)', direction='down', amount=3) - Scroll at position
- wait() - Wait for page/app to load
- finished(content='description') - Task is complete

Coordinates are normalized 0-1000 for both x and y axes.
Only output ONE action per step."""


class VisionAgent:
    """Visuomotor agent: screenshot → UI-TARS → action → xdotool, in a loop."""

    def __init__(
        self,
        ollama: OllamaClient,
        controller: ShadowDesktopController,
        neotx: NeoTXClient,
        task_manager: TaskManager,
        model: str = "avil/UI-TARS",
        max_steps: int = 50,
        timeout: float = 300.0,
        screenshot_interval: float = 1.0,
        approval_timeout: float = 60.0,
        history_window: int = 8,
        screen_width: int = 1920,
        screen_height: int = 1080,
    ):
        self._ollama = ollama
        self._executor = ActionExecutor(controller)
        self._controller = controller
        self._neotx = neotx
        self._task_manager = task_manager
        self._model = model
        self._max_steps = max_steps
        self._timeout = timeout
        self._screenshot_interval = screenshot_interval
        self._approval_timeout = approval_timeout
        self._history_window = history_window
        self._screen_width = screen_width
        self._screen_height = screen_height

    async def run_task(self, task_id: UUID, goal: str) -> TaskStatus:
        """Execute a multi-step GUI task. Returns final status."""
        state = self._task_manager.get_task(task_id)
        if not state:
            return TaskStatus.FAILED

        self._task_manager.update_task(task_id, status=TaskStatus.RUNNING)

        # VLMs via Ollama don't support the 'system' role with images.
        # Merge system prompt into the first user message instead.
        system_text = SYSTEM_PROMPT.format(goal=goal)
        messages: list[dict] = []
        deadline = time.monotonic() + self._timeout

        try:
            for step in range(self._max_steps):
                # Check timeout
                if time.monotonic() > deadline:
                    logger.warning("Task %s timed out at step %d", task_id, step)
                    self._task_manager.update_task(
                        task_id, status=TaskStatus.FAILED, error="Timeout"
                    )
                    return TaskStatus.FAILED

                # Check cancellation
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
                        task_id, status=TaskStatus.FAILED, error=str(e)
                    )
                    return TaskStatus.FAILED

                screenshot_b64 = base64.b64encode(screenshot).decode("ascii")

                # Build user message — first step includes system prompt
                if step == 0:
                    user_content = f"{system_text}\n\nStep 1. What should I do first?"
                else:
                    user_content = f"Step {step + 1}. What should I do next to: {goal}"

                messages.append({"role": "user", "content": user_content})

                # Call UI-TARS
                try:
                    response = await self._ollama.chat(
                        self._model, messages, images=[screenshot]
                    )
                except Exception as e:
                    logger.error("Ollama inference failed at step %d: %s", step, e)
                    self._task_manager.update_task(
                        task_id, status=TaskStatus.FAILED, error=f"Inference error: {e}"
                    )
                    return TaskStatus.FAILED

                raw_text = response.get("message", {}).get("content", "")
                logger.info("Step %d raw: %s", step, raw_text[:200])

                # Parse response
                try:
                    parsed = parse_uitars_response(raw_text)
                    action = to_vision_action(parsed, self._screen_width, self._screen_height)
                    action.tier = classify_tier(action)
                except ValueError as e:
                    logger.warning("Parse error at step %d: %s", step, e)
                    messages.append({"role": "assistant", "content": raw_text})
                    continue

                # Update task state
                self._task_manager.update_task(
                    task_id, current_step=step + 1, last_action=action,
                )

                # Send task update to NEO-TX
                await self._safe_task_update(task_id, step + 1, action)

                # Handle approval flow
                if action.tier == ActionTier.APPROVE:
                    approved = await self._handle_approval(
                        task_id, action, screenshot_b64, step + 1, goal
                    )
                    if not approved:
                        self._task_manager.update_task(task_id, status=TaskStatus.DENIED)
                        return TaskStatus.DENIED
                elif action.tier == ActionTier.NOTIFY:
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

                # Trim history (sliding window)
                if len(messages) > self._history_window + 1:
                    messages = [messages[0]] + messages[-(self._history_window):]

                # Brief pause before next screenshot
                await asyncio.sleep(self._screenshot_interval)

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
                task_id, status=TaskStatus.FAILED, error=str(e)
            )
            return TaskStatus.FAILED

    async def analyze_single(
        self, screenshot: bytes, goal: str
    ) -> VisionAnalyzeResponse:
        """One-shot: analyze a screenshot and return the next action (no execution)."""
        # Merge system prompt into user message (VLMs don't support system role)
        system_text = SYSTEM_PROMPT.format(goal=goal)
        messages = [
            {"role": "user", "content": f"{system_text}\n\nWhat should I do to: {goal}"},
        ]

        start = time.monotonic()
        response = await self._ollama.chat(self._model, messages, images=[screenshot])
        elapsed_ms = (time.monotonic() - start) * 1000

        raw_text = response.get("message", {}).get("content", "")
        parsed = parse_uitars_response(raw_text)
        action = to_vision_action(parsed, self._screen_width, self._screen_height)
        action.tier = classify_tier(action)

        return VisionAnalyzeResponse(
            action=action,
            model=self._model,
            inference_ms=elapsed_ms,
        )

    async def _handle_approval(
        self, task_id: UUID, action: VisionAction,
        screenshot_b64: str, step: int, goal: str,
    ) -> bool:
        """Pause agent and request approval from NEO-TX. Returns True if approved."""
        state = self._task_manager.get_task(task_id)
        if not state:
            return False

        self._task_manager.update_task(task_id, status=TaskStatus.WAITING_APPROVAL)
        state.approval_event.clear()
        state.approved = None

        try:
            await self._neotx.request_approval(ApprovalRequest(
                task_id=task_id, action=action, screenshot_b64=screenshot_b64,
                step=step, goal=goal, timeout_seconds=int(self._approval_timeout),
            ))
        except Exception as e:
            logger.warning("Failed to send approval request: %s", e)

        # Wait for approval signal
        try:
            await asyncio.wait_for(
                state.approval_event.wait(), timeout=self._approval_timeout
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
        """Send a notification to NEO-TX for NOTIFY-tier actions."""
        try:
            await self._neotx.notify(NotifyRequest(
                task_id=task_id, action=action,
                message=f"Step {step}: {action.action} — {action.reasoning}",
                step=step, screenshot_b64=screenshot_b64,
            ))
        except Exception as e:
            logger.warning("Failed to send notification: %s", e)

    async def _safe_task_update(
        self, task_id: UUID, step: int, action: VisionAction,
    ):
        """Send task update to NEO-TX (best-effort)."""
        try:
            state = self._task_manager.get_task(task_id)
            if state:
                await self._neotx.task_update(TaskUpdateRequest(
                    task_id=task_id, status=state.status,
                    current_step=step, last_action=action,
                ))
        except Exception:
            pass  # Non-critical
