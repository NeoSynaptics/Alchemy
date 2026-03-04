"""Playwright GUI Agent — main agent loop.

Snapshot → Qwen3 14B (think: true) → action → execute → repeat.
When stuck, escalates to UI-TARS 7B vision fallback (Tier 1.5).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum

from alchemy.core.parser import ParseError, PlaywrightAction, parse_playwright_response
from alchemy.core.prompts import SYSTEM_PROMPT, format_action_log_entry, format_user_prompt
from alchemy.core.escalation import StuckDetector, VisionEscalation
from alchemy.core.executor import ExecutionError, execute_action
from alchemy.core.snapshot import capture_snapshot
from alchemy.core.trace import AgentTrace, make_trace_entry

logger = logging.getLogger(__name__)


class AgentStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    WAITING_APPROVAL = "waiting_approval"


@dataclass
class StepResult:
    """Result of a single agent step."""

    step: int
    action: PlaywrightAction
    success: bool
    error: str | None = None
    snapshot_text: str = ""
    inference_ms: float = 0.0
    execution_ms: float = 0.0
    escalated: bool = False  # Was this step handled by vision fallback?


@dataclass
class AgentResult:
    """Final result of a complete agent run."""

    status: AgentStatus
    steps: list[StepResult] = field(default_factory=list)
    total_steps: int = 0
    total_ms: float = 0.0
    error: str | None = None
    escalation_count: int = 0  # How many times vision fallback was used
    trace: AgentTrace | None = None  # Full trace for replay/debugging


class PlaywrightAgent:
    """Accessibility tree → LLM → action loop.

    Args:
        ollama_client: OllamaClient for inference.
        model: Ollama model name (e.g., "qwen3:14b").
        max_steps: Maximum steps before forced stop.
        think: Enable Qwen3 thinking mode.
        temperature: LLM temperature.
        max_tokens: Max tokens per inference call.
        settle_timeout: Time to wait for page to settle after action (ms).
        approval_checker: Optional callback for irreversible action detection.
        vision_escalation: Optional VisionEscalation for Tier 1.5 fallback.
        stuck_detector: Optional StuckDetector (created automatically if escalation is set).
    """

    def __init__(
        self,
        ollama_client,
        model: str = "qwen3:14b",
        max_steps: int = 50,
        think: bool = True,
        temperature: float = 0.1,
        max_tokens: int = 300,
        settle_timeout: float = 5000,
        approval_checker=None,
        vision_escalation: VisionEscalation | None = None,
        stuck_detector: StuckDetector | None = None,
        enable_trace: bool = False,
    ):
        self._ollama = ollama_client
        self._model = model
        self._max_steps = max_steps
        self._think = think
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._settle_timeout = settle_timeout
        self._approval_checker = approval_checker
        self._escalation = vision_escalation
        self._stuck_detector = stuck_detector or (StuckDetector() if vision_escalation else None)
        self._enable_trace = enable_trace

    async def run_task(self, task: str, page) -> AgentResult:
        """Execute a task using the Playwright agent loop.

        Args:
            task: Natural language task description.
            page: Playwright Page object to operate on.

        Returns:
            AgentResult with status, steps, and timing.
        """
        start = time.monotonic()
        action_log: list[str] = []
        steps: list[StepResult] = []
        consecutive_errors = 0
        max_consecutive_errors = 3
        escalation_count = 0
        trace = AgentTrace(task=task, started_at=start * 1000) if self._enable_trace else None

        logger.info("Starting task: %s", task)

        for step_num in range(1, self._max_steps + 1):
            try:
                # --- Pre-check: skip Qwen3 if page is too complex ---
                # When complexity exceeds threshold, Qwen3 will always fail.
                # Go straight to vision model to avoid wasting ~14s per step.
                if self._stuck_detector and self._escalation:
                    snapshot = await capture_snapshot(page)
                    ref_count = snapshot.text.count("[ref=e")
                    logger.debug("Step %d: url=%s refs=%d", step_num, page.url[:80], ref_count)
                    pre_reason = self._stuck_detector.check(ref_count=ref_count)

                    if pre_reason:
                        logger.info(
                            "Step %d: pre-check escalation (%s, %d refs), skipping Qwen3",
                            step_num, pre_reason.value, ref_count,
                        )
                        esc_result = await self._escalation.escalate(
                            page=page,
                            task=task,
                            recent_actions=action_log[-5:],
                            reason=pre_reason,
                        )

                        if esc_result.success:
                            esc_step = await self._execute_escalation(
                                page, esc_result, step_num,
                            )
                            steps.append(esc_step)
                            escalation_count += 1

                            log_entry = format_action_log_entry(
                                step=step_num,
                                action_type=f"[VISION] {esc_result.action_type}",
                                text=esc_result.thought[:50] if esc_result.thought else None,
                                success=esc_step.success,
                                error=esc_step.error,
                            )
                            action_log.append(log_entry)

                            # Don't reset stuck detector — let it keep tracking
                            consecutive_errors = 0

                            # Reset click history after type (page likely navigated)
                            if esc_result.action_type == "type":
                                self._escalation.reset_clicks()

                            try:
                                await page.wait_for_load_state(
                                    "networkidle", timeout=self._settle_timeout,
                                )
                            except Exception:
                                pass

                            continue  # Next step

                # --- Run Qwen3 14B (normal path) ---
                step_result = await self._run_step(task, page, action_log, step_num)

                # --- Post-step stuck detection + escalation ---
                if self._stuck_detector and self._escalation:
                    if step_result.success:
                        sig = f"{step_result.action.type}@{step_result.action.ref or ''}"
                        self._stuck_detector.record_success(sig)
                    else:
                        self._stuck_detector.record_parse_failure()

                    reason = self._stuck_detector.check()

                    if reason and not step_result.success:
                        logger.info(
                            "Step %d: stuck detected (%s), escalating to vision",
                            step_num, reason.value,
                        )
                        esc_result = await self._escalation.escalate(
                            page=page,
                            task=task,
                            recent_actions=action_log[-5:],
                            reason=reason,
                        )

                        if esc_result.success:
                            esc_step = await self._execute_escalation(
                                page, esc_result, step_num,
                            )
                            steps.append(esc_step)
                            escalation_count += 1

                            log_entry = format_action_log_entry(
                                step=step_num,
                                action_type=f"[VISION] {esc_result.action_type}",
                                text=esc_result.thought[:50] if esc_result.thought else None,
                                success=esc_step.success,
                                error=esc_step.error,
                            )
                            action_log.append(log_entry)

                            self._stuck_detector.reset()
                            consecutive_errors = 0

                            # Reset click history after type (page likely navigated)
                            if esc_result.action_type == "type":
                                self._escalation.reset_clicks()

                            try:
                                await page.wait_for_load_state(
                                    "networkidle", timeout=self._settle_timeout,
                                )
                            except Exception:
                                pass

                            continue

                # --- Normal flow (no escalation triggered) ---
                steps.append(step_result)

                # Record trace entry
                if trace:
                    action_str = step_result.action.type
                    if step_result.action.ref:
                        action_str += f" @{step_result.action.ref}"
                    if step_result.action.text:
                        action_str += f' "{step_result.action.text[:50]}"'
                    trace.record(make_trace_entry(
                        step=step_num,
                        snapshot_text=step_result.snapshot_text,
                        prompt=f"[step {step_num}]",
                        llm_output=step_result.action.thought[:200],
                        action_str=action_str,
                        success=step_result.success,
                        inference_ms=step_result.inference_ms,
                        execution_ms=step_result.execution_ms,
                        escalated=step_result.escalated,
                        error=step_result.error,
                    ))

                # Update action log
                log_entry = format_action_log_entry(
                    step=step_num,
                    action_type=step_result.action.type,
                    ref=step_result.action.ref,
                    text=step_result.action.text,
                    success=step_result.success,
                    error=step_result.error,
                )
                action_log.append(log_entry)

                # Check for completion
                if step_result.action.type == "done":
                    logger.info("Task completed in %d steps", step_num)
                    return AgentResult(
                        status=AgentStatus.COMPLETED,
                        steps=steps,
                        total_steps=step_num,
                        total_ms=(time.monotonic() - start) * 1000,
                        escalation_count=escalation_count,
                        trace=trace,
                    )

                # Check approval gate
                if self._approval_checker and step_result.action.type != "wait":
                    needs_approval = self._approval_checker(step_result.action)
                    if needs_approval:
                        logger.info("Step %d: approval required for %s", step_num, step_result.action.type)
                        return AgentResult(
                            status=AgentStatus.WAITING_APPROVAL,
                            steps=steps,
                            total_steps=step_num,
                            total_ms=(time.monotonic() - start) * 1000,
                            escalation_count=escalation_count,
                            trace=trace,
                        )

                # Track consecutive errors
                if step_result.success:
                    consecutive_errors = 0
                else:
                    consecutive_errors += 1
                    if consecutive_errors >= max_consecutive_errors:
                        return AgentResult(
                            status=AgentStatus.FAILED,
                            steps=steps,
                            total_steps=step_num,
                            total_ms=(time.monotonic() - start) * 1000,
                            error=f"Failed {max_consecutive_errors} consecutive steps",
                            escalation_count=escalation_count,
                            trace=trace,
                        )

                # Wait for page to settle after action
                if step_result.action.type not in ("wait", "done"):
                    try:
                        await page.wait_for_load_state(
                            "networkidle", timeout=self._settle_timeout
                        )
                    except Exception:
                        pass  # Timeout is fine — page might not have network activity

            except Exception as e:
                logger.error("Step %d unexpected error: %s", step_num, e)
                steps.append(StepResult(
                    step=step_num,
                    action=PlaywrightAction(type="error", thought=str(e)),
                    success=False,
                    error=str(e),
                ))
                consecutive_errors += 1
                if consecutive_errors >= max_consecutive_errors:
                    break

        # Exhausted max steps
        return AgentResult(
            status=AgentStatus.FAILED,
            steps=steps,
            total_steps=len(steps),
            total_ms=(time.monotonic() - start) * 1000,
            error=f"Reached maximum steps ({self._max_steps})",
            escalation_count=escalation_count,
            trace=trace,
        )

    async def _run_step(
        self, task: str, page, action_log: list[str], step_num: int,
    ) -> StepResult:
        """Execute a single step: snapshot → infer → parse → execute."""
        # 1. Capture accessibility snapshot
        snapshot = await capture_snapshot(page)

        # 2. Build prompt
        user_prompt = format_user_prompt(
            task=task,
            snapshot_text=snapshot.text,
            action_log=action_log,
            step=step_num,
        )

        # 3. Infer via Qwen3 14B
        t0 = time.monotonic()
        raw_response = await self._infer(user_prompt)
        inference_ms = (time.monotonic() - t0) * 1000

        logger.debug("Step %d LLM response (%.0fms): %s", step_num, inference_ms, raw_response[:200])

        # 4. Parse action
        try:
            action = parse_playwright_response(raw_response)
        except ParseError as e:
            logger.warning("Step %d parse error: %s", step_num, e)
            return StepResult(
                step=step_num,
                action=PlaywrightAction(type="error", thought=raw_response[:200]),
                success=False,
                error=f"Parse error: {e}",
                snapshot_text=snapshot.text,
                inference_ms=inference_ms,
            )

        # 5. Execute action
        t1 = time.monotonic()
        try:
            await execute_action(
                page=page,
                action_type=action.type,
                ref=action.ref,
                ref_map=snapshot.ref_map,
                text=action.text,
                direction=action.direction,
                key_name=action.key_name,
            )
            execution_ms = (time.monotonic() - t1) * 1000
            logger.info(
                "Step %d: %s %s (infer=%.0fms, exec=%.0fms)",
                step_num, action.type, action.ref or "", inference_ms, execution_ms,
            )
            return StepResult(
                step=step_num,
                action=action,
                success=True,
                snapshot_text=snapshot.text,
                inference_ms=inference_ms,
                execution_ms=execution_ms,
            )

        except ExecutionError as e:
            execution_ms = (time.monotonic() - t1) * 1000
            logger.warning("Step %d execution error: %s", step_num, e)
            return StepResult(
                step=step_num,
                action=action,
                success=False,
                error=str(e),
                snapshot_text=snapshot.text,
                inference_ms=inference_ms,
                execution_ms=execution_ms,
            )

    async def _execute_escalation(
        self, page, esc_result, step_num: int,
    ) -> StepResult:
        """Execute a vision-guided escalation action on the Playwright page.

        Maps UI-TARS output (pixel coordinates) to Playwright mouse actions.
        """
        t0 = time.monotonic()
        action_type = esc_result.action_type

        try:
            if action_type == "click" and esc_result.x is not None:
                await page.mouse.click(esc_result.x, esc_result.y)
                logger.info(
                    "Step %d [VISION]: click(%d, %d) — %s",
                    step_num, esc_result.x, esc_result.y, esc_result.thought[:80],
                )
            elif action_type == "double_click" and esc_result.x is not None:
                await page.mouse.dblclick(esc_result.x, esc_result.y)
            elif action_type == "type" and esc_result.text:
                url_before = page.url

                # Strategy 1: Find a search/text input by common selectors
                typed = False
                search_selectors = [
                    'input[type="search"]',
                    'input[name="search"]',
                    'input[name="q"]',
                    'input[role="searchbox"]',
                    'textarea:focus',
                    'input:focus',
                ]
                for sel in search_selectors:
                    try:
                        loc = page.locator(sel).first
                        if await loc.count() > 0:
                            await loc.fill(esc_result.text)
                            await loc.press("Enter")
                            typed = True
                            logger.info("Step %d [VISION]: fill+Enter(%r) via %s", step_num, esc_result.text[:50], sel)
                            break
                    except Exception:
                        continue

                # Strategy 2: Fall back to keyboard
                if not typed:
                    await page.keyboard.type(esc_result.text, delay=20)
                    await page.wait_for_timeout(300)
                    await page.keyboard.press("Enter")
                    logger.info("Step %d [VISION]: type+Enter %r (keyboard fallback)", step_num, esc_result.text[:50])

                # Wait for page navigation
                try:
                    await page.wait_for_load_state("networkidle", timeout=8000)
                except Exception:
                    pass

                logger.info("Step %d [VISION]: url %s → %s", step_num, url_before[:60], page.url[:60])
            elif action_type == "scroll":
                direction = esc_result.direction or "down"
                delta = -300 if direction == "up" else 300
                x = esc_result.x or 0
                y = esc_result.y or 0
                await page.mouse.move(x, y)
                await page.mouse.wheel(0, delta)
                logger.info("Step %d [VISION]: scroll %s", step_num, direction)
            elif action_type in ("wait", "done"):
                logger.info("Step %d [VISION]: %s", step_num, action_type)
            else:
                raise ExecutionError(
                    f"Unsupported escalation action: {action_type}"
                )

            execution_ms = (time.monotonic() - t0) * 1000
            return StepResult(
                step=step_num,
                action=PlaywrightAction(
                    type=action_type,
                    thought=f"[VISION] {esc_result.thought}",
                ),
                success=True,
                inference_ms=esc_result.inference_ms,
                execution_ms=execution_ms,
                escalated=True,
            )

        except Exception as e:
            execution_ms = (time.monotonic() - t0) * 1000
            logger.error("Step %d [VISION] execution failed: %s", step_num, e)
            return StepResult(
                step=step_num,
                action=PlaywrightAction(
                    type=action_type,
                    thought=f"[VISION] {esc_result.thought}",
                ),
                success=False,
                error=str(e),
                inference_ms=esc_result.inference_ms,
                execution_ms=execution_ms,
                escalated=True,
            )

    async def _infer(self, user_prompt: str) -> str:
        """Call Qwen3 14B via Ollama with think mode support."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        result = await self._ollama.chat_think(
            model=self._model,
            messages=messages,
            think=self._think,
            options={"temperature": self._temperature, "num_predict": self._max_tokens},
        )

        if result["thinking"]:
            logger.debug("LLM thinking: %s", result["thinking"][:200])

        content = result["content"]
        if not content:
            logger.warning("LLM content empty, using thinking field as fallback")
            content = result["thinking"]

        return content
