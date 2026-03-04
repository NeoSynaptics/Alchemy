"""Playwright GUI Agent — main agent loop.

Snapshot → Qwen3 14B (think: true) → action → execute → repeat.
No vision, no screenshots, no coordinates. Pure structured data.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum

from alchemy.agent.pw_action_parser import ParseError, PlaywrightAction, parse_playwright_response
from alchemy.agent.pw_prompts import SYSTEM_PROMPT, format_action_log_entry, format_user_prompt
from alchemy.playwright.executor import ExecutionError, execute_action
from alchemy.playwright.snapshot import capture_snapshot

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


@dataclass
class AgentResult:
    """Final result of a complete agent run."""

    status: AgentStatus
    steps: list[StepResult] = field(default_factory=list)
    total_steps: int = 0
    total_ms: float = 0.0
    error: str | None = None


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
    ):
        self._ollama = ollama_client
        self._model = model
        self._max_steps = max_steps
        self._think = think
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._settle_timeout = settle_timeout
        self._approval_checker = approval_checker

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

        logger.info("Starting task: %s", task)

        for step_num in range(1, self._max_steps + 1):
            try:
                step_result = await self._run_step(task, page, action_log, step_num)
                steps.append(step_result)

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

    async def _infer(self, user_prompt: str) -> str:
        """Call Qwen3 14B via Ollama with think mode support."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        # Build payload manually to support think parameter
        client = self._ollama._ensure_client()
        payload = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "keep_alive": self._ollama._keep_alive,
            "options": {
                "temperature": self._temperature,
                "num_predict": self._max_tokens,
            },
        }

        # Qwen3 defaults to thinking mode — must explicitly set think: false to disable
        payload["think"] = self._think

        resp = await client.post("/api/chat", json=payload)
        resp.raise_for_status()
        data = resp.json()

        # With think: true, reasoning goes to 'thinking', answer to 'content'
        content = data.get("message", {}).get("content", "")
        thinking = data.get("message", {}).get("thinking", "")

        if thinking:
            logger.debug("LLM thinking: %s", thinking[:200])

        if not content:
            # Fallback: if content is empty, try to use thinking
            logger.warning("LLM content empty, using thinking field as fallback")
            content = thinking

        return content
