"""GateReviewer — LLM-backed tool call reviewer for Claude Code approvals.

Second-class module: borrows inference from core's shared Ollama client.
Uses think:false + low token budget for sub-second decisions.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass

from alchemy.gate.policies import PolicyDecision, check_static_policy

logger = logging.getLogger(__name__)

_DECISION_RE = re.compile(r"\b(accept|deny|other)\b", re.IGNORECASE)

_REVIEW_PROMPT = """\
You are a security reviewer for a coding assistant (Claude Code).
The assistant wants to perform this action:

Tool: {tool_name}
Arguments: {args}
Project: {project}

Decide if this action is safe to auto-approve.

Rules:
- ACCEPT: routine development actions (install deps, run builds, write code files, git operations)
- DENY: destructive actions (delete production data, expose secrets, force-push main)
- OTHER: unclear or risky — let a human decide

Answer with exactly one word: accept, deny, or other"""


@dataclass
class GateResult:
    """Result matching the AlchemyVS GateResult interface."""

    action: str  # "accept" | "deny" | "other"
    reason: str
    tier: str  # "static" | "ask_ollama"
    latency_ms: float = 0.0
    model: str = ""


class GateReviewer:
    """Reviews Claude Code tool calls using static policies + Qwen3 14B fallback."""

    def __init__(
        self,
        ollama_client,
        model: str = "qwen3:14b",
        timeout: float = 5.0,
    ):
        self._ollama = ollama_client
        self._model = model
        self._timeout = timeout

    async def review(
        self,
        tool_name: str,
        args: dict[str, str],
        project_context: dict[str, str] | None = None,
    ) -> GateResult:
        """Review a tool call and return accept/deny/other.

        1. Check static policies (instant, no inference).
        2. If ambiguous, ask Qwen3 14B (think:false, ~800ms).
        3. On timeout/error, fail-open (accept).
        """
        start = time.monotonic()

        # --- Tier 1: Static policies ---
        decision, reason = check_static_policy(tool_name, args)

        if decision == PolicyDecision.ACCEPT:
            ms = (time.monotonic() - start) * 1000
            logger.debug("Gate ACCEPT (static): %s — %s (%.0fms)", tool_name, reason, ms)
            return GateResult(action="accept", reason=reason, tier="static", latency_ms=ms)

        if decision == PolicyDecision.DENY:
            ms = (time.monotonic() - start) * 1000
            logger.warning("Gate DENY (static): %s — %s (%.0fms)", tool_name, reason, ms)
            return GateResult(action="deny", reason=reason, tier="static", latency_ms=ms)

        # --- Tier 2: Ask Qwen3 ---
        project = project_context.get("project", "unknown") if project_context else "unknown"

        prompt = _REVIEW_PROMPT.format(
            tool_name=tool_name,
            args=_format_args(args),
            project=project,
        )

        try:
            raw = await asyncio.wait_for(
                self._infer(prompt),
                timeout=self._timeout,
            )
            ms = (time.monotonic() - start) * 1000

            action = _parse_decision(raw)
            logger.info(
                "Gate %s (ollama): %s %s — %r (%.0fms)",
                action.upper(), tool_name, _format_args(args)[:60], raw[:50], ms,
            )
            return GateResult(
                action=action,
                reason=raw.strip()[:200],
                tier="ask_ollama",
                latency_ms=ms,
                model=self._model,
            )

        except asyncio.TimeoutError:
            ms = (time.monotonic() - start) * 1000
            logger.warning("Gate timeout (%.0fms), fail-open accept: %s", ms, tool_name)
            return GateResult(
                action="accept",
                reason=f"timeout after {ms:.0f}ms",
                tier="ask_ollama",
                latency_ms=ms,
                model=self._model,
            )

        except Exception as e:
            ms = (time.monotonic() - start) * 1000
            logger.warning("Gate error, fail-open accept: %s — %s", tool_name, e)
            return GateResult(
                action="accept",
                reason=f"error: {e}",
                tier="ask_ollama",
                latency_ms=ms,
                model=self._model,
            )

    async def _infer(self, prompt: str) -> str:
        """Quick Qwen3 call — think:false, low tokens."""
        messages = [{"role": "user", "content": prompt}]
        result = await self._ollama.chat_think(
            model=self._model,
            messages=messages,
            think=False,
            options={"temperature": 0.0, "num_predict": 50},
        )
        return result.get("content") or result.get("thinking") or ""


def _parse_decision(raw: str) -> str:
    """Extract accept/deny/other from model output."""
    match = _DECISION_RE.search(raw.lower())
    if match:
        return match.group(1).lower()
    return "other"


def _format_args(args: dict[str, str]) -> str:
    """Format args dict for display/prompt."""
    if not args:
        return "{}"
    parts = [f"{k}={v!r}" for k, v in args.items()]
    return ", ".join(parts)
