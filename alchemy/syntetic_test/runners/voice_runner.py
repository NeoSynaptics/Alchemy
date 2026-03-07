"""AlchemyVoice E2E test runner.

Steps:
  1. Check if Voice server is reachable (health)
  2. Check microphone availability
  3. Cold-start: send first message, measure response time
  4. Assert we got a valid response
  5. Send second message immediately after, measure warm response time
  6. Assert second response
  7. Report pass/fail with timings

Timeout protection: if any step exceeds thresholds, flag for GPU test routing.
"""

from __future__ import annotations

import asyncio
import time
from typing import Callable

import httpx


# Thresholds (seconds)
HEALTH_TIMEOUT = 5
COLD_RESPONSE_TIMEOUT = 60      # First message — model might need loading
WARM_RESPONSE_TIMEOUT = 30      # Second message — model should be hot
FREEZE_THRESHOLD = 90           # If total exceeds this, route to GPU test


class VoiceTestRunner:
    def __init__(
        self,
        voice_url: str,
        log_fn: Callable[[str, str], None],
        done_fn: Callable[[bool, bool], None],
        root=None,
    ):
        self.url = voice_url
        self.log = log_fn
        self.done = done_fn
        self.root = root
        self._passed = True
        self._should_route_gpu = False
        self._total_start = 0.0

    async def run(self):
        self._total_start = time.monotonic()
        try:
            await self._step_health()
            await self._step_mic_check()
            cold_ms = await self._step_cold_message()
            warm_ms = await self._step_warm_message()
            await self._step_summary(cold_ms, warm_ms)
        except _GpuRouteError as e:
            self.log(f"GPU ROUTE: {e}", "warn")
            self._passed = False
            self._should_route_gpu = True
        except Exception as e:
            self.log(f"UNEXPECTED ERROR: {e}", "fail")
            self._passed = False
        finally:
            self.done(self._passed, self._should_route_gpu)

    # ------------------------------------------------------------------
    # Steps
    # ------------------------------------------------------------------

    async def _step_health(self):
        self.log("[1/5] Checking Voice server health...", "info")
        try:
            async with httpx.AsyncClient(timeout=HEALTH_TIMEOUT) as c:
                r = await c.get(f"{self.url}/health")
            if r.status_code == 200:
                body = r.json()
                self.log(f"  Server OK — status: {body.get('status', 'ok')}", "pass")
            else:
                self._fail(f"  Health returned {r.status_code}")
        except httpx.ConnectError:
            self._fail(f"  Cannot connect to {self.url} — is AlchemyVoice running?")
        except httpx.ReadTimeout:
            self._fail(f"  Health check timed out after {HEALTH_TIMEOUT}s")

    async def _step_mic_check(self):
        self.log("[2/5] Checking microphone...", "info")
        try:
            # Voice status tells us if the pipeline can access audio
            async with httpx.AsyncClient(timeout=HEALTH_TIMEOUT) as c:
                r = await c.get(f"{self.url}/v1/voice/status")
            if r.status_code == 200:
                data = r.json()
                if data.get("voice_enabled", False):
                    self.log(f"  Voice enabled, state: {data.get('state', '?')}", "pass")
                else:
                    self.log("  Voice pipeline disabled in config — skipping mic", "warn")
            else:
                self.log(f"  Voice status returned {r.status_code} — mic check skipped", "warn")
        except Exception as e:
            self.log(f"  Mic check skipped: {e}", "warn")

    async def _step_cold_message(self) -> float:
        """Send first message — model may be cold. Return response time in ms."""
        self.log("[3/5] Cold start: sending first message...", "info")
        msg = "Hello, this is a synthetic test. Please respond briefly."

        t0 = time.monotonic()
        response_text, status = await self._send_chat(msg, COLD_RESPONSE_TIMEOUT)
        elapsed_ms = (time.monotonic() - t0) * 1000

        self._check_freeze()

        if status != 200:
            if elapsed_ms > (COLD_RESPONSE_TIMEOUT * 1000 * 0.9):
                raise _GpuRouteError(
                    f"Cold message timed out after {elapsed_ms:.0f}ms — possible GPU issue"
                )
            self._fail(f"  Cold message failed with status {status}")
            return elapsed_ms

        if not response_text or len(response_text.strip()) == 0:
            self._fail("  Got empty response from cold message")
            return elapsed_ms

        self.log(f"  Response ({elapsed_ms:.0f}ms): {_truncate(response_text, 80)}", "pass")

        # Flag if suspiciously slow but not timed out
        if elapsed_ms > 30_000:
            self.log(f"  WARNING: Cold response took {elapsed_ms:.0f}ms (>30s)", "warn")

        return elapsed_ms

    async def _step_warm_message(self) -> float:
        """Send second message — model should be warm now."""
        self.log("[4/5] Warm follow-up: sending second message...", "info")
        msg = "What did I just say to you?"

        t0 = time.monotonic()
        response_text, status = await self._send_chat(msg, WARM_RESPONSE_TIMEOUT)
        elapsed_ms = (time.monotonic() - t0) * 1000

        self._check_freeze()

        if status != 200:
            if elapsed_ms > (WARM_RESPONSE_TIMEOUT * 1000 * 0.9):
                raise _GpuRouteError(
                    f"Warm message timed out after {elapsed_ms:.0f}ms — possible GPU freeze"
                )
            self._fail(f"  Warm message failed with status {status}")
            return elapsed_ms

        if not response_text or len(response_text.strip()) == 0:
            self._fail("  Got empty response from warm message")
            return elapsed_ms

        self.log(f"  Response ({elapsed_ms:.0f}ms): {_truncate(response_text, 80)}", "pass")
        return elapsed_ms

    async def _step_summary(self, cold_ms: float, warm_ms: float):
        self.log("[5/5] Summary", "info")
        self.log(f"  Cold response:  {cold_ms:>8.0f} ms", "dim")
        self.log(f"  Warm response:  {warm_ms:>8.0f} ms", "dim")
        total = time.monotonic() - self._total_start
        self.log(f"  Total test:     {total:>8.1f} s", "dim")

        if self._passed:
            self.log("RESULT: ALL CHECKS PASSED", "pass")
        else:
            self.log("RESULT: SOME CHECKS FAILED", "fail")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _send_chat(self, message: str, timeout: float) -> tuple[str, int]:
        """Send a chat message via SSE stream, collect full response."""
        try:
            async with httpx.AsyncClient(timeout=timeout) as c:
                r = await c.post(
                    f"{self.url}/v1/chat/stream",
                    json={"message": message, "source": "api"},
                )
                if r.status_code != 200:
                    return "", r.status_code

                # Parse SSE response
                full_text = ""
                for line in r.text.split("\n"):
                    if line.startswith("data: "):
                        try:
                            import json
                            chunk = json.loads(line[6:])
                            full_text += chunk.get("content", "")
                        except (json.JSONDecodeError, KeyError):
                            pass
                return full_text, 200

        except httpx.ReadTimeout:
            return "", 408
        except httpx.ConnectError:
            return "", 503
        except Exception as e:
            self.log(f"  Chat error: {e}", "fail")
            return "", 500

    def _check_freeze(self):
        """If total elapsed time is too long, something is wrong with GPU."""
        elapsed = time.monotonic() - self._total_start
        if elapsed > FREEZE_THRESHOLD:
            raise _GpuRouteError(
                f"Total test time {elapsed:.0f}s exceeds {FREEZE_THRESHOLD}s freeze threshold"
            )

    def _fail(self, msg: str):
        self.log(msg, "fail")
        self._passed = False


class _GpuRouteError(Exception):
    """Raised when we suspect GPU issues and should route to GPU test."""


def _truncate(s: str, n: int) -> str:
    s = s.replace("\n", " ").strip()
    return s[:n] + "..." if len(s) > n else s
