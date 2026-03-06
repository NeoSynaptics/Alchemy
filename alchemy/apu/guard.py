"""AlchemyGuard — health-check and auto-start for the Alchemy server.

Any app that depends on Alchemy as its base MUST use AlchemyGuard to ensure
the Alchemy server is running before making requests.

Usage (in any dependent app):
    from alchemy.apu.guard import AlchemyGuard

    guard = AlchemyGuard()          # uses default http://localhost:8000
    await guard.ensure_running()    # blocks until Alchemy is up (or raises)

    # Or run as a background loop (keeps checking every N seconds):
    await guard.start_watchdog()    # non-blocking, spawns background task
    ...
    await guard.stop_watchdog()
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
import sys
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

# Default Alchemy server location
_ALCHEMY_ROOT = Path(__file__).resolve().parent.parent.parent  # alchemy/apu/guard.py -> repo root
_DEFAULT_HOST = "http://localhost:8000"
_HEALTH_ENDPOINT = "/v1/apu/status"
_PING_TIMEOUT = 5.0
_START_TIMEOUT = 60.0
_WATCHDOG_INTERVAL = 30.0
_MAX_RETRIES = 12  # 12 * 5s = 60s


class AlchemyGuard:
    """Ensures Alchemy server is running. Auto-starts it if not.

    Parameters:
        host: Alchemy server URL (default: http://localhost:8000)
        auto_start: If True, attempt to launch Alchemy when it's down
        watchdog_interval: Seconds between health checks in watchdog mode
        server_command: Command to start Alchemy (default: make server in repo root)
    """

    def __init__(
        self,
        host: str = _DEFAULT_HOST,
        *,
        auto_start: bool = True,
        watchdog_interval: float = _WATCHDOG_INTERVAL,
        server_command: list[str] | None = None,
    ) -> None:
        self._host = host.rstrip("/")
        self._auto_start = auto_start
        self._watchdog_interval = watchdog_interval
        self._server_command = server_command or self._default_command()
        self._watchdog_task: asyncio.Task | None = None
        self._server_process: subprocess.Popen | None = None

    @staticmethod
    def _default_command() -> list[str]:
        """Default command to start Alchemy server."""
        if sys.platform == "win32":
            return [sys.executable, "-m", "uvicorn", "alchemy.server:app",
                    "--host", "0.0.0.0", "--port", "8000"]
        return ["make", "-C", str(_ALCHEMY_ROOT), "server"]

    # --- Health Check ---

    async def is_alive(self) -> bool:
        """Check if Alchemy server is responding."""
        try:
            async with httpx.AsyncClient(timeout=_PING_TIMEOUT) as client:
                resp = await client.get(f"{self._host}{_HEALTH_ENDPOINT}")
                return resp.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException, httpx.ReadError):
            return False

    async def wait_until_ready(self, timeout: float = _START_TIMEOUT) -> bool:
        """Poll until Alchemy responds or timeout is reached."""
        elapsed = 0.0
        interval = 2.0
        while elapsed < timeout:
            if await self.is_alive():
                logger.info("Alchemy server is ready at %s", self._host)
                return True
            await asyncio.sleep(interval)
            elapsed += interval
        logger.error("Alchemy server did not become ready within %.0fs", timeout)
        return False

    # --- Auto-Start ---

    def _start_server(self) -> None:
        """Launch Alchemy server as a detached subprocess."""
        if self._server_process and self._server_process.poll() is None:
            logger.debug("Alchemy server process already running (pid=%d)", self._server_process.pid)
            return

        logger.info("Starting Alchemy server: %s", " ".join(self._server_command))
        self._server_process = subprocess.Popen(
            self._server_command,
            cwd=str(_ALCHEMY_ROOT),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
        )
        logger.info("Alchemy server started (pid=%d)", self._server_process.pid)

    async def ensure_running(self) -> bool:
        """Check if Alchemy is running. If not, start it and wait for readiness.

        Returns True if Alchemy is confirmed running. Raises RuntimeError if
        it cannot be started within the timeout.
        """
        if await self.is_alive():
            return True

        if not self._auto_start:
            raise RuntimeError(
                f"Alchemy server is not running at {self._host} and auto_start is disabled"
            )

        logger.warning("Alchemy server not responding at %s — force starting", self._host)
        self._start_server()

        if not await self.wait_until_ready():
            raise RuntimeError(
                f"Alchemy server failed to start within {_START_TIMEOUT}s. "
                f"Command: {' '.join(self._server_command)}"
            )
        return True

    # --- Watchdog (Background Loop) ---

    async def start_watchdog(self) -> None:
        """Start a background task that monitors Alchemy health."""
        if self._watchdog_task and not self._watchdog_task.done():
            logger.debug("Watchdog already running")
            return

        self._watchdog_task = asyncio.create_task(self._watchdog_loop())
        logger.info("AlchemyGuard watchdog started (interval=%.0fs)", self._watchdog_interval)

    async def stop_watchdog(self) -> None:
        """Stop the background watchdog."""
        if self._watchdog_task:
            self._watchdog_task.cancel()
            try:
                await self._watchdog_task
            except asyncio.CancelledError:
                pass
            self._watchdog_task = None
            logger.info("AlchemyGuard watchdog stopped")

    async def _watchdog_loop(self) -> None:
        """Periodically check Alchemy health and restart if needed."""
        while True:
            try:
                if not await self.is_alive():
                    logger.warning("AlchemyGuard: server down — attempting restart")
                    try:
                        await self.ensure_running()
                    except RuntimeError as e:
                        logger.error("AlchemyGuard: restart failed — %s", e)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("AlchemyGuard: unexpected error in watchdog")

            await asyncio.sleep(self._watchdog_interval)
