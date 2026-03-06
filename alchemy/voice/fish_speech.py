"""Fish Speech subprocess manager — start/stop/health check.

Runs Fish Speech as a separate OS process for clean VRAM isolation.
Killing the process instantly frees all GPU memory it held.

The subprocess uses its own Python environment (e.g., a dedicated conda env)
to avoid dependency conflicts with AlchemyVoice.
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
import sys
import time

logger = logging.getLogger(__name__)


class FishSpeechProcess:
    """Manages the Fish Speech API server as a subprocess."""

    def __init__(
        self,
        port: int = 8080,
        checkpoint_path: str = "checkpoints/openaudio-s1-mini",
        decoder_path: str | None = None,
        decoder_config: str = "modded_dac_vq",
        listen_host: str = "127.0.0.1",
        startup_timeout: float = 60.0,
        compile: bool = False,
        python_exe: str | None = None,
        working_dir: str | None = None,
    ) -> None:
        self._port = port
        self._checkpoint_path = checkpoint_path
        self._decoder_path = decoder_path or f"{checkpoint_path}/codec.pth"
        self._decoder_config = decoder_config
        self._listen_host = listen_host
        self._startup_timeout = startup_timeout
        self._compile = compile
        self._python_exe = python_exe or sys.executable
        self._working_dir = working_dir
        self._process: subprocess.Popen | None = None
        self._healthy = False

    @property
    def port(self) -> int:
        return self._port

    @property
    def base_url(self) -> str:
        return f"http://{self._listen_host}:{self._port}"

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    @property
    def is_healthy(self) -> bool:
        return self._healthy and self.is_running

    async def start(self) -> None:
        """Start Fish Speech subprocess and wait for health check."""
        if self.is_running:
            return

        cmd = self._build_command()
        logger.info("Starting Fish Speech: %s", " ".join(cmd))

        loop = asyncio.get_event_loop()
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
        self._process = await loop.run_in_executor(
            None,
            lambda: subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                cwd=self._working_dir,
                creationflags=creationflags,
            ),
        )

        await self._wait_for_ready()
        self._healthy = True
        logger.info("Fish Speech ready on port %d (PID=%d)", self._port, self._process.pid)

    async def stop(self) -> None:
        """Kill the Fish Speech subprocess (frees all VRAM instantly)."""
        if not self._process:
            return

        pid = self._process.pid
        logger.info("Stopping Fish Speech (PID=%d)", pid)

        loop = asyncio.get_event_loop()
        if sys.platform == "win32":
            await loop.run_in_executor(
                None,
                lambda: subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(pid)],
                    capture_output=True,
                    timeout=10,
                ),
            )
        else:
            import signal

            self._process.send_signal(signal.SIGTERM)
            try:
                await asyncio.wait_for(
                    loop.run_in_executor(None, self._process.wait),
                    timeout=5.0,
                )
            except asyncio.TimeoutError:
                self._process.kill()

        self._process = None
        self._healthy = False
        await asyncio.sleep(0.5)
        logger.info("Fish Speech stopped, VRAM freed")

    async def health_check(self) -> bool:
        """Check if the Fish Speech server is responsive."""
        import httpx

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(f"{self.base_url}/v1/health", timeout=2.0)
                return resp.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException):
            return False

    async def _wait_for_ready(self) -> None:
        """Poll health endpoint until server is ready or timeout."""
        import httpx

        deadline = time.monotonic() + self._startup_timeout
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                raise RuntimeError(
                    f"Fish Speech process exited with code {self._process.returncode}"
                )
            try:
                async with httpx.AsyncClient() as client:
                    resp = await client.get(f"{self.base_url}/v1/health", timeout=2.0)
                    if resp.status_code == 200:
                        return
            except (httpx.ConnectError, httpx.TimeoutException):
                pass
            await asyncio.sleep(1.0)

        raise TimeoutError(f"Fish Speech did not start within {self._startup_timeout}s")

    def _build_command(self) -> list[str]:
        """Build the subprocess command for Fish Speech API server."""
        cmd = [
            self._python_exe,
            "tools/api_server.py",
            "--listen",
            f"{self._listen_host}:{self._port}",
            "--llama-checkpoint-path",
            self._checkpoint_path,
            "--decoder-checkpoint-path",
            self._decoder_path,
            "--decoder-config-name",
            self._decoder_config,
        ]
        if self._compile:
            cmd.append("--compile")
        return cmd
