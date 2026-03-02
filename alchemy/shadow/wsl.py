"""WslRunner — Windows → WSL2 bridge for shadow desktop commands.

Every shadow desktop operation (start Xvfb, capture screenshot, run xdotool)
goes through this class. It wraps subprocess calls to `wsl -d <distro>`.
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RunResult:
    """Result of a WSL command execution."""
    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0


class WslRunner:
    """Execute commands inside WSL2 from Windows."""

    def __init__(self, distro: str = "Ubuntu", display_num: int = 99):
        self.distro = distro
        self.display_num = display_num

    def is_available(self) -> bool:
        """Check if WSL2 and the target distro are reachable (sync, blocks up to 5s)."""
        try:
            result = subprocess.run(
                ["wsl", "-d", self.distro, "--", "echo", "ok"],
                capture_output=True, text=True, timeout=5,
            )
            return result.returncode == 0 and "ok" in result.stdout
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            return False

    async def run(self, cmd: str, timeout: float = 30.0, env_display: bool = True) -> RunResult:
        """Run a command inside WSL2 and wait for it to complete.

        Args:
            cmd: Shell command to execute inside WSL2.
            timeout: Max seconds to wait.
            env_display: If True, sets DISPLAY=:<display_num> before the command.
        """
        if env_display:
            full_cmd = f"DISPLAY=:{self.display_num} {cmd}"
        else:
            full_cmd = cmd

        logger.debug("WSL run: %s", full_cmd)

        try:
            proc = await asyncio.create_subprocess_exec(
                "wsl", "-d", self.distro, "--", "bash", "-c", full_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
            result = RunResult(
                returncode=proc.returncode or 0,
                stdout=stdout_bytes.decode("utf-8", errors="replace"),
                stderr=stderr_bytes.decode("utf-8", errors="replace"),
            )
            if not result.ok:
                logger.warning("WSL command failed (rc=%d): %s\n%s", result.returncode, cmd, result.stderr)
            return result

        except asyncio.TimeoutError:
            logger.error("WSL command timed out after %.1fs: %s", timeout, cmd)
            proc.kill()
            return RunResult(returncode=-1, stdout="", stderr=f"Timeout after {timeout}s")

    async def run_bg(self, cmd: str, env_display: bool = True) -> asyncio.subprocess.Process:
        """Start a command inside WSL2 in the background (non-blocking).

        Returns the subprocess — caller is responsible for cleanup.
        """
        if env_display:
            full_cmd = f"DISPLAY=:{self.display_num} {cmd}"
        else:
            full_cmd = cmd

        logger.debug("WSL run_bg: %s", full_cmd)

        proc = await asyncio.create_subprocess_exec(
            "wsl", "-d", self.distro, "--", "bash", "-c", full_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        return proc

    async def read_file(self, wsl_path: str) -> bytes:
        """Read a file from inside WSL2 and return its bytes.

        Args:
            wsl_path: Absolute path inside WSL2 (e.g., /tmp/screenshot.png).
        """
        result = await self.run(f"cat {wsl_path}", env_display=False)
        if not result.ok:
            raise FileNotFoundError(f"WSL file not found: {wsl_path} — {result.stderr}")
        # For binary files, we need raw bytes — re-run without text decoding
        proc = await asyncio.create_subprocess_exec(
            "wsl", "-d", self.distro, "--", "cat", wsl_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_bytes, _ = await proc.communicate()
        return stdout_bytes

    async def file_exists(self, wsl_path: str) -> bool:
        """Check if a file exists inside WSL2."""
        result = await self.run(f"test -f {wsl_path} && echo yes", env_display=False)
        return "yes" in result.stdout
