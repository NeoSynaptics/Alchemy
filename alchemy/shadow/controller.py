"""ShadowDesktopController — orchestrates the shadow desktop lifecycle.

Uses WslRunner to start/stop Xvfb, Fluxbox, x11vnc, noVNC inside WSL2.
Provides screenshot capture and health checks.
"""

from __future__ import annotations

import logging

from alchemy.schemas import (
    ShadowHealthResponse,
    ShadowStartRequest,
    ShadowStartResponse,
    ShadowStatus,
    ShadowStopResponse,
)
from alchemy.shadow.wsl import WslRunner

logger = logging.getLogger(__name__)

# Path to shadow desktop scripts (as seen from WSL2 via /mnt/c/...)
_REPO_WSL_PATH = "/mnt/c/Users/info/GitHub/Alchemy"
_SCREENSHOT_PATH = "/tmp/alchemy_screenshot.png"


class ShadowDesktopController:
    """Manage the shadow desktop lifecycle via WSL2."""

    def __init__(
        self,
        wsl: WslRunner,
        display_num: int = 99,
        vnc_port: int = 5900,
        novnc_port: int = 6080,
        resolution: str = "1920x1080x24",
    ):
        self.wsl = wsl
        self.display_num = display_num
        self.vnc_port = vnc_port
        self.novnc_port = novnc_port
        self.resolution = resolution

    def _run_script(self, script_name: str, *args: str | int) -> str:
        """Build a command that runs a WSL script with CRLF stripping.

        Scripts live on /mnt/c/ (Windows FS) and may have \\r\\n line endings.
        Pipe through tr to fix before executing.
        """
        args_str = " ".join(str(a) for a in args)
        return f"cd {_REPO_WSL_PATH} && tr -d '\\r' < wsl/{script_name} | bash -s -- {args_str}"

    async def start(self, req: ShadowStartRequest | None = None) -> ShadowStartResponse:
        """Start the shadow desktop (Xvfb + Fluxbox + x11vnc + noVNC)."""
        if req:
            self.display_num = req.display_num
            self.resolution = req.resolution
            self.wsl.display_num = req.display_num

        cmd = self._run_script("start_shadow.sh", self.display_num, self.vnc_port, self.novnc_port, self.resolution)
        result = await self.wsl.run(cmd, timeout=30.0, env_display=False)

        if not result.ok:
            logger.error("Shadow desktop start failed: %s", result.stderr)
            return ShadowStartResponse(
                status=ShadowStatus.ERROR,
                display=f":{self.display_num}",
                vnc_url=f"localhost:{self.vnc_port}",
                novnc_url="",
            )

        return ShadowStartResponse(
            status=ShadowStatus.RUNNING,
            display=f":{self.display_num}",
            vnc_url=f"localhost:{self.vnc_port}",
            novnc_url=f"http://localhost:{self.novnc_port}/vnc.html?autoconnect=true",
        )

    async def stop(self) -> ShadowStopResponse:
        """Stop all shadow desktop services."""
        cmd = self._run_script("stop_shadow.sh", self.display_num)
        result = await self.wsl.run(cmd, timeout=15.0, env_display=False)

        if not result.ok:
            logger.warning("Shadow desktop stop may have failed: %s", result.stderr)

        return ShadowStopResponse(
            status=ShadowStatus.STOPPED,
            message="Shadow desktop stopped",
        )

    async def health(self) -> ShadowHealthResponse:
        """Check if all shadow desktop services are running."""
        cmd = self._run_script("health_check.sh", self.display_num, self.vnc_port, self.novnc_port)
        result = await self.wsl.run(cmd, timeout=10.0, env_display=False)

        # Parse [OK] / [FAIL] lines from health_check.sh output
        output = result.stdout
        xvfb = "[OK] Xvfb" in output
        fluxbox = "[OK] Fluxbox" in output
        vnc = "[OK] x11vnc" in output
        novnc = "[OK] noVNC" in output

        all_ok = xvfb and fluxbox and vnc and novnc
        status = ShadowStatus.RUNNING if all_ok else ShadowStatus.STOPPED

        return ShadowHealthResponse(
            status=status,
            xvfb_running=xvfb,
            fluxbox_running=fluxbox,
            vnc_running=vnc,
            novnc_running=novnc,
            display=f":{self.display_num}",
        )

    async def screenshot(self) -> bytes:
        """Capture a screenshot from the shadow desktop. Returns PNG bytes."""
        # Use scrot to capture the display
        result = await self.wsl.run(
            f"scrot -o {_SCREENSHOT_PATH}",
            timeout=10.0,
        )
        if not result.ok:
            # Fallback: try xwd + convert
            result = await self.wsl.run(
                f"xwd -root -silent | convert xwd:- png:{_SCREENSHOT_PATH}",
                timeout=10.0,
            )
            if not result.ok:
                raise RuntimeError(f"Screenshot capture failed: {result.stderr}")

        # Read the PNG file back
        png_bytes = await self.wsl.read_file(_SCREENSHOT_PATH)
        if not png_bytes or png_bytes[:4] != b"\x89PNG":
            raise RuntimeError("Screenshot capture returned invalid data")

        return png_bytes

    async def execute(self, cmd: str) -> str:
        """Run an arbitrary command on the shadow desktop.

        The DISPLAY variable is set automatically.
        Use this for xdotool, launching apps, etc.
        """
        result = await self.wsl.run(cmd, timeout=15.0)
        return result.stdout
