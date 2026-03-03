"""ShadowDesktopController — orchestrates the shadow desktop lifecycle.

Uses WslRunner to start/stop Xvfb, Fluxbox, x11vnc, noVNC inside WSL2.
Provides screenshot capture with configurable format and resolution.
"""

from __future__ import annotations

import logging
import os

from alchemy.schemas import (
    ShadowHealthResponse,
    ShadowStartRequest,
    ShadowStartResponse,
    ShadowStatus,
    ShadowStopResponse,
)
from alchemy.shadow.wsl import WslRunner

logger = logging.getLogger(__name__)


class ShadowDesktopController:
    """Manage the shadow desktop lifecycle via WSL2."""

    def __init__(
        self,
        wsl: WslRunner,
        display_num: int = 99,
        vnc_port: int = 5900,
        novnc_port: int = 6080,
        resolution: str = "1920x1080x24",
        screenshot_format: str = "jpeg",
        screenshot_jpeg_quality: int = 85,
        screenshot_resize_width: int = 0,
        screenshot_resize_height: int = 0,
        repo_wsl_path: str = "/mnt/c/Users/info/GitHub/Alchemy",
    ):
        self.wsl = wsl
        self.display_num = display_num
        self.vnc_port = vnc_port
        self.novnc_port = novnc_port
        self.resolution = resolution
        self._screenshot_format = screenshot_format.lower()
        self._jpeg_quality = screenshot_jpeg_quality
        self._resize_width = screenshot_resize_width
        self._resize_height = screenshot_resize_height
        self._repo_wsl_path = repo_wsl_path
        self._screenshot_path = f"/tmp/alchemy_screenshot_{os.getpid()}"

    def _run_script(self, script_name: str, *args: str | int) -> str:
        """Build a command that runs a WSL script with CRLF stripping."""
        args_str = " ".join(str(a) for a in args)
        return f"cd {self._repo_wsl_path} && tr -d '\\r' < wsl/{script_name} | bash -s -- {args_str}"

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
        """Capture a screenshot from the shadow desktop.

        Returns image bytes in the configured format (JPEG or PNG).
        If resize dimensions are set, scales the image to reduce visual tokens.
        """
        use_jpeg = self._screenshot_format == "jpeg"
        ext = "jpg" if use_jpeg else "png"
        output_path = f"{self._screenshot_path}.{ext}"

        # Build capture + optional resize pipeline
        if self._resize_width > 0 and self._resize_height > 0:
            resize = f"{self._resize_width}x{self._resize_height}"
            if use_jpeg:
                cmd = (
                    f"scrot -o {self._screenshot_path}.png && "
                    f"convert {self._screenshot_path}.png "
                    f"-resize {resize} -quality {self._jpeg_quality} "
                    f"{output_path}"
                )
            else:
                cmd = (
                    f"scrot -o {self._screenshot_path}.png && "
                    f"convert {self._screenshot_path}.png -resize {resize} {output_path}"
                )
        elif use_jpeg:
            cmd = (
                f"scrot -o {self._screenshot_path}.png && "
                f"convert {self._screenshot_path}.png -quality {self._jpeg_quality} {output_path}"
            )
        else:
            cmd = f"scrot -o {output_path}"
            output_path = f"{self._screenshot_path}.png"

        result = await self.wsl.run(cmd, timeout=10.0)
        if not result.ok:
            # Fallback: try xwd + convert
            if use_jpeg:
                fallback = (
                    f"xwd -root -silent | convert xwd:- "
                    f"-quality {self._jpeg_quality} {output_path}"
                )
            else:
                fallback = f"xwd -root -silent | convert xwd:- png:{output_path}"
            result = await self.wsl.run(fallback, timeout=10.0)
            if not result.ok:
                raise RuntimeError(f"Screenshot capture failed: {result.stderr}")

        img_bytes = await self.wsl.read_file(output_path)

        if not img_bytes:
            raise RuntimeError("Screenshot capture returned empty data")
        if use_jpeg:
            if img_bytes[:2] != b"\xff\xd8":
                raise RuntimeError("Screenshot capture returned invalid JPEG data")
        else:
            if img_bytes[:4] != b"\x89PNG":
                raise RuntimeError("Screenshot capture returned invalid PNG data")

        return img_bytes

    async def execute(self, cmd: str) -> str:
        """Run an arbitrary command on the shadow desktop."""
        result = await self.wsl.run(cmd, timeout=15.0)
        return result.stdout
