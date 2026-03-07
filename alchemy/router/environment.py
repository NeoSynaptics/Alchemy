"""Environment detection — discovers what's available on Windows.

Detects installed apps and system info from the Windows host.
Results are cached on startup and refreshed on demand.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Well-known apps on Windows (filtered from Get-StartApps output)
_WINDOWS_KEYWORDS = [
    "spotify", "chrome", "firefox", "edge", "discord", "slack", "teams",
    "vscode", "visual studio", "notepad", "terminal", "powershell",
    "outlook", "thunderbird", "vlc", "steam", "obs",
    "file explorer", "paint", "calculator", "photos",
]


@dataclass
class EnvironmentSnapshot:
    """Point-in-time snapshot of the execution environment."""

    os_type: str = "Windows"
    resolution: str = "1920x1080"
    windows_apps: list[str] = field(default_factory=list)
    windows_version: str = "Windows"

    @property
    def all_apps(self) -> list[str]:
        """All detected apps."""
        return sorted(set(self.windows_apps))

    def apps_for_category(self, keywords: list[str]) -> list[str]:
        """Find apps matching any of the given keywords."""
        result = []
        for app in self.all_apps:
            app_lower = app.lower()
            if any(kw.lower() in app_lower for kw in keywords):
                result.append(app)
        return result


class EnvironmentDetector:
    """Detects installed apps and system info from Windows."""

    async def detect(self) -> EnvironmentSnapshot:
        """Run full detection. Safe to call even if PowerShell unavailable."""
        windows_result = await self._detect_windows()

        snap = EnvironmentSnapshot()

        if isinstance(windows_result, dict):
            snap.windows_apps = windows_result.get("apps", [])
            snap.windows_version = windows_result.get("version", snap.windows_version)

        logger.info(
            "Environment detected: %d windows apps",
            len(snap.windows_apps),
        )
        return snap

    async def _detect_windows(self) -> dict:
        """Detect apps from Windows host via PowerShell."""
        result: dict = {}

        try:
            # Get Windows version
            ver_proc = await asyncio.create_subprocess_exec(
                "powershell", "-NoProfile", "-Command",
                "(Get-CimInstance Win32_OperatingSystem).Caption",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            ver_out, _ = await asyncio.wait_for(ver_proc.communicate(), timeout=10.0)
            result["version"] = ver_out.decode("utf-8", errors="replace").strip() or "Windows"
        except (asyncio.TimeoutError, FileNotFoundError, OSError):
            result["version"] = "Windows"

        try:
            # Get installed apps via Get-StartApps
            apps_proc = await asyncio.create_subprocess_exec(
                "powershell", "-NoProfile", "-Command",
                "Get-StartApps | Select-Object -ExpandProperty Name",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            apps_out, _ = await asyncio.wait_for(apps_proc.communicate(), timeout=15.0)
            raw_apps = apps_out.decode("utf-8", errors="replace").splitlines()

            # Filter to well-known apps (avoid noise from system entries)
            apps = []
            for app_name in raw_apps:
                app_name = app_name.strip()
                if not app_name:
                    continue
                app_lower = app_name.lower()
                if any(kw in app_lower for kw in _WINDOWS_KEYWORDS):
                    apps.append(app_name)
            result["apps"] = apps

        except (asyncio.TimeoutError, FileNotFoundError, OSError):
            result["apps"] = []

        return result
