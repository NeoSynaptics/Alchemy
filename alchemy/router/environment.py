"""Environment detection — discovers what's available on both platforms.

Detects installed apps and system info from the WSL2 shadow desktop
(where actions execute) and the Windows host (for awareness).
Results are cached on startup and refreshed on demand.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

from alchemy.shadow.wsl import WslRunner

logger = logging.getLogger(__name__)

# Well-known apps to probe for on the shadow desktop (Linux)
_SHADOW_PROBES = [
    "firefox", "chromium-browser", "google-chrome",
    "vlc", "totem", "rhythmbox", "audacious",
    "nautilus", "thunar", "pcmanfm",
    "xterm", "xfce4-terminal", "lxterminal", "gnome-terminal",
    "gedit", "mousepad", "nano", "vim",
    "libreoffice", "gimp", "inkscape",
    "rofi", "dmenu",
]

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

    # Shadow desktop (primary — where actions execute)
    os_type: str = "Ubuntu (WSL2)"
    desktop: str = "Fluxbox"
    resolution: str = "1920x1080"
    shadow_apps: list[str] = field(default_factory=list)
    taskbar_position: str = "bottom"
    default_browser: str = "firefox"
    has_audio: bool = False
    search_method: str = "fluxbox menu"

    # Windows host (secondary — for awareness)
    windows_apps: list[str] = field(default_factory=list)
    windows_version: str = "Windows"

    @property
    def all_apps(self) -> list[str]:
        """Combined app list from both platforms, deduplicated."""
        return sorted(set(self.shadow_apps + self.windows_apps))

    def apps_for_category(self, keywords: list[str]) -> list[str]:
        """Find apps matching any of the given keywords."""
        result = []
        for app in self.all_apps:
            app_lower = app.lower()
            if any(kw.lower() in app_lower for kw in keywords):
                result.append(app)
        return result


class EnvironmentDetector:
    """Detects installed apps and system info from both platforms."""

    def __init__(self, wsl: WslRunner | None = None):
        self._wsl = wsl

    async def detect(self) -> EnvironmentSnapshot:
        """Run full detection. Safe to call even if WSL/PowerShell unavailable."""
        shadow_task = self._detect_shadow()
        windows_task = self._detect_windows()

        shadow_result, windows_result = await asyncio.gather(
            shadow_task, windows_task, return_exceptions=True
        )

        snap = EnvironmentSnapshot()

        if isinstance(shadow_result, dict):
            snap.os_type = shadow_result.get("os_type", snap.os_type)
            snap.shadow_apps = shadow_result.get("apps", [])
            snap.default_browser = shadow_result.get("browser", snap.default_browser)
            snap.has_audio = shadow_result.get("has_audio", False)
            snap.search_method = shadow_result.get("search_method", snap.search_method)
        else:
            logger.warning("Shadow detection failed: %s", shadow_result)

        if isinstance(windows_result, dict):
            snap.windows_apps = windows_result.get("apps", [])
            snap.windows_version = windows_result.get("version", snap.windows_version)
        else:
            logger.warning("Windows detection failed: %s", windows_result)

        logger.info(
            "Environment detected: %d shadow apps, %d windows apps",
            len(snap.shadow_apps), len(snap.windows_apps),
        )
        return snap

    async def _detect_shadow(self) -> dict:
        """Detect apps and config from WSL2 shadow desktop."""
        if not self._wsl:
            return {"apps": [], "os_type": "Unknown (no WSL)"}

        result: dict = {}

        # OS type
        os_result = await self._wsl.run(
            "cat /etc/os-release 2>/dev/null | grep PRETTY_NAME | cut -d'\"' -f2",
            env_display=False,
        )
        result["os_type"] = os_result.stdout.strip() or "Ubuntu (WSL2)"

        # Probe for installed apps
        apps = []
        probe_cmd = " && ".join(
            f"which {app} >/dev/null 2>&1 && echo {app}" for app in _SHADOW_PROBES
        )
        probe_result = await self._wsl.run(probe_cmd, env_display=False)
        if probe_result.ok:
            apps = [line.strip() for line in probe_result.stdout.splitlines() if line.strip()]
        result["apps"] = apps

        # Default browser
        browser_result = await self._wsl.run(
            "xdg-settings get default-web-browser 2>/dev/null | sed 's/.desktop//'",
            env_display=False,
        )
        browser = browser_result.stdout.strip()
        if browser:
            result["browser"] = browser

        # PulseAudio check
        audio_result = await self._wsl.run("pactl info >/dev/null 2>&1 && echo yes", env_display=False)
        result["has_audio"] = "yes" in audio_result.stdout

        # Search method (rofi > dmenu > fluxbox menu)
        if "rofi" in apps:
            result["search_method"] = "rofi (Alt+F2 or run rofi -show run)"
        elif "dmenu" in apps:
            result["search_method"] = "dmenu (Alt+F2 or run dmenu_run)"
        else:
            result["search_method"] = "fluxbox menu (right-click desktop)"

        return result

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
