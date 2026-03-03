"""Tests for environment detection — all WSL/PowerShell calls mocked."""

from unittest.mock import AsyncMock, patch

import pytest

from alchemy.router.environment import EnvironmentDetector, EnvironmentSnapshot
from alchemy.shadow.wsl import RunResult


class TestEnvironmentSnapshot:
    def test_all_apps_deduplicates(self):
        snap = EnvironmentSnapshot(
            shadow_apps=["firefox", "vlc"],
            windows_apps=["firefox", "spotify"],
        )
        assert sorted(snap.all_apps) == ["firefox", "spotify", "vlc"]

    def test_all_apps_empty(self):
        snap = EnvironmentSnapshot()
        assert snap.all_apps == []

    def test_apps_for_category(self):
        snap = EnvironmentSnapshot(
            shadow_apps=["vlc", "firefox"],
            windows_apps=["spotify", "discord"],
        )
        media = snap.apps_for_category(["spotify", "vlc", "music"])
        assert "vlc" in media
        assert "spotify" in media
        assert "firefox" not in media

    def test_apps_for_category_case_insensitive(self):
        snap = EnvironmentSnapshot(windows_apps=["Spotify"])
        assert snap.apps_for_category(["spotify"]) == ["Spotify"]


class TestEnvironmentDetector:
    @pytest.fixture
    def mock_wsl(self):
        wsl = AsyncMock()
        wsl.run = AsyncMock()
        return wsl

    async def test_detect_shadow_apps(self, mock_wsl):
        # OS
        mock_wsl.run.side_effect = [
            RunResult(0, "Ubuntu 22.04.3 LTS", ""),       # os-release
            RunResult(0, "firefox\nvlc\nxterm\n", ""),     # which probes
            RunResult(0, "firefox", ""),                    # xdg-settings
            RunResult(0, "yes", ""),                        # pactl
        ]

        detector = EnvironmentDetector(wsl=mock_wsl)
        snap = await detector.detect()

        assert "Ubuntu" in snap.os_type
        assert "firefox" in snap.shadow_apps
        assert "vlc" in snap.shadow_apps
        assert snap.default_browser == "firefox"
        assert snap.has_audio is True

    async def test_detect_without_wsl(self):
        detector = EnvironmentDetector(wsl=None)
        snap = await detector.detect()

        assert snap.shadow_apps == []
        assert "no WSL" in snap.os_type or snap.os_type == "Ubuntu (WSL2)"

    async def test_detect_windows_apps(self):
        with patch("alchemy.router.environment.asyncio.create_subprocess_exec") as mock_exec:
            # Mock both PowerShell calls
            ver_proc = AsyncMock()
            ver_proc.communicate = AsyncMock(
                return_value=(b"Microsoft Windows 11 Pro\n", b"")
            )

            apps_proc = AsyncMock()
            apps_proc.communicate = AsyncMock(
                return_value=(b"Spotify\nGoogle Chrome\nNotepad\nSome Random App\n", b"")
            )

            mock_exec.side_effect = [ver_proc, apps_proc]

            detector = EnvironmentDetector(wsl=None)
            snap = await detector.detect()

            assert "Windows 11" in snap.windows_version
            assert "Spotify" in snap.windows_apps
            assert "Google Chrome" in snap.windows_apps
            # "Some Random App" should be filtered out
            assert "Some Random App" not in snap.windows_apps

    async def test_detect_windows_timeout_graceful(self):
        import asyncio

        with patch("alchemy.router.environment.asyncio.create_subprocess_exec") as mock_exec:
            mock_exec.side_effect = asyncio.TimeoutError()

            detector = EnvironmentDetector(wsl=None)
            snap = await detector.detect()

            # Should not crash, just return empty
            assert snap.windows_apps == []
            assert snap.windows_version == "Windows"

    async def test_detect_shadow_failure_graceful(self, mock_wsl):
        mock_wsl.run.side_effect = Exception("WSL not available")

        with patch("alchemy.router.environment.asyncio.create_subprocess_exec") as mock_exec:
            mock_exec.side_effect = FileNotFoundError("no powershell")

            detector = EnvironmentDetector(wsl=mock_wsl)
            snap = await detector.detect()

            # Should not crash
            assert isinstance(snap, EnvironmentSnapshot)
