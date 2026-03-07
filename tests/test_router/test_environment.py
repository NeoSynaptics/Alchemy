"""Tests for environment detection — all PowerShell calls mocked."""

from unittest.mock import AsyncMock, patch

import pytest

from alchemy.router.environment import EnvironmentDetector, EnvironmentSnapshot


class TestEnvironmentSnapshot:
    def test_all_apps_deduplicates(self):
        snap = EnvironmentSnapshot(
            windows_apps=["firefox", "spotify", "firefox"],
        )
        assert sorted(snap.all_apps) == ["firefox", "spotify"]

    def test_all_apps_empty(self):
        snap = EnvironmentSnapshot()
        assert snap.all_apps == []

    def test_apps_for_category(self):
        snap = EnvironmentSnapshot(
            windows_apps=["spotify", "discord", "vlc", "firefox"],
        )
        media = snap.apps_for_category(["spotify", "vlc", "music"])
        assert "vlc" in media
        assert "spotify" in media
        assert "firefox" not in media

    def test_apps_for_category_case_insensitive(self):
        snap = EnvironmentSnapshot(windows_apps=["Spotify"])
        assert snap.apps_for_category(["spotify"]) == ["Spotify"]

    def test_default_fields(self):
        snap = EnvironmentSnapshot()
        assert snap.os_type == "Windows"
        assert snap.resolution == "1920x1080"
        assert snap.windows_apps == []
        assert snap.windows_version == "Windows"


class TestEnvironmentDetector:
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

            detector = EnvironmentDetector()
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

            detector = EnvironmentDetector()
            snap = await detector.detect()

            # Should not crash, just return empty
            assert snap.windows_apps == []
            assert snap.windows_version == "Windows"

    async def test_detect_powershell_not_found_graceful(self):
        with patch("alchemy.router.environment.asyncio.create_subprocess_exec") as mock_exec:
            mock_exec.side_effect = FileNotFoundError("no powershell")

            detector = EnvironmentDetector()
            snap = await detector.detect()

            # Should not crash
            assert isinstance(snap, EnvironmentSnapshot)
            assert snap.windows_apps == []
