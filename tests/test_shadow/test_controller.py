"""ShadowDesktopController unit tests — mock WslRunner."""

from unittest.mock import AsyncMock

import pytest

from alchemy.schemas import ShadowStatus
from alchemy.shadow.controller import ShadowDesktopController
from alchemy.shadow.wsl import RunResult, WslRunner


@pytest.fixture
def mock_wsl():
    wsl = AsyncMock(spec=WslRunner)
    wsl.distro = "Ubuntu"
    wsl.display_num = 99
    return wsl


@pytest.fixture
def controller(mock_wsl):
    return ShadowDesktopController(
        wsl=mock_wsl,
        display_num=99,
        vnc_port=5900,
        novnc_port=6080,
        resolution="1920x1080x24",
    )


class TestStart:
    @pytest.mark.asyncio
    async def test_start_success(self, controller, mock_wsl):
        mock_wsl.run.return_value = RunResult(
            returncode=0,
            stdout="=== Shadow Desktop Running ===",
            stderr="",
        )
        result = await controller.start()
        assert result.status == ShadowStatus.RUNNING
        assert result.display == ":99"
        assert "6080" in result.novnc_url

    @pytest.mark.asyncio
    async def test_start_failure(self, controller, mock_wsl):
        mock_wsl.run.return_value = RunResult(
            returncode=1, stdout="", stderr="Xvfb failed",
        )
        result = await controller.start()
        assert result.status == ShadowStatus.ERROR


class TestStop:
    @pytest.mark.asyncio
    async def test_stop(self, controller, mock_wsl):
        mock_wsl.run.return_value = RunResult(
            returncode=0, stdout="=== Shadow Desktop Stopped ===", stderr="",
        )
        result = await controller.stop()
        assert result.status == ShadowStatus.STOPPED


class TestHealth:
    @pytest.mark.asyncio
    async def test_health_all_ok(self, controller, mock_wsl):
        mock_wsl.run.return_value = RunResult(
            returncode=0,
            stdout=(
                "  [OK] Xvfb on :99\n"
                "  [OK] Fluxbox\n"
                "  [OK] x11vnc on port 5900\n"
                "  [OK] noVNC on port 6080\n"
                "=== All services healthy ==="
            ),
            stderr="",
        )
        result = await controller.health()
        assert result.status == ShadowStatus.RUNNING
        assert result.xvfb_running is True
        assert result.fluxbox_running is True
        assert result.vnc_running is True
        assert result.novnc_running is True

    @pytest.mark.asyncio
    async def test_health_partial(self, controller, mock_wsl):
        mock_wsl.run.return_value = RunResult(
            returncode=1,
            stdout=(
                "  [OK] Xvfb on :99\n"
                "  [FAIL] Fluxbox not running\n"
                "  [OK] x11vnc on port 5900\n"
                "  [FAIL] noVNC not running\n"
            ),
            stderr="",
        )
        result = await controller.health()
        assert result.status == ShadowStatus.STOPPED
        assert result.xvfb_running is True
        assert result.fluxbox_running is False
        assert result.vnc_running is True
        assert result.novnc_running is False


class TestScreenshot:
    @pytest.mark.asyncio
    async def test_screenshot_success(self, controller, mock_wsl):
        png_data = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        mock_wsl.run.return_value = RunResult(returncode=0, stdout="", stderr="")
        mock_wsl.read_file.return_value = png_data

        result = await controller.screenshot()
        assert result[:4] == b"\x89PNG"

    @pytest.mark.asyncio
    async def test_screenshot_failure(self, controller, mock_wsl):
        mock_wsl.run.return_value = RunResult(
            returncode=1, stdout="", stderr="scrot not found",
        )
        mock_wsl.read_file.return_value = b""

        with pytest.raises(RuntimeError):
            await controller.screenshot()


class TestExecute:
    @pytest.mark.asyncio
    async def test_execute(self, controller, mock_wsl):
        mock_wsl.run.return_value = RunResult(
            returncode=0, stdout="1920 1080\n", stderr="",
        )
        result = await controller.execute("xdotool getdisplaygeometry")
        assert "1920" in result
