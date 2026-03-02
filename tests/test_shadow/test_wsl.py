"""WslRunner unit tests — mock subprocess calls."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from alchemy.shadow.wsl import RunResult, WslRunner


@pytest.fixture
def wsl():
    return WslRunner(distro="Ubuntu", display_num=99)


class TestRunResult:
    def test_ok_true(self):
        r = RunResult(returncode=0, stdout="hello", stderr="")
        assert r.ok is True

    def test_ok_false(self):
        r = RunResult(returncode=1, stdout="", stderr="error")
        assert r.ok is False


class TestIsAvailable:
    def test_available(self, wsl):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="ok\n")
            assert wsl.is_available() is True

    def test_not_available(self, wsl):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stdout="")
            assert wsl.is_available() is False

    def test_timeout(self, wsl):
        import subprocess as sp
        with patch("subprocess.run", side_effect=sp.TimeoutExpired("wsl", 10)):
            assert wsl.is_available() is False


class TestRun:
    @pytest.mark.asyncio
    async def test_run_success(self, wsl):
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"output\n", b""))
        mock_proc.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await wsl.run("echo hello")
            assert result.ok is True
            assert "output" in result.stdout

    @pytest.mark.asyncio
    async def test_run_with_display(self, wsl):
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"", b""))
        mock_proc.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec:
            await wsl.run("xdotool click 1", env_display=True)
            # Verify the command includes DISPLAY=:99
            call_args = mock_exec.call_args[0]
            bash_cmd = call_args[-1]  # last arg is the bash -c command
            assert "DISPLAY=:99" in bash_cmd

    @pytest.mark.asyncio
    async def test_run_without_display(self, wsl):
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"", b""))
        mock_proc.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec:
            await wsl.run("ls /tmp", env_display=False)
            call_args = mock_exec.call_args[0]
            bash_cmd = call_args[-1]
            assert "DISPLAY" not in bash_cmd

    @pytest.mark.asyncio
    async def test_run_failure(self, wsl):
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"", b"error msg"))
        mock_proc.returncode = 1

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await wsl.run("bad command")
            assert result.ok is False
            assert "error msg" in result.stderr


class TestReadFile:
    @pytest.mark.asyncio
    async def test_read_file(self, wsl):
        png_header = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(png_header, b""))
        mock_proc.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            data = await wsl.read_file("/tmp/screenshot.png")
            assert data[:4] == b"\x89PNG"
