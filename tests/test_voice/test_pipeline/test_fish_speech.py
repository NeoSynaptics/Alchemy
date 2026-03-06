"""Tests for FishSpeechProcess — subprocess lifecycle with mocked Popen."""

import subprocess
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from alchemy.voice.fish_speech import FishSpeechProcess


class TestFishSpeechProcess:
    def test_init_defaults(self):
        proc = FishSpeechProcess()
        assert proc.port == 8080
        assert proc.is_running is False
        assert proc.is_healthy is False

    def test_base_url(self):
        proc = FishSpeechProcess(port=9000, listen_host="0.0.0.0")
        assert proc.base_url == "http://0.0.0.0:9000"

    def test_build_command(self):
        proc = FishSpeechProcess(
            port=8085,
            checkpoint_path="ckpt/fish",
            decoder_config="firefly_gan_vq",
        )
        cmd = proc._build_command()
        assert sys.executable in cmd[0]
        assert "--listen" in cmd
        assert "127.0.0.1:8085" in cmd
        assert "--llama-checkpoint-path" in cmd
        assert "ckpt/fish" in cmd

    def test_build_command_with_compile(self):
        proc = FishSpeechProcess(compile=True)
        cmd = proc._build_command()
        assert "--compile" in cmd

    def test_build_command_without_compile(self):
        proc = FishSpeechProcess(compile=False)
        cmd = proc._build_command()
        assert "--compile" not in cmd

    async def test_start_launches_subprocess(self):
        proc = FishSpeechProcess(startup_timeout=2.0)

        mock_popen = MagicMock()
        mock_popen.poll.return_value = None  # Process is running
        mock_popen.pid = 12345

        import httpx

        mock_response = httpx.Response(200, request=httpx.Request("GET", "http://test/"))

        with patch("subprocess.Popen", return_value=mock_popen):
            with patch("httpx.AsyncClient") as mock_client_cls:
                mock_client = AsyncMock()
                mock_client.get = AsyncMock(return_value=mock_response)
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=False)
                mock_client_cls.return_value = mock_client

                await proc.start()

        assert proc.is_healthy is True

    async def test_start_already_running_is_noop(self):
        proc = FishSpeechProcess()
        mock_popen = MagicMock()
        mock_popen.poll.return_value = None
        proc._process = mock_popen

        # Should return without starting a new process
        await proc.start()
        assert proc._process is mock_popen

    async def test_stop_kills_process(self):
        proc = FishSpeechProcess()
        mock_popen = MagicMock()
        mock_popen.poll.return_value = None
        mock_popen.pid = 99999
        proc._process = mock_popen
        proc._healthy = True

        with patch("subprocess.run") as mock_run:
            await proc.stop()
            if sys.platform == "win32":
                mock_run.assert_called_once()
                call_args = mock_run.call_args[0][0]
                assert "taskkill" in call_args
                assert "99999" in call_args

        assert proc._process is None
        assert proc.is_healthy is False

    async def test_stop_when_not_running_is_noop(self):
        proc = FishSpeechProcess()
        await proc.stop()  # Should not raise

    async def test_start_timeout_raises(self):
        proc = FishSpeechProcess(startup_timeout=0.5)

        mock_popen = MagicMock()
        mock_popen.poll.return_value = None

        import httpx

        with patch("subprocess.Popen", return_value=mock_popen):
            with patch("httpx.AsyncClient") as mock_client_cls:
                mock_client = AsyncMock()
                mock_client.get = AsyncMock(side_effect=httpx.ConnectError("refused"))
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=False)
                mock_client_cls.return_value = mock_client

                with pytest.raises(TimeoutError, match="did not start"):
                    await proc.start()

    async def test_start_process_exits_early_raises(self):
        proc = FishSpeechProcess(startup_timeout=2.0)

        mock_popen = MagicMock()
        mock_popen.poll.return_value = 1  # Exited with error
        mock_popen.returncode = 1

        with patch("subprocess.Popen", return_value=mock_popen):
            with pytest.raises(RuntimeError, match="exited with code 1"):
                await proc.start()

    async def test_health_check_success(self):
        proc = FishSpeechProcess()

        import httpx

        mock_response = httpx.Response(200, request=httpx.Request("GET", "http://test/"))

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await proc.health_check()
            assert result is True

    async def test_health_check_failure(self):
        proc = FishSpeechProcess()

        import httpx

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(side_effect=httpx.ConnectError("refused"))
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await proc.health_check()
            assert result is False
