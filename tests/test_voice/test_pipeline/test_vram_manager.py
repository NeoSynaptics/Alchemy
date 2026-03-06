"""Tests for VRAMManager — state transitions, lock, dual-GPU no-op."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from alchemy.voice.vram_manager import GPUMode, VRAMManager, VRAMSlot


class TestVRAMManagerInit:
    def test_defaults(self):
        mgr = VRAMManager()
        assert mgr.mode == GPUMode.SINGLE
        assert mgr.current_slot == VRAMSlot.EMPTY
        assert mgr.is_active is True

    def test_dual_mode(self):
        mgr = VRAMManager(mode=GPUMode.DUAL)
        assert mgr.is_active is False

    async def test_start_close(self):
        mgr = VRAMManager()
        await mgr.start()
        assert mgr._client is not None
        await mgr.close()
        assert mgr._client is None


class TestVRAMManagerSingleGPU:
    async def _make_mgr(self):
        mgr = VRAMManager(
            mode=GPUMode.SINGLE,
            ollama_host="http://test-ollama:11434",
            gpu_model="qwen3:14b",
        )
        # Mock the httpx client
        mock_client = AsyncMock()
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        mgr._client = mock_client
        return mgr, mock_client

    async def test_ensure_stt(self):
        mgr, _ = await self._make_mgr()
        mock_stt = AsyncMock()

        await mgr.ensure_stt(mock_stt)

        assert mgr.current_slot == VRAMSlot.WHISPER
        mock_stt.load.assert_called_once()

    async def test_ensure_stt_unloads_qwen3_first(self):
        mgr, mock_client = await self._make_mgr()
        mgr._current_slot = VRAMSlot.QWEN3  # Qwen3 is resident

        mock_stt = AsyncMock()
        await mgr.ensure_stt(mock_stt)

        # Should have called Ollama to unload Qwen3
        assert mock_client.post.call_count >= 1
        unload_call = mock_client.post.call_args_list[0]
        payload = unload_call[1]["json"]
        assert payload["keep_alive"] == "0"

        assert mgr.current_slot == VRAMSlot.WHISPER

    async def test_release_stt(self):
        mgr, _ = await self._make_mgr()
        mgr._current_slot = VRAMSlot.WHISPER

        mock_stt = AsyncMock()
        await mgr.release_stt(mock_stt)

        assert mgr.current_slot == VRAMSlot.EMPTY
        mock_stt.unload.assert_called_once()

    async def test_ensure_llm(self):
        mgr, mock_client = await self._make_mgr()

        await mgr.ensure_llm()

        assert mgr.current_slot == VRAMSlot.QWEN3
        # Should have called Ollama to preload
        preload_call = mock_client.post.call_args_list[-1]
        payload = preload_call[1]["json"]
        assert payload["keep_alive"] == "30m"
        assert payload["model"] == "qwen3:14b"

    async def test_release_llm(self):
        mgr, mock_client = await self._make_mgr()
        mgr._current_slot = VRAMSlot.QWEN3

        await mgr.release_llm()

        assert mgr.current_slot == VRAMSlot.EMPTY
        unload_call = mock_client.post.call_args_list[-1]
        payload = unload_call[1]["json"]
        assert payload["keep_alive"] == "0"

    async def test_ensure_tts(self):
        mgr, _ = await self._make_mgr()

        mock_fish = AsyncMock()
        await mgr.ensure_tts(mock_fish)

        assert mgr.current_slot == VRAMSlot.FISH_SPEECH
        mock_fish.start.assert_called_once()

    async def test_release_tts(self):
        mgr, _ = await self._make_mgr()
        mgr._current_slot = VRAMSlot.FISH_SPEECH

        mock_fish = AsyncMock()
        await mgr.release_tts(mock_fish)

        assert mgr.current_slot == VRAMSlot.EMPTY
        mock_fish.stop.assert_called_once()

    async def test_restore_idle(self):
        mgr, mock_client = await self._make_mgr()
        mgr._current_slot = VRAMSlot.EMPTY

        await mgr.restore_idle()

        assert mgr.current_slot == VRAMSlot.QWEN3
        preload_call = mock_client.post.call_args_list[-1]
        payload = preload_call[1]["json"]
        assert payload["keep_alive"] == "30m"

    async def test_full_swap_sequence(self):
        """Verify the full single-GPU swap: idle → stt → llm → tts → idle."""
        mgr, mock_client = await self._make_mgr()
        mock_stt = AsyncMock()
        mock_fish = AsyncMock()

        # Start from idle (Qwen3 resident)
        mgr._current_slot = VRAMSlot.QWEN3

        # STT phase
        await mgr.ensure_stt(mock_stt)
        assert mgr.current_slot == VRAMSlot.WHISPER

        await mgr.release_stt(mock_stt)
        assert mgr.current_slot == VRAMSlot.EMPTY

        # LLM phase
        await mgr.ensure_llm()
        assert mgr.current_slot == VRAMSlot.QWEN3

        await mgr.release_llm()
        assert mgr.current_slot == VRAMSlot.EMPTY

        # TTS phase
        await mgr.ensure_tts(mock_fish)
        assert mgr.current_slot == VRAMSlot.FISH_SPEECH

        await mgr.release_tts(mock_fish)
        assert mgr.current_slot == VRAMSlot.EMPTY

        # Restore idle
        await mgr.restore_idle()
        assert mgr.current_slot == VRAMSlot.QWEN3


class TestVRAMManagerDualGPU:
    async def test_all_methods_are_noop(self):
        mgr = VRAMManager(mode=GPUMode.DUAL)
        assert mgr.is_active is False

        mock_stt = AsyncMock()
        mock_fish = AsyncMock()

        # None of these should do anything
        await mgr.ensure_stt(mock_stt)
        mock_stt.load.assert_not_called()

        await mgr.release_stt(mock_stt)
        mock_stt.unload.assert_not_called()

        await mgr.ensure_llm()
        await mgr.release_llm()

        await mgr.ensure_tts(mock_fish)
        mock_fish.start.assert_not_called()

        await mgr.release_tts(mock_fish)
        mock_fish.stop.assert_not_called()

        await mgr.restore_idle()

        # Slot should never change
        assert mgr.current_slot == VRAMSlot.EMPTY

    async def test_no_http_calls_in_dual_mode(self):
        mgr = VRAMManager(mode=GPUMode.DUAL)
        mock_client = AsyncMock()
        mgr._client = mock_client

        await mgr.ensure_llm()
        await mgr.release_llm()
        await mgr.restore_idle()

        mock_client.post.assert_not_called()


class TestGPUModeEnum:
    def test_single(self):
        assert GPUMode("single") == GPUMode.SINGLE

    def test_dual(self):
        assert GPUMode("dual") == GPUMode.DUAL
