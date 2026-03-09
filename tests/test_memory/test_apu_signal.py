"""Tests for APUSignal — activity → APU model warmth signals."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from alchemy.memory.cache.apu_signal import APUSignal, ACTIVITY_MODEL_MAP


@pytest.fixture
def mock_orchestrator():
    orch = AsyncMock()
    orch.app_activate = AsyncMock()
    orch.app_deactivate = AsyncMock()
    return orch


class TestAPUSignal:
    @pytest.mark.asyncio
    async def test_emit_coding(self, mock_orchestrator):
        signal = APUSignal(mock_orchestrator)
        await signal._emit("coding")

        mock_orchestrator.app_activate.assert_awaited_once_with(
            "memory_stm",
            ACTIVITY_MODEL_MAP["coding"],
            module_tier="infra",
        )
        assert signal.current_activity == "coding"

    @pytest.mark.asyncio
    async def test_emit_idle_deactivates(self, mock_orchestrator):
        signal = APUSignal(mock_orchestrator)
        signal._current_activity = "coding"

        await signal._emit("idle")

        mock_orchestrator.app_deactivate.assert_awaited_once_with("memory_stm")
        # idle has no models, so app_activate should not be called
        mock_orchestrator.app_activate.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_emit_deactivates_previous(self, mock_orchestrator):
        signal = APUSignal(mock_orchestrator)
        signal._current_activity = "coding"

        await signal._emit("research")

        mock_orchestrator.app_deactivate.assert_awaited_once_with("memory_stm")
        mock_orchestrator.app_activate.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_emit_handles_orchestrator_failure(self, mock_orchestrator):
        mock_orchestrator.app_activate.side_effect = RuntimeError("GPU busy")
        signal = APUSignal(mock_orchestrator)

        # Should not raise
        await signal._emit("coding")
        assert signal.current_activity == "coding"
