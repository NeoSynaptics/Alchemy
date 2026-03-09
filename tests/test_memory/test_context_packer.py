"""Tests for ContextPacker — builds context_pack for LLM injection."""

import time

import pytest

from alchemy.memory.cache.context import ContextPacker
from alchemy.memory.cache.store import STMStore


@pytest.fixture
def stm(tmp_path):
    s = STMStore(tmp_path / "stm.db")
    s.init()
    return s


@pytest.fixture
def packer(stm):
    return ContextPacker(stm)


class TestContextPacker:
    def test_build_empty(self, packer):
        pack = packer.build()
        assert pack["activity"] == "idle"
        assert pack["recent"] == []
        assert pack["apps"] == []
        assert isinstance(pack["preferences"], dict)
        assert isinstance(pack["generated_at"], float)
        assert "Current activity: idle" in pack["text_summary"]

    def test_build_with_events(self, stm, packer):
        now = time.time()
        stm.insert(event_type="x", summary="Editing main.py", app_name="VS Code", ts=now)
        stm.insert(event_type="x", summary="Running pytest", app_name="Terminal", ts=now)

        packer.set_activity("coding")
        pack = packer.build()

        assert pack["activity"] == "coding"
        assert len(pack["recent"]) == 2
        assert "VS Code" in pack["apps"]
        assert "coding" in pack["text_summary"]

    def test_build_with_preferences(self, stm, packer):
        stm.set_preference("language", "Python")
        pack = packer.build()
        assert pack["preferences"]["language"] == "Python"
        assert "Python" in pack["text_summary"]
