"""Tests for nested settings groups (backward compat + new form)."""

from __future__ import annotations

from config.settings import (
    DesktopSettings,
    GateSettings,
    OllamaSettings,
    ResearchSettings,
    Settings,
)


class TestNestedSettings:
    def test_nested_groups_exist(self):
        s = Settings()
        assert isinstance(s.ollama, OllamaSettings)
        assert isinstance(s.gate, GateSettings)
        assert isinstance(s.desktop, DesktopSettings)
        assert isinstance(s.research, ResearchSettings)

    def test_nested_defaults_match_flat(self):
        s = Settings()
        # Gate
        assert s.gate.enabled == s.gate_enabled
        assert s.gate.model == s.gate_model
        assert s.gate.timeout == s.gate_timeout
        # Desktop
        assert s.desktop.enabled == s.desktop_enabled
        assert s.desktop.model == s.desktop_model
        assert s.desktop.max_steps == s.desktop_max_steps
        # Ollama
        assert s.ollama.host == s.ollama_host
        assert s.ollama.keep_alive == s.ollama_keep_alive
        # Research
        assert s.research.enabled == s.research_enabled
        assert s.research.model == s.research_model
        assert s.research.top_k == s.research_top_k

    def test_env_nested_delimiter_configured(self):
        assert Settings.model_config.get("env_nested_delimiter") == "__"

    def test_all_nested_groups_have_defaults(self):
        """Every nested group should work with no args."""
        s = Settings()
        for name in [
            "ollama", "server", "auth", "screenshot",
            "click", "router", "pw", "pw_escalation", "gui_actor",
            "desktop", "gate", "research", "word", "voice",
        ]:
            group = getattr(s, name)
            assert group is not None, f"Nested group {name} is None"
