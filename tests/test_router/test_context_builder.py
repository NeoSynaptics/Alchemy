"""Tests for the context builder — full context assembly."""

import pytest

from alchemy.router.categories import TaskCategory
from alchemy.router.context_builder import ContextBuilder
from alchemy.router.environment import EnvironmentSnapshot


@pytest.fixture
def env():
    return EnvironmentSnapshot(
        os_type="Ubuntu 22.04 (WSL2)",
        desktop="Fluxbox",
        resolution="1920x1080",
        shadow_apps=["firefox", "vlc", "xterm", "nautilus"],
        default_browser="firefox",
        has_audio=True,
        search_method="fluxbox menu (right-click desktop)",
        windows_apps=["Spotify", "Google Chrome", "Discord", "Visual Studio Code"],
        windows_version="Windows 11 Pro",
    )


class TestContextBuilder:
    def test_build_includes_environment(self, env):
        builder = ContextBuilder(env)
        ctx = builder.build("open spotify")

        assert "## Environment" in ctx
        assert "Ubuntu 22.04" in ctx
        assert "Fluxbox" in ctx
        assert "1920x1080" in ctx

    def test_build_includes_shadow_apps(self, env):
        builder = ContextBuilder(env)
        ctx = builder.build("open firefox")

        assert "firefox" in ctx
        assert "vlc" in ctx

    def test_build_includes_windows_apps(self, env):
        builder = ContextBuilder(env)
        ctx = builder.build("open spotify")

        assert "Spotify" in ctx
        assert "Windows 11 Pro" in ctx

    def test_build_includes_audio_status(self, env):
        builder = ContextBuilder(env)
        ctx = builder.build("play music")

        assert "PulseAudio available" in ctx

    def test_build_no_audio(self, env):
        env.has_audio = False
        builder = ContextBuilder(env)
        ctx = builder.build("play music")

        assert "not configured" in ctx

    def test_build_includes_category_hint(self, env):
        builder = ContextBuilder(env)
        ctx = builder.build("open spotify and play music")

        assert "## Task Context" in ctx
        assert "media" in ctx.lower()

    def test_build_includes_recovery(self, env):
        builder = ContextBuilder(env)
        ctx = builder.build("open firefox")

        assert "## If Stuck" in ctx

    def test_build_includes_completion(self, env):
        builder = ContextBuilder(env)
        ctx = builder.build("play a song")

        assert "## Completion" in ctx
        assert "playing" in ctx.lower()

    def test_build_without_category_hints(self, env):
        builder = ContextBuilder(env, category_hints=False)
        ctx = builder.build("open spotify")

        assert "## Task Context" not in ctx
        # But environment and recovery should still be there
        assert "## Environment" in ctx
        assert "## If Stuck" in ctx

    def test_build_without_recovery(self, env):
        builder = ContextBuilder(env, recovery_nudges=False)
        ctx = builder.build("open spotify")

        assert "## If Stuck" not in ctx
        assert "## Environment" in ctx

    def test_build_without_completion(self, env):
        builder = ContextBuilder(env, completion_criteria=False)
        ctx = builder.build("open spotify")

        assert "## Completion" not in ctx
        assert "## Environment" in ctx

    def test_build_minimal(self, env):
        builder = ContextBuilder(
            env,
            category_hints=False,
            recovery_nudges=False,
            completion_criteria=False,
        )
        ctx = builder.build("anything")

        assert "## Environment" in ctx
        assert "## Task Context" not in ctx
        assert "## If Stuck" not in ctx
        assert "## Completion" not in ctx

    def test_build_general_category(self, env):
        builder = ContextBuilder(env)
        ctx = builder.build("do something random")

        # Should still produce valid context even for GENERAL
        assert "## Environment" in ctx

    def test_app_tagging_shows_platform(self, env):
        builder = ContextBuilder(env)
        ctx = builder.build("open spotify and play music")

        # Spotify is on Windows, VLC is on shadow
        assert "(shadow)" in ctx or "(Windows)" in ctx

    def test_empty_environment(self):
        env = EnvironmentSnapshot()
        builder = ContextBuilder(env)
        ctx = builder.build("play music")

        # Should not crash, should produce valid output
        assert "## Environment" in ctx
        assert "## Task Context" in ctx
