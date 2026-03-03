"""Context builder — assembles enriched prompt context from all sources.

Takes an EnvironmentSnapshot, task goal, and settings to produce
a context block that gets injected into the vision agent's system prompt.
The model gets orientation — not micromanagement.
"""

from __future__ import annotations

import logging

from alchemy.router.categories import TaskCategory, classify_task, get_hint
from alchemy.router.completion import get_completion
from alchemy.router.environment import EnvironmentSnapshot
from alchemy.router.recovery import get_recovery

logger = logging.getLogger(__name__)

# App keywords for category-specific filtering
_MEDIA_KEYWORDS = ["spotify", "vlc", "rhythmbox", "audacious", "totem", "music", "media"]
_COMM_KEYWORDS = ["slack", "discord", "teams", "thunderbird", "outlook", "mail", "telegram"]
_DEV_KEYWORDS = ["code", "vscode", "visual studio", "vim", "terminal", "powershell"]
_FILE_KEYWORDS = ["nautilus", "thunar", "explorer", "pcmanfm", "file"]


class ContextBuilder:
    """Assembles enriched context for the vision agent's system prompt."""

    def __init__(
        self,
        env: EnvironmentSnapshot,
        *,
        category_hints: bool = True,
        recovery_nudges: bool = True,
        completion_criteria: bool = True,
    ):
        self._env = env
        self._category_hints = category_hints
        self._recovery_nudges = recovery_nudges
        self._completion_criteria = completion_criteria

    def build(self, goal: str) -> str:
        """Build the full context block for injection into the system prompt.

        Returns a multi-section string ready for template insertion.
        """
        category = classify_task(goal)

        sections: list[str] = []

        # Always include environment
        sections.append(self._environment_block())

        # Category hint (with app-specific placeholders filled)
        if self._category_hints:
            hint = self._fill_hint(category)
            if hint:
                sections.append(f"## Task Context\n{hint}")

        # Recovery nudge
        if self._recovery_nudges:
            recovery = get_recovery(category)
            sections.append(f"## If Stuck\n{recovery}")

        # Completion criteria
        if self._completion_criteria:
            completion = get_completion(category)
            sections.append(f"## Completion\n{completion}")

        return "\n\n".join(sections)

    @property
    def category_for(self) -> type:
        """Expose classify_task for external use."""
        return classify_task

    def _environment_block(self) -> str:
        """Format the environment snapshot as a readable block."""
        env = self._env
        lines = ["## Environment"]
        lines.append(f"- Shadow Desktop: {env.os_type}, {env.desktop}, {env.resolution}")

        if env.shadow_apps:
            lines.append(f"- Shadow Apps: {', '.join(env.shadow_apps)}")

        lines.append(f"- Windows Host: {env.windows_version}")

        if env.windows_apps:
            lines.append(f"- Windows Apps: {', '.join(env.windows_apps)}")

        audio_status = "available" if env.has_audio else "not configured"
        lines.append(f"- Audio: PulseAudio {audio_status} on shadow desktop")
        lines.append(f"- App launcher: {env.search_method}")

        return "\n".join(lines)

    def _fill_hint(self, category: TaskCategory) -> str:
        """Fill category hint template with environment-specific values."""
        env = self._env
        raw_hint = get_hint(category)

        # Build substitution values from environment
        media_apps = env.apps_for_category(_MEDIA_KEYWORDS)
        comm_apps = env.apps_for_category(_COMM_KEYWORDS)
        dev_apps = env.apps_for_category(_DEV_KEYWORDS)
        file_apps = env.apps_for_category(_FILE_KEYWORDS)

        # Tag apps with their platform
        def _tag(apps: list[str]) -> str:
            if not apps:
                return "none detected"
            tagged = []
            for app in apps:
                if app in env.shadow_apps:
                    tagged.append(f"{app} (shadow)")
                elif app in env.windows_apps:
                    tagged.append(f"{app} (Windows)")
                else:
                    tagged.append(app)
            return ", ".join(tagged)

        subs = {
            "media_apps": _tag(media_apps),
            "default_browser": env.default_browser,
            "file_manager": _tag(file_apps) if file_apps else "nautilus or terminal",
            "comm_apps": _tag(comm_apps),
            "dev_apps": _tag(dev_apps),
            "desktop": env.desktop,
        }

        try:
            return raw_hint.format(**subs)
        except KeyError:
            # If a placeholder is missing, return with unfilled placeholders stripped
            return raw_hint
