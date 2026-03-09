"""ContextPacker — assembles the context_pack dict injected into LLM calls.

The context_pack is a lightweight dict that any LLM call can prepend
as a system message section. It represents "what the user has been
doing recently" — sourced entirely from STM (short-term memory).

Usage:
    pack = packer.build()
    # Then in a system message:
    system = f"User context:\\n{pack['text_summary']}\\n\\nNow respond..."
"""

from __future__ import annotations

import time
from typing import Any

from alchemy.memory.cache.store import STMStore


class ContextPacker:
    """Builds a context_pack from current STM state."""

    def __init__(self, stm: STMStore) -> None:
        self._stm = stm
        self._current_activity: str = "idle"

    def set_activity(self, activity: str) -> None:
        self._current_activity = activity

    def build(self) -> dict[str, Any]:
        """Build context pack from current STM state.

        Returns a dict with:
          activity       — detected activity string
          recent         — list of summary strings (last 1min)
          apps           — active app names (last 1hr)
          preferences    — user preference dict
          generated_at   — unix timestamp
          text_summary   — pre-formatted text for LLM injection
        """
        recent_events = self._stm.recent(window_minutes=1, limit=10)
        recent_summaries = [e.summary for e in recent_events if e.summary]
        active_apps = self._stm.active_apps(last_hours=1)
        preferences = self._stm.get_preferences()

        text_summary = self._format(
            self._current_activity, recent_summaries, active_apps, preferences
        )

        return {
            "activity": self._current_activity,
            "recent": recent_summaries,
            "apps": active_apps,
            "preferences": preferences,
            "generated_at": time.time(),
            "text_summary": text_summary,
        }

    @staticmethod
    def _format(
        activity: str,
        recent: list[str],
        apps: list[str],
        prefs: dict[str, str],
    ) -> str:
        lines: list[str] = [f"Current activity: {activity}"]
        if apps:
            lines.append(f"Active apps: {', '.join(apps[:5])}")
        if recent:
            lines.append("Recent context:")
            for s in recent[:5]:
                lines.append(f"  - {s}")
        if prefs:
            lines.append("User preferences:")
            for k, v in list(prefs.items())[:5]:
                lines.append(f"  {k}: {v}")
        return "\n".join(lines)
