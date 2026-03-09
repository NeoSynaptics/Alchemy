"""ActivityClassifier — uses qwen3:3b to decide what the user is currently doing.

Runs every 60 seconds inside the STM purge loop. Classifies the last
few STM event summaries into one of a fixed set of activity categories,
then signals the APU which models to keep warm.

Categories: coding | visual_task | research | writing | voice_chat | idle
"""

from __future__ import annotations

import asyncio
import logging
import time

from alchemy.adapters.ollama import OllamaClient
from alchemy.memory.cache.store import STMStore

logger = logging.getLogger(__name__)

_CATEGORIES = ["coding", "visual_task", "research", "writing", "voice_chat", "idle"]

_SYSTEM_PROMPT = """\
You classify desktop activity into exactly one category.
Categories: coding, visual_task, research, writing, voice_chat, idle
Output ONLY the category word, nothing else."""

_USER_TEMPLATE = """\
Recent activity summaries:
{summaries}

Active apps: {apps}

Category:"""


class ActivityClassifier:
    """Classifies current user activity using qwen3:3b every 60s."""

    def __init__(
        self,
        ollama: OllamaClient,
        stm: STMStore,
        model: str = "qwen3:3b",
        interval: int = 60,
    ) -> None:
        self._ollama = ollama
        self._stm = stm
        self._model = model
        self._interval = interval
        self.current_activity: str = "idle"
        self.last_classified_at: float = 0.0
        self._task: asyncio.Task | None = None
        self._on_change_callbacks: list = []

    def on_activity_change(self, callback) -> None:
        """Register a callback(activity: str) called when activity changes."""
        self._on_change_callbacks.append(callback)

    def start(self) -> None:
        self._task = asyncio.create_task(
            self._classify_loop(), name="memory:classifier"
        )

    def stop(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()

    async def _classify_loop(self) -> None:
        while True:
            await asyncio.sleep(self._interval)
            try:
                await self._classify()
            except Exception:
                logger.warning("Activity classification failed", exc_info=True)

    async def _classify(self) -> None:
        recent = self._stm.recent(window_minutes=5, limit=6)
        if not recent:
            self.current_activity = "idle"
            return

        summaries = "\n".join(f"- {e.summary}" for e in recent if e.summary)
        apps = ", ".join(self._stm.active_apps(last_hours=1)[:5])

        prompt = _USER_TEMPLATE.format(summaries=summaries or "(none)", apps=apps or "(none)")

        result = await self._ollama.chat(
            model=self._model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            options={"temperature": 0.0, "num_predict": 10},
        )
        raw: str = result.get("message", {}).get("content", "").strip().lower()

        if not raw:
            logger.warning("Classifier returned empty response — keeping %s", self.current_activity)
            return

        # Exact match first, then substring (longest first to avoid collision)
        matched: str | None = None
        if raw in _CATEGORIES:
            matched = raw
        else:
            for cat in sorted(_CATEGORIES, key=len, reverse=True):
                if cat in raw:
                    matched = cat
                    break

        if matched:
            if matched != self.current_activity:
                logger.info("Activity: %s → %s", self.current_activity, matched)
                self.current_activity = matched
                for cb in self._on_change_callbacks:
                    try:
                        cb(matched)
                    except Exception:
                        logger.debug("Activity change callback failed", exc_info=True)
        else:
            logger.warning("Classifier returned unrecognized output: %r — falling back to idle", raw)
            self.current_activity = "idle"

        self.last_classified_at = time.time()
