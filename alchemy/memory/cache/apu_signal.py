"""APUSignal — translates detected activity into APU model warmth signals.

The STM (short-term memory) dictates what models the APU should keep warm.
This class is the bridge: it watches the ActivityClassifier and calls
orchestrator.app_activate() / app_deactivate() when activity changes.

Only STM signals the APU. LTM never does.
"""

from __future__ import annotations

import asyncio
import logging
import time

logger = logging.getLogger(__name__)

# Activity → list of models to keep warm in VRAM
ACTIVITY_MODEL_MAP: dict[str, list[str]] = {
    "coding":       ["qwen3:14b", "qwen2.5-coder:14b"],
    "visual_task":  ["qwen2.5vl:7b"],
    "research":     ["qwen3:14b", "nomic-embed-text"],
    "writing":      ["qwen3:14b"],
    "voice_chat":   [],  # voice stack manages its own VRAM
    "idle":         [],  # demote non-resident models
}

_APP_NAME = "memory_stm"
_SIGNAL_REFRESH_SECONDS = 600  # re-signal even if activity unchanged (keep_alive)


class APUSignal:
    """Sends model warmth signals to StackOrchestrator based on user activity."""

    def __init__(self, orchestrator) -> None:
        self._orchestrator = orchestrator
        self._current_activity: str = ""
        self._last_signal_at: float = 0.0
        self._task: asyncio.Task | None = None

    def start(self, classifier) -> None:
        """Start watching the classifier and emitting APU signals."""
        self._task = asyncio.create_task(
            self._signal_loop(classifier), name="memory:apu_signal"
        )

    def stop(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()

    async def _signal_loop(self, classifier) -> None:
        while True:
            await asyncio.sleep(10)  # check every 10s
            activity = classifier.current_activity
            now = time.time()
            needs_refresh = (now - self._last_signal_at) >= _SIGNAL_REFRESH_SECONDS

            if activity == self._current_activity and not needs_refresh:
                continue

            await self._emit(activity)

    async def _emit(self, activity: str) -> None:
        models = ACTIVITY_MODEL_MAP.get(activity, [])
        try:
            # Deactivate previous slot
            if self._current_activity:
                await self._orchestrator.app_deactivate(_APP_NAME)

            # Activate new slot
            if models:
                await self._orchestrator.app_activate(
                    _APP_NAME, models, module_tier="infra"
                )

            logger.info("APU signal: activity=%s warm=%s", activity, models)
        except Exception:
            logger.debug("APU signal failed (non-critical)", exc_info=True)

        self._current_activity = activity
        self._last_signal_at = time.time()

    @property
    def current_activity(self) -> str:
        return self._current_activity
