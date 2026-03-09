"""ScreenshotSummarizer — turns a screenshot into one descriptive sentence.

Uses qwen2.5vl:7b (the proven AlchemyFlow vision model) via OllamaClient.chat()
with the image attached. Returns a short caption describing what the user
is doing at that moment.
"""

from __future__ import annotations

import logging

from alchemy.adapters.ollama import OllamaClient

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a desktop activity summarizer. "
    "Given a screenshot, output ONE sentence describing what the user is doing right now. "
    "Be specific about visible apps, file names, or content on screen. "
    "No preamble, no punctuation at the end."
)


class ScreenshotSummarizer:
    """Captions a screenshot with a single sentence using qwen2.5vl:7b."""

    def __init__(self, ollama: OllamaClient, model: str = "qwen2.5vl:7b") -> None:
        self._ollama = ollama
        self._model = model

    async def summarize(self, image_bytes: bytes) -> str:
        """Return a one-sentence caption for the screenshot.

        Falls back to empty string on failure so capture loop never crashes.
        """
        try:
            result = await self._ollama.chat(
                model=self._model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": "What is the user doing?"},
                ],
                images=[image_bytes],
                options={"num_ctx": 8192, "temperature": 0.0, "num_predict": 128},
            )
            content: str = result.get("message", {}).get("content", "").strip()
            return content
        except Exception:
            logger.warning("Screenshot summarization failed", exc_info=True)
            return ""
