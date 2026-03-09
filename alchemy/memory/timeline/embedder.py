"""EmbeddingClient — wraps nomic-embed-text via OllamaClient.embed().

Thin wrapper that adds the model name and truncates text to stay
within token limits.
"""

from __future__ import annotations

import logging

from alchemy.adapters.ollama import OllamaClient

logger = logging.getLogger(__name__)

# nomic-embed-text context window is 8192 tokens (~6000 words).
# Truncate at characters as a cheap proxy.
_MAX_CHARS = 4000


class EmbeddingClient:
    """Generates embeddings using nomic-embed-text via Ollama."""

    def __init__(self, ollama: OllamaClient, model: str = "nomic-embed-text") -> None:
        self._ollama = ollama
        self._model = model

    async def embed(self, text: str) -> list[float]:
        """Embed text. Returns 768-dim float list.

        Truncates to _MAX_CHARS before sending. Raises on model failure.
        """
        truncated = text[:_MAX_CHARS]
        try:
            return await self._ollama.embed(self._model, truncated)
        except Exception:
            logger.exception("Embedding failed for text[:50]=%r", text[:50])
            raise
