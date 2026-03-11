"""AlchemyWord text generation via Ollama."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from config.settings import settings

if TYPE_CHECKING:
    from alchemy.adapters.ollama import OllamaClient

logger = logging.getLogger(__name__)

_MODE_PROMPTS = {
    "summarize": "Summarize the following text concisely. Output only the summary.",
    "rewrite": "Rewrite the following text to improve clarity and flow. Output only the rewritten text.",
    "expand": "Expand the following text with more detail and examples. Output only the expanded text.",
    "translate": "Translate the following text to {target_language}. Output only the translation.",
}

VALID_MODES = set(_MODE_PROMPTS.keys())


async def generate(
    client: OllamaClient,
    prompt: str,
    mode: str,
    model: str = "qwen3:14b",
    target_language: str = "English",
) -> str:
    """Generate text using Ollama.

    Parameters
    ----------
    client: OllamaClient instance
    prompt: The user's input text
    mode: One of "summarize", "rewrite", "expand", "translate"
    model: Ollama model name
    target_language: Target language for translate mode

    Returns
    -------
    Generated text string.
    """
    if mode not in VALID_MODES:
        raise ValueError(f"Invalid mode '{mode}'. Must be one of: {sorted(VALID_MODES)}")

    system_prompt = _MODE_PROMPTS[mode]
    if mode == "translate":
        system_prompt = system_prompt.format(target_language=target_language)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    result = await client.chat(
        model=model,
        messages=messages,
        options={
            "temperature": settings.word.temperature,
            "num_predict": settings.word.max_tokens,
        },
    )

    return result["message"]["content"]
