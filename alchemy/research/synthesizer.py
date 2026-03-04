"""LLM-powered query decomposition, relevance scoring, and synthesis."""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass

from alchemy.research.collector import PageContent

logger = logging.getLogger(__name__)


DECOMPOSE_SYSTEM = (
    "You are a research query decomposer. Given a user question, produce 8-12 "
    "diverse search queries that would find the information needed to answer "
    "comprehensively. Return ONLY a JSON array of strings, nothing else.\n\n"
    "Example:\n"
    'User: "What are the performance differences between React and Vue in 2026?"\n'
    '["React vs Vue performance benchmarks 2026", '
    '"React rendering speed comparison", '
    '"Vue 3 performance improvements", '
    '"React fiber architecture performance", '
    '"Vue virtual DOM efficiency", '
    '"React vs Vue bundle size comparison", '
    '"React vs Vue memory usage", '
    '"React concurrent features performance", '
    '"Vue composition API performance", '
    '"React vs Vue large application scaling"]'
)


SYNTHESIZE_SYSTEM = (
    "You are a research synthesizer. Given source materials and a user question, "
    "write a comprehensive, well-structured answer. Requirements:\n"
    "1. Answer the question directly and thoroughly\n"
    "2. Cite sources by number [1], [2] etc. when using information from them\n"
    "3. Be factual — do not invent information not in the sources\n"
    "4. If sources conflict, note the disagreement\n"
    "5. Structure with clear paragraphs, use markdown formatting\n"
    "6. At the end, list sources as:\n"
    "   Sources:\n"
    "   [1] Title — brief excerpt capturing the key point\n"
    "   [2] Title — brief excerpt\n"
    "   ...\n"
    "Do NOT include raw URLs. Only titles and brief excerpts."
)


@dataclass
class SynthesisResult:
    """Output of the synthesis step."""

    answer: str
    sources: list[dict]  # [{title: str, excerpt: str}]
    inference_ms: float = 0.0


class Synthesizer:
    """Handles LLM calls for query decomposition and final synthesis.

    Args:
        ollama_client: The shared OllamaClient instance from app.state.
        model: Ollama model name (e.g. "qwen3:14b").
        think: Whether to enable Qwen3 thinking mode.
        temperature: LLM temperature.
        max_tokens: Max tokens per LLM call.
    """

    def __init__(
        self,
        ollama_client,
        model: str = "qwen3:14b",
        think: bool = False,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ):
        self._ollama = ollama_client
        self._model = model
        self._think = think
        self._temperature = temperature
        self._max_tokens = max_tokens

    async def decompose_query(self, query: str, max_queries: int = 10) -> list[str]:
        """Decompose a user query into multiple search sub-queries.

        Args:
            query: The user's natural language question.
            max_queries: Maximum number of sub-queries to generate.

        Returns:
            List of search query strings.
        """
        t0 = time.monotonic()

        content = await self._infer(
            system=DECOMPOSE_SYSTEM,
            user=query,
        )

        queries = self._parse_json_array(content)
        elapsed = (time.monotonic() - t0) * 1000

        queries = queries[:max_queries]

        logger.info("Decomposed query into %d sub-queries (%.0fms)", len(queries), elapsed)
        return queries

    def score_relevance(
        self, query: str, pages: list[PageContent], top_k: int = 5
    ) -> list[PageContent]:
        """Quick keyword-overlap relevance scoring (no LLM call).

        Scores each page by counting how many query keywords appear in
        the page text. Returns top-K pages sorted by score descending.

        Args:
            query: Original user query.
            pages: List of PageContent with extracted text.
            top_k: Number of top pages to return.

        Returns:
            Top K PageContent objects sorted by relevance.
        """
        query_words = set(query.lower().split())
        stop = {
            "the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "to",
            "for", "of", "and", "or", "but", "with", "what", "how", "why", "when",
            "where", "which", "who",
        }
        query_words -= stop

        valid = [p for p in pages if p.fetch_ok and p.text]

        if not query_words:
            return valid[:top_k]

        scored: list[tuple[float, PageContent]] = []

        for page in valid:
            text_lower = page.text.lower()
            hits = sum(1 for w in query_words if w in text_lower)
            score = hits / len(query_words)
            length_bonus = min(page.word_count / 1000, 1.0) * 0.1
            scored.append((score + length_bonus, page))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = [page for _, page in scored[:top_k]]

        logger.info("Relevance scoring: %d pages -> top %d", len(scored), len(top))
        return top

    async def synthesize(self, query: str, pages: list[PageContent]) -> SynthesisResult:
        """Synthesize a structured answer from the top pages.

        Args:
            query: Original user query.
            pages: Ranked list of PageContent to use as sources.

        Returns:
            SynthesisResult with answer and source attributions.
        """
        source_blocks: list[str] = []
        for i, page in enumerate(pages, 1):
            text_preview = page.text[:2000]
            source_blocks.append(f"[Source {i}] {page.title}\n{text_preview}")

        context = "\n\n---\n\n".join(source_blocks)
        user_prompt = f"Question: {query}\n\nSources:\n{context}"

        t0 = time.monotonic()
        raw = await self._infer(system=SYNTHESIZE_SYSTEM, user=user_prompt)
        inference_ms = (time.monotonic() - t0) * 1000

        sources = self._extract_source_attributions(raw, pages)

        return SynthesisResult(answer=raw, sources=sources, inference_ms=inference_ms)

    async def _infer(self, system: str, user: str) -> str:
        """Call Qwen3 via Ollama with think mode support."""
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        result = await self._ollama.chat_think(
            model=self._model,
            messages=messages,
            think=self._think,
            options={"temperature": self._temperature, "num_predict": self._max_tokens},
        )

        if result["thinking"]:
            logger.debug("LLM thinking: %s", result["thinking"][:200])

        content = result["content"]
        if not content and result["thinking"]:
            logger.warning("LLM content empty, using thinking field as fallback")
            content = result["thinking"]

        return content

    @staticmethod
    def _parse_json_array(text: str) -> list[str]:
        """Extract a JSON array of strings from LLM output.

        Handles cases where the LLM wraps the array in markdown code fences.
        """
        # Strip markdown code fences if present
        cleaned = re.sub(r"```(?:json)?\s*", "", text)
        cleaned = cleaned.strip()

        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, list):
                return [str(q) for q in parsed if q]
        except json.JSONDecodeError:
            pass

        # Fallback: try to find array in the text
        match = re.search(r"\[.*\]", cleaned, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group())
                if isinstance(parsed, list):
                    return [str(q) for q in parsed if q]
            except json.JSONDecodeError:
                pass

        logger.warning("Failed to parse JSON array from LLM output")
        return []

    @staticmethod
    def _extract_source_attributions(
        answer: str, pages: list[PageContent]
    ) -> list[dict]:
        """Extract source attribution entries from the synthesized answer.

        Returns list of {title, excerpt} dicts. Tries to parse the Sources:
        section from the answer. Falls back to generating from page metadata.
        """
        sources: list[dict] = []

        sources_match = re.search(
            r"Sources:\s*\n(.*)", answer, re.DOTALL | re.IGNORECASE
        )
        if sources_match:
            source_text = sources_match.group(1)
            for line in source_text.strip().split("\n"):
                line = line.strip()
                if not line:
                    continue
                m = re.match(r"\[?\d+\]?\s*(.+?)(?:\s*[—\-]\s*(.+))?$", line)
                if m:
                    sources.append({
                        "title": m.group(1).strip(),
                        "excerpt": (m.group(2) or "").strip(),
                    })

        # Fallback: use page metadata if parsing produced nothing
        if not sources:
            for page in pages:
                if page.fetch_ok and page.text:
                    sources.append({
                        "title": page.title,
                        "excerpt": page.text[:150].strip() + "...",
                    })

        return sources
