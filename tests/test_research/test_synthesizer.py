"""Tests for Synthesizer — LLM decomposition + scoring + synthesis."""

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from alchemy.research.collector import PageContent
from alchemy.research.synthesizer import Synthesizer, SynthesisResult


def _make_mock_ollama(response_content: str, thinking: str = "") -> MagicMock:
    """Create a mock OllamaClient that returns a controlled response via chat_think()."""
    mock_ollama = MagicMock()

    async def mock_chat_think(model, messages, think=True, options=None, seed=None):
        return {"content": response_content, "thinking": thinking, "total_duration": 0}

    mock_ollama.chat_think = mock_chat_think
    return mock_ollama


def _make_page(title: str, text: str, word_count: int = 100, fetch_ok: bool = True) -> PageContent:
    return PageContent(
        url=f"https://example.com/{title.lower().replace(' ', '-')}",
        title=title,
        text=text,
        fetch_ok=fetch_ok,
        word_count=word_count,
    )


class TestDecomposeQuery:
    async def test_decompose_returns_queries(self):
        mock_ollama = _make_mock_ollama('["query one", "query two", "query three"]')
        synth = Synthesizer(ollama_client=mock_ollama)

        result = await synth.decompose_query("test question")

        assert result == ["query one", "query two", "query three"]

    async def test_decompose_handles_markdown_fences(self):
        mock_ollama = _make_mock_ollama('```json\n["q1", "q2"]\n```')
        synth = Synthesizer(ollama_client=mock_ollama)

        result = await synth.decompose_query("test")

        assert result == ["q1", "q2"]

    async def test_decompose_caps_at_max(self):
        import json
        queries = [f"q{i}" for i in range(20)]
        mock_ollama = _make_mock_ollama(json.dumps(queries))
        synth = Synthesizer(ollama_client=mock_ollama)

        result = await synth.decompose_query("test", max_queries=5)

        assert len(result) == 5

    async def test_decompose_fallback_on_bad_json(self):
        mock_ollama = _make_mock_ollama("this is not json at all")
        synth = Synthesizer(ollama_client=mock_ollama)

        result = await synth.decompose_query("test")

        assert result == []

    async def test_decompose_extracts_array_from_prose(self):
        mock_ollama = _make_mock_ollama(
            'Here are the queries:\n["embedded q1", "embedded q2"]\nHope that helps!'
        )
        synth = Synthesizer(ollama_client=mock_ollama)

        result = await synth.decompose_query("test")

        assert result == ["embedded q1", "embedded q2"]


class TestScoreRelevance:
    def test_basic_ranking(self):
        synth = Synthesizer(ollama_client=MagicMock())

        pages = [
            _make_page("Low", "unrelated content about cooking recipes"),
            _make_page("High", "python programming language performance benchmarks speed"),
            _make_page("Mid", "python tutorial for beginners"),
        ]

        result = synth.score_relevance("python performance benchmarks", pages, top_k=2)

        assert len(result) == 2
        assert result[0].title == "High"

    def test_filters_failed_pages(self):
        synth = Synthesizer(ollama_client=MagicMock())

        pages = [
            _make_page("Good", "relevant content python", fetch_ok=True),
            _make_page("Bad", "", fetch_ok=False),
        ]

        result = synth.score_relevance("python", pages, top_k=5)

        assert len(result) == 1
        assert result[0].title == "Good"

    def test_respects_top_k(self):
        synth = Synthesizer(ollama_client=MagicMock())

        pages = [_make_page(f"Page {i}", "relevant content python") for i in range(10)]

        result = synth.score_relevance("python", pages, top_k=3)

        assert len(result) == 3

    def test_empty_query_words_returns_first_k(self):
        synth = Synthesizer(ollama_client=MagicMock())

        pages = [_make_page(f"Page {i}", "some content") for i in range(5)]

        # All stop words
        result = synth.score_relevance("the is a an", pages, top_k=2)

        assert len(result) == 2

    def test_length_bonus(self):
        synth = Synthesizer(ollama_client=MagicMock())

        # Same keyword density but different lengths
        pages = [
            _make_page("Short", "python code", word_count=50),
            _make_page("Long", "python code", word_count=2000),
        ]

        result = synth.score_relevance("python code", pages, top_k=2)

        # Long page should rank higher due to length bonus
        assert result[0].title == "Long"


class TestSynthesize:
    async def test_returns_answer(self):
        answer_text = "Here is the answer.\n\nSources:\n[1] Page One — key finding"
        mock_ollama = _make_mock_ollama(answer_text)
        synth = Synthesizer(ollama_client=mock_ollama)

        pages = [_make_page("Page One", "source content here")]
        result = await synth.synthesize("test question", pages)

        assert isinstance(result, SynthesisResult)
        assert "Here is the answer" in result.answer
        assert result.inference_ms > 0

    async def test_extracts_sources(self):
        answer_text = (
            "The answer.\n\nSources:\n"
            "[1] First Source — important finding\n"
            "[2] Second Source — another insight"
        )
        mock_ollama = _make_mock_ollama(answer_text)
        synth = Synthesizer(ollama_client=mock_ollama)

        pages = [
            _make_page("First Source", "content 1"),
            _make_page("Second Source", "content 2"),
        ]
        result = await synth.synthesize("test", pages)

        assert len(result.sources) == 2
        assert result.sources[0]["title"] == "First Source"
        assert result.sources[0]["excerpt"] == "important finding"

    async def test_fallback_sources(self):
        # No Sources: section in answer
        mock_ollama = _make_mock_ollama("Just an answer with no sources section.")
        synth = Synthesizer(ollama_client=mock_ollama)

        pages = [_make_page("Fallback Page", "some content here for excerpt")]
        result = await synth.synthesize("test", pages)

        assert len(result.sources) == 1
        assert result.sources[0]["title"] == "Fallback Page"
        assert "some content" in result.sources[0]["excerpt"]

    async def test_truncates_source_text(self):
        captured_messages = []

        async def mock_chat_think(model, messages, think=True, options=None, seed=None):
            captured_messages.append(messages)
            return {"content": "Answer.\n\nSources:\n[1] P — e", "thinking": "", "total_duration": 0}

        mock_ollama = MagicMock()
        mock_ollama.chat_think = mock_chat_think
        synth = Synthesizer(ollama_client=mock_ollama)

        long_text = "x" * 5000
        pages = [_make_page("Long", long_text)]

        await synth.synthesize("test", pages)

        # The synthesize call is the second chat_think call (first is decompose fallback or direct)
        user_msg = captured_messages[-1][1]["content"]
        # Source text should be truncated to ~2000 chars
        assert len(user_msg) < 3000  # 2000 source + query + formatting


class TestInfer:
    async def test_think_false(self):
        captured = {}

        async def mock_chat_think(model, messages, think=True, options=None, seed=None):
            captured["think"] = think
            captured["model"] = model
            return {"content": "response", "thinking": "", "total_duration": 0}

        mock_ollama = MagicMock()
        mock_ollama.chat_think = mock_chat_think
        synth = Synthesizer(ollama_client=mock_ollama, think=False)

        result = await synth._infer(system="sys", user="usr")

        assert captured["think"] is False
        assert result == "response"

    async def test_empty_content_uses_thinking(self):
        mock_ollama = _make_mock_ollama("", thinking="fallback from thinking")
        synth = Synthesizer(ollama_client=mock_ollama)

        result = await synth._infer(system="sys", user="usr")

        assert result == "fallback from thinking"

    async def test_uses_correct_model(self):
        captured = {}

        async def mock_chat_think(model, messages, think=True, options=None, seed=None):
            captured["model"] = model
            return {"content": "response", "thinking": "", "total_duration": 0}

        mock_ollama = MagicMock()
        mock_ollama.chat_think = mock_chat_think
        synth = Synthesizer(ollama_client=mock_ollama, model="custom-model:7b")

        await synth._infer(system="sys", user="usr")

        assert captured["model"] == "custom-model:7b"


class TestParseJsonArray:
    def test_clean_json(self):
        result = Synthesizer._parse_json_array('["a", "b", "c"]')
        assert result == ["a", "b", "c"]

    def test_markdown_fenced(self):
        result = Synthesizer._parse_json_array('```json\n["a", "b"]\n```')
        assert result == ["a", "b"]

    def test_embedded_array(self):
        result = Synthesizer._parse_json_array('Here you go:\n["a", "b"]\nDone!')
        assert result == ["a", "b"]

    def test_invalid_json(self):
        result = Synthesizer._parse_json_array("not json at all")
        assert result == []

    def test_filters_empty_entries(self):
        result = Synthesizer._parse_json_array('["a", "", "b", null]')
        assert result == ["a", "b"]


class TestExtractSourceAttributions:
    def test_parses_sources_section(self):
        answer = "Answer.\n\nSources:\n[1] Title A — excerpt A\n[2] Title B — excerpt B"
        pages = [_make_page("Title A", "text"), _make_page("Title B", "text")]

        result = Synthesizer._extract_source_attributions(answer, pages)

        assert len(result) == 2
        assert result[0]["title"] == "Title A"
        assert result[0]["excerpt"] == "excerpt A"

    def test_fallback_to_page_metadata(self):
        answer = "Just an answer, no sources."
        pages = [_make_page("Page Title", "the page content here")]

        result = Synthesizer._extract_source_attributions(answer, pages)

        assert len(result) == 1
        assert result[0]["title"] == "Page Title"
        assert "the page content" in result[0]["excerpt"]

    def test_handles_em_dash_and_regular_dash(self):
        answer = "Answer.\n\nSources:\n[1] Title — em dash\n[2] Other - regular dash"
        pages = []

        result = Synthesizer._extract_source_attributions(answer, pages)

        assert len(result) == 2
        assert result[0]["excerpt"] == "em dash"
        assert result[1]["excerpt"] == "regular dash"
