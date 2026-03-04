"""Tests for Research API endpoints."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from alchemy.api.research_api import router, _tasks


class TestResearchAPI:
    def test_research_endpoint_exists(self):
        """POST /v1/research endpoint is registered."""
        paths = [r.path for r in router.routes]
        assert "/research" in paths

    def test_status_endpoint_exists(self):
        """GET /v1/research/{id}/status endpoint is registered."""
        paths = [r.path for r in router.routes]
        assert "/research/{task_id}/status" in paths


class TestGetResearchStatus:
    async def test_task_not_found(self):
        """GET status for non-existent task returns 404."""
        from httpx import ASGITransport, AsyncClient
        from alchemy.server import app

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/v1/research/nonexistent/status")
        assert resp.status_code == 404

    async def test_task_found_returns_status(self):
        from uuid import uuid4

        from alchemy.research.engine import PipelineStage, ResearchProgress
        from alchemy.research.synthesizer import SynthesisResult

        progress = ResearchProgress()
        progress.stage = PipelineStage.COMPLETED
        progress.queries_generated = 5
        progress.pages_fetched = 3
        progress.pages_used = 2
        progress.total_ms = 5000.0
        progress.result = SynthesisResult(
            answer="The answer",
            sources=[{"title": "Source 1", "excerpt": "excerpt 1"}],
            inference_ms=100.0,
        )

        test_task_id = str(uuid4())
        _tasks[test_task_id] = {
            "task_id": test_task_id,
            "status": "completed",
            "progress": progress,
        }

        try:
            from httpx import ASGITransport, AsyncClient
            from alchemy.server import app

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get(f"/v1/research/{test_task_id}/status")

            assert resp.status_code == 200
            data = resp.json()
            assert data["task_id"] == test_task_id
            assert data["result"]["answer"] == "The answer"
            assert len(data["result"]["sources"]) == 1
        finally:
            _tasks.pop(test_task_id, None)


class TestSubmitResearch:
    async def test_disabled_returns_503(self):
        from httpx import ASGITransport, AsyncClient
        from alchemy.server import app

        app.state.ollama_client = MagicMock()

        with patch("config.settings.settings") as mock_settings:
            mock_settings.research_enabled = False

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    "/v1/research",
                    json={"query": "test"},
                )
            assert resp.status_code == 503

    async def test_direct_mode_no_urls_returns_400(self):
        from httpx import ASGITransport, AsyncClient
        from alchemy.server import app

        # Ensure ollama_client exists
        app.state.ollama_client = MagicMock()

        with patch("config.settings.settings") as mock_settings:
            mock_settings.research_enabled = True
            mock_settings.research_model = "qwen3:14b"
            mock_settings.research_think = False
            mock_settings.research_temperature = 0.3
            mock_settings.research_max_tokens = 2048
            mock_settings.research_max_queries = 10
            mock_settings.research_max_pages = 8
            mock_settings.research_fetch_timeout = 15.0
            mock_settings.research_top_k = 5

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    "/v1/research",
                    json={"query": "test", "mode": "direct", "urls": []},
                )
            assert resp.status_code == 400

    async def test_no_ollama_returns_503(self):
        from httpx import ASGITransport, AsyncClient
        from alchemy.server import app

        # Remove ollama client
        app.state.ollama_client = None

        with patch("config.settings.settings") as mock_settings:
            mock_settings.research_enabled = True

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    "/v1/research",
                    json={"query": "test"},
                )
            assert resp.status_code == 503
