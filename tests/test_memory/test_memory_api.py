"""Tests for AlchemyMemory API endpoints."""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from alchemy.memory.cache.store import STMStore
from alchemy.memory.timeline.store import TimelineStore


@pytest.fixture
def memory_system(tmp_path):
    """Create a minimal MemorySystem-like mock with real stores."""
    timeline = TimelineStore(tmp_path / "timeline.db")
    timeline.init()
    stm = STMStore(tmp_path / "stm.db")
    stm.init()

    mem = MagicMock()
    mem.timeline = timeline
    mem.stm = stm
    mem.capture = MagicMock()
    mem.capture.is_running.return_value = True
    mem.capture.ingest_event = AsyncMock(return_value=1)
    mem.vectors = MagicMock()
    mem.vectors.count.return_value = 0
    mem.classifier = MagicMock()
    mem.classifier.current_activity = "coding"
    mem.classifier.last_classified_at = time.time()
    mem.context_packer = MagicMock()
    mem.context_packer.build.return_value = {
        "activity": "coding",
        "recent": ["Editing code"],
        "apps": ["VS Code"],
        "preferences": {"language": "Python"},
        "generated_at": time.time(),
        "text_summary": "Current activity: coding",
    }
    mem.searcher = AsyncMock()
    mem.searcher.search = AsyncMock(return_value=[])
    mem.settings = MagicMock()
    mem.settings.storage_path = str(tmp_path)
    mem.settings.max_stm_results = 5
    return mem


@pytest.fixture
def client(memory_system):
    from fastapi import FastAPI
    from alchemy.memory.api.memory_api import router

    app = FastAPI()
    app.state.memory_system = memory_system
    app.include_router(router)
    return TestClient(app)


class TestMemoryAPI:
    def test_health(self, client):
        resp = client.get("/v1/memory/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "running"
        assert data["activity"] == "coding"

    def test_timeline_recent_empty(self, client):
        resp = client.get("/v1/memory/timeline/recent")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_timeline_recent_with_events(self, client, memory_system):
        memory_system.timeline.insert(
            event_type="screenshot",
            summary="Editing Python",
            app_name="VS Code",
        )
        resp = client.get("/v1/memory/timeline/recent?limit=5")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["summary"] == "Editing Python"

    def test_stm_context(self, client):
        resp = client.get("/v1/memory/stm/context")
        assert resp.status_code == 200
        data = resp.json()
        assert data["activity"] == "coding"
        assert "Editing code" in data["recent"]

    def test_stm_activity(self, client):
        resp = client.get("/v1/memory/stm/activity")
        assert resp.status_code == 200
        data = resp.json()
        assert data["activity"] == "coding"

    def test_timeline_ingest(self, client):
        resp = client.post("/v1/memory/timeline/ingest", json={
            "event_type": "voice",
            "summary": "User said hello",
            "source": "voice",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ingested"

    def test_stm_flush(self, client):
        resp = client.delete("/v1/memory/stm/flush")
        assert resp.status_code == 200
        assert "deleted" in resp.json()

    def test_screenshot_not_found(self, client):
        resp = client.get("/v1/memory/screenshot/999")
        assert resp.status_code == 404

    def test_search_returns_task_id(self, client):
        resp = client.post("/v1/memory/search", json={"query": "python code"})
        assert resp.status_code == 200
        data = resp.json()
        assert "task_id" in data
        assert data["status"] == "started"
