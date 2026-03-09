"""Tests for TimelineStore — SQLite long-term memory."""

import time

import pytest

from alchemy.memory.timeline.store import TimelineStore


@pytest.fixture
def store(tmp_path):
    s = TimelineStore(tmp_path / "timeline.db")
    s.init()
    return s


class TestTimelineStore:
    def test_init_creates_db(self, tmp_path):
        db_path = tmp_path / "sub" / "timeline.db"
        store = TimelineStore(db_path)
        store.init()
        assert db_path.exists()

    def test_insert_and_get(self, store):
        eid = store.insert(
            event_type="screenshot",
            summary="User is editing code in VS Code",
            source="desktop",
            app_name="VS Code",
        )
        assert eid == 1

        event = store.get(eid)
        assert event is not None
        assert event.event_type == "screenshot"
        assert event.summary == "User is editing code in VS Code"
        assert event.source == "desktop"
        assert event.app_name == "VS Code"
        assert event.screenshot_path is None
        assert event.chroma_id is None

    def test_insert_with_screenshot_path(self, store):
        eid = store.insert(
            event_type="screenshot",
            summary="Browsing web",
            screenshot_path="D:/AlchemyMemory/screenshots/2026/03/09/123.jpg",
        )
        event = store.get(eid)
        assert event.screenshot_path == "D:/AlchemyMemory/screenshots/2026/03/09/123.jpg"

    def test_update_chroma_id(self, store):
        eid = store.insert(event_type="voice", summary="Said hello")
        store.update_chroma_id(eid, "chroma-001")
        event = store.get(eid)
        assert event.chroma_id == "chroma-001"

    def test_get_nonexistent(self, store):
        assert store.get(999) is None

    def test_recent(self, store):
        store.insert(event_type="screenshot", summary="First", ts=100.0)
        store.insert(event_type="voice", summary="Second", ts=200.0)
        store.insert(event_type="screenshot", summary="Third", ts=300.0)

        events = store.recent(limit=2)
        assert len(events) == 2
        assert events[0].summary == "Third"  # Most recent first
        assert events[1].summary == "Second"

    def test_recent_filter_by_type(self, store):
        store.insert(event_type="screenshot", summary="Screen1", ts=100.0)
        store.insert(event_type="voice", summary="Voice1", ts=200.0)
        store.insert(event_type="screenshot", summary="Screen2", ts=300.0)

        events = store.recent(limit=10, event_types=["voice"])
        assert len(events) == 1
        assert events[0].event_type == "voice"

    def test_query_time_range(self, store):
        store.insert(event_type="screenshot", summary="Old", ts=100.0)
        store.insert(event_type="screenshot", summary="Mid", ts=200.0)
        store.insert(event_type="screenshot", summary="New", ts=300.0)

        events = store.query_time_range(start_ts=150.0, end_ts=250.0)
        assert len(events) == 1
        assert events[0].summary == "Mid"

    def test_count(self, store):
        assert store.count() == 0
        store.insert(event_type="a", summary="x")
        store.insert(event_type="b", summary="y")
        assert store.count() == 2

    def test_stats(self, store):
        store.insert(event_type="screenshot", summary="x")
        store.insert(event_type="voice", summary="y")
        stats = store.stats()
        assert stats["total_events"] == 2
        assert stats["by_type"]["screenshot"] == 1
        assert stats["by_type"]["voice"] == 1

    def test_meta_json_roundtrip(self, store):
        eid = store.insert(
            event_type="action",
            summary="Clicked button",
            meta={"x": 100, "y": 200, "target": "Submit"},
        )
        event = store.get(eid)
        assert event.meta == {"x": 100, "y": 200, "target": "Submit"}
