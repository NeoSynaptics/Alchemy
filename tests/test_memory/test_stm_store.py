"""Tests for STMStore — short-term memory with TTL."""

import time

import pytest

from alchemy.memory.cache.store import STMStore


@pytest.fixture
def store(tmp_path):
    s = STMStore(tmp_path / "stm.db", purge_interval=60)
    s.init()
    return s


class TestSTMStore:
    def test_init_creates_db(self, tmp_path):
        db_path = tmp_path / "sub" / "stm.db"
        store = STMStore(db_path)
        store.init()
        assert db_path.exists()

    def test_insert_and_recent(self, store):
        now = time.time()
        store.insert(
            event_type="screenshot",
            summary="Coding in Python",
            app_name="VS Code",
            ts=now,
        )
        events = store.recent(window_minutes=1, limit=10)
        assert len(events) == 1
        assert events[0].summary == "Coding in Python"
        assert events[0].app_name == "VS Code"

    def test_ttl_expiry(self, store):
        old_ts = time.time() - 10
        store.insert(
            event_type="screenshot",
            summary="Old event",
            ttl_seconds=5,  # Expired 5 seconds ago
            ts=old_ts,
        )
        deleted = store.purge_expired()
        assert deleted == 1

        events = store.recent(window_minutes=60)
        assert len(events) == 0

    def test_active_apps(self, store):
        now = time.time()
        store.insert(event_type="x", summary="a", app_name="VS Code", ts=now)
        store.insert(event_type="x", summary="b", app_name="Chrome", ts=now)
        store.insert(event_type="x", summary="c", app_name="VS Code", ts=now)

        apps = store.active_apps(last_hours=1)
        assert set(apps) == {"VS Code", "Chrome"}

    def test_preferences(self, store):
        store.set_preference("language", "Python", confidence=0.9)
        store.set_preference("editor", "VS Code", confidence=0.8)

        prefs = store.get_preferences()
        assert prefs["language"] == "Python"
        assert prefs["editor"] == "VS Code"

    def test_preference_update(self, store):
        store.set_preference("language", "Python", confidence=0.5)
        store.set_preference("language", "Rust", confidence=0.9)

        prefs = store.get_preferences()
        assert prefs["language"] == "Rust"

    def test_stats(self, store):
        now = time.time()
        store.insert(event_type="x", summary="a", ts=now)
        store.set_preference("k", "v")

        stats = store.stats()
        assert stats["active_events"] == 1
        assert stats["preferences"] == 1

    def test_recent_excludes_expired(self, store):
        now = time.time()
        # Insert one expired, one active
        store.insert(event_type="x", summary="expired", ttl_seconds=0, ts=now - 10)
        store.insert(event_type="x", summary="active", ttl_seconds=3600, ts=now)

        events = store.recent(window_minutes=60)
        assert len(events) == 1
        assert events[0].summary == "active"
