"""Tests for AlchemyConnect offline message queue."""

import pytest

from alchemy.connect.protocol import AlchemyMessage
from alchemy.connect.queue import OfflineQueue


@pytest.fixture
def queue(tmp_path):
    return OfflineQueue(data_dir=tmp_path / "connect", max_per_device=5)


class TestOfflineQueue:
    def test_enqueue_and_drain(self, queue):
        msg = AlchemyMessage(agent="chat", type="message", payload={"text": "hi"})
        queue.enqueue("device1", msg)
        messages = queue.drain("device1")
        assert len(messages) == 1
        assert messages[0].agent == "chat"
        assert messages[0].payload["text"] == "hi"

    def test_drain_empties_queue(self, queue):
        msg = AlchemyMessage(agent="chat", type="message")
        queue.enqueue("device1", msg)
        queue.drain("device1")
        assert queue.drain("device1") == []

    def test_drain_empty(self, queue):
        assert queue.drain("nonexistent") == []

    def test_fifo_order(self, queue):
        for i in range(3):
            queue.enqueue("d1", AlchemyMessage(
                agent="chat", type="message", payload={"i": i},
            ))
        messages = queue.drain("d1")
        assert [m.payload["i"] for m in messages] == [0, 1, 2]

    def test_max_per_device(self, queue):
        for i in range(10):
            queue.enqueue("d1", AlchemyMessage(
                agent="chat", type="message", payload={"i": i},
            ))
        # Max is 5, so oldest should be trimmed
        messages = queue.drain("d1")
        assert len(messages) == 5
        # Should have the last 5 (indices 5-9)
        assert messages[0].payload["i"] == 5
        assert messages[-1].payload["i"] == 9

    def test_devices_isolated(self, queue):
        queue.enqueue("d1", AlchemyMessage(agent="chat", type="m", payload={"d": 1}))
        queue.enqueue("d2", AlchemyMessage(agent="chat", type="m", payload={"d": 2}))
        m1 = queue.drain("d1")
        m2 = queue.drain("d2")
        assert len(m1) == 1
        assert len(m2) == 1
        assert m1[0].payload["d"] == 1
        assert m2[0].payload["d"] == 2

    def test_count(self, queue):
        assert queue.count("d1") == 0
        queue.enqueue("d1", AlchemyMessage(agent="chat", type="m"))
        queue.enqueue("d1", AlchemyMessage(agent="chat", type="m"))
        assert queue.count("d1") == 2

    def test_clear(self, queue):
        for _ in range(3):
            queue.enqueue("d1", AlchemyMessage(agent="chat", type="m"))
        deleted = queue.clear("d1")
        assert deleted == 3
        assert queue.count("d1") == 0
