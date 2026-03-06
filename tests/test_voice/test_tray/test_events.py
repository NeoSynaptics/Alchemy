"""Tests for TrayEventBus — pure Python, no GUI needed."""

import asyncio
from unittest.mock import MagicMock

from alchemy.voice.tray.events import TrayEvent, TrayEventBus, TrayMessage


class TestTrayEventBus:
    def test_push_and_poll(self):
        bus = TrayEventBus()
        msg = TrayMessage(event=TrayEvent.NOTIFICATION, data={"key": "value"})
        bus.push(msg)
        result = bus.poll()
        assert result is msg

    def test_poll_empty_returns_none(self):
        bus = TrayEventBus()
        assert bus.poll() is None

    def test_pending_count(self):
        bus = TrayEventBus()
        assert bus.pending_count() == 0
        bus.push(TrayMessage(event=TrayEvent.NOTIFICATION, data={}))
        bus.push(TrayMessage(event=TrayEvent.TASK_UPDATE, data={}))
        assert bus.pending_count() == 2

    def test_fifo_order(self):
        bus = TrayEventBus()
        msg1 = TrayMessage(event=TrayEvent.NOTIFICATION, data={"order": 1})
        msg2 = TrayMessage(event=TrayEvent.TASK_UPDATE, data={"order": 2})
        bus.push(msg1)
        bus.push(msg2)
        assert bus.poll() is msg1
        assert bus.poll() is msg2

    def test_event_types(self):
        assert TrayEvent.APPROVAL_REQUEST == "approval_request"
        assert TrayEvent.NOTIFICATION == "notification"
        assert TrayEvent.TASK_UPDATE == "task_update"
        assert TrayEvent.VOICE_STATE == "voice_state"

    async def test_respond_sets_future(self):
        loop = asyncio.get_event_loop()
        bus = TrayEventBus()
        bus.bind_loop(loop)

        future = loop.create_future()
        msg = TrayMessage(
            event=TrayEvent.APPROVAL_REQUEST,
            data={},
            response_future=future,
        )

        bus.respond(msg, approved=True)
        # Let the call_soon_threadsafe callback run
        await asyncio.sleep(0.05)
        assert future.done()
        assert future.result() is True

    async def test_respond_deny(self):
        loop = asyncio.get_event_loop()
        bus = TrayEventBus()
        bus.bind_loop(loop)

        future = loop.create_future()
        msg = TrayMessage(
            event=TrayEvent.APPROVAL_REQUEST,
            data={},
            response_future=future,
        )

        bus.respond(msg, approved=False)
        await asyncio.sleep(0.05)
        assert future.result() is False

    def test_respond_no_future_is_safe(self):
        bus = TrayEventBus()
        bus.bind_loop(MagicMock())
        msg = TrayMessage(event=TrayEvent.NOTIFICATION, data={})
        # Should not raise
        bus.respond(msg, approved=True)

    def test_respond_no_loop_is_safe(self):
        bus = TrayEventBus()
        future = MagicMock()
        msg = TrayMessage(
            event=TrayEvent.APPROVAL_REQUEST,
            data={},
            response_future=future,
        )
        # No loop bound — should not raise
        bus.respond(msg, approved=True)

    def test_message_repr(self):
        msg = TrayMessage(event=TrayEvent.NOTIFICATION, data={"a": 1})
        r = repr(msg)
        assert "NOTIFICATION" in r
        # Future should not appear in repr
        assert "response_future" not in r
