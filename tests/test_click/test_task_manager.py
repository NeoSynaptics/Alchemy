"""Task manager tests — state management and signaling."""

import asyncio
from uuid import uuid4

import pytest

from alchemy.click.task_manager import TaskManager
from alchemy.schemas import TaskStatus


class TestTaskManager:
    def test_create_and_get(self):
        tm = TaskManager()
        tid = uuid4()
        state = tm.create_task(tid, "test goal")
        assert state.task_id == tid
        assert state.goal == "test goal"
        assert state.status == TaskStatus.PENDING
        assert tm.get_task(tid) is state

    def test_get_missing(self):
        tm = TaskManager()
        assert tm.get_task(uuid4()) is None

    def test_update(self):
        tm = TaskManager()
        tid = uuid4()
        tm.create_task(tid, "goal")
        tm.update_task(tid, status=TaskStatus.RUNNING, current_step=5)
        state = tm.get_task(tid)
        assert state.status == TaskStatus.RUNNING
        assert state.current_step == 5

    def test_approve_sets_event(self):
        tm = TaskManager()
        tid = uuid4()
        tm.create_task(tid, "goal")
        tm.approve(tid)
        state = tm.get_task(tid)
        assert state.approved is True
        assert state.approval_event.is_set()

    def test_deny_sets_event_and_cancel(self):
        tm = TaskManager()
        tid = uuid4()
        tm.create_task(tid, "goal")
        tm.deny(tid)
        state = tm.get_task(tid)
        assert state.approved is False
        assert state.approval_event.is_set()
        assert state.cancel_event.is_set()

    def test_to_status_response(self):
        tm = TaskManager()
        tid = uuid4()
        tm.create_task(tid, "goal")
        resp = tm.to_status_response(tid)
        assert resp is not None
        assert resp.task_id == tid
        assert resp.status == TaskStatus.PENDING

    def test_to_status_response_missing(self):
        tm = TaskManager()
        assert tm.to_status_response(uuid4()) is None

    async def test_cancel_cancels_asyncio_task(self):
        tm = TaskManager()
        tid = uuid4()
        tm.create_task(tid, "goal")

        async def dummy():
            await asyncio.sleep(100)

        task = asyncio.create_task(dummy())
        tm.register_agent_task(tid, task)
        tm.cancel(tid)
        await asyncio.sleep(0)  # let cancellation propagate
        assert task.cancelled()
