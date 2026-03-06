"""Tests for TaskPlanner — plan lifecycle and execution tracking."""

from alchemy.voice.planner.decomposer import SubTaskStatus
from alchemy.voice.planner.planner import TaskPlanner


class TestTaskPlanner:
    def test_simple_goal_passthrough(self):
        planner = TaskPlanner()
        plan = planner.plan("Open Chrome")
        assert plan.is_simple is True
        assert len(plan.sub_tasks) == 1
        assert plan.sub_tasks[0].description == "Open Chrome"

    def test_complex_goal_decomposed(self):
        planner = TaskPlanner()
        plan = planner.plan("Open Chrome and then go to Gmail and then compose an email")
        assert plan.is_simple is False
        assert len(plan.sub_tasks) == 3

    def test_conjunction_split(self):
        planner = TaskPlanner()
        plan = planner.plan("Download the file, then open it, after that print it")
        # Should split on "then" and "after that"
        assert len(plan.sub_tasks) >= 2

    def test_plan_stored(self):
        planner = TaskPlanner()
        plan = planner.plan("Do something")
        retrieved = planner.get_plan(plan.id)
        assert retrieved is plan

    def test_remove_plan(self):
        planner = TaskPlanner()
        plan = planner.plan("Do something")
        planner.remove_plan(plan.id)
        assert planner.get_plan(plan.id) is None

    def test_active_plans(self):
        planner = TaskPlanner()
        assert len(planner.active_plans()) == 0
        planner.plan("A")
        planner.plan("B")
        assert len(planner.active_plans()) == 2

    def test_plan_from_model_output(self):
        planner = TaskPlanner()
        model_text = "1. Open browser\n2. Go to site\n3. Click button"
        plan = planner.plan_from_model_output("Do task", model_text)
        assert len(plan.sub_tasks) == 3
        assert plan.goal == "Do task"

    def test_decompose_prompt(self):
        prompt = TaskPlanner.decompose_prompt("Send email to alice")
        assert "Send email to alice" in prompt
        assert "Steps:" in prompt


class TestPlanExecution:
    def test_advance_marks_complete(self):
        planner = TaskPlanner()
        plan = planner.plan("A and then B and then C")

        # Start first task
        desc = planner.advance(plan.id)
        assert desc is not None
        assert plan.sub_tasks[0].status == SubTaskStatus.RUNNING

        # Advance to second
        desc = planner.advance(plan.id)
        assert plan.sub_tasks[0].status == SubTaskStatus.COMPLETED
        assert plan.sub_tasks[1].status == SubTaskStatus.RUNNING

    def test_advance_returns_none_when_done(self):
        planner = TaskPlanner()
        plan = planner.plan("Just one thing")
        # Start it
        planner.advance(plan.id)
        # Complete it
        desc = planner.advance(plan.id)
        # Nothing left
        assert desc is None

    def test_fail_current(self):
        planner = TaskPlanner()
        plan = planner.plan("A and then B")
        planner.advance(plan.id)  # Start A
        planner.fail_current(plan.id, "Connection lost")
        assert plan.sub_tasks[0].status == SubTaskStatus.FAILED
        assert plan.sub_tasks[0].error == "Connection lost"

    def test_advance_unknown_plan(self):
        planner = TaskPlanner()
        from uuid import uuid4
        assert planner.advance(uuid4()) is None

    def test_fail_unknown_plan(self):
        planner = TaskPlanner()
        from uuid import uuid4
        # Should not raise
        planner.fail_current(uuid4(), "error")
