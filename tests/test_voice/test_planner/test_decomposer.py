"""Tests for task decomposition — complexity heuristics and plan parsing."""

from alchemy.voice.planner.decomposer import (
    SubTaskStatus,
    TaskPlan,
    is_complex_goal,
    parse_plan_from_text,
)


class TestComplexityDetection:
    def test_simple_question(self):
        assert is_complex_goal("What time is it?") is False

    def test_simple_command(self):
        assert is_complex_goal("Open Chrome") is False

    def test_tell_me(self):
        assert is_complex_goal("Tell me a joke") is False

    def test_multi_step_and_then(self):
        assert is_complex_goal("Open Chrome and then navigate to Google") is True

    def test_multi_step_first_second(self):
        assert is_complex_goal("First open the file, second edit it") is True

    def test_multi_step_finally(self):
        assert is_complex_goal("Do the thing, finally close the window") is True

    def test_long_goal_is_complex(self):
        goal = " ".join(["word"] * 25)
        assert is_complex_goal(goal) is True

    def test_short_ambiguous_is_simple(self):
        assert is_complex_goal("Play some music") is False


class TestParsePlan:
    def test_numbered_steps(self):
        text = """1. Open Chrome
2. Navigate to gmail.com
3. Click compose"""
        plan = parse_plan_from_text(text, "Send an email")
        assert len(plan.sub_tasks) == 3
        assert plan.sub_tasks[0].description == "Open Chrome"
        assert plan.sub_tasks[1].description == "Navigate to gmail.com"
        assert plan.sub_tasks[2].description == "Click compose"
        assert plan.sub_tasks[0].order == 1

    def test_numbered_with_parens(self):
        text = """1) Open browser
2) Go to website
3) Log in"""
        plan = parse_plan_from_text(text, "Login")
        assert len(plan.sub_tasks) == 3

    def test_bullet_fallback(self):
        text = """- Open browser
- Navigate to site
- Click button"""
        plan = parse_plan_from_text(text, "Do stuff")
        assert len(plan.sub_tasks) == 3
        assert plan.sub_tasks[0].description == "Open browser"

    def test_empty_text_single_task(self):
        plan = parse_plan_from_text("", "Just do it")
        assert len(plan.sub_tasks) == 1
        assert plan.is_simple is True
        assert plan.sub_tasks[0].description == "Just do it"

    def test_goal_preserved(self):
        plan = parse_plan_from_text("1. Step one", "My goal")
        assert plan.goal == "My goal"


class TestTaskPlan:
    def test_current_task(self):
        plan = parse_plan_from_text("1. A\n2. B\n3. C", "Test")
        assert plan.current_task.description == "A"

    def test_progress(self):
        plan = parse_plan_from_text("1. A\n2. B\n3. C", "Test")
        assert plan.progress == (0, 3)
        plan.sub_tasks[0].complete()
        assert plan.progress == (1, 3)

    def test_is_complete(self):
        plan = parse_plan_from_text("1. A\n2. B", "Test")
        assert plan.is_complete is False
        plan.sub_tasks[0].complete()
        plan.sub_tasks[1].complete()
        assert plan.is_complete is True

    def test_has_failed(self):
        plan = parse_plan_from_text("1. A\n2. B", "Test")
        assert plan.has_failed is False
        plan.sub_tasks[0].fail("Error occurred")
        assert plan.has_failed is True

    def test_skipped_counts_as_complete(self):
        plan = parse_plan_from_text("1. A\n2. B", "Test")
        plan.sub_tasks[0].complete()
        plan.sub_tasks[1].skip("Not needed")
        assert plan.is_complete is True

    def test_summary(self):
        plan = parse_plan_from_text("1. Open browser\n2. Navigate", "Browse")
        plan.sub_tasks[0].complete()
        summary = plan.summary()
        assert "OK" in summary
        assert "Open browser" in summary
