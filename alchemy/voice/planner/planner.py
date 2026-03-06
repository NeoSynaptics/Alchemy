"""Task planner — orchestrates goal decomposition and execution tracking.

For simple goals: pass-through to SmartRouter as-is.
For complex goals: decompose with 14B → execute sub-tasks sequentially →
track progress → handle failures.
"""

from __future__ import annotations

import logging
from uuid import UUID

from alchemy.voice.planner.decomposer import (
    SubTaskStatus,
    TaskPlan,
    is_complex_goal,
    parse_plan_from_text,
)

logger = logging.getLogger(__name__)


_DECOMPOSE_PROMPT = """Break down this goal into concrete, sequential steps.
Each step should be a single action that can be executed independently.
Number each step. Keep it concise (3-7 steps max).

Goal: {goal}

Steps:"""


class TaskPlanner:
    """Decomposes complex goals and tracks execution progress.

    Usage:
        planner = TaskPlanner()
        plan = planner.plan("Send an email to alice and then close the browser")
        # plan.is_simple = False, plan.sub_tasks = [...]
        # Execute each sub-task, update status, check plan.is_complete
    """

    def __init__(self) -> None:
        self._active_plans: dict[UUID, TaskPlan] = {}

    def plan(self, goal: str) -> TaskPlan:
        """Create a plan for a goal (local heuristic only, no model call).

        For model-assisted decomposition, use ``plan_with_model()``.
        """
        if not is_complex_goal(goal):
            task_plan = TaskPlan(goal=goal, is_simple=True)
            task_plan.sub_tasks.append(
                __import__("alchemy.voice.planner.decomposer", fromlist=["SubTask"]).SubTask(
                    description=goal, order=1
                )
            )
            self._active_plans[task_plan.id] = task_plan
            return task_plan

        # Complex goal — try to split on conjunctions
        return self._split_on_conjunctions(goal)

    def plan_from_model_output(self, goal: str, model_text: str) -> TaskPlan:
        """Parse a plan from 14B model output (used after asking model to decompose)."""
        task_plan = parse_plan_from_text(model_text, goal)
        self._active_plans[task_plan.id] = task_plan
        return task_plan

    def get_plan(self, plan_id: UUID) -> TaskPlan | None:
        return self._active_plans.get(plan_id)

    def remove_plan(self, plan_id: UUID) -> None:
        self._active_plans.pop(plan_id, None)

    def active_plans(self) -> list[TaskPlan]:
        return list(self._active_plans.values())

    @staticmethod
    def decompose_prompt(goal: str) -> str:
        """Return the prompt to send to 14B for decomposition."""
        return _DECOMPOSE_PROMPT.format(goal=goal)

    def _split_on_conjunctions(self, goal: str) -> TaskPlan:
        """Split a goal on 'and then', 'then', 'after that' etc."""
        import re

        parts = re.split(
            r"\s+(?:and then|then|after that|finally|next)\s+",
            goal,
            flags=re.IGNORECASE,
        )

        task_plan = TaskPlan(goal=goal)

        if len(parts) <= 1:
            # Can't split — single task
            task_plan.is_simple = True
            from alchemy.voice.planner.decomposer import SubTask

            task_plan.sub_tasks.append(SubTask(description=goal, order=1))
        else:
            from alchemy.voice.planner.decomposer import SubTask

            for i, part in enumerate(parts):
                cleaned = part.strip().rstrip(".,;")
                if cleaned:
                    task_plan.sub_tasks.append(
                        SubTask(description=cleaned, order=i + 1)
                    )

        self._active_plans[task_plan.id] = task_plan
        return task_plan

    def advance(self, plan_id: UUID) -> str | None:
        """Mark current sub-task as complete and return the next one's description.

        Returns None if plan is complete or not found.
        """
        plan = self._active_plans.get(plan_id)
        if plan is None:
            return None

        current = plan.current_task
        if current and current.status == SubTaskStatus.RUNNING:
            current.complete()

        next_task = plan.current_task
        if next_task is None:
            return None

        next_task.status = SubTaskStatus.RUNNING
        return next_task.description

    def fail_current(self, plan_id: UUID, error: str = "") -> None:
        """Mark the current sub-task as failed."""
        plan = self._active_plans.get(plan_id)
        if plan is None:
            return

        current = plan.current_task
        if current:
            current.fail(error)
