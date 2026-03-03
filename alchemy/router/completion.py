"""Completion criteria — tells the model what "done" looks like.

Per-category definitions so the model doesn't stop too early
or loop forever. These are hints, not hard rules.
"""

from __future__ import annotations

from alchemy.router.categories import TaskCategory

COMPLETION_CRITERIA: dict[TaskCategory, str] = {
    TaskCategory.MEDIA: (
        "Task is complete when media is actively playing — "
        "look for visible playback controls, a moving progress bar, "
        "or an active waveform/visualizer."
    ),
    TaskCategory.WEB: (
        "Task is complete when the target page or content is fully loaded and visible. "
        "If searching, the search results should be displayed. "
        "If navigating, the destination page should be rendered."
    ),
    TaskCategory.FILE: (
        "Task is complete when the file operation is confirmed — "
        "file visible in the new location, dialog closed successfully, "
        "or terminal output confirms the operation."
    ),
    TaskCategory.COMMUNICATION: (
        "Task is complete when the message has been sent (confirmation visible) "
        "or the compose window shows the drafted content ready for review. "
        "Do NOT auto-send without explicit approval."
    ),
    TaskCategory.DEVELOPMENT: (
        "Task is complete when the code change is saved, the command has finished executing, "
        "or the expected output is visible in the terminal/editor."
    ),
    TaskCategory.SYSTEM: (
        "Task is complete when the setting change is applied and confirmed — "
        "the UI reflects the new state, or a terminal command confirms the change."
    ),
    TaskCategory.GENERAL: (
        "Task is complete when the stated goal is visibly achieved on screen. "
        "Use the finished() action to describe what was accomplished."
    ),
}


def get_completion(category: TaskCategory) -> str:
    """Get the completion criteria for a task category."""
    return COMPLETION_CRITERIA.get(category, COMPLETION_CRITERIA[TaskCategory.GENERAL])
