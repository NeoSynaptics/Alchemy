"""Recovery strategies — per-category fallback nudges.

Injected once in the system prompt so the vision model knows
what to try if it gets stuck. Light guidance, not rigid rules.
"""

from __future__ import annotations

from alchemy.router.categories import TaskCategory

RECOVERY_STRATEGIES: dict[TaskCategory, str] = {
    TaskCategory.MEDIA: (
        "If the media app isn't visible: check the taskbar, try the Start menu, "
        "or open it via search (Win key + type). "
        "If no dedicated app exists, use the browser instead."
    ),
    TaskCategory.WEB: (
        "If the browser doesn't respond: wait 3 seconds for it to load, "
        "check if a dialog or popup is blocking, or try clicking the address bar directly. "
        "If the browser isn't open, look for it in the taskbar or launch it from the app menu."
    ),
    TaskCategory.FILE: (
        "If the file manager isn't responding: try using the terminal for file operations "
        "(ls, cp, mv, rm). If a dialog is stuck, press Escape and retry. "
        "Check the correct path before proceeding with destructive operations."
    ),
    TaskCategory.COMMUNICATION: (
        "If the messaging app isn't loading: check your internet connection indicator, "
        "wait for the app to fully load before interacting, and verify you're in the "
        "correct conversation or compose window before typing."
    ),
    TaskCategory.DEVELOPMENT: (
        "If the editor or terminal isn't responding: try clicking inside the window first, "
        "check if a save dialog or prompt is blocking. For terminal commands, ensure the "
        "previous command has finished before typing the next one."
    ),
    TaskCategory.SYSTEM: (
        "If the settings panel isn't accessible: try the Start menu and search for Settings, "
        "or use PowerShell for system commands. "
        "Some settings may require restarting the affected service."
    ),
    TaskCategory.GENERAL: (
        "If stuck: try right-clicking for context menus, check the taskbar for running apps, "
        "or use the Start menu to find what you need. "
        "If an app is unresponsive, try clicking elsewhere first, then back on the target."
    ),
}


def get_recovery(category: TaskCategory) -> str:
    """Get the recovery strategy for a task category."""
    return RECOVERY_STRATEGIES.get(category, RECOVERY_STRATEGIES[TaskCategory.GENERAL])
