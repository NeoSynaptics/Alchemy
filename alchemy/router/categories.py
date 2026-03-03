"""Task category taxonomy — lightweight keyword classification.

Classifies raw goal strings into broad categories to provide the vision
model with orientation. NOT an ML classifier — just pattern matching
to give the 72B a map, not GPS turn-by-turn.
"""

from __future__ import annotations

import re
from enum import Enum


class TaskCategory(str, Enum):
    MEDIA = "media"
    WEB = "web"
    FILE = "file"
    COMMUNICATION = "communication"
    DEVELOPMENT = "development"
    SYSTEM = "system"
    GENERAL = "general"


# Patterns: (compiled regex, category, weight)
# Higher weight wins when multiple categories match.
_PATTERNS: list[tuple[re.Pattern, TaskCategory, int]] = [
    # Media
    (re.compile(r"\b(spotify|music|song|playlist|play|pause|volume|audio|video|youtube|vlc|media\s*player|stream|podcast|mp3|mp4)\b", re.I), TaskCategory.MEDIA, 10),
    # Web
    (re.compile(r"\b(browser|firefox|chrome|url|website|web\s*page|search|google|bing|http|navigate|bookmark|tab)\b", re.I), TaskCategory.WEB, 10),
    # File
    (re.compile(r"\b(file|folder|directory|copy|move|rename|delete|create\s+file|save\s+as|download|upload|zip|extract|nautilus|explorer)\b", re.I), TaskCategory.FILE, 10),
    # Communication
    (re.compile(r"\b(email|mail|message|slack|discord|teams|send|compose|reply|chat|telegram|whatsapp)\b", re.I), TaskCategory.COMMUNICATION, 10),
    # Development
    (re.compile(r"\b(code|terminal|git|editor|vscode|vim|compile|debug|run\s+script|python|npm|pip|IDE)\b", re.I), TaskCategory.DEVELOPMENT, 10),
    # System
    (re.compile(r"\b(settings|wifi|bluetooth|update|install|uninstall|display|brightness|network|printer|permission|system)\b", re.I), TaskCategory.SYSTEM, 8),
]

# Per-category orientation hints. {placeholders} are filled by ContextBuilder.
CATEGORY_HINTS: dict[TaskCategory, str] = {
    TaskCategory.MEDIA: (
        "This is a media task. "
        "Available media apps: {media_apps}. "
        "Audio output depends on PulseAudio configuration."
    ),
    TaskCategory.WEB: (
        "This is a web browsing task. "
        "Default browser: {default_browser}. "
        "Look for the browser in the taskbar or application menu."
    ),
    TaskCategory.FILE: (
        "This is a file management task. "
        "File manager: {file_manager}. "
        "Use the file manager or terminal for file operations."
    ),
    TaskCategory.COMMUNICATION: (
        "This is a communication task. "
        "Available: {comm_apps}. "
        "Be careful with send actions — they may require approval."
    ),
    TaskCategory.DEVELOPMENT: (
        "This is a development task. "
        "Available: {dev_apps}. "
        "Terminal is always available for command-line operations."
    ),
    TaskCategory.SYSTEM: (
        "This is a system configuration task. "
        "Desktop: {desktop}. "
        "System settings may be accessible via the desktop menu or terminal commands."
    ),
    TaskCategory.GENERAL: (
        "Check the taskbar and application menu for available tools."
    ),
}


def classify_task(goal: str) -> TaskCategory:
    """Classify a goal string into a task category.

    Uses weighted keyword matching. Returns GENERAL if no patterns match.
    """
    scores: dict[TaskCategory, int] = {}

    for pattern, category, weight in _PATTERNS:
        matches = pattern.findall(goal)
        if matches:
            scores[category] = scores.get(category, 0) + len(matches) * weight

    if not scores:
        return TaskCategory.GENERAL

    return max(scores, key=scores.get)


def get_hint(category: TaskCategory) -> str:
    """Get the raw hint template for a category. Placeholders need filling."""
    return CATEGORY_HINTS.get(category, CATEGORY_HINTS[TaskCategory.GENERAL])
