"""Alchemy context router — lightweight goal enrichment for the vision agent.

Public API:
    ContextBuilder      — assembles enriched prompt context
    EnvironmentDetector — discovers installed apps and system info
    EnvironmentSnapshot — point-in-time environment data
    classify_task       — goal string → TaskCategory
    TaskCategory        — task taxonomy enum
    classify_tier_contextual — context-aware action tier classification
"""

from alchemy.router.categories import TaskCategory, classify_task
from alchemy.router.context_builder import ContextBuilder
from alchemy.router.environment import EnvironmentDetector, EnvironmentSnapshot
from alchemy.router.tier import classify_tier_contextual

__all__ = [
    "ContextBuilder",
    "EnvironmentDetector",
    "EnvironmentSnapshot",
    "TaskCategory",
    "classify_task",
    "classify_tier_contextual",
]
