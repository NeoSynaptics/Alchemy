"""AlchemyClick — the overarching GUI automation behavior contract.

AlchemyClick defines HOW the agent behaves when interacting with any GUI.
It owns the shared rules: task lifecycle, approval gates, action tier
classification, and the 10 proven patterns.

Two execution paths live underneath:

  AlchemyBrowser (alchemy.click.browser)
    Playwright accessibility tree + Qwen3 14B → ref-based actions.
    For web apps and Electron apps (Chrome, VS Code, Spotify, Slack).

  AlchemyFlow (alchemy.click.flow)
    Screenshot → Qwen2.5-VL 7B → pixel coordinates → ghost cursor.
    For native Win32 apps and anything without a DOM.

Shared components at this level:
  - TaskManager: task lifecycle + approval signaling (used by both paths)
  - Patterns: the 10 proven capabilities registry
  - Parent manifest: declares the module to Alchemy's registry
"""

from alchemy.click.manifest import MANIFEST
from alchemy.click.task_manager import TaskManager
from alchemy.click.functions import (
    ALCHEMY_BROWSER,
    ALCHEMY_CLICK,
    ALCHEMY_FLOW,
    ALCHEMY_FLOW_AGENT,
    ALCHEMY_FLOW_VS,
    Visibility,
    all_functions,
    browser,
    click,
    dispatch_browser,
    dispatch_click,
    dispatch_flow,
    dispatch_flow_vs,
    external_functions,
    flow,
    get_function,
    internal_functions,
)

__all__ = [
    "MANIFEST",
    "TaskManager",
    # Function registry
    "ALCHEMY_CLICK",
    "ALCHEMY_FLOW",
    "ALCHEMY_FLOW_AGENT",
    "ALCHEMY_FLOW_VS",
    "ALCHEMY_BROWSER",
    "Visibility",
    "get_function",
    "all_functions",
    "external_functions",
    "internal_functions",
    # Dispatchers
    "dispatch_click",
    "dispatch_flow",
    "dispatch_flow_vs",
    "dispatch_browser",
    # Shorthand
    "click",
    "flow",
    "browser",
]
