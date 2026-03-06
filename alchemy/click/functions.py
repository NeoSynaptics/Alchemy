"""AlchemyClick function call registry.

Three callable functions with dual visibility:
  - alchemy_click   (parent dispatcher — routes to Flow or Browser)
  - alchemy_flow    (vision + ghost cursor, native desktop)
  - alchemy_browser (Playwright + a11y tree, web/Electron)

Each function can be:
  - Called internally by Alchemy core / other modules (always available)
  - Called externally by users via API (controlled by Visibility flag)

AlchemyClick auto-selects the right sub-function based on context,
but callers can also invoke alchemy_flow or alchemy_browser directly.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Awaitable
from uuid import UUID, uuid4

from alchemy.schemas import (
    ClickCallRequest,
    ClickCallResult,
    ClickTarget,
    TaskStatus,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Visibility
# ---------------------------------------------------------------------------

class Visibility(str, Enum):
    """Who can call this function."""
    INTERNAL = "internal"    # Dev / core only — hidden from external API
    EXTERNAL = "external"    # Exposed to users via API
    BOTH = "both"            # Internal tool + external function


# ---------------------------------------------------------------------------
# Function definition
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ClickFunction:
    """A registered AlchemyClick callable."""
    name: str
    description: str
    visibility: Visibility
    target: ClickTarget           # Which path this represents
    handler: str                  # Dotted path to the async handler
    params: tuple[str, ...] = ()  # Accepted parameter names beyond 'goal'


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_FUNCTIONS: dict[str, ClickFunction] = {}


def register(fn: ClickFunction) -> ClickFunction:
    """Add a function to the registry."""
    _FUNCTIONS[fn.name] = fn
    return fn


def get_function(name: str) -> ClickFunction | None:
    return _FUNCTIONS.get(name)


def all_functions() -> list[ClickFunction]:
    return list(_FUNCTIONS.values())


def external_functions() -> list[ClickFunction]:
    """Functions visible to external callers."""
    return [f for f in _FUNCTIONS.values()
            if f.visibility in (Visibility.EXTERNAL, Visibility.BOTH)]


def internal_functions() -> list[ClickFunction]:
    """Functions available to internal callers."""
    return [f for f in _FUNCTIONS.values()
            if f.visibility in (Visibility.INTERNAL, Visibility.BOTH)]


# ---------------------------------------------------------------------------
# The three registered functions
# ---------------------------------------------------------------------------

ALCHEMY_CLICK = register(ClickFunction(
    name="alchemy_click",
    description=(
        "GUI automation dispatcher. Accepts a goal and auto-routes to "
        "AlchemyFlow (native desktop) or AlchemyBrowser (web/Electron). "
        "Set target='flow' or target='browser' to force a path."
    ),
    visibility=Visibility.BOTH,
    target=ClickTarget.AUTO,
    handler="alchemy.click.functions.dispatch_click",
    params=("goal", "target", "url", "cdp_endpoint", "context", "callback_url"),
))

ALCHEMY_FLOW = register(ClickFunction(
    name="alchemy_flow",
    description=(
        "Vision-based desktop automation. Screenshot -> Qwen2.5-VL 7B -> "
        "pixel coordinates -> ghost cursor -> SendInput. "
        "Works on any visible UI element on any Windows app."
    ),
    visibility=Visibility.BOTH,
    target=ClickTarget.FLOW,
    handler="alchemy.click.functions.dispatch_flow",
    params=("goal", "context", "callback_url"),
))

ALCHEMY_FLOW_AGENT = register(ClickFunction(
    name="alchemy_flow_agent",
    description=(
        "Fused vision-action primitive. Screenshot -> VLM -> click/drag/type "
        "in one atomic step. Internal core building block for all FlowAgent "
        "consumers. Not user-callable."
    ),
    visibility=Visibility.INTERNAL,
    target=ClickTarget.FLOW,
    handler="alchemy.click.flow.flow_agent.FlowAgent.step",
    params=("goal", "messages", "screenshot", "execute"),
))

ALCHEMY_FLOW_VS = register(ClickFunction(
    name="alchemy_flow_vs",
    description=(
        "Internal VS Code automation agent. Uses FlowAgent to click buttons "
        "and relay text inside VS Code. Toggle on/off via "
        "settings.agents.flow_vs.enabled."
    ),
    visibility=Visibility.INTERNAL,
    target=ClickTarget.FLOW,
    handler="alchemy.click.functions.dispatch_flow_vs",
    params=("goal", "label", "text", "target_element"),
))

ALCHEMY_BROWSER = register(ClickFunction(
    name="alchemy_browser",
    description=(
        "Playwright-based web/Electron automation. Accessibility tree + "
        "Qwen3 14B reasoning -> ref-based actions. "
        "Covers Chrome, VS Code, Spotify, Slack, Discord, Notion."
    ),
    visibility=Visibility.BOTH,
    target=ClickTarget.BROWSER,
    handler="alchemy.click.functions.dispatch_browser",
    params=("goal", "url", "cdp_endpoint", "context", "callback_url"),
))


# ---------------------------------------------------------------------------
# Routing logic
# ---------------------------------------------------------------------------

def _resolve_target(req: ClickCallRequest) -> ClickTarget:
    """Decide Flow vs Browser when target is AUTO."""
    if req.target != ClickTarget.AUTO:
        return req.target

    # Heuristic: if a URL or CDP endpoint is provided -> Browser
    if req.url or req.cdp_endpoint:
        return ClickTarget.BROWSER

    # Default: Flow (native desktop)
    return ClickTarget.FLOW


# ---------------------------------------------------------------------------
# Dispatchers (called by API and internal code)
# ---------------------------------------------------------------------------

async def dispatch_click(
    req: ClickCallRequest,
    *,
    app_state: Any = None,
) -> ClickCallResult:
    """Parent dispatcher — routes to Flow or Browser based on target."""
    resolved = _resolve_target(req)
    logger.info(
        "alchemy_click: goal=%r target=%s -> resolved=%s",
        req.goal, req.target.value, resolved.value,
    )

    if resolved == ClickTarget.BROWSER:
        return await dispatch_browser(req, app_state=app_state)
    return await dispatch_flow(req, app_state=app_state)


async def dispatch_flow(
    req: ClickCallRequest,
    *,
    app_state: Any = None,
) -> ClickCallResult:
    """Invoke AlchemyFlow (vision agent) directly."""
    from alchemy.click.task_manager import TaskManager
    from alchemy.click.flow.vision_agent import VisionAgent
    from alchemy.clients.voice_callback import VoiceCallbackClient
    from alchemy.models.ollama_client import OllamaClient
    from config.settings import settings

    task_id = uuid4()
    now = datetime.now(timezone.utc)

    # Resolve dependencies from app_state or create fresh
    ollama = _get_or_raise(app_state, "ollama_client", OllamaClient)
    desktop_agent = getattr(app_state, "desktop_agent", None) if app_state else None
    controller = desktop_agent._controller if desktop_agent else None
    task_manager = _get_or_create(app_state, "task_manager", TaskManager)

    voice_cb = VoiceCallbackClient(base_url=req.callback_url)
    task_manager.create_task(task_id, req.goal)

    width = settings.desktop_screenshot_width or 1920
    height = settings.desktop_screenshot_height or 1080

    agent = VisionAgent(
        ollama=ollama,
        controller=controller,
        voice_cb=voice_cb,
        task_manager=task_manager,
        model=settings.ollama_cpu_model,
        max_steps=settings.click_max_steps,
        timeout=settings.click_timeout,
        screenshot_interval=settings.click_screenshot_interval,
        approval_timeout=settings.click_approval_timeout,
        history_window=settings.click_history_window,
        screen_width=width,
        screen_height=height,
        use_streaming=settings.click_use_streaming,
        temperature=settings.ollama_temperature,
        max_tokens=settings.ollama_max_tokens,
    )

    async def _run():
        try:
            return await agent.run_task(task_id, req.goal)
        finally:
            await voice_cb.close()

    async_task = asyncio.create_task(_run())
    task_manager.register_agent_task(task_id, async_task)

    logger.info("alchemy_flow: started task %s goal=%r", task_id, req.goal)
    return ClickCallResult(
        task_id=task_id,
        status=TaskStatus.PENDING,
        target_used=ClickTarget.FLOW,
        created_at=now,
    )


async def dispatch_flow_vs(
    req: ClickCallRequest,
    *,
    app_state: Any = None,
) -> ClickCallResult:
    """Invoke AlchemyFlowVS (hidden VS Code subroutine). Internal only."""
    from config.settings import settings

    if not settings.agents.flow_vs.enabled:
        task_id = uuid4()
        now = datetime.now(timezone.utc)
        logger.warning("alchemy_flow_vs: disabled via settings")
        return ClickCallResult(
            task_id=task_id,
            status=TaskStatus.FAILED,
            target_used=ClickTarget.FLOW,
            created_at=now,
        )

    from alchemy.agents.flow_vs import FlowVSAgent

    task_id = uuid4()
    now = datetime.now(timezone.utc)

    agent = _get_or_create(app_state, "flow_vs_agent", FlowVSAgent)

    if not agent.running:
        await agent.start()

    logger.info("alchemy_flow_vs: dispatched task %s goal=%r", task_id, req.goal)
    return ClickCallResult(
        task_id=task_id,
        status=TaskStatus.PENDING,
        target_used=ClickTarget.FLOW,
        created_at=now,
    )


async def dispatch_browser(
    req: ClickCallRequest,
    *,
    app_state: Any = None,
) -> ClickCallResult:
    """Invoke AlchemyBrowser (Playwright agent) directly."""
    from alchemy.click.task_manager import TaskManager
    from alchemy.core.agent import PlaywrightAgent

    task_id = uuid4()
    now = datetime.now(timezone.utc)
    task_manager = _get_or_create(app_state, "task_manager", TaskManager)
    task_manager.create_task(task_id, req.goal)

    logger.info("alchemy_browser: started task %s goal=%r url=%r", task_id, req.goal, req.url)
    return ClickCallResult(
        task_id=task_id,
        status=TaskStatus.PENDING,
        target_used=ClickTarget.BROWSER,
        created_at=now,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_or_raise(app_state: Any, attr: str, cls: type) -> Any:
    """Get dependency from app_state or raise."""
    if app_state and hasattr(app_state, attr):
        val = getattr(app_state, attr)
        if val is not None:
            return val
    raise RuntimeError(
        f"AlchemyClick: required dependency '{attr}' not available. "
        f"Ensure the server is initialized or pass it explicitly."
    )


def _get_or_create(app_state: Any, attr: str, cls: type) -> Any:
    """Get dependency from app_state or create a fresh instance."""
    if app_state and hasattr(app_state, attr):
        val = getattr(app_state, attr)
        if val is not None:
            return val
    return cls()


# ---------------------------------------------------------------------------
# Convenience: direct call interface (for internal dev use)
# ---------------------------------------------------------------------------

async def click(goal: str, **kwargs) -> ClickCallResult:
    """Shorthand: call alchemy_click with defaults."""
    return await dispatch_click(ClickCallRequest(goal=goal, **kwargs))


async def flow(goal: str, **kwargs) -> ClickCallResult:
    """Shorthand: call alchemy_flow directly."""
    req = ClickCallRequest(goal=goal, target=ClickTarget.FLOW, **kwargs)
    return await dispatch_flow(req)


async def browser(goal: str, **kwargs) -> ClickCallResult:
    """Shorthand: call alchemy_browser directly."""
    req = ClickCallRequest(goal=goal, target=ClickTarget.BROWSER, **kwargs)
    return await dispatch_browser(req)
