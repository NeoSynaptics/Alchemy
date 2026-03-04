"""Execute parsed actions on Playwright pages using ref-based locators.

Maps ref IDs back to ARIA role+name locators for deterministic interaction.
No pixel coordinates, no guessing — refs come from the accessibility tree.
"""

from __future__ import annotations

import logging

from alchemy.playwright.snapshot import RefEntry

logger = logging.getLogger(__name__)


class ExecutionError(Exception):
    """Raised when an action cannot be executed."""


async def execute_action(
    page,
    action_type: str,
    ref: str | None,
    ref_map: dict[str, RefEntry],
    text: str | None = None,
    direction: str | None = None,
    key_name: str | None = None,
    timeout: float = 5000,
) -> bool:
    """Execute a single action on the page.

    Args:
        page: Playwright Page object.
        action_type: One of: click, type, scroll, key, select, wait, done.
        ref: Element ref ID (e.g., "e5"). Required for click, type, select.
        ref_map: Mapping from ref IDs to RefEntry (role, name, index).
        text: Text to type or option to select.
        direction: Scroll direction ("up" or "down").
        key_name: Key to press (e.g., "Enter", "Tab", "Escape").
        timeout: Action timeout in milliseconds.

    Returns:
        True if action was executed successfully.

    Raises:
        ExecutionError: If the action cannot be executed.
    """
    if action_type in ("done", "wait"):
        if action_type == "wait":
            try:
                await page.wait_for_load_state("networkidle", timeout=timeout)
            except Exception:
                pass  # Timeout is okay for wait
        return True

    if action_type == "scroll":
        delta = -300 if direction == "up" else 300
        await page.mouse.wheel(0, delta)
        return True

    if action_type == "key":
        if not key_name:
            raise ExecutionError("key action requires a key_name")
        await page.keyboard.press(key_name)
        return True

    # Actions that require a ref
    if not ref:
        raise ExecutionError(f"{action_type} action requires a ref")

    entry = ref_map.get(ref)
    if not entry:
        raise ExecutionError(f"Unknown ref: {ref}")

    locator = _build_locator(page, entry)

    if action_type == "click":
        await locator.click(timeout=timeout)
        logger.debug("Clicked %s (%s '%s')", ref, entry.role, entry.name)
        return True

    if action_type == "type":
        if text is None:
            raise ExecutionError("type action requires text")
        await locator.fill(text, timeout=timeout)
        logger.debug("Typed into %s: %r", ref, text[:50])
        return True

    if action_type == "select":
        if text is None:
            raise ExecutionError("select action requires text")
        await locator.select_option(label=text, timeout=timeout)
        logger.debug("Selected '%s' in %s", text, ref)
        return True

    raise ExecutionError(f"Unknown action type: {action_type}")


def _build_locator(page, entry: RefEntry):
    """Build a Playwright locator from a RefEntry.

    Uses get_by_role with name matching. If multiple elements share the
    same role+name, uses .nth() to pick the right one.
    """
    locator = page.get_by_role(entry.role, name=entry.name)

    if entry.index > 0:
        locator = locator.nth(entry.index)

    return locator
