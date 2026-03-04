"""Capture and format Playwright accessibility tree for LLM consumption.

Walks the accessibility tree, assigns ref IDs to interactive/named elements,
and formats as indented text that a 14B text model can read and act on.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Roles that are interactive — always get a ref
INTERACTIVE_ROLES = frozenset({
    "link", "button", "textbox", "checkbox", "radio", "slider",
    "combobox", "menuitem", "tab", "listitem", "treeitem", "option",
    "switch", "searchbox", "spinbutton",
})

# Roles that are structural containers — get a ref only if named
CONTAINER_ROLES = frozenset({
    "WebArea", "navigation", "main", "banner", "contentinfo", "form",
    "search", "complementary", "group", "list", "tree", "table", "row",
    "toolbar", "menu", "menubar", "tablist", "dialog", "alert", "region",
})

# Properties to include in formatted output
DISPLAY_PROPS = frozenset({
    "level", "valuetext", "valuemin", "valuemax", "checked", "pressed",
    "expanded", "selected", "disabled", "required", "readonly",
})


@dataclass
class RefEntry:
    """Mapping entry: ref ID → element locator info."""

    role: str
    name: str
    index: int = 0  # nth match for same role+name


@dataclass
class SnapshotResult:
    """Formatted accessibility tree + ref mapping."""

    text: str
    ref_map: dict[str, RefEntry] = field(default_factory=dict)
    element_count: int = 0


def _should_assign_ref(role: str, name: str) -> bool:
    """Decide if an element gets a ref ID."""
    if role in INTERACTIVE_ROLES:
        return True
    if role in CONTAINER_ROLES:
        return False
    # Headings, images, cells — assign ref if they have a name
    return bool(name)


def _format_node(
    node: dict,
    ref_counter: list[int],
    ref_map: dict[str, RefEntry],
    role_counts: dict[tuple[str, str], int],
    depth: int = 0,
    max_depth: int = 10,
    max_elements: int = 150,
) -> list[str]:
    """Recursively format an accessibility tree node."""
    if depth > max_depth or ref_counter[0] > max_elements:
        return []

    role = node.get("role", "unknown")
    name = node.get("name", "")

    # Skip the root WebArea — just process children
    if role == "WebArea" and depth == 0:
        lines = []
        for child in node.get("children", []):
            lines.extend(
                _format_node(child, ref_counter, ref_map, role_counts, depth, max_depth, max_elements)
            )
        return lines

    # Skip empty unnamed nodes
    if not name and role in CONTAINER_ROLES and not node.get("children"):
        return []

    indent = "  " * depth
    parts = [f"{indent}- {role}"]

    if name:
        # Truncate very long names
        display_name = name[:120] + "..." if len(name) > 120 else name
        parts.append(f'"{display_name}"')

    # Assign ref if appropriate
    ref_id = None
    if _should_assign_ref(role, name):
        ref_counter[0] += 1
        ref_id = f"e{ref_counter[0]}"
        parts.append(f"[ref={ref_id}]")

        # Track how many times we've seen this role+name combo
        key = (role, name)
        idx = role_counts.get(key, 0)
        role_counts[key] = idx + 1

        ref_map[ref_id] = RefEntry(role=role, name=name, index=idx)

    # Add properties
    for prop in DISPLAY_PROPS:
        val = node.get(prop)
        if val is not None:
            parts.append(f"[{prop}={val}]")

    lines = [" ".join(parts)]

    # Process children
    for child in node.get("children", []):
        lines.extend(
            _format_node(child, ref_counter, ref_map, role_counts, depth + 1, max_depth, max_elements)
        )

    return lines


async def capture_snapshot(
    page,
    max_elements: int = 150,
    max_depth: int = 10,
) -> SnapshotResult:
    """Capture the accessibility tree and format it for LLM consumption.

    Args:
        page: Playwright Page object.
        max_elements: Maximum number of ref-labeled elements.
        max_depth: Maximum tree depth to traverse.

    Returns:
        SnapshotResult with formatted text, ref mapping, and element count.
    """
    try:
        tree = await page.accessibility.snapshot()
    except Exception as e:
        logger.error("Failed to capture accessibility snapshot: %s", e)
        return SnapshotResult(text="[Error: Could not capture accessibility tree]")

    if not tree:
        return SnapshotResult(text="[Empty page — no accessibility tree]")

    ref_counter = [0]
    ref_map: dict[str, RefEntry] = {}
    role_counts: dict[tuple[str, str], int] = {}

    lines = _format_node(
        tree, ref_counter, ref_map, role_counts,
        depth=0, max_depth=max_depth, max_elements=max_elements,
    )

    text = "\n".join(lines)
    logger.debug("Snapshot: %d elements, %d refs, %d lines", ref_counter[0], len(ref_map), len(lines))

    return SnapshotResult(text=text, ref_map=ref_map, element_count=ref_counter[0])
