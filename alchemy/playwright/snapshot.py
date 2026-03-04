"""Capture and format Playwright accessibility tree for LLM consumption.

Uses Playwright's aria_snapshot() API (replaces deprecated page.accessibility).
Parses the YAML-like output, assigns ref IDs to interactive elements, and builds
a ref_map so the executor can locate elements by ref.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Roles that are interactive — always get a ref
INTERACTIVE_ROLES = frozenset({
    "link", "button", "textbox", "checkbox", "radio", "slider",
    "combobox", "menuitem", "tab", "listitem", "treeitem", "option",
    "switch", "searchbox", "spinbutton",
})

# Unicode control characters that some pages inject (RTL markers, etc.)
_UNICODE_JUNK_RE = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060-\u206f\ufeff]")

# Match aria_snapshot lines: "  - role "name" [props]" or "  - 'role "name"': value"
_ROLE_NAME_RE = re.compile(
    r"^(\s*-\s+)"               # indent + "- "
    r"(?:')?(\w+)\s+"           # optional quote + role + space
    r'"([^"]*)"'                # "name"
    r"(?:')?(.*)$"              # optional closing quote + rest
)

# Match unnamed role lines: "  - role:" or "  - role"
_ROLE_ONLY_RE = re.compile(r"^(\s*-\s+)(\w+)([:].*)$")


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
        raw = await page.locator(":root").aria_snapshot()
    except Exception as e:
        logger.error("Failed to capture aria snapshot: %s", e)
        return SnapshotResult(text="[Error: Could not capture accessibility tree]")

    if not raw:
        return SnapshotResult(text="[Empty page — no accessibility tree]")

    # Strip unicode control chars that break encoding
    raw = _UNICODE_JUNK_RE.sub("", raw)

    # Parse lines, inject ref IDs on interactive elements
    ref_counter = [0]
    ref_map: dict[str, RefEntry] = {}
    role_counts: dict[tuple[str, str], int] = {}
    annotated_lines: list[str] = []

    for line in raw.split("\n"):
        stripped = line.strip()

        # Skip /url: metadata lines (Playwright artifact)
        if stripped.startswith("- /url:"):
            continue

        # Try to match "- role "name" ..."
        m = _ROLE_NAME_RE.match(line)
        if m and ref_counter[0] < max_elements:
            prefix, role, name, rest = m.groups()
            if role in INTERACTIVE_ROLES:
                ref_counter[0] += 1
                ref_id = f"e{ref_counter[0]}"

                key = (role, name)
                idx = role_counts.get(key, 0)
                role_counts[key] = idx + 1
                ref_map[ref_id] = RefEntry(role=role, name=name, index=idx)

                # Inject ref marker after the name
                line = f'{prefix}{role} "{name}" [ref={ref_id}]{rest}'

        annotated_lines.append(line)

    text = "\n".join(annotated_lines)
    logger.debug(
        "Snapshot: %d interactive refs, %d lines",
        len(ref_map),
        len(annotated_lines),
    )

    return SnapshotResult(text=text, ref_map=ref_map, element_count=ref_counter[0])
