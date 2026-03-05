"""Static accept/deny policies for the Gate module.

Two-tier system:
  Tier 1 (instant) — known-safe tools and commands are auto-accepted,
                      known-destructive commands are auto-denied.
  Tier 2 (review)  — anything ambiguous goes to the LLM reviewer.
"""

from __future__ import annotations

import re
from enum import Enum


class PolicyDecision(str, Enum):
    ACCEPT = "accept"
    DENY = "deny"
    REVIEW = "review"


# --- Safe tools (always accept, no inference needed) ---
_SAFE_TOOLS: set[str] = {
    "Read", "Glob", "Grep", "WebFetch", "WebSearch",
    "TodoWrite", "AskUserQuestion",
}

# --- Safe Bash command prefixes (always accept) ---
_SAFE_BASH_PREFIXES: list[str] = [
    "git status", "git diff", "git log", "git branch", "git show",
    "git remote", "git stash list", "git rev-parse",
    "ls", "pwd", "echo", "cat ", "head ", "tail ", "wc ",
    "npm test", "npm run test", "npm run lint", "npm run build",
    "npx jest", "npx tsc", "npx eslint",
    "pytest", "python -m pytest", "python -c ",
    "pip list", "pip show", "pip freeze",
    "node --version", "npm --version", "python --version",
    "which ", "where ", "type ",
    "date", "hostname", "whoami",
    "grep ", "rg ", "find ", "fd ",
]

# --- Destructive patterns (always deny) ---
_DENY_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"rm\s+-r[f ]*\s*/(?!\S*node_modules)", re.IGNORECASE),  # rm -r[f] /anything
    re.compile(r"rm\s+-r[f ]*\s*~", re.IGNORECASE),   # rm -r[f] ~
    re.compile(r"rm\s+-r[f ]*\s*\.$", re.IGNORECASE),  # rm -r[f] .
    re.compile(r"git\s+push\s+.*--force\s+.*(?:main|master)", re.IGNORECASE),
    re.compile(r"git\s+push\s+-f\s+.*(?:main|master)", re.IGNORECASE),
    re.compile(r"DROP\s+(?:TABLE|DATABASE)", re.IGNORECASE),
    re.compile(r"TRUNCATE\s+TABLE", re.IGNORECASE),
    re.compile(r"--no-verify", re.IGNORECASE),
    re.compile(r"format\s+[a-zA-Z]:", re.IGNORECASE),  # format C:
    re.compile(r"mkfs\.", re.IGNORECASE),
    re.compile(r"dd\s+if=.+of=/dev/", re.IGNORECASE),
]

# --- Sensitive file patterns (deny writes to these) ---
_SENSITIVE_FILE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\.env$", re.IGNORECASE),
    re.compile(r"\.env\.", re.IGNORECASE),
    re.compile(r"credentials", re.IGNORECASE),
    re.compile(r"secrets?\.", re.IGNORECASE),
    re.compile(r"\.pem$", re.IGNORECASE),
    re.compile(r"\.key$", re.IGNORECASE),
    re.compile(r"id_rsa", re.IGNORECASE),
    re.compile(r"\.ssh/", re.IGNORECASE),
]


def check_static_policy(
    tool_name: str,
    args: dict[str, str],
) -> tuple[PolicyDecision, str]:
    """Check static policies for a tool call.

    Returns:
        (PolicyDecision, reason) — ACCEPT/DENY are final, REVIEW means
        the call should be forwarded to the LLM reviewer.
    """
    # --- Safe tools (no args inspection needed) ---
    if tool_name in _SAFE_TOOLS:
        return PolicyDecision.ACCEPT, f"safe tool: {tool_name}"

    # --- Bash commands ---
    if tool_name == "Bash":
        command = args.get("command", "")
        return _check_bash_command(command)

    # --- Write/Edit: check for sensitive files ---
    if tool_name in ("Write", "Edit", "NotebookEdit"):
        file_path = args.get("file_path", "")
        for pat in _SENSITIVE_FILE_PATTERNS:
            if pat.search(file_path):
                return PolicyDecision.DENY, f"sensitive file: {file_path}"
        # Non-sensitive writes → review (model decides)
        return PolicyDecision.REVIEW, f"file write: {file_path}"

    # --- Everything else → review ---
    return PolicyDecision.REVIEW, f"unknown tool: {tool_name}"


def _check_bash_command(command: str) -> tuple[PolicyDecision, str]:
    """Check a Bash command against static policies."""
    stripped = command.strip()

    if not stripped:
        return PolicyDecision.ACCEPT, "empty command"

    # Check destructive patterns first (deny takes priority)
    for pat in _DENY_PATTERNS:
        if pat.search(stripped):
            return PolicyDecision.DENY, f"destructive command: {stripped[:80]}"

    # Check safe prefixes
    lower = stripped.lower()
    for prefix in _SAFE_BASH_PREFIXES:
        if lower.startswith(prefix.lower()):
            return PolicyDecision.ACCEPT, f"safe command: {stripped[:80]}"

    # Ambiguous → review
    return PolicyDecision.REVIEW, f"bash: {stripped[:80]}"
