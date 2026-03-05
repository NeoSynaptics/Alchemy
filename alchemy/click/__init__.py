"""AlchemyClick — two-tier GUI automation (click agent).

Tier 1 (PRIMARY): Playwright accessibility tree → Qwen3 14B → ref-based actions.
  No screenshots, no coordinates. Pure structured data. Fast + reliable.
  Covers: Chrome, Electron apps (VS Code, Spotify, Slack, Discord, Notion).

Tier 2 (FALLBACK): Screenshot → Qwen2.5-VL 7B → coordinate-based actions → xdotool.
  For native Win32 apps without DOM/accessibility tree access.

For APPROVE-tier actions, the click agent pauses and requests human confirmation
before executing irreversible operations.
"""

from alchemy.click.manifest import MANIFEST

__all__ = ["MANIFEST"]
