"""Shim — parser accessible via alchemy.click.browser.action_parser."""
from alchemy.click.browser.action_parser import (  # noqa: F401
    ParseError,
    PlaywrightAction,
    parse_playwright_response,
    _extract_ref,
    _extract_quoted_text,
)
