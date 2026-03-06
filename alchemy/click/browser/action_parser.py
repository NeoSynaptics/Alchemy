"""AlchemyBrowser action parser — re-export from core."""
from alchemy.core.parser import (  # noqa: F401
    ParseError,
    PlaywrightAction,
    parse_playwright_response,
    _extract_ref,
    _extract_quoted_text,
)
