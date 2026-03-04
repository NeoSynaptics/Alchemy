"""Centralized logging configuration for Alchemy.

Call ``setup_logging()`` once at server startup to configure structured
logging with level control via the ``LOG_LEVEL`` environment variable.
"""

from __future__ import annotations

import logging
import os
import sys


def setup_logging() -> None:
    """Configure root logger with structured format and env-driven level."""
    level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(handler)

    # Quiet noisy third-party loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
