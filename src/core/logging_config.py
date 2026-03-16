"""Centralized logging configuration for the application.

This module provides a single entry point to configure logging so that
all modules can rely on consistent formatting and levels. The setup is
idempotent: calling setup_logging multiple times will not duplicate handlers.
"""

import logging
from logging import Handler
from typing import Optional


_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def _has_handlers(logger: logging.Logger) -> bool:
    """Check if the logger already has non-null handlers attached."""
    return any(isinstance(h, Handler) for h in logger.handlers)


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure root logger with a sensible default format.

    Args:
        level: Logging level name (e.g., "DEBUG", "INFO").

    Returns:
        The configured root logger.
    """
    root = logging.getLogger()

    if not _has_handlers(root):
        logging.basicConfig(level=level, format=_LOG_FORMAT, datefmt=_DATE_FORMAT)
    else:
        root.setLevel(level)

    return root
