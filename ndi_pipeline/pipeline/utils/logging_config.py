from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler

# Re-export the existing setup_logging so callers only need to import from here
from src.core.logging_config import setup_logging as _base_setup_logging


def setup_logging(
    level: str = "INFO",
    log_file: str | None = None,
) -> logging.Logger:
    """Configure root logger; optionally add a rotating file sink.

    Args:
        level:    Logging level name ("DEBUG", "INFO", etc.).
        log_file: Optional path for a rotating log file (10 MB, 3 backups).

    Returns:
        The configured root logger.
    """
    root = _base_setup_logging(level)

    if log_file:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        # Avoid duplicate file handlers if called more than once
        existing_paths = {
            getattr(h, "baseFilename", None)
            for h in root.handlers
            if isinstance(h, RotatingFileHandler)
        }
        if os.path.abspath(log_file) not in existing_paths:
            fh = RotatingFileHandler(
                log_file,
                maxBytes=10 * 1024 * 1024,
                backupCount=3,
                encoding="utf-8",
            )
            fh.setFormatter(logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            ))
            root.addHandler(fh)

    return root
