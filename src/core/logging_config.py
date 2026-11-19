"""Centralized logging configuration for the application.

This module provides a single entry point to configure logging so that
all modules can rely on consistent formatting and levels. The setup is
idempotent: calling setup_logging multiple times will not duplicate handlers.
"""

import logging
import threading
import time
from logging import Handler


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


# ── throttled logging for hot paths ──────────────────────────────────────────
# Handlers that must never break a per-frame loop used to be ``except: pass``,
# which hid real failures (e.g. a MAKCU read error silently disabling aiming).
# Logging them every frame would flood the console instead, so log each distinct
# problem at most once per interval.

_throttle_lock = threading.Lock()
_throttle_last: dict = {}


def log_throttled(logger: logging.Logger, key: str, level: int, msg: str, *args,
                  interval_s: float = 5.0, exc_info: bool = False) -> bool:
    """Log ``msg % args`` unless ``key`` was already logged within ``interval_s``.

    ``key`` identifies the call site/problem (not the formatted text), so repeated
    failures of one kind are one line per interval while different problems each
    get their own. Returns True when a record was emitted. Cheap enough for the
    inference loop: the lock is only taken when something actually failed.
    """
    now = time.monotonic()
    with _throttle_lock:
        last = _throttle_last.get(key)
        if last is not None and now - last < interval_s:
            return False
        _throttle_last[key] = now
    logger.log(level, msg, *args, exc_info=exc_info)
    return True


def reset_log_throttle() -> None:
    """Forget all throttle state (tests)."""
    with _throttle_lock:
        _throttle_last.clear()
