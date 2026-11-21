"""One way to tell the user the result of an action.

Convention (see CLAUDE.md "UI conventions"): short results and validation messages are a non-blocking
``InfoBar`` through :func:`notify`; a modal ``QMessageBox`` is only for confirmations (Yes/No before
something destructive or slow) and for long diagnostics the user has to read and copy. Every page used
to pick one at random, with its own duration and position.
"""
from __future__ import annotations

# How long each kind stays on screen (ms). Errors stay the longest so they can be read.
DURATION_MS = {"success": 3000, "info": 4000, "warning": 6000, "error": 8000}


def notify(parent, kind: str, title: str, content: str = ""):
    """Show an InfoBar of ``kind`` ("success" | "info" | "warning" | "error") at the top of ``parent``."""
    from qfluentwidgets import InfoBar, InfoBarPosition      # deferred: this module's table stays importable without Qt
    if kind not in DURATION_MS:
        kind = "info"
    return getattr(InfoBar, kind)(
        title, content, duration=DURATION_MS[kind], isClosable=True,
        position=InfoBarPosition.TOP, parent=parent)
