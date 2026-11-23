"""What detection does while the aim key is not held: one choice instead of two switches.

``keep_detecting`` and ``idle_detect_enabled`` were exposed as separate settings (the first
only on the web, the second on both) although together they describe one three-way choice:

* ``full``    — keep detecting at the normal rate (``keep_detecting`` on, idle off)
* ``reduced`` — keep detecting, but only every ``ai_loop.IDLE_DETECT_INTERVAL_S`` (50 ms) (``keep_detecting`` on,
  ``idle_detect_enabled`` on)
* ``stop``    — no detection (and no capture/decode work) until aiming starts
  (``keep_detecting`` off)

The persisted fields are unchanged; this module maps them to and from the one choice, so
existing config files load exactly as before. ``ai_loop`` already ignores the reduced rate
while Always Aim or Always Auto-Fire is on, so those switches no longer need to rewrite
``idle_detect_enabled`` (they used to, losing the user's own choice for good).

Pure Python: shared by the Qt Inference page and Web Control.
"""
from __future__ import annotations

FULL, REDUCED, STOP = 'full', 'reduced', 'stop'
MODES = (FULL, REDUCED, STOP)


def mode_from_config(config) -> str:
    """The choice the two saved fields describe."""
    if not bool(getattr(config, 'keep_detecting', True)):
        return STOP
    return REDUCED if bool(getattr(config, 'idle_detect_enabled', False)) else FULL


def apply_mode(config, mode: str) -> str:
    """Write the saved fields for ``mode`` (an unknown value means ``full``); returns it.

    ``stop`` leaves ``idle_detect_enabled`` alone: it has no effect while detection is
    stopped, and the user's reduced-rate preference is still there if they switch back."""
    mode = mode if mode in MODES else FULL
    if mode == STOP:
        config.keep_detecting = False
    else:
        config.keep_detecting = True
        config.idle_detect_enabled = mode == REDUCED
    return mode


def reduced_rate_ignored(config) -> bool:
    """True while the reduced rate is chosen but cannot apply (Always Aim / Always Auto-Fire
    keep detection at full speed); the UI shows a note instead of changing the setting."""
    return mode_from_config(config) == REDUCED and bool(
        getattr(config, 'always_aim', False) or getattr(config, 'always_auto_fire', False))
