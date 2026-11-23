"""FOV mode: one choice instead of a switch plus a mode.

"Reduce FOV on Active Target" (``fov_reduce_on_target_enabled``) and "Shrink Mode" (``fov_shrink_mode``)
described one four-way choice with an extra "off" state sitting outside the combo:

* ``off``      — the FOV is not changed while a target is locked (the switch is off)
* ``fixed`` / ``adaptive`` / ``dynamic`` — shrink while locked (the switch is on, this mode)
* ``anchor``   — Anchor FOV: the FOV centre follows an anchor; it works without the switch

The persisted fields are unchanged; this module maps them to and from the one choice, so existing
config files, presets and tests behave exactly as before (same pattern as ``core/detect_mode.py``).

Pure Python: shared by the Qt Inference page and Web Control.
"""
from __future__ import annotations

OFF, FIXED, ADAPTIVE, DYNAMIC, ANCHOR = 'off', 'fixed', 'adaptive', 'dynamic', 'anchor'
MODES = (OFF, FIXED, ADAPTIVE, DYNAMIC, ANCHOR)
SHRINK_MODES = (FIXED, ADAPTIVE, DYNAMIC)


def mode_from_config(config) -> str:
    """The choice the two saved fields describe."""
    saved = str(getattr(config, 'fov_shrink_mode', ADAPTIVE))
    if saved == ANCHOR:
        return ANCHOR
    if not bool(getattr(config, 'fov_reduce_on_target_enabled', False)):
        return OFF
    return saved if saved in SHRINK_MODES else ADAPTIVE


def apply_mode(config, mode: str) -> str:
    """Write the saved fields for ``mode`` (an unknown value means ``off``); returns it.

    ``off`` leaves the saved shrink mode alone (so switching back restores it) unless it was
    ``anchor``, which would still read as Anchor: that falls back to ``adaptive``. ``anchor``
    leaves the on/off switch alone, as it never needed it."""
    mode = mode if mode in MODES else OFF
    if mode == OFF:
        config.fov_reduce_on_target_enabled = False
        if str(getattr(config, 'fov_shrink_mode', ADAPTIVE)) == ANCHOR:
            config.fov_shrink_mode = ADAPTIVE
    elif mode == ANCHOR:
        config.fov_shrink_mode = ANCHOR
    else:
        config.fov_reduce_on_target_enabled = True
        config.fov_shrink_mode = mode
    return mode


def shrinking(config) -> bool:
    """True when the FOV shrinks while a target is locked (Fixed / Adaptive / Dynamic)."""
    return mode_from_config(config) in SHRINK_MODES
