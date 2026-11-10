"""Detection-box line style shared by every surface that draws boxes.

The in-game overlay (``gui/overlay.py``) and the baked-in capture preview
(``screen_capture._draw_detection_overlay``) both call these, so the two
always agree. The Web ESP client (``web_overlay/app.js``) mirrors the same
rules in JS. Pure Python: no Qt/cv2 import, so it is testable anywhere.
"""

from __future__ import annotations

MAX_THICKNESS = 10
MAX_CORNER_PCT = 50


def box_thickness(config, confidence: float) -> int:
    """Line thickness in px.

    ``box_line_thickness`` 0 = Auto: 1-3 px by detection confidence (the
    original look). 1-10 = that fixed thickness for every box.
    """
    fixed = int(getattr(config, 'box_line_thickness', 0) or 0)
    if fixed > 0:
        return min(MAX_THICKNESS, fixed)
    return max(1, min(3, 1 + round(float(confidence) * 2)))


def highlight_thickness(config, auto: int) -> int:
    """Thickness for an extra outline drawn over a box (locked / fading lock).

    Follows a fixed ``box_line_thickness`` so the outline never hides a
    thicker box underneath; keeps its own ``auto`` width otherwise.
    """
    fixed = int(getattr(config, 'box_line_thickness', 0) or 0)
    return min(MAX_THICKNESS, fixed) if fixed > 0 else auto


def corner_length(config, width: float, height: float) -> int:
    """Length in px of each corner-bracket arm.

    ``box_corner_length_pct`` 0 = Auto: 15% of the shorter side, kept within
    6-24 px (the original look). 1-50 = that percentage of the shorter side,
    with no clamp other than at least 2 px and at most half the side, so the
    two brackets on one edge never overlap.
    """
    side = max(0.0, min(abs(float(width)), abs(float(height))))
    pct = int(getattr(config, 'box_corner_length_pct', 0) or 0)
    if pct > 0:
        pct = min(MAX_CORNER_PCT, pct)
        return int(max(2, min(side / 2.0, side * pct / 100.0)))
    return max(6, min(24, int(side * 0.15)))


# ── Overlay element line widths (FOV / Detection Range / Tracer / Predicted point) ──
# Unlike box_line_thickness these have no "Auto": 1-10 px, defaulting to the width each
# surface always drew, so an existing config looks exactly as before.
FOV_LINE_DEFAULT = 2
DETECT_RANGE_LINE_DEFAULT = 1
TRACER_LINE_DEFAULT = 2
AIM_MARKER_LINE_DEFAULT = 2


def line_thickness(config, attr: str, default: int) -> int:
    """Width in px of an overlay line: ``config.<attr>`` clamped to 1..MAX_THICKNESS,
    ``default`` when unset or not a usable number."""
    raw = getattr(config, attr, default)
    try:
        value = int(raw)
    except (TypeError, ValueError, OverflowError):
        return default
    if value <= 0:
        return default
    return min(MAX_THICKNESS, value)


def fov_thickness(config) -> int:
    return line_thickness(config, 'fov_line_thickness', FOV_LINE_DEFAULT)


def detect_range_thickness(config) -> int:
    return line_thickness(config, 'detect_range_line_thickness', DETECT_RANGE_LINE_DEFAULT)


def tracer_thickness(config) -> int:
    return line_thickness(config, 'tracer_line_thickness', TRACER_LINE_DEFAULT)


def aim_marker_thickness(config) -> int:
    return line_thickness(config, 'aim_marker_line_thickness', AIM_MARKER_LINE_DEFAULT)
