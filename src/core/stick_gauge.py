"""Geometry for the virtual-stick gauge drawn on every overlay surface."""

from __future__ import annotations

import math

GAUGE_RADIUS_PX = 70
GAUGE_MARGIN_PX = 24


def gauge_center(width: float, height: float, radius: float = GAUGE_RADIUS_PX,
                 margin: float = GAUGE_MARGIN_PX) -> tuple[float, float]:
    """Bottom-right of the drawable area."""
    return width - margin - radius, height - margin - radius


def stick_gauge_geometry(x: float, y: float, dz: float, game_dz: float,
                         radius: float = GAUGE_RADIUS_PX) -> dict:
    """Dot offset (clamped to the outer ring, +y down) and ring radii in px."""
    try:
        x = float(x)
        y = float(y)
    except (TypeError, ValueError):
        x = y = 0.0
    if not (math.isfinite(x) and math.isfinite(y)):
        x = y = 0.0
    mag = math.hypot(x, y)
    if mag > 1.0:
        x /= mag
        y /= mag
        mag = 1.0
    clamp = lambda v: max(0.0, min(1.0, float(v or 0.0)))  # noqa: E731
    return {
        "radius": float(radius),
        "dot_dx": x * radius,
        "dot_dy": y * radius,
        "dz_r": clamp(dz) * radius,
        "game_dz_r": clamp(game_dz) * radius,
        "deflected": mag > 0.0,
    }
