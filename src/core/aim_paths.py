"""Small aim-math helpers: output EMA smoothing and the adaptive prediction lead.

Movement is always driven by the PID; the old alternative movement paths
(linear / exponential / bezier / adaptive / perlin) were removed.
"""

from __future__ import annotations


def ema_step(previous: float, current: float, factor: float) -> float:
    f = max(0.0, min(1.0, float(factor)))
    return current * f + previous * (1.0 - f)


def adaptive_lead_s(horizon_s: float, mouse_speed_px_s: float) -> float:
    """Stretch a time lead when the mouse is moving slowly.

    A slow correction takes longer to arrive, so the lead grows toward
    40% of an estimated completion time (100 px / speed), scaled by the
    user's horizon relative to 0.10 s, and clamped to 20–300 ms.
    """
    if mouse_speed_px_s <= 1.0:
        return horizon_s
    estimated = 100.0 / mouse_speed_px_s
    scaled = (estimated * 0.4) * (horizon_s / 0.10)
    return max(0.02, min(0.30, scaled))
