"""Optional aim-move shaping.

The default path is the PID. The other paths travel a fraction of the
remaining error each frame (1 - sensitivity), the same control law a
curve-based aimer uses instead of a PID. Bezier controls are bowed
perpendicular to the error so the sample is not identical to a straight lerp.
"""

from __future__ import annotations

import math

PATHS = ("pid", "linear", "exponential", "bezier", "adaptive", "perlin")

# One step of shape_movement() is a 60 Hz step. The aim loop often runs
# several times faster; scale_path_step() shrinks the step with the real
# tick so a 240 Hz loop does not travel four times too far.
REFERENCE_DT = 1.0 / 60.0

# Same per-command ceiling Aimmy applies before handing the step to a device.
PATH_STEP_LIMIT = 150.0


def scale_path_step(dx: float, dy: float, dt: float) -> tuple[float, float]:
    """Shrink a 60 Hz path step to this aim tick. A stall does not enlarge it."""
    if dt <= 0.0 or dt >= REFERENCE_DT:
        return dx, dy
    scale = dt / REFERENCE_DT
    return dx * scale, dy * scale


def clamp_path_step(dx: float, dy: float, limit: float = PATH_STEP_LIMIT) -> tuple[float, float]:
    return max(-limit, min(limit, dx)), max(-limit, min(limit, dy))


def shape_movement(
    error_x: float,
    error_y: float,
    path: str,
    sensitivity: float,
    curve: float,
    phase: float,
) -> tuple[float, float]:
    t = max(0.01, min(0.99, 1.0 - float(sensitivity)))
    kind = path if path in PATHS and path != "pid" else "linear"
    if kind == "exponential":
        # 1 - (sensitivity - 0.2), then cubed. Using (1 - sensitivity)^3
        # alone left the default 0.80 sensitivity at 0.8% of the error per
        # step, which never accumulated into a MAKCU report.
        exp_t = max(0.05, min(0.95, 1.0 - (float(sensitivity) - 0.2)))
        scale = exp_t ** 3
        return error_x * scale, error_y * scale
    if kind == "bezier":
        return _bezier(error_x, error_y, t, curve)
    if kind == "adaptive":
        if math.hypot(error_x, error_y) < 100.0:
            return error_x * t, error_y * t
        return _bezier(error_x, error_y, t, curve)
    if kind == "perlin":
        bx, by = error_x * t, error_y * t
        length = math.hypot(error_x, error_y) or 1.0
        px, py = -error_y / length, error_x / length
        noise = _noise(phase) * 8.0
        return bx + px * noise, by + py * noise
    return error_x * t, error_y * t


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


def _bezier(ex: float, ey: float, t: float, curve: float) -> tuple[float, float]:
    length = math.hypot(ex, ey)
    if length < 1e-6:
        return 0.0, 0.0
    px, py = -ey / length, ex / length
    bow = float(curve) * length
    c1x, c1y = ex / 3.0 + px * bow, ey / 3.0 + py * bow
    c2x, c2y = 2.0 * ex / 3.0 - px * bow, 2.0 * ey / 3.0 - py * bow
    u = 1.0 - t
    x = 3 * u * u * t * c1x + 3 * u * t * t * c2x + t * t * t * ex
    y = 3 * u * u * t * c1y + 3 * u * t * t * c2y + t * t * t * ey
    return x, y


def _noise(x: float) -> float:
    i = math.floor(x)
    f = x - i
    u = f * f * (3.0 - 2.0 * f)

    def h(n: float) -> float:
        v = (int(n) * 1664525 + 1013904223) & 0xFFFFFFFF
        return v / 0xFFFFFFFF * 2.0 - 1.0

    return h(i) * (1.0 - u) + h(i + 1) * u
