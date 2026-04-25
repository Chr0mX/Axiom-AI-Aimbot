"""
smart_tracker.py — Lightweight EMA-based target position smoother.

Sits between detection output and PID input.
No velocity prediction, no target extrapolation — purely reactive smoothing.

Pipeline position:
    Detection (x, y)  →  SmartTracker.update()  →  PID error calculation
"""

from __future__ import annotations

import math
from typing import Optional, Tuple


class SmartTracker:
    """
    Lightweight target position smoother using Exponential Moving Average (EMA).

    Design goals:
    - Zero latency spikes: O(1) per update, no heap allocations
    - No overshoot prediction: operates only on observed coordinates
    - Safe cold start: first call seeds state without discontinuity
    - Toggleable: when disabled, update() is a pass-through

    EMA formula:
        smoothed = alpha * raw + (1 - alpha) * prev_smoothed

        alpha = 1.0  →  pure pass-through (identical to disabled)
        alpha = 0.6  →  moderate smoothing (recommended default)
        alpha → 0    →  very heavy smoothing (high lag)

    Velocity dampening (optional, very light):
        When enabled and the raw target jumps more than `_DAMPEN_THRESHOLD`
        pixels per frame, the contribution of the raw coordinate is reduced
        proportionally.  This is NOT prediction — it only slightly softens
        large sudden jumps to prevent the EMA from being pulled away too fast.
        PID corrects the remaining error on subsequent frames.
    """

    _DAMPEN_THRESHOLD: float = 25.0   # px/frame above which dampening kicks in

    def __init__(self, alpha: float = 0.6) -> None:
        self.alpha: float = max(0.0, min(1.0, alpha))
        self.velocity_dampen: bool = False
        # Scale applied to displacement excess above threshold (0 = full stop, 1 = no effect)
        self.dampen_factor: float = 0.85

        self._sx: Optional[float] = None    # EMA-smoothed X
        self._sy: Optional[float] = None    # EMA-smoothed Y
        self._prev_x: Optional[float] = None
        self._prev_y: Optional[float] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, raw_x: float, raw_y: float) -> Tuple[float, float]:
        """
        Ingest a new raw detection coordinate; return the smoothed coordinate.

        Args:
            raw_x, raw_y: Raw target position from the detector.

        Returns:
            (smoothed_x, smoothed_y) ready for PID error calculation.
        """
        # Cold start — seed with first observation (no lag on first frame)
        if self._sx is None:
            self._sx = raw_x
            self._sy = raw_y
            self._prev_x = raw_x
            self._prev_y = raw_y
            return raw_x, raw_y

        eff_x, eff_y = raw_x, raw_y

        # Optional velocity dampening — reduces influence of sudden large jumps
        if self.velocity_dampen:
            vx = raw_x - self._prev_x  # type: ignore[operator]
            vy = raw_y - self._prev_y  # type: ignore[operator]
            velocity = math.hypot(vx, vy)
            if velocity > self._DAMPEN_THRESHOLD:
                # Reduce the raw displacement excess, keeping direction intact
                excess_ratio = (velocity - self._DAMPEN_THRESHOLD) / velocity
                scale = max(self.dampen_factor, 1.0 - excess_ratio * (1.0 - self.dampen_factor))
                eff_x = self._prev_x + vx * scale  # type: ignore[operator]
                eff_y = self._prev_y + vy * scale  # type: ignore[operator]

        # EMA update
        a = self.alpha
        self._sx = a * eff_x + (1.0 - a) * self._sx  # type: ignore[operator]
        self._sy = a * eff_y + (1.0 - a) * self._sy  # type: ignore[operator]

        self._prev_x = raw_x
        self._prev_y = raw_y

        return self._sx, self._sy  # type: ignore[return-value]

    def reset(self) -> None:
        """Clear all cached state.  Call when no targets are detected."""
        self._sx = None
        self._sy = None
        self._prev_x = None
        self._prev_y = None
