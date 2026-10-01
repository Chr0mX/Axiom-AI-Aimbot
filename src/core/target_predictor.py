"""Velocity-based predictive aiming.

Stores the last N observed (x, y, t) positions and extrapolates forward by a
configurable time horizon.  Resets automatically when the apparent velocity
exceeds a sanity cap (detection jump rather than real motion).
"""

from __future__ import annotations

from collections import deque
from typing import Tuple


class VelocityPredictor:
    """Lightweight constant-velocity predictor for target aim-point smoothing."""

    def __init__(
        self,
        history_len: int = 3,
        max_velocity_px_per_s: float = 1200.0,
    ) -> None:
        self._history: deque[Tuple[float, float, float]] = deque(maxlen=history_len)
        self._max_velocity = max_velocity_px_per_s

    def reset(self) -> None:
        """Clear history (call when target is lost or aim deactivated)."""
        self._history.clear()

    def update(
        self,
        x: float,
        y: float,
        t: float,
        prediction_horizon_s: float,
    ) -> Tuple[float, float]:
        """Record the current observation and return the predicted future position.

        Args:
            x: Current target X coordinate (screen space).
            y: Current target Y coordinate (screen space).
            t: Timestamp from time.perf_counter().
            prediction_horizon_s: How far ahead to predict (seconds).

        Returns:
            (predicted_x, predicted_y) — falls back to (x, y) when history is
            too short or velocity exceeds the sanity cap.
        """
        self._history.append((x, y, t))

        if len(self._history) < 2:
            return x, y

        # Use the oldest and newest points to estimate velocity.
        x0, y0, t0 = self._history[0]
        x1, y1, t1 = self._history[-1]
        dt = t1 - t0
        if dt <= 0:
            return x, y

        vx = (x1 - x0) / dt
        vy = (y1 - y0) / dt

        speed = (vx * vx + vy * vy) ** 0.5
        if speed > self._max_velocity:
            # Velocity spike — likely a detection jump; discard history.
            self.reset()
            self._history.append((x, y, t))
            return x, y

        predicted_x = x1 + vx * prediction_horizon_s
        predicted_y = y1 + vy * prediction_horizon_s
        return predicted_x, predicted_y


class EmaPredictor:
    """EMA position and velocity, then a time lead.

    Same shape as a smoothed constant-velocity estimate: the position is
    an exponential moving average, velocity is an EMA of that position's
    frame-to-frame rate, and the returned point is ema + velocity * horizon.
    """

    def __init__(self, alpha: float = 0.5) -> None:
        self.alpha = alpha
        self.reset()

    def reset(self) -> None:
        self._initialized = False
        self._ema_x = self._ema_y = 0.0
        self._vx = self._vy = 0.0
        self._prev_x = self._prev_y = 0.0

    def update(
        self,
        x: float,
        y: float,
        t: float,
        prediction_horizon_s: float,
    ) -> Tuple[float, float]:
        if not self._initialized:
            self._ema_x = self._prev_x = x
            self._ema_y = self._prev_y = y
            self._last_t = t
            self._initialized = True
            return x, y

        dt = t - self._last_t
        dt = 0.001 if dt <= 0 else min(0.1, dt)
        a = self.alpha
        self._ema_x = a * x + (1.0 - a) * self._ema_x
        self._ema_y = a * y + (1.0 - a) * self._ema_y
        nvx = (self._ema_x - self._prev_x) / dt
        nvy = (self._ema_y - self._prev_y) / dt
        self._vx = a * nvx + (1.0 - a) * self._vx
        self._vy = a * nvy + (1.0 - a) * self._vy
        self._prev_x, self._prev_y = self._ema_x, self._ema_y
        self._last_t = t
        return (
            self._ema_x + self._vx * prediction_horizon_s,
            self._ema_y + self._vy * prediction_horizon_s,
        )


class RollingVelocityPredictor:
    """Average of the last few per-frame position deltas, times a frame lead.

    Lead is a frame count, not seconds: a lead of 3 means "three frames of
    the recent average step ahead of the current point."
    """

    def __init__(self, history_len: int = 5) -> None:
        self._vx: deque[float] = deque(maxlen=history_len)
        self._vy: deque[float] = deque(maxlen=history_len)
        self._prev: Tuple[float, float] | None = None

    def reset(self) -> None:
        self._vx.clear()
        self._vy.clear()
        self._prev = None

    def update(
        self,
        x: float,
        y: float,
        t: float,
        lead_frames: float,
    ) -> Tuple[float, float]:
        del t  # frame-count lead does not use the timestamp
        if self._prev is None:
            self._prev = (x, y)
            return x, y
        px, py = self._prev
        self._vx.append(x - px)
        self._vy.append(y - py)
        self._prev = (x, y)
        ax = sum(self._vx) / len(self._vx)
        ay = sum(self._vy) / len(self._vy)
        return x + ax * lead_frames, y + ay * lead_frames
