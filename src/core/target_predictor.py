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

    def reconfigure(self, history_len: int, max_velocity_px_per_s: float) -> None:
        """Apply new settings in place, preserving observations.

        Mirrors KalmanFilter2D.reconfigure(). Callers previously reached into
        the private attributes and rebuilt the deque on *every frame* just to
        pick up settings that almost never change; this makes that a no-op
        unless something actually changed.
        """
        self._max_velocity = max_velocity_px_per_s
        if history_len != self._history.maxlen:
            # Rebuild only on a real change. deque(existing, maxlen=n) keeps
            # the most recent n entries, so tightening the window drops the
            # oldest rather than losing the track.
            self._history = deque(self._history, maxlen=history_len)

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
