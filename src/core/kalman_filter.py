"""2D constant-velocity Kalman filter for aim-point smoothing."""
from __future__ import annotations

import numpy as np


class KalmanFilter2D:
    """Constant-velocity Kalman filter operating in 2D screen space.

    State: [x, y, vx, vy]  (position + velocity)
    Measurement: [x, y]    (raw detection coordinates)

    process_noise:      Q diagonal scale — how much velocity can change per frame.
                        Lower = smoother but slower to react to direction changes.
    measurement_noise:  R diagonal scale — how much we trust the detector.
                        Lower = reacts faster but noisier.
    dt:                 Fallback per-update time step, used only when a real
                        elapsed time isn't passed to update() (backward
                        compatible with any caller that doesn't supply one).
    """

    def __init__(
        self,
        process_noise: float = 0.01,
        measurement_noise: float = 0.1,
        dt: float = 1.0,
    ) -> None:
        self._dt = dt
        self._initialized = False

        # Measurement matrix H (2×4): we only observe x, y
        self._H = np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0]],
            dtype=np.float64,
        )

        # Base process/measurement noise (2×2 / 4×4 diagonals). F and the
        # effective Q/R are rebuilt per-update from these plus the actual
        # elapsed time and measurement_noise_scale — see update().
        self._Q_base = np.eye(4, dtype=np.float64) * process_noise
        self._R_base = np.eye(2, dtype=np.float64) * measurement_noise

        # Estimate covariance P (4×4) — start with high uncertainty
        self._P = np.eye(4, dtype=np.float64) * 1000.0

        # State estimate x_hat (4×1)
        self._x = np.zeros((4, 1), dtype=np.float64)

    def reset(self) -> None:
        """Clear filter state (call when target is lost)."""
        self._initialized = False
        self._P = np.eye(4, dtype=np.float64) * 1000.0
        self._x = np.zeros((4, 1), dtype=np.float64)

    def reconfigure(self, process_noise: float, measurement_noise: float) -> None:
        """Hot-swap noise parameters without resetting state."""
        self._Q_base = np.eye(4, dtype=np.float64) * process_noise
        self._R_base = np.eye(2, dtype=np.float64) * measurement_noise

    @staticmethod
    def _build_F(dt: float) -> np.ndarray:
        return np.array(
            [
                [1, 0, dt, 0],
                [0, 1, 0, dt],
                [0, 0, 1,  0],
                [0, 0, 0,  1],
            ],
            dtype=np.float64,
        )

    def update(
        self,
        x: float,
        y: float,
        dt: float | None = None,
        measurement_noise_scale: float = 1.0,
    ) -> tuple[float, float]:
        """Feed one measurement and return the filtered position estimate.

        Args:
            x, y: Raw detected position this frame.
            dt: Real elapsed time (seconds) since the previous update(). When
                omitted, falls back to the constructor's fixed dt (the
                original, frame-rate-dependent behavior) — existing callers
                that don't pass this see no change.
            measurement_noise_scale: Multiplies R for just this call, e.g. to
                trust a low-confidence or small/distant detection less
                without permanently reconfiguring the filter. 1.0 = no change.
        """
        z = np.array([[x], [y]], dtype=np.float64)

        if not self._initialized:
            # Bootstrap: set position from first measurement, zero velocity
            self._x[0, 0] = x
            self._x[1, 0] = y
            self._x[2, 0] = 0.0
            self._x[3, 0] = 0.0
            self._initialized = True
            return x, y

        step_dt = self._dt if dt is None else max(1e-4, float(dt))
        F = self._build_F(step_dt)
        # Process noise scales with elapsed time so the filter behaves
        # consistently across variable frame timing rather than assuming a
        # fixed tick — a fixed self._dt=1.0 baked into F at construction and
        # never revisited made the filter's effective smoothing/lag implicitly
        # frame-rate dependent.
        Q = self._Q_base * step_dt
        R = self._R_base * max(1e-6, float(measurement_noise_scale))

        # --- Predict ---
        x_pred = F @ self._x
        P_pred = F @ self._P @ F.T + Q

        # --- Update (Kalman gain) ---
        S = self._H @ P_pred @ self._H.T + R
        K = P_pred @ self._H.T @ np.linalg.inv(S)

        self._x = x_pred + K @ (z - self._H @ x_pred)
        self._P = (np.eye(4, dtype=np.float64) - K @ self._H) @ P_pred

        return float(self._x[0, 0]), float(self._x[1, 0])
