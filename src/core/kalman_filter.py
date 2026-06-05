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
    """

    def __init__(
        self,
        process_noise: float = 0.01,
        measurement_noise: float = 0.1,
        dt: float = 1.0,
    ) -> None:
        self._dt = dt
        self._initialized = False

        # State transition matrix F (4×4)
        self._F = np.array(
            [
                [1, 0, dt, 0],
                [0, 1, 0, dt],
                [0, 0, 1,  0],
                [0, 0, 0,  1],
            ],
            dtype=np.float64,
        )

        # Measurement matrix H (2×4): we only observe x, y
        self._H = np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0]],
            dtype=np.float64,
        )

        # Process noise covariance Q (4×4)
        self._Q = np.eye(4, dtype=np.float64) * process_noise

        # Measurement noise covariance R (2×2)
        self._R = np.eye(2, dtype=np.float64) * measurement_noise

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
        self._Q = np.eye(4, dtype=np.float64) * process_noise
        self._R = np.eye(2, dtype=np.float64) * measurement_noise

    def update(self, x: float, y: float) -> tuple[float, float]:
        """Feed one measurement and return the filtered position estimate."""
        z = np.array([[x], [y]], dtype=np.float64)

        if not self._initialized:
            # Bootstrap: set position from first measurement, zero velocity
            self._x[0, 0] = x
            self._x[1, 0] = y
            self._x[2, 0] = 0.0
            self._x[3, 0] = 0.0
            self._initialized = True
            return x, y

        # --- Predict ---
        x_pred = self._F @ self._x
        P_pred = self._F @ self._P @ self._F.T + self._Q

        # --- Update (Kalman gain) ---
        S = self._H @ P_pred @ self._H.T + self._R
        K = P_pred @ self._H.T @ np.linalg.inv(S)

        self._x = x_pred + K @ (z - self._H @ x_pred)
        self._P = (np.eye(4, dtype=np.float64) - K @ self._H) @ P_pred

        return float(self._x[0, 0]), float(self._x[1, 0])
