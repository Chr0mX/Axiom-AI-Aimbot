# tests/test_kalman_filter.py
"""Regression tests for src/core/kalman_filter.py's KalmanFilter2D.

Covers the two fixes made alongside the Helios-inspired motion-prediction
work: update() now accepts a real per-call `dt` (instead of a fixed value
baked into F once at construction) and a `measurement_noise_scale` (so a
low-confidence/small detection can be trusted less without permanently
reconfiguring the filter).

kalman_filter.py only imports numpy (no win_utils/cv2), so this module is
directly importable wherever numpy is available -- no import-time stubbing
needed, unlike ai_aiming.py's own tests.
"""

from core.kalman_filter import KalmanFilter2D


class TestBootstrap:
    def test_first_update_returns_raw_position_unchanged(self):
        kf = KalmanFilter2D()
        x, y = kf.update(123.0, 456.0)
        assert (x, y) == (123.0, 456.0)

    def test_first_update_ignores_dt_and_noise_scale(self):
        """Bootstrap sets position with zero velocity regardless of any
        dt/measurement_noise_scale passed on the very first call."""
        kf = KalmanFilter2D()
        x, y = kf.update(50.0, 60.0, dt=999.0, measurement_noise_scale=0.001)
        assert (x, y) == (50.0, 60.0)


class TestVariableDt:
    """Fixes the filter's dt being hardcoded at construction and never
    revisited -- a real per-update dt must actually change the predicted
    step, not just be accepted and ignored."""

    def test_larger_dt_extrapolates_further_ahead(self):
        """Isolates dt's effect on the constant-velocity prediction step (F)
        from Kalman-gain blending: with the next measurement effectively
        untrusted (measurement_noise_scale enormous), the returned position
        is essentially the pure motion-model prediction x + vx*dt, so a
        bigger dt must move further given the same learned velocity."""
        kf_small = KalmanFilter2D(process_noise=0.01, measurement_noise=0.01)
        kf_large = KalmanFilter2D(process_noise=0.01, measurement_noise=0.01)

        # Identical history for both up to this point: bootstrap at (0,0),
        # then one real update establishing a positive rightward velocity.
        kf_small.update(0.0, 0.0, dt=1.0)
        kf_small.update(10.0, 0.0, dt=1.0)
        kf_large.update(0.0, 0.0, dt=1.0)
        kf_large.update(10.0, 0.0, dt=1.0)

        x_small, _ = kf_small.update(999.0, 0.0, dt=0.1, measurement_noise_scale=1e9)
        x_large, _ = kf_large.update(999.0, 0.0, dt=5.0, measurement_noise_scale=1e9)

        assert x_large > x_small

    def test_omitted_dt_falls_back_to_constructor_dt(self):
        """Backward compatibility: a caller that never passes dt must see
        the exact same fixed-dt behavior as before this change."""
        kf_explicit = KalmanFilter2D(dt=1.0)
        kf_implicit = KalmanFilter2D(dt=1.0)

        kf_explicit.update(0.0, 0.0, dt=1.0)
        kf_implicit.update(0.0, 0.0)  # no dt -> falls back to constructor's dt=1.0

        x_explicit, y_explicit = kf_explicit.update(10.0, 0.0, dt=1.0)
        x_implicit, y_implicit = kf_implicit.update(10.0, 0.0)  # still no dt

        assert x_explicit == x_implicit
        assert y_explicit == y_implicit


class TestMeasurementNoiseScale:
    """A large measurement_noise_scale must make the filter trust a new
    measurement less, leaning more on its own motion-model prediction --
    the mechanism ai_aiming.py's _kalman_noise_scale() drives from detection
    confidence/box size."""

    def test_high_noise_scale_moves_less_toward_new_measurement(self):
        kf_trusting = KalmanFilter2D(process_noise=0.01, measurement_noise=0.1)
        kf_skeptical = KalmanFilter2D(process_noise=0.01, measurement_noise=0.1)

        kf_trusting.update(100.0, 100.0, dt=1.0)
        kf_skeptical.update(100.0, 100.0, dt=1.0)

        x_trusting, _ = kf_trusting.update(200.0, 100.0, dt=1.0, measurement_noise_scale=1.0)
        x_skeptical, _ = kf_skeptical.update(200.0, 100.0, dt=1.0, measurement_noise_scale=1e6)

        # Both estimates move toward the new measurement (200), but the
        # skeptical filter (scale=1e6, R dwarfs the prior covariance) must
        # move dramatically less than the trusting one (scale=1.0).
        assert abs(x_trusting - 100.0) > abs(x_skeptical - 100.0)
        assert x_skeptical < x_trusting

    def test_default_scale_is_unchanged_behavior(self):
        """measurement_noise_scale=1.0 (the default) must be mathematically
        identical to omitting it -- no accidental behavior change for
        existing callers that don't pass this new argument."""
        kf_a = KalmanFilter2D(process_noise=0.01, measurement_noise=0.1)
        kf_b = KalmanFilter2D(process_noise=0.01, measurement_noise=0.1)

        kf_a.update(0.0, 0.0, dt=1.0)
        kf_b.update(0.0, 0.0, dt=1.0)

        x_a, y_a = kf_a.update(10.0, 5.0, dt=1.0)
        x_b, y_b = kf_b.update(10.0, 5.0, dt=1.0, measurement_noise_scale=1.0)

        assert x_a == x_b
        assert y_a == y_b


class TestResetAndReconfigure:
    def test_reset_clears_state_back_to_bootstrap(self):
        kf = KalmanFilter2D()
        kf.update(1.0, 1.0, dt=1.0)
        kf.update(2.0, 2.0, dt=1.0)
        kf.reset()
        # Post-reset, the next update() must behave like a fresh bootstrap.
        x, y = kf.update(50.0, 60.0)
        assert (x, y) == (50.0, 60.0)

    def test_reconfigure_does_not_reset_position_state(self):
        kf = KalmanFilter2D(process_noise=0.01, measurement_noise=0.1)
        kf.update(1.0, 1.0, dt=1.0)  # bootstrap
        kf.reconfigure(process_noise=0.5, measurement_noise=0.5)
        # Still initialized -- reconfigure only swaps noise parameters.
        x, y = kf.update(2.0, 2.0, dt=1.0)
        # A real KF update (not a second bootstrap) blends toward, but
        # doesn't necessarily equal, the raw measurement.
        assert x != 2.0 or y != 2.0
