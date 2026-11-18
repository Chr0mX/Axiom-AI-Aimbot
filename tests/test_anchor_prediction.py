"""Tests for core/anchor_prediction.py (pure Python, runs anywhere).

Accuracy tests compare the prediction against where the simulated target
actually is one horizon later, and against the "no prediction" baseline (the
current position).
"""
import math
import random
from types import SimpleNamespace

import pytest

from core.anchor_prediction import (
    TRAIL_LEN, AnchorParams, AnchorPredictor, ModeAdjust, _axis_lead, adjust_for_mode,
)

W, H = 40.0, 80.0
T0 = 1000.0


def make(**kw):
    p = AnchorParams()
    p.horizon_s = 0.05
    p.max_distance_px = 500.0
    p.confidence_threshold = 0.2
    for k, v in kw.items():
        setattr(p, k, v)
    return AnchorPredictor(p)


def simulate(pred, pos, fps, seconds, *, horizon=0.05, skip=None, agility=0.0, t_off=0.0):
    """Feed ``pos(t)`` -> (x, y) at ``fps``; return [(t, pred_xy, truth_xy, now_xy)]
    for every fed frame."""
    out = []
    n = int(seconds * fps)
    for i in range(n):
        if skip is not None and skip(i):
            continue
        t = i / fps + t_off
        x, y = pos(t)
        pred.update(x, y, T0 + t, W, H, det_conf=0.95, agility=agility)
        px, py = pred.predict()
        tx, ty = pos(t + horizon)
        out.append((t, (px, py), (tx, ty), (x, y)))
    return out


def err(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def mean_err(rows, t_from, key_pred=1):
    sel = [r for r in rows if r[0] >= t_from]
    return sum(err(r[key_pred], r[2]) for r in sel) / len(sel)


class TestStationary:
    def test_no_lead_and_exact_position(self):
        p = make()
        rows = simulate(p, lambda t: (500.0, 300.0), 60, 1.0)
        for _, pr, _, now in rows[5:]:
            assert pr == pytest.approx(now, abs=1e-6)
        assert p.lead_dist == 0.0

    def test_jitter_does_not_become_a_lead(self):
        rng = random.Random(1)
        p = make()
        rows = simulate(p, lambda t: (500.0 + rng.uniform(-1, 1), 300.0 + rng.uniform(-1, 1)), 60, 2.0)
        assert max(r[1][0] - 500.0 for r in rows[10:]) < 6.0


class TestConstantVelocity:
    @pytest.mark.parametrize("fps", [30, 60, 144, 240])
    def test_accurate_at_any_frame_rate(self, fps):
        p = make()
        pos = lambda t: (200.0 + 600.0 * t, 400.0 - 150.0 * t)
        rows = simulate(p, pos, fps, 1.5)
        pred_err = mean_err(rows, 0.6)
        base_err = mean_err(rows, 0.6, key_pred=3)
        assert base_err > 30.0                      # standing still would be ~31 px off
        assert pred_err < 3.0
        assert pred_err < base_err / 8.0

    def test_dropped_frames_and_jittery_timestamps(self):
        rng = random.Random(7)
        p = make()
        pos = lambda t: (200.0 + 500.0 * t, 400.0)
        rows = simulate(p, pos, 90, 2.0, skip=lambda i: rng.random() < 0.25)
        assert mean_err(rows, 0.8) < 4.0
        assert mean_err(rows, 0.8) < mean_err(rows, 0.8, key_pred=3) / 4.0

    def test_noisy_measurements_still_beat_baseline(self):
        rng = random.Random(3)
        p = make(smoothing=0.5)
        pos_clean = lambda t: (200.0 + 400.0 * t, 400.0)
        n = 120
        errs, base = [], []
        for i in range(n):
            t = i / 60.0
            x, y = pos_clean(t)
            p.update(x + rng.gauss(0, 1.5), y + rng.gauss(0, 1.5), T0 + t, W, H, 0.9)
            px, py = p.predict()
            if i > 30:
                errs.append(err((px, py), pos_clean(t + 0.05)))
                base.append(err((x, y), pos_clean(t + 0.05)))
        assert sum(errs) / len(errs) < sum(base) / len(base) / 2.5


class TestAcceleration:
    def test_acceleration_term_improves_prediction(self):
        pos = lambda t: (100.0 + 0.5 * 1500.0 * t * t, 300.0)
        kw = dict(smoothing=0.3, max_speed_px_s=5000.0, horizon_s=0.1)
        with_a = make(accel_influence=1.0, **kw)
        no_a = make(accel_influence=0.0, **kw)
        rows_a = simulate(with_a, pos, 120, 1.5, horizon=0.1)
        rows_n = simulate(no_a, pos, 120, 1.5, horizon=0.1)
        e_a, e_n = mean_err(rows_a, 0.7), mean_err(rows_n, 0.7)
        base = mean_err(rows_a, 0.7, key_pred=3)
        assert e_a < e_n < base
        assert e_a < base / 6.0

    def test_deceleration_never_reverses_the_lead(self):
        p = make(accel_influence=1.0, smoothing=0.2)
        p.params.horizon_s = 0.4
        v0, decel = 900.0, 3000.0
        t_stop = v0 / decel

        def pos(t):
            if t < t_stop:
                return 100.0 + v0 * t - 0.5 * decel * t * t, 200.0
            return 100.0 + v0 * t_stop - 0.5 * decel * t_stop ** 2, 200.0

        rows = simulate(p, pos, 120, 1.0, horizon=0.4)
        stop_x = pos(t_stop)[0]
        for t, pr, _, now in rows:
            if t < t_stop:
                assert pr[0] >= now[0] - 1e-6        # while slowing down: never leads backwards
            else:
                assert abs(pr[0] - stop_x) < 80.0    # after the stop: bounded overshoot even at a 0.4 s horizon
        assert rows[-1][1][0] == pytest.approx(stop_x, abs=1.0)

    def test_axis_lead_stops_at_standstill(self):
        assert _axis_lead(100.0, -1000.0, 0.5) == pytest.approx(5.0)
        assert _axis_lead(100.0, -1000.0, 0.05) == pytest.approx(100 * 0.05 - 0.5 * 1000 * 0.05 ** 2)
        assert _axis_lead(-100.0, 1000.0, 0.5) == pytest.approx(-5.0)
        # Speeding up: the speed gained over the horizon is capped at the current speed.
        assert _axis_lead(0.0, 500.0, 0.1) == pytest.approx(0.0)
        assert _axis_lead(100.0, 5000.0, 0.1) == pytest.approx(100 * 0.1 + 0.5 * 1000.0 * 0.1 ** 2)
        assert _axis_lead(100.0, 500.0, 0.1) == pytest.approx(100 * 0.1 + 0.5 * 500.0 * 0.1 ** 2)


class TestDirectionChange:
    @staticmethod
    def zigzag(t):
        return (500.0 + 600.0 * t, 300.0) if t < 0.5 else (500.0 + 600.0 * 0.5 - 600.0 * (t - 0.5), 300.0)

    def test_recovers_after_a_sudden_reversal(self):
        p = make(smoothing=0.1)
        rows = simulate(p, self.zigzag, 120, 1.5)
        assert mean_err(rows, 0.9) < 3.0

    def test_agility_cuts_the_post_reversal_lag(self):
        calm = make(smoothing=0.7)
        agile = make(smoothing=0.7)
        rows_c = simulate(calm, self.zigzag, 120, 1.0)
        rows_a = simulate(agile, self.zigzag, 120, 1.0, agility=1.0)
        window = lambda rows: [r for r in rows if 0.5 <= r[0] <= 0.75]
        e_c = sum(err(r[1], r[2]) for r in window(rows_c))
        e_a = sum(err(r[1], r[2]) for r in window(rows_a))
        assert e_a < e_c

    def test_reversal_does_not_overshoot_wildly(self):
        p = make(smoothing=0.4)
        rows = simulate(p, self.zigzag, 120, 1.5)
        assert max(err(r[1], r[3]) for r in rows) < 120.0   # lead stays near horizon*speed


class TestTemporaryLoss:
    def test_short_gap_coasts_with_bounded_drift(self):
        p = make()
        v = 600.0
        for i in range(30):
            t = i / 60.0
            p.update(100.0 + v * t, 200.0, T0 + t, W, H, 0.95)
        t_last = 29 / 60.0
        x_last = 100.0 + v * t_last
        gap = 0.2
        p.update(x_last + v * gap, 200.0, T0 + t_last + gap, W, H, 0.95)
        px, _ = p.predict()
        truth_now = x_last + v * gap
        assert abs(p.x - truth_now) < v * gap
        assert p.lead_dist <= p.params.max_distance_px
        assert p.n == 31                                   # continued, not restarted

    def test_long_gap_restarts_clean(self):
        p = make()
        for i in range(30):
            p.update(100.0 + 600.0 * i / 60.0, 200.0, T0 + i / 60.0, W, H, 0.95)
        p.update(900.0, 500.0, T0 + 0.5 + 1.0, W, H, 0.95)
        assert p.n == 1
        assert (p.vx, p.vy, p.ax, p.ay) == (0.0, 0.0, 0.0, 0.0)
        assert p.predict() == (900.0, 500.0)
        assert p.lead_dist == 0.0

    def test_reacquisition_has_no_stale_lead(self):
        p = make()
        for i in range(30):
            p.update(100.0 + 600.0 * i / 60.0, 200.0, T0 + i / 60.0, W, H, 0.95)
        p.reset()
        p.update(800.0, 200.0, T0 + 10.0, W, H, 0.95)
        assert p.predict() == (800.0, 200.0)


class TestTargetSwitchAndGuards:
    def test_impossible_speed_is_treated_as_a_new_target(self):
        p = make(max_speed_px_s=1200.0)
        for i in range(20):
            p.update(100.0 + 300.0 * i / 60.0, 200.0, T0 + i / 60.0, W, H, 0.95)
        p.update(1500.0, 900.0, T0 + 20 / 60.0, W, H, 0.95)    # a jump of ~1500 px in 16 ms
        assert p.n == 1
        assert p.vx == 0.0

    def test_non_finite_and_out_of_order_samples_are_ignored(self):
        p = make()
        p.update(100.0, 100.0, T0, W, H)
        p.update(110.0, 100.0, T0 + 0.016, W, H)
        snap = (p.x, p.y, p.vx, p.n, p.last_t)
        p.update(float('nan'), 100.0, T0 + 0.03, W, H)
        p.update(120.0, 100.0, T0 + 0.01, W, H)        # older than last_t
        p.update(120.0, 100.0, T0 + 0.016, W, H)       # same timestamp
        assert (p.x, p.y, p.vx, p.n, p.last_t) == snap

    def test_low_detection_confidence_suppresses_the_lead(self):
        pos = lambda t: (200.0 + 600.0 * t, 400.0)
        hi = make(confidence_threshold=0.6)
        lo = make(confidence_threshold=0.6)
        for i in range(60):
            t = i / 60.0
            hi.update(*pos(t), T0 + t, W, H, 0.95)
            lo.update(*pos(t), T0 + t, W, H, 0.05)
            hi.predict()
            lo.predict()
        assert hi.lead_dist > 20.0
        assert lo.lead_dist == 0.0
        assert lo.confidence < hi.confidence

    def test_lead_fades_in_with_confidence_not_a_step(self):
        p = make(confidence_threshold=0.5)
        leads = []
        for i in range(40):
            t = i / 60.0
            p.update(200.0 + 600.0 * t, 400.0, T0 + t, W, H, 0.95)
            p.predict()
            leads.append(p.lead_dist)
        steps = [b - a for a, b in zip(leads, leads[1:])]
        assert max(leads) == pytest.approx(30.0, abs=1.0)
        assert max(steps) < 0.6 * max(leads)        # ramps over several frames, no single jump

    def test_max_distance_cap_and_strength(self):
        pos = lambda t: (100.0 + 1000.0 * t, 200.0)
        capped = make(max_distance_px=25.0)
        capped.params.horizon_s = 0.5
        free = make(max_distance_px=0.0)
        free.params.horizon_s = 0.5
        half = make(max_distance_px=0.0, strength=0.5)
        half.params.horizon_s = 0.5
        for p in (capped, free, half):
            simulate(p, pos, 60, 1.0, horizon=0.5)
        assert capped.lead_dist == pytest.approx(25.0)
        assert free.lead_dist > 400.0
        assert half.lead_dist == pytest.approx(free.lead_dist * 0.5, rel=0.02)

    def test_velocity_influence_scales_the_lead(self):
        pos = lambda t: (100.0 + 600.0 * t, 200.0)
        full, none = make(accel_influence=0.0), make(accel_influence=0.0, velocity_influence=0.0)
        for p in (full, none):
            simulate(p, pos, 60, 1.0)
        assert full.lead_dist > 20.0
        assert none.lead_dist == 0.0


class TestMultipleObjects:
    def test_instances_are_independent(self):
        a, b = make(), make()
        for i in range(40):
            t = i / 60.0
            a.update(100.0 + 500.0 * t, 100.0, T0 + t, W, H, 0.95)
            b.update(900.0 - 300.0 * t, 600.0, T0 + t, W, H, 0.95)
        a.predict()
        b.predict()
        assert a.vx == pytest.approx(500.0, rel=0.05)
        assert b.vx == pytest.approx(-300.0, rel=0.05)
        a_before = (a.x, a.vx, a.n)
        b.reset()
        assert (a.x, a.vx, a.n) == a_before
        assert b.n == 0


class TestFovClamp:
    @staticmethod
    def moving(n=24, **kw):
        p = make(max_distance_px=0.0, **kw)
        p.params.horizon_s = 0.5
        for i in range(n):
            t = i / 60.0
            p.update(500.0 + 400.0 * t, 300.0, T0 + t, W, H, 0.95)
            p.predict()
        return p

    def test_without_a_fov_the_lead_is_unbounded(self):
        p = self.moving()
        assert p.predict()[0] > p.x + 150.0

    def test_rectangle_fov_clamps_the_predicted_point(self):
        p = self.moving()
        p.set_fov(500.0, 300.0, 400.0, 300.0)             # right edge at x=700
        px, py = p.predict()
        assert px == pytest.approx(700.0)
        assert p.lead_dist == pytest.approx(math.hypot(px - p.x, py - p.y))
        assert p.lead_dist < 150.0

    def test_ellipse_fov_clamps_radially(self):
        p = self.moving()
        p.set_fov(500.0, 300.0, 400.0, 300.0, ellipse=True)
        px, py = p.predict()
        nx, ny = (px - 500.0) / 200.0, (py - 300.0) / 150.0
        assert nx * nx + ny * ny <= 1.0 + 1e-9

    def test_anchor_already_outside_the_fov_keeps_its_lead_direction(self):
        p = self.moving(n=40)
        p.set_fov(100.0, 300.0, 100.0, 100.0)            # anchor (~760) is far outside
        assert p.predict()[0] > p.x                      # not flipped back towards the FOV

    def test_fov_survives_reset(self):
        p = self.moving()
        p.set_fov(500.0, 300.0, 400.0, 300.0)
        p.reset()
        assert p._fov is not None
        p.set_fov(0, 0, 0, 0)
        assert p._fov is None


class TestParamsAndTrail:
    def test_update_from_config_reads_fields_and_ignores_garbage(self):
        cfg = SimpleNamespace(prediction_horizon_ms=40.0, anchor_strength=0.7, anchor_smoothing='x',
                              anchor_velocity_influence=1.2, anchor_accel_influence=float('nan'),
                              anchor_max_distance_pct=30, anchor_confidence_threshold=0.5,
                              prediction_max_velocity=900.0, fov_size=300, fov_height=200,
                              fov_effective_size=200, fov_effective_height=100)
        p = AnchorParams().update_from(cfg)
        assert p.horizon_s == pytest.approx(0.04)
        assert p.strength == 0.7
        assert p.smoothing == 0.4                     # unparsable -> default kept
        assert p.accel_influence == 0.35              # NaN -> default kept
        assert p.max_speed_px_s == 900.0
        # 30 % of the shorter side of the *effective* (shrunk) FOV, not the configured one.
        assert p.max_distance_px == pytest.approx(30.0)
        assert AnchorParams().update_from(SimpleNamespace()).horizon_s == pytest.approx(0.01)
        # Falls back to the configured FOV when no shrink is in effect.
        plain = AnchorParams().update_from(SimpleNamespace(anchor_max_distance_pct=50, fov_size=200, fov_height=160))
        assert plain.max_distance_px == pytest.approx(80.0)
        # Percentage 0 = no distance cap (the FOV clamp still applies).
        assert AnchorParams().update_from(SimpleNamespace(anchor_max_distance_pct=0, fov_size=200)).max_distance_px == 0.0

    def test_trail_is_ordered_oldest_first_and_wraps(self):
        p = make()
        for i in range(TRAIL_LEN + 5):
            p.update(float(i * 3), 0.0, T0 + i / 60.0, W, H, 0.95)
        tr = p.trail()
        assert len(tr) == TRAIL_LEN
        xs = [pt[0] for pt in tr]
        assert xs == sorted(xs)
        assert xs[-1] == pytest.approx(p.x)


class TestModeAdjust:
    @staticmethod
    def tracker(turn=0.0, stab=0.9, hit=1.0, conf=0.95):
        return SimpleNamespace(turn_ema=turn, stab_ema=stab, hit_ema=hit, conf_ema=conf, edge_ratio=0.0)

    def test_fixed_and_missing_tracker_are_neutral(self):
        out = ModeAdjust()
        for mode, tr in (('fixed', self.tracker()), ('adaptive', None), ('bogus', self.tracker())):
            adjust_for_mode(mode, tr, out)
            assert (out.strength_scale, out.horizon_scale, out.agility) == (1.0, 1.0, 0.0)

    def test_adaptive_eases_off_towards_the_fov_edge(self):
        out = ModeAdjust()
        centre = adjust_for_mode('adaptive', self.tracker(), out).strength_scale
        edge_tr = self.tracker()
        edge_tr.edge_ratio = 0.95
        mid_tr = self.tracker()
        mid_tr.edge_ratio = 0.4
        edge = adjust_for_mode('adaptive', edge_tr, out).strength_scale
        mid = adjust_for_mode('adaptive', mid_tr, out).strength_scale
        assert edge == pytest.approx(centre * 0.6, rel=0.05)
        assert mid == pytest.approx(centre, rel=0.01)       # no penalty well inside the FOV

    def test_adaptive_trusts_consistent_motion_more(self):
        out = ModeAdjust()
        steady = adjust_for_mode('adaptive', self.tracker(), out).strength_scale
        turning = adjust_for_mode('adaptive', self.tracker(turn=0.45, stab=0.4, hit=0.5, conf=0.6), out).strength_scale
        assert steady > 0.9
        assert 0.35 <= turning < steady - 0.3
        assert out.horizon_scale == 1.0 and out.agility == 0.0

    def test_dynamic_shortens_horizon_on_turns_and_is_agile(self):
        out = ModeAdjust()
        adjust_for_mode('dynamic', self.tracker(), out)
        clean_h, clean_s = out.horizon_scale, out.strength_scale
        assert out.agility == 1.0
        assert clean_h > 1.0
        adjust_for_mode('dynamic', self.tracker(turn=0.4), out)
        assert out.horizon_scale < clean_h * 0.6
        assert out.strength_scale < clean_s

    def test_dynamic_mutes_less_than_adaptive_when_shaky(self):
        out = ModeAdjust()
        shaky = self.tracker(turn=0.3, stab=0.4, hit=0.6, conf=0.7)
        a = adjust_for_mode('adaptive', shaky, out).strength_scale
        d = adjust_for_mode('dynamic', shaky, out).strength_scale
        assert d > a
