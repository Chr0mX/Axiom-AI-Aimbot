"""Tests for core/anchor_prediction.py (pure Python, runs anywhere).

Accuracy tests compare the prediction with where the simulated box actually is
one horizon later, and with the "no prediction" baseline (the current box).
"""
import math
import random
from types import SimpleNamespace

import pytest

from core.anchor_prediction import (
    MODES, AnchorBank, AnchorParams, AnchorTrack, AnchorView, FovAdjuster, _axis_lead, anchor_active,
)

W, H = 40.0, 80.0
T0 = 1000.0


def box(cx, cy, w=W, h=H):
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def make(mode="velocity_accel", **kw):
    p = AnchorParams()
    p.mode = mode
    p.horizon_s = 0.10
    p.accel_comp = 1.0
    p.conf_threshold = 0.2
    p.max_speed_px_s = 1e6
    p.history_len = 6
    for k, v in kw.items():
        setattr(p, k, v)
    return AnchorTrack(p)


def run(tr, pos, fps, seconds, *, horizon=None, skip=None, conf=0.95, t_off=0.0):
    """Feed pos(t) -> (cx, cy). Rows: (t, predicted centre, true centre at t+H, current centre)."""
    H_ = tr.params.horizon_s if horizon is None else horizon
    rows = []
    for i in range(int(seconds * fps)):
        if skip is not None and skip(i):
            continue
        t = i / fps + t_off
        cx, cy = pos(t)
        tr.update(T0 + t, box(cx, cy), conf)
        p = tr.predict()
        rows.append((t, (p.pcx, p.pcy), pos(t + H_), (cx, cy)))
    return rows


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def mean_err(rows, t_from, key=1):
    sel = [r for r in rows if r[0] >= t_from]
    return sum(dist(r[key], r[2]) for r in sel) / len(sel)


class TestGateAndParams:
    def test_anchor_active_needs_enable_and_a_real_mode(self):
        assert not anchor_active(SimpleNamespace())
        assert not anchor_active(SimpleNamespace(anchor_enabled=True, anchor_mode="disabled"))
        assert not anchor_active(SimpleNamespace(anchor_enabled=False, anchor_mode="adaptive"))
        for m in MODES[1:]:
            assert anchor_active(SimpleNamespace(anchor_enabled=True, anchor_mode=m))

    def test_update_from_reads_fields_ignores_garbage_and_uses_configured_fov(self):
        cfg = SimpleNamespace(
            anchor_mode="smoothed", anchor_horizon_ms=150, anchor_strength=0.7, anchor_velocity_smoothing="x",
            anchor_accel_comp=float("nan"), anchor_confidence_threshold=0.5, anchor_history_len=99,
            anchor_max_distance_pct=30, fov_size=300, fov_height=200,
            fov_effective_size=100, fov_effective_height=50, prediction_max_velocity=900,
            anchor_fov_influence_adaptive=0.2, anchor_fov_influence_dynamic=0.9)
        p = AnchorParams().update_from(cfg)
        assert p.mode == "smoothed" and p.horizon_s == pytest.approx(0.15) and p.strength == 0.7
        assert p.smoothing == 0.4 and p.accel_comp == 0.35        # garbage / NaN -> defaults kept
        assert p.conf_threshold == 0.5 and p.history_len == 12     # clamped to the buffer
        assert p.max_distance_px == pytest.approx(60.0)            # 30 % of min(300, 200): the *configured* FOV
        assert p.max_speed_px_s == 900 and p.influence_adaptive == 0.2 and p.influence_dynamic == 0.9
        assert AnchorParams().update_from(SimpleNamespace(anchor_mode="bogus")).mode == "adaptive"


class TestEstimator:
    def test_fit_recovers_exact_velocity_and_acceleration(self):
        tr = make("velocity_accel", history_len=8)
        pos = lambda t: (100 + 300 * t + 400 * t * t, 50 - 100 * t - 200 * t * t)
        run(tr, pos, 60, 0.3)
        t_last = (int(0.3 * 60) - 1) / 60
        vlx, vly, ax, ay, _, _, _, _, tau_bar = tr._fit(6, True)
        assert ax == pytest.approx(800, rel=1e-6) and ay == pytest.approx(-400, rel=1e-6)
        # the line's slope is the velocity at the window's mean time; moving it to the newest sample
        # by the acceleration gives the exact end-of-window velocity
        assert vlx - ax * tau_bar == pytest.approx(300 + 800 * t_last, rel=1e-6)
        assert vly - ay * tau_bar == pytest.approx(-100 - 400 * t_last, rel=1e-6)

    def test_strong_acceleration_passes_the_noise_gate_and_is_recovered_exactly(self):
        tr = make("velocity_accel", history_len=8)
        pos = lambda t: (100 + 300 * t + 2000 * t * t, 50 - 100 * t - 1000 * t * t)    # a = (4000, -2000) px/s^2
        run(tr, pos, 60, 0.3)
        t_last = (int(0.3 * 60) - 1) / 60
        assert tr.ax == pytest.approx(4000, rel=1e-6) and tr.ay == pytest.approx(-2000, rel=1e-6)
        assert tr.vx == pytest.approx(300 + 4000 * t_last, rel=1e-6)
        assert tr.vy == pytest.approx(-100 - 2000 * t_last, rel=1e-6)

    def test_weak_acceleration_leaves_the_low_noise_linear_velocity_untouched(self):
        tr = make("velocity_accel", history_len=6)
        run(tr, lambda t: (100 + 300 * t + 400 * t * t, 300), 60, 0.3)     # 800 px/s^2 is under the noise floor
        t_last = (int(0.3 * 60) - 1) / 60
        assert tr.ax == 0.0
        assert abs(tr.vx - (300 + 800 * t_last)) < 800 * 0.05              # = v at the window mean, off by a*|tau_bar|

    def test_size_velocity_is_tracked(self):
        tr = make("linear")
        for i in range(10):
            t = i / 60
            tr.update(T0 + t, box(300, 300, w=40 + 60 * t, h=80 + 120 * t), 0.9)
        assert tr.vw == pytest.approx(60, rel=1e-6) and tr.vh == pytest.approx(120, rel=1e-6)

    def test_history_length_limits_the_window(self):
        # 600 px/s until t=0.15 s, then a kink down to 100 px/s.
        pos = lambda t: (200 + 600 * t, 300) if t < 0.15 else (200 + 90 + 100 * (t - 0.15), 300)
        short, long_ = make("linear", history_len=4), make("linear", history_len=12)
        for tr in (short, long_):
            run(tr, pos, 60, 0.25)
        # the short window sees only the new regime; the long one is still contaminated by the old one
        assert short.vx == pytest.approx(100, abs=1e-6)
        assert long_.vx > 150

    def test_ring_buffer_wraps_without_error(self):
        tr = make("velocity_accel", history_len=12)
        run(tr, lambda t: (100 + 200 * t, 100), 60, 2.0)
        assert tr.samples == 12 and tr.vx == pytest.approx(200, rel=1e-6)


class TestModes:
    # Acceleration is only trusted when it clears its own noise floor, which needs a long enough
    # window (the estimate's error falls with the 4th power of the window span) or a strong enough
    # acceleration: 4000 px/s^2 over 12 samples at 60 FPS (a 0.18 s window) qualifies.
    accel_pos = staticmethod(lambda t: (100 + 0.5 * 4000 * t * t, 300))

    def test_linear_ignores_acceleration_velocity_accel_uses_it(self):
        errs = {}
        for mode in ("linear", "velocity_accel"):
            tr = make(mode, history_len=12, max_speed_px_s=1e6)
            rows = run(tr, self.accel_pos, 60, 0.9)
            errs[mode] = mean_err(rows, 0.5)
            errs["base"] = mean_err(rows, 0.5, key=3)
            errs[mode + "_ax"] = tr.pred.ax
        assert errs["linear_ax"] == 0.0 and errs["velocity_accel_ax"] > 1000.0
        assert errs["velocity_accel"] < errs["linear"] < errs["base"]

    def test_acceleration_that_cannot_be_told_from_jitter_is_ignored(self):
        rng = random.Random(4)
        tr = make("velocity_accel", history_len=6)
        worst = 0.0
        for i in range(300):                                   # constant velocity + 1 px detector jitter
            t = i / 60
            tr.update(T0 + t, box(200 + 400 * t + rng.gauss(0, 1.0), 300 + rng.gauss(0, 1.0)), 0.9)
            p = tr.predict()
            if i > 30:
                worst = max(worst, math.hypot(p.ax, p.ay))
        assert worst < 1200.0 and tr.acc_thr > 500.0           # the noise floor is far above what leaks through

    def test_smoothed_reduces_prediction_jitter_vs_raw(self):
        rng = random.Random(5)
        noise = [(rng.gauss(0, 1.5), rng.gauss(0, 1.5)) for _ in range(240)]

        def jitter(mode):
            tr = make(mode, smoothing=0.7, accel_comp=0.0)
            leads = []
            for i in range(240):
                t = i / 60
                tr.update(T0 + t, box(200 + 300 * t + noise[i][0], 300 + noise[i][1]), 0.9)
                p = tr.predict()
                if i > 30:
                    leads.append(p.dx)
            m = sum(leads) / len(leads)
            return math.sqrt(sum((v - m) ** 2 for v in leads) / len(leads))

        assert jitter("smoothed") < jitter("linear") * 0.6

    def test_adaptive_cuts_the_lead_after_a_direction_change_then_recovers(self):
        def zig(t):
            return (500 + 600 * t, 300) if t < 0.5 else (500 + 300 - 600 * (t - 0.5), 300)

        sm, ad = make("smoothed", smoothing=0.3), make("adaptive", smoothing=0.3)
        worst = {}
        for name, tr in (("sm", sm), ("ad", ad)):
            rows = run(tr, zig, 120, 1.3)
            worst[name] = max(dist(r[1], r[2]) for r in rows if 0.5 <= r[0] <= 0.7)
            assert mean_err(rows, 1.0) < 6.0                       # both settle again
        assert worst["ad"] < worst["sm"]

    def test_every_mode_works_on_the_same_motion(self):
        for mode in MODES[1:]:
            tr = make(mode)
            rows = run(tr, lambda t: (200 + 500 * t, 300), 60, 1.5)
            assert mean_err(rows, 0.6) < mean_err(rows, 0.6, key=3) / 5


class TestMotionRegimes:
    @pytest.mark.parametrize("mode", MODES[1:])
    def test_stationary_box_gets_no_meaningful_lead_from_detector_jitter(self, mode):
        rng = random.Random(1)
        tr = make(mode)
        worst = 0.0
        for i in range(600):
            tr.update(T0 + i / 60, box(500 + rng.uniform(-1, 1), 300 + rng.uniform(-1, 1)), 0.9)
            p = tr.predict()
            if i > 30:
                worst = max(worst, p.dist)
        assert worst < 2.5

    def test_exactly_still_box_is_invalid(self):
        tr = make("adaptive")
        for i in range(60):
            tr.update(T0 + i / 60, box(500, 300), 0.9)
        p = tr.predict()
        assert not p.valid and p.dist == 0.0

    def test_slow_target_gets_a_small_proportional_lead(self):
        tr = make("linear")
        run(tr, lambda t: (200 + 60 * t, 300), 60, 2.0)           # 60 px/s, box height 80 -> dead-zone 20 px/s
        p = tr.predict()
        assert p.valid and 3.0 < p.dist < 5.0                      # (60-20) px/s * 0.1 s

    @pytest.mark.parametrize("fps", [30, 60, 144, 240])
    def test_fast_target_is_predicted_accurately_at_any_frame_rate(self, fps):
        tr = make("smoothed", smoothing=0.0, accel_comp=0.0)
        rows = run(tr, lambda t: (200 + 900 * t, 400 - 200 * t), fps, 1.5)
        pe, base = mean_err(rows, 0.6), mean_err(rows, 0.6, key=3)
        assert base > 80 and pe < 6.0 and pe < base / 12

    def test_prediction_error_is_lower_than_holding_the_last_box(self):
        for mode in MODES[1:]:
            rows = run(make(mode), lambda t: (200 + 400 * t, 300 + 100 * math.sin(3 * t)), 60, 2.0)
            assert mean_err(rows, 0.6) < 0.5 * mean_err(rows, 0.6, key=3)


class TestDirectionChange:
    @staticmethod
    def zig(t):
        return (500 + 600 * t, 300) if t < 0.5 else (500 + 300 - 600 * (t - 0.5), 300)

    def test_recovers_and_never_runs_away(self):
        tr = make("adaptive", smoothing=0.2)
        rows = run(tr, self.zig, 120, 1.5)
        assert mean_err(rows, 0.9) < 3.0
        assert max(dist(r[1], r[3]) for r in rows) < 80.0           # lead stays near horizon * speed

    def test_deceleration_never_reverses_the_lead(self):
        v0, decel = 900.0, 3000.0
        t_stop = v0 / decel
        pos = lambda t: (100 + v0 * min(t, t_stop) - 0.5 * decel * min(t, t_stop) ** 2, 200)
        tr = make("velocity_accel", smoothing=0.2, horizon_s=0.4, history_len=6)
        rows = run(tr, pos, 120, 1.0, horizon=0.4)
        stop_x = pos(t_stop)[0]
        for t, pr, _, now in rows:
            if t < t_stop:
                assert pr[0] >= now[0] - 1e-6
            else:
                assert abs(pr[0] - stop_x) < 80.0
        assert rows[-1][1][0] == pytest.approx(stop_x, abs=1.0)

    def test_axis_lead_decelerates_to_standstill_and_caps_acceleration(self):
        assert _axis_lead(100.0, -1000.0, 0.5) == pytest.approx(5.0)
        assert _axis_lead(-100.0, 1000.0, 0.5) == pytest.approx(-5.0)
        assert _axis_lead(0.0, 500.0, 0.1) == pytest.approx(0.0)    # no motion from near-rest
        assert _axis_lead(100.0, 500.0, 0.1) == pytest.approx(100 * 0.1 + 0.5 * 500 * 0.01)


class TestDetectionLoss:
    def _moving(self, n=30, v=600.0):
        tr = make("smoothed")
        for i in range(n):
            tr.update(T0 + i / 60, box(100 + v * i / 60, 200), 0.95)
        return tr

    def test_coasting_drift_is_bounded_and_expires(self):
        tr = self._moving()
        last = tr.last_t
        hold = tr.predict(0.0)
        hold_conf, hold_c = hold.confidence, (hold.cx, hold.cy)
        far = tr.predict(0.0, now=last + 0.25)                     # coasting (same reused object as `hold`)
        assert dist((far.pcx, far.pcy), hold_c) <= 600 * 0.15 + 1e-6
        assert far.confidence < hold_conf
        gone = tr.predict(0.0, now=last + 0.5)
        assert not gone.valid and (gone.pcx, gone.pcy) == (gone.cx, gone.cy)

    def test_short_gap_continues_the_track_without_stale_acceleration(self):
        tr = self._moving()
        t = tr.last_t + 0.2
        tr.update(t, box(100 + 600 * (29 / 60 + 0.2), 200), 0.95)
        assert tr.ax == 0.0 and tr.samples == 1
        assert tr.t_start < t                                       # not restarted

    def test_long_gap_restarts_clean(self):
        tr = self._moving()
        tr.update(tr.last_t + 1.0, box(900, 500), 0.95)
        assert (tr.vx, tr.vy, tr.ax, tr.ay) == (0.0, 0.0, 0.0, 0.0) and tr.samples == 1
        assert not tr.predict().valid

    def test_reacquisition_after_loss_is_not_contaminated(self):
        tr = self._moving()
        tr.reset()
        tr.update(T0 + 50, box(800, 200), 0.95)
        p = tr.predict()
        assert (p.pcx, p.pcy) == (800.0, 200.0) and not p.valid


class TestMultipleObjects:
    def test_tracks_are_independent(self):
        bank = AnchorBank(make("smoothed").params)
        for i in range(40):
            t = T0 + i / 60
            bank.update(1, t, box(100 + 500 * i / 60, 100), 0.9)
            bank.update(2, t, box(900 - 300 * i / 60, 600), 0.9)
        a, b = bank.get(1), bank.get(2)
        assert a.vx == pytest.approx(500, rel=0.05) and b.vx == pytest.approx(-300, rel=0.05)
        before = (a.vx, a.samples)
        b.reset()
        assert (a.vx, a.samples) == before and len(bank) == 2

    def test_new_id_starts_with_empty_history(self):
        bank = AnchorBank(make("smoothed").params)
        for i in range(30):
            bank.update(1, T0 + i / 60, box(100 + 600 * i / 60, 100), 0.9)
        fresh = bank.update(2, T0 + 30 / 60, box(100 + 600 * 30 / 60, 100), 0.9)
        assert fresh.samples == 1 and fresh.vx == 0.0 and not fresh.predict().valid

    def test_reassigned_id_jumping_across_the_screen_restarts_the_track(self):
        bank = AnchorBank(make("smoothed", max_speed_px_s=1200.0).params)
        for i in range(30):
            bank.update(7, T0 + i / 60, box(100 + 300 * i / 60, 100), 0.9)
        tr = bank.update(7, T0 + 30 / 60, box(1500, 900), 0.9)      # a different object got the same id
        assert tr.samples == 1 and tr.vx == 0.0

    def test_prune_drops_only_stale_tracks(self):
        bank = AnchorBank()
        bank.update(1, T0, box(100, 100))
        bank.update(2, T0 + 5, box(100, 100))
        assert bank.prune(T0 + 5.5, max_age_s=2.0) == 1 and bank.get(1) is None and bank.get(2) is not None


class TestVariableFps:
    def test_dropped_frames_and_jittery_timestamps(self):
        rng = random.Random(7)
        tr = make("smoothed", smoothing=0.1, accel_comp=0.0)
        rows = run(tr, lambda t: (200 + 500 * t, 400), 90, 2.0, skip=lambda i: rng.random() < 0.25)
        assert mean_err(rows, 0.8) < 5.0 and mean_err(rows, 0.8) < mean_err(rows, 0.8, key=3) / 4

    def test_duplicate_and_out_of_order_timestamps_are_ignored(self):
        tr = make("linear")
        tr.update(T0, box(100, 100))
        tr.update(T0 + 0.016, box(110, 100))
        snap = (tr.cx, tr.vx, tr.samples, tr.last_t)
        for bad in (T0 + 0.016, T0 + 0.01, float("nan")):
            assert tr.update(bad, box(150, 100)) is False
        assert tr.update(T0 + 0.03, (float("inf"), 0, 1, 1)) is False
        assert (tr.cx, tr.vx, tr.samples, tr.last_t) == snap


class TestConfidenceAndGuards:
    def test_low_detection_confidence_suppresses_the_lead(self):
        hi, lo = make("smoothed", conf_threshold=0.6), make("smoothed", conf_threshold=0.6)
        for i in range(60):
            t = T0 + i / 60
            hi.update(t, box(200 + 600 * i / 60, 400), 0.95)
            lo.update(t, box(200 + 600 * i / 60, 400), 0.05)
        assert hi.predict().valid and hi.predict().dist > 20
        assert not lo.predict().valid and lo.predict().dist == 0.0 and lo.confidence < hi.confidence

    def test_lead_fades_in_instead_of_jumping(self):
        tr = make("linear", conf_threshold=0.5)
        leads = []
        for i in range(40):
            tr.update(T0 + i / 60, box(200 + 600 * i / 60, 400), 0.95)
            leads.append(tr.predict().dist)
        steps = [b - a for a, b in zip(leads, leads[1:])]
        assert max(leads) == pytest.approx(60.0, abs=1.5) and max(steps) < 0.6 * max(leads)

    def test_max_distance_cap_and_strength(self):
        def lead(**kw):
            tr = make("linear", horizon_s=0.5, **kw)
            run(tr, lambda t: (100 + 1000 * t, 200), 60, 1.0, horizon=0.5)
            return tr.predict().dist
        assert lead(max_distance_px=25.0) == pytest.approx(25.0)
        free = lead(max_distance_px=0.0)
        assert free > 400 and lead(strength=0.5) == pytest.approx(free * 0.5, rel=0.02)

    def test_predicted_box_is_clamped_to_the_image(self):
        tr = make("linear", horizon_s=0.5)
        tr.set_bounds(800, 600)
        run(tr, lambda t: (100 + 600 * t, 300), 60, 1.0, horizon=0.5)
        p = tr.predict()
        assert p.px2 <= 800.0 + 1e-9 and p.px1 >= 0.0 and p.py1 >= 0.0 and p.py2 <= 600.0
        assert p.dist == pytest.approx(math.hypot(p.dx, p.dy))

    def test_predicted_size_change_is_limited(self):
        tr = make("linear", horizon_s=2.0)
        for i in range(30):
            t = i / 60
            tr.update(T0 + t, box(300, 300, w=40 + 400 * t, h=80), 0.95)
        p = tr.predict(2.0)
        assert (p.px2 - p.px1) <= 1.5 * (p.x2 - p.x1) + 1e-6

    def test_impossible_speed_is_treated_as_a_new_object(self):
        tr = make("smoothed", max_speed_px_s=1200.0)
        for i in range(20):
            tr.update(T0 + i / 60, box(100 + 300 * i / 60, 200), 0.95)
        tr.update(T0 + 20 / 60, box(1500, 900), 0.95)
        assert tr.samples == 1 and tr.vx == 0.0

    def test_objects_entering_and_leaving_the_image(self):
        tr = make("smoothed", smoothing=0.1, accel_comp=0.0)
        tr.set_bounds(1000, 600)
        rows = run(tr, lambda t: (-100 + 700 * t, 300), 60, 2.0)    # enters from the left, leaves on the right
        for _, pr, _, _ in rows:
            assert math.isfinite(pr[0])
        p = tr.predict()
        assert p.px2 <= 1000.0 + 1e-9


class TestFovAdjuster:
    @staticmethod
    def view(**kw):
        v = AnchorView()
        v.valid = True
        v.confidence = 1.0
        v.t = T0
        v.edge_pred = 0.9
        v.speed_now = 200.0
        v.speed_pred = 400.0
        v.radial_speed = 0.0
        v.turn_pred = 0.0
        for k, val in kw.items():
            setattr(v, k, val)
        return v

    def settle(self, adj, view, edge_now=0.4, speed=200.0, turn=0.0, infl=(1.0, 1.0), frames=120, fps=60):
        t = T0
        for _ in range(frames):
            t += 1 / fps
            view.t = t
            adj.update(view, edge_now, speed, turn, 200.0, infl[0], infl[1], t)
        return adj

    def test_invalid_view_or_zero_influence_gives_exactly_zero(self):
        adj = FovAdjuster()
        bad = self.view(valid=False)
        assert (self.settle(adj, bad).edge_adj, adj.speed_adj, adj.turn_adj) == (0.0, 0.0, 0.0)
        adj = FovAdjuster()
        self.settle(adj, self.view(), infl=(0.0, 0.0))
        assert (adj.edge_adj, adj.speed_adj, adj.turn_adj) == (0.0, 0.0, 0.0)

    def test_approaching_the_boundary_expands_early_but_is_capped(self):
        adj = self.settle(FovAdjuster(), self.view(edge_pred=1.4), edge_now=0.4)
        assert 0.0 < adj.edge_adj <= 0.25 + 1e-9

    def test_prediction_well_inside_the_fov_never_expands_it(self):
        adj = self.settle(FovAdjuster(), self.view(edge_pred=0.45), edge_now=0.3)
        assert adj.edge_adj == 0.0

    def test_moving_away_never_expands_and_never_goes_below_zero_edge(self):
        adj = self.settle(FovAdjuster(), self.view(edge_pred=0.1), edge_now=0.8)
        assert -0.4 - 1e-9 <= adj.edge_adj < 0.0
        adj = self.settle(FovAdjuster(), self.view(edge_pred=0.0), edge_now=0.02)
        assert adj.edge_adj >= -0.02 - 1e-9

    def test_low_confidence_weakens_the_correction(self):
        full = self.settle(FovAdjuster(), self.view(edge_pred=1.2, confidence=1.0), edge_now=0.5).edge_adj
        weak = self.settle(FovAdjuster(), self.view(edge_pred=1.2, confidence=0.2), edge_now=0.5).edge_adj
        assert 0.0 < weak < full

    def test_dynamic_terms_follow_predicted_speed_turn_and_outward_motion(self):
        adj = self.settle(FovAdjuster(), self.view(speed_pred=600.0, radial_speed=100.0, turn_pred=0.4), turn=0.1)
        assert adj.speed_adj > 100.0 and 0.0 < adj.turn_adj <= 0.4
        slow = self.settle(FovAdjuster(), self.view(speed_now=400.0, speed_pred=200.0), speed=400.0)
        assert slow.speed_adj < 0.0 and slow.speed_adj >= -200.0 * 0.5 - 1e-9   # slowing counts for half

    def test_a_flickering_prediction_does_not_make_the_correction_oscillate(self):
        adj = FovAdjuster()
        t = T0
        signs = []
        for i in range(240):
            t += 1 / 60
            v = self.view(edge_pred=1.2 if i % 2 == 0 else 0.1)
            v.t = t
            adj.update(v, 0.5, 200.0, 0.0, 200.0, 1.0, 1.0, t)
            if i > 60:
                signs.append(adj.edge_adj)
        assert max(signs) - min(signs) < 0.08

    def test_stale_view_decays_back_to_zero(self):
        adj = self.settle(FovAdjuster(), self.view(edge_pred=1.4), edge_now=0.4)
        assert adj.edge_adj > 0
        stale = self.view(edge_pred=1.4)
        t = T0 + 10
        for k in range(120):
            t += 1 / 60
            adj.update(stale, 0.4, 200.0, 0.0, 200.0, 1.0, 1.0, t)     # view.t never refreshed
        assert adj.edge_adj == 0.0

    def test_view_fill_derives_edge_radial_speed_and_turn_from_a_prediction(self):
        from core.target_tracking import edge_ratio
        tr = make("velocity_accel")
        run(tr, lambda t: (500 + 400 * t, 300), 60, 1.0)
        pred = tr.predict()
        v = AnchorView()
        v.fill(pred, 0.1, 400.0, 300.0, 400.0, 400.0, False, T0 + 1)
        assert v.valid
        assert v.edge_pred == pytest.approx(edge_ratio(pred.predicted_box(), 400.0, 300.0, 400.0, 400.0, False))
        assert v.radial_speed == pytest.approx(pred.vx, rel=1e-6)       # moving straight away from the crosshair
        assert v.turn_pred < 0.01
        v.fill(AnchorTrack().predict(), 0.1, 0, 0, 100, 100, False, T0)  # empty track -> invalid
        assert not v.valid
