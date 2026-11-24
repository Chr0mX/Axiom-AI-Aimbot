"""Auto lead / strafe response / occlusion coasting in the Anchor estimator (pure Python)."""
from __future__ import annotations

import hashlib
import math
import random

import pytest

from core import anchor_prediction as ap
from core.anchor_prediction import (
    AnchorParams, AnchorTrack, aim_track_active, aim_uses_anchor, auto_aim_point, auto_coast_point,
    make_auto_track, predict_aim_point,
)

T0 = 1000.0


def box(cx, cy, w=40.0, h=80.0):
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def strafe_x(speed=350.0, period=0.6):
    def f(t):
        u = (t % period) / period
        return 400.0 + (speed * period / 2.0) * (2 * u if u < 0.5 else 2 * (1 - u))
    return f


def feed(tr, fn, fps, seconds, noise=1.0, seed=3, skip=()):
    rng = random.Random(seed)
    for i in range(int(seconds * fps)):
        if i in skip:
            continue
        t = i / fps
        x = fn(t) + rng.gauss(0, noise)
        tr.update(T0 + t, box(x, 500.0), 0.9)


# ── defaults are untouched: golden digests recorded from the pre-change estimator ────────────────────

GOLDEN = {
    'linear': ['d97baf04c1282cb9', '408d291a675ff9fd', '194adb734b65eb2b'],
    'velocity_accel': ['9747b093e4742b82', '7c733549966f0070', 'f37c2bda04714950'],
    'smoothed': ['2ed7a0bac320f246', '1c5fc84206d1eb90', '904962071e9bfd84'],
    'adaptive': ['570f303556335927', 'ab3af9445da579e9', '176c78b1ddf40784'],
}


@pytest.mark.parametrize('mode', list(GOLDEN))
@pytest.mark.parametrize('idx,fps', [(0, 30), (1, 60), (2, 144)])
def test_defaults_are_bit_identical_to_the_previous_estimator(mode, idx, fps):
    p = AnchorParams()
    p.mode, p.history_len, p.horizon_s = mode, 6, 0.05
    assert (p.strafe, p.coast_s, p.auto) == (0.0, 0.0, False)
    tr = AnchorTrack(p)
    rng = random.Random(7)
    out = []
    for i in range(240):
        t = i / fps
        if i in (60, 61, 62):
            continue
        x = 500 + 300 * math.sin(2.1 * t) + rng.gauss(0, 1)
        y = 500 + 80 * math.sin(3.3 * t)
        tr.update(T0 + t, (x - 20, y - 40, x + 20, y + 40), 0.9)
        pr = tr.predict(0.05, T0 + t)
        out.append((round(pr.pcx, 6), round(pr.pcy, 6), pr.valid))
    assert hashlib.sha1(repr(out).encode()).hexdigest()[:16] == GOLDEN[mode][idx]


# ── strafe response (reversal detector) ──────────────────────────────────────────────────────────────

def _frames_to_flip(fps, strafe=0.6, noise=1.0, seed=3):
    """Milliseconds from the true reversal (t = 0.6 s) until the velocity estimate reaches 80 % of the new speed."""
    tr = make_auto_track(strafe_response=strafe, coast_ms=0.0)
    fn = strafe_x(350.0, 1.2)             # peak at t = 0.6 s
    rng = random.Random(seed)
    for i in range(int(1.4 * fps)):
        t = i / fps
        tr.update(T0 + t, box(fn(t) + rng.gauss(0, noise), 500.0), 0.9)
        if t > 0.6 and tr.vx < -280.0:
            return (t - 0.6) * 1000.0
    return math.inf


@pytest.mark.parametrize('fps', [30, 60, 144, 240])
def test_reversal_flips_the_estimate_quickly_at_any_rate(fps):
    assert _frames_to_flip(fps) <= 130.0


def test_reversal_detector_is_never_slower_and_helps_at_low_rates():
    def mean_flip(fps, strafe):
        return sum(_frames_to_flip(fps, strafe=strafe, seed=sd) for sd in range(6)) / 6.0
    for fps in (30, 60, 144):
        assert mean_flip(fps, 0.6) <= mean_flip(fps, 0.0) + 1.0
    assert mean_flip(30, 0.6) + 10.0 < mean_flip(30, 0.0)


@pytest.mark.parametrize('fps', [30, 60, 144])
@pytest.mark.parametrize('speed', [0.0, 40.0, 350.0, 900.0])
def test_no_false_reversals_on_steady_motion_with_noise(fps, speed):
    for seed in range(5):
        tr = make_auto_track(strafe_response=1.0, coast_ms=0.0)
        feed(tr, lambda t: 300.0 + speed * t, fps, 6.0, noise=1.0, seed=seed)
        assert tr.reversals == 0, (fps, speed, seed)


def test_reversal_counter_follows_the_real_turns():
    tr = make_auto_track(strafe_response=0.8, coast_ms=0.0)
    feed(tr, strafe_x(350.0, 0.6), 60, 3.0)         # 10 turns in 3 s (every 0.3 s)
    assert 7 <= tr.reversals <= 12


def test_strafe_zero_never_fires_and_leaves_turn_signal_alone():
    tr = make_auto_track(strafe_response=0.0, coast_ms=0.0)
    feed(tr, strafe_x(350.0, 0.6), 60, 3.0)
    assert tr.reversals == 0 and tr.reversal_t == 0.0


def test_reversal_clears_turn_signal_and_cuts_history():
    tr = make_auto_track(strafe_response=0.8, coast_ms=0.0)
    fn = strafe_x(350.0, 1.2)
    n_before = None
    for i in range(int(1.0 * 60)):
        t = i / 60
        tr.update(T0 + t, box(fn(t), 500.0), 0.9)
        if tr.reversals:
            n_before = tr.samples
            break
    assert n_before is not None and n_before <= 4
    assert tr.turn_ema == 0.0 and tr.since_reversal(tr.last_t) == 0.0


def test_strafe_response_works_with_the_plain_anchor_method_too():
    p = AnchorParams()
    p.mode, p.strafe = 'adaptive', 0.8
    tr = AnchorTrack(p)
    feed(tr, strafe_x(350.0, 0.6), 60, 3.0)
    assert tr.reversals > 0


def test_sensitivity_scales_with_the_setting():
    low, high = make_auto_track(0.05, 0.0), make_auto_track(1.0, 0.0)
    for tr in (low, high):
        feed(tr, strafe_x(120.0, 0.6), 60, 4.0, noise=2.0)
    assert high.reversals >= low.reversals


# ── occlusion coasting ───────────────────────────────────────────────────────────────────────────────

def _track_moving(vx=350.0, coast_ms=400.0, fps=60, seconds=1.0):
    tr = make_auto_track(strafe_response=0.0, coast_ms=coast_ms)
    feed(tr, lambda t: 300.0 + vx * t, fps, seconds, noise=0.5)
    return tr


def test_coast_is_off_by_default_and_when_window_is_zero():
    assert AnchorParams().coast_s == 0.0
    tr = _track_moving(coast_ms=0.0)
    assert tr.coast_offset(tr.last_t + 0.1) is None


def test_coast_travel_profile_is_continuous_monotonic_and_bounded():
    S = 0.4
    xs = [ap._coast_travel(i * 0.005, S) for i in range(0, 120)]
    assert xs[0] == 0.0
    assert all(b >= a for a, b in zip(xs, xs[1:]))
    assert ap._coast_travel(S, S) == pytest.approx(0.5 * S + 0.25 * S)      # hold half at full speed, fade the rest
    assert ap._coast_travel(10.0, S) == ap._coast_travel(S, S)               # never beyond the window
    assert ap._coast_travel(-1.0, S) == 0.0 and ap._coast_travel(0.1, 0.0) == 0.0


def test_coast_follows_the_target_then_stops_at_the_window():
    tr = _track_moving()
    last = tr.last_t
    # first 200 ms: full speed -> ~ v * dt
    dx, dy = tr.coast_offset(last + 0.1)
    assert dx == pytest.approx(350.0 * 0.1, rel=0.2) and abs(dy) < 1e-6
    # end of the window: bounded travel (0.75 * v * window at most)
    dx_end, _ = tr.coast_offset(last + 0.4)
    assert dx_end <= 350.0 * 0.4 * 0.75 * 1.05
    assert tr.coast_offset(last + 0.401) is None                              # expired: crosshair stops


def test_coast_returns_none_for_bad_inputs_and_untrustworthy_tracks():
    tr = _track_moving()
    assert tr.coast_offset(tr.last_t) is None                                 # not hidden yet
    assert tr.coast_offset(tr.last_t - 1.0) is None
    assert tr.coast_offset(float('nan')) is None
    fresh = make_auto_track(0.0, 400.0)
    assert fresh.coast_offset(T0 + 0.1) is None                               # no history
    fresh.update(T0, box(100, 100), 0.9)
    assert fresh.coast_offset(T0 + 0.1) is None                               # one sample: no velocity
    low = _track_moving()
    low.confidence = 0.0
    assert low.coast_offset(low.last_t + 0.1) is None


def test_standing_target_coasts_in_place():
    tr = make_auto_track(0.0, 400.0)
    feed(tr, lambda t: 500.0, 60, 1.0, noise=0.3)
    assert tr.coast_offset(tr.last_t + 0.2) == (0.0, 0.0)


def test_unstable_heading_shortens_the_coast():
    tr = _track_moving()
    steady = tr.coast_offset(tr.last_t + 0.2)[0]
    tr.turn_ema = 0.6
    assert tr.coast_offset(tr.last_t + 0.2)[0] < 0.5 * steady


def test_coast_distance_cap():
    tr = _track_moving()
    tr.params.max_distance_px = 30.0
    dx, dy = tr.coast_offset(tr.last_t + 0.3)
    assert math.hypot(dx, dy) <= 30.0 + 1e-9


def test_track_survives_the_gap_only_with_coasting_enabled():
    for coast_ms, survives in ((400.0, True), (0.0, False)):
        tr = make_auto_track(0.0, coast_ms)
        feed(tr, lambda t: 300.0 + 350.0 * t, 60, 1.0, noise=0.0)
        t_start = tr.t_start
        t = 1.0 + 0.35
        tr.update(T0 + t, box(300.0 + 350.0 * t, 500.0), 0.9)
        assert (tr.t_start == t_start) == survives, coast_ms


def test_track_restarts_after_a_gap_beyond_the_window():
    tr = make_auto_track(0.0, 400.0)
    feed(tr, lambda t: 300.0 + 350.0 * t, 60, 1.0, noise=0.0)
    t_start = tr.t_start
    t = 1.0 + 0.65
    tr.update(T0 + t, box(300.0 + 350.0 * t, 500.0), 0.9)
    assert tr.t_start != t_start and tr.samples == 1


def test_velocity_is_kept_through_a_coasted_gap_and_relearned_fast():
    tr = make_auto_track(0.0, 400.0)
    feed(tr, lambda t: 300.0 + 350.0 * t, 60, 1.0, noise=0.0)
    t = 1.0 + 0.3
    tr.update(T0 + t, box(300.0 + 350.0 * t, 500.0), 0.9)
    assert tr.vx == pytest.approx(350.0, rel=0.05)                            # not decayed to ~0
    tr.update(T0 + t + 1 / 60, box(300.0 + 350.0 * (t + 1 / 60), 500.0), 0.9)
    assert tr.vx == pytest.approx(350.0, rel=0.05) and tr.predict(0.03, T0 + t + 1 / 60).valid


def test_reversal_detector_does_not_fire_across_a_gap():
    tr = make_auto_track(1.0, 400.0)
    feed(tr, lambda t: 300.0 + 350.0 * t, 60, 1.0, noise=0.0)
    n = tr.reversals
    t = 1.0 + 0.3
    tr.update(T0 + t, box(300.0 + 350.0 * 1.0 - 100.0, 500.0), 0.9)          # reappears behind: no history to compare
    assert tr.reversals == n


def test_auto_coast_point_uses_the_aim_offset_inside_the_box():
    tr = make_auto_track(0.0, 400.0)
    t = 0.0
    for i in range(60):
        t = i / 60
        x = 300.0 + 350.0 * t
        auto_aim_point(tr, T0 + t, box(x, 500.0), x + 5.0, 480.0, 0.025, 0.9)   # aim point 5 px right, 20 px up
    p = auto_coast_point(tr, T0 + t + 0.1, 0.025)
    assert p is not None
    assert p[1] == pytest.approx(480.0, abs=1e-6)
    assert p[0] > tr.cx + 5.0


def test_coast_and_lead_are_never_both_applied_in_one_frame():
    tr = make_auto_track(0.5, 400.0)
    for i in range(60):
        t = i / 60
        auto_aim_point(tr, T0 + t, box(300 + 350 * t, 500), 300 + 350 * t, 500, 0.025, 0.9)
    assert auto_coast_point(tr, tr.last_t, 0.025) is None                    # a frame with a detection: no coasting


# ── auto horizon ─────────────────────────────────────────────────────────────────────────────────────

def test_auto_horizon_is_age_plus_output_delay_and_clamped():
    tr = make_auto_track(0.0, 0.0, max_ms=60.0)
    assert tr.auto_horizon(0.025, 0.016) == pytest.approx(0.025 + ap._AUTO_OUT_S)
    tr = make_auto_track(0.0, 0.0, max_ms=60.0)
    assert tr.auto_horizon(0.5, 0.016) == pytest.approx(0.06)                # capped
    tr = make_auto_track(0.0, 0.0, max_ms=60.0)
    assert tr.auto_horizon(0.0, 0.016) == pytest.approx(ap._AUTO_OUT_S)
    for bad in (float('nan'), float('inf'), -1.0):
        t2 = make_auto_track(0.0, 0.0)
        h = t2.auto_horizon(bad, 0.016)
        assert math.isfinite(h) and ap._AUTO_MIN_H_S <= h <= 0.06


def test_auto_horizon_smooths_age_jitter_and_follows_load():
    tr = make_auto_track(0.0, 0.0, max_ms=200.0)
    rng = random.Random(1)
    hs = [tr.auto_horizon(0.025 + rng.gauss(0, 0.01), 1 / 60) for _ in range(120)]
    spread = max(hs[60:]) - min(hs[60:])
    assert spread < 0.01                                                      # 10 ms sigma input -> far calmer output
    for _ in range(120):                                                      # pipeline slows to 60 ms
        h = tr.auto_horizon(0.060, 1 / 60)
    assert h == pytest.approx(0.070, abs=0.004)


def test_auto_lead_matches_a_hand_tuned_horizon_without_being_told():
    """Lead error with the measured age equals the error of a track hand-tuned to the true latency."""
    L_pipe, L_out = 0.025, 0.010
    fn = lambda t: 200.0 + 350.0 * t                                          # noqa: E731
    auto = make_auto_track(0.0, 0.0)
    p = AnchorParams()
    p.mode, p.history_len, p.accel_comp, p.horizon_s, p.conf_threshold = 'velocity_accel', 6, 0.5, L_pipe + L_out, 0.35
    manual = AnchorTrack(p)
    fixed10 = AnchorTrack(AnchorParams())
    fixed10.params.mode = 'velocity_accel'
    rng = random.Random(2)
    err = {'auto': 0.0, 'manual': 0.0, 'fixed10': 0.0}
    n = 0
    for i in range(240):
        t = i / 60
        x = fn(t) + rng.gauss(0, 1.0)
        a = auto_aim_point(auto, T0 + t, box(x, 500), x, 500, L_pipe, 0.9)[0]
        m = predict_aim_point(manual, T0 + t, box(x, 500), x, 500, L_pipe + L_out, 0.9)[0]
        f = predict_aim_point(fixed10, T0 + t, box(x, 500), x, 500, 0.010, 0.9)[0]
        if t > 0.5:
            truth = fn(t + L_pipe + L_out)
            for k, v in (('auto', a), ('manual', m), ('fixed10', f)):
                err[k] += (v - truth) ** 2
            n += 1
    rm = {k: math.sqrt(v / n) for k, v in err.items()}
    assert rm['auto'] < 0.5 * rm['fixed10']
    assert rm['auto'] <= rm['manual'] * 1.5 + 0.5


# ── gating helpers / config mapping ──────────────────────────────────────────────────────────────────

class Cfg:
    def __init__(self, **kw):
        self.prediction_enabled = True
        self.prediction_method = 'velocity'
        self.anchor_mode = 'adaptive'
        self.__dict__.update(kw)


def test_method_and_track_gating():
    assert not aim_uses_anchor(Cfg()) and not aim_track_active(Cfg())
    assert aim_uses_anchor(Cfg(prediction_method='anchor')) and aim_uses_anchor(Cfg(prediction_method='auto'))
    assert not aim_uses_anchor(Cfg(prediction_method='auto', prediction_enabled=False))
    assert aim_track_active(Cfg(occlusion_coast_enabled=True))
    assert aim_track_active(Cfg(prediction_enabled=False, occlusion_coast_enabled=True))


def test_update_from_maps_the_new_settings():
    p = AnchorParams().update_from(Cfg(prediction_method='auto', strafe_response=0.7, occlusion_coast_enabled=True,
                                       occlusion_coast_ms=300, auto_lead_max_ms=45, anchor_mode='smoothed'))
    assert p.auto and p.strafe == pytest.approx(0.7) and p.coast_s == pytest.approx(0.3)
    assert p.auto_max_s == pytest.approx(0.045)
    assert p.mode == 'velocity_accel' and p.accel_comp == 0.5                  # Auto fixes the estimator
    q = AnchorParams().update_from(Cfg(prediction_method='anchor', occlusion_coast_enabled=False, occlusion_coast_ms=300,
                                       anchor_mode='smoothed'))
    assert not q.auto and q.coast_s == 0.0 and q.mode == 'smoothed'
    bad = AnchorParams().update_from(Cfg(strafe_response=float('nan'), occlusion_coast_enabled=True,
                                         occlusion_coast_ms=float('inf'), auto_lead_max_ms='x'))
    assert math.isfinite(bad.strafe) and bad.coast_s >= 0.0 and math.isfinite(bad.coast_s)


def test_per_frame_cost_stays_small():
    import time
    tr = make_auto_track(0.6, 400.0)
    pts = [(500 + 80 * math.sin(i / 20), 500 + 40 * math.sin(i / 13)) for i in range(5000)]
    s = time.perf_counter()
    for i, (x, y) in enumerate(pts):
        auto_aim_point(tr, T0 + i / 144.0, box(x, y), x, y, 0.025, 0.9)
    per = (time.perf_counter() - s) / len(pts) * 1e6
    assert per < 150.0          # generous CI bound; measured ~15 us
