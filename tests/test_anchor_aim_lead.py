"""Tests for the Anchor aim-lead Prediction Method (``core.anchor_prediction.predict_aim_point``
and ``aim_uses_anchor``) — pure Python, runs anywhere.

The helper feeds the locked box to one ``AnchorTrack`` and returns the aim point led by the
track's predicted box one aim-lead horizon ahead. ``process_aiming()``'s own wiring (method
dispatch, compensation, acquisition guard, resets, the ``disabled`` mode switch) is covered in
``tests/test_ai_aiming.py::TestAnchorAimLead``.
"""
import math
import random
from types import SimpleNamespace

import pytest

from core.anchor_prediction import AnchorParams, AnchorTrack, aim_uses_anchor, predict_aim_point

T0 = 1000.0
FPS = 60.0
H = 0.10                                  # aim-lead horizon used below (s)
SPEED = 600.0                             # px/s of the constant-velocity target


def make(mode="adaptive", **kw):
    p = AnchorParams()
    p.mode = mode
    p.max_speed_px_s = 5000.0
    p.history_len = 6
    for k, v in kw.items():
        setattr(p, k, v)
    return AnchorTrack(p)


def box(cx, cy, w=40.0, h=80.0):
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def feed(track, pos, n, *, fps=FPS, horizon=H, conf=0.9, aim_dy=-20.0, skip=()):
    """Feed ``pos(t) -> (cx, cy)`` for n frames; return [(frame, led_x, led_y, aim_x, aim_y)]."""
    out = []
    for i in range(n):
        if i in skip:
            continue
        t = i / fps
        cx, cy = pos(t)
        ax, ay = cx, cy + aim_dy
        lx, ly = predict_aim_point(track, T0 + t, box(cx, cy), ax, ay, horizon, conf)
        out.append((i, lx, ly, ax, ay))
    return out


def lead_x(row):
    return row[1] - row[3]


def line(t):
    return 400.0 + SPEED * t, 500.0


class TestConstantVelocity:
    @pytest.mark.parametrize("mode", ["linear", "velocity_accel", "smoothed", "adaptive"])
    def test_lead_is_velocity_times_horizon(self, mode):
        rows = feed(make(mode), line, 40)
        assert lead_x(rows[-1]) == pytest.approx(SPEED * H, rel=0.05)
        assert rows[-1][2] - rows[-1][4] == pytest.approx(0.0, abs=1e-6)       # no Y motion -> no Y lead

    @pytest.mark.parametrize("horizon", [0.0, 0.02, 0.05, 0.20])
    def test_lead_scales_with_the_horizon(self, horizon):
        rows = feed(make("linear"), line, 40, horizon=horizon)
        assert lead_x(rows[-1]) == pytest.approx(SPEED * horizon, rel=0.05, abs=1e-6)

    def test_first_frames_have_no_lead(self):
        rows = feed(make(), line, 3)
        assert all(lead_x(r) == 0.0 for r in rows)

    def test_diagonal_motion_leads_both_axes(self):
        rows = feed(make("linear"), lambda t: (400 + 300 * t, 300 + 450 * t), 40)
        r = rows[-1]
        assert r[1] - r[3] == pytest.approx(30.0, rel=0.05)
        assert r[2] - r[4] == pytest.approx(45.0, rel=0.05)

    def test_prediction_beats_no_prediction_for_an_accelerating_target(self):
        def accel(t):
            return 400.0 + 0.5 * 1500.0 * t * t, 500.0
        for mode in ("linear", "velocity_accel", "adaptive"):
            errs_pred, errs_none = [], []
            tr = make(mode, accel_comp=1.0)
            for i in range(90):
                t = i / FPS
                cx, cy = accel(t)
                led = predict_aim_point(tr, T0 + t, box(cx, cy), cx, cy, H, 0.95)
                if i >= 40:
                    true_x = accel(t + H)[0]
                    errs_pred.append(abs(led[0] - true_x))
                    errs_none.append(abs(cx - true_x))
            assert sum(errs_pred) < 0.3 * sum(errs_none), mode


class TestStationaryAndNoise:
    def test_stationary_target_gets_no_lead(self):
        rows = feed(make(), lambda t: (500.0, 500.0), 40)
        assert all(r[1] == r[3] and r[2] == r[4] for r in rows)

    def test_detector_jitter_stays_under_a_pixel(self):
        rows = feed(make(), lambda t: (500 + 0.8 * math.sin(t * 37), 500 + 0.8 * math.cos(t * 29)), 90)
        assert max(abs(lead_x(r)) for r in rows) < 1.0
        assert max(abs(r[2] - r[4]) for r in rows) < 1.0

    def test_low_detector_confidence_never_exceeds_the_full_lead(self):
        for conf in (0.1, 0.5, 1.0):
            rows = feed(make(), line, 40, conf=conf)
            assert 0.0 <= lead_x(rows[-1]) <= SPEED * H * 1.01


class TestDirectionChange:
    @staticmethod
    def reversing(t):
        return (400.0 + SPEED * t if t < 0.5 else 400.0 + SPEED * 0.5 - SPEED * (t - 0.5)), 500.0

    def test_lead_never_overshoots_the_true_lead_and_flips_within_a_few_frames(self):
        rows = feed(make(), self.reversing, 70)
        assert max(abs(lead_x(r)) for r in rows) <= SPEED * H * 1.01
        after = [lead_x(r) for r in rows if r[0] >= 30]                      # reversal at frame 30
        assert after[0] > 0.0                                                # still the old direction at the turn
        assert min(after[:12]) <= 0.0                                        # gone / flipped inside 12 frames
        assert after[-1] < 0.0                                               # and leading the new direction


class TestGapsAndReentry:
    def test_a_couple_of_missed_frames_keep_the_lead(self):
        rows = feed(make("linear"), line, 40, skip=(20, 21))
        after = [lead_x(r) for r in rows if r[0] in (22, 23, 24)]
        assert all(v == pytest.approx(SPEED * H, rel=0.05) for v in after)

    def test_a_long_gap_restarts_the_track(self):
        tr = make("linear")
        feed(tr, line, 20)
        t = 20 / FPS + 0.6                                                    # target gone for 0.6 s
        r1 = predict_aim_point(tr, T0 + t, box(900.0, 500.0), 900.0, 480.0, H, 0.9)
        r2 = predict_aim_point(tr, T0 + t + 1 / FPS, box(910.0, 500.0), 910.0, 480.0, H, 0.9)
        assert r1 == (900.0, 480.0)                                           # no lead from the stale history
        assert r2 == (910.0, 480.0)                                           # one sample again -> still none

    def test_a_teleport_is_not_read_as_velocity(self):
        tr = make("linear", max_speed_px_s=1200.0)
        feed(tr, lambda t: (400.0 + 60.0 * t, 500.0), 20)
        led = predict_aim_point(tr, T0 + 20 / FPS, box(800.0, 500.0), 800.0, 480.0, H, 0.9)
        assert led == (800.0, 480.0)

    def test_variable_frame_intervals(self):
        rng = random.Random(3)
        tr = make()
        t = 0.0
        last = None
        for _ in range(80):
            t += rng.uniform(0.010, 0.025)                                    # 40-100 FPS jitter
            x = 400.0 + SPEED * t
            led = predict_aim_point(tr, T0 + t, box(x, 500.0), x, 500.0, H, 0.9)
            last = led[0] - x
        assert last == pytest.approx(SPEED * H, rel=0.08)

    def test_out_of_order_and_non_finite_samples_are_ignored(self):
        tr = make("linear")
        feed(tr, line, 10)
        n = tr.samples
        assert predict_aim_point(tr, T0 + 0.01, box(1.0, 1.0), 7.0, 5.0, H) == (7.0, 5.0)          # old timestamp
        assert predict_aim_point(tr, T0 + 0.2, (float("nan"), 0.0, 1.0, 1.0), 7.0, 5.0, H) == (7.0, 5.0)
        ax, ay = predict_aim_point(tr, T0 + 0.2, box(1.0, 1.0), float("nan"), 5.0, H)
        assert math.isnan(ax) and ay == 5.0
        assert tr.samples == n


class TestAimPointInsideTheBox:
    def test_the_aim_point_keeps_its_relative_place_when_the_box_grows(self):
        tr = make("linear")
        r = None
        for i in range(30):
            t = i / FPS
            s = 1.0 + 0.5 * t                                                  # box grows 50 %/s
            w, h = 40.0 * s, 80.0 * s
            cx = 400.0 + 300.0 * t
            ax, ay = cx, 500.0 - 0.25 * h                                      # a quarter of the height above centre
            r = predict_aim_point(tr, T0 + t, box(cx, 500.0, w, h), ax, ay, H, 0.9)
        p = tr.pred
        assert p.valid
        rel_now = (ay - p.y1) / (p.y2 - p.y1)
        rel_led = (r[1] - p.py1) / (p.py2 - p.py1)
        assert rel_led == pytest.approx(rel_now, abs=1e-6)
        assert p.px1 <= r[0] <= p.px2 and p.py1 <= r[1] <= p.py2

    def test_translation_invariance(self):
        """A constant offset applied to box and aim point (what own-mouse / camera drift
        compensation does) changes nothing about the lead."""
        a = feed(make(), line, 40)
        b = feed(make(), lambda t: (line(t)[0] + 321.0, line(t)[1] - 77.0), 40)
        assert lead_x(a[-1]) == pytest.approx(lead_x(b[-1]), abs=1e-6)


class TestSafeguards:
    def test_max_distance_caps_the_lead(self):
        rows = feed(make("linear", max_distance_px=10.0), line, 40)
        assert lead_x(rows[-1]) == pytest.approx(10.0, abs=1e-6)

    def test_tracks_are_independent(self):
        a, b = make("linear"), make("linear")
        ra = feed(a, line, 40)
        rb = feed(b, lambda t: (900.0 - 300.0 * t, 200.0), 40)
        assert lead_x(ra[-1]) == pytest.approx(SPEED * H, rel=0.05)
        assert lead_x(rb[-1]) == pytest.approx(-300.0 * H, rel=0.05)

    def test_history_stays_bounded(self):
        tr = make()
        feed(tr, line, 5000)
        assert tr.samples <= 12                                              # the ring's capacity, however long it runs


class TestAimUsesAnchor:
    @staticmethod
    def cfg(**kw):
        base = dict(prediction_enabled=True, prediction_method="anchor", anchor_mode="adaptive")
        base.update(kw)
        return SimpleNamespace(**base)

    def test_truth_table(self):
        assert aim_uses_anchor(self.cfg())
        assert not aim_uses_anchor(self.cfg(prediction_enabled=False))
        assert not aim_uses_anchor(self.cfg(prediction_method="velocity"))
        assert not aim_uses_anchor(self.cfg(anchor_mode="disabled"))
        for mode in ("linear", "velocity_accel", "smoothed", "adaptive"):
            assert aim_uses_anchor(self.cfg(anchor_mode=mode))

    def test_independent_of_the_fov_layer_switch(self):
        assert aim_uses_anchor(self.cfg(anchor_enabled=False))
        assert aim_uses_anchor(self.cfg(anchor_enabled=True))

    def test_missing_fields_default_to_off(self):
        assert not aim_uses_anchor(SimpleNamespace())
