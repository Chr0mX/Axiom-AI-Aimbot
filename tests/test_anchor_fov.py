"""Anchor Prediction x Adaptive/Dynamic FOV, lock association and backward compatibility.

``GOLDEN`` holds digests of the FOV sizes/scales produced by the ORIGINAL code
(before Anchor Prediction existed) on scripted scenarios. With Anchor
Prediction off — attributes absent, disabled, or a stock ``Config`` — the
current code must reproduce them exactly.
"""
import hashlib
import math
from types import SimpleNamespace

import pytest

from core.ai_loop_state import LoopState
from core.ai_loop_utils import anchor_on_match, anchor_ref_box, anchor_reset, compute_effective_fov
from core.target_tracking import TargetTracker, edge_ratio, find_continuation

T0 = 1000.0
CX, CY = 500.0, 500.0

GOLDEN = {
    "fixed/False/approach": "b2f99e7ecd2169d5", "fixed/False/steady": "b2f99e7ecd2169d5",
    "fixed/False/zigzag": "b2f99e7ecd2169d5",
    "adaptive/False/approach": "d6bd1526397c790a", "adaptive/False/steady": "b2f99e7ecd2169d5",
    "adaptive/False/zigzag": "b295b8e3721a010d",
    "dynamic/False/approach": "7414325b33e88edd", "dynamic/False/steady": "b2f99e7ecd2169d5",
    "dynamic/False/zigzag": "8e240274fc88219e",
    "dynamic/True/approach": "82ddbfff4dcf0387", "dynamic/True/steady": "b2f99e7ecd2169d5",
    "dynamic/True/zigzag": "3fe50ea3bc751187",
}
SCENARIOS = [(m, sg, k) for m, sg in (("fixed", False), ("adaptive", False), ("dynamic", False), ("dynamic", True))
             for k in ("approach", "steady", "zigzag")]

ANCHOR_ON = dict(
    anchor_enabled=True, anchor_mode="adaptive", anchor_horizon_ms=150.0, anchor_strength=1.0,
    anchor_velocity_smoothing=0.3, anchor_accel_comp=1.0, anchor_confidence_threshold=0.2,
    anchor_history_len=6, anchor_max_distance_pct=100.0,
    anchor_fov_influence_adaptive=1.0, anchor_fov_influence_dynamic=1.0,
    prediction_max_velocity=5000.0,
)


def cfg(mode, safeguards=False, **extra):
    c = SimpleNamespace(
        fov_size=300, fov_height=300, fov_reduce_on_target_enabled=True, fov_min_size_pct=40.0,
        fov_min_size_duration=0.5, fov_recovery_duration_ms=800, fov_expand_delay_ms=300,
        fov_shrink_mode=mode, fov_dynamic_safeguards=safeguards, min_confidence=0.5,
        tracking_confidence=0.0, fov_circle_filter_enabled=False,
        screenshot_method="mss", screen_width=1000, screen_height=1000)
    for k, v in extra.items():
        setattr(c, k, v)
    return c


def path(kind, n, fps):
    pts = []
    for i in range(n):
        t = i / fps
        if kind == "approach":      # accelerates outward toward the FOV edge, then turns back
            x = CX + 10 + 0.5 * 250 * t * t if t < 0.8 else CX + 10 + 0.5 * 250 * 0.64 - 200 * (t - 0.8)
            y = CY + 20 * math.sin(2 * t)
        elif kind == "steady":
            x, y = CX + 15 * math.sin(t), CY + 10 * math.cos(1.3 * t)
        else:
            x, y = CX + 90 * math.sin(4 * t), CY + 60 * math.sin(7 * t)
        pts.append((x, y))
    return pts


def drive(c, kind, fps=60, n=240, pts=None, with_anchor=True, skip=None):
    st = LoopState()
    rows = []
    for i, (x, y) in enumerate(pts if pts is not None else path(kind, n, fps)):
        t = T0 + i / fps
        w, h = compute_effective_fov(c, st, t)
        if skip is not None and skip(i):
            rows.append((w, h, st.fov_scale))
            continue
        box = [x - 20, y - 40, x + 20, y + 40]
        st.locked_box = box
        c.fov_effective_size, c.fov_effective_height = w, h
        er = edge_ratio(box, CX, CY, w, h, False)
        st.tracker.on_match(t, box, 0.9, edge_ratio=er, same_target=(i > 0))
        if with_anchor:
            anchor_on_match(c, st, box, 0.9, t, CX, CY, w, h)
        rows.append((w, h, st.fov_scale))
    return rows, st


def digest(rows):
    return hashlib.sha256(",".join(f"{a}/{b}/{c:.9f}" for a, b, c in rows).encode()).hexdigest()[:16]


def scales(rows):
    return [r[2] for r in rows]


class TestDisabledIsIdenticalToTheOriginal:
    @pytest.mark.parametrize("mode,sg,kind", SCENARIOS)
    def test_anchor_attributes_absent(self, mode, sg, kind):
        rows, _ = drive(cfg(mode, sg), kind)
        assert digest(rows) == GOLDEN[f"{mode}/{sg}/{kind}"]

    @pytest.mark.parametrize("mode,sg,kind", SCENARIOS)
    def test_anchor_disabled_with_every_setting_present(self, mode, sg, kind):
        extra = dict(ANCHOR_ON, anchor_enabled=False)
        rows, st = drive(cfg(mode, sg, **extra), kind)
        assert digest(rows) == GOLDEN[f"{mode}/{sg}/{kind}"]
        assert len(st.anchor) == 0 and not st.anchor_view.valid     # nothing was even tracked

    @pytest.mark.parametrize("mode,sg,kind", SCENARIOS)
    def test_mode_disabled_is_off_even_when_enabled(self, mode, sg, kind):
        extra = dict(ANCHOR_ON, anchor_mode="disabled")
        rows, st = drive(cfg(mode, sg, **extra), kind)
        assert digest(rows) == GOLDEN[f"{mode}/{sg}/{kind}"] and len(st.anchor) == 0

    @pytest.mark.parametrize("mode,sg,kind", SCENARIOS[3:6] + SCENARIOS[9:])
    def test_zero_influence_matches_the_original(self, mode, sg, kind):
        extra = dict(ANCHOR_ON, anchor_fov_influence_adaptive=0.0, anchor_fov_influence_dynamic=0.0)
        rows, _ = drive(cfg(mode, sg, **extra), kind)
        assert digest(rows) == GOLDEN[f"{mode}/{sg}/{kind}"]

    def test_a_stock_config_object_behaves_like_the_original(self):
        from unittest.mock import patch
        from core.config import Config
        for mode, sg, kind in (("adaptive", False, "approach"), ("dynamic", True, "zigzag")):
            with patch("core.config._get_screen_size", return_value=(1000, 1000)):
                c = Config()
            c.fov_size = c.fov_height = 300
            c.fov_reduce_on_target_enabled = True
            c.fov_min_size_pct, c.fov_min_size_duration = 40.0, 0.5
            c.fov_recovery_duration_ms, c.fov_expand_delay_ms = 800, 300
            c.fov_shrink_mode, c.fov_dynamic_safeguards = mode, sg
            c.min_confidence, c.tracking_confidence = 0.5, 0.0
            assert c.anchor_enabled is False
            rows, _ = drive(c, kind)
            assert digest(rows) == GOLDEN[f"{mode}/{sg}/{kind}"]

    def test_fixed_mode_ignores_the_prediction(self):
        on, _ = drive(cfg("fixed", False, **ANCHOR_ON), "approach")
        assert digest(on) == GOLDEN["fixed/False/approach"]


class TestAdaptiveFov:
    def test_expands_earlier_when_heading_for_the_boundary(self):
        off, _ = drive(cfg("adaptive"), "approach")
        on, _ = drive(cfg("adaptive", **ANCHOR_ON), "approach")
        so, sf = scales(on), scales(off)
        approach = range(20, 55)                                   # target moving outward
        assert all(so[i] >= sf[i] - 1e-9 for i in approach)
        # Velocity-only anticipation (this 250 px/s^2 acceleration is below the noise floor and ignored),
        # filtered over 0.25 s: a modest but clear head start.
        assert max(so[i] - sf[i] for i in approach) > 0.012

    def test_a_target_staying_near_the_centre_causes_no_extra_expansion(self):
        off, _ = drive(cfg("adaptive"), "steady")
        on, _ = drive(cfg("adaptive", **ANCHOR_ON), "steady")
        assert max(abs(a - b) for a, b in zip(scales(on), scales(off))) < 0.01

    def test_fov_stays_inside_the_configured_bounds(self):
        for mode, sg, kind in SCENARIOS:
            rows, _ = drive(cfg(mode, sg, **ANCHOR_ON), kind)
            for w, h, s in rows:
                assert 0.4 - 1e-9 <= s <= 1.0 + 1e-9 and 120 <= w <= 300 and 120 <= h <= 300

    def test_confidence_scales_the_influence(self):
        full = max(scales(drive(cfg("adaptive", **dict(ANCHOR_ON, anchor_fov_influence_adaptive=1.0)), "approach")[0]))
        c_half = dict(ANCHOR_ON, anchor_fov_influence_adaptive=0.25)
        rows_half, _ = drive(cfg("adaptive", **c_half), "approach")
        off, _ = drive(cfg("adaptive"), "approach")
        gain_full = sum(a - b for a, b in zip(scales(drive(cfg("adaptive", **ANCHOR_ON), "approach")[0]), scales(off)))
        gain_half = sum(a - b for a, b in zip(scales(rows_half), scales(off)))
        assert 0.0 < gain_half < gain_full and full <= 1.0


class TestDynamicFov:
    def test_anticipates_acceleration(self):
        off, _ = drive(cfg("dynamic"), "approach")
        on, _ = drive(cfg("dynamic", **ANCHOR_ON), "approach")
        assert scales(on)[45] > scales(off)[45] + 0.05              # opens up while the target is still speeding up
        assert all(a >= b - 1e-9 for a, b in zip(scales(on)[:60], scales(off)[:60]))

    def test_turning_back_is_anticipated_through_the_predicted_direction_change(self):
        rows, st = drive(cfg("dynamic", **ANCHOR_ON), "zigzag")
        assert all(math.isfinite(r[2]) for r in rows)
        assert st.anchor_fov.turn_adj >= 0.0

    def test_transitions_are_rate_limited_and_do_not_oscillate(self):
        bound = (1.0 - 0.4) / 0.5 / 60 + 1e-9                       # span / min(shrink, recovery) * dt

        def flips(s, th=1e-4):
            d = [b - a for a, b in zip(s, s[1:]) if abs(b - a) > th]
            return sum(1 for a, b in zip(d, d[1:]) if a * b < 0)

        def jerk(s):                                                # sum of |second difference|
            return sum(abs(s[i + 1] - 2 * s[i] + s[i - 1]) for i in range(1, len(s) - 1))

        total_on = total_off = 0.0
        for mode, sg in (("adaptive", False), ("dynamic", False), ("dynamic", True)):
            for kind in ("approach", "zigzag"):
                on, _ = drive(cfg(mode, sg, **ANCHOR_ON), kind)
                off, _ = drive(cfg(mode, sg), kind)
                s_on, s_off = scales(on), scales(off)
                assert max(abs(b - a) for a, b in zip(s_on, s_on[1:])) <= bound
                assert flips(s_on, 3e-3) <= flips(s_off, 3e-3) + 2          # reversals of real size (micro-wiggles aside)
                assert jerk(s_on) <= 1.6 * jerk(s_off) + 0.02              # measured x0.85 .. x1.57 (tiny dynamic/approach baseline)
                total_on += jerk(s_on)
                total_off += jerk(s_off)
        assert total_on <= 1.05 * total_off                                # overall: measured x0.96 -- no added oscillation

    def test_flickering_detections_do_not_add_oscillation(self):
        pts = []
        for i in range(240):                                        # target creeping outward with +-25 px detector flicker
            t = i / 60
            pts.append((CX + 30 + 120 * t + (25 if i % 2 else -25), CY + (12 if i % 3 else -12)))
        on, _ = drive(cfg("dynamic", **ANCHOR_ON), "x", pts=pts)
        off, _ = drive(cfg("dynamic"), "x", pts=pts)

        def flips(s):
            d = [b - a for a, b in zip(s, s[1:]) if abs(b - a) > 1e-4]
            return sum(1 for a, b in zip(d, d[1:]) if a * b < 0)
        assert flips(scales(on)) <= flips(scales(off)) + 6


class TestIndependenceAndCombination:
    def test_adaptive_mode_ignores_the_dynamic_influence(self):
        a = drive(cfg("adaptive", **dict(ANCHOR_ON, anchor_fov_influence_dynamic=0.0)), "approach")[0]
        b = drive(cfg("adaptive", **dict(ANCHOR_ON, anchor_fov_influence_dynamic=1.0)), "approach")[0]
        assert digest(a) == digest(b)

    def test_dynamic_without_safeguards_ignores_the_adaptive_influence(self):
        a = drive(cfg("dynamic", False, **dict(ANCHOR_ON, anchor_fov_influence_adaptive=0.0)), "approach")[0]
        b = drive(cfg("dynamic", False, **dict(ANCHOR_ON, anchor_fov_influence_adaptive=1.0)), "approach")[0]
        assert digest(a) == digest(b)

    def test_dynamic_with_safeguards_uses_both_influences(self):
        base = dict(ANCHOR_ON)
        only_dyn = drive(cfg("dynamic", True, **dict(base, anchor_fov_influence_adaptive=0.0)), "approach")[0]
        only_ada = drive(cfg("dynamic", True, **dict(base, anchor_fov_influence_dynamic=0.0)), "approach")[0]
        both = drive(cfg("dynamic", True, **base), "approach")[0]
        off = drive(cfg("dynamic", True), "approach")[0]
        digests = {digest(x) for x in (only_dyn, only_ada, both, off)}
        assert len(digests) == 4                                    # each influence changes the outcome
        assert sum(scales(both)) > sum(scales(off))

    def test_fov_reduce_off_leaves_the_full_fov(self):
        rows, _ = drive(cfg("adaptive", **dict(ANCHOR_ON, fov_reduce_on_target_enabled=False)), "approach")
        assert all(r[:2] == (300, 300) for r in rows)


class TestTrackIdentityAndLoss:
    def test_new_lock_gets_a_new_id_and_a_clean_history(self):
        c = cfg("adaptive", **ANCHOR_ON)
        st = LoopState()
        for i in range(30):
            t = T0 + i / 60
            box = [CX + 10 * i - 20, CY - 40, CX + 10 * i + 20, CY + 40]
            st.tracker.on_match(t, box, 0.9, edge_ratio=0.3, same_target=i > 0)
            anchor_on_match(c, st, box, 0.9, t, CX, CY, 300, 300)
        first = st.tracker.lock_id
        old = st.anchor.get(first)
        assert old.samples >= 6 and old.vx > 300
        t = T0 + 30 / 60
        box = [900, 100, 940, 180]                                   # a different target elsewhere
        st.tracker.on_match(t, box, 0.9, edge_ratio=0.3, same_target=False)
        anchor_on_match(c, st, box, 0.9, t, CX, CY, 300, 300)
        new = st.anchor.get(st.tracker.lock_id)
        assert st.tracker.lock_id == first + 1 and new is not old
        assert new.samples == 1 and new.vx == 0.0 and not st.anchor_view.valid

    def test_same_target_keeps_its_id(self):
        tr = TargetTracker()
        tr.on_match(T0, [0, 0, 10, 10], 0.9, same_target=False)
        first = tr.lock_id
        tr.on_match(T0 + 0.016, [1, 0, 11, 10], 0.9, same_target=True)
        assert tr.lock_id == first
        tr.release(T0 + 1)
        tr.on_match(T0 + 2, [0, 0, 10, 10], 0.9, same_target=False)
        assert tr.lock_id == first + 1                                # monotonic: ids are never reused

    def test_temporary_detection_loss_lets_the_correction_fade_out(self):
        c = cfg("dynamic", **ANCHOR_ON)
        rows, st = drive(c, "approach", skip=lambda i: 100 <= i < 150)   # locked_box held, no detections for 0.8 s
        assert st.anchor_fov.edge_adj == 0.0 or abs(st.anchor_fov.edge_adj) < 0.02
        assert all(math.isfinite(r[2]) for r in rows)

    def test_reset_clears_everything(self):
        c = cfg("adaptive", **dict(ANCHOR_ON, show_anchor_debug=True))
        c.anchor_debug = None
        _, st = drive(c, "approach")
        assert len(st.anchor) == 1 and c.anchor_debug is not None
        anchor_reset(c, st)
        assert len(st.anchor) == 0 and not st.anchor_view.valid and c.anchor_debug is None
        assert (st.anchor_fov.edge_adj, st.anchor_fov.speed_adj, st.anchor_fov.turn_adj) == (0.0, 0.0, 0.0)


class TestDebugSnapshot:
    def test_published_only_while_the_toggle_is_on(self):
        c = cfg("adaptive", **ANCHOR_ON)
        c.anchor_debug = None
        drive(c, "approach")
        assert c.anchor_debug is None
        c2 = cfg("adaptive", **dict(ANCHOR_ON, show_anchor_debug=True))
        c2.anchor_debug = None
        drive(c2, "approach", n=30)                                  # target still speeding up, away from the crosshair
        snap = c2.anchor_debug
        assert snap is not None and len(snap) == 15 and all(math.isfinite(v) for v in snap)
        x1, y1, x2, y2, px1, py1, px2, py2, cx, cy, pcx, pcy, vx, vy, conf = snap
        assert pcx >= cx and vx > 0 and 0.0 <= conf <= 1.0 and (px2 - px1) > 0
        c2.show_anchor_debug = False
        drive(c2, "approach", n=5)
        assert c2.anchor_debug is None


class TestAssociation:
    @staticmethod
    def _locked_fast_target(fps=10):
        c = cfg("adaptive", **dict(ANCHOR_ON, anchor_horizon_ms=100.0))
        st = LoopState()
        for i in range(8):                                           # 800 px/s sampled at only 10 Hz: 80 px per frame
            t = T0 + i / fps
            x = 200 + 80 * i
            box = [x - 20, 480, x + 20, 560]
            st.tracker.on_match(t, box, 0.9, edge_ratio=0.2, same_target=i > 0)
            anchor_on_match(c, st, box, 0.9, t, CX, CY, 600, 600)
            st.locked_box = box
        return c, st, T0 + 8 / fps

    def test_predicted_box_picks_the_real_target_over_an_overlapping_rival(self):
        c, st, now = self._locked_fast_target()
        true_box = [820, 480, 860, 560]                              # where the target really is now (x=840)
        rival = [770, 480, 810, 560]                                 # a slower box still overlapping the old position
        boxes = [rival, true_box]
        kw = dict(locked_box=st.locked_box, tracker=st.tracker, config=c, now=now, iou_thresh=0.1)
        base_idx, _ = find_continuation(boxes, **kw)
        ref = anchor_ref_box(c, st, now)
        assert ref is not None
        idx, ambiguous = find_continuation(boxes, **kw, ref_box=ref)
        assert base_idx == 0                                         # without prediction it re-locks the wrong box
        assert idx == 1 and not ambiguous

    def test_without_a_reference_box_nothing_changes(self):
        c, st, now = self._locked_fast_target()
        boxes = [[770, 480, 810, 560]]
        kw = dict(locked_box=st.locked_box, tracker=st.tracker, config=c, now=now, iou_thresh=0.1)
        assert find_continuation(boxes, **kw) == find_continuation(boxes, **kw, ref_box=None)

    def test_ref_box_is_none_when_off_or_not_confident(self):
        c, st, now = self._locked_fast_target()
        assert anchor_ref_box(c, st, now) is not None
        c.anchor_enabled = False
        assert anchor_ref_box(c, st, now) is None
        c.anchor_enabled = True
        assert anchor_ref_box(c, st, now + 5.0) is None              # track expired
        assert anchor_ref_box(cfg("adaptive"), LoopState(), now) is None
