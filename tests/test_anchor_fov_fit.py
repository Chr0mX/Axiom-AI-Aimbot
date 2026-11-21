"""Anchor FOV "Fit FOV to target box while aiming" (core/anchor_fov.py).

While aim is active and a target is locked the FOV shrinks to the padded detection box (never above
the configured FOV Width/Height) and glides back when aim is released. Pure Python: the real controller
is driven with simulated time.
"""
import math
from types import SimpleNamespace

import pytest

from core import anchor_fov as af
from core.target_tracking import TargetTracker

REGION = {"left": 0, "top": 0, "width": 1000, "height": 1000}
CROSS = (500.0, 500.0)
T0 = 1000.0
DT = 1.0 / 60.0
BOX = (480.0, 440.0, 520.0, 520.0)         # 40 x 80 target


def cfg(**kw):
    c = SimpleNamespace(
        fov_shrink_mode="anchor", fov_size=300, fov_height=300, crosshairX=500, crosshairY=500,
        afov_source="detection", afov_position="center", afov_offset_x=0, afov_offset_y=0,
        afov_follow_speed=100.0, afov_smoothing=0.0, afov_max_speed=0,
        afov_predict_time_ms=0, afov_lost_behavior="hold", afov_lost_delay_ms=150,
        afov_reacquire="smooth", afov_max_radius=400, afov_dynamic_radius=False,
        fov_circle_filter_enabled=False, afov_fit_box_enabled=True, afov_fit_padding_pct=30)
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def new_state(locked=True):
    return SimpleNamespace(afov=None, tracker=TargetTracker(), locked_box=BOX if locked else None)


def run(c, st, frames, boxes=(BOX,), aiming=True, t0=T0):
    """Advance `frames` frames; return (w, h) of the last one."""
    out = None
    for i in range(frames):
        out = af.update_anchor_fov(c, st, list(boxes), CROSS[0], CROSS[1], REGION, t0 + i * DT, aiming=aiming)[2:]
    return out


def settle(c, st, boxes=(BOX,), aiming=True, t0=T0, seconds=2.0):
    return run(c, st, int(seconds / DT), boxes, aiming, t0)


class TestFitSize:
    def test_size_is_the_padded_box_shape(self):
        w, h = settle(cfg(), new_state())
        assert w == pytest.approx(40 * 1.3, abs=1.5) and h == pytest.approx(80 * 1.3, abs=1.5)

    def test_hard_floor(self):
        tiny = (498.0, 495.0, 502.0, 505.0)                       # 4 x 10 px
        st = new_state()
        st.locked_box = tiny
        w, h = settle(cfg(), st, boxes=(tiny,))
        assert w == pytest.approx(2 * af.MIN_RADIUS_PX, abs=1.5) and h == pytest.approx(2 * af.MIN_RADIUS_PX, abs=1.5)

    def test_never_larger_than_the_configured_fov(self):
        big = (300.0, 300.0, 700.0, 700.0)                        # 400 x 400 > FOV 300
        st = new_state()
        st.locked_box = big
        w, h = settle(cfg(afov_fit_padding_pct=200), st, boxes=(big,))
        assert w == pytest.approx(300, abs=1.5) and h == pytest.approx(300, abs=1.5)

    def test_rectangular_fov_caps_each_axis(self):
        c = cfg(fov_size=300, fov_height=120)                     # wide, short FOV
        w, h = settle(c, new_state())
        assert w == pytest.approx(40 * 1.3, abs=1.5) and h == pytest.approx(80 * 1.3, abs=1.5)   # 104 < 120: not capped
        w2, h2 = settle(cfg(fov_size=300, fov_height=60, afov_fit_padding_pct=200), new_state())
        assert h2 == pytest.approx(60, abs=1.5)                  # 240 wanted, capped by the configured FOV height

    def test_padding_zero_and_two_hundred_percent(self):
        w0, h0 = settle(cfg(afov_fit_padding_pct=0), new_state())
        assert w0 == pytest.approx(max(2 * af.MIN_RADIUS_PX, 40), abs=1.5) and h0 == pytest.approx(80, abs=1.5)
        w2, h2 = settle(cfg(afov_fit_padding_pct=200), new_state())
        assert w2 == pytest.approx(120, abs=1.5) and h2 == pytest.approx(240, abs=1.5)


class TestWhenItApplies:
    def test_off_by_default_is_the_normal_fov(self):
        w, h = settle(cfg(afov_fit_box_enabled=False), new_state())
        assert (w, h) == (300, 300)

    def test_not_aiming_keeps_the_full_size(self):
        assert settle(cfg(), new_state(), aiming=False) == (300, 300)

    def test_no_lock_yet_keeps_the_full_size(self):
        """Acquisition must still see every candidate."""
        assert settle(cfg(), new_state(locked=False)) == (300, 300)

    def test_manual_source_has_no_box(self):
        assert settle(cfg(afov_source="manual"), new_state()) == (300, 300)

    def test_tracker_source_uses_the_tracker_box(self):
        st = new_state()
        tr = st.tracker
        for i in range(120):                                       # keep the tracker's lock alive while the FOV settles
            tr.on_match(T0 + i * DT, BOX, 0.9, same_target=(i > 0))
        w, h = settle(cfg(afov_source="tracker"), st)
        assert w == pytest.approx(40 * 1.3, abs=2.0) and h == pytest.approx(80 * 1.3, abs=2.0)


class TestReleaseAndLoss:
    def test_release_glides_back_without_a_jump(self):
        c, st = cfg(), new_state()
        fitted = settle(c, st)
        sizes = []
        for i in range(int(1.5 / DT)):
            sizes.append(af.update_anchor_fov(c, st, [BOX], CROSS[0], CROSS[1], REGION, T0 + 3 + i * DT, aiming=False)[2])
        assert sizes[0] < 80 and sizes[-1] == 300                       # first frame barely moves, ends at the full FOV
        assert all(b >= a for a, b in zip(sizes, sizes[1:]))            # monotonic, no overshoot
        assert max(b - a for a, b in zip(sizes, sizes[1:])) < 30         # never a jump
        assert fitted[0] < 80

    def test_a_short_detection_gap_keeps_the_fit(self):
        c, st = cfg(afov_lost_delay_ms=300), new_state()
        w0, _ = settle(c, st)
        w, _ = run(c, st, 6, boxes=(), t0=T0 + 3)                       # 100 ms without detections, inside the 300 ms grace
        assert w == pytest.approx(w0, abs=1.5)

    def test_after_the_grace_the_fov_expands_again(self):
        c, st = cfg(afov_lost_delay_ms=100), new_state()
        w0, _ = settle(c, st)
        w, _ = run(c, st, int(1.5 / DT), boxes=(), t0=T0 + 3)
        assert w == 300 and w0 < 80

    def test_lost_search_widens_to_the_maximum_even_when_it_was_fitted(self):
        c, st = cfg(afov_lost_behavior="search", afov_lost_delay_ms=50), new_state()
        settle(c, st)
        w, _ = run(c, st, int(2.0 / DT), boxes=(), t0=T0 + 3)
        assert w == pytest.approx(2 * 400, abs=2)

    def test_fit_beats_dynamic_radius_while_a_box_exists(self):
        c = cfg(afov_dynamic_radius=True)
        w, h = settle(c, new_state())
        assert w == pytest.approx(52, abs=1.5)


class TestUnchangedWhenOff:
    def test_default_output_equals_the_plain_anchor_fov(self):
        """With the option off the width/height are exactly what they were before the option existed."""
        a, b = cfg(afov_fit_box_enabled=False), cfg(afov_fit_box_enabled=False)
        del b.afov_fit_box_enabled                                    # an old config object without the field at all
        sa, sb = new_state(), new_state()
        for i in range(120):
            ra = af.update_anchor_fov(a, sa, [BOX], CROSS[0], CROSS[1], REGION, T0 + i * DT, aiming=True)
            rb = af.update_anchor_fov(b, sb, [BOX], CROSS[0], CROSS[1], REGION, T0 + i * DT)
            assert ra == rb
        assert ra[2:] == (300, 300)

    @pytest.mark.parametrize("fov_h", [300, 150])
    def test_aspect_is_kept_without_fit(self, fov_h):
        w, h = settle(cfg(afov_fit_box_enabled=False, fov_height=fov_h), new_state())
        assert (w, h) == (300, fov_h)


class TestRobustness:
    @pytest.mark.parametrize("bad", [(10.0, 10.0, 10.0, 10.0), (float("nan"), 0.0, 5.0, 5.0), (5.0, 5.0, 1.0, 1.0)])
    def test_degenerate_boxes_never_crash_or_produce_nonfinite_sizes(self, bad):
        st = new_state()
        st.locked_box = bad
        w, h = settle(cfg(), st, boxes=(bad,), seconds=0.5)
        assert math.isfinite(w) and math.isfinite(h) and 1 <= w <= 800 and 1 <= h <= 800

    def test_sample_size_helper(self):
        assert af._box_size((10, 20, 50, 100)) == (40.0, 80.0)
        assert af._box_size((10, 20, 10, 100)) == (0.0, 0.0)
        assert af._box_size((float("inf"), 0, 1, 1)) == (0.0, 0.0)


class TestConfigFields:
    @staticmethod
    def new_config():
        from unittest.mock import patch
        from core import config as cfgmod
        with patch("core.config._get_screen_size", return_value=(1920, 1080)):
            return cfgmod.Config()

    def test_defaults_are_off_and_30_percent(self):
        c = self.new_config()
        assert c.afov_fit_box_enabled is False and c.afov_fit_padding_pct == 30

    def test_round_trip_through_the_grouped_file(self):
        c = self.new_config()
        c.afov_fit_box_enabled, c.afov_fit_padding_pct = True, 55
        d = c.to_dict()["aim"]["fov_anchor"]["fit_box"]
        assert d == {"enabled": True, "padding_pct": 55}
        c2 = self.new_config()
        c2.from_dict(c.to_dict())
        assert (c2.afov_fit_box_enabled, c2.afov_fit_padding_pct) == (True, 55)

    @pytest.mark.parametrize("raw,want", [(-5, 0), (999, 200), (float("nan"), 30), ("x", 30), (45.6, 45)])
    def test_padding_is_clamped(self, raw, want):
        from core import config as cfgmod
        c = self.new_config()
        c.afov_fit_padding_pct = raw
        cfgmod.validate_config(c)
        assert c.afov_fit_padding_pct == want

    def test_it_is_an_aim_preset_field(self):
        from core.config_manager import _AIM_PRESET_FIELDS
        assert {"afov_fit_box_enabled", "afov_fit_padding_pct"} <= set(_AIM_PRESET_FIELDS)
