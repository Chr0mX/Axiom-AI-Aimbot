"""Anchor FOV mode (core/anchor_fov.py): a FOV whose centre follows an anchor.

Covers the ten scenarios of the feature request: stationary / moving / rapid /
noisy anchors, prediction, temporary and permanent loss, reacquisition, mode
switching and the existing modes staying untouched.  Pure Python -- the
controller and the real TargetTracker are driven with simulated time.
"""
import math
import random
from types import SimpleNamespace

import pytest

from core import anchor_fov as af
from core.ai_loop_utils import compute_effective_fov, filter_boxes_by_fov
from core.target_tracking import TargetTracker

CROSS = (320.0, 320.0)
REGION = {"left": 0, "top": 0, "width": 640, "height": 640}
T0 = 1000.0


def cfg(**kw):
    c = SimpleNamespace(
        fov_shrink_mode="anchor", fov_size=200, fov_height=200, crosshairX=320, crosshairY=320,
        afov_source="detection", afov_position="center", afov_offset_x=0, afov_offset_y=0,
        afov_follow_speed=12.0, afov_lost_behavior="hold", fov_circle_filter_enabled=False)
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def new_state():
    return SimpleNamespace(afov=None, tracker=TargetTracker(), locked_box=None)


def box_at(x, y, w=40.0, h=80.0):
    return [x - w / 2, y - h / 2, x + w / 2, y + h / 2]


def run(c, state, frames, fps=144.0, start=T0, tracker_feed=False):
    """frames: callable(i, t) -> (boxes, locked_box).  Returns per-frame centres."""
    out = []
    for i in range(frames_count(frames)):
        t = start + i / fps
        boxes, locked = frames_fn(frames)(i, t)
        state.locked_box = locked
        if tracker_feed and locked is not None:
            state.tracker.on_match(t, locked, 0.9, same_target=(state.tracker.last_box is not None))
        cx, cy, w, h = af.update_anchor_fov(c, state, boxes, CROSS[0], CROSS[1], REGION, t)
        out.append((cx, cy, w, h))
    return out


def frames_count(f):
    return f[0]


def frames_fn(f):
    return f[1]


def seq(n, fn):
    return (n, fn)


# 1 ─ stationary object ───────────────────────────────────────────────────────
@pytest.mark.parametrize("fps", [60.0, 144.0, 240.0])
def test_follows_stationary_object_and_settles(fps):
    c, s = cfg(), new_state()
    b = box_at(450, 250)
    out = run(c, s, seq(int(fps * 2), lambda i, t: ([b], list(b))), fps)
    cx, cy, w, h = out[-1]
    assert (cx, cy) == pytest.approx((450, 250), abs=0.5)
    assert (w, h) == (200, 200)                      # radius 100, aspect 1:1
    assert max(abs(out[-1][0] - out[-30][0]), abs(out[-1][1] - out[-30][1])) < 0.1   # no drift
    # monotone approach, never overshoots
    xs = [o[0] for o in out]
    assert all(b2 >= a - 1e-9 for a, b2 in zip(xs, xs[1:])) and max(xs) <= 450 + 1e-6


def test_starts_at_crosshair_and_never_jumps():
    c, s = cfg(), new_state()
    out = run(c, s, seq(5, lambda i, t: ([box_at(600, 600)], box_at(600, 600))))
    assert out[0][:2] == pytest.approx(CROSS)         # first frame: exactly the crosshair
    steps = [math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(out, out[1:])]
    assert max(steps) < 0.5 * math.hypot(600 - 320, 600 - 320)


# 2 ─ moving object ───────────────────────────────────────────────────────────
def test_follows_moving_object_with_small_lag():
    c, s = cfg(afov_follow_speed=20.0), new_state()
    v = 200.0
    out = run(c, s, seq(288, lambda i, t: ([box_at(100 + v * (t - T0), 300)], box_at(100 + v * (t - T0), 300))))
    cx, cy, _, _ = out[-1]
    truth = 100 + v * (287 / 144.0)
    lag = truth - cx
    assert 0 < lag < 0.25 * 100        # behind by roughly v / follow_speed, well inside the radius
    assert cy == pytest.approx(300, abs=0.5)


# 3 ─ rapid movement ──────────────────────────────────────────────────────────
def test_rapid_movement_respects_max_speed_and_stays_continuous():
    c, s = cfg(afov_follow_speed=100.0), new_state()
    fps = 144.0
    out = run(c, s, seq(144, lambda i, t: ([box_at(40 + 3000 * (t - T0) % 560, 300)], box_at(40 + 3000 * (t - T0) % 560, 300))), fps)
    steps = [math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(out, out[1:])]
    assert max(steps) <= af.MAX_SPEED_PX_S / fps + 1e-6
    assert all(0 <= o[0] <= 640 and 0 <= o[1] <= 640 for o in out)       # kept inside the region


# 4 ─ noisy boxes ─────────────────────────────────────────────────────────────
def test_noisy_boxes_are_smoothed():
    rnd = random.Random(7)
    noise = [(rnd.gauss(0, 4), rnd.gauss(0, 4)) for _ in range(600)]

    def spread():
        c, s = cfg(afov_follow_speed=15.0), new_state()
        out = run(c, s, seq(600, lambda i, t: ([box_at(400 + noise[i][0], 300 + noise[i][1])], box_at(400 + noise[i][0], 300 + noise[i][1]))))
        tail = [o[0] for o in out[300:]]
        m = sum(tail) / len(tail)
        return math.sqrt(sum((x - m) ** 2 for x in tail) / len(tail))

    raw = 4.0
    assert spread() < raw * 0.6             # the fixed input smoothing plus the follow lag average the jitter






# 6 ─ temporary loss ──────────────────────────────────────────────────────────
def test_temporary_loss_keeps_position_then_resumes():
    c = cfg(afov_lost_delay_ms=150)
    s = new_state()
    b = box_at(450, 250)
    out = run(c, s, seq(144, lambda i, t: ([b], list(b))))
    before = out[-1]
    gone = run(c, s, seq(14, lambda i, t: ([], None)), start=T0 + 1.0)       # ~97 ms < 150 ms
    assert s.afov.phase == af.PH_GRACE
    assert all(g[:2] == pytest.approx(before[:2], abs=0.5) for g in gone)
    assert gone[-1][2] == before[2]                                          # radius unchanged
    back = run(c, s, seq(30, lambda i, t: ([b], list(b))), start=T0 + 1.2)
    assert s.afov.phase == af.PH_FOLLOW and back[-1][:2] == pytest.approx((450, 250), abs=0.5)


# 7 ─ permanent loss (all three behaviours) ───────────────────────────────────
def settle(c, s, x=450, y=250):
    b = box_at(x, y)
    return run(c, s, seq(288, lambda i, t: ([b], list(b))))


def test_lost_hold_stays_put():
    c, s = cfg(afov_lost_behavior="hold"), new_state()
    settle(c, s)
    out = run(c, s, seq(576, lambda i, t: ([], None)), start=T0 + 2.0)
    assert s.afov.phase == af.PH_HOLD and s.afov.lost
    assert out[-1][:2] == pytest.approx((450, 250), abs=0.5)
    assert out[-1][2] == 200


def test_lost_return_glides_back_to_crosshair():
    c, s = cfg(afov_lost_behavior="return"), new_state()
    settle(c, s)
    out = run(c, s, seq(576, lambda i, t: ([], None)), start=T0 + 2.0)
    assert s.afov.phase == af.PH_RETURN
    assert out[-1][:2] == pytest.approx(CROSS, abs=0.5)
    steps = [math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(out, out[1:])]
    assert max(steps) < 12                                   # glides, never teleports


def test_lost_search_expands_then_shrinks_back_on_reacquire():
    c, s = cfg(afov_lost_behavior="search"), new_state()
    settle(c, s)
    out = run(c, s, seq(432, lambda i, t: ([], None)), start=T0 + 2.0)
    assert s.afov.phase == af.PH_SEARCH
    assert out[-1][2] == pytest.approx(600, abs=2)           # radius -> max
    assert out[-1][:2] == pytest.approx((450, 250), abs=0.5)  # position held
    back = run(c, s, seq(432, lambda i, t: ([box_at(450, 250)], None)), start=T0 + 5.0)
    assert back[-1][2] == pytest.approx(200, abs=2)
    radii = [o[2] for o in out]
    assert all(b >= a - 1e-9 for a, b in zip(radii, radii[1:]))              # smooth, monotone


# 8 ─ reacquisition ───────────────────────────────────────────────────────────
@pytest.mark.parametrize("behavior", ["hold", "return", "search"])
def test_reacquires_a_new_object(behavior):
    # the new object has to appear inside the FOV where the behaviour left it
    new = {"hold": box_at(480, 280), "return": box_at(330, 330), "search": box_at(200, 400)}[behavior]
    c, s = cfg(afov_lost_behavior=behavior), new_state()
    settle(c, s)
    run(c, s, seq(288, lambda i, t: ([], None)), start=T0 + 2.0)
    out = run(c, s, seq(288, lambda i, t: ([new], None)), start=T0 + 4.0)
    assert s.afov.phase == af.PH_FOLLOW
    assert out[-1][:2] == pytest.approx(((new[0] + new[2]) / 2, (new[1] + new[3]) / 2), abs=0.5)



def test_detection_with_lock_follows_the_locked_target_not_the_nearest():
    c, s = cfg(), new_state()
    locked, other = box_at(500, 500), box_at(330, 330)
    out = run(c, s, seq(288, lambda i, t: ([other, locked], list(locked))))
    assert out[-1][:2] == pytest.approx((500, 500), abs=0.5)


def test_detection_without_lock_picks_nearest_in_fov_only():
    c, s = cfg(), new_state()
    near, far = box_at(350, 330), box_at(600, 600)
    out = run(c, s, seq(288, lambda i, t: ([far, near], None)))
    assert out[-1][:2] == pytest.approx((350, 330), abs=0.5)
    c2, s2 = cfg(), new_state()
    out2 = run(c2, s2, seq(60, lambda i, t: ([far], None)))        # outside the FOV: nothing to follow
    assert out2[-1][:2] == pytest.approx(CROSS, abs=0.01) and s2.afov.lost


# sources / positions / offsets ───────────────────────────────────────────────
def test_anchor_point_positions_and_offsets():
    b = [100, 200, 140, 300]
    assert af.anchor_point(b, "center", 0, 0) == (120.0, 250.0)
    assert af.anchor_point(b, "top", 0, 0) == (120.0, 200.0)
    assert af.anchor_point(b, "bottom", 0, 0) == (120.0, 300.0)
    assert af.anchor_point(b, "custom", 5, -10) == (125.0, 240.0)
    assert af.anchor_point(b, "top", -4, 6) == (116.0, 206.0)


def test_position_setting_changes_where_the_fov_settles():
    for pos, y in (("top", 250 - 40), ("bottom", 250 + 40), ("center", 250)):
        c, s = cfg(afov_position=pos), new_state()
        out = settle(c, s)
        assert out[-1][1] == pytest.approx(y, abs=0.5)
    c, s = cfg(afov_offset_x=10, afov_offset_y=-20), new_state()
    assert settle(c, s)[-1][:2] == pytest.approx((460, 230), abs=0.5)


def test_manual_source_is_never_lost_and_clamps_to_the_region():
    # Manual = crosshair + the same X/Y offset pair every other source adds to its anchor point
    c, s = cfg(afov_source="manual", afov_offset_x=100, afov_offset_y=-50), new_state()
    out = run(c, s, seq(288, lambda i, t: ([], None)))
    assert out[-1][:2] == pytest.approx((420, 270), abs=0.5) and not s.afov.lost
    c, s = cfg(afov_source="manual", afov_offset_x=4000, afov_offset_y=4000), new_state()
    out = run(c, s, seq(288, lambda i, t: ([], None)))
    assert out[-1][0] <= 640 and out[-1][1] <= 640


def test_tracker_source_follows_the_tracker_and_survives_short_gaps():
    c, s = cfg(afov_source="tracker"), new_state()
    b = box_at(450, 250)
    fps = 144.0
    for i in range(288):
        t = T0 + i / fps
        s.locked_box = b
        s.tracker.on_match(t, b, 0.9, same_target=(i > 0))
        out = af.update_anchor_fov(c, s, [], CROSS[0], CROSS[1], REGION, t)      # no detections at all
    assert out[:2] == pytest.approx((450, 250), abs=0.5)
    assert not s.afov.lost
    # no lock -> nothing to follow
    c2, s2 = cfg(afov_source="tracker"), new_state()
    af.update_anchor_fov(c2, s2, [b], 320, 320, REGION, T0)
    assert s2.afov.lost


# radius ──────────────────────────────────────────────────────────────────────
def test_size_is_the_fov_width_in_every_mode():
    """The Anchor FOV is FOV Width wide (radius = Width / 2), like every other FOV mode."""
    for width in (80, 200, 440):
        c, s = cfg(fov_size=width, fov_height=width), new_state()
        assert settle(c, s)[-1][2] == width


def test_search_radius_is_headroom_never_a_cap_on_the_base_size():
    # a base radius above the search radius must not shrink the FOV
    c, s = cfg(fov_size=800, fov_height=800), new_state()
    assert settle(c, s)[-1][2] == 800
    # the hard floor stops a tiny FOV Width from collapsing the radius
    c, s = cfg(fov_size=10, fov_height=10), new_state()
    assert settle(c, s)[-1][2] == int(2 * af.MIN_RADIUS_PX)



def test_rectangular_fov_keeps_its_aspect_ratio():
    c, s = cfg(fov_size=200, fov_height=100), new_state()
    out = settle(c, s)
    assert out[-1][2:] == (200, 100)


# robustness ──────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_invalid_coordinates_are_ignored(bad):
    c, s = cfg(), new_state()
    out = run(c, s, seq(10, lambda i, t: ([[bad, 300, bad, 340], box_at(400, 300)], None)))
    assert all(math.isfinite(v) for o in out for v in o)
    c, s = cfg(afov_source="manual", afov_manual_x=bad), new_state()
    out = run(c, s, seq(10, lambda i, t: ([], None)))
    assert all(math.isfinite(v) for o in out for v in o)


def test_stalls_and_clock_glitches_do_not_move_the_fov():
    c, s = cfg(), new_state()
    out = settle(c, s)
    stalled = af.update_anchor_fov(c, s, [box_at(100, 100)], 320, 320, REGION, T0 + 500.0)
    assert stalled[:2] == pytest.approx(out[-1][:2], abs=0.01)
    back = af.update_anchor_fov(c, s, [box_at(100, 100)], 320, 320, REGION, T0 + 10.0)   # clock went back
    assert back[:2] == pytest.approx(out[-1][:2], abs=0.01)


def test_unusual_settings_cannot_break_the_controller():
    c = cfg(afov_follow_speed=float("nan"), afov_radius="x",
            afov_min_radius=None, afov_lost_behavior="nonsense", afov_source=42)
    s = new_state()
    out = run(c, s, seq(30, lambda i, t: ([box_at(400, 300)], None)))
    assert all(math.isfinite(v) for o in out for v in o) and out[-1][2] > 0


def test_stable_at_high_inference_fps():
    for fps in (240.0, 500.0, 1000.0):
        c, s = cfg(), new_state()
        nb = box_at(500, 250)
        out = run(c, s, seq(int(fps), lambda i, t: ([nb], list(nb))), fps)
        assert out[-1][:2] == pytest.approx((500, 250), abs=1.0)
        xs = [o[0] for o in out]
        assert max(xs) <= 500 + 1e-6                                       # never overshoots


# 9 ─ mode switching ──────────────────────────────────────────────────────────
def test_switching_modes_resets_cleanly():
    c, s = cfg(), new_state()
    settle(c, s)
    assert af.fov_center(c) == pytest.approx((450, 250), abs=0.5) and c.afov_active
    c.fov_shrink_mode = "adaptive"
    assert af.fov_center(c) == (320.0, 320.0)           # readers fall back at once, even before the loop runs
    af.clear_anchor_fov(c, s)
    assert c.fov_center_x is None and not c.afov_active and not s.afov.initialised
    c.fov_shrink_mode = "anchor"
    out = af.update_anchor_fov(c, s, [box_at(500, 500)], 320, 320, REGION, T0 + 50)
    assert out[:2] == pytest.approx(CROSS)               # re-enters at the crosshair, no jump


def test_clear_is_a_no_op_for_other_modes():
    c = SimpleNamespace(fov_shrink_mode="adaptive")
    s = new_state()
    af.clear_anchor_fov(c, s)
    assert not hasattr(c, "afov_active") and s.afov is None


def test_published_state_is_the_controllers_real_state():
    c, s = cfg(), new_state()
    out = settle(c, s)
    ctrl = s.afov
    assert (c.fov_center_x, c.fov_center_y) == (ctrl.cx, ctrl.cy)
    assert c.fov_effective_size == out[-1][2] and c.afov_has_anchor and not c.afov_lost
    assert (c.afov_anchor_x, c.afov_anchor_y) == (ctrl.ax, ctrl.ay)
    assert not hasattr(c, "afov_pred_x")                   # the look-ahead marker is gone


# 10 ─ existing modes untouched ───────────────────────────────────────────────
def test_filter_uses_the_moved_centre():
    boxes = [[600, 300, 640, 340], [300, 300, 340, 340]]
    near_cross, _ = filter_boxes_by_fov(boxes, [0.9, 0.9], 320, 320, 100, SimpleNamespace(), fov_height=100)
    near_anchor, _ = filter_boxes_by_fov(boxes, [0.9, 0.9], 620, 320, 100, SimpleNamespace(), fov_height=100)
    assert near_cross == [boxes[1]] and near_anchor == [boxes[0]]


def test_compute_effective_fov_ignores_anchor_selection_values():
    # 'anchor' is handled before compute_effective_fov in the loop; calling it anyway
    # (e.g. from a stale caller) must behave like the plain fixed shrink, not crash.
    from core.ai_loop_state import LoopState
    c = SimpleNamespace(fov_size=300, fov_height=300, fov_reduce_on_target_enabled=False,
                        fov_shrink_mode="anchor")
    assert compute_effective_fov(c, LoopState(), T0) == (300, 300)


def _new_config():
    from unittest.mock import patch
    from core import config as cfgmod
    with patch("core.config._get_screen_size", return_value=(1920, 1080)):
        return cfgmod.Config()


def test_config_defaults_persistence_and_validation(tmp_path):
    from core import config as cfgmod
    Config = _new_config
    c = Config()
    assert c.fov_shrink_mode == "adaptive"
    assert c.afov_lost_behavior == "hold" and c.fov_center_x is None
    d = c.to_dict()
    assert d["aim"]["fov_anchor"]["source"] == "detection"
    assert "fov_center_x" not in str(d) and "afov_active" not in str(d)      # runtime-only
    c.fov_shrink_mode = "anchor"
    c.afov_source, c.afov_position = "tracker", "top"
    c.afov_lost_behavior = "search"
    c2 = Config()
    c2.from_dict(c.to_dict())
    assert (c2.fov_shrink_mode, c2.afov_source, c2.afov_position) == ("anchor", "tracker", "top")
    assert c2.afov_lost_behavior == "search"
    # an old config without any of it loads the defaults
    old = Config()
    old.from_dict({"aim": {"fov_size": 250}})
    assert old.afov_source == "detection" and old.fov_shrink_mode == "adaptive"
    # garbage is clamped
    bad = Config()
    bad.afov_source, bad.afov_lost_behavior, bad.afov_position = "zzz", 7, "custom"
    bad.afov_follow_speed = float("nan")
    bad.fov_shrink_mode = "bogus"
    cfgmod.validate_config(bad)
    assert bad.afov_source == "detection" and bad.afov_lost_behavior == "hold"
    assert bad.afov_position == "center"
    assert bad.afov_follow_speed == 12.0 and bad.fov_shrink_mode == "adaptive"


def test_anchor_settings_are_aim_preset_eligible():
    from core.config_manager import _AIM_PRESET_FIELDS
    assert {"afov_source", "afov_lost_behavior", "afov_follow_speed"} <= set(_AIM_PRESET_FIELDS)
    assert "fov_center_x" not in _AIM_PRESET_FIELDS
