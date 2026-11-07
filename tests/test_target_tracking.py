"""Tests for core/target_tracking.py (lock state machine, target selection,
acquisition/tracking confidence split) and the FOV shrink modes in
ai_loop_utils.compute_effective_fov.

All pure logic — no win32/Qt — so it runs on any platform.
"""
from types import SimpleNamespace

import pytest

from core.target_tracking import (
    TargetTracker, TrackState, detection_confidence_floor,
    filter_by_acquisition_confidence, hold_lock_on_miss, persistence_s, select_target,
    tracking_confidence,
)


def _cfg(**kw):
    base = dict(
        min_confidence=0.8, tracking_confidence=0.0, sticky_lock_enabled=True,
        lock_persistence_ms=250, lost_target_timeout_ms=1500,
        min_lock_duration_ms=150, target_switch_threshold=0.35,
        reacquire_radius_px=150, target_priority_mode='distance',
        target_priority_confidence_weight=0.5,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def _item(box, conf=0.9, crosshair=(0, 0)):
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return ((cx - crosshair[0]) ** 2 + (cy - crosshair[1]) ** 2, conf, cx, cy, box)


# ── state machine ───────────────────────────────────────────────────────────
class TestStateMachine:
    def test_full_lifecycle(self):
        t = TargetTracker()
        assert t.state is TrackState.SEARCHING
        box = [100, 100, 120, 120]
        t.on_match(1.0, box, 0.9, same_target=False)
        assert t.state is TrackState.ACQUIRED
        t.on_match(1.2, box, 0.9)                       # past the settle time
        assert t.state is TrackState.TRACKING
        t.on_miss(1.3)
        assert t.state is TrackState.TEMPORARILY_LOST
        assert t.lost_duration(1.8) == pytest.approx(0.5)
        t.on_match(1.8, box, 0.9)
        assert t.state is TrackState.REACQUIRED
        t.on_match(2.0, box, 0.9)
        assert t.state is TrackState.TRACKING
        t.on_miss(2.1)
        assert not t.expire(3.0, 1.5)                   # 0.9 s unseen < timeout
        assert t.expire(3.7, 1.5)                       # 1.6 s unseen
        assert t.state is TrackState.RELEASED
        t.on_miss(3.8)
        assert t.state is TrackState.SEARCHING
        assert t.last_center is None                    # forgotten

    def test_new_target_switch_restarts_signals(self):
        t = TargetTracker()
        t.on_match(1.0, [0, 0, 10, 10], 0.9, same_target=False)
        t.on_match(1.3, [0, 0, 10, 10], 0.9)
        assert t.state is TrackState.TRACKING and t.hit_ema > 0.5
        t.on_match(1.4, [200, 200, 210, 210], 0.9, same_target=False)
        assert t.state is TrackState.ACQUIRED
        assert t.hit_ema == 0.5 and t.acquired_t == 1.4

    def test_misses_lower_consistency(self):
        t = TargetTracker()
        t.on_match(1.0, [0, 0, 10, 10], 0.9, same_target=False)
        for i in range(1, 6):
            t.on_match(1.0 + i * 0.05, [0, 0, 10, 10], 0.9)
        before = t.hit_ema
        for i in range(1, 6):
            t.on_miss(1.25 + i * 0.05)
        assert t.hit_ema < before

    @pytest.mark.parametrize("fps", [30, 60, 240])
    def test_signals_are_frame_rate_independent(self, fps):
        t = TargetTracker()
        t.on_match(10.0, [0, 0, 10, 10], 0.9, same_target=False)
        dt = 1.0 / fps
        for i in range(1, fps + 1):                     # one second of steady tracking
            t.on_match(10.0 + i * dt, [0, 0, 10, 10], 0.9)
        # Same wall-clock time -> ~same smoothed value regardless of FPS.
        assert t.hit_ema == pytest.approx(1.0 - 0.5 * 2.718281828 ** (-1.0 / 0.4), abs=0.02)

    def test_reacquire_radius_uses_velocity(self):
        t = TargetTracker()
        t.on_match(1.0, [0, 0, 10, 10], 0.9, same_target=False)
        for i in range(1, 11):                          # moving right ~500 px/s
            x = i * 0.02 * 500
            t.on_match(1.0 + i * 0.02, [x, 0, x + 10, 10], 0.9)
        t.on_miss(1.22)
        far = [260, 0, 270, 10]                         # ~160 px past the last seen position
        assert t.within_reacquire_radius(far, 1.22, 130) is None        # not yet: no time to extrapolate
        assert t.within_reacquire_radius(far, 1.42, 130) is not None    # velocity carries the search forward
        assert t.within_reacquire_radius([900, 900, 910, 910], 1.42, 130) is None


# ── confidence split ────────────────────────────────────────────────────────
class TestConfidenceSplit:
    def test_default_is_single_threshold(self):
        c = _cfg()
        assert tracking_confidence(c) == 0.8 and detection_confidence_floor(c) == 0.8

    def test_lower_tracking_lowers_detector_floor(self):
        c = _cfg(tracking_confidence=0.5)
        assert tracking_confidence(c) == 0.5 and detection_confidence_floor(c) == 0.5

    def test_tracking_above_acquisition_is_ignored(self):
        assert tracking_confidence(_cfg(tracking_confidence=0.95)) == 0.8

    def test_sticky_off_means_single_threshold(self):
        assert detection_confidence_floor(_cfg(tracking_confidence=0.5, sticky_lock_enabled=False)) == 0.8

    def test_low_conf_dropped_unless_continuing_lock(self):
        c = _cfg(tracking_confidence=0.5)
        tr = TargetTracker()
        locked = [100, 100, 120, 120]
        tr.on_match(1.0, locked, 0.9, same_target=False)
        boxes = [[102, 102, 122, 122], [400, 400, 420, 420], [500, 500, 520, 520]]
        confs = [0.55, 0.55, 0.95]
        b, cf, ids = filter_by_acquisition_confidence(boxes, confs, [0, 0, 0], c, tr, locked, 1.1)
        assert b == [boxes[0], boxes[2]] and cf == [0.55, 0.95] and ids == [0, 0]

    def test_noop_returns_same_objects_when_equal(self):
        c = _cfg()
        boxes, confs, ids = [[0, 0, 1, 1]], [0.9], [0]
        out = filter_by_acquisition_confidence(boxes, confs, ids, c, TargetTracker(), None, 1.0)
        assert out[0] is boxes and out[1] is confs and out[2] is ids


# ── selection ───────────────────────────────────────────────────────────────
class TestSelectTarget:
    def _locked(self, t0=1.0):
        tr = TargetTracker()
        box = [100, 100, 120, 120]
        tr.on_match(t0, box, 0.9, same_target=False)
        return tr, box

    def test_sticks_with_overlapping_target_inside_min_lock(self):
        tr, box = self._locked()
        mine, rival = _item([101, 101, 121, 121]), _item([5, 5, 15, 15])   # rival is far closer
        ordered = sorted([mine, rival], key=lambda i: i[0])
        sel, cont = select_target(ordered, locked_box=box, tracker=tr, config=_cfg(),
                                  now=1.05, iou_thresh=0.3, norm=200)
        assert sel is mine and cont

    def test_switch_needs_min_lock_and_score_advantage(self):
        tr, box = self._locked()
        mine, rival = _item([101, 101, 121, 121]), _item([5, 5, 15, 15])
        ordered = sorted([mine, rival], key=lambda i: i[0])
        kw = dict(locked_box=box, tracker=tr, iou_thresh=0.3, norm=200)
        # old enough but advantage below threshold -> keep
        sel, _ = select_target(ordered, config=_cfg(target_switch_threshold=0.99), now=2.0, **kw)
        assert sel is mine
        # old enough and big advantage -> switch
        sel, cont = select_target(ordered, config=_cfg(), now=2.0, **kw)
        assert sel is rival and not cont
        # big advantage but lock too young -> keep
        sel, _ = select_target(ordered, config=_cfg(min_lock_duration_ms=5000), now=2.0, **kw)
        assert sel is mine

    def test_reacquires_by_radius_when_overlap_lost(self):
        tr, box = self._locked()
        tr.on_miss(1.1)
        moved = _item([170, 100, 190, 120])             # no overlap, 70 px away
        other = _item([600, 600, 620, 620])
        ordered = sorted([other, moved], key=lambda i: i[0])
        sel, cont = select_target(ordered, locked_box=None, tracker=tr, config=_cfg(),
                                  now=1.3, iou_thresh=0.3, norm=200)
        assert sel is moved and cont

    def test_memory_expires_after_lost_timeout(self):
        tr, box = self._locked()
        tr.on_miss(1.1)
        moved = _item([170, 100, 190, 120])
        sel, cont = select_target([moved], locked_box=None, tracker=tr,
                                  config=_cfg(lost_target_timeout_ms=200), now=2.0,
                                  iou_thresh=0.3, norm=200)
        assert sel is moved and not cont                # fresh pick, not a reacquisition

    def test_holds_instead_of_hopping_inside_min_lock(self):
        tr, box = self._locked()
        elsewhere = _item([600, 600, 620, 620])
        sel, _ = select_target([elsewhere], locked_box=box, tracker=tr, config=_cfg(),
                               now=1.05, iou_thresh=0.3, norm=200)
        assert sel is None
        sel, cont = select_target([elsewhere], locked_box=box, tracker=tr, config=_cfg(),
                                  now=2.0, iou_thresh=0.3, norm=200)
        assert sel is elsewhere and not cont            # lock old enough -> free to move on

    def test_new_target_needs_acquisition_confidence(self):
        c = _cfg(tracking_confidence=0.5)
        weak = _item([10, 10, 20, 20], conf=0.6)
        strong = _item([300, 300, 310, 310], conf=0.9)
        sel, _ = select_target([weak, strong], locked_box=None, tracker=TargetTracker(),
                               config=c, now=1.0, iou_thresh=0.3, norm=200)
        assert sel is strong
        sel, _ = select_target([weak], locked_box=None, tracker=TargetTracker(),
                               config=c, now=1.0, iou_thresh=0.3, norm=200)
        assert sel is None

    def test_persistence_never_exceeds_timeout(self):
        assert persistence_s(_cfg(lock_persistence_ms=2000, lost_target_timeout_ms=500)) == 0.5


# ── FOV shrink modes ────────────────────────────────────────────────────────
@pytest.fixture
def fov():
    from core import ai_loop_utils
    return ai_loop_utils.compute_effective_fov


def _fov_cfg(**kw):
    base = dict(
        fov_size=200, fov_height=200, fov_reduce_on_target_enabled=True,
        fov_min_size_pct=50.0, fov_min_size_duration=1.0, fov_shrink_mode='fixed',
        fov_expand_delay_ms=0, fov_recovery_duration_ms=0, min_confidence=0.8,
        tracking_confidence=0.0,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def _state(locked=True):
    from core.ai_loop_state import LoopState
    return LoopState(locked_box=[0, 0, 10, 10] if locked else None)


def _settle(state, conf=0.95, stab=0.9, hit=1.0, speed=0.0, turn=0.0, edge=0.1):
    tr = state.tracker
    tr.conf_ema, tr.stab_ema, tr.hit_ema = conf, stab, hit
    tr.speed_ema, tr.turn_ema, tr.edge_ratio = speed, turn, edge


class TestFovModes:
    def test_fixed_delay_then_recovery(self, fov):
        c = _fov_cfg(fov_expand_delay_ms=300, fov_recovery_duration_ms=500, fov_min_size_duration=0.0)
        s = _state()
        assert fov(c, s, 10.0)[0] == 100                # instant shrink (duration 0)
        s.locked_box = None                             # released
        assert fov(c, s, 10.1)[0] == 100                # inside the expansion delay
        assert fov(c, s, 10.35)[0] == 100               # delay (300 ms from 10.1) not over yet
        # Released at 10.1, delay ends 10.4; 200 ms into the 500 ms recovery = 70 %.
        assert fov(c, s, 10.6)[0] == pytest.approx(140, abs=1)
        # Sparse calls must not count the delay toward recovery.
        s2 = _state()
        fov(c, s2, 20.0)
        s2.locked_box = None
        fov(c, s2, 20.1)
        assert fov(c, s2, 20.6)[0] == pytest.approx(140, abs=1)
        assert fov(c, s, 11.2)[0] == 200                # fully recovered

    def test_fixed_holds_minimum_while_locked_and_relocks_smoothly(self, fov):
        c = _fov_cfg(fov_expand_delay_ms=300, fov_recovery_duration_ms=500, fov_min_size_duration=0.0)
        s = _state()
        fov(c, s, 10.0)
        s.locked_box = None
        fov(c, s, 10.1)
        fov(c, s, 10.5 + 0.1)                           # mid recovery
        mid = s.fov_scale
        assert 0.5 < mid < 1.0
        s.locked_box = [0, 0, 10, 10]                   # new lock mid-recovery
        assert fov(c, s, 10.61)[0] == 100               # shrink is instant (duration 0) from current scale

    @pytest.mark.parametrize("step", [0.004, 0.016, 0.1])
    def test_shrink_is_frame_rate_independent(self, fov, step):
        c = _fov_cfg(fov_min_size_duration=1.0)
        s = _state()
        t = 100.0
        fov(c, s, t)
        end = t + 0.5
        while t < end - 1e-9:
            t += step
            fov(c, s, min(t, end))
        assert s.fov_scale == pytest.approx(0.75, abs=1e-6)

    def test_adaptive_contracts_when_stable_and_central(self, fov):
        c = _fov_cfg(fov_shrink_mode='adaptive')
        s = _state()
        _settle(s)
        sizes = [fov(c, s, 50.0 + i * 0.05)[0] for i in range(60)]
        assert sizes[-1] == 100                          # reached the minimum
        assert all(a >= b for a, b in zip(sizes, sizes[1:]))   # monotone: no oscillation

    def test_adaptive_expands_toward_boundary_and_on_low_confidence(self, fov):
        c = _fov_cfg(fov_shrink_mode='adaptive', fov_recovery_duration_ms=400)
        s = _state()
        _settle(s)
        for i in range(60):
            fov(c, s, 50.0 + i * 0.05)
        assert s.fov_scale == pytest.approx(0.5)
        _settle(s, edge=0.95)                            # drifting to the FOV edge
        for i in range(30):
            fov(c, s, 53.0 + i * 0.05)
        assert s.fov_scale > 0.9
        _settle(s)                                       # stable again -> contracts again
        for i in range(80):
            fov(c, s, 55.0 + i * 0.05)
        assert s.fov_scale == pytest.approx(0.5)
        _settle(s, conf=0.80, stab=0.2, hit=0.3)         # shaky tracking
        for i in range(40):
            fov(c, s, 60.0 + i * 0.05)
        assert s.fov_scale > 0.9

    def test_dynamic_expands_on_rapid_motion_and_turns(self, fov):
        c = _fov_cfg(fov_shrink_mode='dynamic', fov_recovery_duration_ms=400)
        s = _state()
        _settle(s)
        for i in range(60):
            fov(c, s, 50.0 + i * 0.05)
        assert s.fov_scale == pytest.approx(0.5)
        _settle(s, speed=600.0, turn=0.8)                # 3 FOV widths/s with direction flips
        for i in range(40):
            fov(c, s, 53.0 + i * 0.05)
        assert s.fov_scale > 0.9
        _settle(s)
        for i in range(80):
            fov(c, s, 55.0 + i * 0.05)
        assert s.fov_scale == pytest.approx(0.5)

    def test_never_below_minimum_or_above_full(self, fov):
        c = _fov_cfg(fov_shrink_mode='dynamic', fov_min_size_pct=30.0)
        s = _state()
        for i in range(200):
            _settle(s, speed=(i % 7) * 100.0, turn=(i % 3) / 3.0, edge=(i % 11) / 10.0)
            size, height = fov(c, s, 1.0 + i * 0.016)
            assert 60 <= size <= 200 and 60 <= height <= 200

    def test_slow_stall_does_not_jump(self, fov):
        """A long gap between calls (e.g. a paused loop) just advances time; the
        result stays inside [min, full]."""
        c = _fov_cfg(fov_shrink_mode='adaptive')
        s = _state()
        _settle(s)
        fov(c, s, 1.0)
        size, _ = fov(c, s, 500.0)
        assert 100 <= size <= 200


# ── persistence / lost timeout bookkeeping (what ai_loop's hold branch calls) ─
class TestHoldLockOnMiss:
    @staticmethod
    def _run(fps, cfg, total_s):
        """Lock a target, then miss every frame for ``total_s``; return
        {elapsed_ms: (holding, state)} sampled at each frame."""
        tr = TargetTracker()
        tr.on_match(100.0, [0, 0, 10, 10], 0.9, same_target=False)
        tr.on_match(100.2, [0, 0, 10, 10], 0.9)
        t0, out = 100.2, []
        n = int(total_s * fps)
        has_lock = True
        for i in range(1, n + 1):
            now = t0 + i / fps
            holding = hold_lock_on_miss(tr, cfg, now, has_lock)
            if not holding:
                has_lock = False          # ai_loop clears locked_box once the grace period ends
            out.append((i / fps, holding, tr.state))
        return out

    @pytest.mark.parametrize("fps", [20, 60, 240])
    def test_grace_period_and_timeout_are_time_based(self, fps):
        cfg = _cfg(lock_persistence_ms=250, lost_target_timeout_ms=1000)
        res = self._run(fps, cfg, 1.5)
        held = [e for e, h, _ in res if h]
        assert held and max(held) < 0.25 + 1.0 / fps          # held for ~250 ms, whatever the FPS
        assert min(e for e, h, _ in res if not h) >= 0.25 - 1e-9
        lost = [e for e, h, st in res if st is TrackState.TEMPORARILY_LOST]
        assert max(lost) < 1.0 + 1.0 / fps                    # still remembered until the timeout
        assert res[-1][2] in (TrackState.RELEASED, TrackState.SEARCHING)

    def test_remembered_between_persistence_and_timeout(self):
        tr = TargetTracker()
        cfg = _cfg(lock_persistence_ms=250, lost_target_timeout_ms=1000)
        tr.on_match(10.0, [0, 0, 10, 10], 0.9, same_target=False)
        assert hold_lock_on_miss(tr, cfg, 10.1, True)
        assert not hold_lock_on_miss(tr, cfg, 10.4, True)     # grace over ...
        assert tr.has_memory(10.5, 1.0)                       # ... target still reacquirable
        assert not hold_lock_on_miss(tr, cfg, 11.2, False)    # timeout reached
        assert not tr.has_memory(11.2, 1.0) and tr.last_center is None

    def test_sticky_off_forgets_immediately(self):
        tr = TargetTracker()
        tr.on_match(10.0, [0, 0, 10, 10], 0.9, same_target=False)
        assert not hold_lock_on_miss(tr, _cfg(sticky_lock_enabled=False), 10.01, True)
        assert tr.last_center is None

    def test_persistence_longer_than_timeout_is_capped(self):
        tr = TargetTracker()
        cfg = _cfg(lock_persistence_ms=2000, lost_target_timeout_ms=300)
        tr.on_match(10.0, [0, 0, 10, 10], 0.9, same_target=False)
        assert hold_lock_on_miss(tr, cfg, 10.1, True)
        assert not hold_lock_on_miss(tr, cfg, 10.4, True)


# ── numeric safeguards and identity gate (GPT-6 review follow-up) ───────────
class TestNumericSafeguards:
    def test_ema_ignores_bad_dt_and_samples(self):
        from core.target_tracking import _ema
        nan, inf = float('nan'), float('inf')
        assert _ema(0.4, 1.0, nan, 0.3) == 0.4
        assert _ema(0.4, 1.0, inf, 0.3) == 0.4
        assert _ema(0.4, nan, 0.01, 0.3) == 0.4
        assert _ema(0.4, inf, 0.01, 0.3) == 0.4
        assert _ema(0.4, 1.0, -1.0, 0.3) == 0.4
        assert _ema(0.4, 1.0, 0.01, 0.0) == 0.4

    def test_ema_is_frame_rate_independent(self):
        from core.target_tracking import _ema
        out = []
        for hz in (20, 60, 180, 240):
            v = 0.0
            for _ in range(hz // 4):                # 0.25 s of constant input
                v = _ema(v, 1.0, 1.0 / hz, 0.25)
            out.append(v)
        assert out == pytest.approx([1 - 2.718281828 ** -1] * 4, rel=1e-4)

    def test_fov_slew_ignores_stale_clock_and_bad_dt(self):
        from core.ai_loop_utils import _slew
        assert _slew(1.0, 0.5, 1.0, 0.4, 0.5, 10.0) == 1.0        # 10 s stall: no jump
        assert _slew(1.0, 0.5, 1.0, 0.4, 0.5, float('nan')) == 1.0
        assert _slew(1.0, 0.5, 1.0, 0.4, 0.5, -1.0) == 1.0
        assert _slew(1.0, 0.5, 1.0, 0.4, 0.5, 0.5) == pytest.approx(0.75)   # exactly span/T * dt
        assert _slew(1.0, 0.5, 0.0, 0.4, 0.5, 0.0) == 0.5          # zero duration = instant

    def test_slew_same_travel_time_at_any_rate(self):
        from core.ai_loop_utils import _slew
        for hz in (30, 60, 180, 240):
            s = 1.0
            for _ in range(hz // 2):
                s = _slew(s, 0.5, 1.0, 0.4, 0.5, 1.0 / hz)
            assert s == pytest.approx(0.75, abs=1e-9), hz


class TestFovConfigValidation:
    def test_fov_bounds_repaired(self):
        pytest.importorskip('numpy')
        from core import config as cfgmod
        c = SimpleNamespace(fov_size=0, fov_height=-5, fov_min_size_pct=0.0, fov_min_size_duration=float('nan'))
        cfgmod._validate_tracking(c)
        assert c.fov_size >= 1 and c.fov_height >= 1
        assert c.fov_min_size_pct == 1.0
        assert c.fov_min_size_duration == 1.0


class TestReacquireAmbiguity:
    def _tr(self):
        t = TargetTracker()
        t.on_match(1.0, [100, 100, 120, 120], 0.9, same_target=False)
        t.on_miss(1.1)
        return t

    def test_two_equally_close_candidates_are_not_claimed(self):
        tr = self._tr()
        a, b = _item([170, 100, 190, 120]), _item([170, 140, 190, 160])   # ~70 px and ~76 px away
        sel, cont = select_target(sorted([a, b], key=lambda i: i[0]), locked_box=None, tracker=tr,
                                  config=_cfg(), now=1.3, iou_thresh=0.3, norm=200)
        assert sel is None and not cont

    def test_clear_winner_still_reacquires(self):
        tr = self._tr()
        near, far = _item([170, 100, 190, 120]), _item([170, 220, 190, 240])
        sel, cont = select_target(sorted([near, far], key=lambda i: i[0]), locked_box=None, tracker=tr,
                                  config=_cfg(), now=1.3, iou_thresh=0.3, norm=200)
        assert sel is near and cont


# ── per-mode behaviour (FOV mode review) ────────────────────────────────────
def _target(mode, **sig):
    """Desired FOV scale (min 50 %) for a locked target with given smoothed signals."""
    from core.ai_loop_utils import _fov_tightness
    c = _fov_cfg(fov_shrink_mode=mode)
    s = _state()
    _settle(s, **sig)
    return 1.0 - 0.5 * _fov_tightness(mode, c, s)


class TestFixedMode:
    def test_always_requests_minimum_whatever_the_signals(self, fov):
        for sig in (dict(), dict(conf=0.5, stab=0.0, hit=0.0), dict(speed=900, turn=1.0, edge=1.0)):
            c = _fov_cfg(fov_shrink_mode='fixed', fov_min_size_duration=0.0)
            s = _state()
            _settle(s, **sig)
            assert fov(c, s, 10.0)[0] == 100

    def test_recovers_after_release(self, fov):
        c = _fov_cfg(fov_shrink_mode='fixed', fov_min_size_duration=0.0, fov_recovery_duration_ms=400)
        s = _state()
        fov(c, s, 10.0)
        s.locked_box = None
        assert fov(c, s, 10.1)[0] == 100                            # release seen; zero delay starts timing here
        assert fov(c, s, 10.3)[0] == pytest.approx(150, abs=1)      # 200 ms of a 400 ms recovery
        assert fov(c, s, 10.6)[0] == 200


class TestAdaptiveMode:
    def test_fresh_lock_is_conservative(self):
        # conf .85 (floor .8) -> conf_q .75; stab/hit start at .5 -> Q .57 -> tightness .175
        assert _target('adaptive', conf=0.85, hit=0.5, stab=0.5, edge=0.0) == pytest.approx(0.9125)

    def test_stable_centred_target_reaches_minimum(self):
        assert _target('adaptive', conf=0.90, hit=1.0, stab=0.9, edge=0.0) == pytest.approx(0.5)

    def test_low_quality_prevents_contraction(self):
        assert _target('adaptive', conf=0.85, hit=0.4, stab=0.3, edge=0.0) == pytest.approx(1.0)

    def test_boundary_proximity_limits_contraction_even_with_perfect_quality(self):
        near = _target('adaptive', conf=0.9, hit=1.0, stab=0.9, edge=0.8)
        assert near == pytest.approx(1 - 0.5 * (0.9 - 0.8) / 0.45)     # edge_q .222
        assert _target('adaptive', conf=0.9, hit=1.0, stab=0.9, edge=0.9) == pytest.approx(1.0)
        assert _target('adaptive', conf=0.9, hit=1.0, stab=0.9, edge=0.45) == pytest.approx(0.5)

    def test_tightness_monotonic_in_each_signal(self):
        base = dict(conf=0.85, hit=0.7, stab=0.6, edge=0.3)
        for key, lo, hi in (('conf', 0.8, 0.95), ('hit', 0.3, 1.0), ('stab', 0.3, 0.9)):
            assert _target('adaptive', **{**base, key: hi}) <= _target('adaptive', **{**base, key: lo})
        assert _target('adaptive', **{**base, 'edge': 0.8}) >= _target('adaptive', **{**base, 'edge': 0.3})

    def test_edge_ratio_rectangle_vs_ellipse(self):
        from core.target_tracking import edge_ratio
        box = [140, 90, 160, 110]                      # centre (150,100) vs crosshair (100,100)
        # nx = 50/100 = .5, ny = 0  -> rect max = .5
        assert edge_ratio(box, 100, 100, 200, 200, False) == pytest.approx(0.5)
        diag = [170, 170, 190, 190]                    # centre (180,180): nx=ny=.8
        assert edge_ratio(diag, 100, 100, 200, 200, False) == pytest.approx(0.8)
        assert edge_ratio(diag, 100, 100, 200, 200, True) == pytest.approx(0.8 * 2 ** 0.5)

    def test_converges_without_oscillation_for_a_stationary_target(self, fov):
        # edge is measured against the CURRENT (shrunk) FOV: negative feedback settles, not cycles.
        for hz in (30, 120, 240):
            c = _fov_cfg(fov_shrink_mode='adaptive', fov_recovery_duration_ms=400)
            s = _state()
            t, tail = 10.0, []
            for i in range(hz * 15):
                t += 1.0 / hz
                _settle(s, edge=0.4 * 100 / (s.fov_scale * 100))   # target 40 px off a 100 px half-extent
                fov(c, s, t)
                tail.append(s.fov_scale)
            last = tail[-hz * 2:]
            assert max(last) - min(last) < 0.01, hz
            assert 0.6 < last[-1] < 0.75                             # ~0.667, centre stays inside


class TestDynamicMode:
    def test_motion_and_turn_terms(self):
        # stationary, straight, perfect signals -> full tightness
        assert _target('dynamic', conf=0.9, hit=1.0, stab=0.9, speed=0, turn=0) == pytest.approx(0.5)
        fast = _target('dynamic', conf=0.9, hit=1.0, stab=0.9, speed=250, turn=0)      # 1.25 widths/s
        turn = _target('dynamic', conf=0.9, hit=1.0, stab=0.9, speed=100, turn=0.5)    # sustained 90 deg
        assert fast == pytest.approx(1 - 0.5 * 0.625)
        assert turn == pytest.approx(1 - 0.5 * 0.375)
        assert fast > 0.5 and turn > fast - 0.2

    def test_speed_monotonic(self):
        vals = [_target('dynamic', conf=0.9, hit=1.0, stab=0.9, speed=v) for v in (0, 100, 200, 300)]
        assert vals == sorted(vals)

    def test_ignores_boundary_proximity_today(self):
        """Characterisation: dynamic has no edge term, so a predictable target at the
        boundary still gets full contraction (adaptive would give ~0.89)."""
        d = _target('dynamic', conf=0.9, hit=1.0, stab=0.9, speed=20, turn=0, edge=0.85)
        a = _target('adaptive', conf=0.9, hit=1.0, stab=0.9, speed=20, turn=0, edge=0.85)
        assert d == pytest.approx(0.5) and a > 0.9

    def test_fresh_lock_is_optimistic_today(self):
        """Characterisation: speed/turn EMAs start at 0, so a brand-new lock scores
        motion_q = turn_q = 1 and dynamic contracts MORE than adaptive at acquisition."""
        d = _target('dynamic', conf=0.85, hit=0.5, stab=0.5)
        a = _target('adaptive', conf=0.85, hit=0.5, stab=0.5)
        assert d == pytest.approx(0.628, abs=1e-3) and a == pytest.approx(0.9125)


class TestSharedSlewController:
    def test_scale_stays_within_bounds(self, fov):
        c = _fov_cfg(fov_shrink_mode='fixed', fov_min_size_pct=30.0, fov_min_size_duration=0.5, fov_recovery_duration_ms=300)
        s = _state()
        t = 10.0
        for i in range(600):
            t += 0.01
            s.locked_box = [0, 0, 10, 10] if (i // 100) % 2 == 0 else None
            fov(c, s, t)
            assert 0.30 - 1e-9 <= s.fov_scale <= 1.0 + 1e-9

    def test_contraction_time_same_at_all_rates(self, fov):
        for hz in (30, 60, 120, 180, 240):
            c = _fov_cfg(fov_shrink_mode='fixed', fov_min_size_duration=1.0)
            s = _state()
            t = 10.0
            fov(c, s, t)
            for _ in range(hz // 2):
                t += 1.0 / hz
                fov(c, s, t)
            assert s.fov_scale == pytest.approx(0.75, abs=1e-6), hz        # half the 1 s travel

    def test_large_dt_does_not_jump(self, fov):
        c = _fov_cfg(fov_shrink_mode='fixed', fov_min_size_duration=1.0)
        s = _state()
        fov(c, s, 10.0)
        assert fov(c, s, 60.0)[0] == 200                                   # 50 s stall: clock restarts
