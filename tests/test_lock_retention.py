"""Lock association / retention FOV tests.

All SIMULATED, pure-logic tests (no win32/Qt/ONNX): they exercise
ai_loop_utils.resolve_retained_lock / track_retained, target_tracking's
find_continuation / select_target and compute_effective_fov, and a replay that
mimics ai_loop's per-frame ordering with synthetic detections. They say nothing
about real capture, real detector output or Windows behaviour.
"""
from types import SimpleNamespace

import pytest

from core.ai_loop_state import LoopState
from core.ai_loop_utils import (
    compute_effective_fov, filter_boxes_by_fov, resolve_retained_lock,
    retention_fov_size, track_retained,
)
from core.target_tracking import (
    TargetTracker, TrackState, build_class_lookup, find_continuation,
    hold_lock_on_miss, select_target,
)

CX = CY = 500.0


def _cfg(**kw):
    base = dict(
        fov_size=400, fov_height=400, fov_reduce_on_target_enabled=True, fov_min_size_pct=50.0,
        fov_min_size_duration=0.0, fov_shrink_mode='fixed', fov_expand_delay_ms=0,
        fov_recovery_duration_ms=0, fov_retention_enabled=True, fov_retention_pct=100.0,
        sticky_lock_enabled=True, min_confidence=0.5, tracking_confidence=0.0,
        lock_persistence_ms=250, lost_target_timeout_ms=1500, min_lock_duration_ms=0,
        target_switch_threshold=0.35, reacquire_radius_px=150, target_priority_mode='distance',
        target_priority_confidence_weight=0.5, fov_circle_filter_enabled=False,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def _box(cx, cy, w=20, h=40):
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


def _locked_state(box, t=10.0, cls=None):
    s = LoopState(locked_box=box)
    s.locked_confidence = 0.9
    s.tracker.on_match(t, box, 0.9, same_target=False)
    s.tracker.locked_class = cls
    return s


# ── retention decision ──────────────────────────────────────────────────────
class TestResolveRetainedLock:
    # fov 400 -> shrunk 200 (control half-extent 100); retention 400 (half 200)
    def _setup(self, lock_x, **cfg_kw):
        c = _cfg(**cfg_kw)
        lock = _box(lock_x, CY)
        s = _locked_state(lock)
        ctrl = filter_boxes_by_fov([lock], [0.9], CX, CY, 200, c, fov_height=200)
        return c, s, lock, ctrl

    def test_locked_target_stays_associated_while_fov_shrinks(self):
        c, s, lock, (cb, cc) = self._setup(CX + 150)             # outside 200 FOV, inside 400
        assert cb == []                                          # the control FOV drops it
        got = resolve_retained_lock(c, s, cb, [lock], [0.9], CX, CY, 10.1, 0.3)
        assert got is not None and got[0] is lock

    def test_disabled_by_default_behaviour_unchanged(self):
        c, s, lock, (cb, cc) = self._setup(CX + 150, fov_retention_enabled=False)
        assert resolve_retained_lock(c, s, cb, [lock], [0.9], CX, CY, 10.1, 0.3) is None

    def test_inside_control_fov_is_normal_selection(self):
        c, s, lock, (cb, cc) = self._setup(CX + 40)
        assert cb == [lock]
        assert resolve_retained_lock(c, s, cb, [lock], [0.9], CX, CY, 10.1, 0.3) is None

    def test_lost_outside_retention_fov(self):
        c, s, lock, (cb, cc) = self._setup(CX + 260)             # beyond even the 400 FOV
        assert resolve_retained_lock(c, s, cb, [lock], [0.9], CX, CY, 10.1, 0.3) is None

    def test_larger_retention_keeps_lock_further_out(self):
        c, s, lock, (cb, cc) = self._setup(CX + 260, fov_retention_pct=200.0)
        assert retention_fov_size(c) == (800, 800)
        assert resolve_retained_lock(c, s, cb, [lock], [0.9], CX, CY, 10.1, 0.3) is not None

    def test_rival_outside_control_fov_never_becomes_the_lock(self):
        c = _cfg()
        lock = _box(CX + 150, CY)                                # real lock drifted outside
        s = _locked_state(lock)
        rival = _box(CX - 180, CY + 20)                          # elsewhere, also outside control FOV
        # the lock vanished from detections this frame; only the rival is visible
        assert resolve_retained_lock(c, s, [], [rival], [0.9], CX, CY, 10.1, 0.3) is None
        assert s.locked_box is lock                              # untouched

    def test_requires_sticky_and_fov_reduce(self):
        for kw in (dict(sticky_lock_enabled=False), dict(fov_reduce_on_target_enabled=False)):
            c, s, lock, (cb, cc) = self._setup(CX + 150, **kw)
            assert resolve_retained_lock(c, s, cb, [lock], [0.9], CX, CY, 10.1, 0.3) is None

    def test_no_lock_means_no_retention(self):
        c = _cfg()
        s = LoopState()
        far = _box(CX + 150, CY)
        assert resolve_retained_lock(c, s, [], [far], [0.9], CX, CY, 10.0, 0.3) is None

    def test_bounded_by_lost_target_timeout(self):
        c, s, lock, (cb, cc) = self._setup(CX + 150, lost_target_timeout_ms=1000)
        assert resolve_retained_lock(c, s, cb, [lock], [0.9], CX, CY, 10.0, 0.3) is not None
        assert resolve_retained_lock(c, s, cb, [lock], [0.9], CX, CY, 10.9, 0.3) is not None
        assert resolve_retained_lock(c, s, cb, [lock], [0.9], CX, CY, 11.2, 0.3) is None   # parked too long
        # coming back inside the control FOV resets the stretch
        resolve_retained_lock(c, s, [lock], [lock], [0.9], CX, CY, 11.3, 0.3)
        assert s.retained_since == 0.0

    def test_ellipse_retention_region(self):
        c, s, lock, (cb, cc) = self._setup(CX + 150, fov_circle_filter_enabled=True)
        diag = _box(CX + 170, CY + 170)                          # inside the square, outside the 400 ellipse
        s2 = _locked_state(diag)
        assert resolve_retained_lock(c, s2, [], [diag], [0.9], CX, CY, 10.1, 0.3) is None


class TestTrackRetained:
    def test_keeps_identity_without_missing(self):
        c = _cfg()
        box = _box(CX + 150, CY)
        s = _locked_state(box)
        moved = _box(CX + 155, CY)
        track_retained(c, s, moved, 0.88, CX, CY, 10.05)
        assert s.locked_box is moved and s.locked_confidence == 0.88
        assert s.tracker.state in (TrackState.ACQUIRED, TrackState.TRACKING)   # not TEMPORARILY_LOST
        assert s.tracker.lost_since == 0.0
        assert s.no_detection_frames == 0
        assert c.display_locked_box == list(moved) and c.display_locked_box_is_decaying is False


# ── reacquisition safeguards ────────────────────────────────────────────────
class TestFindContinuation:
    def _tr(self, cls=None):
        t = TargetTracker()
        t.on_match(1.0, [100, 100, 120, 120], 0.9, same_target=False)
        t.locked_class = cls
        t.on_miss(1.1)
        return t

    def test_class_mismatch_is_never_a_continuation(self):
        t = self._tr(cls=1)
        overlapping_other_class = [102, 102, 122, 122]
        look = build_class_lookup([overlapping_other_class], [0])
        idx, amb = find_continuation([overlapping_other_class], locked_box=[100, 100, 120, 120],
                                     tracker=t, config=_cfg(), now=1.2, iou_thresh=0.3, class_of=look)
        assert idx is None and not amb

    def test_class_match_or_unknown_passes(self):
        t = self._tr(cls=1)
        b = [102, 102, 122, 122]
        for look in (build_class_lookup([b], [1]), None, build_class_lookup([b], [None])):
            idx, _ = find_continuation([b], locked_box=[100, 100, 120, 120], tracker=t,
                                       config=_cfg(), now=1.2, iou_thresh=0.3, class_of=look)
            assert idx == 0

    def test_ambiguous_candidates_hold(self):
        t = self._tr()
        a, b = [170, 100, 190, 120], [170, 140, 190, 160]
        idx, amb = find_continuation([a, b], locked_box=None, tracker=t, config=_cfg(), now=1.3, iou_thresh=0.3)
        assert idx is None and amb

    def test_implausible_jump_outside_radius_rejected(self):
        t = self._tr()
        far = [700, 700, 720, 720]
        idx, amb = find_continuation([far], locked_box=None, tracker=t, config=_cfg(), now=1.3, iou_thresh=0.3)
        assert idx is None and not amb

    def test_legitimate_reacquisition_with_velocity_still_works(self):
        t = TargetTracker()
        t.on_match(1.0, [100, 100, 120, 120], 0.9, same_target=False)
        for i in range(1, 6):                                    # moving +300 px/s in x
            t.on_match(1.0 + i * 0.05, [100 + 15 * i, 100, 120 + 15 * i, 120], 0.9)
        t.on_miss(1.3)
        ahead = [265, 100, 285, 120]                             # 0.45 s later, ahead of last position
        idx, amb = find_continuation([ahead], locked_box=None, tracker=t, config=_cfg(reacquire_radius_px=100),
                                     now=1.45, iou_thresh=0.3)
        assert idx == 0

    def test_select_target_class_gate(self):
        t = self._tr(cls=1)
        b = [102, 102, 122, 122]
        item = ((111 - 0) ** 2, 0.9, 112, 112, b)
        look = build_class_lookup([b], [0])
        sel, cont = select_target([item], locked_box=[100, 100, 120, 120], tracker=t, config=_cfg(),
                                  now=1.2, iou_thresh=0.3, norm=200, class_of=look)
        assert not cont          # different class: not the same target


# ── confidence decay during misses ─────────────────────────────────────────
class TestMissDecay:
    def test_confidence_decays_with_elapsed_time(self):
        t = TargetTracker()
        t.on_match(1.0, [0, 0, 10, 10], 0.95, same_target=False)
        t.on_miss(1.1)
        c1 = t.conf_ema
        t.on_miss(1.6)
        assert t.conf_ema < c1 < 0.95
        assert t.hit_ema < 0.5 and t.stab_ema < 0.5

    def test_decay_is_frame_rate_independent(self):
        outs = []
        for hz in (20, 60, 240):
            t = TargetTracker()
            t.on_match(1.0, [0, 0, 10, 10], 0.95, same_target=False)
            now = 1.0
            for _ in range(hz // 2):                              # 0.5 s of misses
                now += 1.0 / hz
                t.on_miss(now)
            outs.append(t.conf_ema)
        assert outs == pytest.approx([outs[1]] * 3, rel=1e-6)

    def test_rises_again_after_reacquire(self):
        t = TargetTracker()
        t.on_match(1.0, [0, 0, 10, 10], 0.95, same_target=False)
        for i in range(1, 11):
            t.on_miss(1.0 + 0.05 * i)
        low = t.conf_ema
        t.on_match(1.6, [0, 0, 10, 10], 0.95)
        assert t.conf_ema > low


# ── FOV recovery + mode compatibility with retention ───────────────────────
class TestFovCompat:
    @pytest.mark.parametrize('mode', ['fixed', 'adaptive', 'dynamic'])
    def test_modes_run_with_retention_enabled(self, mode):
        c = _cfg(fov_shrink_mode=mode, fov_min_size_duration=1.0, fov_recovery_duration_ms=400)
        s = LoopState(locked_box=[0, 0, 10, 10])
        t = 10.0
        for _ in range(240):
            t += 1 / 120
            w, h = compute_effective_fov(c, s, t)
            assert 200 <= w <= 400 and 200 <= h <= 400

    def test_fov_recovers_after_the_lock_is_lost(self):
        c = _cfg(fov_recovery_duration_ms=400)
        s = LoopState(locked_box=[0, 0, 10, 10])
        assert compute_effective_fov(c, s, 10.0) == (200, 200)
        s.locked_box = None                                      # released
        compute_effective_fov(c, s, 10.1)
        assert compute_effective_fov(c, s, 10.6) == (400, 400)

    def test_retained_lock_keeps_fov_shrunk_no_flapping(self):
        c = _cfg()
        s = LoopState(locked_box=[0, 0, 10, 10])
        sizes = []
        t = 10.0
        for _ in range(60):
            t += 1 / 60
            sizes.append(compute_effective_fov(c, s, t)[0])      # lock held (retained) the whole time
        assert set(sizes) == {200}


# ── config ──────────────────────────────────────────────────────────────────
def _make_config():
    from unittest.mock import patch
    with patch("core.config._get_screen_size", return_value=(1920, 1080)):
        from core.config import Config
        return Config()


class TestRetentionConfig:
    def test_validation_clamps(self):
        from core import config as cfgmod
        c = SimpleNamespace(fov_retention_enabled=1, fov_retention_pct=float('nan'))
        cfgmod._validate_tracking(c)
        assert c.fov_retention_enabled is True and c.fov_retention_pct == 100.0
        for raw, want in ((50, 100.0), (150, 150.0), (9999, 400.0)):
            c.fov_retention_pct = raw
            cfgmod._validate_tracking(c)
            assert c.fov_retention_pct == want

    def test_defaults_off_and_full_fov(self):
        c = _make_config()
        assert c.fov_retention_enabled is False and c.fov_retention_pct == 100.0

    def test_existing_config_loads_unchanged(self):
        c = _make_config()
        before = c.to_dict()
        old_file = c.to_dict()
        old_file['aim']['fov_reduce'].pop('retention_enabled', None)
        old_file['aim']['fov_reduce'].pop('retention_pct', None)
        c2 = _make_config()
        c2.from_dict(old_file)
        assert c2.fov_retention_enabled is False and c2.fov_retention_pct == 100.0
        assert c2.to_dict() == before

    def test_round_trip(self):
        a = _make_config()
        a.fov_retention_enabled, a.fov_retention_pct = True, 175.0
        b = _make_config()
        b.from_dict(a.to_dict())
        assert b.fov_retention_enabled is True and b.fov_retention_pct == 175.0


# ── replay: mimic ai_loop's per-frame ordering with synthetic detections ───
def _frame(cfg, state, dets, t, use_retention=True):
    """One frame of ai_loop's ordering: effective FOV -> control-FOV filter ->
    retention step -> selection / hold.  dets = [(box, conf, cls)].
    Returns 'aimed' | 'tracked' | 'miss'."""
    boxes = [d[0] for d in dets]
    confs = [d[1] for d in dets]
    class_of = build_class_lookup(boxes, [d[2] for d in dets]) if boxes else None
    w, h = compute_effective_fov(cfg, state, t)
    cb, cc = filter_boxes_by_fov(boxes, confs, CX, CY, w, cfg, fov_height=h)
    iou = 0.3
    if use_retention:
        r = resolve_retained_lock(cfg, state, cb, boxes, confs, CX, CY, t, iou, class_of)
        if r is not None:
            track_retained(cfg, state, r[0], r[1], CX, CY, t)
            return 'tracked'
    if cb:
        items = [(((b[0] + b[2]) / 2 - CX) ** 2 + ((b[1] + b[3]) / 2 - CY) ** 2, cc[i],
                  (b[0] + b[2]) / 2, (b[1] + b[3]) / 2, b) for i, b in enumerate(cb)]
        items.sort(key=lambda x: x[0])
        sel, cont = select_target(items, locked_box=state.locked_box, tracker=state.tracker, config=cfg,
                                  now=t, iou_thresh=iou, norm=200, class_of=class_of)
        if sel is not None:
            state.tracker.on_match(t, sel[4], sel[1], same_target=bool(cont))
            state.locked_box, state.locked_confidence = sel[4], sel[1]
            return 'aimed'
    holding = hold_lock_on_miss(state.tracker, cfg, t, state.locked_box is not None)
    if not holding:
        state.locked_box = None
    return 'miss'


class TestReplay:
    """SIMULATED replay: a locked target walks out past the shrunk FOV and back,
    with a rival standing near the edge."""

    def _run(self, retention):
        c = _cfg(fov_retention_enabled=retention, min_lock_duration_ms=0)
        s = LoopState()
        t, log, locks = 10.0, [], []
        mine = _box(CX + 20, CY)
        rival = _box(CX - 60, CY - 30)                           # inside the shrunk FOV the whole time
        # frames: acquire at centre, then drift to x=+150 (outside shrunk 200 FOV, inside 400), then back
        path = [20] * 10 + list(range(20, 151, 10)) + [150] * 20 + list(range(150, 19, -10)) + [20] * 5
        for x in path:
            t += 1 / 30
            mine = _box(CX + x, CY)
            dets = [(mine, 0.9, 0), (rival, 0.85, 0)]
            log.append(_frame(c, s, dets, t, use_retention=retention))
            locks.append(s.locked_box is mine)
        return log, locks

    def test_without_retention_the_lock_is_stolen_or_lost(self):
        log, locks = self._run(False)
        assert not all(locks)                                    # lock left "mine" at some point

    def test_with_retention_the_lock_never_leaves_the_original_target(self):
        log, locks = self._run(True)
        assert all(locks)
        assert 'tracked' in log and log[-1] == 'aimed'           # tracked outside, aiming resumed on return
        # never aimed at the rival; no aim frames while outside the control FOV
        first_tracked = log.index('tracked')
        assert all(x in ('tracked',) for x in log[first_tracked:first_tracked + 15])

    def test_loss_outside_retention_then_normal_acquisition(self):
        c = _cfg(lost_target_timeout_ms=300, lock_persistence_ms=100)
        s = LoopState()
        t = 10.0
        for _ in range(10):
            t += 1 / 30
            _frame(c, s, [(_box(CX + 20, CY), 0.9, 0)], t)
        gone = [(_box(CX + 330, CY), 0.9, 0)]                    # left even the retention FOV
        for _ in range(30):
            t += 1 / 30
            _frame(c, s, gone, t)
        assert s.locked_box is None
        newt = [(_box(CX - 30, CY), 0.9, 0)]
        t += 1 / 30
        assert _frame(c, s, newt, t) == 'aimed'                  # reacquired via the normal path
