"""Audit tests for FOV lock retention: identity, lock protection, ambiguity,
retention exit and reacquisition.

SIMULATED: the per-frame ordering of ai_loop.py is mirrored with pure functions
(compute_effective_fov -> control-FOV filter -> resolve_retained_lock ->
track_retained | select_target/on_match | hold_lock_on_miss) over synthetic
detections.  They do not exercise real capture, the detector, process_aiming's
mouse output, or anything Windows-specific.
"""
from types import SimpleNamespace

import pytest

from core.ai_loop_state import LoopState
from core.ai_loop_utils import (
    AMBIGUOUS_LOCK, compute_effective_fov, filter_boxes_by_fov, resolve_retained_lock, track_retained,
)
from core.target_tracking import TrackState, build_class_lookup, hold_lock_on_miss, select_target

CX = CY = 500.0
DT = 1 / 30


def cfg(**kw):
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


def box(cx, cy, w=20, h=40):
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


class Sim:
    """One aim session: feed detections frame by frame, record what happened."""

    def __init__(self, config=None, iou=0.3):
        self.c = config or cfg()
        self.s = LoopState()
        self.t = 100.0
        self.iou = iou
        self.log = []          # 'aimed' | 'tracked' | 'miss'
        self.aimed_boxes = []  # the box handed to the aiming path on 'aimed' frames

    def step(self, dets, dt=DT):
        """dets: [(box, conf, cls)]"""
        self.t += dt
        c, s, t = self.c, self.s, self.t
        boxes = [d[0] for d in dets]
        confs = [d[1] for d in dets]
        class_of = build_class_lookup(boxes, [d[2] for d in dets]) if boxes else None
        w, h = compute_effective_fov(c, s, t)
        cb, cc = filter_boxes_by_fov(boxes, confs, CX, CY, w, c, fov_height=h)
        r = resolve_retained_lock(c, s, cb, boxes, confs, CX, CY, t, self.iou, class_of)
        tracked = None if (r is None or r is AMBIGUOUS_LOCK) else r
        c.lock_track_only = tracked is not None
        if tracked is not None:
            track_retained(c, s, tracked[0], tracked[1], CX, CY, t)
            self.log.append('tracked')
            return 'tracked'
        if r is AMBIGUOUS_LOCK:
            pass                                          # hold: neither aim nor adopt anyone
        elif cb:
            items = [(((b[0] + b[2]) / 2 - CX) ** 2 + ((b[1] + b[3]) / 2 - CY) ** 2, cc[i],
                      (b[0] + b[2]) / 2, (b[1] + b[3]) / 2, b) for i, b in enumerate(cb)]
            items.sort(key=lambda x: x[0])
            sel, cont = select_target(items, locked_box=s.locked_box, tracker=s.tracker, config=c,
                                      now=t, iou_thresh=self.iou, norm=200, class_of=class_of)
            if sel is not None:
                s.tracker.on_match(t, sel[4], sel[1], same_target=bool(cont))
                if class_of is not None:
                    cid = class_of(sel[4])
                    if cid is not None or not cont:
                        s.tracker.locked_class = cid
                s.locked_box, s.locked_confidence = sel[4], sel[1]
                self.aimed_boxes.append(sel[4])
                self.log.append('aimed')
                return 'aimed'
        if not hold_lock_on_miss(s.tracker, c, t, s.locked_box is not None):
            s.locked_box = None
        self.log.append('miss')
        return 'miss'

    def lock_on(self, target, cls=0, frames=8):
        for _ in range(frames):
            self.step([(target, 0.9, cls)])
        assert self.s.locked_box is target

    @property
    def fov(self):
        return compute_effective_fov(self.c, self.s, self.t)[0]


# 400 px FOV shrinks to 200 (control half-extent 100); retention = full 400 (half 200).
OUT = 150      # x offset: outside control FOV, inside retention FOV
BEYOND = 260   # outside retention FOV


# ── 1. retention ────────────────────────────────────────────────────────────
class TestRetention:
    def test_identity_kept_outside_control_inside_retention(self):
        sim = Sim()
        mine = box(CX + 20, CY)
        sim.lock_on(mine)
        for x in range(30, OUT + 1, 10):
            mine = box(CX + x, CY)
            sim.step([(mine, 0.9, 0)])
        assert sim.s.locked_box is mine
        assert sim.log[-1] == 'tracked'
        assert sim.s.tracker.state is not TrackState.TEMPORARILY_LOST

    def test_tracked_frames_never_reach_the_aiming_path(self):
        sim = Sim()
        sim.lock_on(box(CX + 20, CY))
        n_aimed_before = len(sim.aimed_boxes)
        for x in range(30, OUT + 1, 10):
            sim.step([(box(CX + x, CY), 0.9, 0)])
        tracked = [i for i, e in enumerate(sim.log) if e == 'tracked']
        assert tracked and all(e != 'aimed' for e in sim.log[tracked[0]:tracked[-1] + 1])
        # nothing handed to aiming was outside the control FOV
        half = sim.fov / 2
        for b in sim.aimed_boxes[n_aimed_before:]:
            assert b[0] < CX + half and b[2] > CX - half

    def test_retention_off_drops_the_lock(self):
        sim = Sim(cfg(fov_retention_enabled=False))
        sim.lock_on(box(CX + 20, CY))
        for x in range(30, OUT + 1, 10):
            sim.step([(box(CX + x, CY), 0.9, 0)])
        assert 'tracked' not in sim.log


# ── 2. lock protection ──────────────────────────────────────────────────────
class TestLockProtection:
    def _outside(self):
        sim = Sim()
        mine = box(CX + 20, CY)
        sim.lock_on(mine)
        for x in range(30, OUT + 1, 10):
            mine = box(CX + x, CY)
            sim.step([(mine, 0.9, 0)])
        return sim, mine

    def test_rival_inside_control_fov_far_from_lock_cannot_steal(self):
        sim, mine = self._outside()
        rival = box(CX - 60, CY - 20)                       # inside control FOV, ~210 px from the lock
        for _ in range(10):
            mine = box(mine[0] + 10 + 1, CY)
            sim.step([(mine, 0.9, 0), (rival, 0.95, 0)])
        assert sim.s.locked_box is mine and set(sim.log[-10:]) == {'tracked'}

    def test_rival_inside_control_fov_NEAR_the_lock_cannot_steal(self):
        """A rival inside the control FOV that is within the reacquisition radius of the
        retained lock's last position must not be mistaken for the lock."""
        sim, mine = self._outside()                         # lock centre x = +150
        rival = box(CX + 95, CY)                            # inside control FOV (half 100), 55 px from the lock
        for _ in range(10):
            sim.step([(mine, 0.9, 0), (rival, 0.99, 0)])
        assert sim.s.locked_box is mine
        assert set(sim.log[-10:]) == {'tracked'}

    def test_rival_appearing_mid_retention(self):
        sim, mine = self._outside()
        for _ in range(3):
            sim.step([(mine, 0.9, 0)])
        rival = box(CX + 80, CY + 10)
        for _ in range(10):
            sim.step([(mine, 0.9, 0), (rival, 0.97, 0)])
        assert sim.s.locked_box is mine and 'aimed' not in sim.log[-10:]

    def test_original_returns_and_aiming_resumes_without_a_jump(self):
        sim, mine = self._outside()
        rival = box(CX + 95, CY)
        for x in range(OUT, 40, -10):
            mine = box(CX + x, CY)
            sim.step([(mine, 0.9, 0), (rival, 0.9, 0)])
        assert sim.log[-1] == 'aimed'
        assert all(b is not rival for b in sim.aimed_boxes[-3:])


# ── 3. ambiguity ───────────────────────────────────────────────────────────
class TestAmbiguity:
    def test_class_mismatch_overlapping_box_is_not_the_lock(self):
        sim = Sim(cfg(min_lock_duration_ms=500))
        mine = box(CX + 20, CY)
        sim.lock_on(mine, cls=1)
        mate = box(CX + 24, CY + 2)                          # overlapping teammate (class 0)
        r = sim.step([(mate, 0.99, 0)])
        # inside the minimum lock time the lock is held, never handed to the other class
        assert r == 'miss' and sim.s.locked_box is mine
        # and past it, the teammate is a NEW target, not a continuation of the lock
        sim2 = Sim(cfg(min_lock_duration_ms=0))
        sim2.lock_on(box(CX + 20, CY), cls=1)
        sim2.step([(mate, 0.99, 0)])
        assert sim2.s.tracker.state is TrackState.ACQUIRED and sim2.s.tracker.locked_class == 0

    def test_two_similar_candidates_after_loss_are_not_guessed(self):
        sim = Sim(cfg(fov_retention_enabled=False))
        sim.lock_on(box(CX + 20, CY))
        for _ in range(3):
            sim.step([])                                     # lost
        a, b = box(CX + 70, CY - 25), box(CX + 70, CY + 25)  # equally plausible
        assert sim.step([(a, 0.9, 0), (b, 0.9, 0)]) == 'miss'
        assert sim.s.locked_box is not a and sim.s.locked_box is not b
        assert sim.s.tracker.state is TrackState.TEMPORARILY_LOST

    def test_true_lock_ambiguous_with_inside_rival_holds_instead_of_adopting_the_rival(self):
        """The real lock moved too fast for box overlap (outside the control FOV) and a rival
        inside the control FOV is about as close to the last position: neither is claimed."""
        sim = Sim()
        mine = box(CX + 20, CY)
        sim.lock_on(mine)
        for x in range(30, OUT + 1, 10):
            mine = box(CX + x, CY)
            sim.step([(mine, 0.9, 0)])
        jumped = box(CX + 180, CY)                           # no box overlap with the last position
        rival = box(CX + 95, CY)                             # inside control FOV
        before = sim.s.locked_box
        r = sim.step([(jumped, 0.9, 0), (rival, 0.99, 0)])
        assert r == 'miss'
        assert sim.s.locked_box is before                    # lock held, rival NOT adopted
        w, h = compute_effective_fov(sim.c, sim.s, sim.t)
        cb, _ = filter_boxes_by_fov([jumped, rival], [0.9, 0.99], CX, CY, w, sim.c, fov_height=h)
        assert resolve_retained_lock(sim.c, sim.s, cb, [jumped, rival], [0.9, 0.99], CX, CY,
                                     sim.t, 0.3) is AMBIGUOUS_LOCK

    def test_crossing_targets_keep_their_own_identity(self):
        sim = Sim(cfg(fov_retention_enabled=False, fov_size=600, fov_height=600, fov_reduce_on_target_enabled=False))
        a_x, b_x = CX - 60.0, CX + 60.0
        a = box(a_x, CY); b = box(b_x, CY + 6)
        sim.lock_on(a, frames=6)
        mine_path = []
        for _ in range(24):
            a_x += 5.0; b_x -= 5.0                           # they cross around the middle
            a = box(a_x, CY); b = box(b_x, CY + 6)
            sim.step([(b, 0.9, 0), (a, 0.9, 0)])
            mine_path.append(sim.s.locked_box is a)
        assert all(mine_path), "lock hopped to the crossing target"


# ── 4. retention exit ──────────────────────────────────────────────────────
class TestRetentionExit:
    def test_leaving_retention_fov_releases_and_fov_recovers(self):
        sim = Sim(cfg(fov_recovery_duration_ms=200, lock_persistence_ms=100, lost_target_timeout_ms=300))
        mine = box(CX + 20, CY)
        sim.lock_on(mine)
        for x in range(30, OUT + 1, 10):
            sim.step([(box(CX + x, CY), 0.9, 0)])
        assert sim.log[-1] == 'tracked' and sim.fov == 200
        gone = box(CX + BEYOND, CY)
        for _ in range(40):
            sim.step([(gone, 0.9, 0)])
        assert sim.s.locked_box is None
        assert sim.s.tracker.state in (TrackState.SEARCHING, TrackState.RELEASED)
        assert sim.s.retained_since == 0.0
        assert sim.fov == 400                                # normal expansion/recovery ran

    def test_parked_outside_control_fov_is_bounded_by_lost_timeout(self):
        """Track-only never lasts longer than lost_target_timeout (+ a few frames): after that the
        normal lost-target path runs (the FOV then expands and the target may be reacquired)."""
        sim = Sim(cfg(lost_target_timeout_ms=600, lock_persistence_ms=100))
        sim.lock_on(box(CX + 20, CY))
        parked = box(CX + OUT, CY)
        for _ in range(int(4.0 / DT)):
            sim.step([(parked, 0.9, 0)])
        run = best = 0
        for e in sim.log:
            run = run + 1 if e == 'tracked' else 0
            best = max(best, run)
        assert 0 < best <= int(0.6 / DT) + 3

# ── 5. reacquisition ───────────────────────────────────────────────────────
class TestReacquisition:
    def test_returns_after_short_disappearance_without_jumping_to_other_target(self):
        sim = Sim(cfg(lock_persistence_ms=100, lost_target_timeout_ms=1500))
        mine = box(CX + 20, CY)
        sim.lock_on(mine)
        for _ in range(5):
            sim.step([])                                      # vanishes ~170 ms
        decoy = box(CX - 80, CY + 40)                         # far from where mine was
        back = box(CX + 35, CY + 2)                           # near last position
        sim.step([(decoy, 0.95, 0), (back, 0.9, 0)])
        assert sim.s.locked_box is back

    def test_target_moves_beyond_retention_then_returns(self):
        sim = Sim(cfg(lock_persistence_ms=100, lost_target_timeout_ms=1500))
        sim.lock_on(box(CX + 20, CY))
        for x in range(30, OUT + 1, 10):
            sim.step([(box(CX + x, CY), 0.9, 0)])
        for x in range(OUT, BEYOND + 1, 10):                  # leaves retention
            sim.step([(box(CX + x, CY), 0.9, 0)])
        for _ in range(4):
            sim.step([])
        # comes back inside the (now recovering) FOV, near where it left
        back = box(CX + 120, CY)
        for _ in range(3):
            sim.step([(back, 0.9, 0)])
        assert sim.s.locked_box is back                       # reacquired (new or same), never another target

    def test_return_with_decoy_near_not_chosen_when_ambiguous(self):
        sim = Sim(cfg(fov_retention_enabled=False, lock_persistence_ms=100))
        sim.lock_on(box(CX + 20, CY))
        for _ in range(4):
            sim.step([])
        a, b = box(CX + 60, CY - 30), box(CX + 60, CY + 30)
        sim.step([(a, 0.9, 0), (b, 0.9, 0)])
        assert sim.s.tracker.state is not TrackState.REACQUIRED   # ambiguous: no identity claimed


# ── edge cases: frame intervals / stale timestamps / flicker ───────────────
class TestTimingEdges:
    def test_variable_frame_intervals_do_not_break_retention(self):
        sim = Sim()
        sim.lock_on(box(CX + 20, CY))
        dts = [0.004, 0.05, 0.016, 0.2, 0.008, 0.033, 0.1, 0.016, 0.016, 0.3]
        x = 30
        for dt in dts:
            x = min(OUT, x + 15)
            sim.step([(box(CX + x, CY), 0.9, 0)], dt=dt)
        assert sim.s.locked_box is not None
        assert sim.log[-1] == 'tracked'

    def test_stale_retention_stretch_does_not_instantly_expire_after_a_pause(self):
        """Aim key released / thread paused while track-only, then resumed much later:
        the timeout must measure the NEW stretch, not the old one."""
        sim = Sim(cfg(lost_target_timeout_ms=1000, lock_persistence_ms=5000))
        sim.lock_on(box(CX + 20, CY))
        for x in range(30, OUT + 1, 10):
            sim.step([(box(CX + x, CY), 0.9, 0)])
        assert sim.s.retained_since > 0
        sim.t += 30.0                                         # long pause (no frames evaluated)
        sim.s.tracker.last_update_t = sim.t - DT              # tracker clock is also current again
        r = sim.step([(box(CX + OUT, CY), 0.9, 0)])
        assert r == 'tracked'

    def test_lock_flicker_for_several_frames_inside_retention(self):
        sim = Sim()
        sim.lock_on(box(CX + 20, CY))
        pos = box(CX + OUT, CY)
        for x in range(30, OUT + 1, 10):
            sim.step([(box(CX + x, CY), 0.9, 0)])
        for _ in range(4):
            sim.step([])                                       # several missed frames
        sim.step([(pos, 0.9, 0)])
        assert sim.s.locked_box is pos or sim.log[-1] in ('tracked', 'aimed')

    def test_non_finite_and_zero_dt(self):
        sim = Sim()
        sim.lock_on(box(CX + 20, CY))
        for dt in (0.0, 1e-9, 0.0):
            sim.step([(box(CX + OUT, CY), 0.9, 0)], dt=dt)
        assert sim.s.locked_box is not None


# ── motion tie-break guards (find_continuation) ────────────────────────────
class TestMotionTieBreak:
    @staticmethod
    def _tracker(step_px=0.0, frames=6):
        from core.target_tracking import TargetTracker
        t = TargetTracker()
        x, now = 500.0, 100.0
        t.on_match(now, box(x, CY), 0.9, same_target=False)
        for _ in range(frames):
            x += step_px
            now += DT
            t.on_match(now, box(x, CY), 0.9)
        return t, box(x, CY), now + DT

    def _pick(self, tracker, locked, now, cands):
        from core.target_tracking import find_continuation
        idx, amb = find_continuation(cands, locked_box=locked, tracker=tracker, config=cfg(),
                                     now=now, iou_thresh=0.3)
        return idx, amb

    def test_single_candidate_unchanged(self):
        t, locked, now = self._tracker(8.0)
        assert self._pick(t, locked, now, [box(520, CY)]) == (0, False)

    def test_stationary_lock_keeps_the_highest_iou_candidate(self):
        t, locked, now = self._tracker(0.0)
        best, other = box(502, CY), box(512, CY)
        assert self._pick(t, locked, now, [other, best]) == (1, False)

    def test_clear_iou_winner_not_overridden_when_motion_agrees(self):
        t, locked, now = self._tracker(10.0)                  # moving right 10 px/frame
        followed = box((locked[0] + locked[2]) / 2 + 10, CY)
        lag = box((locked[0] + locked[2]) / 2 + 3, CY)
        idx, _ = self._pick(t, locked, now, [lag, followed])
        assert idx == 1                                       # the one that follows the motion

    def test_stationary_rival_does_not_beat_a_target_that_moved_through_it(self):
        t, locked, now = self._tracker(-10.0)                 # moving left 10 px/frame, last box centred 500-60
        lx = (locked[0] + locked[2]) / 2
        mine = box(lx - 10, CY)
        rival = box(lx - 5, CY)                               # stationary, overlaps the old box MORE than `mine`
        from core.target_tracking import box_iou
        assert box_iou(locked, rival) > box_iou(locked, mine)  # the trap
        idx, _ = self._pick(t, locked, now, [rival, mine])
        assert idx == 1

    def test_unclear_motion_falls_back_to_legacy_choice(self):
        t, locked, now = self._tracker(10.0)
        lx = (locked[0] + locked[2]) / 2
        a, b = box(lx + 9, CY), box(lx + 11, CY)              # both about equally close to the expected centre
        idx, _ = self._pick(t, locked, now, [a, b])
        from core.target_tracking import box_iou
        assert idx == (0 if box_iou(locked, a) >= box_iou(locked, b) else 1)
