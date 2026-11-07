"""Target-lock state machine and the smoothed tracking signals that drive the
adaptive/dynamic FOV shrinking modes.

Pure Python (no numpy / Qt / win32), so it is importable and unit-testable
anywhere.  Everything here is driven by elapsed time (``now`` is a
monotonic-ish float in seconds supplied by the caller), never by frame
counts, so behaviour is the same at 30 or 300 inference FPS.

States (``TrackState``)::

    SEARCHING          no target, nothing remembered
    ACQUIRED           a brand-new target was just locked
    TRACKING           lock is settled and being matched every frame
    TEMPORARILY_LOST   lock missed detections; identity remembered for reacquisition
    REACQUIRED         the remembered target was found again
    RELEASED           lock given up (timeout / explicit release); becomes SEARCHING

Transitions::

    SEARCHING/RELEASED --match(new)-->  ACQUIRED
    ACQUIRED/REACQUIRED --settle time-> TRACKING
    ACQUIRED/TRACKING/REACQUIRED --miss--> TEMPORARILY_LOST
    TEMPORARILY_LOST --match(same)-->   REACQUIRED
    TEMPORARILY_LOST --timeout------->  RELEASED -> SEARCHING
    any --match(new target)----------> ACQUIRED   (switch)
"""
from __future__ import annotations

import math
from enum import Enum
from typing import List, Optional, Sequence, Tuple


class TrackState(str, Enum):
    SEARCHING = "searching"
    ACQUIRED = "acquired"
    TRACKING = "tracking"
    TEMPORARILY_LOST = "temporarily_lost"
    REACQUIRED = "reacquired"
    RELEASED = "released"


# A freshly acquired / reacquired lock counts as settled after this long.
_SETTLE_S = 0.10
# Longest gap folded into one EMA step (a stall must not whip the filters).
_MAX_EMA_DT_S = 0.5
# How far ahead (seconds) the last velocity may extrapolate the lost target.
_MAX_EXTRAPOLATE_S = 0.5
# Speeds below this (px/s) give no meaningful direction for the turn signal.
_MIN_TURN_SPEED = 30.0

# EMA time constants (seconds).
_TAU_CONF = 0.30
_TAU_HIT = 0.40
_TAU_STAB = 0.40
_TAU_SPEED = 0.25
_TAU_TURN = 0.30
_TAU_EDGE = 0.12


# Reacquisition by radius needs the best candidate to be closer than the runner-up
# by this much (fraction of the radius, at least the pixel floor).
_REACQUIRE_AMBIGUITY_FRAC = 0.2
_REACQUIRE_MIN_MARGIN_PX = 8.0

# When several detections overlap the lock, the motion-consistent one beats the
# highest-IoU one only if it is clearly closer to where the lock should be.
_MAX_EXPECT_DT_S = 0.1
_TIEBREAK_MIN_PX = 3.0
_TIEBREAK_BOX_FRAC = 0.2


def _ema(prev: float, sample: float, dt: float, tau: float) -> float:
    """Time-constant EMA: frame-rate independent smoothing."""
    if not (dt > 0.0) or not math.isfinite(dt) or not math.isfinite(sample) or tau <= 0.0:
        return prev        # no time passed / bad input: keep the estimate
    alpha = 1.0 - math.exp(-min(dt, _MAX_EMA_DT_S) / tau)
    return prev + (sample - prev) * alpha


def clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    union = ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / union if union > 0.0 else 0.0


def box_center(box: Sequence[float]) -> Tuple[float, float]:
    return (box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5


class TargetTracker:
    """Lock lifecycle + smoothed quality signals for one tracked target."""

    def __init__(self) -> None:
        self.reset()

    # ── lifecycle ────────────────────────────────────────────────────────
    def reset(self) -> None:
        self.state: TrackState = TrackState.SEARCHING
        self.state_since: float = 0.0
        self.acquired_t: float = 0.0
        self.last_seen_t: float = 0.0
        self.lost_since: float = 0.0
        self.last_update_t: float = 0.0
        self.last_box: Optional[List[float]] = None
        self.last_center: Optional[Tuple[float, float]] = None
        self.last_conf: float = 0.0
        self.vx: float = 0.0
        self.vy: float = 0.0
        # Smoothed signals (see fov mode code in ai_loop_utils).
        self.conf_ema: float = 0.0
        self.hit_ema: float = 0.0
        self.stab_ema: float = 0.0
        self.speed_ema: float = 0.0
        self.turn_ema: float = 0.0
        self.edge_ratio: float = 0.0
        self.locked_class: Optional[int] = None   # class id of the locked target, when the model reports one
        self.inst_vx: float = 0.0   # un-smoothed velocity of the last frame-to-frame step (px/s)
        self.inst_vy: float = 0.0

    def _set_state(self, state: TrackState, now: float) -> None:
        if state is not self.state:
            self.state = state
            self.state_since = now

    def _dt(self, now: float) -> float:
        return max(0.0, now - self.last_update_t) if self.last_update_t > 0.0 else 0.0

    def on_match(
        self,
        now: float,
        box: Sequence[float],
        conf: float,
        edge_ratio: float = 0.0,
        same_target: bool = True,
    ) -> None:
        """The lock was matched to ``box`` this frame.

        ``same_target`` False starts a new lock (ACQUIRED, signals restart
        pessimistically so FOV contraction has to be earned again).
        """
        if self.state is TrackState.RELEASED:
            self._set_state(TrackState.SEARCHING, now)
        cx, cy = box_center(box)
        dt = self._dt(now)
        was_lost = self.state is TrackState.TEMPORARILY_LOST
        has_memory = self.last_center is not None

        if not same_target or not has_memory or self.state is TrackState.SEARCHING:
            self.acquired_t = now
            self.conf_ema = float(conf)
            self.hit_ema = 0.5
            self.stab_ema = 0.5
            self.speed_ema = 0.0
            self.turn_ema = 0.0
            self.vx = self.vy = 0.0
            self.inst_vx = self.inst_vy = 0.0
            self.edge_ratio = float(edge_ratio)
            self.locked_class = None
            self._set_state(TrackState.ACQUIRED, now)
        else:
            prev_box = self.last_box
            iou = box_iou(prev_box, box) if prev_box is not None else 0.0
            self.conf_ema = _ema(self.conf_ema, float(conf), dt, _TAU_CONF)
            self.hit_ema = _ema(self.hit_ema, 1.0, dt, _TAU_HIT)
            self.stab_ema = _ema(self.stab_ema, iou, dt, _TAU_STAB)
            self.edge_ratio = _ema(self.edge_ratio, float(edge_ratio), dt, _TAU_EDGE)
            if dt > 1e-3 and self.last_center is not None and not was_lost:
                nvx = (cx - self.last_center[0]) / dt
                nvy = (cy - self.last_center[1]) / dt
                self.inst_vx, self.inst_vy = nvx, nvy
                speed = math.hypot(nvx, nvy)
                old_speed = math.hypot(self.vx, self.vy)
                if speed > _MIN_TURN_SPEED and old_speed > _MIN_TURN_SPEED:
                    cos = (nvx * self.vx + nvy * self.vy) / (speed * old_speed)
                    turn = (1.0 - max(-1.0, min(1.0, cos))) * 0.5
                    self.turn_ema = _ema(self.turn_ema, turn, dt, _TAU_TURN)
                else:
                    self.turn_ema = _ema(self.turn_ema, 0.0, dt, _TAU_TURN)
                self.speed_ema = _ema(self.speed_ema, speed, dt, _TAU_SPEED)
                # Velocity itself is lightly smoothed so one noisy box does
                # not fling the reacquisition prediction.
                self.vx = _ema(self.vx, nvx, dt, _TAU_SPEED)
                self.vy = _ema(self.vy, nvy, dt, _TAU_SPEED)
            if was_lost:
                self._set_state(TrackState.REACQUIRED, now)
            elif (self.state in (TrackState.ACQUIRED, TrackState.REACQUIRED)
                  and now - self.state_since >= _SETTLE_S):
                self._set_state(TrackState.TRACKING, now)

        self.last_box = [float(v) for v in box[:4]]
        self.last_center = (cx, cy)
        self.last_conf = float(conf)
        self.last_seen_t = now
        self.lost_since = 0.0
        self.last_update_t = now

    def on_miss(self, now: float) -> None:
        """No valid detection of the locked target this frame."""
        if self.state in (TrackState.SEARCHING, TrackState.RELEASED):
            self._set_state(TrackState.SEARCHING, now)
            return
        dt = self._dt(now)
        self.hit_ema = _ema(self.hit_ema, 0.0, dt, _TAU_HIT)
        self.stab_ema = _ema(self.stab_ema, 0.0, dt, _TAU_STAB)
        # Nothing was detected, so the confidence signal must not stay at its last
        # (possibly high) value through a long miss; it decays like the others.
        self.conf_ema = _ema(self.conf_ema, 0.0, dt, _TAU_CONF)
        if self.state is not TrackState.TEMPORARILY_LOST:
            self.lost_since = now
            self._set_state(TrackState.TEMPORARILY_LOST, now)
        self.last_update_t = now

    def release(self, now: float) -> None:
        """Give the lock up and forget the target."""
        if self.state in (TrackState.SEARCHING, TrackState.RELEASED):
            return
        self.reset()
        self.state = TrackState.RELEASED
        self.state_since = now

    def expire(self, now: float, timeout_s: float) -> bool:
        """Release a lost target that has been unseen for ``timeout_s``."""
        if self.state is TrackState.RELEASED:
            self._set_state(TrackState.SEARCHING, now)
        if self.state is TrackState.TEMPORARILY_LOST and self.lost_duration(now) >= timeout_s:
            self.release(now)
            return True
        return False

    # ── queries ──────────────────────────────────────────────────────────
    @property
    def locked(self) -> bool:
        return self.state in (TrackState.ACQUIRED, TrackState.TRACKING,
                              TrackState.REACQUIRED, TrackState.TEMPORARILY_LOST)

    def lost_duration(self, now: float) -> float:
        if self.state is not TrackState.TEMPORARILY_LOST:
            return 0.0
        return max(0.0, now - self.lost_since)

    def lock_age(self, now: float) -> float:
        """Seconds since the current target was first acquired (0 if none)."""
        if not self.locked or self.acquired_t <= 0.0:
            return 0.0
        return max(0.0, now - self.acquired_t)

    def has_memory(self, now: float, timeout_s: float) -> bool:
        """True while a remembered target can still be reacquired."""
        if self.last_center is None or not self.locked:
            return False
        return self.lost_duration(now) < timeout_s

    def expected_center(self, now: float) -> Optional[Tuple[float, float]]:
        """Where a continuing target should be this frame: the last centre advanced
        by its LAST step's velocity (un-smoothed, so a recent reversal counts at once,
        unlike the EMA velocity) for at most 0.1 s."""
        if self.last_center is None:
            return None
        dt = min(max(0.0, now - self.last_update_t), _MAX_EXPECT_DT_S) if self.last_update_t > 0.0 else 0.0
        return self.last_center[0] + self.inst_vx * dt, self.last_center[1] + self.inst_vy * dt

    def predicted_center(self, now: float) -> Optional[Tuple[float, float]]:
        """Last known position, advanced by the last velocity while lost."""
        if self.last_center is None:
            return None
        gap = min(self.lost_duration(now), _MAX_EXTRAPOLATE_S)
        return self.last_center[0] + self.vx * gap, self.last_center[1] + self.vy * gap

    def within_reacquire_radius(self, box: Sequence[float], now: float, radius_px: float) -> Optional[float]:
        """Distance from ``box`` to the remembered target if inside the radius."""
        if self.last_center is None or radius_px <= 0.0:
            return None
        cx, cy = box_center(box)
        best = math.hypot(cx - self.last_center[0], cy - self.last_center[1])
        pred = self.predicted_center(now)
        if pred is not None:
            best = min(best, math.hypot(cx - pred[0], cy - pred[1]))
        return best if best <= radius_px else None


# ── config-derived helpers (getattr fallbacks keep stubs/old configs working) ─

def acquisition_confidence(config) -> float:
    return float(getattr(config, 'min_confidence', 0.8))


def tracking_confidence(config) -> float:
    """Confidence needed to *keep* a lock. 0 / unset / above acquisition means
    "same as acquisition" (the pre-split single-threshold behaviour)."""
    acq = acquisition_confidence(config)
    if not getattr(config, 'sticky_lock_enabled', False):
        return acq          # no lock to keep -> a single threshold
    tc = float(getattr(config, 'tracking_confidence', 0.0) or 0.0)
    return acq if tc <= 0.0 or tc > acq else tc


def detection_confidence_floor(config) -> float:
    """Threshold handed to the detector so lock-keeping detections survive."""
    return min(acquisition_confidence(config), tracking_confidence(config))


def persistence_s(config) -> float:
    """Aim-hold grace period, never longer than the lost-target timeout."""
    p = max(0.0, float(getattr(config, 'lock_persistence_ms', 250)) / 1000.0)
    return min(p, lost_timeout_s(config))


def lost_timeout_s(config) -> float:
    return max(0.0, float(getattr(config, 'lost_target_timeout_ms', 1500)) / 1000.0)


def filter_by_acquisition_confidence(
    boxes, confidences, class_ids, config, tracker: TargetTracker, locked_box, now: float,
):
    """Drop detections below the acquisition threshold unless they can be a
    continuation of the current/remembered lock (overlap or inside the
    reacquisition radius).  No-op (and no allocation) when tracking
    confidence equals acquisition confidence."""
    acq = acquisition_confidence(config)
    if tracking_confidence(config) >= acq or not boxes:
        return boxes, confidences, class_ids
    radius = float(getattr(config, 'reacquire_radius_px', 150))
    memory = locked_box is not None or tracker.has_memory(now, lost_timeout_s(config))
    keep = []
    for i, box in enumerate(boxes):
        c = confidences[i] if i < len(confidences) else 1.0
        if c >= acq:
            keep.append(i)
        elif memory and ((locked_box is not None and box_iou(locked_box, box) > 0.0)
                         or tracker.within_reacquire_radius(box, now, radius) is not None):
            keep.append(i)
    if len(keep) == len(boxes):
        return boxes, confidences, class_ids
    return ([boxes[i] for i in keep],
            [confidences[i] for i in keep] if confidences else confidences,
            [class_ids[i] for i in keep] if class_ids is not None and len(class_ids) == len(boxes) else class_ids)


def edge_ratio(box: Sequence[float], crosshair_x: float, crosshair_y: float,
               fov_w: float, fov_h: float, elliptical: bool) -> float:
    """Target centre's distance from the FOV centre as a fraction of the FOV
    half-extent (0 = centred, 1 = on the boundary)."""
    cx, cy = box_center(box)
    nx = abs(cx - crosshair_x) / max(1.0, fov_w * 0.5)
    ny = abs(cy - crosshair_y) / max(1.0, fov_h * 0.5)
    return math.hypot(nx, ny) if elliptical else max(nx, ny)


def candidate_score(item, mode: str, conf_weight: float, norm: float) -> float:
    """0..1, higher is better, mirroring the target-priority sort key.

    ``item`` is process_aiming's ``(distance_sq, conf, tx, ty, box)`` tuple.
    ``norm`` (pixels) is the distance at which the proximity score hits 0.
    """
    if mode == 'confidence':
        return clamp01(float(item[1]))
    key = item[0] * (1.0 - item[1] * conf_weight) if mode == 'composite' else item[0]
    return 1.0 - min(1.0, math.sqrt(max(0.0, key)) / max(1.0, norm))


def _motion_consistent(boxes, strong, locked_box, tracker: TargetTracker, now: float) -> int:
    """Among detections that all overlap the lock enough, keep the highest-IoU one
    unless another is clearly closer to the lock's expected position (a stationary
    or crossing rival can out-overlap the real target's old box when the target
    moves more than a box width per frame)."""
    best_i = strong[0][1]
    exp = tracker.expected_center(now)
    if exp is None:
        return best_i
    def dist(i):
        cx, cy = box_center(boxes[i])
        return math.hypot(cx - exp[0], cy - exp[1])
    near_i = min((i for _, i in strong), key=dist)
    if near_i == best_i:
        return best_i
    width = max(1.0, locked_box[2] - locked_box[0])
    if dist(best_i) - dist(near_i) >= max(_TIEBREAK_MIN_PX, _TIEBREAK_BOX_FRAC * width):
        return near_i
    return best_i


def find_continuation(boxes, *, locked_box, tracker: TargetTracker, config, now: float,
                      iou_thresh: float, class_of=None):
    """Which of ``boxes`` (if any) is the current/remembered lock?

    Returns ``(index | None, ambiguous)``.  Box overlap with ``locked_box``
    (>= ``iou_thresh``) wins; otherwise, while the tracker still remembers the
    target, the nearest box inside the reacquisition radius of its last /
    velocity-extrapolated position -- unless the runner-up is about as close
    (``ambiguous``: don't guess).  ``class_of(box)`` (optional) gives a box's
    class id; a box whose class differs from the locked target's is never a
    continuation.  Pure and allocation-light: runs once or twice per frame.
    """
    lc = tracker.locked_class
    if class_of is not None and lc is not None:
        def ok(box):
            c = class_of(box)
            return c is None or c == lc
    else:
        ok = None

    idx = None
    if locked_box is not None:
        cands = []
        for i, box in enumerate(boxes):
            if ok is not None and not ok(box):
                continue
            iou = box_iou(locked_box, box)
            if iou > 0.0:
                cands.append((iou, i))
        if cands:
            cands.sort(reverse=True)
            if cands[0][0] >= iou_thresh:
                idx = cands[0][1]
                strong = [c for c in cands if c[0] >= iou_thresh]
                if len(strong) > 1:          # crossing / overlapping detections: use motion too
                    idx = _motion_consistent(boxes, strong, locked_box, tracker, now)
    if idx is not None or not tracker.has_memory(now, lost_timeout_s(config)):
        return idx, False

    radius = float(getattr(config, 'reacquire_radius_px', 150))
    dists = []
    for i, box in enumerate(boxes):
        if ok is not None and not ok(box):
            continue
        d = tracker.within_reacquire_radius(box, now, radius)
        if d is not None:
            dists.append((d, i))
    if not dists:
        return None, False
    dists.sort()
    margin = max(_REACQUIRE_MIN_MARGIN_PX, _REACQUIRE_AMBIGUITY_FRAC * radius)
    if len(dists) > 1 and dists[1][0] - dists[0][0] < margin:
        return None, True
    return dists[0][1], False


def build_class_lookup(boxes, class_ids):
    """``box -> class id`` (identity based; the box objects survive the FOV
    filters) or None when the model reports no per-box classes."""
    if class_ids is None or len(class_ids) != len(boxes) or not boxes:
        return None
    table = {id(b): c for b, c in zip(boxes, class_ids)}
    return lambda box: table.get(id(box))


def select_target(valid_targets, *, locked_box, tracker: TargetTracker, config,
                  now: float, iou_thresh: float, norm: float, class_of=None):
    """Sticky-lock target choice.  ``valid_targets`` is already sorted by
    priority (best first).  Returns ``(item, continued)``:

    * ``item`` None  -> no acceptable target this frame; hold the lock.
    * ``continued``  -> the pick is the same target as the current/remembered
      lock (matched by box overlap, else by the reacquisition radius around its
      last known / velocity-extrapolated position).

    A rival only displaces a continuing lock when the lock is older than
    ``min_lock_duration_ms`` AND outscores it by ``target_switch_threshold``.
    A brand-new target needs the acquisition confidence; the lock itself is
    kept down to the tracking confidence.
    """
    idx, ambiguous = find_continuation(
        [it[4] for it in valid_targets], locked_box=locked_box, tracker=tracker,
        config=config, now=now, iou_thresh=iou_thresh, class_of=class_of)
    if ambiguous:
        return None, False
    cont = valid_targets[idx] if idx is not None else None

    acq = acquisition_confidence(config)
    split = tracking_confidence(config) < acq
    winner = valid_targets[0] if not split else next(
        (it for it in valid_targets if it[1] >= acq), None)
    min_lock_s = max(0.0, float(getattr(config, 'min_lock_duration_ms', 0)) / 1000.0)

    if cont is not None:
        if winner is not None and winner is not cont:
            mode = str(getattr(config, 'target_priority_mode', 'distance'))
            weight = float(getattr(config, 'target_priority_confidence_weight', 0.5))
            advantage = candidate_score(winner, mode, weight, norm) - candidate_score(cont, mode, weight, norm)
            if (tracker.lock_age(now) >= min_lock_s
                    and advantage >= float(getattr(config, 'target_switch_threshold', 0.0))):
                return winner, False
        return cont, True

    if winner is None:
        return None, False
    if locked_box is not None and tracker.locked and tracker.lock_age(now) < min_lock_s:
        return None, False      # locked target momentarily missing: don't hop inside the minimum lock time
    return winner, False


def hold_lock_on_miss(tracker: TargetTracker, config, now: float, has_lock: bool) -> bool:
    """Per-frame bookkeeping for a frame with no accepted target.

    Returns True while the aim state (PID, smoothing, prediction) should be
    held: the lock exists and has been missing for less than
    ``lock_persistence_ms``.  Once that grace period is over the caller resets
    its aim state but the *tracker* keeps remembering the target, so a
    reacquisition inside ``reacquire_radius_px`` is still recognised, until
    ``lost_target_timeout_ms`` releases it for good.  Sticky lock off means no
    lock memory at all (the original behaviour).  Everything is measured in
    elapsed time, so it is independent of the inference frame rate.
    """
    if not getattr(config, 'sticky_lock_enabled', False):
        tracker.release(now)
        return False
    tracker.on_miss(now)
    holding = has_lock and tracker.lost_duration(now) < persistence_s(config)
    if tracker.expire(now, lost_timeout_s(config)):
        return False
    return holding
