from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class LoopState:
    """Status management for AI loop"""

    last_pid_update: float = 0.0

    last_method_check_time: float = 0.0
    cached_mouse_move_method: str = 'mouse_event'

    pid_check_interval: float = 1.0
    method_check_interval: float = 2.0

    aiming_start_time: float = 0.0

    # Sticky target lock — hold the selected target across short detection gaps.
    # locked_box/locked_confidence are updated on every successful selection
    # (regardless of sticky_lock_enabled) so callers can read back what aiming
    # actually picked this frame; only *acting* on a hold across a gap is
    # gated by sticky_lock_enabled elsewhere.
    locked_box: Optional[List[float]] = field(default=None)
    locked_confidence: float = 0.0
    no_detection_frames: int = 0

    # FOV-reduce-on-target — the timestamp the current lock's shrink ramp
    # started at; 0.0 means "no ramp running right now." Set once, on the
    # None→non-None edge of locked_box, not every frame the target stays
    # locked — see ai_loop.py's fov filtering call site for why that
    # distinction matters (same class of continuous-reset bug already fixed
    # once for the MAKCU disengage delay: setting it every frame would pin
    # elapsed-time-since-acquisition near zero forever and the ramp could
    # never progress). Left non-zero for the entire lock once armed — the
    # FOV ramps from full size down to fov_min_size_pct% over
    # fov_min_size_duration seconds, then simply stays at the minimum
    # (compute_effective_fov() clamps ramp progress at 1.0) for the rest of
    # the lock; only losing the lock resets this back to 0.0 so the next
    # acquisition starts its own fresh ramp from full size.
    fov_reduce_since: float = 0.0

    # Y-reduce velocity gate — track target Y position across frames to estimate vy.
    aim_y_last_target_y: float = 0.0
    aim_y_last_target_t: float = 0.0

    # Camera motion compensation — global scene shift estimated by phase correlation in
    # _preprocess_worker; written there, read in process_aiming to cancel shake-induced error.
    cam_shift_x: float = 0.0
    cam_shift_y: float = 0.0

    # Running integral of cam_shift_x/y — a free-running "camera position
    # relative to an arbitrary reference frame," accumulated once per
    # _preprocess_worker tick alongside cam_shift_x/y itself. process_aiming
    # subtracts this from the raw detected target position before handing it
    # to the velocity predictor/Kalman filter, so their frame-to-frame deltas
    # reflect true target motion rather than apparent motion caused by the
    # aimbot's own correction or camera shake/recoil; the same drift is added
    # back before the PID error is computed (which must stay in real, current
    # screen coordinates). Its absolute value is meaningless — only frame-to-
    # frame consistency matters — so no reset is needed on target loss, only
    # when cam_motion_comp_enabled itself is off (mirrors cam_shift_x/y).
    cam_drift_x: float = 0.0
    cam_drift_y: float = 0.0

    # Timestamp the current lock was acquired — set once on a genuinely new
    # target (the same "is this a continuation of the previous pick"
    # _box_iou() check already used for aim_y_last_target_t below), not every
    # frame the target stays locked. Drives process_aiming()'s brief
    # acquisition-phase guard: for _ACQUISITION_GUARD_S after this timestamp,
    # the predictor/Kalman are kept re-bootstrapping instead of accumulating
    # normally, so the crosshair's own snap onto a fresh target is never
    # misread as the target moving. Unconditional (not gated by any feature
    # flag) — unlike fov_reduce_since, which resets to 0 whenever its own
    # feature is off, this always tracks the true lock-acquisition time.
    lock_acquired_t: float = 0.0

    # Elapsed-time tracking for KalmanFilter2D's per-update dt (fixes the
    # filter's own dt being hardcoded at construction and never revisited) —
    # 0.0 means "no previous update to diff against," matching aim_y_last_target_t's
    # own "0.0 = not yet primed" convention. Reset alongside kalman resets
    # (target loss, kalman_enabled toggled off, acquisition-guard re-bootstrap).
    kalman_last_t: float = 0.0

    # Sub-pixel carry — accumulates the fractional remainder that integer truncation
    # discards each frame so micro-corrections are never silently lost and the
    # crosshair converges exactly onto the aim point. Applies to all mouse backends.
    aim_carry_x: float = 0.0
    aim_carry_y: float = 0.0
