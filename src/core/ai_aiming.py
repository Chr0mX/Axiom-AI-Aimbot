from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING, List, Optional, Tuple

from win_utils import send_mouse_move

from .ai_loop_state import LoopState
from .ai_loop_utils import anchor_on_match, anchor_ref_box
from .humanization import apply_humanization
from .inference import PIDController
from .kalman_filter import KalmanFilter2D
from .aim_paths import adaptive_lead_s, ema_step
from .anchor_prediction import auto_aim_point, predict_aim_point
from .target_predictor import VelocityPredictor
from .target_tracking import box_iou, edge_ratio, lost_timeout_s, select_target

if TYPE_CHECKING:
    from .config import Config

# Module-level singleton — shared across process_aiming calls.
_predictor: Optional[VelocityPredictor] = None
_kalman: Optional[KalmanFilter2D] = None


def _get_kalman(config: Config) -> KalmanFilter2D:
    """Return (and lazily create/reconfigure) the module-level KalmanFilter2D."""
    global _kalman
    pn = float(getattr(config, 'kalman_process_noise', 0.01))
    mn = float(getattr(config, 'kalman_measurement_noise', 0.1))
    if _kalman is None:
        _kalman = KalmanFilter2D(process_noise=pn, measurement_noise=mn)
    else:
        _kalman.reconfigure(pn, mn)
    return _kalman


def _get_predictor(config: Config) -> VelocityPredictor:
    """Return (and lazily create/reconfigure) the module-level VelocityPredictor."""
    global _predictor
    history_len = int(getattr(config, 'prediction_history_len', 3))
    max_vel = float(getattr(config, 'prediction_max_velocity', 1200.0))
    if _predictor is None:
        _predictor = VelocityPredictor(history_len=history_len, max_velocity_px_per_s=max_vel)
    else:
        _predictor._max_velocity = max_vel
        _predictor._history = type(_predictor._history)(
            _predictor._history, maxlen=history_len
        )
    _predictor.lsq_fit = bool(getattr(config, 'prediction_lsq_fit', False))
    _predictor.velocity_deadzone = float(getattr(config, 'prediction_velocity_deadzone_px_s', 0.0) or 0.0)
    return _predictor


def _reset_unused_predictors(method: str) -> None:
    if method != "velocity" and _predictor is not None:
        _predictor.reset()


# How often the aim-lead anchor track re-reads the shared anchor_* settings (seconds).
_ANCHOR_AIM_PARAMS_REFRESH_S = 0.25


# Strafe response: how long after a detected reversal the output side stays "fast" (Aim Step Smoothing bypassed).
_REVERSAL_OUTPUT_S = 0.10


def _track_clock(config: Config, state: LoopState, method: str):
    """``(sample time, capture->now age)`` for the aim-lead track. Samples are timed by when their frame was
    captured whenever that is known and the track needs it (Auto, or the capture-timestamp option), so inference jitter never reads as target motion; ``age`` is what Auto adds to its horizon."""
    cap = state.frame_capture_t
    now = time.perf_counter()
    if cap > 0.0 and (method == 'auto' or getattr(config, 'prediction_capture_timestamps', False)):
        return cap, max(0.0, now - cap)
    return now, 0.0


def _refresh_aim_track(config: Config, state: LoopState, now_t: float):
    """The aim-lead track, with its parameters re-read from the config every ``_ANCHOR_AIM_PARAMS_REFRESH_S``."""
    track = state.aim_anchor
    # 0.0 = never refreshed (fresh state / after anchor_reset): load the config right away
    # instead of relying on the clock already being past the refresh interval.
    if (state.aim_anchor_params_t == 0.0
            or now_t - state.aim_anchor_params_t >= _ANCHOR_AIM_PARAMS_REFRESH_S
            or state.aim_anchor_params_t > now_t):
        track.params.update_from(config)
        state.aim_anchor_params_t = now_t
    return track


def _anchor_aim_lead(config: Config, state: LoopState, box, comp_dx: float, comp_dy: float,
                     aim_x: float, aim_y: float, now_t: float, horizon_s: float,
                     det_conf: float):
    """Prediction Method 'anchor': lead the (already compensated) aim point with the
    Anchor estimator. ``box`` is the locked box in raw screen coordinates; it is shifted
    by the same camera-drift / own-mouse compensation the aim point carries
    (``comp_dx/comp_dy``) so our own correction never reads as target velocity. Uses the
    aim-lead horizon, not ``anchor_horizon_ms`` (that one is the FOV layer's)."""
    track = _refresh_aim_track(config, state, now_t)
    if track.params.mode == 'disabled':          # stale value only: never stored (config.migrate_anchor_settings)
        return aim_x, aim_y
    shifted = (box[0] + comp_dx, box[1] + comp_dy, box[2] + comp_dx, box[3] + comp_dy)
    out = predict_aim_point(track, now_t, shifted, aim_x, aim_y, horizon_s, det_conf)
    return out


def _auto_aim_lead(config: Config, state: LoopState, box, comp_dx: float, comp_dy: float,
                   aim_x: float, aim_y: float, now_t: float, age_s: float, det_conf: float):
    """Prediction Method 'auto': like 'anchor', but the horizon is the measured capture->aim age plus the
    nominal output delay (see ``AnchorTrack.auto_horizon``) instead of a fixed number of milliseconds."""
    track = _refresh_aim_track(config, state, now_t)
    shifted = (box[0] + comp_dx, box[1] + comp_dy, box[2] + comp_dx, box[3] + comp_dy)
    return auto_aim_point(track, now_t, shifted, aim_x, aim_y, age_s, det_conf)


def calculate_aim_target(
    box: List[float],
    aim_part: str,
    head_height_ratio: float,
    config=None,
) -> Tuple[float, float]:
    """Calculate aim-point coordinates from a detection box."""

    abs_x1, abs_y1, abs_x2, abs_y2 = box
    box_w, box_h = abs_x2 - abs_x1, abs_y2 - abs_y1
    box_center_x = abs_x1 + box_w * 0.5

    # --- Distance-adaptive ratio ---
    # Scales head_height_ratio inversely with box height so the aim point stays
    # on the head at all ranges. Large box (close) → ratio shrinks; small box
    # (far) → ratio grows. Clamped to [0.4×, 2.5×] of the nominal value.
    ratio = head_height_ratio
    if config is not None and getattr(config, 'aim_adaptive_ratio_enabled', False) and box_h > 0:
        ref_h = float(getattr(config, 'aim_adaptive_ratio_ref_h', 80.0))
        scale = ref_h / max(box_h, 1.0)
        ratio = max(head_height_ratio * 0.4, min(head_height_ratio * 2.5, head_height_ratio * scale))

    # --- Posture-aware targeting ---
    # When box_w / box_h exceeds the threshold the player is likely crouching,
    # sliding, or prone. Fall back to center-mass to avoid overshooting above them
    # (the X/Y offsets below still apply on that path).
    posture_center = False
    if config is not None and getattr(config, 'aim_posture_aware_enabled', False) and box_h > 0:
        threshold = float(getattr(config, 'aim_crouch_aspect_threshold', 1.2))
        posture_center = box_w / box_h >= threshold

    if posture_center:
        target_x = box_center_x
        target_y = abs_y1 + box_h * 0.5
    elif aim_part in ('custom', 'center'):       # 'center' (old "Smart") was the same code path: kept as an alias
        target_x = box_center_x
        pct = float(getattr(config, 'aim_custom_y_pct', 30.0)) / 100.0
        target_y = abs_y1 + box_h * pct
    elif aim_part == 'head':
        target_x = box_center_x
        target_y = abs_y1 + box_h * ratio * 0.5
    else:
        target_x = box_center_x
        head_h = box_h * ratio
        target_y = (abs_y1 + head_h + abs_y2) * 0.5

    if config is not None:
        x_frac = max(-1.0, min(1.0, float(getattr(config, 'aim_x_offset_frac', 0.0) or 0.0)))
        y_frac = max(-1.0, min(1.0, float(getattr(config, 'aim_y_offset_frac', 0.0) or 0.0)))
        target_x += box_w * x_frac
        target_y += box_h * y_frac

    # TODO: Per-class routing — when model outputs separate head/body class IDs, bypass
    #       the ratio formula: aim at box center for head class, body formula for body class.
    #       Requires passing class_id and config._detect_class_names here.

    # TODO: Confidence-weighted fallback — blend aim point toward center-mass when detection
    #       confidence is below a threshold (partially occluded target). Config:
    #       aim_low_conf_threshold, aim_low_conf_blend (0–1).

    return target_x, target_y


# How long after a fresh target lock to keep re-bootstrapping the predictor/
# Kalman filter instead of letting them accumulate normally (see
# process_aiming()'s "Acquisition-phase guard" below). Matches the fixed
# window a comparable third-party inference platform's own anchor-prediction
# design uses for the same purpose; not exposed as a config field since it's
# a low-level implementation constant, not a user-facing tuning knob (same
# precedent as _adaptive_sticky_iou's own hardcoded area thresholds).
_ACQUISITION_GUARD_S = 0.024


def _adaptive_sticky_iou(base_iou: float, box: list, fov_size: float) -> float:
    # Replaces fixed lock_iou_threshold with area-scaled version.
    # Ported from Someone_idea/sticky_aim.py StickyTargetLock._adaptive_iou_threshold().
    # Small/far targets get a looser threshold so box jitter doesn't break the lock.
    area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    s = max(0.55, min(1.65, (max(0.1, fov_size / 240.0)) ** 2))
    b = float(base_iou)
    if area < 350 * s:   return max(0.05, b * 0.34)
    if area < 900 * s:   return max(0.06, b * 0.48)
    if area < 2200 * s:  return max(0.08, b * 0.62)
    if area < 4500 * s:  return max(0.05, b * 0.48)
    return max(0.04, b * 0.34)


def _apply_per_frame_cap(move_x: float, move_y: float, config) -> tuple:
    # Hard cap on pixels-per-frame to prevent instant lock-on snaps.
    # A mouse travelling 300+ px in a single 16ms frame is physically implausible for a human.
    # Ported from Someone_idea/ai_aiming.py.
    max_pf = float(getattr(config, 'max_move_per_frame_px', 0) or 0)
    if max_pf <= 0:
        return move_x, move_y
    mag = math.hypot(move_x, move_y)
    if mag > max_pf:
        scale = max_pf / mag
        return move_x * scale, move_y * scale
    return move_x, move_y


# Base IoU for "this box is the locked target" (scaled by box size in _adaptive_sticky_iou). It used to be a
# slider; the area scaling already multiplies it by 0.34-0.62, so the slider barely mattered.
_STICKY_IOU_BASE = 0.3


def sticky_iou_threshold(config: Config, state: LoopState) -> float:
    """Overlap needed to treat a box as the locked target (adaptive to its size)."""
    base = _STICKY_IOU_BASE
    if state.locked_box is not None:
        return _adaptive_sticky_iou(base, state.locked_box, float(getattr(config, 'fov_size', 222)))
    return base


def process_aiming(
    config: Config,
    boxes: List[List[float]],
    crosshair_x: int,
    crosshair_y: int,
    pid_x: PIDController,
    pid_y: PIDController,
    mouse_method: str,
    state: LoopState,
    current_time: float,
    confidences: List[float] | None = None,
    class_of=None,
) -> Optional[bool]:
    """Returns False when no target was accepted this frame (e.g. the locked
    target is momentarily missing inside the minimum lock time) so the caller
    runs its hold/decay path; None/True otherwise.

    Aiming logic: target selection → optional prediction/smoothing → PID → mouse move.

    Per-frame pipeline: select a target from `boxes` (sticky lock + priority
    scoring), optionally run it through velocity prediction
    (`target_predictor.py`) and/or Kalman smoothing (`kalman_filter.py`),
    feed the result to the X/Y PID controllers, then apply Y-axis recoil
    suppression and humanization (micro-jitter, motion variation, speed
    shaping, micro-stutter, reaction variability — see `humanization.py`'s
    `HumanizationConfig`) before dispatching the mouse move.
    """

    aim_part = config.aim_part
    head_height_ratio = config.head_height_ratio
    confidences = confidences or []

    valid_targets = []
    for i, box in enumerate(boxes):
        target_x, target_y = calculate_aim_target(box, aim_part, head_height_ratio, config)
        moveX = target_x - crosshair_x
        moveY = target_y - crosshair_y
        distance_sq = moveX * moveX + moveY * moveY
        conf = confidences[i] if i < len(confidences) else 0.5
        valid_targets.append((distance_sq, conf, target_x, target_y, box))

    if valid_targets:
        priority_mode = str(getattr(config, 'target_priority_mode', 'distance'))
        conf_weight = float(getattr(config, 'target_priority_confidence_weight', 0.5))
        if priority_mode == 'confidence':
            valid_targets.sort(key=lambda x: -(x[1]))
        elif priority_mode == 'composite':
            valid_targets.sort(key=lambda x: x[0] * (1.0 - x[1] * conf_weight))
        else:
            valid_targets.sort(key=lambda x: x[0])

        # --- Sticky target lock (optional) ---
        # When enabled, prefer the previously locked target over a new closest one,
        # preventing aim from snapping to a different target mid-track.
        sticky = getattr(config, 'sticky_lock_enabled', False)
        _continued = False
        if sticky:
            state.tracker.expire(current_time, lost_timeout_s(config))
            # Replaced fixed lock_iou_threshold with adaptive area-scaled version (Someone_idea).
            iou_thresh = sticky_iou_threshold(config, state)
            _fov_ref = max(float(getattr(config, 'fov_effective_size', 0) or 0),
                           float(getattr(config, 'fov_effective_height', 0) or 0),
                           float(getattr(config, 'fov_size', 222)))
            # Overlap match -> reacquisition radius -> switch-threshold /
            # minimum-lock-duration gating all live in select_target().
            selected, _continued = select_target(
                valid_targets, locked_box=state.locked_box, tracker=state.tracker,
                config=config, now=current_time, iou_thresh=iou_thresh, norm=_fov_ref * 0.5,
                class_of=class_of,
                ref_box=anchor_ref_box(config, state, current_time))
            if selected is None:
                return False
        else:
            selected = valid_targets[0]
        _prev_locked_box = state.locked_box
        _, _conf, target_x, target_y, selected_box = selected
        # Always record what got selected — independent of sticky_lock_enabled —
        # so callers (e.g. single_target_mode's box-list reduction in ai_loop.py)
        # can see the actual post-lock pick instead of re-deriving a lock-blind
        # one, which is what let single_target_mode silently defeat sticky lock.
        state.locked_box = selected_box
        state.locked_confidence = _conf

        # The Y-recoil-suppression velocity-restore gate (below) computes how
        # fast the CURRENT target is moving vertically frame-to-frame. If the
        # top pick swaps to a physically different target this same frame
        # (boxes stayed non-empty, so the full target-loss reset in
        # ai_loop.py never runs), the stale last-target Y/timestamp belongs
        # to the old target — dividing a cross-target Y jump by one frame's
        # dt can produce a huge spurious velocity that wrongly disables Y
        # suppression right when it's needed. Reset the gate's timestamp
        # whenever the selected box isn't a continuation of the previous one.
        _iou_thresh = _STICKY_IOU_BASE
        if _prev_locked_box is None or box_iou(_prev_locked_box, selected_box) < _iou_thresh:
            state.aim_y_last_target_t = 0.0
            # Same "is this genuinely a new target" edge as above — arms the
            # acquisition-phase guard below (state.lock_acquired_t) exactly
            # once per fresh lock, not every frame the target stays locked.
            state.lock_acquired_t = current_time

        # Lock lifecycle + smoothed signals for the adaptive/dynamic FOV modes.
        _fw = float(getattr(config, 'fov_effective_size', 0) or getattr(config, 'fov_size', 222))
        _fh = float(getattr(config, 'fov_effective_height', 0) or getattr(config, 'fov_height', _fw) or _fw)
        _same = _continued or (_prev_locked_box is not None
                               and box_iou(_prev_locked_box, selected_box) >= _iou_thresh)
        state.tracker.on_match(
            current_time, selected_box, _conf,
            edge_ratio=edge_ratio(selected_box, crosshair_x, crosshair_y, _fw, _fh,
                                  bool(getattr(config, 'fov_circle_filter_enabled', False))),
            same_target=_same)
        anchor_on_match(config, state, selected_box, _conf, current_time,
                        crosshair_x, crosshair_y, _fw, _fh)
        if class_of is not None:
            _cid = class_of(selected_box)
            if _cid is not None or not _same:
                state.tracker.locked_class = _cid

        if sticky:
            state.no_detection_frames = 0
            config.display_locked_box = list(selected_box)
            config.display_locked_box_is_decaying = False

        # --- Camera-drift-compensated coordinate frame for prediction/smoothing ---
        # The predictor and Kalman filter both estimate target velocity from
        # how its position changes frame-to-frame. Feeding them the raw
        # detected position lets the aimbot's own correction — or camera
        # shake/recoil — look exactly like target motion to that estimate,
        # worst of all right after a fresh lock when the camera is still
        # rotating hard toward the target. state.cam_drift_x/y is a running
        # integral of the phase-correlation-measured background shift (see
        # ai_loop.py's _preprocess_worker), so subtracting it here yields a
        # world-relative position whose frame-to-frame deltas reflect only
        # real target motion; it's added back below before the PID error is
        # computed, which must stay in real, current screen coordinates. This
        # is independent of (and in addition to) the existing per-frame
        # cam_shift_x/y subtraction on the error itself further down, which
        # damps this frame's shake in the PID rather than the prediction
        # history — the two serve different purposes and don't double-count.
        _cam_comp = getattr(config, 'cam_motion_comp_enabled', False)
        if _cam_comp:
            pred_x = target_x - state.cam_drift_x
            pred_y = target_y - state.cam_drift_y
        else:
            pred_x, pred_y = target_x, target_y

        # --- Acquisition-phase guard ---
        # For a short window right after a fresh lock, the crosshair is still
        # snapping onto the target under our own correction — that
        # transition looks exactly like target motion to a naive velocity
        # estimate. Keep re-bootstrapping the predictor/Kalman (reset, then
        # treat this frame's position as a fresh zero-velocity start) until
        # the window elapses, so neither ever computes a velocity across the
        # lock-acquisition jump itself.
        _in_acquisition = (current_time - state.lock_acquired_t) < _ACQUISITION_GUARD_S

        # --- Velocity prediction (optional) ---
        # Extrapolates the aim point forward by prediction_horizon_ms to
        # compensate for capture->inference->move pipeline latency — this is
        # a genuinely different job from Kalman's below (anticipating motion
        # vs. denoising the point), not a duplicate of it. Restored after
        # being merged away: on a PID with Kd=0 on both axes (no derivative
        # term of its own), this was the *only* thing in the whole pipeline
        # compensating for target motion during that latency window — with
        # Kalman off (as it commonly is), removing this left literally
        # nothing anticipating a moving target, which read as the aimbot
        # feeling permanently a step behind.
        t_trk = None                                   # sample time of the aim-lead track this frame (None = not fed)
        if getattr(config, 'prediction_enabled', False):
            method = str(getattr(config, 'prediction_method', 'velocity') or 'velocity')
            if method not in ('velocity', 'anchor', 'auto'):
                method = 'velocity'
            if _in_acquisition:
                _reset_unused_predictors('')
                if method == 'velocity':
                    _get_predictor(config).reset()
                else:
                    state.aim_anchor.reset()
            horizon_s = float(getattr(config, 'prediction_horizon_ms', 10.0)) / 1000.0
            if getattr(config, 'prediction_adaptive_lead', False):
                horizon_s = adaptive_lead_s(horizon_s, float(getattr(state, 'last_mouse_speed_px_s', 0.0)))
            # Optionally time each sample by when its frame was captured, so
            # a slow inference doesn't look like the target slowing down.
            if getattr(config, 'prediction_capture_timestamps', False) and state.frame_capture_t > 0.0:
                now_t = state.frame_capture_t
            else:
                now_t = time.perf_counter()
            if method == 'auto':
                t_trk, _age = _track_clock(config, state, method)
                pred_x, pred_y = _auto_aim_lead(
                    config, state, selected_box, pred_x - target_x, pred_y - target_y,
                    pred_x, pred_y, t_trk, _age, _conf)
            elif method == 'anchor':
                t_trk, _age = _track_clock(config, state, method)
                pred_x, pred_y = _anchor_aim_lead(
                    config, state, selected_box, pred_x - target_x, pred_y - target_y,
                    pred_x, pred_y, t_trk, horizon_s, _conf)
            else:
                pred_x, pred_y = _get_predictor(config).update(pred_x, pred_y, now_t, horizon_s)
            _reset_unused_predictors(method)
            if method == 'velocity' and state.aim_anchor.samples:
                state.aim_anchor.reset()
        else:
            _reset_unused_predictors('')
            if state.aim_anchor.samples:
                state.aim_anchor.reset()

        # --- Strafe response, output side ---
        # A reversal the estimator just detected: take the new heading at once instead of letting the
        # state that was built up for the old one (PID integral, Kalman lag, Aim Step Smoothing) drag.
        _fast = False
        if t_trk is not None and getattr(config, 'strafe_response', 0.0) > 0.0:
            _rt = state.aim_anchor.reversal_t
            if _rt > 0.0 and 0.0 <= t_trk - _rt <= _REVERSAL_OUTPUT_S:
                _fast = True
                if _rt != state.reversal_seen_t:
                    state.reversal_seen_t = _rt
                    pid_x.reset_integral()
                    pid_y.reset_integral()
                    if _kalman is not None:
                        _kalman.reset()

        # --- Kalman filter aim-point smoothing (optional) ---
        if getattr(config, 'kalman_enabled', False):
            kf = _get_kalman(config)
            if _in_acquisition:
                kf.reset()
            pred_x, pred_y = kf.update(pred_x, pred_y)
        else:
            if _kalman is not None:
                _kalman.reset()

        if _cam_comp:
            target_x, target_y = pred_x + state.cam_drift_x, pred_y + state.cam_drift_y
        else:
            target_x, target_y = pred_x, pred_y

        # --- Publish the post-prediction point for overlay.py ---
        # overlay.py's own per-box aim marker is always the raw,
        # pre-prediction point (it recomputes calculate_aim_target() itself
        # from the live box list, with no access to this frame's Kalman/
        # VelocityPredictor output). Publish the actual locked target's
        # post-prediction point here so a second, distinctly-colored marker
        # can show what prediction_enabled/kalman_enabled are actually doing
        # — active only when at least one of them is on; with both off,
        # pred_x/pred_y equal the raw point exactly, so a second marker at
        # the same spot would just be visual noise.
        config.aim_predicted_x = target_x
        config.aim_predicted_y = target_y
        config.aim_prediction_active = bool(getattr(config, 'prediction_enabled', False)) or \
            bool(getattr(config, 'kalman_enabled', False))

        errorX = target_x - crosshair_x
        errorY = target_y - crosshair_y

        # --- Camera motion compensation — cancel shake-induced scene shift ---
        if _cam_comp:
            errorX -= state.cam_shift_x
            errorY -= state.cam_shift_y

        if state.sticky_last_t > 0:
            _sdt = current_time - state.sticky_last_t
            if _sdt > 0:
                state.last_aim_dt = _sdt           # feeds last_mouse_speed_px_s (Adaptive Lead)
        state.sticky_last_t = current_time

        _finish_aim_move(
            config, state, pid_x, pid_y, mouse_method, current_time,
            errorX, errorY, target_y, selected_box[3] - selected_box[1], _fast,
        )

        return


def _finish_aim_move(
    config, state, pid_x, pid_y, mouse_method, current_time,
    errorX, errorY, target_y, box_height, fast: bool = False,
) -> None:
    del box_height
    # Opt-in: pass the frame time so the PID's P/I terms are time-normalised.
    _now = current_time if getattr(config, 'pid_time_normalize', False) else None
    dx, dy = pid_x.update(errorX, _now), pid_y.update(errorY, _now)

    if getattr(config, 'aim_ema_smoothing_enabled', False):
        if fast:                                   # just after a detected reversal: no smoothing lag
            state.aim_ema_x, state.aim_ema_y = dx, dy
        else:
            factor = float(getattr(config, 'aim_ema_smoothing', 0.5) or 0.5)
            dx = ema_step(state.aim_ema_x, dx, factor)
            dy = ema_step(state.aim_ema_y, dy, factor)
            state.aim_ema_x, state.aim_ema_y = dx, dy

    # Track target Y velocity for the velocity-restore gate
    if state.aim_y_last_target_t > 0:
        _y_dt = current_time - state.aim_y_last_target_t
        _vy = (target_y - state.aim_y_last_target_y) / _y_dt if _y_dt > 0 else 0.0
    else:
        _vy = 0.0
    state.aim_y_last_target_y = target_y
    state.aim_y_last_target_t = current_time

    if getattr(config, 'aim_y_reduce_enabled', False) and state.aiming_start_time > 0:
        aim_duration = current_time - state.aiming_start_time
        delay = getattr(config, 'aim_y_reduce_delay', 0.6)
        if aim_duration > delay:
            suppress = True
            settle_px = float(getattr(config, 'aim_y_reduce_settle_px', 0.0))
            if settle_px > 0 and abs(errorY) > settle_px:
                suppress = False
            if suppress:
                vel_restore = float(getattr(config, 'aim_y_vel_restore_px_s', 0.0))
                if vel_restore > 0 and abs(_vy) > vel_restore:
                    suppress = False
            if suppress:
                floor = float(getattr(config, 'aim_y_reduce_floor', 0.0))
                ramp = float(getattr(config, 'aim_y_reduce_ramp', 0.0))
                if ramp > 0:
                    t_past = aim_duration - delay
                    factor = 1.0 - min(1.0, t_past / ramp) * (1.0 - floor)
                else:
                    factor = floor
                dy *= factor

    _hcfg = getattr(config, 'humanization', None)
    if _hcfg is not None and _hcfg.enabled:
        dx, dy = apply_humanization(dx, dy, _hcfg)

    raw_x = dx + state.aim_carry_x
    raw_y = dy + state.aim_carry_y
    move_x = int(raw_x)
    move_y = int(raw_y)
    state.aim_carry_x = raw_x - move_x
    state.aim_carry_y = raw_y - move_y

    if getattr(config, 'max_move_per_frame_px', 0) > 0:
        _mx, _my = _apply_per_frame_cap(float(move_x), float(move_y), config)
        move_x, move_y = int(round(_mx)), int(round(_my))

    dt = float(getattr(state, 'last_aim_dt', 0.0) or 0.0)
    if dt > 0:
        state.last_mouse_speed_px_s = math.hypot(move_x, move_y) / dt

    if move_x != 0 or move_y != 0:
        send_mouse_move(move_x, move_y, method=mouse_method)


def apply_idle_micro_jitter(config: "Config", state: LoopState, mouse_method: str) -> None:
    """
    Optional companion to process_aiming(): simulate Humanization's
    Micro-Jitter tremor while the aim key is held but no target is
    currently locked (e.g. ADS on an empty angle, or the brief gap before
    an unlocked target reacquires). process_aiming() never runs on these
    frames (ai_loop.py only calls it under `is_aiming and boxes`), so
    without this the crosshair sits perfectly still while "aiming" —
    reads as robotic next to the on-target case.

    Reuses apply_humanization(0.0, 0.0, hcfg) unchanged rather than
    reimplementing the jitter math: with a zero-magnitude input, Speed
    Shaping / Motion Variation both no-op via their own
    `magnitude > 0.0` guards, and only Micro-Jitter's amplitude floor
    (micro_jitter_base, scaled by intensity) fires — the exact same
    per-axis noise draw used on-target, just with no proportional term.

    Gated on `humanization.micro_jitter_idle_enabled` — a separate opt-in
    from `micro_jitter_enabled` (which only gates the on-target case) —
    so this is fully off by default and existing users/presets see no
    behavior change until they flip it on.

    Called from ai_loop.py's `else` branch (aimed_this_frame is False),
    only when `is_aiming` is still True that frame.
    """
    hcfg = getattr(config, 'humanization', None)
    if hcfg is None or not hcfg.enabled:
        return
    if not (hcfg.micro_jitter_enabled and getattr(hcfg, 'micro_jitter_idle_enabled', False)):
        return

    dx, dy = apply_humanization(0.0, 0.0, hcfg)

    # Same sub-pixel carry as the main path, so idle-jitter's fractional
    # remainder isn't silently dropped either.
    raw_x = dx + state.aim_carry_x
    raw_y = dy + state.aim_carry_y
    move_x = int(raw_x)
    move_y = int(raw_y)
    state.aim_carry_x = raw_x - move_x
    state.aim_carry_y = raw_y - move_y

    if move_x != 0 or move_y != 0:
        send_mouse_move(move_x, move_y, method=mouse_method)
