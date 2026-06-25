from __future__ import annotations

import itertools
import math
import random
import time
from typing import TYPE_CHECKING, List, Optional, Tuple

from win_utils import send_mouse_move, is_makcu_connected

from .ai_loop_state import LoopState
from .humanization import apply_humanization
from .inference import PIDController
from .kalman_filter import KalmanFilter2D
from .target_predictor import VelocityPredictor

if TYPE_CHECKING:
    from .config import Config

def _box_iou(a: List[float], b: List[float]) -> float:
    """Intersection over Union for two [x1, y1, x2, y2] boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0.0:
        return 0.0
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


# Module-level singletons — shared across process_aiming calls.
_predictor: Optional[VelocityPredictor] = None

# Jitter pattern iterator cache: reloaded whenever jitter_pattern_file changes.
_jitter_pattern_cache: dict = {"file": None, "iter": None}


def _load_jitter_pattern(path_str: str) -> list:
    from pathlib import Path
    from core.jitter_recorder import _load_pattern
    return _load_pattern(Path(path_str))["frames"]
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
    return _predictor


def calculate_aim_target(box: List[float], aim_part: str, head_height_ratio: float) -> Tuple[float, float]:
    """Calculate aim-point coordinates from a detection box."""

    abs_x1, abs_y1, abs_x2, abs_y2 = box
    box_w, box_h = abs_x2 - abs_x1, abs_y2 - abs_y1
    box_center_x = abs_x1 + box_w * 0.5

    if aim_part == 'head':
        target_x = box_center_x
        target_y = abs_y1 + box_h * head_height_ratio * 0.5
    else:
        target_x = box_center_x
        head_h = box_h * head_height_ratio
        target_y = (abs_y1 + head_h + abs_y2) * 0.5

    return target_x, target_y


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


def _apply_adaptive_deadzone(
    error_x: float, error_y: float, box_height: float, config
) -> tuple:
    # Zero out small errors so the cursor doesn't micro-jitter when already on-target.
    # Deadzone grows with target proximity because detection noise is larger for close targets.
    # Ported from Someone_idea/ai_aiming.py.
    try:
        if not getattr(config, 'aim_deadzone_enabled', False):
            return error_x, error_y
        fov = float(getattr(config, 'fov_size', 222) or 222)
        h_norm = box_height / max(fov, 1.0)
        dz_min = float(getattr(config, 'aim_deadzone_min_px', 0.4))
        dz_close = float(getattr(config, 'aim_deadzone_close_px', 0.2))
        t = min(1.0, h_norm / 0.28)
        deadzone = dz_min + (dz_close * 9.0 - dz_min) * (t ** 0.85)
        deadzone = max(0.16, min(deadzone, fov * 0.075))
        mag = math.hypot(error_x, error_y)
        if mag < deadzone:
            return 0.0, 0.0
        scale = min(1.0, (mag - deadzone) / (deadzone * 0.3 + 1e-6))
        ratio = (mag - deadzone) / mag
        return error_x * ratio * scale, error_y * ratio * scale
    except Exception:
        return error_x, error_y


def _apply_lateral_overshoot_brake(
    error_x: float, error_y: float, box: list, config
) -> tuple:
    # Slows horizontal correction when vertical error dominates, mimicking the human
    # tendency to settle onto a target rather than diagonal-snapping.
    # Ported from Someone_idea/ai_aiming.py.
    try:
        if not getattr(config, 'aim_lateral_brake_enabled', False):
            return error_x, error_y
        ex = abs(error_x)
        ey = abs(error_y)
        if ey < 1e-6:
            return error_x, error_y
        dom_trigger = float(getattr(config, 'aim_lateral_brake_dom_trigger', 1.12))
        dominance = ex / max(ey, 1e-6)
        if dominance < dom_trigger:
            return error_x, error_y
        dom_max = float(getattr(config, 'aim_lateral_brake_dom_max', 3.0))
        strength = float(getattr(config, 'aim_lateral_brake_strength', 0.75))
        min_scale = float(getattr(config, 'aim_lateral_brake_min_scale', 0.26))
        t = min(1.0, (dominance - dom_trigger) / max(dom_max - dom_trigger, 0.1))
        brake_raw = 1.0 - (1.0 - min_scale) * (t ** 0.9) * strength
        x_scale = max(min_scale, min(1.0, brake_raw))
        return error_x * x_scale, error_y
    except Exception:
        return error_x, error_y


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
) -> None:
    """Aiming logic: direct detection coordinates → PID → mouse move.

    SmartTracker (velocity prediction) and Bezier-curve offset have been
    removed. The cursor moves to the raw detection coordinate each frame
    with no temporal smoothing or path interpolation.
    """

    aim_part = config.aim_part
    head_height_ratio = config.head_height_ratio
    config._current_confidences = confidences or []

    valid_targets = []
    confidences = getattr(config, '_current_confidences', [])
    for i, box in enumerate(boxes):
        target_x, target_y = calculate_aim_target(box, aim_part, head_height_ratio)
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
        selected = valid_targets[0]
        if sticky and state.locked_box is not None:
            # Replaced fixed lock_iou_threshold with adaptive area-scaled version (Someone_idea).
            _base_iou = float(getattr(config, 'lock_iou_threshold', 0.3))
            if getattr(config, 'sticky_adaptive_iou', True):
                iou_thresh = _adaptive_sticky_iou(_base_iou, state.locked_box, float(getattr(config, 'fov_size', 222)))
            else:
                iou_thresh = _base_iou
            best_item, best_iou = None, 0.0
            for item in valid_targets:
                iou = _box_iou(state.locked_box, item[4])
                if iou > best_iou:
                    best_iou, best_item = iou, item
            if best_iou >= iou_thresh and best_item is not None:
                selected = best_item
        _, _conf, target_x, target_y, selected_box = selected
        if sticky:
            state.locked_box = selected_box
            state.no_detection_frames = 0
            config.display_locked_box = list(selected_box)
            config.display_locked_box_is_decaying = False

        # --- Box EMA — smooth raw box coords to suppress size-jitter wobble ---
        # Runs after sticky lock (which needs the raw box for IOU matching) but
        # before aim-point computation so that frame-to-frame box size variance
        # doesn't propagate into target_x / target_y.
        if getattr(config, 'box_ema_enabled', False):
            raw_box = list(selected_box)
            if state.smoothed_box is None:
                state.smoothed_box = raw_box[:]
            else:
                ax = float(getattr(config, 'box_ema_alpha_x', 0.8))
                ay = float(getattr(config, 'box_ema_alpha_y', 0.5))
                sb = state.smoothed_box
                state.smoothed_box = [
                    ax * raw_box[0] + (1.0 - ax) * sb[0],
                    ay * raw_box[1] + (1.0 - ay) * sb[1],
                    ax * raw_box[2] + (1.0 - ax) * sb[2],
                    ay * raw_box[3] + (1.0 - ay) * sb[3],
                ]
            target_x, target_y = calculate_aim_target(state.smoothed_box, aim_part, head_height_ratio)
            selected_box = state.smoothed_box
        else:
            state.smoothed_box = list(selected_box)

        # --- Velocity prediction (optional) ---
        if getattr(config, 'prediction_enabled', False):
            predictor = _get_predictor(config)
            horizon_s = float(getattr(config, 'prediction_horizon_ms', 10.0)) / 1000.0
            target_x, target_y = predictor.update(target_x, target_y, time.perf_counter(), horizon_s)
        else:
            if _predictor is not None:
                _predictor.reset()

        # --- Kalman filter aim-point smoothing (optional, UI-exclusive with EMA) ---
        if getattr(config, 'kalman_enabled', False):
            kf = _get_kalman(config)
            target_x, target_y = kf.update(target_x, target_y)
        else:
            if _kalman is not None:
                _kalman.reset()

        # --- EMA aim-point smoothing (optional) ---
        # Smooths the target coordinate before feeding to PID, reducing jitter
        # without introducing the drift risk of full Kalman filtering.
        if getattr(config, 'ema_enabled', False):
            alpha = float(getattr(config, 'ema_alpha', 0.7))
            if state.smooth_x == 0.0 and state.smooth_y == 0.0:
                # Bootstrap on first frame so the aim doesn't spring from (0, 0).
                state.smooth_x = target_x
                state.smooth_y = target_y
            else:
                state.smooth_x = alpha * target_x + (1.0 - alpha) * state.smooth_x
                state.smooth_y = alpha * target_y + (1.0 - alpha) * state.smooth_y
            target_x, target_y = state.smooth_x, state.smooth_y
        else:
            state.smooth_x = 0.0
            state.smooth_y = 0.0

        errorX = target_x - crosshair_x
        errorY = target_y - crosshair_y

        # --- Camera motion compensation — cancel shake-induced scene shift ---
        if getattr(config, 'cam_motion_comp_enabled', False):
            errorX -= state.cam_shift_x
            errorY -= state.cam_shift_y

        # --- Adaptive deadzone (new feature from Someone_idea) ---
        if getattr(config, 'aim_deadzone_enabled', False):
            errorX, errorY = _apply_adaptive_deadzone(errorX, errorY, selected_box[3] - selected_box[1], config)
            if errorX == 0.0 and errorY == 0.0:
                return

        # --- Lateral overshoot brake (new feature from Someone_idea) ---
        if getattr(config, 'aim_lateral_brake_enabled', False):
            errorX, errorY = _apply_lateral_overshoot_brake(errorX, errorY, selected_box, config)

        dx, dy = pid_x.update(errorX), pid_y.update(errorY)

        # Track target Y velocity for the velocity-restore gate (independent of prediction_enabled)
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
                # Error-based gate: skip suppression until crosshair has settled vertically
                settle_px = float(getattr(config, 'aim_y_reduce_settle_px', 0.0))
                if settle_px > 0 and abs(errorY) > settle_px:
                    suppress = False
                # Velocity-aware gate: restore full Y if target is moving vertically fast enough
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

        # Apply humanization layer (post-PID, pre-rounding, pre-injection).
        # Operates only on dx/dy; never touches PID state or coordinate space.
        _hcfg = getattr(config, 'humanization', None)
        if _hcfg is not None and _hcfg.enabled:
            _result = apply_humanization(dx, dy, _hcfg)
            if _result is None:
                # Reaction variability: suppress this frame's injection.
                # PID error persists and is corrected on the next frame.
                return
            dx, dy = _result

        move_x, move_y = int(round(dx)), int(round(dy))

        # --- Per-frame pixel cap (new feature from Someone_idea) ---
        if getattr(config, 'max_move_per_frame_px', 0) > 0:
            _mx, _my = _apply_per_frame_cap(float(move_x), float(move_y), config)
            move_x, move_y = int(round(_mx)), int(round(_my))

        if getattr(config, 'jitter_enabled', False) and (move_x != 0 or move_y != 0):
            j = float(getattr(config, 'jitter_strength', 1.5))
            move_x += int(random.uniform(-j, j))
            move_y += int(random.uniform(-j, j))

        # --- Smart jitter: fires when box is small (far target) ---
        if getattr(config, 'smart_jitter_enabled', False):
            lmb_gate = getattr(config, 'smart_jitter_lmb_gate', True)
            if not lmb_gate:
                is_shooting = True
            elif getattr(config, 'mouse_move_method', '') == 'makcu' and is_makcu_connected():
                from win_utils.makcu_mouse import makcu_mouse as _mm
                is_shooting = _mm.lmb_held
            else:
                is_shooting = state.aiming_start_time > 0
            if is_shooting:
                box_h = selected_box[3] - selected_box[1]
                detect_size = float(getattr(config, 'detect_range_size', 350))
                threshold_pct = float(getattr(config, 'smart_jitter_box_threshold_pct', 15.0))
                if detect_size > 0 and (box_h / detect_size) * 100.0 < threshold_pct:
                    sj = max(0.0, float(getattr(config, 'smart_jitter_strength', 6.0)))
                    pattern_file = getattr(config, 'jitter_pattern_file', '')
                    if pattern_file:
                        cache = _jitter_pattern_cache
                        if cache["file"] != pattern_file:
                            try:
                                frames = _load_jitter_pattern(pattern_file)
                                cache["iter"] = itertools.cycle(frames)
                                cache["file"] = pattern_file
                            except Exception:
                                cache["iter"] = None
                                cache["file"] = None
                        if cache["iter"]:
                            _mult = max(1, int(getattr(config, 'jitter_speed_multiplier', 1)))
                            for _ in range(_mult):
                                f = next(cache["iter"])
                                move_x += int(f["dx"])
                                move_y += int(f["dy"])
                        else:
                            angle = random.uniform(0, math.tau)
                            r = random.uniform(0, sj)
                            move_x += int(r * math.cos(angle))
                            move_y += int(r * math.sin(angle))
                    else:
                        angle = random.uniform(0, math.tau)
                        r = random.uniform(0, sj)
                        move_x += int(r * math.cos(angle))
                        move_y += int(r * math.sin(angle))

        if move_x != 0 or move_y != 0:
            send_mouse_move(move_x, move_y, method=mouse_method)
    else:
        sticky = getattr(config, 'sticky_lock_enabled', False)
        if sticky and state.locked_box is not None:
            decay = int(getattr(config, 'lock_decay_frames', 15))
            state.no_detection_frames += 1
            config.display_locked_box_is_decaying = True
            if state.no_detection_frames < decay:
                # Hold aim — PID keeps last error; no mouse move this frame
                return
            # Decay expired — clear lock and reset
            state.locked_box = None
            state.no_detection_frames = 0
            state.smoothed_box = None
            config.display_locked_box = None
            config.display_locked_box_is_decaying = False
        state.smoothed_box = None
        pid_x.reset()
        pid_y.reset()
        state.aim_y_last_target_y = 0.0
        state.aim_y_last_target_t = 0.0
        if _kalman is not None:
            _kalman.reset()
