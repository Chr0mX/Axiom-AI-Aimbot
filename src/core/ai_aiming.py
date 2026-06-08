from __future__ import annotations

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
            iou_thresh = float(getattr(config, 'lock_iou_threshold', 0.3))
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

        # --- Velocity prediction (optional) ---
        if getattr(config, 'prediction_enabled', False):
            predictor = _get_predictor(config)
            horizon_s = float(getattr(config, 'prediction_horizon_ms', 10.0)) / 1000.0
            target_x, target_y = predictor.update(target_x, target_y, time.perf_counter(), horizon_s)
            config.tracker_has_prediction = True
        else:
            config.tracker_has_prediction = False
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

        dx, dy = pid_x.update(errorX), pid_y.update(errorY)

        if getattr(config, 'aim_y_reduce_enabled', False) and state.aiming_start_time > 0:
            aim_duration = current_time - state.aiming_start_time
            delay = getattr(config, 'aim_y_reduce_delay', 0.6)

            if aim_duration > delay:
                dy = 0.0

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
            config.display_locked_box = None
            config.display_locked_box_is_decaying = False
        pid_x.reset()
        pid_y.reset()
        if _kalman is not None:
            _kalman.reset()
