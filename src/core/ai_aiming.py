from __future__ import annotations

import random
import time
from typing import TYPE_CHECKING, List, Optional, Tuple

from win_utils import send_mouse_move

from .ai_loop_state import LoopState
from .humanization import apply_humanization
from .inference import PIDController
from .target_predictor import VelocityPredictor

if TYPE_CHECKING:
    from .config import Config

# Module-level predictor singleton — shared across process_aiming calls.
_predictor: Optional[VelocityPredictor] = None


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
        _, _conf, target_x, target_y, _box = valid_targets[0]

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

        if move_x != 0 or move_y != 0:
            send_mouse_move(move_x, move_y, method=mouse_method)
    else:
        pid_x.reset()
        pid_y.reset()
