from __future__ import annotations

import random
from typing import TYPE_CHECKING, List, Tuple

from win_utils import send_mouse_move

from .ai_loop_state import LoopState
from .inference import PIDController
from .smart_tracker import SmartTracker

if TYPE_CHECKING:
    from .config import Config


def calculate_aim_target(box: List[float], aim_part: str, head_height_ratio: float) -> Tuple[float, float]:
    """計算瞄準點座標"""

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
) -> None:
    """處理瞄準邏輯 (包含卡爾曼濾波預判和幽靈目標/貝塞爾曲線偏移)"""

    aim_part = config.aim_part
    head_height_ratio = config.head_height_ratio

    valid_targets = []
    for box in boxes:
        target_x, target_y = calculate_aim_target(box, aim_part, head_height_ratio)
        moveX = target_x - crosshair_x
        moveY = target_y - crosshair_y
        distance_sq = moveX * moveX + moveY * moveY
        valid_targets.append((distance_sq, target_x, target_y, box))

    if valid_targets:
        valid_targets.sort(key=lambda x: x[0])
        _, target_x, target_y, box = valid_targets[0]

        tracker_enabled = getattr(config, 'tracker_enabled', False)
        if tracker_enabled:
            if state.smart_tracker is None:
                state.smart_tracker = SmartTracker(
                    smoothing_factor=getattr(config, 'tracker_smoothing_factor', 0.5),
                    stop_threshold=getattr(config, 'tracker_stop_threshold', 20.0),
                )
                state.tracker_last_time = current_time
            else:
                state.smart_tracker.alpha = getattr(config, 'tracker_smoothing_factor', 0.5)
                state.smart_tracker.stop_threshold = getattr(config, 'tracker_stop_threshold', 20.0)

            current_box_tuple = tuple(box)
            if state.tracker_last_target_box is not None:
                last_box = state.tracker_last_target_box
                last_cx = (last_box[0] + last_box[2]) * 0.5
                last_cy = (last_box[1] + last_box[3]) * 0.5
                curr_cx = (box[0] + box[2]) * 0.5
                curr_cy = (box[1] + box[3]) * 0.5
                box_distance_sq = (curr_cx - last_cx) ** 2 + (curr_cy - last_cy) ** 2
                if box_distance_sq > 40000:
                    state.smart_tracker.reset()
            state.tracker_last_target_box = current_box_tuple

            dt = current_time - state.tracker_last_time
            if dt <= 0:
                dt = 0.01
            state.tracker_last_time = current_time

            state.smart_tracker.update(target_x, target_y, dt)

            prediction_time = getattr(config, 'tracker_prediction_time', 0.05)
            pred_x, pred_y = state.smart_tracker.get_predicted_position(prediction_time)

            config.tracker_current_x = target_x
            config.tracker_current_y = target_y
            config.tracker_predicted_x = pred_x
            config.tracker_predicted_y = pred_y
            config.tracker_has_prediction = True

            target_x, target_y = pred_x, pred_y
        else:
            config.tracker_has_prediction = False
            if state.smart_tracker is not None:
                state.smart_tracker.reset()
                state.smart_tracker = None

        errorX = target_x - crosshair_x
        errorY = target_y - crosshair_y

        if getattr(config, 'bezier_curve_enabled', False):
            if not state.target_locked:
                state.target_locked = True
                state.bezier_curve_scalar = random.uniform(-1.0, 1.0)

            strength = float(getattr(config, 'bezier_curve_strength', 0.35))
            perp_x = -errorY
            perp_y = errorX

            offset_x = perp_x * strength * state.bezier_curve_scalar
            offset_y = perp_y * strength * state.bezier_curve_scalar

            errorX += offset_x
            errorY += offset_y
        else:
            state.target_locked = False

        dx, dy = pid_x.update(errorX), pid_y.update(errorY)

        if getattr(config, 'aim_y_reduce_enabled', False) and state.aiming_start_time > 0:
            aim_duration = current_time - state.aiming_start_time
            delay = getattr(config, 'aim_y_reduce_delay', 0.6)

            if aim_duration > delay:
                dy = 0.0

        move_x, move_y = int(round(dx)), int(round(dy))

        if move_x != 0 or move_y != 0:
            send_mouse_move(move_x, move_y, method=mouse_method)
    else:
        state.target_locked = False
        pid_x.reset()
        pid_y.reset()
