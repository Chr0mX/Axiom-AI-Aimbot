from __future__ import annotations

import random
from typing import TYPE_CHECKING, List, Tuple

from win_utils import send_mouse_move

from .ai_loop_state import LoopState
from .humanization import apply_humanization
from .inference import PIDController

if TYPE_CHECKING:
    from .config import Config


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
    """Aiming pipeline: Detection → PID → Mouse"""

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

        config.tracker_has_prediction = False

        errorX = target_x - crosshair_x
        errorY = target_y - crosshair_y

        dx, dy = pid_x.update(errorX), pid_y.update(errorY)

        # Y-axis recoil reduction: gradually zero vertical movement after aiming
        if getattr(config, 'aim_y_reduce_enabled', False) and state.aiming_start_time > 0:
            aim_duration = current_time - state.aiming_start_time
            delay = getattr(config, 'aim_y_reduce_delay', 0.6)
            if aim_duration > delay:
                dy = 0.0

        # ── Humanization: post-PID delta shaping ─────────────────────────────
        # Operates only on (dx, dy); never touches PID state or coordinate space.
        # Applies speed_multiplier, speed shaping, jitter, stutter, etc.
        _hcfg = getattr(config, 'humanization', None)
        if _hcfg is not None and _hcfg.enabled:
            _result = apply_humanization(dx, dy, _hcfg)
            if _result is None:
                # Reaction variability: suppress this frame's injection.
                # PID error persists and is corrected on the next frame.
                return
            dx, dy = _result

        # Legacy jitter (independent of humanization layer)
        if getattr(config, 'jitter_enabled', False) and (dx != 0 or dy != 0):
            j = float(getattr(config, 'jitter_strength', 1.5))
            dx += random.uniform(-j, j)
            dy += random.uniform(-j, j)

        move_x, move_y = int(round(dx)), int(round(dy))
        if move_x != 0 or move_y != 0:
            send_mouse_move(move_x, move_y, method=mouse_method)

    else:
        # No targets — reset PID state
        pid_x.reset()
        pid_y.reset()
