from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class LoopState:
    """Status management for AI loop"""

    last_pid_update: float = 0.0
    last_ddxoft_stats_time: float = 0.0

    last_method_check_time: float = 0.0
    cached_mouse_move_method: str = 'mouse_event'

    pid_check_interval: float = 1.0
    ddxoft_stats_interval: float = 30.0
    method_check_interval: float = 2.0

    aiming_start_time: float = 0.0

    # EMA smoothing state — running average of the aim-point coordinates.
    # Reset to 0.0 when aiming starts; converges to the target on the first frame.
    smooth_x: float = 0.0
    smooth_y: float = 0.0

    # Sticky target lock — hold the selected target across short detection gaps.
    locked_box: Optional[List[float]] = field(default=None)
    no_detection_frames: int = 0

    # Box EMA — running average of selected box coords [x1, y1, x2, y2].
    smoothed_box: Optional[List[float]] = field(default=None)

    # Y-reduce velocity gate — track target Y position across frames to estimate vy.
    aim_y_last_target_y: float = 0.0
    aim_y_last_target_t: float = 0.0

    # Camera motion compensation — global scene shift estimated by phase correlation in
    # _preprocess_worker; written there, read in process_aiming to cancel shake-induced error.
    cam_shift_x: float = 0.0
    cam_shift_y: float = 0.0
