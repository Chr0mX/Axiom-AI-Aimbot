from __future__ import annotations

from dataclasses import dataclass


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
