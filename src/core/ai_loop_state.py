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

    # EMA smoothing state — running average of the aim-point coordinates.
    # Reset to 0.0 (and smooth_initialized to False) when aiming starts;
    # converges to the target on the first frame. smooth_initialized is the
    # actual "needs bootstrap" signal — smooth_x/y == 0.0 used to double as
    # that sentinel, which wrongly re-bootstrapped on a legitimately
    # smoothed aim-point of exactly (0, 0).
    smooth_x: float = 0.0
    smooth_y: float = 0.0
    smooth_initialized: bool = False

    # Sticky target lock — hold the selected target across short detection gaps.
    # locked_box/locked_confidence are updated on every successful selection
    # (regardless of sticky_lock_enabled) so callers can read back what aiming
    # actually picked this frame; only *acting* on a hold across a gap is
    # gated by sticky_lock_enabled elsewhere.
    locked_box: Optional[List[float]] = field(default=None)
    locked_confidence: float = 0.0
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

    # Sub-pixel carry — accumulates the fractional remainder that integer truncation
    # discards each frame so micro-corrections are never silently lost and the
    # crosshair converges exactly onto the aim point. Applies to all mouse backends.
    aim_carry_x: float = 0.0
    aim_carry_y: float = 0.0
