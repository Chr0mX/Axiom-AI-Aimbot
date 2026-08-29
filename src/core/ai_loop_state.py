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

    # Sub-pixel carry — accumulates the fractional remainder that integer truncation
    # discards each frame so micro-corrections are never silently lost and the
    # crosshair converges exactly onto the aim point. Applies to all mouse backends.
    aim_carry_x: float = 0.0
    aim_carry_y: float = 0.0
