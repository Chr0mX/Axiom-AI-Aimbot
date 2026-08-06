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

    # Recorded-jitter-pattern carry — same idea as aim_carry_*, but for
    # replayed pattern frames. Patterns are normalized to zero net
    # displacement; truncating each frame's dx/dy independently breaks that
    # (sum of int(x) != int(sum of x)) and walks the crosshair off-target
    # over a long burst. Carrying the fraction preserves the invariant.
    jitter_carry_x: float = 0.0
    jitter_carry_y: float = 0.0

    # Wall-clock timestamp of the last PID update, so the controller can be
    # given a real dt instead of assuming a fixed step. 0.0 = no previous
    # frame (fresh, or just reset on target loss), which the PID treats as
    # "use the reference step" rather than guessing.
    pid_last_update_t: float = 0.0
