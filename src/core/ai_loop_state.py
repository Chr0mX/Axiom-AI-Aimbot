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

    # Auto Un-ADS — forced-release window bookkeeping. unads_release_active
    # gates both the ai_loop.py is_aiming override (keeps process_aiming()
    # running through the release window) and ai_aiming.py's movement-send
    # skip. unads_pending_transition is a one-shot signal set by
    # process_aiming()'s decision logic and consumed+cleared by
    # ai_loop_utils.apply_unads_transition() the same iteration.
    unads_release_active: bool = False
    unads_release_start_time: float = 0.0
    unads_clear_hold_start: float = 0.0     # when the reengage clear condition started holding; 0 = not holding
    unads_pending_transition: str = ''      # '' | 'release' | 'reengage'
    unads_active_vks: List[int] = field(default_factory=list)  # generic-path snapshot of held AimKeys at release time
    unads_last_target_x: float = 0.0
    unads_last_target_y: float = 0.0
    unads_last_target_t: float = 0.0
