"""Timing / easing rules for every interface animation — pure Python (no Qt).

One table of durations and one on/off switch (``Config.ui_animations``) so the whole app moves with
the same rhythm and turns instant at once. Qt widgets read ``duration()`` when they build a tween;
painters use ``ease_out_cubic`` / ``lerp_color`` to blend states without owning any timers.
"""
from __future__ import annotations

from typing import Sequence, Tuple

# milliseconds
DUR_FAST = 120      # hover / press feedback, switch thumb
DUR = 180           # page fade, dropdowns, row highlights
DUR_SLOW = 260      # navigation capsule, dock changes, theme cross-fade

_enabled = True


def set_animations_enabled(enabled: bool) -> None:
    global _enabled
    _enabled = bool(enabled)


def animations_enabled() -> bool:
    return _enabled


def duration(ms: int) -> int:
    """``ms`` when animations are on, else 0 (callers apply the end state at once)."""
    return int(ms) if _enabled else 0


def clamp01(t: float) -> float:
    return 0.0 if t <= 0.0 else 1.0 if t >= 1.0 else t


def ease_out_cubic(t: float) -> float:
    t = clamp01(t)
    return 1.0 - (1.0 - t) ** 3


def ease_in_out_cubic(t: float) -> float:
    t = clamp01(t)
    return 4 * t * t * t if t < 0.5 else 1.0 - (-2 * t + 2) ** 3 / 2


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * clamp01(t)


def lerp_color(c1: Sequence[int], c2: Sequence[int], t: float) -> Tuple[int, ...]:
    """Component-wise blend of two RGB(A) tuples of equal length."""
    return tuple(int(round(lerp(a, b, t))) for a, b in zip(c1, c2))
