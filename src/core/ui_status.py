"""State for the bottom status pill (CPU % + quick preset buttons) — pure Python (no Qt).

The Qt widget only draws what these functions return, so the rules (how many slots, which one is
active, what an empty slot says, how CPU is printed) are testable without a display.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

SLOT_COUNT = 5


def format_cpu(percent: Optional[float]) -> str:
    """``'31%'`` for a reading, ``'—'`` when there is none (psutil missing / first sample)."""
    if percent is None:
        return "—"
    try:
        p = float(percent)
    except (TypeError, ValueError):
        return "—"
    if p != p:                      # NaN
        return "—"
    return f"{max(0, min(100, round(p)))}%"


def normalize_slots(slots: Optional[Sequence], count: int = SLOT_COUNT) -> List[str]:
    """Exactly ``count`` names (padded with '' / truncated), never None — same rule as Web Control."""
    names = [str(s) if s else "" for s in (slots or [])][:count]
    return names + [""] * (count - len(names))


def slot_states(slots: Optional[Sequence], active_name: str = "", count: int = SLOT_COUNT) -> List[Dict]:
    """One dict per slot: ``index`` (0-based), ``label`` (1-based text), ``name``, ``assigned``,
    ``active`` (the slot whose preset is the one currently loaded)."""
    names = normalize_slots(slots, count)
    active_name = active_name or ""
    out = []
    for i, name in enumerate(names):
        out.append({"index": i, "label": str(i + 1), "name": name,
                    "assigned": bool(name), "active": bool(name) and name == active_name})
    return out


def active_label(active_preset: str = "", active_config: str = "", empty: str = "None", limit: int = 22) -> str:
    """Name shown at the pill's right: the loaded preset wins over a loaded config (as on the HUD)."""
    name = (active_preset or "").strip() or (active_config or "").strip() or empty
    return name if len(name) <= limit else name[: limit - 1] + "…"
