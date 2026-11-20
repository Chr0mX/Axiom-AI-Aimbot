"""Rules that tie one aiming setting to another, in one place.

The Qt Aim page, the Web Control API and anything else that changes the Move Method all
need the same side effect, so it lives here instead of being re-implemented in each UI.
Pure Python.
"""
from __future__ import annotations

# Move methods that also deliver the click (a device or network backend): the click must go
# through the same backend or the click and the movement would come from different places.
CLICK_FOLLOWS_MOVE = ('makcu', 'vigem_network')


def click_method_for_move(move_method: str, current_click: str) -> str:
    """The click method that should be in effect after the Move Method becomes ``move_method``:
    the same backend for makcu / vigem_network, otherwise the user's current click choice."""
    return move_method if move_method in CLICK_FOLLOWS_MOVE else current_click
