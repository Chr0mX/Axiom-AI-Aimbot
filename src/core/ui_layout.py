"""Layout rules for the glass pages — pure Python (no Qt).

Pages stack ``SettingCardGroup``s; on wide windows they flow into two balanced columns, on narrow
ones into one. The Qt band widget only applies what these functions decide.
"""
from __future__ import annotations

from typing import List, Sequence

MIN_COLUMN_W = 440    # a column narrower than this cramps slider rows (title + slider + value pill)
COLUMN_GAP = 20


def columns_for_width(width: int, max_columns: int = 2, min_column_w: int = MIN_COLUMN_W, gap: int = COLUMN_GAP) -> int:
    """How many columns fit in ``width`` px (1..max_columns)."""
    n = max(1, int(max_columns))
    while n > 1 and (width - gap * (n - 1)) / n < min_column_w:
        n -= 1
    return n


def assign_columns(heights: Sequence[float], n_columns: int) -> List[int]:
    """Column index per item: each item goes to the currently shortest column (ties: leftmost), in
    order, so the columns end up balanced and the first items sit at the top of the left column.

    A ``0`` height (hidden group) still gets a column — the shortest one at that moment.
    """
    n = max(1, int(n_columns))
    if n == 1:
        return [0] * len(heights)
    totals = [0.0] * n
    out: List[int] = []
    for h in heights:
        col = min(range(n), key=lambda c: (totals[c], c))
        out.append(col)
        totals[col] += max(0.0, float(h))
    return out


def split_bands(kinds: Sequence[str]) -> List[List[int]]:
    """Group consecutive ``'group'`` items into bands; every other item (``'span'``) is its own band.

    >>> split_bands(['group', 'group', 'span', 'group'])
    [[0, 1], [2], [3]]
    """
    bands: List[List[int]] = []
    for i, kind in enumerate(kinds):
        if kind == 'group' and bands and kinds[bands[-1][0]] == 'group':
            bands[-1].append(i)
        else:
            bands.append([i])
    return bands
