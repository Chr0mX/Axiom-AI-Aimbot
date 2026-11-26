"""Two-column flow for the groups of a settings page (rules live in ``core.ui_layout``).

``TwoColumnBand`` owns consecutive ``GlassSettingCardGroup``s. On a wide page it spreads them over
two balanced columns, on a narrow one it stacks them in one. Groups are **moved, never recreated**,
and only when the assignment actually changes (column count crossing the threshold, or a group
becoming visible/hidden), so a window resize or a conditional group never flickers or resets state.
"""
from __future__ import annotations

from typing import List

from PyQt6.QtCore import QEvent, QObject, QTimer
from PyQt6.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget

from core import ui_layout


class TwoColumnBand(QWidget):
    def __init__(self, parent: QWidget | None = None, max_columns: int = 2):
        super().__init__(parent)
        self._max = max_columns
        self._groups: List[QWidget] = []
        self._columns: List[QVBoxLayout] = []
        self._key = None
        self._pending = False
        self._row = QHBoxLayout(self)
        self._row.setContentsMargins(0, 0, 0, 0)
        self._row.setSpacing(ui_layout.COLUMN_GAP)
        self._build(1)

    # ------------------------------------------------------------------ structure
    def addGroup(self, group: QWidget) -> None:
        self._groups.append(group)
        group.setParent(self)
        group.installEventFilter(self)
        self._key = None
        self._schedule()

    def groups(self) -> List[QWidget]:
        return list(self._groups)

    def columnCount(self) -> int:
        return len(self._columns)

    def _build(self, n: int) -> None:
        while self._row.count():
            self._row.takeAt(0)
        for col in self._columns:
            col.setParent(None)
        self._columns = []
        for _ in range(n):
            col = QVBoxLayout()
            col.setContentsMargins(0, 0, 0, 0)
            col.setSpacing(20)
            self._row.addLayout(col, 1)
            self._columns.append(col)

    # ------------------------------------------------------------------ reflow
    def _schedule(self) -> None:
        if not self._pending:
            self._pending = True
            QTimer.singleShot(0, self._reflow)

    def eventFilter(self, obj: QObject, e: QEvent) -> bool:
        if e.type() in (QEvent.Type.Show, QEvent.Type.Hide):
            self._schedule()
        return False

    def resizeEvent(self, e) -> None:
        super().resizeEvent(e)
        self._schedule()

    def _reflow(self) -> None:
        self._pending = False
        n = ui_layout.columns_for_width(self.width() or 10_000, self._max)
        vis = tuple(g.isVisibleTo(self) for g in self._groups)
        n = min(n, max(1, sum(vis)))        # a lone visible group uses the full width
        key = (n, vis)
        if key == self._key:
            return
        self._key = key
        col_w = max(1, (self.width() - ui_layout.COLUMN_GAP * (n - 1)) // n) if self.width() else 600
        heights = []
        for g, shown in zip(self._groups, vis):
            if not shown:
                heights.append(0)
            else:
                h = g.heightForWidth(col_w) if g.hasHeightForWidth() else g.sizeHint().height()
                heights.append(max(h, g.sizeHint().height()))
        assign = ui_layout.assign_columns(heights, n)
        if n != len(self._columns):
            self._build(n)
        else:       # same column count: detach groups, keep the layouts
            for col in self._columns:
                while col.count():
                    col.takeAt(0)
        for g, c in zip(self._groups, assign):
            self._columns[c].addWidget(g)
        for col in self._columns:
            col.addStretch(1)
        self.updateGeometry()
