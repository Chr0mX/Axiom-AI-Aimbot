"""Bottom status pill: CPU %, quick preset buttons 1..N and the loaded preset's name.

Draws what ``core.ui_status`` decides. Clicking a numbered button emits ``slotClicked(index)``;
the window loads the preset (or explains how to assign one). It owns no timer — the window's
existing 1 s sync tick calls ``update_state()``.
"""
from __future__ import annotations

from typing import List, Optional

from PyQt6.QtCore import QEvent, QPointF, QRectF, QSize, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPen
from PyQt6.QtWidgets import QToolTip, QWidget

from core import ui_status
from ..theme_colors import ThemeColors
from .glass import accent_palette, signals
from .motion import DUR_FAST, Tween

H = 36
BTN = 26
GAP = 6
PADX = 14


class StatusPill(QWidget):
    slotClicked = pyqtSignal(int)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setFixedHeight(H)
        self._cpu: Optional[float] = None
        self._states: List[dict] = ui_status.slot_states([], "")
        self._label = "None"
        self._cpuCaption = "CPU"
        self._presetCaption = "PRESET"
        self._emptyTip = "Empty slot"
        self._hot = -1
        self._hover = [Tween(self, 0.0) for _ in self._states]
        for t in self._hover:
            t.changed.connect(lambda _v: self.update())
        self._font = QFont(self.font())
        self._font.setPixelSize(12)
        self._font.setWeight(QFont.Weight.DemiBold)
        signals.appearanceChanged.connect(self.update)

    # ------------------------------------------------------------------ state
    def update_state(self, cpu: Optional[float], slots, active_preset: str, active_config: str,
                     none_text: str = "None") -> None:
        states = ui_status.slot_states(slots, active_preset)
        label = ui_status.active_label(active_preset, active_config, none_text)
        if cpu == self._cpu and states == self._states and label == self._label:
            return
        self._cpu, self._states, self._label = cpu, states, label
        self.updateGeometry()
        self.update()

    def setCaptions(self, cpu: str, preset: str, empty_tip: str) -> None:
        self._cpuCaption, self._presetCaption, self._emptyTip = cpu, preset, empty_tip
        self.updateGeometry()
        self.update()

    # ------------------------------------------------------------------ geometry
    def _cells(self):
        fm = QFontMetrics(self._font)
        cpu_text = f"{self._cpuCaption} {ui_status.format_cpu(self._cpu)}"
        cpu_w = fm.horizontalAdvance(cpu_text)
        cap_w = fm.horizontalAdvance(self._presetCaption)
        lab_w = fm.horizontalAdvance(self._label)
        return cpu_text, cpu_w, cap_w, lab_w

    def sizeHint(self) -> QSize:
        _t, cpu_w, cap_w, lab_w = self._cells()
        n = len(self._states)
        w = PADX + cpu_w + 18 + cap_w + 10 + n * BTN + (n - 1) * GAP + 18 + lab_w + PADX
        return QSize(int(w), H)

    def _buttonRects(self) -> List[QRectF]:
        _t, cpu_w, cap_w, _l = self._cells()
        x = (self.width() - self.sizeHint().width()) / 2 + PADX + cpu_w + 18 + cap_w + 10
        return [QRectF(x + i * (BTN + GAP), (H - BTN) / 2, BTN, BTN) for i in range(len(self._states))]

    # ------------------------------------------------------------------ painting
    def paintEvent(self, e) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        pal = accent_palette()
        sh = self.sizeHint().width()
        pill = QRectF((self.width() - sh) / 2 + 0.5, 0.5, sh - 1, H - 1)
        p.setBrush(ThemeColors.GLASS_PILL.qcolor())
        p.setPen(QPen(ThemeColors.GLASS_PILL_BORDER.qcolor(), 1))
        p.drawRoundedRect(pill, H / 2, H / 2)
        text, muted = ThemeColors.GLASS_TEXT.qcolor(), ThemeColors.GLASS_TEXT_MUTED.qcolor()
        p.setFont(self._font)
        cpu_text, cpu_w, cap_w, lab_w = self._cells()
        x0 = pill.left() + PADX
        p.setPen(text)
        p.drawText(QRectF(x0, 0, cpu_w + 2, H), int(Qt.AlignmentFlag.AlignVCenter), cpu_text)
        p.setPen(muted)
        p.drawText(QRectF(x0 + cpu_w + 18, 0, cap_w + 2, H), int(Qt.AlignmentFlag.AlignVCenter), self._presetCaption)
        for i, (rect, st) in enumerate(zip(self._buttonRects(), self._states)):
            h = self._hover[i].value
            if st["active"]:
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QColor(*pal.glow))
                p.drawEllipse(rect.adjusted(-3, -3, 3, 3))
                p.setBrush(QColor(*pal.base))
                p.drawEllipse(rect)
                p.setPen(QColor(*pal.on_accent))
            else:
                fill = ThemeColors.GLASS_ROW_HOVER.qcolor()
                fill.setAlphaF(min(1.0, 0.35 + 0.65 * h) if st["assigned"] else 0.18 + 0.3 * h)
                p.setBrush(fill)
                p.setPen(QPen(ThemeColors.GLASS_PILL_BORDER.qcolor(), 1))
                p.drawEllipse(rect)
                p.setPen(text if st["assigned"] else muted)
            p.drawText(rect, int(Qt.AlignmentFlag.AlignCenter), st["label"])
        p.setPen(muted)
        right = QRectF(pill.right() - PADX - lab_w - 2, 0, lab_w + 2, H)
        p.drawText(right, int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight), self._label)

    # ------------------------------------------------------------------ input
    def _slotAt(self, pos) -> int:
        for i, r in enumerate(self._buttonRects()):
            if r.adjusted(-2, -4, 2, 4).contains(QPointF(pos)):
                return i
        return -1

    def _setHot(self, i: int) -> None:
        if i == self._hot:
            return
        if 0 <= self._hot < len(self._hover):
            self._hover[self._hot].to(0.0, DUR_FAST)
        self._hot = i
        if 0 <= i < len(self._hover):
            self._hover[i].to(1.0, DUR_FAST)
        self.setCursor(Qt.CursorShape.PointingHandCursor if i >= 0 else Qt.CursorShape.ArrowCursor)

    def mouseMoveEvent(self, e) -> None:
        self._setHot(self._slotAt(e.position()))

    def leaveEvent(self, e) -> None:
        self._setHot(-1)
        super().leaveEvent(e)

    def mouseReleaseEvent(self, e) -> None:
        if e.button() == Qt.MouseButton.LeftButton:
            i = self._slotAt(e.position())
            if i >= 0:
                self.slotClicked.emit(i)

    def event(self, e) -> bool:
        if e.type() == QEvent.Type.ToolTip:
            i = self._slotAt(e.pos())
            if 0 <= i < len(self._states):
                st = self._states[i]
                QToolTip.showText(e.globalPos(), st["name"] if st["assigned"] else self._emptyTip, self)
            else:
                QToolTip.hideText()
            return True
        return super().event(e)
