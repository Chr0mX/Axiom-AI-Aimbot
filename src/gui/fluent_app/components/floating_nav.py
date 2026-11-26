"""Floating pill navigation (replaces the vendored left navigation panel).

* one rounded glass pill holding an icon per page;
* a single **accent capsule** slides to the active item (``DUR_SLOW`` tween, retargeted from where it
  currently is on rapid clicks) while the active item's label expands and the previous one folds;
* hover = soft wash, press = capsule dips; inactive items are icon-only with a tooltip (all ten labels
  would not fit the title row), the active one shows its label (``compact`` hides it on narrow windows);
* two orientations — ``top`` (horizontal, in the title row) and ``left`` (vertical rail); dragging the
  pill far enough emits ``dockRequested`` with the dock it was dropped on;
* arrow keys move the selection, so the nav is usable without a mouse.

The widget only *reports* the chosen route (``routeChanged``); the window switches the page and calls
``setCurrent`` again when the page changes by other means (remote conversion redirect, model page).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from PyQt6.QtCore import QEvent, QPoint, QPointF, QRectF, QSize, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QFontMetrics, QLinearGradient, QPainter, QPen
from PyQt6.QtWidgets import QToolTip, QWidget

from ..theme_colors import ThemeColors
from .glass import accent_palette, signals
from .icon_utils import tint_icon
from .motion import DUR_FAST, DUR_SLOW, Tween

ITEM = 38            # icon cell (square)
PAD = 4              # pill inner padding
GAP = 2
LABEL_GAP = 7
ICON = 18
DRAG_THRESHOLD = 12


@dataclass
class NavEntry:
    route: str
    icon: object
    label: str


class FloatingNav(QWidget):
    routeChanged = pyqtSignal(str)
    dockRequested = pyqtSignal(str)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._entries: List[NavEntry] = []
        self._expand: List[Tween] = []      # label expansion 0..1 per item
        self._hover: List[Tween] = []       # hover wash 0..1 per item
        self._current = -1
        self._hot = -1
        self._pressed = -1
        self._press_pos: Optional[QPoint] = None
        self._press_origin = QPoint()
        self._dragging = False
        self._orient = "top"
        self._compact = False
        self._cap = Tween(self, 1.0)            # capsule progress start->end
        self._cap_start = QRectF()
        self._cap_end = QRectF()
        self._press_dip = Tween(self, 0.0)
        self._cap.changed.connect(lambda _v: self.update())
        self._press_dip.changed.connect(lambda _v: self.update())
        self._icon_cache: Dict[Tuple[str, int], object] = {}
        self._font = QFont(self.font())
        self._font.setPixelSize(13)
        self._font.setWeight(QFont.Weight.DemiBold)
        signals.appearanceChanged.connect(self._invalidateIcons)

    # ------------------------------------------------------------------ model
    def setItems(self, entries: List[NavEntry]) -> None:
        self._entries = list(entries)
        self._expand = [self._tween(0.0) for _ in entries]
        self._hover = [self._tween(0.0) for _ in entries]
        self._current = -1
        self._icon_cache.clear()
        self.updateGeometry()
        self.update()

    def _tween(self, v: float) -> Tween:
        t = Tween(self, v)
        t.changed.connect(lambda _v: self.update())
        return t

    def retranslate(self, labels: Dict[str, str]) -> None:
        for e in self._entries:
            if e.route in labels:
                e.label = labels[e.route]
        self.updateGeometry()
        self._snapCapsule()
        self.update()

    def currentRoute(self) -> str:
        return self._entries[self._current].route if 0 <= self._current < len(self._entries) else ""

    def orientation(self) -> str:
        return self._orient

    def setOrientation(self, orient: str) -> None:
        orient = "left" if orient == "left" else "top"
        if orient == self._orient:
            return
        self._orient = orient
        self.updateGeometry()
        self._snapCapsule()
        self.update()

    def setCompact(self, compact: bool) -> None:
        """Hide the active label (narrow title row)."""
        if compact == self._compact:
            return
        self._compact = compact
        self.updateGeometry()
        self._snapCapsule()
        self.update()

    # ------------------------------------------------------------------ geometry
    def _labelWidth(self, i: int) -> float:
        if self._orient != "top" or self._compact:
            return 0.0
        return QFontMetrics(self._font).horizontalAdvance(self._entries[i].label) + LABEL_GAP + 6

    def _layout(self, fractions: Optional[List[float]] = None) -> Tuple[List[QRectF], float]:
        """Item rects (widget coordinates) and the pill's content extent for the given expansion fractions."""
        n = len(self._entries)
        fr = fractions if fractions is not None else [t.value for t in self._expand]
        rects: List[QRectF] = []
        if self._orient == "left":
            y = PAD
            for _ in range(n):
                rects.append(QRectF((self.width() - ITEM) / 2, y, ITEM, ITEM))
                y += ITEM + GAP
            return rects, (y - GAP + PAD if n else 2 * PAD)
        widths = [ITEM + self._labelWidth(i) * max(0.0, min(1.0, fr[i])) for i in range(n)]
        total = sum(widths) + GAP * max(0, n - 1) + 2 * PAD
        x = (self.width() - total) / 2 + PAD
        for w in widths:
            rects.append(QRectF(x, (self.height() - ITEM) / 2, w, ITEM))
            x += w + GAP
        return rects, total

    def _maxExtent(self) -> int:
        n = len(self._entries)
        if self._orient == "left":
            return int(2 * PAD + n * ITEM + max(0, n - 1) * GAP)
        widest = max((self._labelWidth(i) for i in range(n)), default=0.0)
        return int(2 * PAD + n * ITEM + max(0, n - 1) * GAP + widest)

    def sizeHint(self) -> QSize:
        if self._orient == "left":
            return QSize(ITEM + 2 * PAD, self._maxExtent())
        return QSize(self._maxExtent(), ITEM + 2 * PAD)

    def minimumSizeHint(self) -> QSize:
        return self.sizeHint()

    def preferredWidth(self, compact: bool) -> int:
        """Width the pill needs in the title row with/without the active label."""
        old = self._compact
        self._compact = compact
        w = self._maxExtent()
        self._compact = old
        return w

    def resizeEvent(self, e) -> None:
        super().resizeEvent(e)
        self._snapCapsule()

    def _finalRect(self, idx: int) -> QRectF:
        n = len(self._entries)
        fr = [1.0 if i == idx else 0.0 for i in range(n)]
        rects, _ = self._layout(fr)
        return rects[idx] if 0 <= idx < len(rects) else QRectF()

    def _snapCapsule(self) -> None:
        if 0 <= self._current < len(self._entries):
            for i, t in enumerate(self._expand):
                t.jump(1.0 if i == self._current and self._labelWidth(i) > 0 else 0.0)
            self._cap_end = self._finalRect(self._current)
            self._cap_start = self._cap_end
            self._cap.jump(1.0)

    # ------------------------------------------------------------------ selection
    def setCurrent(self, route: str, emit: bool = False, animate: bool = True) -> None:
        idx = next((i for i, e in enumerate(self._entries) if e.route == route), -1)
        if idx < 0 or idx == self._current:
            return
        old = self._current
        shown = self._capsuleRect()
        self._current = idx
        if not animate or old < 0:
            self._snapCapsule()
        else:
            self._cap_start = shown
            self._cap_end = self._finalRect(idx)
            for i, t in enumerate(self._expand):
                want = 1.0 if i == idx and self._labelWidth(i) > 0 else 0.0
                t.to(want, DUR_SLOW)
            self._cap.jump(0.0)
            self._cap.to(1.0, DUR_SLOW)
        self.update()
        if emit:
            self.routeChanged.emit(route)

    def _capsuleRect(self) -> QRectF:
        t = self._cap.value
        a, b = self._cap_start, self._cap_end
        if a.isNull():
            return b
        return QRectF(a.x() + (b.x() - a.x()) * t, a.y() + (b.y() - a.y()) * t,
                      a.width() + (b.width() - a.width()) * t, a.height() + (b.height() - a.height()) * t)

    # ------------------------------------------------------------------ painting
    def _invalidateIcons(self) -> None:
        self._icon_cache.clear()
        self.update()

    def _icon(self, i: int, color: QColor):
        key = (self._entries[i].route, color.rgba())
        pm = self._icon_cache.get(key)
        if pm is None:
            pm = tint_icon(self._entries[i].icon, color, ICON)
            if len(self._icon_cache) > 64:
                self._icon_cache.clear()
            self._icon_cache[key] = pm
        return pm

    def paintEvent(self, e) -> None:
        if not self._entries:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        pal = accent_palette()
        rects, extent = self._layout()
        vertical = self._orient == "left"
        if vertical:
            pill = QRectF(0.5, 0.5, self.width() - 1, extent - 1)
        else:
            pill = QRectF((self.width() - extent) / 2 + 0.5, 0.5, extent - 1, self.height() - 1)
        radius = min(pill.width(), pill.height()) / 2 if not vertical else (ITEM + 2 * PAD) / 2
        p.setBrush(ThemeColors.GLASS_PILL.qcolor())
        p.setPen(QPen(ThemeColors.GLASS_PILL_BORDER.qcolor(), 1))
        p.drawRoundedRect(pill, radius, radius)

        # accent capsule (glow rings + gradient fill), dipping a little while pressed
        if self._current >= 0:
            cap = self._capsuleRect()
            dip = self._press_dip.value * 1.5
            cap = cap.adjusted(dip, dip, -dip, -dip)
            r = min(cap.width(), cap.height()) / 2
            p.setPen(Qt.PenStyle.NoPen)
            for grow, alpha in ((5, 18), (3, 30), (1.5, 44)):
                c = QColor(*pal.base, alpha)
                p.setBrush(c)
                p.drawRoundedRect(cap.adjusted(-grow, -grow, grow, grow), r + grow, r + grow)
            g = QLinearGradient(cap.topLeft(), cap.topRight() if not vertical else cap.bottomLeft())
            g.setColorAt(0, QColor(*pal.hi))
            g.setColorAt(1, QColor(*pal.base))
            p.setBrush(g)
            p.drawRoundedRect(cap, r, r)

        muted = ThemeColors.GLASS_TEXT_MUTED.qcolor()
        strong = ThemeColors.GLASS_TEXT.qcolor()
        on = QColor(*pal.on_accent)
        for i, rect in enumerate(rects):
            h = self._hover[i].value
            if h > 0.01 and i != self._current:
                wash = ThemeColors.GLASS_ROW_HOVER.qcolor()
                wash.setAlphaF(min(1.0, wash.alphaF() * 1.6 * h))
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(wash)
                p.drawRoundedRect(rect, min(rect.width(), rect.height()) / 2, min(rect.width(), rect.height()) / 2)
            active = i == self._current
            frac = self._expand[i].value if active else 0.0
            if active:
                color = on
            else:
                color = QColor(muted)
                color = QColor(
                    int(muted.red() + (strong.red() - muted.red()) * h),
                    int(muted.green() + (strong.green() - muted.green()) * h),
                    int(muted.blue() + (strong.blue() - muted.blue()) * h))
            pm = self._icon(i, color)
            p.drawPixmap(QPointF(rect.x() + (ITEM - ICON) / 2, rect.y() + (rect.height() - ICON) / 2), pm)
            if active and frac > 0.02 and not vertical and not self._compact:
                p.setFont(self._font)
                tc = QColor(on)
                tc.setAlphaF(max(0.0, min(1.0, (frac - 0.25) / 0.75)))
                p.setPen(tc)
                text_rect = QRectF(rect.x() + ITEM + LABEL_GAP - 4, rect.y(), max(0.0, rect.width() - ITEM - LABEL_GAP + 4), rect.height())
                p.drawText(text_rect, int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft),
                           self._entries[i].label)
        if self.hasFocus() and self._current >= 0:
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.setPen(QPen(QColor(*pal.base, 200), 1.5, Qt.PenStyle.DashLine))
            p.drawRoundedRect(pill.adjusted(-1, -1, 1, 1), radius + 1, radius + 1)

    # ------------------------------------------------------------------ input
    def _indexAt(self, pos: QPoint) -> int:
        rects, _ = self._layout()
        for i, r in enumerate(rects):
            if r.adjusted(-GAP / 2, -4, GAP / 2, 4).contains(QPointF(pos)):
                return i
        return -1

    def _setHot(self, idx: int) -> None:
        if idx == self._hot:
            return
        if 0 <= self._hot < len(self._hover):
            self._hover[self._hot].to(0.0, DUR_FAST)
        self._hot = idx
        if 0 <= idx < len(self._hover):
            self._hover[idx].to(1.0, DUR_FAST)

    def enterEvent(self, e) -> None:
        super().enterEvent(e)

    def leaveEvent(self, e) -> None:
        self._setHot(-1)
        super().leaveEvent(e)

    def mouseMoveEvent(self, e) -> None:
        if self._press_pos is not None and e.buttons() & Qt.MouseButton.LeftButton:
            delta = e.globalPosition().toPoint() - self._press_pos
            if not self._dragging and (abs(delta.x()) > DRAG_THRESHOLD or abs(delta.y()) > DRAG_THRESHOLD):
                self._dragging = True
                self._press_dip.to(0.0, DUR_FAST)
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
            if self._dragging:
                self.move(self._press_origin + delta)
            return
        self._setHot(self._indexAt(e.position().toPoint()))

    def mousePressEvent(self, e) -> None:
        if e.button() == Qt.MouseButton.LeftButton:
            self._press_pos = e.globalPosition().toPoint()
            self._press_origin = self.pos()
            self._pressed = self._indexAt(e.position().toPoint())
            self._dragging = False
            self.setFocus(Qt.FocusReason.MouseFocusReason)
            if self._pressed >= 0:
                self._press_dip.to(1.0, DUR_FAST)
        e.accept()

    def mouseReleaseEvent(self, e) -> None:
        if e.button() != Qt.MouseButton.LeftButton:
            return
        was_drag, idx = self._dragging, self._pressed
        self._press_pos, self._pressed, self._dragging = None, -1, False
        self._press_dip.to(0.0, DUR_FAST)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        if was_drag:
            self.dockRequested.emit(self._dockFromPosition())
            return
        if idx >= 0 and idx == self._indexAt(e.position().toPoint()):
            route = self._entries[idx].route
            if idx != self._current:
                self.setCurrent(route, emit=True)

    def _dockFromPosition(self) -> str:
        """'left' when the pill was dropped over the left fifth of the window below the title row."""
        win = self.window()
        c = self.mapTo(win, self.rect().center())
        return "left" if c.x() < win.width() * 0.2 and c.y() > 56 else "top"

    def keyPressEvent(self, e) -> None:
        step = {Qt.Key.Key_Left: -1, Qt.Key.Key_Up: -1, Qt.Key.Key_Right: 1, Qt.Key.Key_Down: 1}.get(e.key())
        if step and self._entries:
            idx = (max(self._current, 0) + step) % len(self._entries)
            self.setCurrent(self._entries[idx].route, emit=True)
            return
        super().keyPressEvent(e)

    def focusInEvent(self, e) -> None:
        super().focusInEvent(e)
        self.update()

    def focusOutEvent(self, e) -> None:
        super().focusOutEvent(e)
        self.update()

    def event(self, e) -> bool:
        if e.type() == QEvent.Type.ToolTip:
            idx = self._indexAt(e.pos())
            if idx >= 0 and (idx != self._current or self._compact or self._orient == "left"):
                QToolTip.showText(e.globalPos(), self._entries[idx].label, self)
            else:
                QToolTip.hideText()
            return True
        return super().event(e)

