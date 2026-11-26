"""Title-bar buttons for the glass window.

``style_caption_buttons`` repaints the vendored minimise / maximise / close buttons as round-washed
glyphs but keeps the widgets themselves (same size and position), because the frameless window's
native maximise hit-testing (Windows 11 snap layouts) is tied to them. ``CaptionIconButton`` is the
matching round icon button used for theme, language and the overflow menu.
"""
from __future__ import annotations

import types

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import QAbstractButton, QWidget

from ..theme_colors import ThemeColors
from .glass import signals
from .icon_utils import tint_icon
from .motion import DUR_FAST, HoverProgress

CIRCLE = 28


def _wash(strength: float, danger: bool = False) -> QColor:
    if danger:
        return QColor(235, 77, 100, int(170 * strength))
    c = ThemeColors.GLASS_ROW_HOVER.qcolor()
    c.setAlphaF(min(1.0, c.alphaF() * 1.8 * strength))
    return c


def _paint_caption(self, e) -> None:
    from qframelesswindow.titlebar.title_bar_buttons import TitleBarButtonState as S
    p = QPainter(self)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    hover = self._state in (S.HOVER, S.PRESSED)
    strength = 1.0 if self._state == S.PRESSED else (0.7 if hover else 0.0)
    danger = self._glyph == "close"
    if strength > 0:
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(_wash(strength, danger))
        cx, cy = self.width() / 2, self.height() / 2
        p.drawEllipse(QRectF(cx - CIRCLE / 2, cy - CIRCLE / 2, CIRCLE, CIRCLE))
    color = QColor(255, 255, 255) if (danger and hover) else (
        ThemeColors.GLASS_TEXT.qcolor() if hover else ThemeColors.GLASS_TEXT_MUTED.qcolor())
    pen = QPen(color, 1.4)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    p.setPen(pen)
    p.setBrush(Qt.BrushStyle.NoBrush)
    cx, cy = self.width() / 2, self.height() / 2
    g = 5.0
    if self._glyph == "min":
        p.drawLine(QPointF(cx - g, cy), QPointF(cx + g, cy))
    elif self._glyph == "max":
        if getattr(self, "_isMax", False):
            p.drawRoundedRect(QRectF(cx - g + 1.5, cy - g, 2 * g - 1.5, 2 * g - 1.5), 1.5, 1.5)
            p.drawLine(QPointF(cx - g, cy - g + 3), QPointF(cx - g, cy + g))
            p.drawLine(QPointF(cx - g, cy + g), QPointF(cx + g - 3, cy + g))
        else:
            p.drawRoundedRect(QRectF(cx - g, cy - g, 2 * g, 2 * g), 2, 2)
    else:
        p.drawLine(QPointF(cx - g + 0.5, cy - g + 0.5), QPointF(cx + g - 0.5, cy + g - 0.5))
        p.drawLine(QPointF(cx - g + 0.5, cy + g - 0.5), QPointF(cx + g - 0.5, cy - g + 0.5))


def style_caption_buttons(title_bar) -> None:
    for btn, glyph in ((title_bar.minBtn, "min"), (title_bar.maxBtn, "max"), (title_bar.closeBtn, "close")):
        btn._glyph = glyph
        btn.paintEvent = types.MethodType(_paint_caption, btn)
        btn.update()


class CaptionIconButton(QAbstractButton):
    """Round 32x32 icon button with an eased hover wash."""

    def __init__(self, icon, tooltip: str = "", parent: QWidget | None = None):
        super().__init__(parent)
        self._icon = icon
        self.setFixedSize(34, 32)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip(tooltip)
        self._hover = HoverProgress(self, DUR_FAST)
        self._pressed = False
        signals.appearanceChanged.connect(self.update)

    def setIconSource(self, icon) -> None:
        self._icon = icon
        self.update()

    def enterEvent(self, e):
        self._hover.set(True)
        super().enterEvent(e)

    def leaveEvent(self, e):
        self._hover.set(False)
        super().leaveEvent(e)

    def mousePressEvent(self, e):
        self._pressed = True
        self.update()
        super().mousePressEvent(e)

    def mouseReleaseEvent(self, e):
        self._pressed = False
        self.update()
        super().mouseReleaseEvent(e)

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        h = max(self._hover.value, 1.0 if self._pressed else 0.0)
        if h > 0.01:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(_wash(h))
            p.drawEllipse(QRectF((self.width() - CIRCLE) / 2, (self.height() - CIRCLE) / 2, CIRCLE, CIRCLE))
        muted, strong = ThemeColors.GLASS_TEXT_MUTED.qcolor(), ThemeColors.GLASS_TEXT.qcolor()
        c = QColor(int(muted.red() + (strong.red() - muted.red()) * h),
                   int(muted.green() + (strong.green() - muted.green()) * h),
                   int(muted.blue() + (strong.blue() - muted.blue()) * h))
        pm = tint_icon(self._icon, c, 16)
        p.drawPixmap(QPointF((self.width() - 16) / 2, (self.height() - 16) / 2), pm)
