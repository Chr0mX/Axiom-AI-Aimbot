"""Accent colour control for the Appearance group: preset swatches + hex field + custom picker.

Emits ``colorChanged('#rrggbb')`` only for user edits; ``setColor`` updates the display silently so
loading a config can never write back.
"""
from __future__ import annotations

from PyQt6.QtCore import QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QConicalGradient, QPainter, QPen
from PyQt6.QtWidgets import QAbstractButton, QColorDialog, QHBoxLayout, QWidget
from qfluentwidgets import LineEdit

from core import ui_accent
from ..theme_colors import ThemeColors
from .glass import signals

SIZE = 24


class _Swatch(QAbstractButton):
    """Round colour dot; ``custom`` draws a rainbow ring and opens the colour dialog instead."""

    def __init__(self, hex_color: str, tooltip: str, custom: bool = False, parent: QWidget | None = None):
        super().__init__(parent)
        self.hex = hex_color
        self.custom = custom
        self.setCheckable(True)
        self.setFixedSize(SIZE + 6, SIZE + 6)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip(tooltip)

    def paintEvent(self, e) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(3, 3, SIZE, SIZE)
        p.setPen(Qt.PenStyle.NoPen)
        if self.custom:
            g = QConicalGradient(r.center(), 0)
            for i in range(7):
                g.setColorAt(i / 6, QColor.fromHsv(int(i * 60) % 360, 190, 245))
            p.setBrush(g)
            p.drawEllipse(r)
            p.setBrush(QColor(0, 0, 0, 90))
            p.drawEllipse(r.adjusted(7, 7, -7, -7))
        else:
            p.setBrush(QColor(self.hex))
            p.drawEllipse(r)
        if self.isChecked():
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.setPen(QPen(ThemeColors.GLASS_TEXT.qcolor(), 2))
            p.drawEllipse(QRectF(1, 1, SIZE + 4, SIZE + 4))
        elif self.underMouse():
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.setPen(QPen(ThemeColors.GLASS_PILL_BORDER.qcolor(), 1.5))
            p.drawEllipse(QRectF(1.5, 1.5, SIZE + 3, SIZE + 3))


class AccentPicker(QWidget):
    colorChanged = pyqtSignal(str)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)
        self._swatches = []
        for label, hx in ui_accent.PRESET_ACCENTS:
            sw = _Swatch(hx, f"{label} {hx}", parent=self)
            sw.clicked.connect(lambda _c=False, h=hx: self._choose(h))
            self._swatches.append(sw)
            row.addWidget(sw)
        self._custom = _Swatch("", "…", custom=True, parent=self)
        self._custom.clicked.connect(self._pickCustom)
        row.addWidget(self._custom)
        self._edit = LineEdit(self)
        self._edit.setFixedWidth(92)
        self._edit.setMaxLength(7)
        self._edit.setPlaceholderText("#7c6cf0")
        self._edit.editingFinished.connect(self._onEdited)
        row.addSpacing(6)
        row.addWidget(self._edit)
        self._color = ui_accent.get_accent()
        self.setColor(self._color)
        signals.appearanceChanged.connect(self.update)

    # -- state
    def color(self) -> str:
        return self._color

    def setColor(self, hex_color: str) -> None:
        self._color = ui_accent.to_hex(ui_accent.parse_hex(hex_color))
        for sw in self._swatches:
            sw.setChecked(sw.hex.lower() == self._color)
        self._custom.setChecked(not any(sw.isChecked() for sw in self._swatches))
        if not self._edit.hasFocus():
            self._edit.setText(self._color)
        self.update()

    def setToolTips(self, custom: str) -> None:
        self._custom.setToolTip(custom)

    # -- user actions
    def _choose(self, hex_color: str) -> None:
        self.setColor(hex_color)
        self.colorChanged.emit(self._color)

    def _pickCustom(self) -> None:
        c = QColorDialog.getColor(QColor(self._color), self.window())
        if c.isValid():
            self._choose(c.name())
        else:
            self.setColor(self._color)

    def _onEdited(self) -> None:
        text = self._edit.text().strip()
        valid = ui_accent.parse_hex(text, default="#000001") != (0, 0, 1)
        if not valid:
            self._edit.setText(self._color)       # reject junk, keep the current accent
            return
        self._choose(ui_accent.to_hex(ui_accent.parse_hex(text)))
