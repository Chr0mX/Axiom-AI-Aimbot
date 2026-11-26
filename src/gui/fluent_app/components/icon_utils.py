"""Icon helpers for the glass chrome."""
from __future__ import annotations

from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QColor, QIcon, QPainter, QPixmap
from qfluentwidgets import FluentIconBase


def to_qicon(icon) -> QIcon:
    """A ``QIcon`` from a FluentIcon / path / QIcon."""
    if isinstance(icon, QIcon):
        return icon
    if isinstance(icon, FluentIconBase):
        return icon.icon()
    return QIcon(str(icon))


def tint_icon(icon, color: QColor, size: int = 18) -> QPixmap:
    """``icon`` rendered at ``size`` px and recoloured to ``color`` (keeps the alpha shape), so the same
    SVG works in light and dark and takes the accent colour."""
    src = to_qicon(icon).pixmap(QSize(size, size))
    out = QPixmap(src.size())
    out.fill(Qt.GlobalColor.transparent)
    p = QPainter(out)
    p.drawPixmap(0, 0, src)
    p.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)
    p.fillRect(out.rect(), color)
    p.end()
    return out
