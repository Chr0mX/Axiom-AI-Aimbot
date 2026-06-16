from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QDialog, QLabel, QPushButton, QToolButton, QVBoxLayout, QWidget,
)


def _frame_to_pixmap(frame, max_w: int, max_h: int) -> "QPixmap | None":
    """Convert a BGRA/BGR numpy frame to a scaled QPixmap. Returns None on error."""
    if frame is None or frame.ndim < 3:
        return None
    h, w = frame.shape[:2]
    ch = frame.shape[2]
    if ch == 4:
        fmt = QImage.Format.Format_ARGB32  # ARGB32 == B,G,R,A in memory on x86
        bpl = w * 4
    elif ch == 3:
        fmt = QImage.Format.Format_BGR888
        bpl = w * 3
    else:
        return None
    img = QImage(frame.data, w, h, bpl, fmt).copy()
    return QPixmap.fromImage(img).scaled(
        max_w, max_h,
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )


def _apply_crop(frame, config) -> "frame":
    """Return frame cropped to detection region if the crop toggle is on."""
    from core.screen_capture import get_preview_region
    if not getattr(config, 'preview_crop_to_detection', False):
        return frame
    region = get_preview_region()
    if not region:
        return frame
    l = max(0, int(region.get('left', 0)))
    t = max(0, int(region.get('top', 0)))
    w = max(1, int(region.get('width', frame.shape[1])))
    h = max(1, int(region.get('height', frame.shape[0])))
    cropped = frame[t:min(frame.shape[0], t + h), l:min(frame.shape[1], l + w)]
    return cropped if cropped.size else frame


class PreviewPopOutWindow(QDialog):
    """Standalone floating window showing the live capture preview."""

    def __init__(self, config, on_close=None, parent=None):
        super().__init__(parent)
        self._config = config
        self._on_close = on_close
        self.setWindowTitle("Axiom – Capture Preview")
        self.resize(640, 360)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        self._label = QLabel()
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setStyleSheet("background: #111;")
        self._label.setText("No signal")
        layout.addWidget(self._label, stretch=1)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh)
        self._timer.start(200)

    def _refresh(self):
        from core.screen_capture import get_preview_frame
        frame = get_preview_frame()
        if frame is None:
            self._label.setText("No signal")
            return
        try:
            frame = _apply_crop(frame, self._config)
        except Exception:
            pass
        pw, ph = self.width() - 8, self.height() - 8
        pixmap = _frame_to_pixmap(frame, pw, ph)
        if pixmap:
            self._label.setPixmap(pixmap)
        else:
            self._label.setText("No signal")

    def closeEvent(self, event):
        self._timer.stop()
        if callable(self._on_close):
            self._on_close()
        super().closeEvent(event)


class CapturePreviewPanel(QWidget):
    """Right-side live preview for NDI / UVC capture feeds.

    Visibility controlled by window.py based on screenshot_method and
    uvc_show_window config. Crop toggled by preview_crop_to_detection config.
    """

    _PANEL_W = 240
    _IMG_W   = 220
    _IMG_H   = 135   # 16:9 at 220 px wide

    def __init__(self, config=None, parent=None):
        super().__init__(parent)
        self._config = config
        self._popout: "PreviewPopOutWindow | None" = None
        self.setFixedWidth(self._PANEL_W)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 12, 8, 8)
        layout.setSpacing(6)

        header = QLabel("Preview")
        header.setStyleSheet("font-weight: bold; font-size: 11px;")
        layout.addWidget(header)

        self._frame_label = QLabel()
        self._frame_label.setFixedSize(self._IMG_W, self._IMG_H)
        self._frame_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._frame_label.setStyleSheet(
            "background: #111; border: 1px solid #333; border-radius: 4px;"
        )
        self._frame_label.setText("No signal")
        layout.addWidget(self._frame_label)

        layout.addStretch(1)

        self._popout_btn = QPushButton("Pop out")
        self._popout_btn.setFixedHeight(28)
        self._popout_btn.clicked.connect(self._onPopOut)
        layout.addWidget(self._popout_btn)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh)

    def setConfig(self, config) -> None:
        self._config = config

    def start(self) -> None:
        if not self._popout:
            self._timer.start(200)

    def stop(self) -> None:
        self._timer.stop()
        self._frame_label.clear()
        self._frame_label.setText("No signal")

    def _refresh(self) -> None:
        from core.screen_capture import get_preview_frame
        frame = get_preview_frame()
        if frame is None:
            self._frame_label.setText("No signal")
            return
        try:
            if self._config is not None:
                frame = _apply_crop(frame, self._config)
        except Exception:
            pass
        pixmap = _frame_to_pixmap(frame, self._IMG_W, self._IMG_H)
        if pixmap:
            self._frame_label.setPixmap(pixmap)
        else:
            self._frame_label.setText("No signal")

    def _onPopOut(self) -> None:
        if self._popout and self._popout.isVisible():
            return
        self._timer.stop()
        self._frame_label.setText("Preview in pop-out")
        self._popout = PreviewPopOutWindow(
            config=self._config,
            on_close=self._onPopOutClosed,
            parent=None,   # top-level so it floats independently
        )
        self._popout.show()

    def _onPopOutClosed(self) -> None:
        self._popout = None
        self._frame_label.setText("No signal")
        self._timer.start(200)
