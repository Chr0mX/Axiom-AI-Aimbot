from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget


class CapturePreviewPanel(QWidget):
    """Right-side live preview for NDI / UVC capture feeds.

    Polls the module-level preview frame from screen_capture at 5 FPS and
    renders it into a QLabel. Hidden entirely when the active capture method
    is not ndi or uvc.
    """

    _PANEL_W  = 240
    _IMG_W    = 220
    _IMG_H    = 135  # 16:9 at 220 px wide

    def __init__(self, parent=None):
        super().__init__(parent)
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

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh)

    def start(self) -> None:
        self._timer.start(200)

    def stop(self) -> None:
        self._timer.stop()
        self._frame_label.clear()
        self._frame_label.setText("No signal")

    def _refresh(self) -> None:
        from core.screen_capture import get_preview_frame
        frame = get_preview_frame()
        if frame is None or frame.ndim < 3:
            self._frame_label.setText("No signal")
            return

        h, w = frame.shape[:2]
        ch = frame.shape[2]
        if ch == 4:
            fmt = QImage.Format.Format_BGRA8888
            bpl = w * 4
        elif ch == 3:
            fmt = QImage.Format.Format_BGR888
            bpl = w * 3
        else:
            return

        img = QImage(frame.data, w, h, bpl, fmt).copy()
        pixmap = QPixmap.fromImage(img).scaled(
            self._IMG_W,
            self._IMG_H,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._frame_label.setPixmap(pixmap)
