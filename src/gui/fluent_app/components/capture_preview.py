import time as _time

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QDialog, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QVBoxLayout, QWidget,
)

from ..language_manager import t


def _frame_to_pixmap(frame, max_w: int, max_h: int) -> "QPixmap | None":
    """Convert a BGRA/BGR numpy frame to a scaled QPixmap. Returns None on error."""
    import numpy as np
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
    # tobytes() forces a contiguous copy — PyQt6 needs bytes, not memoryview
    data = np.ascontiguousarray(frame).tobytes()
    img = QImage(data, w, h, bpl, fmt)
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


def _capture_interval_ms(config) -> int:
    """Return capture interval in ms, honouring preview_fps_cap (0 = uncapped)."""
    base = max(1, int(getattr(config, 'screenshot_interval', 0.016) * 1000))
    cap = getattr(config, 'preview_fps_cap', 0)
    if cap > 0:
        return max(base, 1000 // cap)
    return base


class _FpsMixin:
    """Mixin that tracks rendered-frame rate and updates a QLabel once per second."""

    def _fps_init(self, label: QLabel) -> None:
        self._fps_label   = label
        self._fps_count   = 0
        self._fps_last_ts = _time.perf_counter()

    def _fps_tick(self) -> None:
        self._fps_count += 1
        now = _time.perf_counter()
        elapsed = now - self._fps_last_ts
        if elapsed >= 1.0:
            fps = self._fps_count / elapsed
            self._fps_label.setText(f"{fps:.0f} fps")
            self._fps_count   = 0
            self._fps_last_ts = now

    def _fps_reset(self) -> None:
        self._fps_count   = 0
        self._fps_last_ts = _time.perf_counter()
        self._fps_label.setText(t("preview_fps_placeholder", "-- fps"))


class PreviewPopOutWindow(_FpsMixin, QDialog):
    """Standalone floating window showing the live capture preview."""

    def __init__(self, config, on_close=None, parent=None):
        super().__init__(parent)
        self._config = config
        self._on_close = on_close
        self.setWindowTitle(t("preview_popout_title", "Axiom – Capture Preview"))
        self.resize(640, 360)

        self._always_on_top = bool(getattr(config, 'uvc_always_on_top', True))
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, self._always_on_top)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)

        fps_label = QLabel(t("preview_fps_placeholder", "-- fps"))
        fps_label.setStyleSheet("font-size: 9px; color: #888;")
        fps_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        layout.addWidget(fps_label)

        self._label = QLabel()
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setStyleSheet("background: #111;")
        self._label.setText(t("preview_no_signal", "No signal"))
        layout.addWidget(self._label, stretch=1)

        self._fps_init(fps_label)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh)
        self._timer.start(_capture_interval_ms(config))

    def _refresh(self):
        wanted_on_top = bool(getattr(self._config, 'uvc_always_on_top', True))
        if wanted_on_top != self._always_on_top:
            self._always_on_top = wanted_on_top
            self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, wanted_on_top)
            self.show()  # re-apply native window flags — required after setWindowFlag()

        from core.screen_capture import get_preview_frame
        frame = get_preview_frame()
        if frame is None:
            self._label.setText(t("preview_no_signal", "No signal"))
            return
        try:
            frame = _apply_crop(frame, self._config)
        except Exception:
            pass
        pw, ph = self.width() - 8, self.height() - 8
        pixmap = _frame_to_pixmap(frame, pw, ph)
        if pixmap:
            self._label.setPixmap(pixmap)
            self._fps_tick()
        else:
            self._label.setText(t("preview_no_signal", "No signal"))

    def closeEvent(self, event):
        self._timer.stop()
        if callable(self._on_close):
            self._on_close()
        super().closeEvent(event)


class CapturePreviewPanel(_FpsMixin, QWidget):
    """Right-side live preview for NDI / UVC capture feeds.

    Visibility controlled by window.py based on screenshot_method and
    uvc_show_window config. Crop toggled by preview_crop_to_detection config.
    Responsive width — expands when the left navigation is collapsed.
    """

    def __init__(self, config=None, parent=None):
        super().__init__(parent)
        self._config = config
        self._popout: "PreviewPopOutWindow | None" = None
        self.setMinimumWidth(180)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 12, 8, 8)
        layout.setSpacing(4)

        # Header row: "Preview" label + FPS counter
        header_row = QHBoxLayout()
        header_row.setContentsMargins(0, 0, 0, 0)
        self._header_label = QLabel(t("preview_header", "Preview"))
        self._header_label.setStyleSheet("font-weight: bold; font-size: 11px;")
        fps_label = QLabel(t("preview_fps_placeholder", "-- fps"))
        fps_label.setStyleSheet("font-size: 9px; color: #888;")
        fps_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        header_row.addWidget(self._header_label)
        header_row.addStretch(1)
        header_row.addWidget(fps_label)
        layout.addLayout(header_row)

        self._frame_label = QLabel()
        self._frame_label.setMinimumSize(160, 90)
        self._frame_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._frame_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._frame_label.setStyleSheet(
            "background: #111; border: 1px solid #333; border-radius: 4px;"
        )
        self._frame_label.setText(t("preview_no_signal", "No signal"))
        layout.addWidget(self._frame_label, stretch=1)

        self._popout_btn = QPushButton(t("preview_popout_btn", "Pop out"))
        self._popout_btn.setFixedHeight(28)
        self._popout_btn.clicked.connect(self._onPopOut)
        layout.addWidget(self._popout_btn)

        self._fps_init(fps_label)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh)

    def setConfig(self, config) -> None:
        self._config = config

    def retranslateUi(self) -> None:
        """Re-apply static labels after a language switch.

        The frame/FPS labels don't need this — they're refreshed continuously
        by the capture timer, which already reads the current language on
        every tick — but the header and pop-out button are set once and
        never touched again otherwise.
        """
        self._header_label.setText(t("preview_header", "Preview"))
        self._popout_btn.setText(t("preview_popout_btn", "Pop out"))

    def start(self) -> None:
        if not self._popout:
            self._fps_reset()
            ms = _capture_interval_ms(self._config)
            self._timer.start(ms)

    def stop(self) -> None:
        self._timer.stop()
        self._frame_label.clear()
        self._frame_label.setText(t("preview_no_signal", "No signal"))
        self._fps_reset()

    def applyFpsCap(self) -> None:
        if self._timer.isActive():
            self._timer.start(_capture_interval_ms(self._config))
        if self._popout and self._popout.isVisible():
            self._popout._timer.start(_capture_interval_ms(self._config))

    def _refresh(self) -> None:
        from core.screen_capture import get_preview_frame
        frame = get_preview_frame()
        if frame is None:
            self._frame_label.setText(t("preview_no_signal", "No signal"))
            return
        try:
            if self._config is not None:
                frame = _apply_crop(frame, self._config)
        except Exception:
            pass
        pw = max(160, self._frame_label.width())
        ph = max(90,  self._frame_label.height())
        pixmap = _frame_to_pixmap(frame, pw, ph)
        if pixmap:
            self._frame_label.setPixmap(pixmap)
            self._fps_tick()
        else:
            self._frame_label.setText(t("preview_no_signal", "No signal"))

    def _onPopOut(self) -> None:
        if self._popout and self._popout.isVisible():
            return
        self._timer.stop()
        self._fps_reset()
        self._frame_label.setText("Preview in pop-out")
        self._popout = PreviewPopOutWindow(
            config=self._config,
            on_close=self._onPopOutClosed,
            parent=None,
        )
        self._popout.show()

    def _onPopOutClosed(self) -> None:
        self._popout = None
        self._frame_label.setText(t("preview_no_signal", "No signal"))
        self._fps_reset()
        ms = _capture_interval_ms(self._config)
        self._timer.start(ms)
