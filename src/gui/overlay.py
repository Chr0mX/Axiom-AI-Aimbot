# overlay.py
"""Overlay Module - Draws FOV boxes and detection boxes"""

from __future__ import annotations

import queue
from typing import List, TYPE_CHECKING

from PyQt6.QtWidgets import QApplication, QWidget
from PyQt6.QtGui import QPainter, QColor, QPen, QFont
from PyQt6.QtCore import Qt, QTimer
import ctypes

if TYPE_CHECKING:
    from core.config import Config

# Import theme color definitions
try:
    from gui.fluent_app.theme_colors import ThemeColors, get_rgba_qcolor
    HAS_THEME_COLORS = True
except ImportError:
    HAS_THEME_COLORS = False


class OverlayColors:
    """Overlay Color Configuration - Integrated with ThemeColors for unified management"""
    
    @staticmethod
    def get_fov_color() -> QColor:
        """FOV box color"""
        if HAS_THEME_COLORS:
            return ThemeColors.OVERLAY_FOV.qcolor()
        return QColor(255, 0, 0, 180)
    
    @staticmethod
    def get_box_color() -> QColor:
        """Detection box color"""
        if HAS_THEME_COLORS:
            return ThemeColors.OVERLAY_BOX.qcolor()
        return QColor(0, 255, 0, 200)
    
    @staticmethod
    def get_confidence_text_color() -> QColor:
        """Confidence text color"""
        if HAS_THEME_COLORS:
            return ThemeColors.OVERLAY_CONFIDENCE_TEXT.qcolor()
        return QColor(255, 255, 0, 220)
    
    @staticmethod
    def get_detect_range_color() -> QColor:
        """Detection range color"""
        if HAS_THEME_COLORS:
            return ThemeColors.OVERLAY_DETECT_RANGE.qcolor()
        return QColor(0, 140, 255, 90)
    
    @staticmethod
    def get_tracker_line_color() -> QColor:
        """Tracker line color"""
        if HAS_THEME_COLORS:
            return ThemeColors.OVERLAY_TRACKER_LINE.qcolor()
        return QColor(255, 255, 255, 50)
    
    @staticmethod
    def get_tracker_current_color() -> QColor:
        """Current observation position color"""
        if HAS_THEME_COLORS:
            return ThemeColors.OVERLAY_TRACKER_CURRENT.qcolor()
        return QColor(0, 255, 255, 60)
    
    @staticmethod
    def get_tracker_predicted_color() -> QColor:
        """Predicted position color"""
        if HAS_THEME_COLORS:
            return ThemeColors.OVERLAY_TRACKER_PREDICTED.qcolor()
        return QColor(255, 0, 255, 80)

    @staticmethod
    def get_tracer_color() -> QColor:
        """Tracer line color (screen center → target)"""
        return QColor(255, 255, 255, 200)


# Predefined box color themes (name → RGBA tuple)
_BOX_THEMES = {
    "default": None,          # uses ThemeColors.OVERLAY_BOX
    "cyan":    (0, 220, 255, 220),
    "red":     (255, 60, 60, 220),
    "yellow":  (255, 210, 0, 220),
    "white":   (255, 255, 255, 200),
    "purple":  (180, 60, 255, 210),
}

class PyQtOverlay(QWidget):
    def __init__(self, boxes_queue, confidences_queue, config):
        super().__init__()
        self.boxes_queue = boxes_queue
        self.confidences_queue = confidences_queue
        self.config = config
        
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        
        self.setGeometry(0, 0, config.width, config.height)
        self.boxes = []
        self.confidences = []
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_overlay)
        
        # 使用與檢測間隔一致的更新頻率
        update_interval_ms = max(int(config.detect_interval * 1000), 16)  # 最小16ms (約60fps)
        self._last_timer_interval_ms = update_interval_ms
        self.timer.start(update_interval_ms)
        
        self.show()
        self.set_click_through()

    def set_click_through(self):
        try:
            hwnd = self.winId().__int__()
            GWL_EXSTYLE = -20
            WS_EX_LAYERED = 0x80000
            WS_EX_TRANSPARENT = 0x20
            style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            style |= WS_EX_LAYERED | WS_EX_TRANSPARENT
            ctypes.windll.user32.SetWindowLongW(hwnd, GWL_EXSTYLE, style)
        except Exception as e:
            print(f"Failed to set mouse click-through: {e}")

    def update_overlay(self) -> None:
        """Fetch latest detection results from the queue and update display"""
        if getattr(self.config, 'screenshot_method', 'mss') in ('uvc', 'ndi'):
            if self.isVisible():
                self.hide()
            return
        if not self.isVisible():
            self.show()

        desired_interval = max(int(self.config.detect_interval * 1000), 16)
        if desired_interval != self._last_timer_interval_ms:
            self.timer.setInterval(desired_interval)
            self._last_timer_interval_ms = desired_interval

        new_boxes = None
        new_confidences = None
        
        try:
            new_boxes = self.boxes_queue.get_nowait()
        except queue.Empty:
            pass
            
        try:
            new_confidences = self.confidences_queue.get_nowait()
        except queue.Empty:
            pass
        
        # Update display data
        if new_boxes is not None:
            self.boxes = new_boxes
        if new_confidences is not None:
            self.confidences = new_confidences
            
        # Redraw only when feature is enabled
        if self.config.AimToggle:
            self.update()

    def draw_corner_box(self, painter, x1, y1, x2, y2):
        """Draw L-shaped corner brackets scaled to target size."""
        box_size = min(abs(x2 - x1), abs(y2 - y1))
        corner_len = max(6, min(24, int(box_size * 0.15)))

        # Top-left
        painter.drawLine(x1, y1, x1 + corner_len, y1)
        painter.drawLine(x1, y1, x1, y1 + corner_len)
        # Top-right
        painter.drawLine(x2, y1, x2 - corner_len, y1)
        painter.drawLine(x2, y1, x2, y1 + corner_len)
        # Bottom-left
        painter.drawLine(x1, y2, x1 + corner_len, y2)
        painter.drawLine(x1, y2, x1, y2 - corner_len)
        # Bottom-right
        painter.drawLine(x2, y2, x2 - corner_len, y2)
        painter.drawLine(x2, y2, x2, y2 - corner_len)

    def draw_fov_corners(self, painter, cx, cy, fov, corner_length=20):
        """Draw FOV corners
        
        Draws L-shaped marker lines at the corners of the FOV box rather than a full box.
        This design reduces visual clutter while still clearly indicating the FOV range.
        
        Args:
            painter: QPainter drawing object
            cx, cy: FOV center coordinates
            fov: FOV box side length (square)
            corner_length: length of corner markers, default 20 pixels
        """
        x1 = cx - fov // 2
        y1 = cy - fov // 2
        x2 = cx + fov // 2
        y2 = cy + fov // 2
        
        # Top-left
        painter.drawLine(x1, y1, x1 + corner_length, y1)  # Horizontal
        painter.drawLine(x1, y1, x1, y1 + corner_length)  # Vertical
        
        # Top-right
        painter.drawLine(x2, y1, x2 - corner_length, y1)  # Horizontal
        painter.drawLine(x2, y1, x2, y1 + corner_length)  # Vertical
        
        # Bottom-left
        painter.drawLine(x1, y2, x1 + corner_length, y2)  # Horizontal
        painter.drawLine(x1, y2, x1, y2 - corner_length)  # Vertical
        
        # 右下角
        painter.drawLine(x2, y2, x2 - corner_length, y2)  # 水平線
        painter.drawLine(x2, y2, x2, y2 - corner_length)  # 垂直線

    def draw_tracker_prediction(self, painter):
        """繪製智慧追蹤預測視覺化"""
        tracker_enabled = getattr(self.config, 'tracker_enabled', False)
        show_prediction = getattr(self.config, 'tracker_show_prediction', True)
        has_prediction = getattr(self.config, 'tracker_has_prediction', False)
        
        if not tracker_enabled or not show_prediction or not has_prediction:
            return
        
        # 取得座標
        current_x = getattr(self.config, 'tracker_current_x', 0)
        current_y = getattr(self.config, 'tracker_current_y', 0)
        predicted_x = getattr(self.config, 'tracker_predicted_x', 0)
        predicted_y = getattr(self.config, 'tracker_predicted_y', 0)
        
        # 如果座標無效，跳過
        if current_x == 0 and current_y == 0:
            return
        
        cx, cy = int(current_x), int(current_y)
        px, py = int(predicted_x), int(predicted_y)
        
        # 繪製連線（從觀測點到預測點）- 使用主題顏色
        line_color = OverlayColors.get_tracker_line_color()
        pen_line = QPen(line_color, 1, Qt.PenStyle.DotLine)
        painter.setPen(pen_line)
        painter.drawLine(cx, cy, px, py)
        
        # 繪製當前觀測位置 - 使用主題顏色
        current_color = OverlayColors.get_tracker_current_color()
        pen_current = QPen(current_color, 2)
        painter.setPen(pen_current)
        current_fill = QColor(current_color)
        current_fill.setAlpha(current_color.alpha() // 2)
        painter.setBrush(current_fill)
        painter.drawEllipse(cx - 2, cy - 2, 4, 4)
        
        # 繪製預測位置 - 使用主題顏色
        predicted_color = OverlayColors.get_tracker_predicted_color()
        pen_predicted = QPen(predicted_color, 2)
        painter.setPen(pen_predicted)
        predicted_fill = QColor(predicted_color)
        predicted_fill.setAlpha(predicted_color.alpha() // 2)
        painter.setBrush(predicted_fill)
        painter.drawEllipse(px - 3, py - 3, 6, 6)
        
        # 重置畫刷
        painter.setBrush(Qt.BrushStyle.NoBrush)

    def draw_tracer_lines(self, painter: QPainter) -> None:
        """Draw lines from screen center to each detected box bottom-center."""
        if not self.boxes:
            return
        cx = int(self.config.crosshairX)
        cy = int(self.config.crosshairY)
        tracer_color = OverlayColors.get_tracer_color()
        pen = QPen(tracer_color, 2, Qt.PenStyle.SolidLine)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        fov_half = int(self.config.fov_size) // 2
        for box in self.boxes:
            x1, y1, x2, y2 = map(int, box)
            bx = (x1 + x2) // 2
            by = (y1 + y2) // 2
            # Only draw if box center is within FOV (already filtered, but guard here too)
            if abs(bx - cx) <= fov_half and abs(by - cy) <= fov_half:
                painter.drawLine(cx, cy, bx, by)

    def paintEvent(self, event):
        if getattr(self.config, 'screenshot_method', 'mss') in ('uvc', 'ndi'):
            return
        if not self.config.AimToggle:
            return
            
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # 讀取顯示設置
        show_fov = getattr(self.config, 'show_fov', True)
        show_boxes = getattr(self.config, 'show_boxes', True)
        show_detect_range = getattr(self.config, 'show_detect_range', False)

        # 繪製 AI 偵測範圍（使用主題顏色）
        if show_detect_range:
            cx, cy = self.config.crosshairX, self.config.crosshairY
            range_size = int(getattr(self.config, 'detect_range_size', self.config.height))
            range_size = max(int(self.config.fov_size), min(int(self.config.height), range_size))
            half = range_size // 2
            x1 = int(cx - half)
            y1 = int(cy - half)
            detect_range_color = OverlayColors.get_detect_range_color()
            pen_range = QPen(detect_range_color, 1)
            painter.setPen(pen_range)
            painter.drawRect(x1, y1, int(range_size), int(range_size))
        
        # 繪製 FOV 框（只顯示四角）- 使用主題顏色
        if show_fov:
            fov = self.config.fov_size
            cx, cy = self.config.crosshairX, self.config.crosshairY
            fov_color = OverlayColors.get_fov_color()
            pen = QPen(fov_color, 2)
            painter.setPen(pen)
            self.draw_fov_corners(painter, cx, cy, fov)

        # 繪製檢測框和置信度 - 使用主題顏色
        if show_boxes and self.boxes:
            theme_key = str(getattr(self.config, 'box_color_theme', 'default')).lower()
            theme_rgba = _BOX_THEMES.get(theme_key)
            if theme_rgba is not None:
                box_color = QColor(*theme_rgba)
            else:
                box_color = OverlayColors.get_box_color()

            show_confidence = self.config.show_confidence
            if show_confidence:
                confidence_color = OverlayColors.get_confidence_text_color()
                pen_text = QPen(confidence_color, 1)
                font = QFont('Arial', 9, QFont.Weight.Bold)
                painter.setFont(font)

            for i, box in enumerate(self.boxes):
                x1, y1, x2, y2 = map(int, box)
                conf = float(self.confidences[i]) if i < len(self.confidences) else 0.5
                # Confidence-based thickness: low conf → 1px, high conf → 3px
                thickness = max(1, min(3, 1 + round(conf * 2)))
                painter.setPen(QPen(box_color, thickness))
                self.draw_corner_box(painter, x1, y1, x2, y2)

                if show_confidence and i < len(self.confidences):
                    painter.setPen(pen_text)
                    painter.drawText(x1 - 20, y1 - 15, f"{conf:.0%}")

        # 繪製追蹤線（從螢幕中心到目標）
        if getattr(self.config, 'show_tracer_line', False):
            self.draw_tracer_lines(painter)

        # 繪製卡爾曼預測視覺化
        self.draw_tracker_prediction(painter)

        # 繪製自訂準心
        self._draw_crosshair(painter)

    def _draw_crosshair(self, painter: QPainter) -> None:
        """Draw a configurable crosshair dot or cross at the crosshair position."""
        if not getattr(self.config, 'show_crosshair', False):
            return
        cx = int(self.config.crosshairX)
        cy = int(self.config.crosshairY)
        size = max(1, int(getattr(self.config, 'crosshair_size', 4)))
        r = int(getattr(self.config, 'crosshair_color_r', 255))
        g = int(getattr(self.config, 'crosshair_color_g', 255))
        b = int(getattr(self.config, 'crosshair_color_b', 255))
        color = QColor(r, g, b, 220)
        pen = QPen(color, 1)
        painter.setPen(pen)
        painter.setBrush(color)
        style = str(getattr(self.config, 'crosshair_style', 'dot'))
        if style == 'cross':
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawLine(cx - size * 3, cy, cx + size * 3, cy)
            painter.drawLine(cx, cy - size * 3, cx, cy + size * 3)
        else:
            painter.drawEllipse(cx - size, cy - size, size * 2, size * 2)
        painter.setBrush(Qt.BrushStyle.NoBrush)
