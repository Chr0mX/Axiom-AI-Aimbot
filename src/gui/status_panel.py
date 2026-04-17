# status_panel.py
"""
Status Panel in Fluent Design Style (Redesigned)
Uses QLayout and QWidget for layout, providing a modern and clean visual effect.
Supports Windows Acrylic blur effect.
"""
import os
import sys
import time
import ctypes
from ctypes import POINTER, pointer, sizeof, byref, WinDLL, c_int
from ctypes.wintypes import DWORD, ULONG
from PyQt6.QtWidgets import (QWidget, QApplication, QVBoxLayout, QHBoxLayout, 
                             QLabel, QFrame, QGraphicsDropShadowEffect, QSpacerItem, QSizePolicy)
from PyQt6.QtGui import (QPainter, QColor, QFont, QPixmap, QLinearGradient, 
                         QBrush, QPainterPath, QPen)
from PyQt6.QtCore import Qt, QTimer, QSize, pyqtSignal

from core.language_manager import get_text, language_manager

# Try to import theme functions from qfluentwidgets
try:
    from qfluentwidgets import isDarkTheme, themeColor
    HAS_FLUENT_WIDGETS = True
except ImportError:
    HAS_FLUENT_WIDGETS = False
    def isDarkTheme():
        return True  # Default to dark theme

# Import theme color definitions
try:
    from gui.fluent_app.theme_colors import ThemeColors, to_css_rgba
    HAS_THEME_COLORS = True
except ImportError:
    HAS_THEME_COLORS = False

# --- Structures required for Win32 Acrylic effect ---
class _ACCENT_POLICY(ctypes.Structure):
    _fields_ = [
        ("AccentState",   DWORD),
        ("AccentFlags",   DWORD),
        ("GradientColor", DWORD),
        ("AnimationId",   DWORD),
    ]

class _WINCOMPATTRDATA(ctypes.Structure):
    _fields_ = [
        ("Attribute",   DWORD),
        ("Data",        POINTER(_ACCENT_POLICY)),
        ("SizeOfData",  ULONG),
    ]

class _MARGINS(ctypes.Structure):
    _fields_ = [
        ("cxLeftWidth",    c_int),
        ("cxRightWidth",   c_int),
        ("cyTopHeight",    c_int),
        ("cyBottomHeight", c_int),
    ]

# Constants
_WCA_ACCENT_POLICY = 19
_ACCENT_ENABLE_ACRYLICBLURBEHIND = 4
_ACCENT_DISABLED = 0

# --- Fluent Design color scheme ---
class FluentColors:
    """Fluent Design Color Scheme - Supports Dark/Light themes
    Now integrated with unified color definitions from the ThemeColors module.
    """
    
    @staticmethod
    def to_css_rgba(color: QColor) -> str:
        return f"rgba({color.red()}, {color.green()}, {color.blue()}, {color.alpha() / 255.0})"

    @staticmethod
    def get_background_color():
        if HAS_THEME_COLORS:
            return ThemeColors.PANEL_BACKGROUND.qcolor()
        if isDarkTheme():
            return QColor(30, 30, 30, 255)  # Dark theme background
        else:
            return QColor(255, 255, 255, 255)  # Light theme background

    @staticmethod
    def get_text_primary_color():
        if HAS_THEME_COLORS:
            return ThemeColors.TEXT_PRIMARY.qcolor()
        return QColor(255, 255, 255) if isDarkTheme() else QColor(26, 26, 26)
        
    @staticmethod
    def get_text_secondary_color():
        if HAS_THEME_COLORS:
            return ThemeColors.TEXT_SECONDARY.qcolor()
        return QColor(160, 160, 160) if isDarkTheme() else QColor(90, 90, 90)

    @staticmethod
    def get_border_color():
        if HAS_THEME_COLORS:
            return ThemeColors.PANEL_BORDER.qcolor()
        return QColor(255, 255, 255, 20) if isDarkTheme() else QColor(0, 0, 0, 15)

    @staticmethod
    def get_accent_color():
        if HAS_THEME_COLORS:
            return ThemeColors.ACCENT.qcolor()
        return QColor(0, 122, 255)  # macOS Blue
    
    @staticmethod
    def get_success_color():
        if HAS_THEME_COLORS:
            return ThemeColors.SUCCESS.qcolor()
        return QColor(52, 199, 89) if not isDarkTheme() else QColor(50, 215, 75)
    
    @staticmethod
    def get_error_color():
        if HAS_THEME_COLORS:
            return ThemeColors.ERROR.qcolor()
        return QColor(255, 59, 48) if not isDarkTheme() else QColor(255, 69, 58)
         
    # Keep static properties for backward compatibility
    @property
    def SUCCESS(self):
        return self.get_success_color()
    
    @property
    def ERROR(self):
        return self.get_error_color()

# Create a global instance for attribute access
_fluent_colors_instance = FluentColors()
FluentColors.SUCCESS = _fluent_colors_instance.get_success_color()
FluentColors.ERROR = _fluent_colors_instance.get_error_color()

class StatusIndicator(QWidget):
    """A simple dot status indicator"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(10, 10) # Slightly smaller
        self._active = False
        self._color = FluentColors.get_error_color()
        
    def set_status(self, active: bool):
        self._active = active
        self._color = FluentColors.get_success_color() if active else FluentColors.get_error_color()
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setBrush(QBrush(self._color))
        painter.setPen(Qt.PenStyle.NoPen)
        # Draw in the middle
        painter.drawEllipse(1, 1, 8, 8)

class StatusRow(QWidget):
    """General status row Widget"""
    def __init__(self, label_text, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # Label
        self.label = QLabel(label_text, self)
        self.label.setObjectName("statusLabel")
        
        # Value (can be text or a Widget)
        self.value_label = QLabel("", self)
        self.value_label.setObjectName("statusValue")
        
        # Elastic space
        layout.addWidget(self.label)
        layout.addStretch()
        layout.addWidget(self.value_label)

    def set_value(self, text, color=None):
        self.value_label.setText(text)
        if color:
            self.value_label.setStyleSheet(f"color: {color};")
        else:
            # Reset to default style
            pass

class StatusPanel(QWidget):
    """
    Status Panel in MacOS style (Widget version)
    Supports Windows Acrylic liquid blur effect.
    """
    # Set to 0 to let it adapt automatically
    PANEL_WIDTH = 0
    PANEL_HEIGHT = 0
    ACRYLIC_PANEL_WIDTH = 0
    ACRYLIC_PANEL_HEIGHT = 0
    BORDER_RADIUS = 24 # MacOS rounded corners

    def __init__(self, config):
        super().__init__()
        self.setObjectName("statusPanelRoot")
        self.config = config
        self._acrylic_enabled = False  # Track if Acrylic is successfully enabled
        self._is_applying_acrylic = False
        self._pending_acrylic_apply = False
        self._shadow_effect = None     # 追蹤陰影特效實例
        
        # --- 視窗基本設定 ---
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool
        )
        # 預設啟用透明背景（acrylic 啟用時會關閉）
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        
        # --- 拖動變數 ---
        self._drag_pos = None
        self._is_dragging = False
        self._auto_nudge_direction = 1

        # --- 這部分是重點：初始化 UI 佈局 ---
        self._init_ui()

        # --- 定時器 ---
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_display)
        self.timer.start(500) # 0.5秒刷新一次狀態

        self.auto_nudge_timer = QTimer(self)
        self.auto_nudge_timer.timeout.connect(self._auto_nudge_panel)
        self.auto_nudge_timer.start(5000)

        # --- 緩存狀態 ---
        self.last_theme_dark = isDarkTheme()
        self.last_aim_state = None
        self.last_model_path = None
        self.last_mouse_method = None
        self.last_language = None
        self._last_acrylic_enabled = None  # 追蹤 config 的 acrylic 開關
        self._last_acrylic_alpha = None    # 追蹤 acrylic 不透明度
        self._last_fps_calc_time = 0.0
        self._last_screenshot_frame_count = 0
        self._last_detection_frame_count = 0

        # 初次設置樣式
        self._apply_panel_size()
        self._update_style()
        self.update_display() # 立即刷新一次內容

    def _update_row_visibility(self):
        """依設定更新狀態面板各元素顯示/隱藏"""
        self.aim_row.setVisible(getattr(self.config, 'status_panel_show_auto_aim', True))
        self.model_row.setVisible(getattr(self.config, 'status_panel_show_model', True))
        self.mouse_row.setVisible(getattr(self.config, 'status_panel_show_mouse_move', True))
        self.mouse_click_row.setVisible(getattr(self.config, 'status_panel_show_mouse_click', True))
        self.screenshot_row.setVisible(getattr(self.config, 'status_panel_show_screenshot_method', True))
        self.screenshot_fps_row.setVisible(getattr(self.config, 'status_panel_show_screenshot_fps', True))
        self.detection_fps_row.setVisible(getattr(self.config, 'status_panel_show_detection_fps', True))

    def _apply_panel_size(self):
        """依據當前模式套用面板尺寸"""
        if self._acrylic_enabled:
            self.setFixedSize(self.ACRYLIC_PANEL_WIDTH, self.ACRYLIC_PANEL_HEIGHT)
        else:
            self.setFixedSize(self.PANEL_WIDTH, self.PANEL_HEIGHT)

    def showEvent(self, event):
        """視窗顯示時套用 Acrylic 效果和圓角"""
        super().showEvent(event)
        # 延遲套用，確保 HWND 已完全建立
        QTimer.singleShot(150, self._applyAcrylicEffect)
        QTimer.singleShot(200, self._applyWindowRoundedCorners)

    def resizeEvent(self, event):
        """視窗大小改變時重新套用圓角 region (Win10 fallback)"""
        super().resizeEvent(event)
        self._applyWindowRoundedCorners()

    def _applyWindowRoundedCorners(self):
        """設定視窗圓角
        
        Win11: 使用 DWM DWMWA_WINDOW_CORNER_PREFERENCE
        Win10 fallback: 使用 CreateRoundRectRgn + SetWindowRgn 裁剪視窗
        """
        if sys.platform != 'win32':
            return
        try:
            hwnd = int(self.winId())
            if hwnd == 0:
                return
            dwmapi = WinDLL("dwmapi")
            
            # 嘗試 Win11 DWM 圓角設定
            try:
                DWMWA_WINDOW_CORNER_PREFERENCE = 33
                DWMWCP_DONOTROUND = c_int(1)
                DWMWCP_ROUND = c_int(2)  # 大圓角
                corner_pref = DWMWCP_ROUND if self._acrylic_enabled else DWMWCP_DONOTROUND
                dwmapi.DwmSetWindowAttribute(
                    hwnd, DWORD(DWMWA_WINDOW_CORNER_PREFERENCE),
                    byref(corner_pref), 4
                )
            except Exception:
                pass
            
            # 保險方案：使用 SetWindowRgn 裁剪圓角
            # 某些情況（例如 Tool 視窗）Win11 的 DWM 圓角不一定穩定生效
            gdi32 = WinDLL("gdi32")
            user32 = WinDLL("user32")
            w, h = self.width(), self.height()
            radius = self.BORDER_RADIUS
            rgn = gdi32.CreateRoundRectRgn(0, 0, w + 1, h + 1, radius, radius)
            user32.SetWindowRgn(hwnd, rgn, True)
        except Exception:
            pass

    def _applyAcrylicEffect(self):
        """應用 Windows Acrylic 液態毛玻璃效果到狀態面板
        
        關鍵技術點：
        1. WA_TranslucentBackground 會建立 Layered Window (WS_EX_LAYERED)，
           會繞過 DWM 合成，因此 Acrylic 無法作用。必須關閉。
        2. 必須呼叫 DwmExtendFrameIntoClientArea 將 DWM 玻璃框延伸至整個客戶區，
           這樣 SetWindowCompositionAttribute 的 Acrylic 效果才能在客戶區域渲染。
        3. paintEvent 中使用 CompositionMode_Source 填充透明色，
           讓 DWM 合成的 Acrylic 效果透出。
        """
        if sys.platform != 'win32':
            return

        if self._is_applying_acrylic:
            self._pending_acrylic_apply = True
            return

        self._is_applying_acrylic = True

        enable = getattr(self.config, 'enable_acrylic', True)
        
        try:
            hwnd = int(self.winId())
            if hwnd == 0:
                return

            user32 = WinDLL("user32")
            dwmapi = WinDLL("dwmapi")

            accentPolicy = _ACCENT_POLICY()
            winCompAttrData = _WINCOMPATTRDATA()
            winCompAttrData.Attribute = _WCA_ACCENT_POLICY
            winCompAttrData.SizeOfData = sizeof(accentPolicy)
            winCompAttrData.Data = pointer(accentPolicy)

            if not enable:
                # 停用 Acrylic - 恢復到普通模式
                accentPolicy.AccentState = _ACCENT_DISABLED
                accentPolicy.GradientColor = 0
                accentPolicy.AccentFlags = 0
                accentPolicy.AnimationId = 0
                user32.SetWindowCompositionAttribute(hwnd, pointer(winCompAttrData))
                
                # 恢復 WA_TranslucentBackground 用於圓角透明
                self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
                self._acrylic_enabled = False
                
                # 恢復陰影
                self._applyShadowEffect()
                self.main_layout.setContentsMargins(10, 10, 10, 10)
                self._apply_panel_size()
                self._update_style()
                self._applyWindowRoundedCorners()
                self.update()
                return

            # === 啟用 Acrylic ===
            
            # 步驟 1: 關閉 WA_TranslucentBackground
            # Layered window (WS_EX_LAYERED) 繞過 DWM，acrylic 無法生效
            self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)
            
            # 步驟 2: 將 DWM 玻璃框延伸到整個客戶區
            margins = _MARGINS(-1, -1, -1, -1)
            dwmapi.DwmExtendFrameIntoClientArea(hwnd, byref(margins))
            
            # 步驟 3: 嘗試設定 Win11 圓角 (DWMWA_WINDOW_CORNER_PREFERENCE = 33)
            try:
                DWMWA_WINDOW_CORNER_PREFERENCE = 33
                DWMWCP_ROUND = c_int(2)  # 圓角
                dwmapi.DwmSetWindowAttribute(
                    hwnd, DWORD(DWMWA_WINDOW_CORNER_PREFERENCE),
                    byref(DWMWCP_ROUND), sizeof(DWMWCP_ROUND)
                )
            except Exception:
                pass  # Win10 不支援，忽略

            # 步驟 4: 計算 gradientColor
            raw_alpha = getattr(self.config, 'acrylic_window_alpha', 187)
            alpha = max(60, min(255, int(raw_alpha)))
            alpha_hex = hex(alpha)[2:].upper().zfill(2)

            is_dark = isDarkTheme()
            if is_dark:
                gradient_str = f"1A1A1A{alpha_hex}"
            else:
                gradient_str = f"F5F5F5{alpha_hex}"

            # 轉換 RRGGBBAA -> AABBGGRR (Win32 byte order)
            gradient_reversed = ''.join(gradient_str[i:i+2] for i in range(6, -1, -2))
            gradient_color = DWORD(int(gradient_reversed, base=16))

            # 步驟 5: 套用 Acrylic Accent Policy
            accentPolicy.AccentState = _ACCENT_ENABLE_ACRYLICBLURBEHIND
            accentPolicy.GradientColor = gradient_color
            accentPolicy.AccentFlags = DWORD(0x20 | 0x40 | 0x80 | 0x100)  # 啟用陰影邊框
            accentPolicy.AnimationId = DWORD(0)

            user32.SetWindowCompositionAttribute(hwnd, pointer(winCompAttrData))
            self._acrylic_enabled = True
            
            # Acrylic 模式不需要 Qt 層面的陰影和邊距
            self._removeShadowEffect()
            self.main_layout.setContentsMargins(0, 0, 0, 0)
            self._apply_panel_size()
            self._update_style()
            self._applyWindowRoundedCorners()
            self.update()

        except Exception as e:
            print(f"[StatusPanel] 套用 Acrylic 效果失敗: {e}")
            self._acrylic_enabled = False
            # 失敗時恢復 WA_TranslucentBackground
            self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        finally:
            self._is_applying_acrylic = False
            if self._pending_acrylic_apply:
                self._pending_acrylic_apply = False
                QTimer.singleShot(0, self._applyAcrylicEffect)

    def _applyShadowEffect(self):
        """套用 Qt 陰影特效（非 Acrylic 模式使用）"""
        if self._shadow_effect is None:
            shadow = QGraphicsDropShadowEffect(self)
            shadow.setBlurRadius(20)
            shadow.setColor(QColor(0, 0, 0, 60))
            shadow.setOffset(0, 4)
            self.container.setGraphicsEffect(shadow)
            self._shadow_effect = shadow

    def _removeShadowEffect(self):
        """移除 Qt 陰影特效（Acrylic 模式下 DWM 提供陰影）"""
        self.container.setGraphicsEffect(None)
        self._shadow_effect = None

    def paintEvent(self, event):
        """自訂繪製 - Acrylic 模式下清除背景讓 DWM 毛玻璃透出"""
        if self._acrylic_enabled:
            painter = QPainter(self)
            # CompositionMode_Source: 直接替換像素（不混合）
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Source)

            # 先清空整個區域，再僅填入圓角客戶區，避免 Acrylic 呈現方角
            painter.fillRect(self.rect(), QColor(0, 0, 0, 0))

            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            path = QPainterPath()
            rect = self.rect().adjusted(0, 0, -1, -1)
            path.addRoundedRect(float(rect.x()), float(rect.y()), float(rect.width()), float(rect.height()),
                                float(self.BORDER_RADIUS), float(self.BORDER_RADIUS))

            # 使用 alpha=1 而非 alpha=0：
            # alpha=0 可能讓 DWM 將區域視為玻璃框，拖動事件被攔截
            painter.fillPath(path, QColor(0, 0, 0, 1))
            painter.end()
        else:
            # 非 Acrylic 模式：使用預設繪製（WA_TranslucentBackground 處理透明）
            super().paintEvent(event)

    def _init_ui(self):
        """初始化 UI 結構"""
        # 主 Layout
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(10, 10, 10, 10) # Space for shadow

        # 背景容器 (QFrame)，負責圓角和背景色
        self.container = QFrame(self)
        self.container.setObjectName("container")
        
        # Shadow Effect（預設啟用，Acrylic 啟用時會移除）
        self._applyShadowEffect()

        self.container_layout = QVBoxLayout(self.container)
        self.container_layout.setContentsMargins(16, 14, 16, 14)
        self.container_layout.setSpacing(6) # Tighter vertical spacing

        # 1. 標題列
        self.header_layout = QHBoxLayout()
        self.header_layout.setSpacing(12)
        
        # Logo PlaceHolder (用來佔位，保持排版不亂)
        self.logo_placeholder = QLabel()
        self.logo_placeholder.setFixedSize(20, 20)
        self.header_layout.addWidget(self.logo_placeholder)

        # 真正的 Logo (浮動顯示，尺寸較大)
        self.logo_label = QLabel(self.container)
        self.logo_label.setFixedSize(32, 32)
        self.logo_label.setScaledContents(True)
        # 手動定位：讓它重疊在左上角 (根據 layout margins 計算)
        # container margins: 16 (left), 14 (top)
        # placeholder 30x40，logo 40x40
        # 水平：讓 logo 左邊超出 placeholder 一些 → x=6
        # 垂直：與 placeholder 頂部對齊 → y=14，使文字垂直居中於 logo
        self.logo_label.move(10, 10)
        self.logo_label.raise_()

        # Title Group (垂直排列 Title 和 Version，或者水平) -> 採用水平
        self.title_label = QLabel("Axiom")
        self.title_label.setObjectName("titleLabel")
        
        self.version_label = QLabel("v6.1")
        self.version_label.setObjectName("versionLabel")

        # self.header_layout.addWidget(self.logo_label) # 移除原本的添加
        self.header_layout.addWidget(self.title_label)
        self.header_layout.addWidget(self.version_label)
        self.header_layout.addStretch()

        # 2. 分隔線 (用 QFrame 模擬)
        self.separator = QFrame()
        self.separator.setFrameShape(QFrame.Shape.HLine)
        self.separator.setFrameShadow(QFrame.Shadow.Plain)
        self.separator.setFixedHeight(1)
        self.separator.setObjectName("separator")

        # 3. 狀態行 - 自動瞄準
        self.aim_row = QWidget()
        self.aim_layout = QHBoxLayout(self.aim_row)
        self.aim_layout.setContentsMargins(0, 0, 0, 0)
        self.aim_layout.setSpacing(8)
        
        self.aim_indicator = StatusIndicator()
        self.aim_text_label = QLabel(get_text('auto_aim'))
        self.aim_text_label.setObjectName("statusLabel")
        self.aim_status_label = QLabel()
        self.aim_status_label.setObjectName("statusValue")

        self.aim_layout.addWidget(self.aim_indicator)
        self.aim_layout.addWidget(self.aim_text_label)
        self.aim_layout.addStretch()
        self.aim_layout.addWidget(self.aim_status_label)

        # 4. 狀態行 - 目前模型
        self.model_row = StatusRow(get_text('status_panel_current_model'))

        # 5. 狀態行 - 滑鼠移動
        self.mouse_row = StatusRow(get_text('mouse_move_method'))

        # 6. 狀態行 - 滑鼠點擊
        self.mouse_click_row = StatusRow(get_text('mouse_click_method'))

        # 7. 狀態行 - 截圖方式
        self.screenshot_row = StatusRow(get_text('screenshot_method'))

        # 8. 狀態行 - 截圖 FPS
        self.screenshot_fps_row = StatusRow(get_text('status_panel_screenshot_fps', 'Screenshot FPS'))

        # 9. 狀態行 - 偵測 FPS
        self.detection_fps_row = StatusRow(get_text('status_panel_detection_fps', 'Detection FPS'))

        # 加入容器
        self.container_layout.addLayout(self.header_layout)
        self.container_layout.addWidget(self.separator)
        self.container_layout.addSpacing(2) 
        self.container_layout.addWidget(self.aim_row)
        self.container_layout.addWidget(self.model_row)
        self.container_layout.addWidget(self.mouse_row)
        self.container_layout.addWidget(self.mouse_click_row)
        self.container_layout.addWidget(self.screenshot_row)
        self.container_layout.addWidget(self.screenshot_fps_row)
        self.container_layout.addWidget(self.detection_fps_row)
        self.container_layout.addStretch()

        self.main_layout.addWidget(self.container)

        self._load_logo()

    def _load_logo(self):
        """載入 Logo"""
        logo_path = os.path.join(os.path.dirname(__file__), 'logo.png')
        
        if os.path.exists(logo_path):
            self.logo_label.setPixmap(QPixmap(logo_path))
        else:
            self.logo_label.clear()

    def _update_style(self):
        """更新 QSS 樣式表"""
        text_primary = FluentColors.to_css_rgba(FluentColors.get_text_primary_color())
        text_secondary = FluentColors.to_css_rgba(FluentColors.get_text_secondary_color())
        border_color = FluentColors.to_css_rgba(FluentColors.get_border_color())
        
        # 根據 Acrylic 是否啟用決定容器背景
        if self._acrylic_enabled:
            # Acrylic 啟用時：容器背景設為透明，讓 DWM 層的毛玻璃效果透出
            # 不需要圓角（DWM 會處理 Win11 圓角，Win10 為矩形即可）
            container_bg = "transparent"
            container_border = "none"
            container_radius = 0
        else:
            # Acrylic 停用時：使用不透明背景 + 圓角
            bg_color_obj = FluentColors.get_background_color()
            container_bg = FluentColors.to_css_rgba(bg_color_obj)
            container_border = f"1px solid {border_color}"
            container_radius = self.BORDER_RADIUS
        
        style_sheet = f"""
            QWidget#statusPanelRoot {{
                background: transparent;
                border: none;
            }}
            QFrame#container {{
                background-color: {container_bg};
                border: {container_border};
                border-radius: {container_radius}px;
            }}
            QLabel#titleLabel {{
                color: {text_primary};
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                font-size: 18px;
                font-weight: 700;
                background: transparent;
            }}
            QLabel#versionLabel {{
                color: {text_secondary};
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                font-size: 14px;
                padding-top: 3px;
                background: transparent;
            }}
            QLabel#statusLabel {{
                color: {text_primary};
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                font-size: 12px;
                background: transparent;
            }}
            QLabel#statusValue {{
                color: {text_primary};
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                font-size: 12px;
                font-weight: 500;
                background: transparent;
            }}
            QFrame#separator {{
                color: {border_color}; 
                background-color: {border_color}; 
                border: none;
            }}
        """
        self.setStyleSheet(style_sheet)

    def update_display(self):
        """更新顯示數據和主題檢測"""
        
        # 1. 檢查是否顯示
        show_panel = getattr(self.config, 'show_status_panel', True)
        if not show_panel:
            if self.isVisible(): self.hide()
            return
        elif not self.isVisible():
            self.show()

        # 2. 檢查主題變化 (簡單輪詢)
        current_theme_dark = isDarkTheme()
        if current_theme_dark != self.last_theme_dark:
            self.last_theme_dark = current_theme_dark
            self._update_style()
            self._load_logo()
            # 主題變化時重新套用 Acrylic（更新毛玻璃顏色）
            self._applyAcrylicEffect()

        self._update_row_visibility()

        # 2.5 檢查 Acrylic 開關變化（避免透明度拖曳時高頻重入原生 API）
        current_acrylic = getattr(self.config, 'enable_acrylic', True)
        acrylic_changed = (current_acrylic != self._last_acrylic_enabled)
        if acrylic_changed:
            self._last_acrylic_enabled = current_acrylic
            self._applyAcrylicEffect()

        # 3. 獲取數據
        current_aim = self.config.AimToggle
        current_model = getattr(self.config, 'model_path', '')
        current_method = getattr(self.config, 'mouse_move_method', 'ddxoft')
        current_screenshot_method = getattr(self.config, 'screenshot_method', 'mss')
        current_lang = language_manager.get_current_language()

        # 檢查是否需要更新 UI 文本 (例如語言改變或狀態改變)
        # 為了簡化，簡單比較關鍵值，或者直接更新所有文字(開銷很小)
        
        # 更新 Auto Aim
        if current_aim:
            self.aim_status_label.setText(get_text("status_panel_on"))
            self.aim_status_label.setStyleSheet(f"color: {FluentColors.to_css_rgba(FluentColors.get_success_color())};")
            self.aim_indicator.set_status(True)
        else:
            self.aim_status_label.setText(get_text("status_panel_off"))
            self.aim_status_label.setStyleSheet(f"color: {FluentColors.to_css_rgba(FluentColors.get_error_color())};")
            self.aim_indicator.set_status(False)
        self.aim_text_label.setText(get_text('auto_aim'))

        # 更新 Model
        model_name = os.path.basename(current_model) if current_model else "None"
        if len(model_name) > 25: model_name = model_name[:22] + "..."
        self.model_row.label.setText(get_text('status_panel_current_model'))
        self.model_row.set_value(model_name)

        # 更新 Mouse Method
        mouse_map = {'sendinput': 'SendInput', 'mouse_event': 'mouse_event', 'ddxoft': 'ddxoft', 'arduino': 'Arduino', 'makcu': 'MAKCU', 'xbox': 'Xbox 360'}
        disp_method = mouse_map.get(current_method, str(current_method))
        
        # 連線狀態檢查
        method_color = None
        if current_method == 'ddxoft':
            try:
                from win_utils import ddxoft_mouse
                if ddxoft_mouse.is_available():
                    disp_method += " ✓"
                    method_color = FluentColors.to_css_rgba(FluentColors.get_success_color())
                else:
                    disp_method += " ✗"
                    method_color = FluentColors.to_css_rgba(FluentColors.get_error_color())
            except ImportError:
                 pass
        elif current_method == 'arduino':
            try:
                from win_utils import is_arduino_connected
                if is_arduino_connected():
                    disp_method += " ✓"
                    method_color = FluentColors.to_css_rgba(FluentColors.get_success_color())
                else:
                    disp_method += " ✗"
                    method_color = FluentColors.to_css_rgba(FluentColors.get_error_color())
            except ImportError:
                disp_method += " ✗"
                method_color = FluentColors.to_css_rgba(FluentColors.get_error_color())
        elif current_method == 'makcu':
            try:
                from win_utils import is_makcu_connected
                if is_makcu_connected():
                    disp_method += " ✓"
                    method_color = FluentColors.to_css_rgba(FluentColors.get_success_color())
                else:
                    disp_method += " ✗"
                    method_color = FluentColors.to_css_rgba(FluentColors.get_error_color())
            except ImportError:
                disp_method += " ✗"
                method_color = FluentColors.to_css_rgba(FluentColors.get_error_color())
        elif current_method == 'xbox':
            try:
                from win_utils import is_xbox_connected
                if is_xbox_connected():
                    disp_method += " ✓"
                    method_color = FluentColors.to_css_rgba(FluentColors.get_success_color())
                else:
                    disp_method += " ✗"
                    method_color = FluentColors.to_css_rgba(FluentColors.get_error_color())
            except ImportError:
                disp_method += " ✗"
                method_color = FluentColors.to_css_rgba(FluentColors.get_error_color())
        
        self.mouse_row.label.setText(get_text('mouse_move_method'))
        self.mouse_row.set_value(disp_method, method_color)

        # 更新 Mouse Click Method
        current_click = getattr(self.config, 'mouse_click_method', 'mouse_event')
        click_map = {'sendinput': 'SendInput', 'mouse_event': 'mouse_event', 'ddxoft': 'ddxoft', 'arduino': 'Arduino', 'makcu': 'MAKCU', 'xbox': 'Xbox 360'}
        disp_click = click_map.get(current_click, str(current_click))
        click_color = None
        if current_click == 'ddxoft':
            try:
                from win_utils import ddxoft_mouse
                if ddxoft_mouse.is_available():
                    disp_click += " ✓"
                    click_color = FluentColors.to_css_rgba(FluentColors.get_success_color())
                else:
                    disp_click += " ✗"
                    click_color = FluentColors.to_css_rgba(FluentColors.get_error_color())
            except ImportError:
                pass
        elif current_click == 'arduino':
            try:
                from win_utils import is_arduino_connected
                if is_arduino_connected():
                    disp_click += " ✓"
                    click_color = FluentColors.to_css_rgba(FluentColors.get_success_color())
                else:
                    disp_click += " ✗"
                    click_color = FluentColors.to_css_rgba(FluentColors.get_error_color())
            except ImportError:
                disp_click += " ✗"
                click_color = FluentColors.to_css_rgba(FluentColors.get_error_color())
        elif current_click == 'makcu':
            try:
                from win_utils import is_makcu_connected
                if is_makcu_connected():
                    disp_click += " ✓"
                    click_color = FluentColors.to_css_rgba(FluentColors.get_success_color())
                else:
                    disp_click += " ✗"
                    click_color = FluentColors.to_css_rgba(FluentColors.get_error_color())
            except ImportError:
                disp_click += " ✗"
                click_color = FluentColors.to_css_rgba(FluentColors.get_error_color())
        elif current_click == 'xbox':
            try:
                from win_utils import is_xbox_connected
                if is_xbox_connected():
                    disp_click += " ✓"
                    click_color = FluentColors.to_css_rgba(FluentColors.get_success_color())
                else:
                    disp_click += " ✗"
                    click_color = FluentColors.to_css_rgba(FluentColors.get_error_color())
            except ImportError:
                disp_click += " ✗"
                click_color = FluentColors.to_css_rgba(FluentColors.get_error_color())
        self.mouse_click_row.label.setText(get_text('mouse_click_method'))
        self.mouse_click_row.set_value(disp_click, click_color)

        # 更新 Screenshot Method
        screenshot_map = {'mss': 'MSS', 'dxcam': 'DXcam', 'uvc': 'UVC', 'ndi': 'NDI'}
        disp_screenshot = screenshot_map.get(current_screenshot_method, str(current_screenshot_method))
        screenshot_color = None

        if current_screenshot_method == 'dxcam':
            try:
                import dxcam  # type: ignore[import-not-found]
                if dxcam is not None:
                    disp_screenshot += " ✓"
                    screenshot_color = FluentColors.to_css_rgba(FluentColors.get_success_color())
                else:
                    disp_screenshot += " ✗"
                    screenshot_color = FluentColors.to_css_rgba(FluentColors.get_error_color())
            except ImportError:
                disp_screenshot += " ✗"
                screenshot_color = FluentColors.to_css_rgba(FluentColors.get_error_color())
        elif current_screenshot_method == 'mss':
            disp_screenshot += " ✓"
            screenshot_color = FluentColors.to_css_rgba(FluentColors.get_success_color())
        elif current_screenshot_method == 'uvc':
            disp_screenshot += " ✓"
            screenshot_color = FluentColors.to_css_rgba(FluentColors.get_success_color())
        elif current_screenshot_method == 'ndi':
            try:
                import ndilib as ndi  # type: ignore[import-not-found]
                if ndi is not None:
                    disp_screenshot += " ✓"
                    screenshot_color = FluentColors.to_css_rgba(FluentColors.get_success_color())
                else:
                    disp_screenshot += " ✗"
                    screenshot_color = FluentColors.to_css_rgba(FluentColors.get_error_color())
            except ImportError:
                disp_screenshot += " ✗"
                screenshot_color = FluentColors.to_css_rgba(FluentColors.get_error_color())

        self.screenshot_row.label.setText(get_text('screenshot_method'))
        self.screenshot_row.set_value(disp_screenshot, screenshot_color)

        # 更新 Screenshot/Detection FPS
        now = time.perf_counter()
        screenshot_count = int(getattr(self.config, 'screenshot_frame_count', 0))
        detection_count = int(getattr(self.config, 'detection_frame_count', 0))
        if self._last_fps_calc_time <= 0.0:
            self._last_fps_calc_time = now
            self._last_screenshot_frame_count = screenshot_count
            self._last_detection_frame_count = detection_count

        elapsed = now - self._last_fps_calc_time
        if elapsed >= 0.5:
            screenshot_fps = max(0.0, (screenshot_count - self._last_screenshot_frame_count) / elapsed)
            detection_fps = max(0.0, (detection_count - self._last_detection_frame_count) / elapsed)

            self._last_fps_calc_time = now
            self._last_screenshot_frame_count = screenshot_count
            self._last_detection_frame_count = detection_count

            self.screenshot_fps_row.set_value(f"{screenshot_fps:.1f}")
            self.detection_fps_row.set_value(f"{detection_fps:.1f}")

        self.screenshot_fps_row.label.setText(get_text('status_panel_screenshot_fps', 'Screenshot FPS'))
        self.detection_fps_row.label.setText(get_text('status_panel_detection_fps', 'Detection FPS'))

    def _auto_nudge_panel(self):
        """每 5 秒水平位移 1px，方向在右/左之間交替"""
        if not self.isVisible() or self._is_dragging:
            return

        self.move(self.x() + self._auto_nudge_direction, self.y())
        self._auto_nudge_direction *= -1

    # --- 拖動邏輯 ---
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            self._is_dragging = True
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()

    def mouseMoveEvent(self, event):
        if self._is_dragging and self._drag_pos is not None:
            self.move(event.globalPosition().toPoint() - self._drag_pos)
            event.accept()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._is_dragging = False
            self._drag_pos = None
            self.setCursor(Qt.CursorShape.OpenHandCursor)
            event.accept()
