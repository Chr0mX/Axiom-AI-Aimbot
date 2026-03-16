# theme_colors.py
"""
Theme Colors Definition Module
Defines colors for all GUI elements in light and dark themes.
Supports loading custom colors from theme_colors.json.
"""

import os
import json
from PyQt6.QtGui import QColor
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any

try:
    from qfluentwidgets import isDarkTheme, themeColor
    HAS_FLUENT_WIDGETS = True
except ImportError:
    HAS_FLUENT_WIDGETS = False
    def isDarkTheme():
        return False


# ============================================================
# JSON Color Config Loader
# ============================================================

class ColorConfigLoader:
    """Loads color configuration from a JSON file"""
    
    _instance = None
    _config: Dict[str, Any] = {}
    _loaded = False
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        if not ColorConfigLoader._loaded:
            self._load_config()
    
    def _load_config(self):
        """Loads the JSON configuration file"""
        try:
            # Try multiple possible paths
            possible_paths = [
                os.path.join(os.path.dirname(__file__), "..", "..", "..", "theme_colors.json"),
                os.path.join(os.path.dirname(__file__), "..", "..", "theme_colors.json"),
                "theme_colors.json",
            ]
            
            for path in possible_paths:
                abs_path = os.path.abspath(path)
                if os.path.exists(abs_path):
                    with open(abs_path, 'r', encoding='utf-8') as f:
                        ColorConfigLoader._config = json.load(f)
                        ColorConfigLoader._loaded = True
                        print(f"[ThemeColors] Custom color config loaded: {abs_path}")
                        return
                        
        except Exception as e:
            print(f"[ThemeColors] Failed to load color config, using defaults: {e}")
        
        ColorConfigLoader._config = {}
        ColorConfigLoader._loaded = True
    
    def get_color(self, *keys, default_light: str = "#000000", default_dark: str = "#FFFFFF") -> Tuple[str, str]:
        """
        Gets color from configuration
        
        Args:
            *keys: Path keys in config, e.g., ("primary_colors", "primary")
            default_light: Default light color value
            default_dark: Default dark color value
        
        Returns:
            (light_color, dark_color) tuple
        """
        try:
            value = ColorConfigLoader._config
            for key in keys:
                value = value[key]
            
            if isinstance(value, dict):
                return (value.get("light", default_light), value.get("dark", default_dark))
            return (default_light, default_dark)
        except (KeyError, TypeError):
            return (default_light, default_dark)
    
    def get_rgba_color(self, *keys, 
                       default_light: Tuple[int, int, int, int] = (0, 0, 0, 255),
                       default_dark: Tuple[int, int, int, int] = (255, 255, 255, 255)) -> Tuple[Tuple, Tuple]:
        """
        Gets RGBA color from configuration
        
        Args:
            *keys: Path keys in config
            default_light: Default light RGBA value
            default_dark: Default dark RGBA value
        
        Returns:
            (light_rgba, dark_rgba) tuple
        """
        try:
            value = ColorConfigLoader._config
            for key in keys:
                value = value[key]
            
            if isinstance(value, dict):
                light = value.get("light", list(default_light))
                dark = value.get("dark", list(default_dark))
                return (tuple(light), tuple(dark))
            return (default_light, default_dark)
        except (KeyError, TypeError):
            return (default_light, default_dark)


# Global config loader instance
_config_loader = ColorConfigLoader.get_instance()


@dataclass
class ColorPair:
    """Color pair for light and dark themes"""
    light: str  # Color hex code for light theme
    dark: str   # Color hex code for dark theme
    
    def get(self) -> str:
        """Returns the corresponding color based on the current theme"""
        return self.dark if isDarkTheme() else self.light
    
    def qcolor(self) -> QColor:
        """Returns QColor object"""
        return QColor(self.get())
    
    @classmethod
    def from_config(cls, *keys, default_light: str = "#000000", default_dark: str = "#FFFFFF") -> 'ColorPair':
        """Creates ColorPair from configuration file"""
        light, dark = _config_loader.get_color(*keys, default_light=default_light, default_dark=default_dark)
        return cls(light=light, dark=dark)


@dataclass  
class ColorPairWithAlpha:
    """Color pair with alpha for light and dark themes"""
    light: Tuple[int, int, int, int]  # (R, G, B, A)
    dark: Tuple[int, int, int, int]   # (R, G, B, A)
    
    def get(self) -> Tuple[int, int, int, int]:
        """Returns the corresponding color based on the current theme"""
        return self.dark if isDarkTheme() else self.light
    
    def qcolor(self) -> QColor:
        """Returns QColor object"""
        r, g, b, a = self.get()
        return QColor(r, g, b, a)
    
    @classmethod
    def from_config(cls, *keys,
                    default_light: Tuple[int, int, int, int] = (0, 0, 0, 255),
                    default_dark: Tuple[int, int, int, int] = (255, 255, 255, 255)) -> 'ColorPairWithAlpha':
        """Creates ColorPairWithAlpha from configuration file"""
        light, dark = _config_loader.get_rgba_color(*keys, default_light=default_light, default_dark=default_dark)
        return cls(light=light, dark=dark)


class ThemeColors:
    """
    Theme Colors Definition Class
    Colors for all GUI elements are defined here.
    Supports loading custom colors from theme_colors.json.
    """
    
    # ============================================================
    # Main Brand / Accent Color
    # ============================================================
    PRIMARY = ColorPair.from_config(
        "primary_colors", "primary",
        default_light="#0078D4",
        default_dark="#4CC2FF"
    )

    ACCENT = ColorPair.from_config(
        "primary_colors", "accent",
        default_light="#0078D4",
        default_dark="#4CC2FF"
    )
    
    # ============================================================
    # Background Colors
    # ============================================================
    BACKGROUND_PRIMARY = ColorPair.from_config(
        "background_colors", "primary",
        default_light="#FFFFFF",
        default_dark="#202020"
    )

    BACKGROUND_SECONDARY = ColorPair.from_config(
        "background_colors", "secondary",
        default_light="#F3F3F3",
        default_dark="#2B2B2B"
    )

    BACKGROUND_TERTIARY = ColorPair.from_config(
        "background_colors", "tertiary",
        default_light="#E5E5E5",
        default_dark="#383838"
    )

    BACKGROUND_CARD = ColorPair.from_config(
        "background_colors", "card",
        default_light="#FFFFFF",
        default_dark="#2D2D2D"
    )

    BACKGROUND_HOVER = ColorPair.from_config(
        "background_colors", "hover",
        default_light="#E9E9E9",
        default_dark="#3D3D3D"
    )

    BACKGROUND_PRESSED = ColorPair.from_config(
        "background_colors", "pressed",
        default_light="#DBDBDB",
        default_dark="#4A4A4A"
    )
    
    # ============================================================
    # 文字色
    # ============================================================
    TEXT_PRIMARY = ColorPair.from_config(
        "text_colors", "primary",
        default_light="#1A1A1A",
        default_dark="#FFFFFF"
    )

    TEXT_SECONDARY = ColorPair.from_config(
        "text_colors", "secondary",
        default_light="#606060",
        default_dark="#ABABAB"
    )

    TEXT_TERTIARY = ColorPair.from_config(
        "text_colors", "tertiary",
        default_light="#8A8A8A",
        default_dark="#787878"
    )

    TEXT_DISABLED = ColorPair.from_config(
        "text_colors", "disabled",
        default_light="#A0A0A0",
        default_dark="#5D5D5D"
    )

    TEXT_LINK = ColorPair.from_config(
        "text_colors", "link",
        default_light="#0067C0",
        default_dark="#4CC2FF"
    )
    
    # ============================================================
    # 邊框色
    # ============================================================
    BORDER_DEFAULT = ColorPair.from_config(
        "border_colors", "default",
        default_light="#D6D6D6",
        default_dark="#454545"
    )

    BORDER_STRONG = ColorPair.from_config(
        "border_colors", "strong",
        default_light="#ABABAB",
        default_dark="#6B6B6B"
    )

    BORDER_SUBTLE = ColorPair.from_config(
        "border_colors", "subtle",
        default_light="#EBEBEB",
        default_dark="#3A3A3A"
    )
    
    # ============================================================
    # 狀態色
    # ============================================================
    SUCCESS = ColorPair.from_config(
        "status_colors", "success",
        default_light="#34C759",
        default_dark="#32D74B"
    )
    
    WARNING = ColorPair.from_config(
        "status_colors", "warning",
        default_light="#FF9500",
        default_dark="#FFD60A"
    )
    
    ERROR = ColorPair.from_config(
        "status_colors", "error",
        default_light="#FF3B30",
        default_dark="#FF453A"
    )
    
    INFO = ColorPair.from_config(
        "status_colors", "info",
        default_light="#007AFF",
        default_dark="#0A84FF"
    )
    
    # ============================================================
    # Overlay (覆蓋層) 顏色
    # ============================================================
    OVERLAY_FOV = ColorPairWithAlpha.from_config(
        "overlay_colors", "fov",
        default_light=(255, 0, 0, 180),
        default_dark=(255, 60, 60, 200)
    )
    
    OVERLAY_BOX = ColorPairWithAlpha.from_config(
        "overlay_colors", "detection_box",
        default_light=(0, 200, 0, 200),
        default_dark=(50, 255, 50, 220)
    )
    
    OVERLAY_CONFIDENCE_TEXT = ColorPairWithAlpha.from_config(
        "overlay_colors", "confidence_text",
        default_light=(255, 200, 0, 220),
        default_dark=(255, 255, 0, 240)
    )
    
    OVERLAY_DETECT_RANGE = ColorPairWithAlpha.from_config(
        "overlay_colors", "detect_range",
        default_light=(0, 100, 200, 80),
        default_dark=(0, 140, 255, 90)
    )
    
    OVERLAY_TRACKER_LINE = ColorPairWithAlpha.from_config(
        "overlay_colors", "tracker_line",
        default_light=(200, 200, 200, 40),
        default_dark=(255, 255, 255, 50)
    )
    
    OVERLAY_TRACKER_CURRENT = ColorPairWithAlpha.from_config(
        "overlay_colors", "tracker_current",
        default_light=(0, 200, 200, 50),
        default_dark=(0, 255, 255, 60)
    )
    
    OVERLAY_TRACKER_PREDICTED = ColorPairWithAlpha.from_config(
        "overlay_colors", "tracker_predicted",
        default_light=(200, 0, 200, 70),
        default_dark=(255, 0, 255, 80)
    )
    
    # ============================================================
    # 狀態面板 (Status Panel) 顏色
    # ============================================================
    PANEL_BACKGROUND = ColorPairWithAlpha.from_config(
        "status_panel_colors", "background",
        default_light=(255, 255, 255, 255),
        default_dark=(30, 30, 30, 255)
    )
    
    PANEL_BORDER = ColorPairWithAlpha.from_config(
        "status_panel_colors", "border",
        default_light=(0, 0, 0, 15),
        default_dark=(255, 255, 255, 20)
    )
    
    PANEL_SHADOW = ColorPairWithAlpha.from_config(
        "status_panel_colors", "shadow",
        default_light=(0, 0, 0, 40),
        default_dark=(0, 0, 0, 60)
    )
    
    PANEL_TITLE = ColorPair.from_config(
        "status_panel_colors", "title",
        default_light="#1A1A1A",
        default_dark="#FFFFFF"
    )
    
    PANEL_VERSION = ColorPair.from_config(
        "status_panel_colors", "version",
        default_light="#8A8A8A",
        default_dark="#707070"
    )
    
    PANEL_SEPARATOR = ColorPairWithAlpha.from_config(
        "status_panel_colors", "separator",
        default_light=(0, 0, 0, 10),
        default_dark=(255, 255, 255, 15)
    )
    
    # ============================================================
    # 導航列 (Navigation) 顏色
    # ============================================================
    NAV_BACKGROUND = ColorPair.from_config(
        "navigation_colors", "background",
        default_light="#F3F3F3",
        default_dark="#202020"
    )
    
    NAV_ITEM_HOVER = ColorPair.from_config(
        "navigation_colors", "item_hover",
        default_light="#E5E5E5",
        default_dark="#353535"
    )
    
    NAV_ITEM_SELECTED = ColorPair.from_config(
        "navigation_colors", "item_selected",
        default_light="#DCDCDC",
        default_dark="#404040"
    )
    
    NAV_ITEM_TEXT = ColorPair.from_config(
        "navigation_colors", "item_text",
        default_light="#3A3A3A",
        default_dark="#E0E0E0"
    )
    
    NAV_ITEM_ICON = ColorPair.from_config(
        "navigation_colors", "item_icon",
        default_light="#3A3A3A",
        default_dark="#E0E0E0"
    )
    
    # ============================================================
    # 設定卡片 (Setting Card) 顏色
    # ============================================================
    CARD_BACKGROUND = ColorPair.from_config(
        "card_colors", "background",
        default_light="#FFFFFF",
        default_dark="#2D2D2D"
    )
    
    CARD_BORDER = ColorPair.from_config(
        "card_colors", "border",
        default_light="#EBEBEB",
        default_dark="#3A3A3A"
    )
    
    CARD_TITLE = ColorPair.from_config(
        "card_colors", "title",
        default_light="#1A1A1A",
        default_dark="#FFFFFF"
    )
    
    CARD_DESCRIPTION = ColorPair.from_config(
        "card_colors", "description",
        default_light="#606060",
        default_dark="#ABABAB"
    )
    
    CARD_ICON = ColorPair.from_config(
        "card_colors", "icon",
        default_light="#3A3A3A",
        default_dark="#CCCCCC"
    )
    
    # ============================================================
    # 按鈕 (Button) 顏色
    # ============================================================
    BUTTON_PRIMARY_BG = ColorPair.from_config(
        "button_colors", "primary", "background",
        default_light="#0078D4",
        default_dark="#4CC2FF"
    )

    BUTTON_PRIMARY_TEXT = ColorPair.from_config(
        "button_colors", "primary", "text",
        default_light="#FFFFFF",
        default_dark="#000000"
    )

    BUTTON_PRIMARY_HOVER = ColorPair.from_config(
        "button_colors", "primary", "hover",
        default_light="#106EBE",
        default_dark="#3DB8FF"
    )

    BUTTON_PRIMARY_PRESSED = ColorPair.from_config(
        "button_colors", "primary", "pressed",
        default_light="#005A9E",
        default_dark="#2AABF0"
    )

    BUTTON_SECONDARY_BG = ColorPair.from_config(
        "button_colors", "secondary", "background",
        default_light="#F0F0F0",
        default_dark="#404040"
    )

    BUTTON_SECONDARY_TEXT = ColorPair.from_config(
        "button_colors", "secondary", "text",
        default_light="#1A1A1A",
        default_dark="#FFFFFF"
    )

    BUTTON_SECONDARY_HOVER = ColorPair.from_config(
        "button_colors", "secondary", "hover",
        default_light="#E5E5E5",
        default_dark="#505050"
    )

    BUTTON_SECONDARY_PRESSED = ColorPair.from_config(
        "button_colors", "secondary", "pressed",
        default_light="#D0D0D0",
        default_dark="#606060"
    )

    BUTTON_DISABLED_BG = ColorPair.from_config(
        "button_colors", "disabled", "background",
        default_light="#F5F5F5",
        default_dark="#2D2D2D"
    )

    BUTTON_DISABLED_TEXT = ColorPair.from_config(
        "button_colors", "disabled", "text",
        default_light="#A0A0A0",
        default_dark="#5D5D5D"
    )
    
    # ============================================================
    # 開關 (Switch) 顏色
    # ============================================================
    SWITCH_OFF_BG = ColorPair.from_config(
        "switch_colors", "off", "background",
        default_light="#CCCCCC",
        default_dark="#4A4A4A"
    )
    
    SWITCH_OFF_THUMB = ColorPair.from_config(
        "switch_colors", "off", "thumb",
        default_light="#FFFFFF",
        default_dark="#9A9A9A"
    )
    
    SWITCH_ON_BG = ColorPair.from_config(
        "switch_colors", "on", "background",
        default_light="#0078D4",
        default_dark="#4CC2FF"
    )
    
    SWITCH_ON_THUMB = ColorPair.from_config(
        "switch_colors", "on", "thumb",
        default_light="#FFFFFF",
        default_dark="#000000"
    )
    
    # ============================================================
    # 滑桿 (Slider) 顏色
    # ============================================================
    SLIDER_TRACK = ColorPair.from_config(
        "slider_colors", "track",
        default_light="#E0E0E0",
        default_dark="#4A4A4A"
    )
    
    SLIDER_FILL = ColorPair.from_config(
        "slider_colors", "fill",
        default_light="#0078D4",
        default_dark="#4CC2FF"
    )
    
    SLIDER_THUMB = ColorPair.from_config(
        "slider_colors", "thumb",
        default_light="#FFFFFF",
        default_dark="#FFFFFF"
    )
    
    SLIDER_THUMB_BORDER = ColorPair.from_config(
        "slider_colors", "thumb_border",
        default_light="#0078D4",
        default_dark="#4CC2FF"
    )
    
    # ============================================================
    # 下拉框 (ComboBox) 顏色
    # ============================================================
    COMBOBOX_BG = ColorPair.from_config(
        "combobox_colors", "background",
        default_light="#FFFFFF",
        default_dark="#2D2D2D"
    )
    
    COMBOBOX_BORDER = ColorPair.from_config(
        "combobox_colors", "border",
        default_light="#D6D6D6",
        default_dark="#454545"
    )
    
    COMBOBOX_TEXT = ColorPair.from_config(
        "combobox_colors", "text",
        default_light="#1A1A1A",
        default_dark="#FFFFFF"
    )
    
    COMBOBOX_DROPDOWN_BG = ColorPair.from_config(
        "combobox_colors", "dropdown_background",
        default_light="#FFFFFF",
        default_dark="#2D2D2D"
    )
    
    COMBOBOX_ITEM_HOVER = ColorPair.from_config(
        "combobox_colors", "item_hover",
        default_light="#E9E9E9",
        default_dark="#3D3D3D"
    )
    
    # ============================================================
    # 輸入框 (Input) 顏色
    # ============================================================
    INPUT_BG = ColorPair.from_config(
        "input_colors", "background",
        default_light="#FFFFFF",
        default_dark="#2D2D2D"
    )
    
    INPUT_BORDER = ColorPair.from_config(
        "input_colors", "border",
        default_light="#D6D6D6",
        default_dark="#454545"
    )
    
    INPUT_BORDER_FOCUS = ColorPair.from_config(
        "input_colors", "border_focus",
        default_light="#0078D4",
        default_dark="#4CC2FF"
    )
    
    INPUT_TEXT = ColorPair.from_config(
        "input_colors", "text",
        default_light="#1A1A1A",
        default_dark="#FFFFFF"
    )
    
    INPUT_PLACEHOLDER = ColorPair.from_config(
        "input_colors", "placeholder",
        default_light="#8A8A8A",
        default_dark="#787878"
    )
    
    # ============================================================
    # 對話框 (Dialog) 顏色
    # ============================================================
    DIALOG_BACKGROUND = ColorPair.from_config(
        "dialog_colors", "background",
        default_light="#FFFFFF",
        default_dark="rgba(45, 45, 45, 0.95)"
    )
    
    DIALOG_OVERLAY = ColorPairWithAlpha.from_config(
        "dialog_colors", "overlay",
        default_light=(0, 0, 0, 80),
        default_dark=(0, 0, 0, 120)
    )
    
    DIALOG_TITLE = ColorPair.from_config(
        "dialog_colors", "title",
        default_light="#1A1A1A",
        default_dark="#FFFFFF"
    )
    
    DIALOG_CONTENT = ColorPair.from_config(
        "dialog_colors", "content",
        default_light="#3A3A3A",
        default_dark="#CCCCCC"
    )
    
    # 對話框項目 (Dialog Item) 顏色 - 用於語言選擇卡片等
    DIALOG_ITEM_BACKGROUND = ColorPair.from_config(
        "dialog_colors", "item_background",
        default_light="rgba(0, 0, 0, 0.03)",
        default_dark="rgba(255, 255, 255, 0.05)"
    )
    
    DIALOG_ITEM_BORDER = ColorPair.from_config(
        "dialog_colors", "item_border",
        default_light="rgba(0, 0, 0, 0.08)",
        default_dark="rgba(255, 255, 255, 0.1)"
    )
    
    DIALOG_ITEM_HOVER = ColorPair.from_config(
        "dialog_colors", "item_hover",
        default_light="rgba(0, 0, 0, 0.06)",
        default_dark="rgba(255, 255, 255, 0.1)"
    )
    
    # ============================================================
    # 提示 (Tooltip) 顏色
    # ============================================================
    TOOLTIP_BG = ColorPair.from_config(
        "tooltip_colors", "background",
        default_light="#2D2D2D",
        default_dark="#F0F0F0"
    )
    
    TOOLTIP_TEXT = ColorPair.from_config(
        "tooltip_colors", "text",
        default_light="#FFFFFF",
        default_dark="#1A1A1A"
    )
    
    # ============================================================
    # 滾動條 (Scrollbar) 顏色
    # ============================================================
    SCROLLBAR_BG = ColorPair.from_config(
        "scrollbar_colors", "background",
        default_light="#F5F5F5",
        default_dark="#2D2D2D"
    )
    
    SCROLLBAR_THUMB = ColorPair.from_config(
        "scrollbar_colors", "thumb",
        default_light="#C0C0C0",
        default_dark="#5A5A5A"
    )
    
    SCROLLBAR_THUMB_HOVER = ColorPair.from_config(
        "scrollbar_colors", "thumb_hover",
        default_light="#A0A0A0",
        default_dark="#707070"
    )
    
    # ============================================================
    # 標籤/徽章 (Tag/Badge) 顏色
    # ============================================================
    TAG_DEFAULT_BG = ColorPair.from_config(
        "tag_colors", "default", "background",
        default_light="#E8E8E8",
        default_dark="#3D3D3D"
    )
    
    TAG_DEFAULT_TEXT = ColorPair.from_config(
        "tag_colors", "default", "text",
        default_light="#3A3A3A",
        default_dark="#E0E0E0"
    )
    
    TAG_PRIMARY_BG = ColorPair.from_config(
        "tag_colors", "primary", "background",
        default_light="#E6F2FF",
        default_dark="#0A3A5A"
    )
    
    TAG_PRIMARY_TEXT = ColorPair.from_config(
        "tag_colors", "primary", "text",
        default_light="#0067C0",
        default_dark="#4CC2FF"
    )
    
    TAG_SUCCESS_BG = ColorPair.from_config(
        "tag_colors", "success", "background",
        default_light="#E6F9ED",
        default_dark="#0A3A1A"
    )
    
    TAG_SUCCESS_TEXT = ColorPair.from_config(
        "tag_colors", "success", "text",
        default_light="#1A8F3C",
        default_dark="#32D74B"
    )
    
    TAG_WARNING_BG = ColorPair.from_config(
        "tag_colors", "warning", "background",
        default_light="#FFF4E6",
        default_dark="#3A2A0A"
    )
    
    TAG_WARNING_TEXT = ColorPair.from_config(
        "tag_colors", "warning", "text",
        default_light="#B86E00",
        default_dark="#FFD60A"
    )
    
    TAG_ERROR_BG = ColorPair.from_config(
        "tag_colors", "error", "background",
        default_light="#FFE6E6",
        default_dark="#3A0A0A"
    )
    
    TAG_ERROR_TEXT = ColorPair.from_config(
        "tag_colors", "error", "text",
        default_light="#CC2929",
        default_dark="#FF453A"
    )


# ============================================================
# 輔助函數
# ============================================================

def get_color(color_pair: ColorPair) -> str:
    """獲取當前主題對應的顏色 (HEX 字串)"""
    return color_pair.get()


def get_qcolor(color_pair: ColorPair) -> QColor:
    """獲取當前主題對應的 QColor 對象"""
    return color_pair.qcolor()


def get_rgba_color(color_pair: ColorPairWithAlpha) -> tuple:
    """獲取當前主題對應的 RGBA 元組"""
    return color_pair.get()


def get_rgba_qcolor(color_pair: ColorPairWithAlpha) -> QColor:
    """獲取當前主題對應的 QColor 對象 (含透明度)"""
    return color_pair.qcolor()


def hex_to_rgb(hex_color: str) -> tuple:
    """將 HEX 顏色轉換為 RGB 元組"""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def rgb_to_hex(r: int, g: int, b: int) -> str:
    """將 RGB 轉換為 HEX 顏色字串"""
    return f"#{r:02X}{g:02X}{b:02X}"


def to_css_rgba(color: QColor) -> str:
    """將 QColor 轉換為 CSS rgba() 字串"""
    return f"rgba({color.red()}, {color.green()}, {color.blue()}, {color.alpha() / 255.0:.2f})"


def to_css_rgb(color: QColor) -> str:
    """將 QColor 轉換為 CSS rgb() 字串"""
    return f"rgb({color.red()}, {color.green()}, {color.blue()})"


# ============================================================
# 樣式表生成器
# ============================================================

class StyleSheetGenerator:
    """根據主題顏色生成樣式表"""
    
    @staticmethod
    def get_card_style() -> str:
        """獲取設定卡片樣式"""
        return f"""
            SettingCard {{
                background-color: {ThemeColors.CARD_BACKGROUND.get()};
                border: 1px solid {ThemeColors.CARD_BORDER.get()};
                border-radius: 18px;
            }}
            SettingCard:hover {{
                background-color: {ThemeColors.BACKGROUND_HOVER.get()};
            }}
        """
    
    @staticmethod
    def get_button_primary_style() -> str:
        """獲取主要按鈕樣式"""
        return f"""
            PrimaryPushButton {{
                background-color: {ThemeColors.BUTTON_PRIMARY_BG.get()};
                color: {ThemeColors.BUTTON_PRIMARY_TEXT.get()};
                border: none;
                border-radius: 14px;
                padding: 8px 16px;
            }}
            PrimaryPushButton:hover {{
                background-color: {ThemeColors.BUTTON_PRIMARY_HOVER.get()};
            }}
            PrimaryPushButton:pressed {{
                background-color: {ThemeColors.BUTTON_PRIMARY_PRESSED.get()};
            }}
            PrimaryPushButton:disabled {{
                background-color: {ThemeColors.BUTTON_DISABLED_BG.get()};
                color: {ThemeColors.BUTTON_DISABLED_TEXT.get()};
            }}
        """
    
    @staticmethod
    def get_button_secondary_style() -> str:
        """獲取次要按鈕樣式"""
        return f"""
            PushButton {{
                background-color: {ThemeColors.BUTTON_SECONDARY_BG.get()};
                color: {ThemeColors.BUTTON_SECONDARY_TEXT.get()};
                border: 1px solid {ThemeColors.BORDER_DEFAULT.get()};
                border-radius: 14px;
                padding: 8px 16px;
            }}
            PushButton:hover {{
                background-color: {ThemeColors.BUTTON_SECONDARY_HOVER.get()};
            }}
            PushButton:pressed {{
                background-color: {ThemeColors.BUTTON_SECONDARY_PRESSED.get()};
            }}
        """
    
    @staticmethod
    def get_input_style() -> str:
        """獲取輸入框樣式"""
        return f"""
            LineEdit, SpinBox {{
                background-color: {ThemeColors.INPUT_BG.get()};
                color: {ThemeColors.INPUT_TEXT.get()};
                border: 1px solid {ThemeColors.INPUT_BORDER.get()};
                border-radius: 14px;
                padding: 6px 12px;
            }}
            LineEdit:focus, SpinBox:focus {{
                border-color: {ThemeColors.INPUT_BORDER_FOCUS.get()};
            }}
        """
    
    @staticmethod
    def get_combobox_style() -> str:
        """獲取下拉框樣式"""
        return f"""
            ComboBox {{
                background-color: {ThemeColors.COMBOBOX_BG.get()};
                color: {ThemeColors.COMBOBOX_TEXT.get()};
                border: 1px solid {ThemeColors.COMBOBOX_BORDER.get()};
                border-radius: 14px;
                padding: 6px 12px;
            }}
        """
    
    @staticmethod
    def get_scrollbar_style() -> str:
        """獲取滾動條樣式"""
        return f"""
            QScrollBar:vertical {{
                background: {ThemeColors.SCROLLBAR_BG.get()};
                width: 8px;
                border-radius: 6px;
            }}
            QScrollBar::handle:vertical {{
                background: {ThemeColors.SCROLLBAR_THUMB.get()};
                border-radius: 6px;
                min-height: 30px;
            }}
            QScrollBar::handle:vertical:hover {{
                background: {ThemeColors.SCROLLBAR_THUMB_HOVER.get()};
            }}
        """
    
    @staticmethod
    def get_status_panel_style() -> str:
        """獲取狀態面板樣式"""
        bg = ThemeColors.PANEL_BACKGROUND.qcolor()
        border = ThemeColors.PANEL_BORDER.qcolor()
        
        return f"""
            #container {{
                background-color: {to_css_rgba(bg)};
                border: 1px solid {to_css_rgba(border)};
                border-radius: 24px;
            }}
            #titleLabel {{
                color: {ThemeColors.PANEL_TITLE.get()};
                font-size: 14px;
                font-weight: bold;
            }}
            #versionLabel {{
                color: {ThemeColors.PANEL_VERSION.get()};
                font-size: 11px;
            }}
            #statusLabel {{
                color: {ThemeColors.TEXT_SECONDARY.get()};
                font-size: 12px;
            }}
            #statusValue {{
                color: {ThemeColors.TEXT_PRIMARY.get()};
                font-size: 12px;
                font-weight: 500;
            }}
            #separator {{
                background-color: {to_css_rgba(ThemeColors.PANEL_SEPARATOR.qcolor())};
            }}
        """
    
    @staticmethod
    def get_dialog_style() -> str:
        """獲取對話框樣式"""
        return f"""
            QDialog {{
                background-color: {ThemeColors.DIALOG_BACKGROUND.get()};
            }}
            QDialog QLabel#titleLabel {{
                color: {ThemeColors.DIALOG_TITLE.get()};
                font-size: 16px;
                font-weight: bold;
            }}
            QDialog QLabel {{
                color: {ThemeColors.DIALOG_CONTENT.get()};
            }}
        """
    
    @staticmethod  
    def get_tooltip_style() -> str:
        """獲取提示樣式"""
        return f"""
            QToolTip {{
                background-color: {ThemeColors.TOOLTIP_BG.get()};
                color: {ThemeColors.TOOLTIP_TEXT.get()};
                border: none;
                border-radius: 14px;
                padding: 6px 10px;
            }}
        """
    
    @staticmethod
    def get_all_styles() -> str:
        """獲取所有樣式的組合"""
        return "\n".join([
            StyleSheetGenerator.get_card_style(),
            StyleSheetGenerator.get_button_primary_style(),
            StyleSheetGenerator.get_button_secondary_style(),
            StyleSheetGenerator.get_input_style(),
            StyleSheetGenerator.get_combobox_style(),
            StyleSheetGenerator.get_scrollbar_style(),
            StyleSheetGenerator.get_tooltip_style(),
        ])


# 導出常用顏色類和函數
__all__ = [
    'ThemeColors',
    'ColorPair', 
    'ColorPairWithAlpha',
    'StyleSheetGenerator',
    'get_color',
    'get_qcolor',
    'get_rgba_color',
    'get_rgba_qcolor',
    'hex_to_rgb',
    'rgb_to_hex',
    'to_css_rgba',
    'to_css_rgb',
]
