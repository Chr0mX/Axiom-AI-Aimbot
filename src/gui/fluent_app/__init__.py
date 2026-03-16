# fluent_app 模組
"""
Fluent Design 風格的 GUI 應用程式模組
"""

from .theme_colors import (
    ThemeColors,
    ColorPair,
    ColorPairWithAlpha,
    StyleSheetGenerator,
    get_color,
    get_qcolor,
    get_rgba_color,
    get_rgba_qcolor,
)

from .theme_manager import (
    ThemeManager,
    get_theme_manager,
    apply_theme_to_app,
    get_current_stylesheet,
)

__all__ = [
    'ThemeColors',
    'ColorPair',
    'ColorPairWithAlpha', 
    'StyleSheetGenerator',
    'get_color',
    'get_qcolor',
    'get_rgba_color',
    'get_rgba_qcolor',
    'ThemeManager',
    'get_theme_manager',
    'apply_theme_to_app',
    'get_current_stylesheet',
]