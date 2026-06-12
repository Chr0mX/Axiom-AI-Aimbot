# fluent_app 模組
"""
Fluent Design 風格的 GUI 應用程式模組
"""

from .theme_colors import (
    ThemeColors,
    ColorPair,
    ColorPairWithAlpha,
    StyleSheetGenerator,
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
    'ThemeManager',
    'get_theme_manager',
    'apply_theme_to_app',
    'get_current_stylesheet',
]