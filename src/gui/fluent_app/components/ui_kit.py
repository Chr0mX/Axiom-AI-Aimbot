"""The widget classes the pages build on, chosen by the interface style (``core.ui_style``).

Pages import ``SettingCard``/``SwitchSettingCard``/``SettingCardGroup``/``Card``/``Slider`` from here
instead of from ``qfluentwidgets`` or ``glass``. With the ``glass`` look they are the translucent
``glass`` widgets; with ``classic`` they are the stock qfluentwidgets ones the app used before the
redesign. Importing this module binds the choice (``ui_style.lock()``): ``main.py`` calls
``ui_style.set_style()`` before the first GUI import, and a later change needs a restart.
"""
from __future__ import annotations

from core import ui_style

IS_GLASS = ui_style.is_glass()
ui_style.lock()

if IS_GLASS:
    from .glass import (
        GlassCard as Card,
        GlassSettingCard as SettingCard,
        GlassSettingCardGroup as SettingCardGroup,
        GlassSlider as Slider,
        GlassSwitchSettingCard as SwitchSettingCard,
        TextStyler,
        style_value_pill,
    )
else:
    from qfluentwidgets import (
        CardWidget as Card,
        SettingCard,
        SettingCardGroup,
        Slider,
        SwitchSettingCard,
    )

    def TextStyler(label, extra: str = "", muted: bool = False):          # noqa: N802 - same name as glass.TextStyler
        """Classic: the plain title look the notes card always had (colour comes from the theme)."""
        label.setStyleSheet("font-size: 14px; font-weight: 600;")
        return None

    def style_value_pill(widget) -> None:
        """Classic: the stock spin box / label look; nothing to restyle."""
        return None


def is_group(widget) -> bool:
    """True for a setting group of the active look (what ``BasePage.addContent`` batches into columns)."""
    return IS_GLASS and isinstance(widget, SettingCardGroup)


__all__ = ["IS_GLASS", "Card", "SettingCard", "SettingCardGroup", "SwitchSettingCard", "Slider",
           "TextStyler", "style_value_pill", "is_group"]
