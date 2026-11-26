"""Which interface look the Qt app uses — pure Python (no Qt).

Two looks share one code base: ``glass`` (the default: floating pill navigation, translucent rows,
accent colour, two-column pages) and ``classic`` (the earlier look: the vendored left navigation, the
stock setting cards, one column). Pages bind their widget classes when ``components/ui_kit`` is first
imported, so the style is chosen **once at start-up** (``main.py`` calls :func:`set_style` right
before the window module is imported) and a change in the settings takes effect after a restart.

``metrics()`` is the one table of numbers that differ between the looks (slider/value-pill widths,
page margins, title size), so the layout code has a single place to read them from.
"""
from __future__ import annotations

import logging
from typing import Dict

logger = logging.getLogger(__name__)

GLASS = 'glass'
CLASSIC = 'classic'
STYLES = (GLASS, CLASSIC)
DEFAULT_STYLE = GLASS

_METRICS: Dict[str, Dict[str, object]] = {
    GLASS: {
        'slider_min_w': 96,        # a two-column card still leaves the text ~200 px
        'slider_max_w': 320,
        'spin_chrome_w': 28,       # value pill: padding + borders, no spin buttons
        'label_pad': 28,
        'page_margins': (28, 14, 28, 28),
        'title_px': 24,
        'title_bold': False,       # DemiBold, set by the page
    },
    CLASSIC: {
        'slider_min_w': 120,
        'slider_max_w': 360,
        'spin_chrome_w': 96,       # the vendored spin box reserves ~80 px for its two buttons
        'label_pad': 8,
        'page_margins': (36, 20, 36, 36),
        'title_px': 32,
        'title_bold': True,
    },
}

_style = DEFAULT_STYLE
_locked = False


def normalize(value) -> str:
    """``'glass'``/``'classic'`` for a valid (any-case) string, else the default style."""
    text = str(value or '').strip().lower()
    return text if text in STYLES else DEFAULT_STYLE


def get_style() -> str:
    return _style


def is_glass() -> bool:
    return _style == GLASS


def metrics(style: str | None = None) -> Dict[str, object]:
    """The numbers of ``style`` (the active one by default); the returned dict is a copy."""
    return dict(_METRICS[normalize(style) if style is not None else _style])


def lock() -> None:
    """Called by ``components/ui_kit`` once it has bound the widget classes: later changes are ignored."""
    global _locked
    _locked = True


def is_locked() -> bool:
    return _locked


def set_style(value) -> str:
    """Choose the look. Ignored (with a warning) once the widget classes are bound; returns the active style."""
    global _style
    wanted = normalize(value)
    if _locked:
        if wanted != _style:
            logger.warning("UI style %r requested after the widgets were bound; staying on %r until restart",
                           wanted, _style)
        return _style
    _style = wanted
    return _style


def _reset_for_tests() -> None:
    global _style, _locked
    _style, _locked = DEFAULT_STYLE, False
