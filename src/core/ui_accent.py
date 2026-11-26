"""Accent colour for the Qt interface — pure Python (no Qt), shared by every glass widget.

One user-chosen ``#rrggbb`` accent (``Config.accent_color``) drives sliders, switches, the active
navigation capsule, focus rings and links. ``derive()`` turns it into the small palette the painters
need (gradient ends, glow, readable text on top of it) so no widget picks its own shades.
"""
from __future__ import annotations

import colorsys
import re
from typing import NamedTuple, Tuple

DEFAULT_ACCENT = "#7c6cf0"
# (label, hex) swatches offered on the Visuals page — violet, purple, pink, blue, teal.
PRESET_ACCENTS: Tuple[Tuple[str, str], ...] = (
    ("Violet", "#7c6cf0"),
    ("Purple", "#a662aa"),
    ("Pink", "#e05fb0"),
    ("Blue", "#4f8cff"),
    ("Teal", "#2fb8a6"),
)

_HEX_RE = re.compile(r"^#?([0-9a-fA-F]{6})$")
RGB = Tuple[int, int, int]


class AccentPalette(NamedTuple):
    base: RGB        # the accent itself (switch on-state, slider fill end, links)
    hi: RGB          # lighter, hue-shifted gradient start
    lo: RGB          # darker shade (pressed, borders)
    glow: Tuple[int, int, int, int]   # translucent halo colour (RGBA)
    soft: Tuple[int, int, int, int]   # translucent fill for selected rows (RGBA)
    on_accent: RGB   # readable text/icon colour drawn on top of ``base``


def parse_hex(value, default: str = DEFAULT_ACCENT) -> RGB:
    """(r, g, b) for ``#rrggbb`` / ``rrggbb``; falls back to ``default`` for anything else."""
    m = _HEX_RE.match(str(value or "").strip())
    if not m:
        m = _HEX_RE.match(default)
    h = m.group(1)
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def to_hex(rgb: RGB) -> str:
    return "#%02x%02x%02x" % tuple(max(0, min(255, int(round(c)))) for c in rgb)


def _hls(rgb: RGB):
    return colorsys.rgb_to_hls(*(c / 255.0 for c in rgb))


def _from_hls(h: float, l: float, s: float) -> RGB:
    r, g, b = colorsys.hls_to_rgb(h % 1.0, max(0.0, min(1.0, l)), max(0.0, min(1.0, s)))
    return int(round(r * 255)), int(round(g * 255)), int(round(b * 255))


def luminance(rgb: RGB) -> float:
    """Relative luminance (WCAG) of an sRGB colour, 0..1."""
    def lin(c):
        c = c / 255.0
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = (lin(c) for c in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def derive(hex_color, dark: bool = True) -> AccentPalette:
    """Palette for ``hex_color``. ``dark`` only changes how strong the glow / selected fill are."""
    base = parse_hex(hex_color)
    h, l, s = _hls(base)
    hi = _from_hls(h - 0.025, min(0.80, l + 0.16), s)      # lighter, nudged toward cooler hue
    lo = _from_hls(h, max(0.12, l - 0.14), s)
    glow = (*base, 110 if dark else 80)
    soft = (*base, 70 if dark else 46)
    on_accent = (255, 255, 255) if luminance(base) < 0.55 else (24, 24, 32)
    return AccentPalette(base, hi, lo, glow, soft, on_accent)


# ---- process-wide current accent (set once by the window from config, read by painters) ----
_current = DEFAULT_ACCENT


def set_accent(hex_color) -> str:
    """Make ``hex_color`` the live accent; returns the normalised ``#rrggbb`` actually stored."""
    global _current
    _current = to_hex(parse_hex(hex_color))
    return _current


def get_accent() -> str:
    return _current
