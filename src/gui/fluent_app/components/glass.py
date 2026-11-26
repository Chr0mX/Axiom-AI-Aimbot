"""Glass widgets: the card / switch / slider / group look of the redesigned interface.

Everything here is a **drop-in subclass** of the vendored qfluentwidgets widget the pages already
use (``SettingCard``, ``SwitchSettingCard``, ``SettingCardGroup``, ``Slider``), with the same
attributes (``titleLabel``, ``contentLabel``, ``hBoxLayout``, ``checkedChanged``, ``vBoxLayout``,
``addSettingCard`` ...). Pages switch to them by changing one import line, so every settings
binding, ``AdvancedVisibility`` shadowing of ``setVisible`` and retranslation path stays as it was.

Visual language
    * a group is a rounded (20 px) translucent container under an accent-bar section title;
    * a row is a transparent strip: bold title + muted wrapped description on the left, the control on
      the right, a hairline divider above every row but the first, a soft wash on hover;
    * switch = 46x24 pill that blends from a glass track to an accent gradient while the thumb slides;
    * slider = 6 px groove, accent-gradient fill, white round thumb with a glow on hover / press;
    * value badge = a centred pill (the spin box / label restyled).

Colours come from ``ThemeColors.GLASS_*`` (overridable in ``theme_colors.json``) and the live accent
(``core.ui_accent``); motion from ``components.motion`` / ``core.ui_motion``. Painters read the
current theme/accent at paint time, so a theme or accent change needs only a repaint.
"""
from __future__ import annotations

import types
from typing import Iterable

from PyQt6.QtCore import QEasingCurve, QEvent, QObject, QRect, QRectF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QLinearGradient, QPainter, QPen
from PyQt6.QtWidgets import QApplication, QFrame, QHBoxLayout, QLabel, QSizePolicy, QToolTip, QVBoxLayout, QWidget
from qfluentwidgets import SettingCard, SwitchSettingCard, Slider, isDarkTheme, qconfig, setCustomStyleSheet
from qfluentwidgets.components.widgets.slider import SliderHandle

from core import ui_accent, ui_motion
from ..theme_colors import ThemeColors
from .motion import HoverProgress

ROW_MIN_H = 52           # a row with a one-line title; descriptions make it grow
ROW_PAD_V = 8
GROUP_RADIUS = 20
ROW_HOVER_RADIUS = 12
QWIDGETSIZE_MAX = 16777215


# ----------------------------------------------------------------------------- helpers
def accent_palette() -> ui_accent.AccentPalette:
    return ui_accent.derive(ui_accent.get_accent(), isDarkTheme())


def _q(rgb_or_rgba) -> QColor:
    return QColor(*rgb_or_rgba)


def _css(rgba) -> str:
    r, g, b, a = rgba
    return f"rgba({r},{g},{b},{a / 255:.3f})"


css_rgba = _css          # public alias for pages that build their own QSS


def _pair_css(pair) -> tuple[str, str]:
    """(light, dark) ``rgba(...)`` strings of a ``ColorPairWithAlpha`` for ``setCustomStyleSheet``."""
    return _css(pair.light), _css(pair.dark)


class _Signals(QObject):
    """Broadcast for look changes the widgets cannot see themselves (accent colour, animation switch)."""
    appearanceChanged = pyqtSignal()


signals = _Signals()


def notify_appearance_changed() -> None:
    signals.appearanceChanged.emit()


# ----------------------------------------------------------------------------- rows
class _GlassRowMixin:
    """Row behaviour shared by plain, switch and slider cards (mix in *before* the vendored card)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._initGlass()

    def _initGlass(self) -> None:
        self._hover = HoverProgress(self)
        # No icon column: the title starts at the card's left padding.
        self.iconLabel.hide()
        lay = self.hBoxLayout
        item = lay.itemAt(1)
        if item is not None and item.spacerItem() is not None:
            lay.takeAt(1)
        # The text column takes the space the vendored card gave to a trailing stretch, and its labels
        # fill it (the vendored AlignLeft keeps a wrapped label at its small size hint).
        for i in range(lay.count() - 1, -1, -1):
            sp = lay.itemAt(i).spacerItem()
            if sp is not None and sp.expandingDirections() & Qt.Orientation.Horizontal:
                lay.takeAt(i)
                break
        lay.setStretchFactor(self.vBoxLayout, 3)
        for lbl in (self.titleLabel, self.contentLabel):
            self.vBoxLayout.removeWidget(lbl)
            self.vBoxLayout.addWidget(lbl)
        lay.setContentsMargins(18, ROW_PAD_V, 0, ROW_PAD_V)
        # The vendored card is fixed at 50/70 px; let the row follow its (wrapped) text instead.
        self.setMinimumHeight(ROW_MIN_H)
        self.setMaximumHeight(QWIDGETSIZE_MAX)
        self.contentLabel.setWordWrap(True)
        self.contentLabel.setMinimumWidth(0)
        self.titleLabel.setMinimumWidth(0)
        self.titleLabel.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        self.contentLabel.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        self.applyGlassStyle()
        signals.appearanceChanged.connect(self.update)

    def applyGlassStyle(self) -> None:
        t_l, t_d = _pair_css(ThemeColors.GLASS_TEXT)
        m_l, m_d = _pair_css(ThemeColors.GLASS_TEXT_MUTED)
        qss = ("QLabel{{color:{t};font-size:14px;font-weight:600;background:transparent}}"
               "QLabel#contentLabel{{color:{m};font-size:12px;font-weight:400}}")
        setCustomStyleSheet(self, qss.format(t=t_l, m=m_l), qss.format(t=t_d, m=m_d))

    # -- height follows the wrapped description at the text column's *actual* width. Qt's own
    # height-for-width cannot see that width through the nested row layouts (it under-measures rows whose
    # control column is wide), so the row measures it after layout and raises its minimum height.
    def _fitHeight(self) -> None:
        self.hBoxLayout.activate()          # make the text column's geometry current before reading it
        text_w = self.vBoxLayout.geometry().width()
        if text_w <= 0:
            return
        need = self.titleLabel.sizeHint().height() + 2 * ROW_PAD_V
        content_h = 0
        if self.contentLabel.isVisibleTo(self) and self.contentLabel.text():
            # font metrics, not label.heightForWidth(): that is clamped by the minimum height set below,
            # so a measurement taken while the row was narrow would never shrink again
            content_h = self.contentLabel.fontMetrics().boundingRect(
                QRect(0, 0, text_w, 100_000), int(Qt.TextFlag.TextWordWrap), self.contentLabel.text()).height()
            need += content_h + 2
        if self.contentLabel.minimumHeight() != content_h:    # the nested layout gives a wrapped label one line otherwise
            self.contentLabel.setMinimumHeight(content_h)
        need = max(ROW_MIN_H, need, self.hBoxLayout.sizeHint().height())   # tall controls (checkbox grids)
        if need != self.minimumHeight():
            self.setMinimumHeight(need)

    # Rows size themselves (``_fitHeight`` -> minimum height); advertising height-for-width as well made
    # the group layouts cache heights measured while a row was still narrow.
    def hasHeightForWidth(self) -> bool:
        return False

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._fitHeight()
        if not getattr(self, "_fitPending", False):       # once more after the parent layout has settled
            self._fitPending = True
            QTimer.singleShot(0, self._fitLater)

    def _fitLater(self) -> None:
        self._fitPending = False
        self._fitHeight()

    # -- hover wash
    def enterEvent(self, e):
        self._hover.set(True)
        super().enterEvent(e)

    def leaveEvent(self, e):
        self._hover.set(False)
        super().leaveEvent(e)

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        h = self._hover.value
        if h > 0.01:
            c = ThemeColors.GLASS_ROW_HOVER.qcolor()
            c.setAlphaF(c.alphaF() * h)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(c)
            p.drawRoundedRect(QRectF(self.rect()).adjusted(6, 2, -6, -2), ROW_HOVER_RADIUS, ROW_HOVER_RADIUS)
        if self.geometry().top() > 0:      # hairline above every row that is not first in its parent
            p.setPen(QPen(ThemeColors.GLASS_DIVIDER.qcolor(), 1))
            p.drawLine(18, 0, self.width() - 18, 0)

    # -- full text only while the title itself is clipped (descriptions wrap instead of clipping)
    def event(self, e):
        if e.type() == QEvent.Type.LayoutRequest:
            self._fitHeight()
        if e.type() == QEvent.Type.ToolTip:
            if self.titleLabel.width() < self.titleLabel.sizeHint().width():
                parts = [self.titleLabel.text()]
                if self.contentLabel.isVisible():
                    parts.append(self.contentLabel.text())
                QToolTip.showText(e.globalPos(), "\n".join(p for p in parts if p), self)
            else:
                QToolTip.hideText()
            return True
        return super().event(e)


class GlassSettingCard(_GlassRowMixin, SettingCard):
    """Plain row: title + description left, any widget(s) added to ``hBoxLayout`` on the right."""


# ----------------------------------------------------------------------------- switch
def _paint_indicator(self, e) -> None:
    p = QPainter(self)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    w, h = self.width(), self.height()
    t = max(0.0, min(1.0, (self.sliderX - 5) / 20.0))
    enabled = self.isEnabled()
    pal = accent_palette()
    p.setOpacity(1.0 if enabled else 0.45)
    track = QRectF(0.5, 0.5, w - 1, h - 1)
    r = h / 2
    p.setPen(QPen(ThemeColors.GLASS_PILL_BORDER.qcolor(), 1))
    p.setBrush(ThemeColors.GLASS_TRACK.qcolor())
    p.drawRoundedRect(track, r, r)
    if t > 0:
        g = QLinearGradient(0, 0, w, 0)
        g.setColorAt(0, _q(pal.hi))
        g.setColorAt(1, _q(pal.base))
        p.setOpacity((1.0 if enabled else 0.45) * t)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(g)
        p.drawRoundedRect(track, r, r)
        p.setOpacity(1.0 if enabled else 0.45)
    if enabled and getattr(self, "isHover", False):
        glow = QColor(*pal.glow)
        glow.setAlpha(int(glow.alpha() * (0.35 + 0.65 * t)))
        p.setPen(QPen(glow, 2))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRoundedRect(track.adjusted(-1, -1, 1, 1), r + 1, r + 1)
    x, d = self.sliderX, 16.0
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor(0, 0, 0, 56))
    p.drawEllipse(QRectF(x, (h - d) / 2 + 1, d, d))
    p.setBrush(QColor(255, 255, 255) if t > 0.5 or not isDarkTheme() else QColor(226, 229, 244))
    p.drawEllipse(QRectF(x, (h - d) / 2, d, d))


def style_switch(button) -> None:
    """Restyle a qfluentwidgets ``SwitchButton`` in place: 46x24 pill, no On/Off text, eased slide."""
    ind = button.indicator
    ind.setFixedSize(46, 24)
    button.setFixedHeight(24)
    button.label.hide()
    button.hBox.setSpacing(0)
    ind.slideAni.setEasingCurve(QEasingCurve(QEasingCurve.Type.OutCubic))
    ind.slideAni.setDuration(ui_motion.duration(160))
    ind.paintEvent = types.MethodType(_paint_indicator, ind)
    signals.appearanceChanged.connect(ind.update)


class GlassSwitchSettingCard(_GlassRowMixin, SwitchSettingCard):
    """Row with an on/off pill switch (same API as ``SwitchSettingCard``)."""

    def _initGlass(self) -> None:
        super()._initGlass()
        style_switch(self.switchButton)


def refresh_motion(root: QWidget) -> None:
    """Re-read the animation switch for widgets that cache a duration (switch thumbs)."""
    from qfluentwidgets.components.widgets.switch_button import Indicator
    for ind in root.findChildren(Indicator):
        ind.slideAni.setDuration(ui_motion.duration(160))


# ----------------------------------------------------------------------------- slider
class _GlassHandle(SliderHandle):
    """White round thumb; ``radius`` (5 idle / 6 hover / 4 pressed, animated by the base class) drives
    its size and the accent halo."""

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        pal = accent_palette()
        c = self.rect().center()
        cx, cy = c.x() + 0.5, c.y() + 0.5
        k = (self.radius - 4) / 2.0          # 0 pressed .. 0.5 idle .. 1 hover
        halo = max(0.0, k - 0.5) * 2 + (0.35 if self.radius <= 4 else 0.0)
        if halo > 0:
            g = QColor(*pal.glow)
            g.setAlpha(int(min(1.0, halo) * g.alpha()))
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(g)
            p.drawEllipse(QRectF(cx - 10.5, cy - 10.5, 21, 21))
        r = 6.5 + k * 1.2
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(0, 0, 0, 60))
        p.drawEllipse(QRectF(cx - r, cy - r + 1, 2 * r, 2 * r))
        p.setBrush(QColor(255, 255, 255))
        p.drawEllipse(QRectF(cx - r, cy - r, 2 * r, 2 * r))
        p.setBrush(_q(pal.base))
        p.drawEllipse(QRectF(cx - 2.6, cy - 2.6, 5.2, 5.2))


class GlassSlider(Slider):
    """Horizontal slider with a gradient-filled groove and the glass thumb (wheel handling is added
    by ``NoWheelSlider``; handle-position sync stays the vendored ``_adjustHandlePos`` contract)."""

    def _postInit(self):
        super()._postInit()
        old = self.handle
        new = _GlassHandle(self)
        new.pressed.connect(self.sliderPressed)
        new.released.connect(self.sliderReleased)
        self.handle = new
        old.hide()
        old.deleteLater()
        new.show()
        self._adjustHandlePos()
        signals.appearanceChanged.connect(self.update)

    def paintEvent(self, e):
        if self.orientation() != Qt.Orientation.Horizontal:
            return super().paintEvent(e)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)
        r = self.handle.width() / 2
        cy = self.handle.height() / 2
        length = max(self.width() - 2 * r, 1.0)
        groove = QRectF(r, cy - 3, length, 6)
        p.setBrush(ThemeColors.GLASS_TRACK.qcolor())
        p.drawRoundedRect(groove, 3, 3)
        span = self.maximum() - self.minimum()
        if span <= 0:
            return
        fill = (self.value() - self.minimum()) / span * length
        if fill <= 0.5:
            return
        pal = accent_palette()
        g = QLinearGradient(groove.left(), 0, groove.left() + max(fill, 24.0), 0)
        g.setColorAt(0, _q(pal.hi))
        g.setColorAt(1, _q(pal.base))
        p.setBrush(g)
        p.drawRoundedRect(QRectF(groove.left(), groove.top(), fill, 6), 3, 3)


# ----------------------------------------------------------------------------- value badge
def style_value_pill(widget: QWidget) -> None:
    """Centred pill badge for a value label or a (still editable) spin box."""
    widget.setAlignment(Qt.AlignmentFlag.AlignCenter)
    widget.setFixedHeight(28)
    hide = getattr(widget, "setSymbolVisible", None)
    if callable(hide):
        hide(False)                        # spin boxes: no up/down buttons, type or use the keys
    f_l, f_d = _pair_css(ThemeColors.GLASS_PILL)
    b_l, b_d = _pair_css(ThemeColors.GLASS_PILL_BORDER)
    t_l, t_d = _pair_css(ThemeColors.GLASS_TEXT)
    pal = accent_palette()
    cls = "QLabel" if isinstance(widget, QLabel) else widget.metaObject().className()
    qss = ("{c}{{background:{f};border:1px solid {b};border-radius:13px;padding:2px 10px;color:{t};"
           "font-size:13px;font-weight:600;min-height:22px}}"
           "{c}:focus{{border:1px solid rgb({r},{g},{bb})}}")
    r, g, bb = pal.base
    setCustomStyleSheet(widget, qss.format(c=cls, f=f_l, b=b_l, t=t_l, r=r, g=g, bb=bb),
                        qss.format(c=cls, f=f_d, b=b_d, t=t_d, r=r, g=g, bb=bb))


class TextStyler(QObject):
    """Keeps a plain ``QLabel``'s colour (and extra QSS) in step with the light/dark theme."""

    def __init__(self, label: QLabel, extra: str = "", muted: bool = False):
        super().__init__(label)
        self._label, self._extra, self._muted = label, extra, muted
        qconfig.themeChanged.connect(self.apply)
        self.apply()

    def apply(self, *_args) -> None:
        pair = ThemeColors.GLASS_TEXT_MUTED if self._muted else ThemeColors.GLASS_TEXT
        self._label.setStyleSheet(f"color:{pair.qcolor().name()};background:transparent;border:none;{self._extra}")


# ----------------------------------------------------------------------------- group
class GlassSettingCardGroup(QWidget):
    """Accent-bar title above a rounded glass container holding the rows.

    API of qfluentwidgets' ``SettingCardGroup``: ``titleLabel``, ``addSettingCard(s)``,
    ``vBoxLayout`` (the container's layout — extra widgets appended there land *after* every card,
    exactly like the old ``cardLayout``-then-``vBoxLayout`` order), ``adjustSize()``.
    """

    def __init__(self, title: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, False)
        self._outer = QVBoxLayout(self)
        self._outer.setContentsMargins(0, 0, 0, 0)
        self._outer.setSpacing(10)
        self._outer.setAlignment(Qt.AlignmentFlag.AlignTop)

        head = QHBoxLayout()
        head.setContentsMargins(4, 0, 0, 0)
        head.setSpacing(8)
        self._bar = _AccentBar(self)
        self.titleLabel = QLabel(title, self)
        head.addWidget(self._bar, 0, Qt.AlignmentFlag.AlignVCenter)
        head.addWidget(self.titleLabel, 1)
        self._outer.addLayout(head)
        if not title:                      # frame-only panel (e.g. the advanced-settings switch)
            self._bar.hide()
            self.titleLabel.hide()
            self._outer.removeItem(head)
            self._outer.setSpacing(0)

        self._frame = GlassCard(self)
        self.vBoxLayout = QVBoxLayout(self._frame)
        self.vBoxLayout.setContentsMargins(0, 0, 0, 0)
        self.vBoxLayout.setSpacing(0)
        self._outer.addWidget(self._frame)
        self._cardCount = 0
        self._restyleTitle()
        qconfig.themeChanged.connect(self._restyleTitle)

    def _restyleTitle(self, *_args) -> None:
        # A plain QLabel is not registered with the Fluent stylesheet machinery, so pick the side now.
        c = ThemeColors.GLASS_TEXT.qcolor()
        self.titleLabel.setStyleSheet(
            f"QLabel{{color:{c.name()};font-size:15px;font-weight:700;background:transparent}}")

    def addSettingCard(self, card: QWidget) -> None:
        card.setParent(self._frame)
        self.vBoxLayout.insertWidget(self._cardCount, card)
        self._cardCount += 1

    def addSettingCards(self, cards: Iterable[QWidget]) -> None:
        for c in cards:
            self.addSettingCard(c)

    def adjustSize(self) -> None:          # call-site compatibility: layouts size the group themselves
        self.updateGeometry()


class _AccentBar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(4, 16)
        signals.appearanceChanged.connect(self.update)

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        pal = accent_palette()
        g = QLinearGradient(0, 0, 0, self.height())
        g.setColorAt(0, _q(pal.hi))
        g.setColorAt(1, _q(pal.base))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(g)
        p.drawRoundedRect(QRectF(self.rect()), 2, 2)


class GlassCard(QFrame):
    """Rounded translucent container (group body, notes card, status panels)."""


    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.NoFrame)

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setBrush(ThemeColors.GLASS_CARD.qcolor())
        p.setPen(QPen(ThemeColors.GLASS_CARD_BORDER.qcolor(), 1))
        p.drawRoundedRect(QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5), GROUP_RADIUS, GROUP_RADIUS)


# ----------------------------------------------------------------------------- generic widget QSS
def _control_qss(kind: str, dark: bool) -> str:
    pair = lambda p: _css(p.dark if dark else p.light)   # noqa: E731
    pal = accent_palette()
    fill, border = pair(ThemeColors.GLASS_PILL), pair(ThemeColors.GLASS_PILL_BORDER)
    text = pair(ThemeColors.GLASS_TEXT)
    hover = _css(ThemeColors.GLASS_ROW_HOVER.dark if dark else ThemeColors.GLASS_ROW_HOVER.light)
    r, g, b = pal.base
    hr, hg, hb = pal.hi
    if kind == "PrimaryPushButton":
        return (f"PrimaryPushButton{{color:rgb{pal.on_accent};border:none;border-radius:12px;padding:5px 16px;"
                f"background:qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 rgb({hr},{hg},{hb}),stop:1 rgb({r},{g},{b}))}}"
                f"PrimaryPushButton:hover{{background:qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 rgb({min(255, hr + 14)},"
                f"{min(255, hg + 14)},{min(255, hb + 14)}),stop:1 rgb({min(255, r + 14)},{min(255, g + 14)},{min(255, b + 14)}))}}"
                f"PrimaryPushButton:pressed{{background:rgb({max(0, r - 18)},{max(0, g - 18)},{max(0, b - 18)})}}"
                f"PrimaryPushButton:disabled{{background:{border};color:{_css(ThemeColors.GLASS_TEXT_MUTED.dark if dark else ThemeColors.GLASS_TEXT_MUTED.light)}}}")
    pad = "5px 34px 5px 12px" if kind == "ComboBox" else "5px 14px"
    return (f"{kind}{{background:{fill};border:1px solid {border};border-radius:12px;color:{text};padding:{pad}}}"
            f"{kind}:hover{{background:{hover};border:1px solid rgb({r},{g},{b})}}"
            f"{kind}:pressed{{background:{border}}}"
            f"{kind}:focus{{border:1px solid rgb({r},{g},{b})}}")


def apply_widget_styles(root: QWidget) -> None:
    """Glass look for the stock controls under ``root`` (buttons, combos, line edits, spin boxes).

    Per-widget ``setCustomStyleSheet`` — the only override the vendored stylesheets allow — with
    light and dark variants at once, so a theme switch needs no re-application; accent changes do.
    """
    from PyQt6.QtWidgets import QAbstractSpinBox, QTextEdit
    from qfluentwidgets import ComboBox, LineEdit, PrimaryPushButton, PushButton

    for kind, cls in (("PushButton", PushButton), ("ComboBox", ComboBox), ("LineEdit", LineEdit)):
        qss_l, qss_d = _control_qss(kind, False), _control_qss(kind, True)
        for w in root.findChildren(cls):
            if isinstance(w, PrimaryPushButton) or w.property("glassPill"):
                continue
            setCustomStyleSheet(w, qss_l, qss_d)
    pl, pd = _control_qss("PrimaryPushButton", False), _control_qss("PrimaryPushButton", True)
    for w in root.findChildren(PrimaryPushButton):
        setCustomStyleSheet(w, pl, pd)
    dark = isDarkTheme()
    fill = css_rgba(ThemeColors.GLASS_PILL.dark if dark else ThemeColors.GLASS_PILL.light)
    border = css_rgba(ThemeColors.GLASS_PILL_BORDER.dark if dark else ThemeColors.GLASS_PILL_BORDER.light)
    text = (ThemeColors.GLASS_TEXT.dark if dark else ThemeColors.GLASS_TEXT.light)[:3]
    for w in root.findChildren(QTextEdit):          # plain Qt text areas are not Fluent-registered
        w.setStyleSheet(f"QTextEdit{{background:{fill};border:1px solid {border};border-radius:12px;"
                        f"color:rgb{text};padding:6px}}")
    for w in root.findChildren(QAbstractSpinBox):
        if w.property("glassPill"):
            continue
        cname = w.metaObject().className()
        setCustomStyleSheet(w, _control_qss(cname, False), _control_qss(cname, True))


def tooltip_qss() -> str:
    """One tooltip look for both themes (dark glass, light text): the app-level stylesheet is applied once —
    changing it re-polishes every widget in the window (seconds), so it must not follow the theme."""
    return ("QToolTip{background-color:rgba(30,32,58,245);color:rgb(240,242,252);border:1px solid rgba(255,255,255,40);"
            "border-radius:8px;padding:6px 10px;font-size:12px}")


def install_tooltip_style() -> None:
    """App-wide tooltip look (plain ``QToolTip``; the only global rule — widgets keep per-widget QSS).
    Call it early (before the pages exist), once."""
    app = QApplication.instance()
    marker = "/*axiom-tooltip*/"
    if app is None or marker in app.styleSheet():
        return
    app.setStyleSheet(app.styleSheet() + marker + tooltip_qss())
