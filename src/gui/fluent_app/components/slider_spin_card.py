# slider_spin_card.py
"""可重用的滑桿+數字輸入卡片組件

Layout rules (why the cards look the same at any window width / Windows scale):

* The slider is *flexible*: it grows with the card up to ``_SLIDER_MAX_W`` and shrinks down
  to ``_SLIDER_MIN_W``. ``slider_width`` is kept only for call-site compatibility — it used
  to be a hard minimum (160-400 px) that could never shrink, so a narrow page pushed the
  number box out of the card.
* The value widget (spin box / value label) gets a *fixed* width computed from the widest
  text it can show plus the stylesheet's own padding and spin buttons, so a value such as
  "1000 ms" or "0.00" is never hidden behind the up/down arrows.
* The title / description labels may shrink before the controls do; descriptions wrap onto more
  lines (the row grows), and a clipped title shows in full in a tooltip (``glass._GlassRowMixin``).

Qt high-DPI scaling is deliberately off in this app (the overlay needs physical pixels), so
sizes here are physical pixels derived from the widget's own font metrics, not constants
for one display scale.
"""

from typing import Callable, Optional
from PyQt6.QtCore import QEvent, Qt, pyqtSignal
from PyQt6.QtGui import QFontMetrics
from PyQt6.QtWidgets import QHBoxLayout, QSizePolicy, QToolTip, QWidget
from qfluentwidgets import FluentIconBase, BodyLabel
from core import ui_style
from .ui_kit import IS_GLASS, SettingCard, style_value_pill
from .no_wheel_widgets import NoWheelSlider as Slider, NoWheelSpinBox as SpinBox, NoWheelDoubleSpinBox as DoubleSpinBox

# Widths differ per interface style (``core.ui_style.metrics``). Glass: the value badge is a pill
# (``glass.style_value_pill``: 2 x 10 px padding + borders, no up/down buttons) and a two-column card
# still leaves the text ~200 px. Classic: the vendored spin box reserves ~80 px for its two buttons.
_METRICS = ui_style.metrics()
_SLIDER_MIN_W = _METRICS["slider_min_w"]    # narrowest usable track
_SLIDER_MAX_W = _METRICS["slider_max_w"]    # longer tracks add precision nobody can use; keeps every page's tracks aligned
_CONTROL_GAP = 12                           # between slider and value widget
_SPIN_CHROME_W = _METRICS["spin_chrome_w"]
_LABEL_PAD = _METRICS["label_pad"]          # the value label's own padding + borders


def text_width(widget: QWidget, text: str) -> int:
    """Width in physical pixels of ``text`` in ``widget``'s font (a few % slack for QSS font changes)."""
    return int(QFontMetrics(widget.font()).horizontalAdvance(text) * 1.1) + 2


def spin_min_width(spin, floor: int = 0) -> int:
    """Smallest width at which the widest value of ``spin`` (with prefix/suffix) stays readable."""
    samples = (spin.minimum(), spin.maximum(), 0)
    widest = max(text_width(spin, f"{spin.prefix()}{spin.textFromValue(v)}{spin.suffix()}") for v in samples)
    return max(widest + _SPIN_CHROME_W, floor)


def label_min_width(label: QWidget, samples, floor: int = 0) -> int:
    """Width for a read-only value label that must fit every string in ``samples``."""
    return max(max(text_width(label, s) for s in samples) + _LABEL_PAD, floor)


class _SliderCardBase(SettingCard):
    """Shared plumbing for the three slider cards: flexible slider + fixed-width value widget."""

    def _installControls(self, slider, value_widget, value_width: int) -> None:
        """Put ``slider`` and ``value_widget`` on the right of the card, right-aligned."""
        if IS_GLASS:
            style_value_pill(value_widget)
            value_widget.setProperty("glassPill", True)       # apply_widget_styles() leaves pills alone
        slider.setMinimumWidth(_SLIDER_MIN_W)
        slider.setMaximumWidth(_SLIDER_MAX_W)
        slider.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        value_widget.setFixedWidth(value_width)

        box = QWidget(self)
        row = QHBoxLayout(box)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(_CONTROL_GAP)
        row.addWidget(slider, 1)
        row.addWidget(value_widget, 0, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                      if IS_GLASS else Qt.AlignmentFlag.AlignRight)
        box.setMinimumWidth(_SLIDER_MIN_W + _CONTROL_GAP + value_width)
        box.setMaximumWidth(_SLIDER_MAX_W + _CONTROL_GAP + value_width)
        box.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self._controlBox = box
        self._valueWidget = value_widget

        # Text first to shrink: a label with an explicit minimum of 0 can be squeezed (and clips)
        # instead of forcing the controls off the right edge of the card.
        for lbl in (self.titleLabel, self.contentLabel):
            lbl.setMinimumWidth(0)
            lbl.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)

        self.hBoxLayout.addWidget(box, 3)
        self.hBoxLayout.addSpacing(16)

    def _setValueWidth(self, width: int) -> None:
        """Re-fit the value widget (e.g. after a new suffix or format function)."""
        self._valueWidget.setFixedWidth(width)
        self._controlBox.setMinimumWidth(_SLIDER_MIN_W + _CONTROL_GAP + width)
        self._controlBox.setMaximumWidth(_SLIDER_MAX_W + _CONTROL_GAP + width)

    def _syncSliderHandle(self) -> None:
        """Move qfluentwidgets' SliderHandle after a blocked setValue().

        That Slider positions its handle only via valueChanged →
        _adjustHandlePos. Config/preset loads call setValue() with signals
        blocked (to avoid feedback loops), so without this the numeric
        SpinBox updates while the thumb stays at the previous position.
        """
        adjust = getattr(self.slider, "_adjustHandlePos", None)
        if callable(adjust):
            adjust()
        self.slider.update()

    if not IS_GLASS:
        def event(self, e):
            """Classic rows: show the full title/description in a tooltip, but only while one is clipped.
            (The glass rows do this themselves in ``glass._GlassRowMixin``.)"""
            if e.type() == QEvent.Type.ToolTip:
                clipped = any(
                    lbl.isVisible() and lbl.width() < lbl.sizeHint().width()
                    for lbl in (self.titleLabel, self.contentLabel)
                )
                if clipped:
                    parts = [self.titleLabel.text()]
                    if self.contentLabel.isVisible():
                        parts.append(self.contentLabel.text())
                    QToolTip.showText(e.globalPos(), "\n".join(p for p in parts if p), self)
                else:
                    QToolTip.hideText()
                return True
            return super().event(e)


class SliderSpinCard(_SliderCardBase):
    """
    可重用的滑桿+SpinBox卡片，支持雙向同步

    用於需要精確數值輸入的設定項，如 FOV、偵測範圍、信心值等
    """
    valueChanged = pyqtSignal(int)

    def __init__(
        self,
        icon: FluentIconBase,
        title: str,
        min_val: int,
        max_val: int,
        suffix: str = "",
        description: str = "",
        slider_width: int = 300,   # kept for call sites; the slider is flexible now
        spin_width: int = 80,      # lower bound for the number box
        parent: Optional[QWidget] = None
    ):
        super().__init__(icon, title, description, parent)

        self._suffix = suffix

        # 創建 Slider
        self.slider = Slider(Qt.Orientation.Horizontal)
        self.slider.setRange(min_val, max_val)

        # 創建 SpinBox
        self.spinBox = SpinBox()
        self.spinBox.setRange(min_val, max_val)
        if suffix:
            self.spinBox.setSuffix(f" {suffix}" if not suffix.startswith(" ") else suffix)

        # 添加到佈局
        self._installControls(self.slider, self.spinBox, spin_min_width(self.spinBox, spin_width))

        # 連接信號 - 雙向同步
        self.slider.valueChanged.connect(self._onSliderChanged)
        self.spinBox.valueChanged.connect(self._onSpinChanged)

    def _onSliderChanged(self, value: int):
        """滑桿改變時同步 SpinBox"""
        self.spinBox.blockSignals(True)
        self.spinBox.setValue(value)
        self.spinBox.blockSignals(False)
        self.valueChanged.emit(value)

    def _onSpinChanged(self, value: int):
        """SpinBox 改變時同步滑桿"""
        self.slider.blockSignals(True)
        self.slider.setValue(value)
        self.slider.blockSignals(False)
        # qfluentwidgets' Slider only moves its handle on valueChanged —
        # blockSignals suppresses that, so force a handle/groove refresh.
        self._syncSliderHandle()
        self.valueChanged.emit(value)

    def setValue(self, value: int):
        """設定值（同時更新滑桿和 SpinBox）"""
        self.slider.blockSignals(True)
        self.spinBox.blockSignals(True)
        self.slider.setValue(value)
        # QSlider.setValue() silently clamps to [min, max] — re-read the
        # slider's own post-clamp value rather than trusting the raw 'value'
        # passed in, so the SpinBox can never disagree with where the thumb
        # actually landed (e.g. a stale/out-of-range value from an older
        # preset file no longer produces a slider/SpinBox mismatch).
        clamped = self.slider.value()
        self.spinBox.setValue(clamped)
        self.slider.blockSignals(False)
        self.spinBox.blockSignals(False)
        self._syncSliderHandle()

    def value(self) -> int:
        """取得當前值"""
        return self.slider.value()


class SliderDoubleSpinCard(_SliderCardBase):
    """
    可重用的滑桿+DoubleSpinBox卡片，支持浮點數雙向同步

    用於需要浮點數精確輸入的設定項，如延遲時間、間隔等
    """
    valueChanged = pyqtSignal(float)

    def __init__(
        self,
        icon: FluentIconBase,
        title: str,
        min_val: float,
        max_val: float,
        decimals: int = 2,
        step: float = 0.01,
        suffix: str = "",
        description: str = "",
        slider_width: int = 200,   # kept for call sites; the slider is flexible now
        spin_width: int = 80,      # lower bound for the number box
        parent: Optional[QWidget] = None
    ):
        super().__init__(icon, title, description, parent)

        self._decimals = decimals
        self._multiplier = 10 ** decimals

        # 創建 Slider (整數範圍)
        self.slider = Slider(Qt.Orientation.Horizontal)
        self.slider.setRange(int(min_val * self._multiplier), int(max_val * self._multiplier))

        # 創建 DoubleSpinBox
        self.spinBox = DoubleSpinBox()
        self.spinBox.setRange(min_val, max_val)
        self.spinBox.setSingleStep(step)
        self.spinBox.setDecimals(decimals)
        if suffix:
            self.spinBox.setSuffix(f" {suffix}" if not suffix.startswith(" ") else suffix)

        # 添加到佈局
        self._installControls(self.slider, self.spinBox, spin_min_width(self.spinBox, spin_width))

        # 連接信號 - 雙向同步
        self.slider.valueChanged.connect(self._onSliderChanged)
        self.spinBox.valueChanged.connect(self._onSpinChanged)

    def _onSliderChanged(self, value: int):
        """滑桿改變時同步 SpinBox"""
        float_val = value / self._multiplier
        self.spinBox.blockSignals(True)
        self.spinBox.setValue(float_val)
        self.spinBox.blockSignals(False)
        self.valueChanged.emit(float_val)

    def _onSpinChanged(self, value: float):
        """SpinBox 改變時同步滑桿"""
        int_val = int(round(value * self._multiplier))
        self.slider.blockSignals(True)
        self.slider.setValue(int_val)
        self.slider.blockSignals(False)
        self._syncSliderHandle()
        self.valueChanged.emit(value)

    def setValue(self, value: float):
        """設定值（同時更新滑桿和 SpinBox）"""
        self.slider.blockSignals(True)
        self.spinBox.blockSignals(True)
        self.slider.setValue(int(round(value * self._multiplier)))
        # Same clamp re-read as SliderSpinCard.setValue() above — derive the
        # DoubleSpinBox's value from where the slider's integer range actually
        # clamped it to, not the raw float, so the two can't disagree.
        self.spinBox.setValue(self.slider.value() / self._multiplier)
        self.slider.blockSignals(False)
        self.spinBox.blockSignals(False)
        self._syncSliderHandle()

    def value(self) -> float:
        """取得當前值"""
        return self.spinBox.value()


class SliderLabelCard(_SliderCardBase):
    """
    可重用的滑桿+標籤卡片，支持自定義格式化

    用於只需顯示數值的設定項，如 PID 參數、百分比等
    """
    valueChanged = pyqtSignal(int)

    def __init__(
        self,
        icon: FluentIconBase,
        title: str,
        min_val: int,
        max_val: int,
        format_func: Optional[Callable[[int], str]] = None,
        description: str = "",
        slider_width: int = 400,   # kept for call sites; the slider is flexible now
        label_width: int = 45,     # lower bound for the value label
        parent: Optional[QWidget] = None
    ):
        super().__init__(icon, title, description, parent)

        # 格式化函數，默認直接轉字符串
        self._format_func = format_func or (lambda v: str(v))
        self._label_floor = label_width

        # 創建 Slider
        self.slider = Slider(Qt.Orientation.Horizontal)
        self.slider.setRange(min_val, max_val)

        # 創建 Label
        self.label = BodyLabel(self._format_func(min_val))
        self.label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        # 添加到佈局
        self._installControls(self.slider, self.label, self._labelWidth())

        # 連接信號
        self.slider.valueChanged.connect(self._onSliderChanged)

    def _labelWidth(self) -> int:
        """Wide enough for the formatted min / mid / max, so the text never clips or makes the row jump."""
        lo, hi = self.slider.minimum(), self.slider.maximum()
        samples = [self._format_func(v) for v in (lo, (lo + hi) // 2, hi)]
        return label_min_width(self.label, samples, self._label_floor)

    def _onSliderChanged(self, value: int):
        """滑桿改變時更新標籤"""
        self.label.setText(self._format_func(value))
        self.valueChanged.emit(value)

    def setValue(self, value: int):
        """設定值"""
        self.slider.blockSignals(True)
        self.slider.setValue(value)
        # Format from the slider's own post-clamp value, not the raw 'value'
        # passed in — QSlider.setValue() silently clamps to [min, max], so an
        # out-of-range value (e.g. a stale/older preset file) would otherwise
        # leave the label showing a number the thumb's actual position
        # disagrees with.
        self.label.setText(self._format_func(self.slider.value()))
        self.slider.blockSignals(False)
        # qfluentwidgets only moves SliderHandle on valueChanged — blocked
        # signals leave the thumb stuck after config/preset loads.
        self._syncSliderHandle()

    def value(self) -> int:
        """取得當前值"""
        return self.slider.value()

    def setFormatFunc(self, func: Callable[[int], str]):
        """設定格式化函數"""
        self._format_func = func
        self.label.setText(self._format_func(self.slider.value()))
        self._setValueWidth(self._labelWidth())
