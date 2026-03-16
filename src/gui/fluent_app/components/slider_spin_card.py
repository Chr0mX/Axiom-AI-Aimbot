# slider_spin_card.py
"""可重用的滑桿+數字輸入卡片組件"""

from typing import Callable, Optional, Union
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QWidget
from qfluentwidgets import SettingCard, FluentIconBase, BodyLabel
from .no_wheel_widgets import NoWheelSlider as Slider, NoWheelSpinBox as SpinBox, NoWheelDoubleSpinBox as DoubleSpinBox


class SliderSpinCard(SettingCard):
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
        slider_width: int = 300,
        spin_width: int = 80,
        parent: Optional[QWidget] = None
    ):
        super().__init__(icon, title, description, parent)

        self._suffix = suffix

        # 創建 Slider
        self.slider = Slider(Qt.Orientation.Horizontal)
        self.slider.setRange(min_val, max_val)
        self.slider.setMinimumWidth(slider_width)

        # 創建 SpinBox
        self.spinBox = SpinBox()
        self.spinBox.setRange(min_val, max_val)
        self.spinBox.setMinimumWidth(spin_width)
        if suffix:
            self.spinBox.setSuffix(f" {suffix}" if not suffix.startswith(" ") else suffix)

        # 添加到佈局
        self.hBoxLayout.addWidget(self.slider, 0, Qt.AlignmentFlag.AlignRight)
        self.hBoxLayout.addWidget(self.spinBox, 0, Qt.AlignmentFlag.AlignRight)
        self.hBoxLayout.addSpacing(16)

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
        self.valueChanged.emit(value)

    def setValue(self, value: int):
        """設定值（同時更新滑桿和 SpinBox）"""
        self.slider.blockSignals(True)
        self.spinBox.blockSignals(True)
        self.slider.setValue(value)
        self.spinBox.setValue(value)
        self.slider.blockSignals(False)
        self.spinBox.blockSignals(False)

    def value(self) -> int:
        """取得當前值"""
        return self.slider.value()


class SliderDoubleSpinCard(SettingCard):
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
        slider_width: int = 200,
        spin_width: int = 80,
        parent: Optional[QWidget] = None
    ):
        super().__init__(icon, title, description, parent)

        self._decimals = decimals
        self._multiplier = 10 ** decimals

        # 創建 Slider (整數範圍)
        self.slider = Slider(Qt.Orientation.Horizontal)
        self.slider.setRange(int(min_val * self._multiplier), int(max_val * self._multiplier))
        self.slider.setMinimumWidth(slider_width)

        # 創建 DoubleSpinBox
        self.spinBox = DoubleSpinBox()
        self.spinBox.setRange(min_val, max_val)
        self.spinBox.setSingleStep(step)
        self.spinBox.setDecimals(decimals)
        self.spinBox.setMinimumWidth(spin_width)
        if suffix:
            self.spinBox.setSuffix(f" {suffix}" if not suffix.startswith(" ") else suffix)

        # 添加到佈局
        self.hBoxLayout.addWidget(self.slider, 0, Qt.AlignmentFlag.AlignRight)
        self.hBoxLayout.addWidget(self.spinBox, 0, Qt.AlignmentFlag.AlignRight)
        self.hBoxLayout.addSpacing(16)

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
        int_val = int(value * self._multiplier)
        self.slider.blockSignals(True)
        self.slider.setValue(int_val)
        self.slider.blockSignals(False)
        self.valueChanged.emit(value)

    def setValue(self, value: float):
        """設定值（同時更新滑桿和 SpinBox）"""
        self.slider.blockSignals(True)
        self.spinBox.blockSignals(True)
        self.slider.setValue(int(value * self._multiplier))
        self.spinBox.setValue(value)
        self.slider.blockSignals(False)
        self.spinBox.blockSignals(False)

    def value(self) -> float:
        """取得當前值"""
        return self.spinBox.value()


class SliderLabelCard(SettingCard):
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
        slider_width: int = 400,
        label_width: int = 45,
        parent: Optional[QWidget] = None
    ):
        super().__init__(icon, title, description, parent)

        # 格式化函數，默認直接轉字符串
        self._format_func = format_func or (lambda v: str(v))

        # 創建 Slider
        self.slider = Slider(Qt.Orientation.Horizontal)
        self.slider.setRange(min_val, max_val)
        self.slider.setMinimumWidth(slider_width)

        # 創建 Label
        self.label = BodyLabel(self._format_func(min_val))
        self.label.setMinimumWidth(label_width)

        # 添加到佈局
        self.hBoxLayout.addWidget(self.slider, 0, Qt.AlignmentFlag.AlignRight)
        self.hBoxLayout.addWidget(self.label, 0, Qt.AlignmentFlag.AlignRight)
        self.hBoxLayout.addSpacing(16)

        # 連接信號
        self.slider.valueChanged.connect(self._onSliderChanged)

    def _onSliderChanged(self, value: int):
        """滑桿改變時更新標籤"""
        self.label.setText(self._format_func(value))
        self.valueChanged.emit(value)

    def setValue(self, value: int):
        """設定值"""
        self.slider.blockSignals(True)
        self.slider.setValue(value)
        self.label.setText(self._format_func(value))
        self.slider.blockSignals(False)

    def value(self) -> int:
        """取得當前值"""
        return self.slider.value()

    def setFormatFunc(self, func: Callable[[int], str]):
        """設定格式化函數"""
        self._format_func = func
        self.label.setText(self._format_func(self.slider.value()))
