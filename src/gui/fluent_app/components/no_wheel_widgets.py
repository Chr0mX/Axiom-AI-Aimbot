# no_wheel_widgets.py
"""自訂控制項 - 禁用滑鼠滾輪調整"""

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QWidget
from qfluentwidgets import Slider, SpinBox, DoubleSpinBox


class NoWheelSlider(Slider):
    """禁用滑鼠滾輪的 Slider，並修復滑鼠移動就會改值的問題"""

    def wheelEvent(self, event):
        # 忽略滾輪事件，讓父容器處理滾動
        event.ignore()
    
    def mouseMoveEvent(self, event):
        # 只有在滑鼠左鍵按下時才允許拖曳改變值
        if event.buttons() & Qt.MouseButton.LeftButton:
            super().mouseMoveEvent(event)
        else:
            event.ignore()


class NoWheelSpinBox(SpinBox):
    """禁用滑鼠滾輪的 SpinBox"""

    def wheelEvent(self, event):
        # 忽略滾輪事件，讓父容器處理滾動
        event.ignore()


class NoWheelDoubleSpinBox(DoubleSpinBox):
    """禁用滑鼠滾輪的 DoubleSpinBox"""

    def wheelEvent(self, event):
        # 忽略滾輪事件，讓父容器處理滾動
        event.ignore()
