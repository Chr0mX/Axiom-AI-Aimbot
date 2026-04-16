# aim_page.py
"""Aim Assist Page - Model Settings, PID, Bezier Curves, Smart Tracking"""

import os
import math
import glob
import threading
from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QFileDialog, QMessageBox, QStackedWidget
from PyQt6.QtGui import QDesktopServices
from qfluentwidgets import (
    SettingCardGroup, ComboBoxSettingCard, SwitchSettingCard,
    PushSettingCard, RangeSettingCard, OptionsSettingCard,
    FluentIcon,
    BodyLabel, ComboBox, PrimaryPushButton, SettingCard,
    qconfig, ConfigItem, OptionsConfigItem, RangeConfigItem,
    BoolValidator, OptionsValidator, RangeValidator, PushButton,
    SegmentedWidget
)
from ..components.no_wheel_widgets import NoWheelSlider as Slider, NoWheelSpinBox as SpinBox, NoWheelDoubleSpinBox as DoubleSpinBox
from ..components.slider_spin_card import SliderSpinCard, SliderLabelCard

from ..base_page import BasePage
from ..language_manager import t


class AimPage(BasePage):
    """Aim Assist Settings Page"""

    def __init__(self, parent=None):
        super().__init__("tab_aim_control", parent)
        self._config = None
        self._initWidgets()
        self._initLayout()
        self._connectSignals()

    def setConfig(self, config):
        """Sets Config instance and loads values"""
        self._config = config
        
        # Dynamically adjust detection range upper limit, supports 2K/4K screens
        if hasattr(self, 'detectRangeCard') and self._config:
            max_h = max(1080, self._config.height)
            self.detectRangeCard.slider.setMaximum(max_h)
            self.detectRangeCard.spinBox.setMaximum(max_h)
            
        self._loadFromConfig()

    def showEvent(self, event):
        """頁面顯示時同步 idle detect 開關狀態（可能被其他頁面修改）"""
        super().showEvent(event)
        if self._config and hasattr(self, 'idleDetectEnableCard'):
            self.idleDetectEnableCard.setChecked(
                getattr(self._config, 'idle_detect_enabled', True)
            )

    def _initWidgets(self):
        """Initializes all controls"""

        # === Model Settings ===
        self.modelGroup = SettingCardGroup(t("model_settings"), self.scrollWidget)

        # Model selection
        self.modelCombo = ComboBox()
        self.modelCombo.setMinimumWidth(200)
        # Note: _refreshModelList() is called during setConfig properly
        self.modelCard = SettingCard(
            FluentIcon.ROBOT,
            t("model"),
            "",
            self.modelGroup
        )
        self.modelCard.hBoxLayout.addWidget(self.modelCombo, 0, Qt.AlignmentFlag.AlignRight)
        self.modelCard.hBoxLayout.addSpacing(16)

        # Inference backend
        self.inferenceBackendCombo = ComboBox()
        self.inferenceBackendCombo.addItems(["Auto", "CUDA", "DirectML", "CPU"])
        self.inferenceBackendCombo.setMinimumWidth(150)
        self.inferenceBackendCard = SettingCard(
            FluentIcon.COMMAND_PROMPT,
            t("inference_backend"),
            "",
            self.modelGroup
        )
        self.inferenceBackendCard.hBoxLayout.addWidget(self.inferenceBackendCombo, 0, Qt.AlignmentFlag.AlignRight)
        self.inferenceBackendCard.hBoxLayout.addSpacing(16)

        # Open Model Folder
        self.openModelFolderBtn = PrimaryPushButton(t("open_model_folder"))
        self.openModelFolderCard = SettingCard(
            FluentIcon.FOLDER,
            t("open_model_folder"),
            "",
            self.modelGroup
        )
        self.openModelFolderCard.hBoxLayout.addWidget(self.openModelFolderBtn, 0, Qt.AlignmentFlag.AlignRight)
        self.openModelFolderCard.hBoxLayout.addSpacing(16)

        # === FOV & Detection Range ===
        self.fovGroup = SettingCardGroup(t("fov_and_detect_range"), self.scrollWidget)

        # FOV Size - using SliderSpinCard
        self.fovCard = SliderSpinCard(
            FluentIcon.ZOOM,
            t("fov_size"),
            50, 500,
            description="",
            parent=self.fovGroup
        )

        # FOV Follow Mouse
        self.fovFollowCard = SwitchSettingCard(
            FluentIcon.MOVE,
            t("fov_follow_mouse"),
            "",
            parent=self.fovGroup
        )

        # AI Detection Range - using SliderSpinCard
        self.detectRangeCard = SliderSpinCard(
            FluentIcon.FULL_SCREEN,
            t("detect_range_size"),
            100, 1080,
            description=t("detect_range_note"),
            parent=self.fovGroup
        )

        # === General Parameters ===
        self.generalGroup = SettingCardGroup(t("general_params"), self.scrollWidget)

        # Detection Interval - using SliderSpinCard
        self.detectIntervalCard = SliderSpinCard(
            FluentIcon.SPEED_HIGH,
            t("detect_interval"),
            1, 100,
            suffix="ms",
            description="",
            parent=self.generalGroup
        )

        # Screenshot Interval - using SliderSpinCard
        self.screenshotIntervalCard = SliderSpinCard(
            FluentIcon.CAMERA,
            t("screenshot_interval"),
            1, 100,
            suffix="ms",
            description="",
            parent=self.generalGroup
        )

        # Minimum Confidence - using SliderSpinCard
        self.confidenceCard = SliderSpinCard(
            FluentIcon.CERTIFICATE,
            t("min_confidence"),
            1, 100,
            suffix="%",
            description="",
            parent=self.generalGroup
        )

        # Aim Part
        self.aimPartCombo = ComboBox()
        self.aimPartCombo.addItems([t("head"), t("body"), t("both")])
        self.aimPartCombo.setMinimumWidth(120)
        self.aimPartCard = SettingCard(
            FluentIcon.PEOPLE,
            t("aim_part"),
            "",
            self.generalGroup
        )
        self.aimPartCard.hBoxLayout.addWidget(self.aimPartCombo, 0, Qt.AlignmentFlag.AlignRight)
        self.aimPartCard.hBoxLayout.addSpacing(16)

        # Mouse Movement Method
        self.mouseMoveCombo = ComboBox()
        self.mouseMoveCombo.addItems(["ddxoft", "mouse_event", "sendinput", "arduino", "makcu", "xbox"])
        self.mouseMoveCombo.setMinimumWidth(150)
        self.mouseMoveCard = SettingCard(
            FluentIcon.FINGERPRINT,
            t("mouse_move_method"),
            "",
            self.generalGroup
        )
        self.mouseMoveCard.hBoxLayout.addWidget(self.mouseMoveCombo, 0, Qt.AlignmentFlag.AlignRight)
        self.mouseMoveCard.hBoxLayout.addSpacing(16)

        # Screenshot Method
        self.screenshotMethodCombo = ComboBox()
        self.screenshotMethodCombo.addItems(["mss", "dxcam", "uvc"])
        self.screenshotMethodCombo.setMinimumWidth(150)
        self.screenshotMethodCard = SettingCard(
            FluentIcon.CAMERA,
            t("screenshot_method"),
            "",
            self.generalGroup
        )
        self.screenshotMethodCard.hBoxLayout.addWidget(self.screenshotMethodCombo, 0, Qt.AlignmentFlag.AlignRight)
        self.screenshotMethodCard.hBoxLayout.addSpacing(16)

        self.uvcDeviceCard = SliderSpinCard(
            FluentIcon.CAMERA,
            "UVC Device Index",
            0, 16,
            suffix="",
            description="",
            parent=self.generalGroup
        )

        self.uvcWidthCard = SliderSpinCard(
            FluentIcon.FULL_SCREEN,
            "UVC Width",
            320, 7680,
            suffix="px",
            description="",
            parent=self.generalGroup
        )

        self.uvcHeightCard = SliderSpinCard(
            FluentIcon.FULL_SCREEN,
            "UVC Height",
            240, 4320,
            suffix="px",
            description="",
            parent=self.generalGroup
        )
        self.uvcWidthCard.setVisible(False)
        self.uvcHeightCard.setVisible(False)

        self.uvcResolutionCombo = ComboBox()
        self.uvcResolutionCombo.setMinimumWidth(180)
        self.uvcResolutionCard = SettingCard(
            FluentIcon.FULL_SCREEN,
            "UVC Resolution",
            "Auto-detect supported resolutions",
            self.generalGroup
        )
        self.uvcResolutionCard.hBoxLayout.addWidget(self.uvcResolutionCombo, 0, Qt.AlignmentFlag.AlignRight)
        self.uvcResolutionCard.hBoxLayout.addSpacing(16)

        self.uvcRefreshResolutionBtn = PushButton(t("refresh"))
        self.uvcRefreshResolutionBtn.setFixedWidth(80)
        self.uvcRefreshResolutionCard = SettingCard(
            FluentIcon.SYNC,
            "Refresh UVC Resolution List",
            "",
            self.generalGroup
        )
        self.uvcRefreshResolutionCard.hBoxLayout.addWidget(self.uvcRefreshResolutionBtn, 0, Qt.AlignmentFlag.AlignRight)
        self.uvcRefreshResolutionCard.hBoxLayout.addSpacing(16)

        self.uvcFpsCard = SliderSpinCard(
            FluentIcon.SPEED_MEDIUM,
            "UVC FPS",
            1, 240,
            suffix="",
            description="",
            parent=self.generalGroup
        )

        self.uvcCaptureMethodCombo = ComboBox()
        self.uvcCaptureMethodCombo.addItems(["dshow", "msmf", "auto", "any"])
        self.uvcCaptureMethodCombo.setMinimumWidth(140)
        self.uvcCaptureMethodCard = SettingCard(
            FluentIcon.CAMERA,
            "UVC Capture Method",
            "Select OpenCV capture backend",
            self.generalGroup
        )
        self.uvcCaptureMethodCard.hBoxLayout.addWidget(self.uvcCaptureMethodCombo, 0, Qt.AlignmentFlag.AlignRight)
        self.uvcCaptureMethodCard.hBoxLayout.addSpacing(16)

        self.uvcPreviewCard = SwitchSettingCard(
            FluentIcon.VIEW,
            "UVC Preview Window",
            "",
            parent=self.generalGroup
        )

        self.uvcPreviewScaleCombo = ComboBox()
        self.uvcPreviewScaleCombo.addItems(["scale_to_fit", "scale_to_canvas", "fit_to_screen"])
        self.uvcPreviewScaleCombo.setMinimumWidth(170)
        self.uvcPreviewScaleCard = SettingCard(
            FluentIcon.FULL_SCREEN,
            "UVC Preview Scale Mode",
            "",
            self.generalGroup
        )
        self.uvcPreviewScaleCard.hBoxLayout.addWidget(self.uvcPreviewScaleCombo, 0, Qt.AlignmentFlag.AlignRight)
        self.uvcPreviewScaleCard.hBoxLayout.addSpacing(16)

        # Always Aim (no need to press aim key)
        self.alwaysAimCard = SwitchSettingCard(
            FluentIcon.FINGERPRINT,
            t("always_aim"),
            "",
            parent=self.generalGroup
        )

        # Keep Detecting (even if aim key is not pressed)
        self.keepDetectingCard = SwitchSettingCard(
            FluentIcon.UPDATE,
            t("keep_detecting"),
            "",
            parent=self.generalGroup
        )

        # Idle Detection Reduce Frequency Enable
        self.idleDetectEnableCard = SwitchSettingCard(
            FluentIcon.SPEED_MEDIUM,
            t("idle_detect_enabled"),
            "",
            parent=self.generalGroup
        )

        # Idle Detection Interval
        self.idleDetectIntervalCard = SliderSpinCard(
            FluentIcon.SPEED_MEDIUM,
            t("idle_detect_interval"),
            5, 500,
            suffix="ms",
            description="",
            parent=self.generalGroup
        )

        self.singleTargetCard = SwitchSettingCard(
            FluentIcon.PEOPLE,
            t("single_target_mode"),
            "",
            parent=self.generalGroup
        )

        # === Arduino Settings (only shown when arduino is selected) ===
        self.arduinoGroup = SettingCardGroup("Arduino", self.scrollWidget)

        # COM Port Selection
        self.comPortCombo = ComboBox()
        self.comPortCombo.setMinimumWidth(120)
        self.comPortCombo.addItem(t("no_com_port"))
        self._refreshComPorts()

        self.comRefreshBtn = PushButton(t("refresh"))
        self.comRefreshBtn.setFixedWidth(80)

        self.comPortCard = SettingCard(
            FluentIcon.CONNECT,
            t("arduino_com_port"),
            "",
            self.arduinoGroup
        )
        self.comPortCard.hBoxLayout.addWidget(self.comPortCombo, 0, Qt.AlignmentFlag.AlignRight)
        self.comPortCard.hBoxLayout.addWidget(self.comRefreshBtn, 0, Qt.AlignmentFlag.AlignRight)
        self.comPortCard.hBoxLayout.addSpacing(16)

        # 連線狀態
        self._isArduinoConnected = False
        self.connectionLabel = BodyLabel(t("disconnected"))
        self.connectionLabel.setStyleSheet("color: #e74c3c; font-weight: bold;")
        self.connectionCard = SettingCard(
            FluentIcon.WIFI,
            t("connected") + " / " + t("disconnected"),
            "",
            self.arduinoGroup
        )
        self.connectionCard.hBoxLayout.addWidget(self.connectionLabel, 0, Qt.AlignmentFlag.AlignRight)
        self.connectionCard.hBoxLayout.addSpacing(16)

        # Arduino 連線/斷線按鈕
        self.arduinoConnectBtn = PushButton(t("arduino_connect"))
        self.arduinoConnectBtn.setFixedWidth(120)
        self.arduinoConnectCard = SettingCard(
            FluentIcon.LINK,
            t("arduino_connect"),
            t("arduino_connect_desc"),
            self.arduinoGroup
        )
        self.arduinoConnectCard.hBoxLayout.addWidget(self.arduinoConnectBtn, 0, Qt.AlignmentFlag.AlignRight)
        self.arduinoConnectCard.hBoxLayout.addSpacing(16)

        # 使用教學
        self.guideBtn = PushButton(t("arduino_guide"))
        self.guideCard = SettingCard(
            FluentIcon.BOOK_SHELF,
            t("arduino_guide"),
            "",
            self.arduinoGroup
        )
        self.guideCard.hBoxLayout.addWidget(self.guideBtn, 0, Qt.AlignmentFlag.AlignRight)
        self.guideCard.hBoxLayout.addSpacing(16)

        # 一鍵硬體偽裝
        self.spoofBtn = PushButton(t("spoof_device"))
        self.spoofCard = SettingCard(
            FluentIcon.VPN,
            t("spoof_device"),
            "",
            self.arduinoGroup
        )
        self.spoofCard.hBoxLayout.addWidget(self.spoofBtn, 0, Qt.AlignmentFlag.AlignRight)
        self.spoofCard.hBoxLayout.addSpacing(16)

        # 驗證偽裝
        self.verifySpoofBtn = PushButton(t("verify_spoof"))
        self.verifySpoofCard = SettingCard(
            FluentIcon.ACCEPT,
            t("verify_spoof"),
            "",
            self.arduinoGroup
        )
        self.verifySpoofCard.hBoxLayout.addWidget(self.verifySpoofBtn, 0, Qt.AlignmentFlag.AlignRight)
        self.verifySpoofCard.hBoxLayout.addSpacing(16)

        # 測試愛心移動
        self.testHeartBtn = PushButton(t("test_move_heart"))
        self.testHeartCard = SettingCard(
            FluentIcon.HEART,
            t("test_move_heart"),
            "",
            self.arduinoGroup
        )
        self.testHeartCard.hBoxLayout.addWidget(self.testHeartBtn, 0, Qt.AlignmentFlag.AlignRight)
        self.testHeartCard.hBoxLayout.addSpacing(16)

        # === Xbox 360 虛擬手把設定（僅在選擇 xbox 時顯示）===
        self.xboxGroup = SettingCardGroup("Xbox 360 Controller", self.scrollWidget)

        # === MAKCU KM Host 設定（僅在選擇 makcu 時顯示）===
        self.makcuGroup = SettingCardGroup("MAKCU", self.scrollWidget)

        # MAKCU COM Port Selection
        self.makcuComPortCombo = ComboBox()
        self.makcuComPortCombo.setMinimumWidth(120)
        self.makcuComPortCombo.addItem(t("no_com_port"))
        self._refreshMakcuComPorts()

        self.makcuComRefreshBtn = PushButton(t("refresh"))
        self.makcuComRefreshBtn.setFixedWidth(80)

        self.makcuComPortCard = SettingCard(
            FluentIcon.CONNECT,
            t("makcu_com_port"),
            "",
            self.makcuGroup
        )
        self.makcuComPortCard.hBoxLayout.addWidget(self.makcuComPortCombo, 0, Qt.AlignmentFlag.AlignRight)
        self.makcuComPortCard.hBoxLayout.addWidget(self.makcuComRefreshBtn, 0, Qt.AlignmentFlag.AlignRight)
        self.makcuComPortCard.hBoxLayout.addSpacing(16)

        # MAKCU 連線狀態
        self._isMakcuConnected = False
        self.makcuConnectionLabel = BodyLabel(t("disconnected"))
        self.makcuConnectionLabel.setStyleSheet("color: #e74c3c; font-weight: bold;")
        self.makcuConnectionCard = SettingCard(
            FluentIcon.WIFI,
            t("connected") + " / " + t("disconnected"),
            "",
            self.makcuGroup
        )
        self.makcuConnectionCard.hBoxLayout.addWidget(self.makcuConnectionLabel, 0, Qt.AlignmentFlag.AlignRight)
        self.makcuConnectionCard.hBoxLayout.addSpacing(16)

        # MAKCU 連線/斷線按鈕
        self.makcuConnectBtn = PushButton(t("makcu_connect"))
        self.makcuConnectBtn.setFixedWidth(120)
        self.makcuConnectCard = SettingCard(
            FluentIcon.LINK,
            t("makcu_connect"),
            t("makcu_connect_desc"),
            self.makcuGroup
        )
        self.makcuConnectCard.hBoxLayout.addWidget(self.makcuConnectBtn, 0, Qt.AlignmentFlag.AlignRight)
        self.makcuConnectCard.hBoxLayout.addSpacing(16)

        # 靈敏度
        self.xboxSensitivityCard = SliderSpinCard(
            FluentIcon.SPEED_HIGH,
            t("xbox_sensitivity"),
            10, 500,
            suffix="%",
            description="",
            parent=self.xboxGroup
        )

        # 死區
        self.xboxDeadzoneCard = SliderSpinCard(
            FluentIcon.REMOVE,
            t("xbox_deadzone"),
            0, 50,
            suffix="%",
            description="",
            parent=self.xboxGroup
        )

        # 連線狀態
        self._isXboxConnected = False
        self.xboxConnectionLabel = BodyLabel(t("disconnected"))
        self.xboxConnectionLabel.setStyleSheet("color: #e74c3c; font-weight: bold;")
        self.xboxConnectionCard = SettingCard(
            FluentIcon.GAME,
            t("connected") + " / " + t("disconnected"),
            "",
            self.xboxGroup
        )
        self.xboxConnectionCard.hBoxLayout.addWidget(self.xboxConnectionLabel, 0, Qt.AlignmentFlag.AlignRight)
        self.xboxConnectionCard.hBoxLayout.addSpacing(16)

        # 手動連線/斷線按鈕
        self.xboxConnectBtn = PushButton(t("xbox_connect"))
        self.xboxConnectBtn.setFixedWidth(120)
        self.xboxConnectCard = SettingCard(
            FluentIcon.WIFI,
            t("xbox_connect"),
            t("xbox_connect_desc"),
            self.xboxGroup
        )
        self.xboxConnectCard.hBoxLayout.addWidget(self.xboxConnectBtn, 0, Qt.AlignmentFlag.AlignRight)
        self.xboxConnectCard.hBoxLayout.addSpacing(16)

        # === PID 參數 ===
        self.pidGroup = SettingCardGroup(t("aim_speed_pid"), self.scrollWidget)

        # X/Y 軸切換器
        self.pidAxisPivot = SegmentedWidget()
        self.pidAxisPivot.addItem(routeKey='x', text=t("horizontal_x"))
        self.pidAxisPivot.addItem(routeKey='y', text=t("vertical_y"))
        self.pidAxisPivot.setCurrentItem('x')
        self.pidAxisPivot.currentItemChanged.connect(self._onPidAxisChanged)

        # 堆疊容器
        self.pidStackedWidget = QStackedWidget()

        # P - 反應速度 X - 使用 SliderLabelCard
        self.pidPxCard = SliderLabelCard(
            FluentIcon.SPEED_HIGH,
            t("reaction_speed_p"),
            0, 100,
            format_func=lambda v: f"{v/100:.2f}",
            parent=self.pidGroup
        )

        # I - 誤差修正 X - 使用 SliderLabelCard
        self.pidIxCard = SliderLabelCard(
            FluentIcon.SYNC,
            t("error_correction_i"),
            0, 100,
            format_func=lambda v: f"{v/100:.2f}",
            parent=self.pidGroup
        )

        # D - 穩定控制 X - 使用 SliderLabelCard
        self.pidDxCard = SliderLabelCard(
            FluentIcon.ALIGNMENT,
            t("stability_suppression_d"),
            0, 100,
            format_func=lambda v: f"{v/100:.2f}",
            parent=self.pidGroup
        )

        # P - 反應速度 Y - 使用 SliderLabelCard
        self.pidPyCard = SliderLabelCard(
            FluentIcon.SPEED_HIGH,
            t("reaction_speed_p"),
            0, 100,
            format_func=lambda v: f"{v/100:.2f}",
            parent=self.pidGroup
        )

        # I - 誤差修正 Y - 使用 SliderLabelCard
        self.pidIyCard = SliderLabelCard(
            FluentIcon.SYNC,
            t("error_correction_i"),
            0, 100,
            format_func=lambda v: f"{v/100:.2f}",
            parent=self.pidGroup
        )

        # D - 穩定控制 Y - 使用 SliderLabelCard
        self.pidDyCard = SliderLabelCard(
            FluentIcon.ALIGNMENT,
            t("stability_suppression_d"),
            0, 100,
            format_func=lambda v: f"{v/100:.2f}",
            parent=self.pidGroup
        )

        # Y軸壓槍速度歸零啟用開關
        self.pidYReduceEnableCard = SwitchSettingCard(
            FluentIcon.CARE_UP_SOLID,
            t("aim_y_reduce_enable"),
            "",
            parent=self.pidGroup
        )

        # Y軸壓槍速度歸零延遲
        self.pidYReduceDelayCard = SliderLabelCard(
            FluentIcon.STOP_WATCH,
            t("aim_y_reduce_delay"),
            0, 500,
            format_func=lambda v: f"{v/100:.2f} s",
            parent=self.pidGroup
        )

        # === 貝塞爾曲線 ===
        self.bezierGroup = SettingCardGroup(t("bezier_curve"), self.scrollWidget)

        # 啟用開關
        self.bezierEnableCard = SwitchSettingCard(
            FluentIcon.CALORIES,
            t("bezier_curve_enable"),
            "",
            parent=self.bezierGroup
        )

        # 曲線彎曲程度 - 使用 SliderLabelCard
        self.bezierStrengthCard = SliderLabelCard(
            FluentIcon.MIX_VOLUMES,
            t("bezier_curve_strength"),
            0, 100,
            format_func=lambda v: f"{v}%",
            parent=self.bezierGroup
        )

        # 曲線分段數 - 使用 SliderLabelCard
        self.bezierStepsCard = SliderLabelCard(
            FluentIcon.MORE,
            t("bezier_curve_steps"),
            2, 20,
            format_func=lambda v: str(v),
            parent=self.bezierGroup
        )

        # === 智慧追蹤 ===
        self.trackerGroup = SettingCardGroup(t("tracker_prediction"), self.scrollWidget)

        # 啟用開關
        self.trackerEnableCard = SwitchSettingCard(
            FluentIcon.RINGER,
            t("tracker_enable"),
            "",
            parent=self.trackerGroup
        )

        # 預判時間 - 使用 SliderLabelCard
        self.trackerTimeCard = SliderLabelCard(
            FluentIcon.HISTORY,
            t("tracker_prediction_time"),
            0, 100,
            format_func=lambda v: f"{v} ms",
            label_width=50,
            parent=self.trackerGroup
        )

        # 速度平滑係數 - 使用 SliderLabelCard
        self.trackerSmoothCard = SliderLabelCard(
            FluentIcon.SPEED_MEDIUM,
            t("tracker_smoothing_factor"),
            0, 100,
            format_func=lambda v: f"{v}%",
            parent=self.trackerGroup
        )

        # 靜止判定速度 - 使用 SliderLabelCard
        self.trackerThresholdCard = SliderLabelCard(
            FluentIcon.STOP_WATCH,
            t("tracker_stop_threshold"),
            0, 100,
            format_func=lambda v: f"{v} px/s",
            label_width=55,
            parent=self.trackerGroup
        )

        # 顯示預判視覺化
        self.trackerShowCard = SwitchSettingCard(
            FluentIcon.VIEW,
            t("tracker_show_prediction"),
            "",
            parent=self.trackerGroup
        )

    def _initLayout(self):
        """排版所有控制項"""
        # 模型設定
        self.modelGroup.addSettingCard(self.modelCard)
        self.modelGroup.addSettingCard(self.inferenceBackendCard)
        self.modelGroup.addSettingCard(self.openModelFolderCard)
        self.addContent(self.modelGroup)

        # FOV 與偵測範圍
        self.fovGroup.addSettingCard(self.fovCard)
        self.fovGroup.addSettingCard(self.fovFollowCard)
        self.fovGroup.addSettingCard(self.detectRangeCard)
        self.fovGroup.addSettingCard(self.screenshotMethodCard)
        self.addContent(self.fovGroup)

        # 通用參數
        self.generalGroup.addSettingCard(self.detectIntervalCard)
        self.generalGroup.addSettingCard(self.screenshotIntervalCard)
        self.generalGroup.addSettingCard(self.confidenceCard)
        self.generalGroup.addSettingCard(self.aimPartCard)
        self.generalGroup.addSettingCard(self.mouseMoveCard)
        self.generalGroup.addSettingCard(self.uvcDeviceCard)
        self.generalGroup.addSettingCard(self.uvcResolutionCard)
        self.generalGroup.addSettingCard(self.uvcRefreshResolutionCard)
        self.generalGroup.addSettingCard(self.uvcWidthCard)
        self.generalGroup.addSettingCard(self.uvcHeightCard)
        self.generalGroup.addSettingCard(self.uvcFpsCard)
        self.generalGroup.addSettingCard(self.uvcCaptureMethodCard)
        self.generalGroup.addSettingCard(self.uvcPreviewCard)
        self.generalGroup.addSettingCard(self.uvcPreviewScaleCard)
        self.generalGroup.addSettingCard(self.alwaysAimCard)
        self.generalGroup.addSettingCard(self.keepDetectingCard)
        self.generalGroup.addSettingCard(self.idleDetectEnableCard)
        self.generalGroup.addSettingCard(self.idleDetectIntervalCard)
        self.generalGroup.addSettingCard(self.singleTargetCard)
        self.addContent(self.generalGroup)

        # Arduino 設定（在滑鼠移動方式下方）
        self.arduinoGroup.addSettingCard(self.comPortCard)
        self.arduinoGroup.addSettingCard(self.connectionCard)
        self.arduinoGroup.addSettingCard(self.arduinoConnectCard)
        self.arduinoGroup.addSettingCard(self.guideCard)
        self.arduinoGroup.addSettingCard(self.spoofCard)
        self.arduinoGroup.addSettingCard(self.verifySpoofCard)
        self.arduinoGroup.addSettingCard(self.testHeartCard)
        self.addContent(self.arduinoGroup)
        # 預設隱藏 Arduino 設定
        self.arduinoGroup.setVisible(False)

        # MAKCU 設定（在 Arduino 設定下方）
        self.makcuGroup.addSettingCard(self.makcuComPortCard)
        self.makcuGroup.addSettingCard(self.makcuConnectionCard)
        self.makcuGroup.addSettingCard(self.makcuConnectCard)
        self.addContent(self.makcuGroup)
        # 預設隱藏 MAKCU 設定
        self.makcuGroup.setVisible(False)

        # Xbox 360 設定（在 MAKCU 設定下方）
        self.xboxGroup.addSettingCard(self.xboxSensitivityCard)
        self.xboxGroup.addSettingCard(self.xboxDeadzoneCard)
        self.xboxGroup.addSettingCard(self.xboxConnectionCard)
        self.xboxGroup.addSettingCard(self.xboxConnectCard)
        self.addContent(self.xboxGroup)
        # 預設隱藏 Xbox 設定
        self.xboxGroup.setVisible(False)

        # === 進階設定（摺疊區域）===

        # PID 參數 - 使用切換式佈局
        # 切換器容器
        pivotWidget = QWidget()
        pivotLayout = QHBoxLayout(pivotWidget)
        pivotLayout.setContentsMargins(16, 8, 16, 8)
        pivotLayout.addWidget(self.pidAxisPivot)
        pivotLayout.addStretch(1)

        # X 軸頁面
        self.pidXPage = QWidget()
        xPageLayout = QVBoxLayout(self.pidXPage)
        xPageLayout.setContentsMargins(0, 0, 0, 0)
        xPageLayout.setSpacing(0)
        xPageLayout.addWidget(self.pidPxCard)
        xPageLayout.addWidget(self.pidIxCard)
        xPageLayout.addWidget(self.pidDxCard)

        # Y 軸頁面
        self.pidYPage = QWidget()
        yPageLayout = QVBoxLayout(self.pidYPage)
        yPageLayout.setContentsMargins(0, 0, 0, 0)
        yPageLayout.setSpacing(0)
        yPageLayout.addWidget(self.pidPyCard)
        yPageLayout.addWidget(self.pidIyCard)
        yPageLayout.addWidget(self.pidDyCard)
        yPageLayout.addWidget(self.pidYReduceEnableCard)
        yPageLayout.addWidget(self.pidYReduceDelayCard)

        # 加入堆疊
        self.pidStackedWidget.addWidget(self.pidXPage)
        self.pidStackedWidget.addWidget(self.pidYPage)

        # 組合到 pidGroup
        self.pidGroup.vBoxLayout.addWidget(pivotWidget)
        self.pidGroup.vBoxLayout.addWidget(self.pidStackedWidget)

        # 貝塞爾曲線
        self.bezierGroup.addSettingCard(self.bezierEnableCard)
        self.bezierGroup.addSettingCard(self.bezierStrengthCard)
        self.bezierGroup.addSettingCard(self.bezierStepsCard)

        # 智慧追蹤
        self.trackerGroup.addSettingCard(self.trackerEnableCard)
        self.trackerGroup.addSettingCard(self.trackerTimeCard)
        self.trackerGroup.addSettingCard(self.trackerSmoothCard)
        self.trackerGroup.addSettingCard(self.trackerThresholdCard)
        self.trackerGroup.addSettingCard(self.trackerShowCard)

        # 將進階設定添加到摺疊區域
        self.addContent(self.pidGroup)
        self.addContent(self.bezierGroup)
        self.addContent(self.trackerGroup)

        self.scrollLayout.addStretch(1)

    def _connectSignals(self):
        """連接信號"""
        # 模型
        self.modelCombo.currentTextChanged.connect(self._onModelChanged)
        self.inferenceBackendCombo.currentTextChanged.connect(self._onInferenceBackendChanged)
        self.openModelFolderBtn.clicked.connect(self._openModelFolder)

        # FOV 與偵測範圍 - 使用新組件的 valueChanged 信號
        self.fovCard.valueChanged.connect(self._onFovChanged)
        self.fovFollowCard.checkedChanged.connect(self._onFovFollowChanged)
        self.detectRangeCard.valueChanged.connect(self._onDetectRangeChanged)

        # 通用參數 - 使用新組件的 valueChanged 信號
        self.detectIntervalCard.valueChanged.connect(self._onDetectIntervalChanged)
        self.screenshotIntervalCard.valueChanged.connect(self._onScreenshotIntervalChanged)
        self.confidenceCard.valueChanged.connect(self._onConfidenceChanged)
        self.aimPartCombo.currentIndexChanged.connect(self._onAimPartChanged)
        self.mouseMoveCombo.currentTextChanged.connect(self._onMouseMoveChanged)
        self.screenshotMethodCombo.currentTextChanged.connect(self._onScreenshotMethodChanged)
        self.uvcDeviceCard.valueChanged.connect(self._onUvcDeviceChanged)
        self.uvcResolutionCombo.currentTextChanged.connect(self._onUvcResolutionChanged)
        self.uvcRefreshResolutionBtn.clicked.connect(self._refreshUvcResolutions)
        self.uvcFpsCard.valueChanged.connect(self._onUvcFpsChanged)
        self.uvcCaptureMethodCombo.currentTextChanged.connect(self._onUvcCaptureMethodChanged)
        self.uvcPreviewCard.checkedChanged.connect(self._onUvcPreviewChanged)
        self.uvcPreviewScaleCombo.currentTextChanged.connect(self._onUvcPreviewScaleModeChanged)
        self.alwaysAimCard.checkedChanged.connect(self._onAlwaysAimChanged)
        self.keepDetectingCard.checkedChanged.connect(self._onKeepDetectingChanged)
        self.idleDetectEnableCard.checkedChanged.connect(self._onIdleDetectEnableChanged)
        self.idleDetectIntervalCard.valueChanged.connect(self._onIdleDetectIntervalChanged)
        self.singleTargetCard.checkedChanged.connect(self._onSingleTargetChanged)

        # Arduino 相關信號
        self.comRefreshBtn.clicked.connect(self._refreshComPorts)
        self.comPortCombo.currentTextChanged.connect(self._onComPortChanged)
        self.arduinoConnectBtn.clicked.connect(self._onArduinoConnectToggle)
        self.guideBtn.clicked.connect(self._onOpenGuide)
        self.spoofBtn.clicked.connect(self._onSpoofDevice)
        self.verifySpoofBtn.clicked.connect(self._onVerifySpoof)
        self.testHeartBtn.clicked.connect(self._onTestHeart)

        # MAKCU 相關信號
        self.makcuComRefreshBtn.clicked.connect(self._refreshMakcuComPorts)
        self.makcuComPortCombo.currentTextChanged.connect(self._onMakcuComPortChanged)
        self.makcuConnectBtn.clicked.connect(self._onMakcuConnectToggle)

        # Xbox 相關信號
        self.xboxSensitivityCard.valueChanged.connect(self._onXboxSensitivityChanged)
        self.xboxDeadzoneCard.valueChanged.connect(self._onXboxDeadzoneChanged)
        self.xboxConnectBtn.clicked.connect(self._onXboxConnectToggle)

        # PID - 使用新組件的 valueChanged 信號
        self.pidPxCard.valueChanged.connect(lambda v: self._onPidChanged('pid_kp_x', v))
        self.pidIxCard.valueChanged.connect(lambda v: self._onPidChanged('pid_ki_x', v))
        self.pidDxCard.valueChanged.connect(lambda v: self._onPidChanged('pid_kd_x', v))
        self.pidPyCard.valueChanged.connect(lambda v: self._onPidChanged('pid_kp_y', v))
        self.pidIyCard.valueChanged.connect(lambda v: self._onPidChanged('pid_ki_y', v))
        self.pidDyCard.valueChanged.connect(lambda v: self._onPidChanged('pid_kd_y', v))
        self.pidYReduceEnableCard.checkedChanged.connect(lambda checked: self._onPidChanged('aim_y_reduce_enabled', checked, is_bool=True))
        self.pidYReduceDelayCard.valueChanged.connect(lambda v: self._onPidChanged('aim_y_reduce_delay', v))

        # 貝塞爾 - 使用新組件的 valueChanged 信號
        self.bezierEnableCard.checkedChanged.connect(self._onBezierEnableChanged)
        self.bezierStrengthCard.valueChanged.connect(self._onBezierStrengthChanged)
        self.bezierStepsCard.valueChanged.connect(self._onBezierStepsChanged)

        # 追蹤 - 使用新組件的 valueChanged 信號
        self.trackerEnableCard.checkedChanged.connect(self._onTrackerEnableChanged)
        self.trackerTimeCard.valueChanged.connect(self._onTrackerTimeChanged)
        self.trackerSmoothCard.valueChanged.connect(self._onTrackerSmoothChanged)
        self.trackerThresholdCard.valueChanged.connect(self._onTrackerThresholdChanged)
        self.trackerShowCard.checkedChanged.connect(self._onTrackerShowChanged)

    def _loadFromConfig(self):
        """從 Config 載入值"""
        if not self._config:
            return

        # 刷新模型列表並選中當前模型（暫時阻斷信號避免覆蓋設定）
        self.modelCombo.blockSignals(True)
        self._refreshModelList()
        model_name = os.path.basename(self._config.model_path)

        # 大小寫不敏感比對
        idx = -1
        for i in range(self.modelCombo.count()):
            if self.modelCombo.itemText(i).lower() == model_name.lower():
                idx = i
                break

        if idx >= 0:
            self.modelCombo.setCurrentIndex(idx)
        self.modelCombo.blockSignals(False)

        backend_map = {
            "auto": "Auto",
            "cuda": "CUDA",
            "directml": "DirectML",
            "cpu": "CPU",
        }
        self.inferenceBackendCombo.blockSignals(True)
        backend_text = backend_map.get(getattr(self._config, "inference_backend", "auto").lower(), "Auto")
        self.inferenceBackendCombo.setCurrentText(backend_text)
        self.inferenceBackendCombo.blockSignals(False)
        self._updateInferenceBackendSubtitle()

        # FOV 與偵測範圍 - 使用新組件的 setValue
        self.fovCard.setValue(self._config.fov_size)
        self.fovFollowCard.setChecked(self._config.fov_follow_mouse)
        self.detectRangeCard.setValue(self._config.detect_range_size)

        # 通用參數 - 使用新組件的 setValue
        interval_ms = int(self._config.detect_interval * 1000)
        self.detectIntervalCard.setValue(interval_ms)
        screenshot_interval_ms = int(getattr(self._config, 'screenshot_interval', self._config.detect_interval) * 1000)
        self.screenshotIntervalCard.setValue(screenshot_interval_ms)
        confidence_pct = int(self._config.min_confidence * 100)
        self.confidenceCard.setValue(confidence_pct)

        aim_parts = ["head", "body", "both"]
        if self._config.aim_part in aim_parts:
            self.aimPartCombo.setCurrentIndex(aim_parts.index(self._config.aim_part))

        mouse_methods = ["ddxoft", "mouse_event", "sendinput", "arduino", "makcu", "xbox"]
        if self._config.mouse_move_method in mouse_methods:
            self.mouseMoveCombo.setCurrentIndex(mouse_methods.index(self._config.mouse_move_method))

        screenshot_methods = ["mss", "dxcam", "uvc"]
        screenshot_method = getattr(self._config, 'screenshot_method', 'mss')
        if screenshot_method in screenshot_methods:
            self.screenshotMethodCombo.setCurrentIndex(screenshot_methods.index(screenshot_method))
        self.uvcDeviceCard.setValue(int(getattr(self._config, 'uvc_device_index', 0)))
        self.uvcCaptureMethodCombo.setCurrentText(str(getattr(self._config, 'uvc_capture_method', 'dshow')))
        self._refreshUvcResolutions()
        resolution_text = str(getattr(self._config, 'uvc_resolution', f"{getattr(self._config, 'uvc_width', self._config.width)}x{getattr(self._config, 'uvc_height', self._config.height)}"))
        idx = self.uvcResolutionCombo.findText(resolution_text)
        if idx < 0:
            self.uvcResolutionCombo.addItem(resolution_text)
            idx = self.uvcResolutionCombo.findText(resolution_text)
        if idx >= 0:
            self.uvcResolutionCombo.setCurrentIndex(idx)
        self.uvcFpsCard.setValue(int(getattr(self._config, 'uvc_fps', 60)))
        self.uvcPreviewCard.setChecked(bool(getattr(self._config, 'uvc_show_window', True)))
        self.uvcPreviewScaleCombo.setCurrentText(str(getattr(self._config, 'uvc_preview_scale_mode', 'scale_to_fit')))
        self._updateUvcControlsVisibility(screenshot_method)
        self.alwaysAimCard.setChecked(getattr(self._config, 'always_aim', False))
        self.keepDetectingCard.setChecked(getattr(self._config, 'keep_detecting', False))
        self.idleDetectEnableCard.setChecked(getattr(self._config, 'idle_detect_enabled', True))
        idle_ms = int(getattr(self._config, 'idle_detect_interval', 0.05) * 1000)
        self.idleDetectIntervalCard.setValue(max(5, min(500, idle_ms)))
        self.singleTargetCard.setChecked(getattr(self._config, 'single_target_mode', False))

        # 根據當前選擇的移動方式顯示/隱藏 Arduino 和 Xbox 設定
        self._updateMethodGroupVisibility(self._config.mouse_move_method)

        # COM 埠
        if self._config.arduino_com_port:
            idx = self.comPortCombo.findText(self._config.arduino_com_port)
            if idx >= 0:
                self.comPortCombo.setCurrentIndex(idx)

        # MAKCU COM 埠
        if getattr(self._config, 'makcu_com_port', ''):
            idx = self.makcuComPortCombo.findText(self._config.makcu_com_port)
            if idx >= 0:
                self.makcuComPortCombo.setCurrentIndex(idx)

        # PID - 使用新組件的 setValue
        self.pidPxCard.setValue(int(self._config.pid_kp_x * 100))
        self.pidIxCard.setValue(int(self._config.pid_ki_x * 100))
        self.pidDxCard.setValue(int(self._config.pid_kd_x * 100))
        self.pidPyCard.setValue(int(self._config.pid_kp_y * 100))
        self.pidIyCard.setValue(int(self._config.pid_ki_y * 100))
        self.pidDyCard.setValue(int(self._config.pid_kd_y * 100))
        self.pidYReduceEnableCard.setChecked(getattr(self._config, 'aim_y_reduce_enabled', False))
        self.pidYReduceDelayCard.setValue(int(getattr(self._config, 'aim_y_reduce_delay', 0.6) * 100))

        # 貝塞爾 - 使用新組件的 setValue
        self.bezierEnableCard.setChecked(self._config.bezier_curve_enabled)
        self.bezierStrengthCard.setValue(int(self._config.bezier_curve_strength * 100))
        self.bezierStepsCard.setValue(self._config.bezier_curve_steps)

        # 追蹤 - 使用新組件的 setValue
        self.trackerEnableCard.setChecked(self._config.tracker_enabled)
        self.trackerTimeCard.setValue(int(self._config.tracker_prediction_time * 1000))
        self.trackerSmoothCard.setValue(int(self._config.tracker_smoothing_factor * 100))
        self.trackerThresholdCard.setValue(int(self._config.tracker_stop_threshold))
        self.trackerShowCard.setChecked(self._config.tracker_show_prediction)

        # Xbox 設定
        self.xboxSensitivityCard.setValue(int(getattr(self._config, 'xbox_sensitivity', 1.0) * 100))
        self.xboxDeadzoneCard.setValue(int(getattr(self._config, 'xbox_deadzone', 0.05) * 100))
        self._updateXboxConnectionStatus()

    def _refreshModelList(self):
        """刷新模型列表"""
        self.modelCombo.clear()
        # aim_page.py 位於 src/gui/fluent_app/pages/，向上 4 層到項目根目錄
        src_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        project_root = os.path.dirname(src_dir)
        model_dir = os.path.join(project_root, "Model")
        if os.path.exists(model_dir):
            models = glob.glob(os.path.join(model_dir, "*.onnx"))
            for m in models:
                self.modelCombo.addItem(os.path.basename(m))

    def _openModelFolder(self):
        """開啟模型資料夾"""
        src_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        project_root = os.path.dirname(src_dir)
        model_dir = os.path.join(project_root, "Model")
        if os.path.exists(model_dir):
            os.startfile(model_dir)

    def _refreshComPorts(self):
        """刷新 COM 埠列表"""
        self.comPortCombo.clear()
        self.comPortCombo.addItem(t("no_com_port"))

        try:
            import serial.tools.list_ports
            ports = serial.tools.list_ports.comports()
            for port in ports:
                self.comPortCombo.addItem(port.device)
        except ImportError:
            pass

    def _updateArduinoVisibility(self, method):
        """根據滑鼠移動方式更新 Arduino 設定的可見性"""
        is_arduino = (method == "arduino")
        self.arduinoGroup.setVisible(is_arduino)

    def _updateMethodGroupVisibility(self, method):
        """根據滑鼠移動方式更新各裝置設定組的可見性"""
        self.arduinoGroup.setVisible(method == "arduino")
        self.makcuGroup.setVisible(method == "makcu")
        self.xboxGroup.setVisible(method == "xbox")

    # === 回調函數 ===
    def _onModelChanged(self, text):
        if self._config and text:
            self._config.model_path = os.path.join("Model", text)

    def _onInferenceBackendChanged(self, text):
        if not self._config:
            return

        backend_map = {
            "Auto": "auto",
            "CUDA": "cuda",
            "DirectML": "directml",
            "CPU": "cpu",
        }
        selected_backend = backend_map.get(text, "auto")
        if getattr(self._config, "inference_backend", "auto") != selected_backend:
            self._config.inference_backend = selected_backend
        self._updateInferenceBackendSubtitle()

    def _updateInferenceBackendSubtitle(self):
        if not hasattr(self, "inferenceBackendCard"):
            return
        provider = getattr(self._config, "current_provider", "Unknown") if self._config else "Unknown"
        self.inferenceBackendCard.contentLabel.setText(
            f"{t('inference_backend_desc')} ({t('inference_backend_current')}: {provider})"
        )

    def _onFovChanged(self, value):
        """FOV 改變"""
        if self._config:
            self._config.fov_size = value

    def _onFovFollowChanged(self, checked):
        if self._config:
            self._config.fov_follow_mouse = checked

    def _onDetectRangeChanged(self, value):
        """偵測範圍改變"""
        if self._config:
            self._config.detect_range_size = value

    def _onDetectIntervalChanged(self, value):
        """偵測間隔改變"""
        if self._config:
            self._config.detect_interval = value / 1000.0

    def _onScreenshotIntervalChanged(self, value):
        """截圖間隔改變"""
        if self._config:
            self._config.screenshot_interval = value / 1000.0

    def _onConfidenceChanged(self, value):
        """信心值改變"""
        if self._config:
            self._config.min_confidence = value / 100.0

    def _onAimPartChanged(self, index):
        if self._config:
            parts = ["head", "body", "both"]
            self._config.aim_part = parts[index]

    def _onMouseMoveChanged(self, text):
        if self._config:
            self._config.mouse_move_method = text
            if text == "ddxoft":
                try:
                    from win_utils import ensure_ddxoft_ready
                    ensure_ddxoft_ready()
                except ImportError:
                    pass
        # 更新設定組的可見性
        self._updateMethodGroupVisibility(text)

    def _onScreenshotMethodChanged(self, text):
        if self._config:
            self._config.screenshot_method = text
        self._updateUvcControlsVisibility(text)
        main_window = self.window()
        if main_window and hasattr(main_window, 'updateVisualsVisibilityForScreenshotMethod'):
            main_window.updateVisualsVisibilityForScreenshotMethod(text)

    def _onUvcDeviceChanged(self, value):
        if self._config:
            self._config.uvc_device_index = int(value)
        self._refreshUvcResolutions()

    def _onUvcResolutionChanged(self, value):
        if self._config:
            text = str(value).strip().lower()
            if 'x' not in text:
                return
            width_str, height_str = text.split('x', 1)
            try:
                self._config.uvc_width = int(width_str)
                self._config.uvc_height = int(height_str)
                self._config.uvc_resolution = f"{self._config.uvc_width}x{self._config.uvc_height}"
            except ValueError:
                return

    def _onUvcFpsChanged(self, value):
        if self._config:
            self._config.uvc_fps = int(value)

    def _onUvcCaptureMethodChanged(self, text):
        if self._config:
            self._config.uvc_capture_method = str(text)
        self._refreshUvcResolutions()

    def _onUvcPreviewChanged(self, checked):
        if self._config:
            self._config.uvc_show_window = bool(checked)

    def _onUvcPreviewScaleModeChanged(self, text):
        if self._config:
            self._config.uvc_preview_scale_mode = str(text)

    def _refreshUvcResolutions(self):
        if not self._config:
            return
        try:
            from core.screen_capture import list_supported_uvc_resolutions
            resolutions = list_supported_uvc_resolutions(
                int(getattr(self._config, 'uvc_device_index', 0)),
                str(getattr(self._config, 'uvc_capture_method', 'dshow')),
            )
        except Exception:
            resolutions = []

        current_text = self.uvcResolutionCombo.currentText().strip()
        self.uvcResolutionCombo.blockSignals(True)
        self.uvcResolutionCombo.clear()
        if resolutions:
            for width, height in resolutions:
                self.uvcResolutionCombo.addItem(f"{width}x{height}")
        else:
            fallback = f"{int(getattr(self._config, 'uvc_width', 1920))}x{int(getattr(self._config, 'uvc_height', 1080))}"
            self.uvcResolutionCombo.addItem(fallback)
        if current_text:
            idx = self.uvcResolutionCombo.findText(current_text)
            if idx >= 0:
                self.uvcResolutionCombo.setCurrentIndex(idx)
        self.uvcResolutionCombo.blockSignals(False)

    def _updateUvcControlsVisibility(self, screenshot_method):
        is_uvc = (screenshot_method == "uvc")
        self.uvcDeviceCard.setVisible(is_uvc)
        self.uvcResolutionCard.setVisible(is_uvc)
        self.uvcRefreshResolutionCard.setVisible(is_uvc)
        self.uvcFpsCard.setVisible(is_uvc)
        self.uvcCaptureMethodCard.setVisible(is_uvc)
        self.uvcPreviewCard.setVisible(is_uvc)
        self.uvcPreviewScaleCard.setVisible(is_uvc)

    def _onAlwaysAimChanged(self, checked):
        if self._config:
            self._config.always_aim = checked
            # 啟用持續自動瞄準時，自動關閉 idle detect
            if checked:
                self._config.idle_detect_enabled = False
                self.idleDetectEnableCard.setChecked(False)

    def _onKeepDetectingChanged(self, checked):
        if self._config:
            self._config.keep_detecting = checked

    def _onIdleDetectEnableChanged(self, checked):
        if self._config:
            self._config.idle_detect_enabled = checked

    def _onIdleDetectIntervalChanged(self, value):
        if self._config:
            self._config.idle_detect_interval = value / 1000.0

    def _onSingleTargetChanged(self, checked):
        if self._config:
            self._config.single_target_mode = checked

    def _onComPortChanged(self, text):
        if self._config and text != t("no_com_port"):
            self._config.arduino_com_port = text

    def _onMakcuComPortChanged(self, text):
        if self._config and text != t("no_com_port"):
            self._config.makcu_com_port = text

    def _onOpenGuide(self):
        """開啟 Arduino 使用教學"""
        guide_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
            "Arduino_User_Guide.html"
        )
        if os.path.exists(guide_path):
            QDesktopServices.openUrl(QUrl.fromLocalFile(guide_path))

    def _onSpoofDevice(self):
        """一鍵硬體偽裝"""
        reply = QMessageBox.question(
            self, t("spoof_confirm_title"),
            t("spoof_confirm_msg").replace("\\n", "\n"),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            try:
                from win_utils.arduino_spoofer import spoof_arduino_board
                success, boards_path = spoof_arduino_board()
                if success:
                    QMessageBox.information(
                        self, t("spoof_success_title"),
                        t("spoof_success_msg").replace("\\n", "\n")
                    )
                else:
                    QMessageBox.warning(
                        self, t("spoof_error_title"),
                        f"Spoof operation returned unsuccessful.\nFile: {boards_path}"
                    )
            except FileNotFoundError as e:
                QMessageBox.warning(self, t("spoof_error_title"), str(e))
            except Exception as e:
                QMessageBox.critical(self, t("spoof_error_title"), f"Error: {e}")

    def _onVerifySpoof(self):
        """驗證偽裝"""
        try:
            from win_utils.arduino_spoofer import verify_spoof
            specific_port = None
            if self._config and self._config.arduino_com_port:
                specific_port = self._config.arduino_com_port
            is_spoofed, message = verify_spoof(specific_port)
            if is_spoofed:
                QMessageBox.information(
                    self, t("verify_success_title"), message
                )
            else:
                QMessageBox.warning(
                    self, t("verify_fail_title"), message
                )
        except Exception as e:
            QMessageBox.critical(
                self, t("verify_fail_title"), f"Error: {e}"
            )

    def _onTestHeart(self):
        """測試愛心移動"""
        reply = QMessageBox.question(
            self, t("test_heart_confirm_title"),
            t("test_heart_confirm_msg").replace("\\n", "\n"),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            import time
            from win_utils.arduino_mouse import arduino_mouse

            if not arduino_mouse.is_connected():
                # 嘗試使用設定中的 COM port 連線
                com_port = self._config.arduino_com_port if self._config else ""
                if not com_port:
                    QMessageBox.warning(
                        self, t("test_heart_confirm_title"),
                        "Arduino not connected. Please set COM port first."
                    )
                    return
                if not arduino_mouse.connect(com_port):
                    QMessageBox.warning(
                        self, t("test_heart_confirm_title"),
                        f"Failed to connect to {com_port}."
                    )
                    return

            def _draw_heart():
                """在背景執行緒中繪製愛心圖案"""
                # 心形參數方程式: x = 16sin³(t), y = 13cos(t) - 5cos(2t) - 2cos(3t) - cos(4t)
                num_steps = 120
                scale = 3.0
                points = []
                for i in range(num_steps + 1):
                    angle = 2 * math.pi * i / num_steps
                    x = 16 * (math.sin(angle) ** 3)
                    y = -(13 * math.cos(angle) - 5 * math.cos(2 * angle)
                           - 2 * math.cos(3 * angle) - math.cos(4 * angle))
                    points.append((x * scale, y * scale))

                # 計算相鄰點之間的增量並發送
                for i in range(1, len(points)):
                    dx = int(round(points[i][0] - points[i - 1][0]))
                    dy = int(round(points[i][1] - points[i - 1][1]))
                    if dx != 0 or dy != 0:
                        arduino_mouse.move(dx, dy)
                    time.sleep(0.015)

            # 在背景執行緒中執行，避免阻塞 GUI
            thread = threading.Thread(target=_draw_heart, daemon=True)
            thread.start()

    def _onPidAxisChanged(self, routeKey: str):
        """切換 PID X/Y 軸頁面"""
        if routeKey == 'x':
            self.pidStackedWidget.setCurrentIndex(0)
        else:
            self.pidStackedWidget.setCurrentIndex(1)

    def _onPidChanged(self, attr, value, is_bool=False):
        if self._config:
            if is_bool:
                setattr(self._config, attr, value)
            else:
                float_val = value / 100.0
                setattr(self._config, attr, float_val)

    def _onBezierEnableChanged(self, checked):
        if self._config:
            self._config.bezier_curve_enabled = checked

    def _onBezierStrengthChanged(self, value):
        if self._config:
            self._config.bezier_curve_strength = value / 100.0

    def _onBezierStepsChanged(self, value):
        if self._config:
            self._config.bezier_curve_steps = value

    def _onTrackerEnableChanged(self, checked):
        if self._config:
            self._config.tracker_enabled = checked

    def _onTrackerTimeChanged(self, value):
        if self._config:
            self._config.tracker_prediction_time = value / 1000.0

    def _onTrackerSmoothChanged(self, value):
        if self._config:
            self._config.tracker_smoothing_factor = value / 100.0

    def _onTrackerThresholdChanged(self, value):
        if self._config:
            self._config.tracker_stop_threshold = float(value)

    def _onTrackerShowChanged(self, checked):
        if self._config:
            self._config.tracker_show_prediction = checked

    # === Arduino 連線回調函數 ===
    # === Arduino 連線回調函數 ===
    def _onArduinoConnectToggle(self):
        """Arduino 連線/斷線切換"""
        try:
            from win_utils import is_arduino_connected, connect_arduino, disconnect_arduino
            if is_arduino_connected():
                disconnect_arduino()
            else:
                com_port = self.comPortCombo.currentText()
                if not com_port or com_port == t("no_com_port"):
                    QMessageBox.warning(
                        self, t("config_error"),
                        t("no_com_port")
                    )
                    return
                success = connect_arduino(com_port)
                if not success:
                    QMessageBox.warning(
                        self, t("config_error"),
                        f"Arduino {t('disconnected')}: {com_port}"
                    )
            self._updateArduinoConnectionStatus()
        except ImportError:
            QMessageBox.warning(
                self, t("config_error"),
                "pyserial not installed.\npip install pyserial"
            )

    def _updateArduinoConnectionStatus(self):
        """更新 Arduino 連線狀態顯示"""
        try:
            from win_utils import is_arduino_connected
            if is_arduino_connected():
                self._isArduinoConnected = True
                self.connectionLabel.setText(t("connected"))
                self.connectionLabel.setStyleSheet("color: #2ecc71; font-weight: bold;")
                self.arduinoConnectBtn.setText(t("arduino_disconnect"))
            else:
                self._isArduinoConnected = False
                self.connectionLabel.setText(t("disconnected"))
                self.connectionLabel.setStyleSheet("color: #e74c3c; font-weight: bold;")
                self.arduinoConnectBtn.setText(t("arduino_connect"))
        except ImportError:
            self.connectionLabel.setText("pyserial N/A")
            self.connectionLabel.setStyleSheet("color: #e74c3c; font-weight: bold;")

    # === MAKCU 連線回調函數 ===
    def _refreshMakcuComPorts(self):
        """刷新 MAKCU COM 埠列表"""
        self.makcuComPortCombo.clear()
        self.makcuComPortCombo.addItem(t("no_com_port"))
        try:
            import serial.tools.list_ports
            ports = serial.tools.list_ports.comports()
            for port in ports:
                self.makcuComPortCombo.addItem(port.device)
        except ImportError:
            pass

    def _onMakcuConnectToggle(self):
        """MAKCU 連線/斷線切換"""
        try:
            from win_utils import is_makcu_connected, connect_makcu, disconnect_makcu
            if is_makcu_connected():
                disconnect_makcu()
            else:
                com_port = self.makcuComPortCombo.currentText()
                if not com_port or com_port == t("no_com_port"):
                    QMessageBox.warning(
                        self, t("config_error"),
                        t("no_com_port")
                    )
                    return
                success = connect_makcu(com_port)
                if not success:
                    QMessageBox.warning(
                        self, t("config_error"),
                        f"MAKCU {t('disconnected')}: {com_port}"
                    )
            self._updateMakcuConnectionStatus()
        except ImportError:
            QMessageBox.warning(
                self, t("config_error"),
                "pyserial not installed.\npip install pyserial"
            )

    def _updateMakcuConnectionStatus(self):
        """更新 MAKCU 連線狀態顯示"""
        try:
            from win_utils import is_makcu_connected
            if is_makcu_connected():
                self._isMakcuConnected = True
                self.makcuConnectionLabel.setText(t("connected"))
                self.makcuConnectionLabel.setStyleSheet("color: #2ecc71; font-weight: bold;")
                self.makcuConnectBtn.setText(t("makcu_disconnect"))
            else:
                self._isMakcuConnected = False
                self.makcuConnectionLabel.setText(t("disconnected"))
                self.makcuConnectionLabel.setStyleSheet("color: #e74c3c; font-weight: bold;")
                self.makcuConnectBtn.setText(t("makcu_connect"))
        except ImportError:
            self.makcuConnectionLabel.setText("pyserial N/A")
            self.makcuConnectionLabel.setStyleSheet("color: #e74c3c; font-weight: bold;")

    # === Xbox 360 回調函數 ===
    def _onXboxSensitivityChanged(self, value):
        """Xbox 靈敏度改變"""
        if self._config:
            self._config.xbox_sensitivity = value / 100.0
            try:
                from win_utils import set_xbox_sensitivity
                set_xbox_sensitivity(value / 100.0)
            except ImportError:
                pass

    def _onXboxDeadzoneChanged(self, value):
        """Xbox 死區改變"""
        if self._config:
            self._config.xbox_deadzone = value / 100.0
            try:
                from win_utils import set_xbox_deadzone
                set_xbox_deadzone(value / 100.0)
            except ImportError:
                pass

    def _onXboxConnectToggle(self):
        """Xbox 手把連線/斷線切換"""
        try:
            from win_utils import is_xbox_connected, connect_xbox, disconnect_xbox
            if is_xbox_connected():
                disconnect_xbox()
            else:
                connect_xbox()
            self._updateXboxConnectionStatus()
        except ImportError:
            QMessageBox.warning(
                self, t("config_error"),
                "vgamepad 未安裝。\n請執行: pip install vgamepad\n並安裝 ViGEmBus 驅動。"
            )

    def _updateXboxConnectionStatus(self):
        """更新 Xbox 連線狀態顯示"""
        try:
            from win_utils import is_xbox_connected, is_xbox_available
            if not is_xbox_available():
                self.xboxConnectionLabel.setText("vgamepad " + t("disconnected"))
                self.xboxConnectionLabel.setStyleSheet("color: #e74c3c; font-weight: bold;")
                self.xboxConnectBtn.setText(t("xbox_connect"))
                return
                
            if is_xbox_connected():
                self._isXboxConnected = True
                self.xboxConnectionLabel.setText(t("connected"))
                self.xboxConnectionLabel.setStyleSheet("color: #2ecc71; font-weight: bold;")
                self.xboxConnectBtn.setText(t("xbox_disconnect"))
            else:
                self._isXboxConnected = False
                self.xboxConnectionLabel.setText(t("disconnected"))
                self.xboxConnectionLabel.setStyleSheet("color: #e74c3c; font-weight: bold;")
                self.xboxConnectBtn.setText(t("xbox_connect"))
        except ImportError:
            self.xboxConnectionLabel.setText("vgamepad N/A")
            self.xboxConnectionLabel.setStyleSheet("color: #e74c3c; font-weight: bold;")

    def retranslateUi(self):
        """刷新翻譯"""
        super().retranslateUi()

        # 群組標題
        self.modelGroup.titleLabel.setText(t("model_settings"))
        self.fovGroup.titleLabel.setText(t("fov_and_detect_range"))
        self.generalGroup.titleLabel.setText(t("general_params"))
        self.pidGroup.titleLabel.setText(t("aim_speed_pid"))
        self.bezierGroup.titleLabel.setText(t("bezier_curve"))
        self.trackerGroup.titleLabel.setText(t("tracker_prediction"))

        # 模型設定
        self.modelCard.titleLabel.setText(t("model"))
        self.inferenceBackendCard.titleLabel.setText(t("inference_backend"))
        self._updateInferenceBackendSubtitle()
        self.openModelFolderCard.titleLabel.setText(t("open_model_folder"))
        self.openModelFolderBtn.setText(t("open_model_folder"))

        # FOV 與偵測範圍
        self.fovCard.titleLabel.setText(t("fov_size"))
        self.fovFollowCard.titleLabel.setText(t("fov_follow_mouse"))
        self.detectRangeCard.titleLabel.setText(t("detect_range_size"))
        self.detectRangeCard.contentLabel.setText(t("detect_range_note"))

        # 通用參數
        self.detectIntervalCard.titleLabel.setText(t("detect_interval"))
        self.screenshotIntervalCard.titleLabel.setText(t("screenshot_interval"))
        self.confidenceCard.titleLabel.setText(t("min_confidence"))
        self.aimPartCard.titleLabel.setText(t("aim_part"))
        self.mouseMoveCard.titleLabel.setText(t("mouse_move_method"))
        self.screenshotMethodCard.titleLabel.setText(t("screenshot_method"))
        self.uvcDeviceCard.titleLabel.setText("UVC Device Index")
        self.uvcResolutionCard.titleLabel.setText("UVC Resolution")
        self.uvcRefreshResolutionCard.titleLabel.setText("Refresh UVC Resolution List")
        self.uvcRefreshResolutionBtn.setText(t("refresh"))
        self.uvcFpsCard.titleLabel.setText("UVC FPS")
        self.uvcCaptureMethodCard.titleLabel.setText("UVC Capture Method")
        self.uvcPreviewCard.titleLabel.setText("UVC Preview Window")
        self.uvcPreviewScaleCard.titleLabel.setText("UVC Preview Scale Mode")
        self.alwaysAimCard.titleLabel.setText(t("always_aim"))
        self.keepDetectingCard.titleLabel.setText(t("keep_detecting"))
        self.idleDetectEnableCard.titleLabel.setText(t("idle_detect_enabled"))
        self.idleDetectIntervalCard.titleLabel.setText(t("idle_detect_interval"))
        self.singleTargetCard.titleLabel.setText(t("single_target_mode"))

        # Arduino 設定
        self.comPortCard.titleLabel.setText(t("arduino_com_port"))
        self.comRefreshBtn.setText(t("refresh"))
        self.connectionCard.titleLabel.setText(t("connected") + " / " + t("disconnected"))
        self.arduinoConnectCard.titleLabel.setText(t("arduino_connect"))
        self.arduinoConnectCard.contentLabel.setText(t("arduino_connect_desc"))
        self._updateArduinoConnectionStatus()
        self.guideCard.titleLabel.setText(t("arduino_guide"))
        self.guideBtn.setText(t("arduino_guide"))
        self.spoofCard.titleLabel.setText(t("spoof_device"))
        self.spoofBtn.setText(t("spoof_device"))
        self.verifySpoofCard.titleLabel.setText(t("verify_spoof"))
        self.verifySpoofBtn.setText(t("verify_spoof"))
        self.testHeartCard.titleLabel.setText(t("test_move_heart"))
        self.testHeartBtn.setText(t("test_move_heart"))

        # MAKCU 設定
        self.makcuComPortCard.titleLabel.setText(t("makcu_com_port"))
        self.makcuComRefreshBtn.setText(t("refresh"))
        self.makcuConnectionCard.titleLabel.setText(t("connected") + " / " + t("disconnected"))
        self.makcuConnectCard.titleLabel.setText(t("makcu_connect"))
        self.makcuConnectCard.contentLabel.setText(t("makcu_connect_desc"))
        self._updateMakcuConnectionStatus()

        # Xbox 設定
        self.xboxSensitivityCard.titleLabel.setText(t("xbox_sensitivity"))
        self.xboxDeadzoneCard.titleLabel.setText(t("xbox_deadzone"))
        self.xboxConnectionCard.titleLabel.setText(t("connected") + " / " + t("disconnected"))
        self.xboxConnectCard.titleLabel.setText(t("xbox_connect"))
        self.xboxConnectCard.contentLabel.setText(t("xbox_connect_desc"))

        # 更新 ComboBox 內容
        current_aim = self.aimPartCombo.currentIndex()
        self.aimPartCombo.clear()
        self.aimPartCombo.addItems([t("head"), t("body"), t("both")])
        self.aimPartCombo.setCurrentIndex(current_aim)

        # PID
        self.pidAxisPivot.setItemText('x', t("horizontal_x"))
        self.pidAxisPivot.setItemText('y', t("vertical_y"))
        self.pidPxCard.titleLabel.setText(t("reaction_speed_p"))
        self.pidIxCard.titleLabel.setText(t("error_correction_i"))
        self.pidDxCard.titleLabel.setText(t("stability_suppression_d"))
        self.pidPyCard.titleLabel.setText(t("reaction_speed_p"))
        self.pidDyCard.titleLabel.setText(t("stability_suppression_d"))
        self.pidYReduceEnableCard.titleLabel.setText(t("aim_y_reduce_enable"))
        self.pidYReduceDelayCard.titleLabel.setText(t("aim_y_reduce_delay"))

        # 貝塞爾
        self.bezierEnableCard.titleLabel.setText(t("bezier_curve_enable"))
        self.bezierStrengthCard.titleLabel.setText(t("bezier_curve_strength"))
        self.bezierStepsCard.titleLabel.setText(t("bezier_curve_steps"))

        # 追蹤
        self.trackerEnableCard.titleLabel.setText(t("tracker_enable"))
        self.trackerTimeCard.titleLabel.setText(t("tracker_prediction_time"))
        self.trackerSmoothCard.titleLabel.setText(t("tracker_smoothing_factor"))
        self.trackerThresholdCard.titleLabel.setText(t("tracker_stop_threshold"))
        self.trackerShowCard.titleLabel.setText(t("tracker_show_prediction"))
