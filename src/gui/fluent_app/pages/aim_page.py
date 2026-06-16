# aim_page.py
"""Aim Assist Page - Move Method, Arduino, Xbox, PID, Smart Jitter, Target Priority, Target Tracking"""

import os
import re
import sys
import subprocess
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QStackedWidget, QMessageBox
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtCore import Qt
from qfluentwidgets import (
    SettingCardGroup, SwitchSettingCard,
    FluentIcon,
    ComboBox, SettingCard,
    SegmentedWidget,
    BodyLabel, PushButton,
)
from ..components.no_wheel_widgets import NoWheelDoubleSpinBox as DoubleSpinBox
from ..components.slider_spin_card import SliderLabelCard, SliderSpinCard

from ..base_page import BasePage
from ..language_manager import t


class AimPage(BasePage):
    """Aim Assist Settings Page"""

    def __init__(self, parent=None):
        super().__init__("tab_aim_control", parent)
        self._config = None
        self._isLoadingConfig = False
        self._isArduinoConnected = False
        self._isXboxConnected = False
        self._initWidgets()
        self._initLayout()
        self._connectSignals()

    def setConfig(self, config):
        """Sets Config instance and loads values"""
        self._config = config
        self._loadFromConfig()

    def _initWidgets(self):
        """Initializes all controls"""

        # === General (Aim Part + Move Method) ===
        self.generalGroup = SettingCardGroup(t("general_params"), self.scrollWidget)

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

        # === Arduino Settings ===
        self.arduinoGroup = SettingCardGroup("Arduino", self.scrollWidget)

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

        self.arduinoBaudCombo = ComboBox()
        self.arduinoBaudCombo.addItems(["115200", "500000", "1000000", "2000000", "4000000"])
        self.arduinoBaudCombo.setMinimumWidth(120)
        self.arduinoBaudCard = SettingCard(
            FluentIcon.SPEED_HIGH,
            t("arduino_baud_rate", "Baud Rate"),
            t("arduino_baud_rate_desc", "⚠ Must match the baud rate in your Arduino sketch"),
            self.arduinoGroup
        )
        self.arduinoBaudCard.hBoxLayout.addWidget(self.arduinoBaudCombo, 0, Qt.AlignmentFlag.AlignRight)
        self.arduinoBaudCard.hBoxLayout.addSpacing(16)

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

        self.guideBtn = PushButton(t("arduino_guide"))
        self.guideCard = SettingCard(
            FluentIcon.BOOK_SHELF,
            t("arduino_guide"),
            "",
            self.arduinoGroup
        )
        self.guideCard.hBoxLayout.addWidget(self.guideBtn, 0, Qt.AlignmentFlag.AlignRight)
        self.guideCard.hBoxLayout.addSpacing(16)

        self.spoofBtn = PushButton(t("spoof_device"))
        self.spoofCard = SettingCard(
            FluentIcon.VPN,
            t("spoof_device"),
            "",
            self.arduinoGroup
        )
        self.spoofCard.hBoxLayout.addWidget(self.spoofBtn, 0, Qt.AlignmentFlag.AlignRight)
        self.spoofCard.hBoxLayout.addSpacing(16)

        self.verifySpoofBtn = PushButton(t("verify_spoof"))
        self.verifySpoofCard = SettingCard(
            FluentIcon.ACCEPT,
            t("verify_spoof"),
            "",
            self.arduinoGroup
        )
        self.verifySpoofCard.hBoxLayout.addWidget(self.verifySpoofBtn, 0, Qt.AlignmentFlag.AlignRight)
        self.verifySpoofCard.hBoxLayout.addSpacing(16)

        self.testHeartBtn = PushButton(t("test_move_heart"))
        self.testHeartCard = SettingCard(
            FluentIcon.HEART,
            t("test_move_heart"),
            "",
            self.arduinoGroup
        )
        self.testHeartCard.hBoxLayout.addWidget(self.testHeartBtn, 0, Qt.AlignmentFlag.AlignRight)
        self.testHeartCard.hBoxLayout.addSpacing(16)

        # === Xbox 360 Controller ===
        self.xboxGroup = SettingCardGroup("Xbox 360 Controller", self.scrollWidget)

        self.xboxSensitivityCard = SliderSpinCard(
            FluentIcon.SPEED_HIGH,
            t("xbox_sensitivity"),
            10, 500,
            suffix="%",
            description="",
            parent=self.xboxGroup
        )

        self.xboxDeadzoneCard = SliderSpinCard(
            FluentIcon.REMOVE,
            t("xbox_deadzone"),
            0, 50,
            suffix="%",
            description="",
            parent=self.xboxGroup
        )

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

        # === PID Parameters ===
        self.pidGroup = SettingCardGroup(t("aim_speed_pid"), self.scrollWidget)

        self.pidAxisPivot = SegmentedWidget()
        self.pidAxisPivot.addItem(routeKey='x', text=t("horizontal_x"))
        self.pidAxisPivot.addItem(routeKey='y', text=t("vertical_y"))
        self.pidAxisPivot.setCurrentItem('x')
        self.pidAxisPivot.currentItemChanged.connect(self._onPidAxisChanged)

        self.pidStackedWidget = QStackedWidget()

        self.pidPxCard = SliderLabelCard(
            FluentIcon.SPEED_HIGH,
            t("reaction_speed_p"),
            0, 100,
            format_func=lambda v: f"{v/100:.2f}",
            parent=self.pidGroup
        )

        self.pidIxCard = SliderLabelCard(
            FluentIcon.SYNC,
            t("error_correction_i"),
            0, 100,
            format_func=lambda v: f"{v/100:.2f}",
            parent=self.pidGroup
        )

        self.pidDxCard = SliderLabelCard(
            FluentIcon.ALIGNMENT,
            t("stability_suppression_d"),
            0, 100,
            format_func=lambda v: f"{v/100:.2f}",
            parent=self.pidGroup
        )

        self.pidPyCard = SliderLabelCard(
            FluentIcon.SPEED_HIGH,
            t("reaction_speed_p"),
            0, 100,
            format_func=lambda v: f"{v/100:.2f}",
            parent=self.pidGroup
        )

        self.pidIyCard = SliderLabelCard(
            FluentIcon.SYNC,
            t("error_correction_i"),
            0, 100,
            format_func=lambda v: f"{v/100:.2f}",
            parent=self.pidGroup
        )

        self.pidDyCard = SliderLabelCard(
            FluentIcon.ALIGNMENT,
            t("stability_suppression_d"),
            0, 100,
            format_func=lambda v: f"{v/100:.2f}",
            parent=self.pidGroup
        )

        self.pidYReduceEnableCard = SwitchSettingCard(
            FluentIcon.CARE_UP_SOLID,
            t("aim_y_reduce_enable"),
            "",
            parent=self.pidGroup
        )

        self.pidYReduceDelayCard = SliderLabelCard(
            FluentIcon.STOP_WATCH,
            t("aim_y_reduce_delay"),
            0, 500,
            format_func=lambda v: f"{v/100:.2f} s",
            parent=self.pidGroup
        )

        self.pidYReduceFloorCard = SliderLabelCard(
            FluentIcon.CARE_DOWN_SOLID,
            "Y Floor",
            0, 100,
            format_func=lambda v: f"{v/100:.2f}",
            description="Min Y multiplier after ramp — 0.00 = full cut, 1.00 = no suppression",
            parent=self.pidGroup
        )

        self.pidYReduceRampCard = SliderLabelCard(
            FluentIcon.STOP_WATCH,
            "Y Ramp Window",
            0, 200,
            format_func=lambda v: f"{v/100:.2f} s",
            description="Time to fade 1.0 → floor after delay (0 = instant cut)",
            parent=self.pidGroup
        )

        self.pidYReduceSettleCard = SliderLabelCard(
            FluentIcon.ALIGNMENT,
            "Y Settle Threshold",
            0, 50,
            format_func=lambda v: "Off" if v == 0 else f"{v} px",
            description="Skip suppression while vertical error > this — waits until aim is settled (0 = off)",
            parent=self.pidGroup
        )

        self.pidYReduceVelCard = SliderLabelCard(
            FluentIcon.SPEED_MEDIUM,
            "Y Velocity Restore",
            0, 500,
            format_func=lambda v: "Off" if v == 0 else f"{v} px/s",
            description="Restore full Y tracking if target moves vertically faster than this (0 = off)",
            parent=self.pidGroup
        )

        # === Anti-Detection (Smart Jitter only) ===
        self.antiDetectionGroup = SettingCardGroup(t("anti_detection", "Anti-Detection"), self.scrollWidget)

        self.smartJitterEnableCard = SwitchSettingCard(
            FluentIcon.MOVE,
            t("smart_jitter_label", "Smart Jitter"),
            t("smart_jitter_desc", "Add jitter when target box is small (far targets). Fires while shooting."),
            parent=self.antiDetectionGroup
        )

        self.smartJitterLmbCard = SwitchSettingCard(
            FluentIcon.FINGERPRINT,
            t("smart_jitter_lmb_label", "Only While Shooting (LMB Held)"),
            t("smart_jitter_lmb_desc", "Jitter only fires when an aim key is held"),
            parent=self.antiDetectionGroup
        )

        self.smartJitterStrengthSpin = DoubleSpinBox()
        self.smartJitterStrengthSpin.setRange(0.1, 200.0)
        self.smartJitterStrengthSpin.setSingleStep(0.5)
        self.smartJitterStrengthSpin.setDecimals(1)
        self.smartJitterStrengthSpin.setSuffix(" px")
        self.smartJitterStrengthSpin.setMinimumWidth(110)
        self.smartJitterLevelCard = SettingCard(
            FluentIcon.SPEED_HIGH,
            t("smart_jitter_level_label", "Jitter Strength"),
            t("smart_jitter_level_desc", "Max pixel offset radius applied per frame while jitter fires"),
            self.antiDetectionGroup
        )
        self.smartJitterLevelCard.hBoxLayout.addWidget(self.smartJitterStrengthSpin, 0)
        self.smartJitterLevelCard.hBoxLayout.addSpacing(16)

        self.smartJitterThreshCard = SliderLabelCard(
            FluentIcon.ZOOM_OUT,
            t("smart_jitter_threshold_label", "Box Size Threshold"),
            1, 50,
            format_func=lambda v: f"{v}%",
            description=t("smart_jitter_threshold_desc", "Jitter fires when box height < this % of detection range"),
            slider_width=160,
            parent=self.antiDetectionGroup
        )

        # === Target Priority ===
        self.targetPriorityGroup = SettingCardGroup(t("target_priority", "Target Priority"), self.scrollWidget)

        self.targetPriorityModeCombo = ComboBox()
        self.targetPriorityModeCombo.addItems(["Distance", "Confidence", "Composite"])
        self.targetPriorityModeCombo.setMinimumWidth(130)
        self.targetPriorityModeCard = SettingCard(
            FluentIcon.PEOPLE,
            t("target_priority_mode", "Priority Mode"),
            t("target_priority_mode_desc", "How to select the best target"),
            self.targetPriorityGroup
        )
        self.targetPriorityModeCard.hBoxLayout.addWidget(self.targetPriorityModeCombo, 0)
        self.targetPriorityModeCard.hBoxLayout.addSpacing(16)

        self.targetPriorityWeightCard = SliderLabelCard(
            FluentIcon.CERTIFICATE,
            t("target_priority_confidence_weight", "Confidence Weight"),
            0, 100,
            format_func=lambda v: f"{v}%",
            description=t("target_priority_weight_desc", "Used in Composite mode only"),
            slider_width=160,
            parent=self.targetPriorityGroup
        )

        # === Target Tracking ===
        self.trackingGroup = SettingCardGroup(t("target_tracking", "Target Tracking"), self.scrollWidget)

        self.emaEnableCard = SwitchSettingCard(
            FluentIcon.SPEED_MEDIUM,
            t("ema_enabled", "EMA Smoothing"),
            t("ema_desc", "Exponential moving average on aim-point coordinates before PID. Reduces jitter."),
            parent=self.trackingGroup
        )

        self.emaAlphaCard = SliderLabelCard(
            FluentIcon.MIX_VOLUMES,
            t("ema_alpha", "EMA Alpha"),
            30, 100,
            format_func=lambda v: f"{v / 100:.2f}",
            description=t("ema_alpha_desc", "1.0 = raw (no smoothing), 0.30 = heavy smoothing"),
            slider_width=160,
            parent=self.trackingGroup
        )

        self.predictionEnableCard = SwitchSettingCard(
            FluentIcon.RINGER,
            t("prediction_enabled", "Velocity Prediction"),
            t("prediction_desc", "Extrapolate target position forward by the prediction horizon."),
            parent=self.trackingGroup
        )

        self.predictionHorizonCard = SliderLabelCard(
            FluentIcon.HISTORY,
            t("prediction_horizon", "Prediction Horizon"),
            5, 50,
            format_func=lambda v: f"{v} ms",
            label_width=55,
            parent=self.trackingGroup
        )

        self.predictionMaxVelCard = SliderLabelCard(
            FluentIcon.SPEED_HIGH,
            t("prediction_max_velocity", "Max Velocity Cap"),
            300, 3000,
            format_func=lambda v: f"{v} px/s",
            label_width=70,
            description=t("prediction_max_vel_desc", "Velocity spikes above this are treated as detection jumps and reset prediction"),
            parent=self.trackingGroup
        )

        self.predictionHistoryCard = SliderLabelCard(
            FluentIcon.HISTORY,
            t("prediction_history", "History Frames"),
            2, 6,
            format_func=lambda v: str(v),
            parent=self.trackingGroup
        )

        self.stickyLockCard = SwitchSettingCard(
            FluentIcon.PIN,
            t("sticky_lock_enabled", "Sticky Target Lock"),
            t("sticky_lock_desc", "Lock onto a target and hold aim across short detection gaps."),
            parent=self.trackingGroup
        )

        self.lockDecayCard = SliderLabelCard(
            FluentIcon.HISTORY,
            t("lock_decay_frames", "Lock Decay Frames"),
            3, 60,
            format_func=lambda v: f"{v} fr",
            description=t("lock_decay_desc", "Frames to hold aim after target is lost before releasing the lock"),
            label_width=55,
            parent=self.trackingGroup
        )

        self.lockIouCard = SliderLabelCard(
            FluentIcon.ZOOM_IN,
            t("lock_iou_threshold", "IoU Match Threshold"),
            10, 70,
            format_func=lambda v: f"{v / 100:.2f}",
            description=t("lock_iou_desc", "Minimum overlap required to match the same target across frames"),
            slider_width=160,
            parent=self.trackingGroup
        )

        self.kalmanEnableCard = SwitchSettingCard(
            FluentIcon.SPEED_HIGH,
            t("kalman_enabled_label", "Kalman Filter"),
            t("kalman_enabled_desc", "2D Kalman filter for aim-point smoothing. Mutually exclusive with EMA."),
            parent=self.trackingGroup
        )

        self.kalmanProcessNoiseCard = SliderLabelCard(
            FluentIcon.MOVE,
            t("kalman_process_noise_label", "Process Noise"),
            1, 100,
            format_func=lambda v: f"{v / 100:.2f}",
            description=t("kalman_noise_desc", "Lower = smoother but slower to react"),
            slider_width=160,
            parent=self.trackingGroup
        )

        self.kalmanMeasNoiseCard = SliderLabelCard(
            FluentIcon.ALIGNMENT,
            t("kalman_meas_noise_label", "Measurement Noise"),
            1, 100,
            format_func=lambda v: f"{v / 100:.2f}",
            description=t("kalman_noise_desc", "Lower = reacts faster but noisier"),
            slider_width=160,
            parent=self.trackingGroup
        )

    def _initLayout(self):
        """Layout all controls"""
        # General
        self.generalGroup.addSettingCard(self.aimPartCard)
        self.generalGroup.addSettingCard(self.mouseMoveCard)
        self.addContent(self.generalGroup)

        # Arduino
        self.arduinoGroup.addSettingCard(self.comPortCard)
        self.arduinoGroup.addSettingCard(self.arduinoBaudCard)
        self.arduinoGroup.addSettingCard(self.connectionCard)
        self.arduinoGroup.addSettingCard(self.arduinoConnectCard)
        self.arduinoGroup.addSettingCard(self.guideCard)
        self.arduinoGroup.addSettingCard(self.spoofCard)
        self.arduinoGroup.addSettingCard(self.verifySpoofCard)
        self.arduinoGroup.addSettingCard(self.testHeartCard)
        self.addContent(self.arduinoGroup)
        self.arduinoGroup.setVisible(False)

        # Xbox
        self.xboxGroup.addSettingCard(self.xboxSensitivityCard)
        self.xboxGroup.addSettingCard(self.xboxDeadzoneCard)
        self.xboxGroup.addSettingCard(self.xboxConnectionCard)
        self.xboxGroup.addSettingCard(self.xboxConnectCard)
        self.addContent(self.xboxGroup)
        self.xboxGroup.setVisible(False)

        # PID - tabbed X/Y layout
        pivotWidget = QWidget()
        pivotLayout = QHBoxLayout(pivotWidget)
        pivotLayout.setContentsMargins(16, 8, 16, 8)
        pivotLayout.addWidget(self.pidAxisPivot)
        pivotLayout.addStretch(1)

        self.pidXPage = QWidget()
        xPageLayout = QVBoxLayout(self.pidXPage)
        xPageLayout.setContentsMargins(0, 0, 0, 0)
        xPageLayout.setSpacing(0)
        xPageLayout.addWidget(self.pidPxCard)
        xPageLayout.addWidget(self.pidIxCard)
        xPageLayout.addWidget(self.pidDxCard)
        xPageLayout.addStretch(1)

        self.pidYPage = QWidget()
        yPageLayout = QVBoxLayout(self.pidYPage)
        yPageLayout.setContentsMargins(0, 0, 0, 0)
        yPageLayout.setSpacing(0)
        yPageLayout.addWidget(self.pidPyCard)
        yPageLayout.addWidget(self.pidIyCard)
        yPageLayout.addWidget(self.pidDyCard)
        yPageLayout.addWidget(self.pidYReduceEnableCard)
        yPageLayout.addWidget(self.pidYReduceDelayCard)
        yPageLayout.addWidget(self.pidYReduceFloorCard)
        yPageLayout.addWidget(self.pidYReduceRampCard)
        yPageLayout.addWidget(self.pidYReduceSettleCard)
        yPageLayout.addWidget(self.pidYReduceVelCard)

        self.pidStackedWidget.addWidget(self.pidXPage)
        self.pidStackedWidget.addWidget(self.pidYPage)

        self.pidGroup.vBoxLayout.addWidget(pivotWidget)
        self.pidGroup.vBoxLayout.addWidget(self.pidStackedWidget)
        self.addContent(self.pidGroup)

        # Anti-Detection (Smart Jitter)
        self.antiDetectionGroup.addSettingCard(self.smartJitterEnableCard)
        self.antiDetectionGroup.addSettingCard(self.smartJitterLmbCard)
        self.antiDetectionGroup.addSettingCard(self.smartJitterLevelCard)
        self.antiDetectionGroup.addSettingCard(self.smartJitterThreshCard)
        self.addContent(self.antiDetectionGroup)

        # Target Priority
        self.targetPriorityGroup.addSettingCard(self.targetPriorityModeCard)
        self.targetPriorityGroup.addSettingCard(self.targetPriorityWeightCard)
        self.addContent(self.targetPriorityGroup)

        # Target Tracking
        self.trackingGroup.addSettingCard(self.emaEnableCard)
        self.trackingGroup.addSettingCard(self.emaAlphaCard)
        self.trackingGroup.addSettingCard(self.predictionEnableCard)
        self.trackingGroup.addSettingCard(self.predictionHorizonCard)
        self.trackingGroup.addSettingCard(self.predictionMaxVelCard)
        self.trackingGroup.addSettingCard(self.predictionHistoryCard)
        self.trackingGroup.addSettingCard(self.stickyLockCard)
        self.trackingGroup.addSettingCard(self.lockDecayCard)
        self.trackingGroup.addSettingCard(self.lockIouCard)
        self.trackingGroup.addSettingCard(self.kalmanEnableCard)
        self.trackingGroup.addSettingCard(self.kalmanProcessNoiseCard)
        self.trackingGroup.addSettingCard(self.kalmanMeasNoiseCard)
        self.addContent(self.trackingGroup)

        self.scrollLayout.addStretch(1)

    def _connectSignals(self):
        """Connect signals"""
        # General
        self.aimPartCombo.currentIndexChanged.connect(self._onAimPartChanged)
        self.mouseMoveCombo.currentTextChanged.connect(self._onMouseMoveChanged)

        # Arduino
        self.comRefreshBtn.clicked.connect(self._refreshComPorts)
        self.comPortCombo.currentTextChanged.connect(self._onComPortChanged)
        self.arduinoConnectBtn.clicked.connect(self._onArduinoConnectToggle)
        self.guideBtn.clicked.connect(self._onOpenGuide)
        self.spoofBtn.clicked.connect(self._onSpoofDevice)
        self.verifySpoofBtn.clicked.connect(self._onVerifySpoof)
        self.testHeartBtn.clicked.connect(self._onTestHeart)
        self.arduinoBaudCombo.currentTextChanged.connect(self._onArduinoBaudChanged)

        # Xbox
        self.xboxSensitivityCard.valueChanged.connect(self._onXboxSensitivityChanged)
        self.xboxDeadzoneCard.valueChanged.connect(self._onXboxDeadzoneChanged)
        self.xboxConnectBtn.clicked.connect(self._onXboxConnectToggle)

        # PID
        self.pidPxCard.valueChanged.connect(lambda v: self._onPidChanged('pid_kp_x', v))
        self.pidIxCard.valueChanged.connect(lambda v: self._onPidChanged('pid_ki_x', v))
        self.pidDxCard.valueChanged.connect(lambda v: self._onPidChanged('pid_kd_x', v))
        self.pidPyCard.valueChanged.connect(lambda v: self._onPidChanged('pid_kp_y', v))
        self.pidIyCard.valueChanged.connect(lambda v: self._onPidChanged('pid_ki_y', v))
        self.pidDyCard.valueChanged.connect(lambda v: self._onPidChanged('pid_kd_y', v))
        self.pidYReduceEnableCard.checkedChanged.connect(lambda checked: self._onPidChanged('aim_y_reduce_enabled', checked, is_bool=True))
        self.pidYReduceDelayCard.valueChanged.connect(lambda v: self._onPidChanged('aim_y_reduce_delay', v))
        self.pidYReduceFloorCard.valueChanged.connect(lambda v: self._onPidChanged('aim_y_reduce_floor', v))
        self.pidYReduceRampCard.valueChanged.connect(lambda v: self._onPidChanged('aim_y_reduce_ramp', v))
        self.pidYReduceSettleCard.valueChanged.connect(
            lambda v: setattr(self._config, 'aim_y_reduce_settle_px', float(v)) if self._config else None)
        self.pidYReduceVelCard.valueChanged.connect(
            lambda v: setattr(self._config, 'aim_y_vel_restore_px_s', float(v)) if self._config else None)

        # Smart Jitter
        self.smartJitterEnableCard.checkedChanged.connect(self._onSmartJitterEnableChanged)
        self.smartJitterLmbCard.checkedChanged.connect(self._onSmartJitterLmbChanged)
        self.smartJitterStrengthSpin.valueChanged.connect(self._onSmartJitterStrengthChanged)
        self.smartJitterThreshCard.valueChanged.connect(self._onSmartJitterThreshChanged)

        # Target Priority
        self.targetPriorityModeCombo.currentTextChanged.connect(self._onTargetPriorityModeChanged)
        self.targetPriorityWeightCard.valueChanged.connect(self._onTargetPriorityWeightChanged)

        # Target Tracking
        self.emaEnableCard.checkedChanged.connect(self._onEmaEnableChanged)
        self.emaAlphaCard.valueChanged.connect(self._onEmaAlphaChanged)
        self.predictionEnableCard.checkedChanged.connect(self._onPredictionEnableChanged)
        self.predictionHorizonCard.valueChanged.connect(self._onPredictionHorizonChanged)
        self.predictionMaxVelCard.valueChanged.connect(self._onPredictionMaxVelChanged)
        self.predictionHistoryCard.valueChanged.connect(self._onPredictionHistoryChanged)
        self.stickyLockCard.checkedChanged.connect(self._onStickyLockChanged)
        self.lockDecayCard.valueChanged.connect(self._onLockDecayChanged)
        self.lockIouCard.valueChanged.connect(self._onLockIouChanged)
        self.kalmanEnableCard.checkedChanged.connect(self._onKalmanEnableChanged)
        self.kalmanProcessNoiseCard.valueChanged.connect(self._onKalmanProcessNoiseChanged)
        self.kalmanMeasNoiseCard.valueChanged.connect(self._onKalmanMeasNoiseChanged)

    def _loadFromConfig(self):
        """Load values from Config"""
        if not self._config:
            return
        self._isLoadingConfig = True
        try:
            # General
            aim_parts = ["head", "body", "both"]
            if self._config.aim_part in aim_parts:
                self.aimPartCombo.setCurrentIndex(aim_parts.index(self._config.aim_part))

            mouse_methods = ["ddxoft", "mouse_event", "sendinput", "arduino", "makcu", "xbox"]
            if self._config.mouse_move_method in mouse_methods:
                self.mouseMoveCombo.setCurrentIndex(mouse_methods.index(self._config.mouse_move_method))

            self._updateMethodGroupVisibility(self._config.mouse_move_method)

            # Arduino
            if self._config.arduino_com_port:
                idx = self.comPortCombo.findText(self._config.arduino_com_port)
                if idx >= 0:
                    self.comPortCombo.setCurrentIndex(idx)
                elif self.comPortCombo.count() > 1:
                    self.comPortCombo.setCurrentIndex(1)
            elif self.comPortCombo.count() > 1:
                self.comPortCombo.setCurrentIndex(1)

            arduino_baud = str(getattr(self._config, 'arduino_baud_rate', 115200))
            if self.arduinoBaudCombo.findText(arduino_baud) < 0:
                arduino_baud = "115200"
            self.arduinoBaudCombo.setCurrentText(arduino_baud)
            self._updateArduinoConnectionStatus()

            # Xbox
            self.xboxSensitivityCard.setValue(int(getattr(self._config, 'xbox_sensitivity', 1.0) * 100))
            self.xboxDeadzoneCard.setValue(int(getattr(self._config, 'xbox_deadzone', 0.05) * 100))
            self._updateXboxConnectionStatus()

            # PID
            self.pidPxCard.setValue(int(self._config.pid_kp_x * 100))
            self.pidIxCard.setValue(int(self._config.pid_ki_x * 100))
            self.pidDxCard.setValue(int(self._config.pid_kd_x * 100))
            self.pidPyCard.setValue(int(self._config.pid_kp_y * 100))
            self.pidIyCard.setValue(int(self._config.pid_ki_y * 100))
            self.pidDyCard.setValue(int(self._config.pid_kd_y * 100))
            self.pidYReduceEnableCard.setChecked(getattr(self._config, 'aim_y_reduce_enabled', False))
            self.pidYReduceDelayCard.setValue(int(getattr(self._config, 'aim_y_reduce_delay', 0.6) * 100))
            self.pidYReduceFloorCard.setValue(int(getattr(self._config, 'aim_y_reduce_floor', 0.0) * 100))
            self.pidYReduceRampCard.setValue(int(getattr(self._config, 'aim_y_reduce_ramp', 0.0) * 100))
            self.pidYReduceSettleCard.setValue(int(getattr(self._config, 'aim_y_reduce_settle_px', 0.0)))
            self.pidYReduceVelCard.setValue(int(getattr(self._config, 'aim_y_vel_restore_px_s', 0.0)))

            # Smart Jitter
            sj_on = bool(getattr(self._config, 'smart_jitter_enabled', False))
            self.smartJitterEnableCard.setChecked(sj_on)
            self.smartJitterLmbCard.setChecked(bool(getattr(self._config, 'smart_jitter_lmb_gate', True)))
            self.smartJitterStrengthSpin.setValue(float(getattr(self._config, 'smart_jitter_strength', 6.0)))
            self.smartJitterThreshCard.setValue(int(getattr(self._config, 'smart_jitter_box_threshold_pct', 15.0)))
            self.smartJitterLmbCard.setEnabled(sj_on)
            self.smartJitterLevelCard.setEnabled(sj_on)
            self.smartJitterThreshCard.setEnabled(sj_on)

            # Target Priority
            mode_map = {"distance": "Distance", "confidence": "Confidence", "composite": "Composite"}
            mode_text = mode_map.get(str(getattr(self._config, 'target_priority_mode', 'distance')), "Distance")
            self.targetPriorityModeCombo.setCurrentText(mode_text)
            self.targetPriorityWeightCard.setValue(int(getattr(self._config, 'target_priority_confidence_weight', 0.5) * 100))

            # Target Tracking
            self.emaEnableCard.setChecked(bool(getattr(self._config, 'ema_enabled', False)))
            self.emaAlphaCard.setValue(int(getattr(self._config, 'ema_alpha', 0.7) * 100))
            self.predictionEnableCard.setChecked(bool(getattr(self._config, 'prediction_enabled', False)))
            self.predictionHorizonCard.setValue(int(getattr(self._config, 'prediction_horizon_ms', 10.0)))
            self.predictionMaxVelCard.setValue(int(getattr(self._config, 'prediction_max_velocity', 1200.0)))
            self.predictionHistoryCard.setValue(int(getattr(self._config, 'prediction_history_len', 3)))
            self.stickyLockCard.setChecked(bool(getattr(self._config, 'sticky_lock_enabled', False)))
            self.lockDecayCard.setValue(int(getattr(self._config, 'lock_decay_frames', 15)))
            self.lockIouCard.setValue(int(getattr(self._config, 'lock_iou_threshold', 0.3) * 100))

            kalman_on = bool(getattr(self._config, 'kalman_enabled', False))
            self.kalmanEnableCard.setChecked(kalman_on)
            self.kalmanProcessNoiseCard.setValue(int(getattr(self._config, 'kalman_process_noise', 0.01) * 100))
            self.kalmanMeasNoiseCard.setValue(int(getattr(self._config, 'kalman_measurement_noise', 0.1) * 100))
            self.kalmanProcessNoiseCard.setEnabled(kalman_on)
            self.kalmanMeasNoiseCard.setEnabled(kalman_on)
            self.emaEnableCard.setEnabled(not kalman_on)
            self.emaAlphaCard.setEnabled(not kalman_on and bool(getattr(self._config, 'ema_enabled', False)))
            if kalman_on:
                self.emaEnableCard.setChecked(False)
                if self._config:
                    self._config.ema_enabled = False
        finally:
            self._isLoadingConfig = False

    # ── Helpers ──────────────────────────────────

    @staticmethod
    def _sorted_com_ports(ports):
        def _num(p):
            m = re.search(r'(\d+)$', p.device)
            return int(m.group(1)) if m else 0
        return sorted(ports, key=_num, reverse=True)

    def _getEmbeddedPythonExe(self) -> str:
        src_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        python_exe = os.path.join(src_dir, "python", "python.exe")
        if os.path.exists(python_exe):
            return python_exe
        return sys.executable

    def _runLocalInstallerScript(self, script_name: str, feature_name: str, capture_output: bool = True) -> bool:
        src_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        script_path = os.path.join(src_dir, script_name)
        if not os.path.exists(script_path):
            QMessageBox.warning(self, f"{feature_name} install failed", f"Missing installer script:\n{script_path}")
            return False
        python_exe = self._getEmbeddedPythonExe()
        install_cmd = [python_exe, script_path]
        try:
            if capture_output:
                result = subprocess.run(install_cmd, check=True, text=True, capture_output=True)
                if result.stdout:
                    print(f"[Dependency][{feature_name}][stdout]\n{result.stdout}")
            else:
                subprocess.run(install_cmd, check=True, text=True)
            return True
        except subprocess.CalledProcessError as exc:
            parts = []
            if getattr(exc, 'stderr', None):
                parts.append(exc.stderr.strip())
            if getattr(exc, 'stdout', None):
                parts.append(exc.stdout.strip())
            error_text = "\n".join(parts) if parts else str(exc)
            QMessageBox.warning(
                self, f"{feature_name} install failed",
                f"Failed command: {' '.join(install_cmd)}\n\n{error_text}\n\nPlease run the installer script manually and try again."
            )
            return False

    def _refreshComPorts(self):
        self.comPortCombo.clear()
        self.comPortCombo.addItem(t("no_com_port"))
        try:
            import serial.tools.list_ports
            for port in self._sorted_com_ports(serial.tools.list_ports.comports()):
                self.comPortCombo.addItem(port.device)
        except ImportError:
            pass

    def _updateMethodGroupVisibility(self, method):
        self.arduinoGroup.setVisible(method == "arduino")
        self.xboxGroup.setVisible(method == "xbox")

    # ── General Callbacks ────────────────────────

    def _onAimPartChanged(self, index):
        if self._config:
            parts = ["head", "body", "both"]
            self._config.aim_part = parts[index]

    def _onMouseMoveChanged(self, text):
        if self._config:
            self._config.mouse_move_method = text
            if text == "makcu":
                self._config.mouse_click_method = "makcu"
            if text == "ddxoft":
                try:
                    from win_utils import ensure_ddxoft_ready
                    ensure_ddxoft_ready()
                except ImportError:
                    pass
        self._updateMethodGroupVisibility(text)
        # Notify keys page to update MAKCU group visibility
        try:
            win = self.window()
            if hasattr(win, 'keysInterface'):
                win.keysInterface._updateMakcuVisibility()
        except Exception:
            pass

    # ── Arduino Callbacks ────────────────────────

    def _onComPortChanged(self, text):
        if self._config and text != t("no_com_port"):
            self._config.arduino_com_port = text

    def _onArduinoBaudChanged(self, text):
        if self._config and not self._isLoadingConfig:
            try:
                self._config.arduino_baud_rate = int(text)
            except ValueError:
                pass

    def _onArduinoConnectToggle(self):
        try:
            from win_utils import is_arduino_connected, connect_arduino, disconnect_arduino
            if is_arduino_connected():
                disconnect_arduino()
            else:
                com_port = self.comPortCombo.currentText()
                if not com_port or com_port == t("no_com_port"):
                    QMessageBox.warning(self, t("config_error"), t("no_com_port"))
                    return
                success = connect_arduino(com_port)
                if not success:
                    QMessageBox.warning(self, t("config_error"),
                                        f"Arduino {t('disconnected')}: {com_port}")
            self._updateArduinoConnectionStatus()
        except ImportError:
            QMessageBox.warning(self, t("config_error"), "pyserial not installed.\npip install pyserial")

    def _updateArduinoConnectionStatus(self):
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

    def _onOpenGuide(self):
        from PyQt6.QtCore import QUrl
        guide_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
            "Arduino_User_Guide.html"
        )
        if os.path.exists(guide_path):
            QDesktopServices.openUrl(QUrl.fromLocalFile(guide_path))

    def _onSpoofDevice(self):
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
                    QMessageBox.information(self, t("spoof_success_title"),
                                            t("spoof_success_msg").replace("\\n", "\n"))
                else:
                    QMessageBox.warning(self, t("spoof_error_title"),
                                        f"Spoof operation returned unsuccessful.\nFile: {boards_path}")
            except FileNotFoundError as e:
                QMessageBox.warning(self, t("spoof_error_title"), str(e))
            except Exception as e:
                QMessageBox.critical(self, t("spoof_error_title"), f"Error: {e}")

    def _onVerifySpoof(self):
        try:
            from win_utils.arduino_spoofer import verify_spoof
            specific_port = None
            if self._config and self._config.arduino_com_port:
                specific_port = self._config.arduino_com_port
            is_spoofed, message = verify_spoof(specific_port)
            if is_spoofed:
                QMessageBox.information(self, t("verify_success_title"), message)
            else:
                QMessageBox.warning(self, t("verify_fail_title"), message)
        except Exception as e:
            QMessageBox.critical(self, t("verify_fail_title"), f"Error: {e}")

    def _onTestHeart(self):
        import math as _math
        import threading as _threading
        import time as _time
        reply = QMessageBox.question(
            self, t("test_heart_confirm_title"),
            t("test_heart_confirm_msg").replace("\\n", "\n"),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            from win_utils.arduino_mouse import arduino_mouse
            if not arduino_mouse.is_connected():
                com_port = self._config.arduino_com_port if self._config else ""
                if not com_port:
                    QMessageBox.warning(self, t("test_heart_confirm_title"),
                                        "Arduino not connected. Please set COM port first.")
                    return
                if not arduino_mouse.connect(com_port):
                    QMessageBox.warning(self, t("test_heart_confirm_title"),
                                        f"Failed to connect to {com_port}.")
                    return

            def _draw_heart():
                num_steps = 120
                scale = 3.0
                points = []
                for i in range(num_steps + 1):
                    angle = 2 * _math.pi * i / num_steps
                    x = 16 * (_math.sin(angle) ** 3)
                    y = -(13 * _math.cos(angle) - 5 * _math.cos(2 * angle)
                           - 2 * _math.cos(3 * angle) - _math.cos(4 * angle))
                    points.append((x * scale, y * scale))
                for i in range(1, len(points)):
                    dx = int(round(points[i][0] - points[i - 1][0]))
                    dy = int(round(points[i][1] - points[i - 1][1]))
                    if dx != 0 or dy != 0:
                        arduino_mouse.move(dx, dy)
                    _time.sleep(0.015)

            _threading.Thread(target=_draw_heart, daemon=True).start()

    # ── Xbox Callbacks ───────────────────────────

    def _onXboxSensitivityChanged(self, value):
        if self._config:
            self._config.xbox_sensitivity = value / 100.0
            try:
                from win_utils import set_xbox_sensitivity
                set_xbox_sensitivity(value / 100.0)
            except ImportError:
                pass

    def _onXboxDeadzoneChanged(self, value):
        if self._config:
            self._config.xbox_deadzone = value / 100.0
            try:
                from win_utils import set_xbox_deadzone
                set_xbox_deadzone(value / 100.0)
            except ImportError:
                pass

    def _onXboxConnectToggle(self):
        try:
            from win_utils import is_xbox_connected, connect_xbox, disconnect_xbox
            if is_xbox_connected():
                disconnect_xbox()
            else:
                connect_xbox()
            self._updateXboxConnectionStatus()
        except ImportError:
            QMessageBox.warning(self, t("config_error"),
                                 "vgamepad not installed.\npip install vgamepad\nInstall ViGEmBus driver.")

    def _updateXboxConnectionStatus(self):
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

    # === PID Callbacks ===

    def _onPidAxisChanged(self, routeKey: str):
        self.pidStackedWidget.setCurrentIndex(0 if routeKey == 'x' else 1)

    def _onPidChanged(self, attr, value, is_bool=False):
        if self._config:
            if is_bool:
                setattr(self._config, attr, value)
            else:
                setattr(self._config, attr, value / 100.0)

    # === Smart Jitter Callbacks ===

    def _onSmartJitterEnableChanged(self, checked):
        if self._config:
            self._config.smart_jitter_enabled = bool(checked)
        self.smartJitterLmbCard.setEnabled(bool(checked))
        self.smartJitterLevelCard.setEnabled(bool(checked))
        self.smartJitterThreshCard.setEnabled(bool(checked))

    def _onSmartJitterLmbChanged(self, checked):
        if self._config:
            self._config.smart_jitter_lmb_gate = bool(checked)

    def _onSmartJitterStrengthChanged(self, value):
        if self._config:
            self._config.smart_jitter_strength = float(value)

    def _onSmartJitterThreshChanged(self, value):
        if self._config:
            self._config.smart_jitter_box_threshold_pct = float(value)

    # === Target Priority Callbacks ===

    def _onTargetPriorityModeChanged(self, text):
        if self._config:
            self._config.target_priority_mode = str(text).lower()

    def _onTargetPriorityWeightChanged(self, value):
        if self._config:
            self._config.target_priority_confidence_weight = value / 100.0

    # === Target Tracking Callbacks ===

    def _onEmaEnableChanged(self, checked):
        if self._config:
            self._config.ema_enabled = bool(checked)
        self.kalmanEnableCard.setEnabled(not checked)
        if checked:
            self.kalmanEnableCard.setChecked(False)
            if self._config:
                self._config.kalman_enabled = False
            self.kalmanProcessNoiseCard.setEnabled(False)
            self.kalmanMeasNoiseCard.setEnabled(False)
        self.emaAlphaCard.setEnabled(bool(checked))

    def _onEmaAlphaChanged(self, value):
        if self._config:
            self._config.ema_alpha = value / 100.0

    def _onPredictionEnableChanged(self, checked):
        if self._config:
            self._config.prediction_enabled = bool(checked)

    def _onPredictionHorizonChanged(self, value):
        if self._config:
            self._config.prediction_horizon_ms = float(value)

    def _onPredictionMaxVelChanged(self, value):
        if self._config:
            self._config.prediction_max_velocity = float(value)

    def _onPredictionHistoryChanged(self, value):
        if self._config:
            self._config.prediction_history_len = int(value)

    def _onStickyLockChanged(self, checked):
        if self._config:
            self._config.sticky_lock_enabled = bool(checked)

    def _onLockDecayChanged(self, value):
        if self._config:
            self._config.lock_decay_frames = int(value)

    def _onLockIouChanged(self, value):
        if self._config:
            self._config.lock_iou_threshold = value / 100.0

    def _onKalmanEnableChanged(self, checked):
        if self._config:
            self._config.kalman_enabled = bool(checked)
        self.emaEnableCard.setEnabled(not checked)
        if checked:
            self.emaEnableCard.setChecked(False)
            if self._config:
                self._config.ema_enabled = False
            self.emaAlphaCard.setEnabled(False)
        self.kalmanProcessNoiseCard.setEnabled(bool(checked))
        self.kalmanMeasNoiseCard.setEnabled(bool(checked))

    def _onKalmanProcessNoiseChanged(self, value):
        if self._config:
            self._config.kalman_process_noise = value / 100.0

    def _onKalmanMeasNoiseChanged(self, value):
        if self._config:
            self._config.kalman_measurement_noise = value / 100.0

    def retranslateUi(self):
        """Refresh translations"""
        super().retranslateUi()

        self.generalGroup.titleLabel.setText(t("general_params"))
        self.aimPartCard.titleLabel.setText(t("aim_part"))
        self.mouseMoveCard.titleLabel.setText(t("mouse_move_method"))

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

        self.xboxSensitivityCard.titleLabel.setText(t("xbox_sensitivity"))
        self.xboxDeadzoneCard.titleLabel.setText(t("xbox_deadzone"))
        self.xboxConnectionCard.titleLabel.setText(t("connected") + " / " + t("disconnected"))
        self.xboxConnectCard.titleLabel.setText(t("xbox_connect"))
        self.xboxConnectCard.contentLabel.setText(t("xbox_connect_desc"))

        self.pidGroup.titleLabel.setText(t("aim_speed_pid"))
        self.pidAxisPivot.setItemText('x', t("horizontal_x"))
        self.pidAxisPivot.setItemText('y', t("vertical_y"))
        self.pidPxCard.titleLabel.setText(t("reaction_speed_p"))
        self.pidIxCard.titleLabel.setText(t("error_correction_i"))
        self.pidDxCard.titleLabel.setText(t("stability_suppression_d"))
        self.pidPyCard.titleLabel.setText(t("reaction_speed_p"))
        self.pidIyCard.titleLabel.setText(t("error_correction_i"))
        self.pidDyCard.titleLabel.setText(t("stability_suppression_d"))
        self.pidYReduceEnableCard.titleLabel.setText(t("aim_y_reduce_enable"))
        self.pidYReduceDelayCard.titleLabel.setText(t("aim_y_reduce_delay"))
        self.pidYReduceFloorCard.titleLabel.setText("Y Floor")
        self.pidYReduceRampCard.titleLabel.setText("Y Ramp Window")
        self.pidYReduceSettleCard.titleLabel.setText("Y Settle Threshold")
        self.pidYReduceVelCard.titleLabel.setText("Y Velocity Restore")

        self.antiDetectionGroup.titleLabel.setText(t("anti_detection", "Anti-Detection"))
        self.smartJitterEnableCard.titleLabel.setText(t("smart_jitter_label", "Smart Jitter"))
        self.smartJitterLmbCard.titleLabel.setText(t("smart_jitter_lmb_label", "Only While Shooting (LMB)"))
        self.smartJitterLevelCard.titleLabel.setText(t("smart_jitter_level_label", "Jitter Strength"))
        self.smartJitterThreshCard.titleLabel.setText(t("smart_jitter_threshold_label", "Box Size Threshold"))

        self.targetPriorityGroup.titleLabel.setText(t("target_priority", "Target Priority"))
        self.targetPriorityModeCard.titleLabel.setText(t("target_priority_mode", "Priority Mode"))
        self.targetPriorityWeightCard.titleLabel.setText(t("target_priority_confidence_weight", "Confidence Weight"))

        self.trackingGroup.titleLabel.setText(t("target_tracking", "Target Tracking"))
        self.emaEnableCard.titleLabel.setText(t("ema_enabled", "EMA Smoothing"))
        self.emaAlphaCard.titleLabel.setText(t("ema_alpha", "EMA Alpha"))
        self.kalmanEnableCard.titleLabel.setText(t("kalman_enabled_label", "Kalman Filter"))
        self.kalmanProcessNoiseCard.titleLabel.setText(t("kalman_process_noise_label", "Process Noise"))
        self.kalmanMeasNoiseCard.titleLabel.setText(t("kalman_meas_noise_label", "Measurement Noise"))

        current_aim = self.aimPartCombo.currentIndex()
        self.aimPartCombo.clear()
        self.aimPartCombo.addItems([t("head"), t("body"), t("both")])
        self.aimPartCombo.setCurrentIndex(current_aim)
