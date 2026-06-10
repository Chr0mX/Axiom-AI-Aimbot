# inference_page.py
"""Inference Page - Model, FOV, Capture, General Parameters, Inference Performance"""

import os
import glob
import sys
import subprocess
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QWidget, QMessageBox
from PyQt6.QtGui import QDesktopServices
from qfluentwidgets import (
    SettingCardGroup, SwitchSettingCard,
    FluentIcon,
    BodyLabel, ComboBox, PrimaryPushButton, SettingCard,
    PushButton, InfoBar, InfoBarPosition
)
from ..components.no_wheel_widgets import NoWheelDoubleSpinBox as DoubleSpinBox
from ..components.slider_spin_card import SliderSpinCard, SliderLabelCard

from ..base_page import BasePage
from ..language_manager import t


class InferencePage(BasePage):
    """Inference Settings Page — Model, FOV, Capture, General Parameters, Inference Performance"""

    def __init__(self, parent=None):
        super().__init__("tab_inference", parent)
        self._config = None
        self._isLoadingConfig = False
        self._initWidgets()
        self._initLayout()
        self._connectSignals()

    def setConfig(self, config):
        self._config = config
        if hasattr(self, 'detectRangeCard') and self._config:
            max_h = max(1080, self._config.height)
            self.detectRangeCard.slider.setMaximum(max_h)
            self.detectRangeCard.spinBox.setMaximum(max_h)
        self._loadFromConfig()

    def showEvent(self, event):
        super().showEvent(event)
        if self._config and hasattr(self, 'idleDetectEnableCard'):
            self.idleDetectEnableCard.setChecked(
                getattr(self._config, 'idle_detect_enabled', True)
            )

    # ──────────────────────────────────────────────
    # Widget initialisation
    # ──────────────────────────────────────────────

    def _initWidgets(self):
        # === Model Settings ===
        self.modelGroup = SettingCardGroup(t("model_settings"), self.scrollWidget)

        self.modelCombo = ComboBox()
        self.modelCombo.setMinimumWidth(200)
        self.modelCard = SettingCard(
            FluentIcon.ROBOT,
            t("model"),
            "",
            self.modelGroup
        )
        self.modelCard.hBoxLayout.addWidget(self.modelCombo, 0, Qt.AlignmentFlag.AlignRight)
        self.modelCard.hBoxLayout.addSpacing(16)

        self.inferenceBackendCombo = ComboBox()
        self.inferenceBackendCombo.addItems(["Auto", "TensorRT", "DirectML", "CPU"])
        self.inferenceBackendCombo.setMinimumWidth(150)
        self.inferenceBackendCard = SettingCard(
            FluentIcon.COMMAND_PROMPT,
            t("inference_backend"),
            "",
            self.modelGroup
        )
        self.inferenceBackendCard.hBoxLayout.addWidget(self.inferenceBackendCombo, 0, Qt.AlignmentFlag.AlignRight)
        self.inferenceBackendCard.hBoxLayout.addSpacing(16)

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

        self.fovCard = SliderSpinCard(
            FluentIcon.ZOOM,
            t("fov_size"),
            50, 500,
            description="",
            parent=self.fovGroup
        )

        self.fovFollowCard = SwitchSettingCard(
            FluentIcon.MOVE,
            t("fov_follow_mouse"),
            "",
            parent=self.fovGroup
        )

        self.fovCircleCard = SwitchSettingCard(
            FluentIcon.REMOVE,
            t("fov_circle_filter", "Circular FOV Filter"),
            t("fov_circle_filter_desc", "Only track targets inside the FOV circle, not the full square region"),
            parent=self.fovGroup
        )

        self.detectRangeCard = SliderSpinCard(
            FluentIcon.FULL_SCREEN,
            t("detect_range_size"),
            100, 1080,
            description=t("detect_range_note"),
            parent=self.fovGroup
        )

        # === General Parameters ===
        self.generalGroup = SettingCardGroup(t("general_params"), self.scrollWidget)

        self.detectIntervalCard = SliderSpinCard(
            FluentIcon.SPEED_HIGH,
            t("detect_interval"),
            1, 100,
            suffix="ms",
            description="",
            parent=self.generalGroup
        )

        self.screenshotIntervalCard = SliderSpinCard(
            FluentIcon.CAMERA,
            t("screenshot_interval"),
            1, 100,
            suffix="ms",
            description="",
            parent=self.generalGroup
        )

        self.autoMatchFpsCard = SwitchSettingCard(
            FluentIcon.SYNC,
            t("auto_match_fps_label", "Sync Detection & Capture Interval"),
            t("auto_match_fps_desc", "Lock capture interval to detection interval"),
            parent=self.generalGroup
        )

        self.confidenceCard = SliderSpinCard(
            FluentIcon.CERTIFICATE,
            t("min_confidence"),
            1, 100,
            suffix="%",
            description="",
            parent=self.generalGroup
        )

        self.semanticFilterCard = SwitchSettingCard(
            FluentIcon.FILTER,
            t("semantic_filter_enabled", "Semantic FP Filter"),
            t("semantic_filter_desc", "Discard trees, vehicles, and HUD elements by class name and geometry"),
            parent=self.generalGroup
        )

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

        self.screenshotMethodCombo = ComboBox()
        self.screenshotMethodCombo.addItems(["mss", "dxcam", "uvc", "ndi"])
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
        self.uvcCaptureMethodCombo.addItems(["msmf", "dshow", "auto", "any"])
        self.uvcCaptureMethodCombo.setMinimumWidth(140)
        self.uvcCaptureMethodCard = SettingCard(
            FluentIcon.CAMERA,
            "UVC Capture Method",
            "msmf recommended for 1080p60 on Windows 10/11",
            self.generalGroup
        )
        self.uvcCaptureMethodCard.hBoxLayout.addWidget(self.uvcCaptureMethodCombo, 0, Qt.AlignmentFlag.AlignRight)
        self.uvcCaptureMethodCard.hBoxLayout.addSpacing(16)

        self.uvcPreviewCard = SwitchSettingCard(
            FluentIcon.VIEW,
            "Capture Preview Window",
            "",
            parent=self.generalGroup
        )

        self.previewCropCard = SwitchSettingCard(
            FluentIcon.ZOOM_IN,
            t("preview_crop_label"),
            t("preview_crop_desc"),
            parent=self.generalGroup
        )

        self.uvcPreviewScaleCombo = ComboBox()
        self.uvcPreviewScaleCombo.addItems(["scale_to_fit", "scale_to_canvas", "fit_to_screen"])
        self.uvcPreviewScaleCombo.setMinimumWidth(170)
        self.uvcPreviewScaleCard = SettingCard(
            FluentIcon.FULL_SCREEN,
            "Capture Preview Scale Mode",
            "",
            self.generalGroup
        )
        self.uvcPreviewScaleCard.hBoxLayout.addWidget(self.uvcPreviewScaleCombo, 0, Qt.AlignmentFlag.AlignRight)
        self.uvcPreviewScaleCard.hBoxLayout.addSpacing(16)

        self.ndiSourceCombo = ComboBox()
        self.ndiSourceCombo.setMinimumWidth(360)
        self.ndiSourceCard = SettingCard(
            FluentIcon.CAMERA,
            "NDI Stream",
            "Select the NDI source to capture",
            self.generalGroup
        )
        self.ndiSourceCard.hBoxLayout.addWidget(self.ndiSourceCombo, 0, Qt.AlignmentFlag.AlignRight)
        self.ndiSourceCard.hBoxLayout.addSpacing(16)

        self.ndiRefreshBtn = PushButton(t("refresh"))
        self.ndiRefreshBtn.setFixedWidth(80)
        self.ndiRefreshCard = SettingCard(
            FluentIcon.SYNC,
            "Refresh NDI Streams",
            "",
            self.generalGroup
        )
        self.ndiRefreshCard.hBoxLayout.addWidget(self.ndiRefreshBtn, 0, Qt.AlignmentFlag.AlignRight)
        self.ndiRefreshCard.hBoxLayout.addSpacing(16)

        self.ndiBandwidthCombo = ComboBox()
        self.ndiBandwidthCombo.addItems(["Highest", "Lowest"])
        self.ndiBandwidthCombo.setMinimumWidth(120)
        self.ndiBandwidthCard = SettingCard(
            FluentIcon.SPEED_HIGH,
            "NDI Bandwidth",
            "Receive bandwidth for the NDI stream",
            self.generalGroup
        )
        self.ndiBandwidthCard.hBoxLayout.addWidget(self.ndiBandwidthCombo, 0, Qt.AlignmentFlag.AlignRight)
        self.ndiBandwidthCard.hBoxLayout.addSpacing(16)

        self.ndiPreResizeCard = SwitchSettingCard(
            FluentIcon.ZOOM,
            "NDI Pre-Resize",
            "Resize NDI frames to model input size in the capture thread — reduces preprocessing load and improves inference FPS",
            parent=self.generalGroup
        )

        self.alwaysAimCard = SwitchSettingCard(
            FluentIcon.FINGERPRINT,
            t("always_aim"),
            "",
            parent=self.generalGroup
        )

        self.keepDetectingCard = SwitchSettingCard(
            FluentIcon.UPDATE,
            t("keep_detecting"),
            "",
            parent=self.generalGroup
        )

        self.idleDetectEnableCard = SwitchSettingCard(
            FluentIcon.SPEED_MEDIUM,
            t("idle_detect_enabled"),
            "",
            parent=self.generalGroup
        )

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

        # === MAKCU Settings ===
        self.makcuGroup = SettingCardGroup("MAKCU", self.scrollWidget)

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

        self.makcuBaudCombo = ComboBox()
        self.makcuBaudCombo.addItems(["115200", "4000000"])
        self.makcuBaudCombo.setMinimumWidth(120)
        self.makcuBaudCard = SettingCard(
            FluentIcon.SPEED_HIGH,
            t("makcu_baud_rate", "Baud Rate"),
            t("makcu_baud_rate_desc", "4000000 = 4 Mbaud — ~35× faster than default, lowest serial latency"),
            self.makcuGroup
        )
        self.makcuBaudCard.hBoxLayout.addWidget(self.makcuBaudCombo, 0, Qt.AlignmentFlag.AlignRight)
        self.makcuBaudCard.hBoxLayout.addSpacing(16)

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

        self.makcuAimButtonCombo = ComboBox()
        self.makcuAimButtonCombo.addItems(["LMB", "RMB", "Off"])
        self.makcuAimButtonCombo.setMinimumWidth(100)
        self.makcuAimButtonCard = SettingCard(
            FluentIcon.FINGERPRINT,
            t("makcu_aim_button", "Aim Trigger Button"),
            t("makcu_aim_button_desc", "Hold this button to aim/track; inference always runs"),
            self.makcuGroup
        )
        self.makcuAimButtonCard.hBoxLayout.addWidget(self.makcuAimButtonCombo, 0, Qt.AlignmentFlag.AlignRight)
        self.makcuAimButtonCard.hBoxLayout.addSpacing(16)

        # === Xbox 360 Settings ===
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

        # === Inference Performance ===
        self.inferPerfGroup = SettingCardGroup(t("inference_performance", "Inference Performance"), self.scrollWidget)

        self.skipLetterboxCard = SwitchSettingCard(
            FluentIcon.SPEED_HIGH,
            t("skip_letterbox_label"),
            t("skip_letterbox_desc"),
            parent=self.inferPerfGroup
        )

        self.cudaIoBindingCard = SwitchSettingCard(
            FluentIcon.COPY,
            t("cuda_io_binding", "CUDA IO Binding"),
            t("cuda_io_binding_desc", "Zero-copy GPU inference. Effective only with CUDA or TensorRT backend."),
            parent=self.inferPerfGroup
        )


    # ──────────────────────────────────────────────
    # Layout
    # ──────────────────────────────────────────────

    def _initLayout(self):
        self.modelGroup.addSettingCard(self.modelCard)
        self.modelGroup.addSettingCard(self.inferenceBackendCard)
        self.modelGroup.addSettingCard(self.openModelFolderCard)
        self.addContent(self.modelGroup)

        self.fovGroup.addSettingCard(self.fovCard)
        self.fovGroup.addSettingCard(self.fovFollowCard)
        self.fovGroup.addSettingCard(self.fovCircleCard)
        self.fovGroup.addSettingCard(self.detectRangeCard)
        self.fovGroup.addSettingCard(self.screenshotMethodCard)
        self.addContent(self.fovGroup)

        self.generalGroup.addSettingCard(self.detectIntervalCard)
        self.generalGroup.addSettingCard(self.screenshotIntervalCard)
        self.generalGroup.addSettingCard(self.autoMatchFpsCard)
        self.generalGroup.addSettingCard(self.confidenceCard)
        self.generalGroup.addSettingCard(self.semanticFilterCard)
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
        self.generalGroup.addSettingCard(self.previewCropCard)
        self.generalGroup.addSettingCard(self.uvcPreviewScaleCard)
        self.generalGroup.addSettingCard(self.ndiSourceCard)
        self.generalGroup.addSettingCard(self.ndiRefreshCard)
        self.generalGroup.addSettingCard(self.ndiBandwidthCard)
        self.generalGroup.addSettingCard(self.ndiPreResizeCard)
        self.generalGroup.addSettingCard(self.alwaysAimCard)
        self.generalGroup.addSettingCard(self.keepDetectingCard)
        self.generalGroup.addSettingCard(self.idleDetectEnableCard)
        self.generalGroup.addSettingCard(self.idleDetectIntervalCard)
        self.generalGroup.addSettingCard(self.singleTargetCard)
        self.addContent(self.generalGroup)

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

        self.makcuGroup.addSettingCard(self.makcuComPortCard)
        self.makcuGroup.addSettingCard(self.makcuBaudCard)
        self.makcuGroup.addSettingCard(self.makcuConnectionCard)
        self.makcuGroup.addSettingCard(self.makcuConnectCard)
        self.makcuGroup.addSettingCard(self.makcuAimButtonCard)
        self.addContent(self.makcuGroup)
        self.makcuGroup.setVisible(False)

        self.xboxGroup.addSettingCard(self.xboxSensitivityCard)
        self.xboxGroup.addSettingCard(self.xboxDeadzoneCard)
        self.xboxGroup.addSettingCard(self.xboxConnectionCard)
        self.xboxGroup.addSettingCard(self.xboxConnectCard)
        self.addContent(self.xboxGroup)
        self.xboxGroup.setVisible(False)

        self.inferPerfGroup.addSettingCard(self.skipLetterboxCard)
        self.inferPerfGroup.addSettingCard(self.cudaIoBindingCard)
        self.addContent(self.inferPerfGroup)

        self.scrollLayout.addStretch(1)

    # ──────────────────────────────────────────────
    # Signal connections
    # ──────────────────────────────────────────────

    def _connectSignals(self):
        self.modelCombo.currentTextChanged.connect(self._onModelChanged)
        self.inferenceBackendCombo.currentTextChanged.connect(self._onInferenceBackendChanged)
        self.openModelFolderBtn.clicked.connect(self._openModelFolder)

        self.fovCard.valueChanged.connect(self._onFovChanged)
        self.fovFollowCard.checkedChanged.connect(self._onFovFollowChanged)
        self.fovCircleCard.checkedChanged.connect(self._onFovCircleChanged)
        self.detectRangeCard.valueChanged.connect(self._onDetectRangeChanged)

        self.detectIntervalCard.valueChanged.connect(self._onDetectIntervalChanged)
        self.screenshotIntervalCard.valueChanged.connect(self._onScreenshotIntervalChanged)
        self.autoMatchFpsCard.checkedChanged.connect(self._onAutoMatchFpsChanged)
        self.confidenceCard.valueChanged.connect(self._onConfidenceChanged)
        self.semanticFilterCard.checkedChanged.connect(self._onSemanticFilterChanged)
        self.aimPartCombo.currentIndexChanged.connect(self._onAimPartChanged)
        self.mouseMoveCombo.currentTextChanged.connect(self._onMouseMoveChanged)
        self.screenshotMethodCombo.currentTextChanged.connect(self._onScreenshotMethodChanged)
        self.uvcDeviceCard.valueChanged.connect(self._onUvcDeviceChanged)
        self.uvcResolutionCombo.currentTextChanged.connect(self._onUvcResolutionChanged)
        self.uvcRefreshResolutionBtn.clicked.connect(self._refreshUvcResolutions)
        self.uvcFpsCard.valueChanged.connect(self._onUvcFpsChanged)
        self.uvcCaptureMethodCombo.currentTextChanged.connect(self._onUvcCaptureMethodChanged)
        self.uvcPreviewCard.checkedChanged.connect(self._onUvcPreviewChanged)
        self.previewCropCard.checkedChanged.connect(self._onPreviewCropChanged)
        self.uvcPreviewScaleCombo.currentTextChanged.connect(self._onUvcPreviewScaleModeChanged)
        self.ndiSourceCombo.currentTextChanged.connect(self._onNdiSourceChanged)
        self.ndiRefreshBtn.clicked.connect(self._refreshNdiSources)
        self.ndiBandwidthCombo.currentTextChanged.connect(self._onNdiBandwidthChanged)
        self.ndiPreResizeCard.checkedChanged.connect(self._onNdiPreResizeChanged)
        self.alwaysAimCard.checkedChanged.connect(self._onAlwaysAimChanged)
        self.keepDetectingCard.checkedChanged.connect(self._onKeepDetectingChanged)
        self.idleDetectEnableCard.checkedChanged.connect(self._onIdleDetectEnableChanged)
        self.idleDetectIntervalCard.valueChanged.connect(self._onIdleDetectIntervalChanged)
        self.singleTargetCard.checkedChanged.connect(self._onSingleTargetChanged)

        self.comRefreshBtn.clicked.connect(self._refreshComPorts)
        self.comPortCombo.currentTextChanged.connect(self._onComPortChanged)
        self.arduinoConnectBtn.clicked.connect(self._onArduinoConnectToggle)
        self.guideBtn.clicked.connect(self._onOpenGuide)
        self.spoofBtn.clicked.connect(self._onSpoofDevice)
        self.verifySpoofBtn.clicked.connect(self._onVerifySpoof)
        self.testHeartBtn.clicked.connect(self._onTestHeart)
        self.arduinoBaudCombo.currentTextChanged.connect(self._onArduinoBaudChanged)

        self.makcuComRefreshBtn.clicked.connect(self._refreshMakcuComPorts)
        self.makcuComPortCombo.currentTextChanged.connect(self._onMakcuComPortChanged)
        self.makcuConnectBtn.clicked.connect(self._onMakcuConnectToggle)
        self.makcuAimButtonCombo.currentTextChanged.connect(self._onMakcuAimButtonChanged)
        self.makcuBaudCombo.currentTextChanged.connect(self._onMakcuBaudChanged)

        self.xboxSensitivityCard.valueChanged.connect(self._onXboxSensitivityChanged)
        self.xboxDeadzoneCard.valueChanged.connect(self._onXboxDeadzoneChanged)
        self.xboxConnectBtn.clicked.connect(self._onXboxConnectToggle)

        self.skipLetterboxCard.checkedChanged.connect(self._onSkipLetterboxChanged)
        self.cudaIoBindingCard.checkedChanged.connect(self._onCudaIoBindingChanged)

    # ──────────────────────────────────────────────
    # Config load
    # ──────────────────────────────────────────────

    def _loadFromConfig(self):
        if not self._config:
            return
        self._isLoadingConfig = True
        try:
            self.modelCombo.blockSignals(True)
            self._refreshModelList()
            model_name = os.path.basename(self._config.model_path or "")
            idx = -1
            for i in range(self.modelCombo.count()):
                if self.modelCombo.itemText(i).lower() == model_name.lower():
                    idx = i
                    break
            if idx >= 0:
                self.modelCombo.setCurrentIndex(idx)
            elif self.modelCombo.count() > 0:
                default_name = "ApexLegendsOrbeet_15k.onnx"
                default_idx = -1
                for i in range(self.modelCombo.count()):
                    if self.modelCombo.itemText(i).lower() == default_name.lower():
                        default_idx = i
                        break
                pick = default_idx if default_idx >= 0 else 0
                self.modelCombo.setCurrentIndex(pick)
                if self._config:
                    self._config.model_path = "Model/" + self.modelCombo.itemText(pick)
            self.modelCombo.blockSignals(False)

            backend_map = {
                "auto": "Auto",
                "tensorrt": "TensorRT",
                "cuda": "TensorRT",
                "directml": "DirectML",
                "cpu": "CPU",
            }
            self.inferenceBackendCombo.blockSignals(True)
            backend_text = backend_map.get(getattr(self._config, "inference_backend", "auto").lower(), "Auto")
            self.inferenceBackendCombo.setCurrentText(backend_text)
            self.inferenceBackendCombo.blockSignals(False)
            self._updateInferenceBackendSubtitle()

            self.fovCard.setValue(self._config.fov_size)
            self.fovFollowCard.setChecked(self._config.fov_follow_mouse)
            self.fovCircleCard.setChecked(bool(getattr(self._config, 'fov_circle_filter_enabled', False)))
            self.detectRangeCard.setValue(self._config.detect_range_size)

            interval_ms = int(self._config.detect_interval * 1000)
            self.detectIntervalCard.setValue(interval_ms)
            screenshot_interval_ms = int(getattr(self._config, 'screenshot_interval', self._config.detect_interval) * 1000)
            self.screenshotIntervalCard.setValue(screenshot_interval_ms)
            _auto_match = bool(getattr(self._config, 'auto_match_fps', False))
            self.autoMatchFpsCard.setChecked(_auto_match)
            self.screenshotIntervalCard.setEnabled(not _auto_match)
            confidence_pct = int(self._config.min_confidence * 100)
            self.confidenceCard.setValue(confidence_pct)
            self.semanticFilterCard.setChecked(bool(getattr(self._config, 'detect_semantic_filter_enabled', False)))

            aim_parts = ["head", "body", "both"]
            if self._config.aim_part in aim_parts:
                self.aimPartCombo.setCurrentIndex(aim_parts.index(self._config.aim_part))

            mouse_methods = ["ddxoft", "mouse_event", "sendinput", "arduino", "makcu", "xbox"]
            if self._config.mouse_move_method in mouse_methods:
                self.mouseMoveCombo.setCurrentIndex(mouse_methods.index(self._config.mouse_move_method))

            screenshot_methods = ["mss", "dxcam", "uvc", "ndi"]
            screenshot_method = getattr(self._config, 'screenshot_method', 'mss')
            if screenshot_method in screenshot_methods:
                self.screenshotMethodCombo.setCurrentIndex(screenshot_methods.index(screenshot_method))

            self.uvcDeviceCard.setValue(int(getattr(self._config, 'uvc_device_index', 0)))
            self.uvcCaptureMethodCombo.setCurrentText(str(getattr(self._config, 'uvc_capture_method', 'msmf')))
            resolution_text = str(getattr(self._config, 'uvc_resolution',
                f"{getattr(self._config, 'uvc_width', self._config.width)}x{getattr(self._config, 'uvc_height', self._config.height)}"))
            if screenshot_method == 'uvc':
                self._refreshUvcResolutions()
                idx = self.uvcResolutionCombo.findText(resolution_text)
                if idx < 0:
                    self.uvcResolutionCombo.addItem(resolution_text)
                    idx = self.uvcResolutionCombo.findText(resolution_text)
                if idx >= 0:
                    self.uvcResolutionCombo.setCurrentIndex(idx)
            else:
                self.uvcResolutionCombo.blockSignals(True)
                self.uvcResolutionCombo.clear()
                self.uvcResolutionCombo.addItem(resolution_text)
                self.uvcResolutionCombo.blockSignals(False)
            self.uvcFpsCard.setValue(int(getattr(self._config, 'uvc_fps', 60)))
            self.uvcPreviewCard.setChecked(bool(getattr(self._config, 'uvc_show_window', True)))
            self.previewCropCard.setChecked(bool(getattr(self._config, 'preview_crop_to_detection', False)))
            self.uvcPreviewScaleCombo.setCurrentText(str(getattr(self._config, 'uvc_preview_scale_mode', 'scale_to_fit')))

            ndi_source = str(getattr(self._config, 'ndi_source_name', '')).strip()
            if screenshot_method == 'ndi':
                self._refreshNdiSources()
                if ndi_source:
                    idx = self.ndiSourceCombo.findText(ndi_source)
                    if idx < 0:
                        self.ndiSourceCombo.addItem(ndi_source)
                        idx = self.ndiSourceCombo.findText(ndi_source)
                    if idx >= 0:
                        self.ndiSourceCombo.setCurrentIndex(idx)
            else:
                self.ndiSourceCombo.blockSignals(True)
                self.ndiSourceCombo.clear()
                if ndi_source:
                    self.ndiSourceCombo.addItem(ndi_source)
                self.ndiSourceCombo.blockSignals(False)
            ndi_bw = str(getattr(self._config, 'ndi_bandwidth', 'highest')).capitalize()
            self.ndiBandwidthCombo.setCurrentText(ndi_bw if ndi_bw in ("Highest", "Lowest") else "Highest")
            self.ndiPreResizeCard.setChecked(bool(getattr(self._config, 'ndi_pre_resize', True)))
            self._updateCaptureControlsVisibility(screenshot_method)

            self.alwaysAimCard.setChecked(getattr(self._config, 'always_aim', False))
            self.keepDetectingCard.setChecked(getattr(self._config, 'keep_detecting', False))
            self.idleDetectEnableCard.setChecked(getattr(self._config, 'idle_detect_enabled', True))
            idle_ms = int(getattr(self._config, 'idle_detect_interval', 0.05) * 1000)
            self.idleDetectIntervalCard.setValue(max(5, min(500, idle_ms)))
            self.singleTargetCard.setChecked(getattr(self._config, 'single_target_mode', False))

            self._updateMethodGroupVisibility(self._config.mouse_move_method)

            if self._config.arduino_com_port:
                idx = self.comPortCombo.findText(self._config.arduino_com_port)
                if idx >= 0:
                    self.comPortCombo.setCurrentIndex(idx)

            if getattr(self._config, 'makcu_com_port', ''):
                idx = self.makcuComPortCombo.findText(self._config.makcu_com_port)
                if idx >= 0:
                    self.makcuComPortCombo.setCurrentIndex(idx)
            _aim_map = {"lmb": "LMB", "rmb": "RMB", "off": "Off"}
            _aim_btn = _aim_map.get(str(getattr(self._config, 'makcu_aim_button', 'lmb')).lower(), "LMB")
            self.makcuAimButtonCombo.setCurrentText(_aim_btn)

            makcu_baud = str(getattr(self._config, 'makcu_baud_rate', 115200))
            if self.makcuBaudCombo.findText(makcu_baud) < 0:
                makcu_baud = "115200"
            self.makcuBaudCombo.setCurrentText(makcu_baud)

            arduino_baud = str(getattr(self._config, 'arduino_baud_rate', 115200))
            if self.arduinoBaudCombo.findText(arduino_baud) < 0:
                arduino_baud = "115200"
            self.arduinoBaudCombo.setCurrentText(arduino_baud)

            self.xboxSensitivityCard.setValue(int(getattr(self._config, 'xbox_sensitivity', 1.0) * 100))
            self.xboxDeadzoneCard.setValue(int(getattr(self._config, 'xbox_deadzone', 0.05) * 100))
            self._updateXboxConnectionStatus()

            self.skipLetterboxCard.setChecked(bool(getattr(self._config, 'skip_letterbox', False)))
            self.cudaIoBindingCard.setChecked(bool(getattr(self._config, 'cuda_io_binding_enabled', False)))

        finally:
            self._isLoadingConfig = False

    # ──────────────────────────────────────────────
    # Helper methods
    # ──────────────────────────────────────────────

    def _refreshModelList(self):
        self.modelCombo.clear()
        src_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        project_root = os.path.dirname(src_dir)
        model_dir = os.path.join(project_root, "Model")
        if os.path.exists(model_dir):
            models = glob.glob(os.path.join(model_dir, "*.onnx"))
            for m in models:
                self.modelCombo.addItem(os.path.basename(m))

    def _openModelFolder(self):
        src_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        project_root = os.path.dirname(src_dir)
        model_dir = os.path.join(project_root, "Model")
        if os.path.exists(model_dir):
            os.startfile(model_dir)

    def _refreshComPorts(self):
        self.comPortCombo.clear()
        self.comPortCombo.addItem(t("no_com_port"))
        try:
            import serial.tools.list_ports
            ports = serial.tools.list_ports.comports()
            for port in ports:
                self.comPortCombo.addItem(port.device)
        except ImportError:
            pass

    def _refreshMakcuComPorts(self):
        self.makcuComPortCombo.clear()
        self.makcuComPortCombo.addItem(t("no_com_port"))
        try:
            import serial.tools.list_ports
            ports = serial.tools.list_ports.comports()
            for port in ports:
                self.makcuComPortCombo.addItem(port.device)
        except ImportError:
            pass

    def _updateMethodGroupVisibility(self, method):
        self.arduinoGroup.setVisible(method == "arduino")
        self.makcuGroup.setVisible(method == "makcu")
        self.xboxGroup.setVisible(method == "xbox")

    def _updateCaptureControlsVisibility(self, screenshot_method):
        is_uvc = (screenshot_method == "uvc")
        is_ndi = (screenshot_method == "ndi")
        self.uvcDeviceCard.setVisible(is_uvc)
        self.uvcResolutionCard.setVisible(is_uvc)
        self.uvcRefreshResolutionCard.setVisible(is_uvc)
        self.uvcFpsCard.setVisible(is_uvc)
        self.uvcCaptureMethodCard.setVisible(is_uvc)
        self.uvcPreviewCard.setVisible(is_uvc or is_ndi)
        self.previewCropCard.setVisible(is_uvc or is_ndi)
        self.uvcPreviewScaleCard.setVisible(is_uvc or is_ndi)
        self.ndiSourceCard.setVisible(is_ndi)
        self.ndiRefreshCard.setVisible(is_ndi)
        self.ndiBandwidthCard.setVisible(is_ndi)
        is_external = is_ndi or is_uvc
        self.fovFollowCard.setVisible(not is_external)
        if is_external and self._config:
            self._config.fov_follow_mouse = False
            self.fovFollowCard.setChecked(False)
        self.ndiPreResizeCard.setVisible(is_ndi)

    def _updateInferenceBackendSubtitle(self):
        if not hasattr(self, "inferenceBackendCard"):
            return
        provider = getattr(self._config, "current_provider", "Unknown") if self._config else "Unknown"
        self.inferenceBackendCard.contentLabel.setText(
            f"{t('inference_backend_desc')} ({t('inference_backend_current')}: {provider})"
        )

    def _refreshUvcResolutions(self):
        if not self._config:
            return
        try:
            from core.screen_capture import list_supported_uvc_resolutions
            resolutions = list_supported_uvc_resolutions(
                int(getattr(self._config, 'uvc_device_index', 0)),
                str(getattr(self._config, 'uvc_capture_method', 'msmf')),
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

    def _refreshNdiSources(self):
        if not self._config:
            return
        try:
            from core.screen_capture import list_available_ndi_source_details
            source_details = list_available_ndi_source_details()
        except Exception:
            source_details = []
        current_name = self.ndiSourceCombo.currentData()
        if not isinstance(current_name, str):
            current_name = self.ndiSourceCombo.currentText().strip()
        configured = str(getattr(self._config, 'ndi_source_name', '')).strip()
        self.ndiSourceCombo.blockSignals(True)
        self.ndiSourceCombo.clear()
        known_names: set = set()
        for detail in source_details:
            name = str(detail.get('name', '')).strip()
            if not name:
                continue
            label = str(detail.get('label', '')).strip() or name
            self.ndiSourceCombo.addItem(label, name)
            known_names.add(name)
        if configured and configured not in known_names:
            self.ndiSourceCombo.addItem(f"{configured} (Unknown @ Unknown fps)", configured)
        fallback_name = configured or (current_name if isinstance(current_name, str) else '')
        if fallback_name:
            for i in range(self.ndiSourceCombo.count()):
                data = self.ndiSourceCombo.itemData(i)
                if isinstance(data, str) and data == fallback_name:
                    self.ndiSourceCombo.setCurrentIndex(i)
                    break
        self.ndiSourceCombo.blockSignals(False)

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

    def _ensureTrtInstalled(self) -> None:
        self._trt_installer_launched = False
        try:
            import onnxruntime as _ort
            if "TensorrtExecutionProvider" in _ort.get_available_providers():
                return
        except Exception:
            pass
        localappdata = os.environ.get("LOCALAPPDATA", "")
        if localappdata:
            trt_libs = os.path.join(localappdata, "AxiomAI", "site-packages", "tensorrt_libs")
            if os.path.isdir(trt_libs):
                for name in os.listdir(trt_libs):
                    if name.lower().startswith("nvinfer") and name.lower().endswith(".dll"):
                        return
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
        bat_path = os.path.join(project_root, "Install TensorRT.bat")
        if not os.path.exists(bat_path):
            InfoBar.warning("TensorRT installer not found", f"Expected: {bat_path}",
                            duration=6000, isClosable=True, position=InfoBarPosition.TOP, parent=self)
            return
        subprocess.Popen([bat_path], shell=True)
        self._trt_installer_launched = True

    def _startRestartCountdown(self) -> None:
        from PyQt6.QtCore import QTimer
        from qfluentwidgets import BodyLabel as _BL
        self._restartCountdown = 5
        self._restartCountdownBar = InfoBar.info(
            t("restart_required", "Restart Required"),
            f"Restarting in {self._restartCountdown}s…",
            duration=-1, isClosable=True, position=InfoBarPosition.TOP, parent=self,
        )
        self._restartTimer = QTimer(self)
        self._restartTimer.timeout.connect(self._onRestartTick)
        self._restartTimer.start(1000)

    def _onRestartTick(self) -> None:
        from qfluentwidgets import BodyLabel as _BL
        self._restartCountdown -= 1
        if self._restartCountdown <= 0:
            self._restartTimer.stop()
            bar = getattr(self, "_restartCountdownBar", None)
            if bar:
                bar.close()
            self._restartApp()
        else:
            bar = getattr(self, "_restartCountdownBar", None)
            if bar:
                for lbl in bar.findChildren(_BL):
                    lbl.setText(f"Restarting in {self._restartCountdown}s…")
                    break

    def _restartApp(self) -> None:
        from core.config import save_config
        if self._config:
            save_config(self._config)
        subprocess.Popen([sys.executable] + sys.argv)
        QApplication.instance().quit()

    # ──────────────────────────────────────────────
    # Callbacks
    # ──────────────────────────────────────────────

    def _onModelChanged(self, text):
        if self._config and text:
            self._config.model_path = os.path.join("Model", text)

    def _onInferenceBackendChanged(self, text):
        if not self._config:
            return
        backend_map = {"Auto": "auto", "TensorRT": "tensorrt", "DirectML": "directml", "CPU": "cpu"}
        prev_backend = getattr(self._config, "inference_backend", "auto")
        selected_backend = backend_map.get(text, "auto")
        if prev_backend != selected_backend:
            self._config.inference_backend = selected_backend

        # Auto-enable CUDA IO Binding when TensorRT is selected
        if selected_backend == "tensorrt" and not self._isLoadingConfig:
            self.cudaIoBindingCard.setChecked(True)
            self._config.cuda_io_binding_enabled = True

        if not self._isLoadingConfig and (selected_backend == "directml" or prev_backend == "directml"):
            if selected_backend == "tensorrt":
                self._ensureTrtInstalled()
                if getattr(self, "_trt_installer_launched", False):
                    InfoBar.info("TensorRT Installer Launched",
                                 "Restart the app after installation completes.",
                                 duration=6000, isClosable=True,
                                 position=InfoBarPosition.TOP, parent=self)
                    self._updateInferenceBackendSubtitle()
                    return
            self._startRestartCountdown()
            return
        self._updateInferenceBackendSubtitle()

    def _onFovChanged(self, value):
        if self._config:
            self._config.fov_size = value

    def _onFovFollowChanged(self, checked):
        if self._config:
            self._config.fov_follow_mouse = checked

    def _onFovCircleChanged(self, checked):
        if self._config:
            self._config.fov_circle_filter_enabled = bool(checked)

    def _onDetectRangeChanged(self, value):
        if self._config:
            self._config.detect_range_size = value

    def _onDetectIntervalChanged(self, value):
        if self._config:
            self._config.detect_interval = value / 1000.0
            if getattr(self._config, 'auto_match_fps', False):
                self._config.screenshot_interval = self._config.detect_interval
                self.screenshotIntervalCard.setValue(value)

    def _onScreenshotIntervalChanged(self, value):
        if self._config:
            self._config.screenshot_interval = value / 1000.0

    def _onAutoMatchFpsChanged(self, checked):
        if self._config:
            self._config.auto_match_fps = bool(checked)
            self.screenshotIntervalCard.setEnabled(not checked)
            if checked:
                self._config.screenshot_interval = self._config.detect_interval
                self.screenshotIntervalCard.setValue(int(self._config.detect_interval * 1000))

    def _onConfidenceChanged(self, value):
        if self._config:
            self._config.min_confidence = value / 100.0

    def _onSemanticFilterChanged(self, checked):
        if self._config:
            self._config.detect_semantic_filter_enabled = bool(checked)

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

    def _onScreenshotMethodChanged(self, text):
        if self._config:
            self._config.screenshot_method = text
        if str(text).strip().lower() == "ndi" and not self._isLoadingConfig:
            has_ran = bool(getattr(self._config, "ndi_installer_ran_once", False))
            if not has_ran:
                if self._runLocalInstallerScript("install_cyndilib.py", "NDI"):
                    self._config.ndi_installer_ran_once = True
        self._updateCaptureControlsVisibility(text)
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

    def _onPreviewCropChanged(self, checked):
        if self._config:
            self._config.preview_crop_to_detection = bool(checked)

    def _onUvcPreviewScaleModeChanged(self, text):
        if self._config:
            self._config.uvc_preview_scale_mode = str(text)

    def _onNdiBandwidthChanged(self, text):
        if self._config:
            self._config.ndi_bandwidth = str(text).lower()

    def _onNdiPreResizeChanged(self, checked):
        if self._config:
            self._config.ndi_pre_resize = bool(checked)

    def _onNdiSourceChanged(self, text):
        if not self._config:
            return
        source_name = self.ndiSourceCombo.currentData()
        if not isinstance(source_name, str) or not source_name.strip():
            source_name = str(text).strip()
        self._config.ndi_source_name = source_name.strip()

    def _onAlwaysAimChanged(self, checked):
        if self._config:
            self._config.always_aim = checked
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

    def _onMakcuConnectToggle(self):
        try:
            from win_utils import is_makcu_connected, connect_makcu, disconnect_makcu
            if is_makcu_connected():
                disconnect_makcu()
            else:
                com_port = self.makcuComPortCombo.currentText()
                if not com_port or com_port == t("no_com_port"):
                    QMessageBox.warning(self, t("config_error"), t("no_com_port"))
                    return
                baud = int(getattr(self._config, 'makcu_baud_rate', 115200)) if self._config else 115200
                success = connect_makcu(com_port, baud)
                if not success:
                    QMessageBox.warning(self, t("config_error"),
                                        f"MAKCU {t('disconnected')}: {com_port}")
            self._updateMakcuConnectionStatus()
        except ImportError:
            QMessageBox.warning(self, t("config_error"), "pyserial not installed.\npip install pyserial")

    def _updateMakcuConnectionStatus(self):
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

    def _onMakcuAimButtonChanged(self, text):
        if self._config:
            self._config.makcu_aim_button = text.lower()

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

    def _onArduinoBaudChanged(self, text):
        if self._config and not self._isLoadingConfig:
            try:
                self._config.arduino_baud_rate = int(text)
            except ValueError:
                pass

    def _onMakcuBaudChanged(self, text):
        if self._config and not self._isLoadingConfig:
            try:
                self._config.makcu_baud_rate = int(text)
            except ValueError:
                pass

    def _onSkipLetterboxChanged(self, checked):
        if self._config:
            self._config.skip_letterbox = bool(checked)

    def _onCudaIoBindingChanged(self, checked):
        if self._config:
            self._config.cuda_io_binding_enabled = bool(checked)

    # ──────────────────────────────────────────────
    # Retranslate
    # ──────────────────────────────────────────────

    def retranslateUi(self):
        super().retranslateUi()

        self.modelGroup.titleLabel.setText(t("model_settings"))
        self.fovGroup.titleLabel.setText(t("fov_and_detect_range"))
        self.generalGroup.titleLabel.setText(t("general_params"))
        self.inferPerfGroup.titleLabel.setText(t("inference_performance", "Inference Performance"))

        self.modelCard.titleLabel.setText(t("model"))
        self.inferenceBackendCard.titleLabel.setText(t("inference_backend"))
        self._updateInferenceBackendSubtitle()
        self.openModelFolderCard.titleLabel.setText(t("open_model_folder"))
        self.openModelFolderBtn.setText(t("open_model_folder"))

        self.fovCard.titleLabel.setText(t("fov_size"))
        self.fovFollowCard.titleLabel.setText(t("fov_follow_mouse"))
        self.detectRangeCard.titleLabel.setText(t("detect_range_size"))
        self.detectRangeCard.contentLabel.setText(t("detect_range_note"))

        self.detectIntervalCard.titleLabel.setText(t("detect_interval"))
        self.screenshotIntervalCard.titleLabel.setText(t("screenshot_interval"))
        self.autoMatchFpsCard.titleLabel.setText(t("auto_match_fps_label", "Sync Detection & Capture Interval"))
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
        self.uvcPreviewCard.titleLabel.setText("Capture Preview Window")
        self.previewCropCard.titleLabel.setText(t("preview_crop_label"))
        self.previewCropCard.contentLabel.setText(t("preview_crop_desc"))
        self.uvcPreviewScaleCard.titleLabel.setText("Capture Preview Scale Mode")
        self.ndiSourceCard.titleLabel.setText("NDI Stream")
        self.ndiRefreshCard.titleLabel.setText("Refresh NDI Streams")
        self.ndiRefreshBtn.setText(t("refresh"))
        self.ndiBandwidthCard.titleLabel.setText("NDI Bandwidth")
        self.ndiPreResizeCard.titleLabel.setText("NDI Pre-Resize")
        self.alwaysAimCard.titleLabel.setText(t("always_aim"))
        self.keepDetectingCard.titleLabel.setText(t("keep_detecting"))
        self.idleDetectEnableCard.titleLabel.setText(t("idle_detect_enabled"))
        self.idleDetectIntervalCard.titleLabel.setText(t("idle_detect_interval"))
        self.singleTargetCard.titleLabel.setText(t("single_target_mode"))
        self.skipLetterboxCard.titleLabel.setText(t("skip_letterbox_label"))
        self.skipLetterboxCard.contentLabel.setText(t("skip_letterbox_desc"))

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

        self.makcuComPortCard.titleLabel.setText(t("makcu_com_port"))
        self.makcuComRefreshBtn.setText(t("refresh"))
        self.makcuConnectionCard.titleLabel.setText(t("connected") + " / " + t("disconnected"))
        self.makcuConnectCard.titleLabel.setText(t("makcu_connect"))
        self.makcuConnectCard.contentLabel.setText(t("makcu_connect_desc"))
        self._updateMakcuConnectionStatus()

        self.xboxSensitivityCard.titleLabel.setText(t("xbox_sensitivity"))
        self.xboxDeadzoneCard.titleLabel.setText(t("xbox_deadzone"))
        self.xboxConnectionCard.titleLabel.setText(t("connected") + " / " + t("disconnected"))
        self.xboxConnectCard.titleLabel.setText(t("xbox_connect"))
        self.xboxConnectCard.contentLabel.setText(t("xbox_connect_desc"))

        current_aim = self.aimPartCombo.currentIndex()
        self.aimPartCombo.clear()
        self.aimPartCombo.addItems([t("head"), t("body"), t("both")])
        self.aimPartCombo.setCurrentIndex(current_aim)
