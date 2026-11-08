# visuals_page.py
"""Visual Settings Page - Display Toggles, Detection Range"""

import webbrowser

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QGridLayout, QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget
from qfluentwidgets import (
    SettingCardGroup, SwitchSettingCard, SettingCard,
    FluentIcon, CheckBox, ComboBox, BodyLabel, PushButton
)
from ..components.slider_spin_card import SliderLabelCard
from ..components.no_wheel_widgets import NoWheelSpinBox

from ..base_page import BasePage
from ..language_manager import t


class VisualsPage(BasePage):
    """Visual Settings Page"""

    # (core/gpu_monitor.py sensor key, English label)
    _GPU_SENSOR_LABELS = (
        ("temp", "Temp"), ("hotspot", "Hotspot"), ("util", "Usage"),
        ("vram", "VRAM"), ("fan", "Fan"), ("core_clock", "Core Clock"),
        ("mem_clock", "Mem Clock"), ("power", "Power"), ("voltage", "Voltage"),
    )

    def __init__(self, parent=None):
        super().__init__("tab_display", parent)
        self._config = None
        self._acrylicRefreshTimer = QTimer(self)
        self._acrylicRefreshTimer.setSingleShot(True)
        self._acrylicRefreshTimer.setInterval(120)
        self._acrylicRefreshTimer.timeout.connect(self._refreshWindowEffect)
        # Debounces a Web HUD server restart after a port-field edit — restart
        # is a real socket rebind, not a cheap config write, so this coalesces
        # rapid spinbox changes (typing, click-and-hold) into a single restart
        # once the user actually stops changing the value.
        self._webEspRestartTimer = QTimer(self)
        self._webEspRestartTimer.setSingleShot(True)
        self._webEspRestartTimer.setInterval(600)
        self._webEspRestartTimer.timeout.connect(self._restartWebEspIfRunning)
        self._initWidgets()
        self._initLayout()
        self._connectSignals()

    def setConfig(self, config):
        """Sets Config instance and loads values"""
        self._config = config
        self._loadFromConfig()

    def _initWidgets(self):
        """Initializes all controls"""

        # === Display Settings ===
        self.displayGroup = SettingCardGroup(t("tab_display"), self.scrollWidget)

        # === Status Panel Settings ===
        self.statusPanelGroup = SettingCardGroup(t("status_panel_settings", "Status Panel"), self.scrollWidget)

        # Show FOV
        self.showFovCard = SwitchSettingCard(
            FluentIcon.ZOOM,
            t("show_fov"),
            "",
            parent=self.displayGroup
        )

        # Show Boxes
        self.showBoxesCard = SwitchSettingCard(
            FluentIcon.CHECKBOX,
            t("show_boxes"),
            "",
            parent=self.displayGroup
        )

        # Full Bounding Box
        self.boxFullRectCard = SwitchSettingCard(
            FluentIcon.CHECKBOX,
            "Full Bounding Box",
            "Show a full rectangle instead of corner brackets",
            parent=self.displayGroup
        )

        # Line widths of the FOV / Detection Range / Tracer Line / Predicted Aim Point. Each sits
        # under its Show toggle and is only visible while that toggle is on.
        def _lineCard(title_key, title, hint_key, hint):
            return SliderLabelCard(
                FluentIcon.PENCIL_INK, t(title_key, title), 1, 10,
                format_func=lambda v: f"{v} px",
                description=t(hint_key, hint), parent=self.displayGroup)
        self.fovThicknessCard = _lineCard("fov_line_thickness", "FOV Line Thickness",
                                          "fov_line_thickness_hint", "Width of the FOV box or ellipse.")
        self.detectRangeThicknessCard = _lineCard("detect_range_line_thickness", "Detection Range Line Thickness",
                                                  "detect_range_line_thickness_hint", "Width of the detection range outline.")
        self.tracerThicknessCard = _lineCard("tracer_line_thickness", "Tracer Line Thickness",
                                             "tracer_line_thickness_hint", "Width of the line from screen center to each target.")
        self.aimMarkerThicknessCard = _lineCard("aim_marker_line_thickness", "Predicted Aim Point Thickness",
                                                "aim_marker_line_thickness_hint", "Width of the predicted aim point circle.")

        # Box line thickness (both styles) and corner-bracket arm length
        self.boxThicknessCard = SliderLabelCard(
            FluentIcon.PENCIL_INK,
            t("box_line_thickness", "Box Line Thickness"),
            0, 10,
            format_func=lambda v: "Auto" if v == 0 else f"{v} px",
            description=t("box_line_thickness_hint", "Thickness of corner brackets and full boxes. Auto = 1-3 px by confidence."),
            parent=self.displayGroup
        )
        self.boxCornerLengthCard = SliderLabelCard(
            FluentIcon.FIT_PAGE,
            t("box_corner_length", "Corner Bracket Length"),
            0, 50,
            format_func=lambda v: "Auto" if v == 0 else f"{v}%",
            description=t("box_corner_length_hint", "Bracket arm length as % of the box's shorter side. Auto = 15% (6-24 px)."),
            parent=self.displayGroup
        )

        # Show Confidence
        self.showConfidenceCard = SwitchSettingCard(
            FluentIcon.CERTIFICATE,
            t("show_confidence"),
            "",
            parent=self.displayGroup
        )

        # Show Status Panel
        self.showStatusCard = SwitchSettingCard(
            FluentIcon.INFO,
            t("show_status_panel"),
            "",
            parent=self.statusPanelGroup
        )

        # Show Detection Range
        self.showDetectRangeCard = SwitchSettingCard(
            FluentIcon.FULL_SCREEN,
            t("show_detect_range"),
            "",
            parent=self.displayGroup
        )

        # Show Tracer Line
        self.showTracerLineCard = SwitchSettingCard(
            FluentIcon.SHARE,
            t("show_tracer_line", "Tracer Line"),
            t("show_tracer_line_hint", "Draw a line from screen center to each detected target"),
            parent=self.displayGroup
        )

        # Show Predicted Aim-Point Marker
        self.showAimPredictionCard = SwitchSettingCard(
            FluentIcon.CERTIFICATE,
            t("show_aim_prediction_marker", "Predicted Aim Point"),
            t("show_aim_prediction_marker_hint",
              "Draw a second marker at the locked target's post-prediction point (only while Target Tracking's prediction or Kalman is on)"),
            parent=self.displayGroup
        )

        # Confidence Box Color Theme
        self.boxThemeCombo = ComboBox()
        self.boxThemeCombo.addItems(["Default", "Cyan", "Red", "Yellow", "White", "Purple"])
        self.boxThemeCombo.setMinimumWidth(110)
        self.boxThemeCard = SettingCard(
            FluentIcon.PALETTE,
            t("box_color_theme", "Box Color Theme"),
            t("box_color_theme_hint", "Color preset for detection boxes"),
            self.displayGroup
        )
        self.boxThemeCard.hBoxLayout.addWidget(self.boxThemeCombo, 0, Qt.AlignmentFlag.AlignRight)
        self.boxThemeCard.hBoxLayout.addSpacing(16)

        # Chroma Speed (rainbow cycle for in-FOV boxes)
        self.chromaSpeedCard = SliderLabelCard(
            FluentIcon.SPEED_HIGH,
            t("chroma_box_speed", "Chroma Speed"),
            0, 20,
            format_func=lambda v: "Off" if v == 0 else f"{v}x",
            description=t("chroma_box_speed_hint", "Rainbow cycle speed for boxes inside FOV. 0 = off."),
            parent=self.displayGroup
        )

        # Status Panel Elements (Checkbox style)
        self.statusPanelElementsCard = SettingCard(
            FluentIcon.INFO,
            t("status_panel_elements", "Status Panel Elements"),
            t("status_panel_elements_hint", "Choose which rows are shown in status panel"),
            self.statusPanelGroup
        )
        self.statusPanelElementsWidget = QWidget(self.statusPanelElementsCard)
        self.statusPanelElementsLayout = QGridLayout(self.statusPanelElementsWidget)
        self.statusPanelElementsLayout.setContentsMargins(0, 0, 0, 0)
        self.statusPanelElementsLayout.setHorizontalSpacing(10)
        self.statusPanelElementsLayout.setVerticalSpacing(4)

        self.spAutoAimCheck = CheckBox(self._shortText("aim_status", "Aim Status"), self.statusPanelElementsWidget)
        self.spAimToggleCheck = CheckBox(t("status_panel_aim_toggle", "Aim Toggle"), self.statusPanelElementsWidget)
        self.spPresetConfigCheck = CheckBox(t("status_panel_preset_config", "Preset/Config"), self.statusPanelElementsWidget)
        self.spModelCheck = CheckBox(self._shortText("status_panel_current_model"), self.statusPanelElementsWidget)
        self.spMouseMoveCheck = CheckBox(self._shortText("mouse_move_method"), self.statusPanelElementsWidget)
        self.spMouseClickCheck = CheckBox(self._shortText("mouse_click_method"), self.statusPanelElementsWidget)
        self.spScreenshotMethodCheck = CheckBox(self._shortText("screenshot_method"), self.statusPanelElementsWidget)
        self.spScreenshotFpsCheck = CheckBox(t("status_panel_screenshot_fps", "Capture FPS"), self.statusPanelElementsWidget)
        self.spDetectionFpsCheck = CheckBox(t("status_panel_detection_fps", "Inference FPS"), self.statusPanelElementsWidget)

        self._statusPanelChecks = [
            self.spAutoAimCheck,
            self.spAimToggleCheck,
            self.spPresetConfigCheck,
            self.spModelCheck,
            self.spMouseMoveCheck,
            self.spMouseClickCheck,
            self.spScreenshotMethodCheck,
            self.spScreenshotFpsCheck,
            self.spDetectionFpsCheck,
        ]
        for index, check in enumerate(self._statusPanelChecks):
            row = index % 2
            col = index // 2
            self.statusPanelElementsLayout.addWidget(check, row, col)

        self.statusPanelElementsWidget.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Preferred)

        self.statusPanelElementsCard.hBoxLayout.addWidget(self.statusPanelElementsWidget, 0, Qt.AlignmentFlag.AlignRight)
        self.statusPanelElementsCard.hBoxLayout.addSpacing(16)

        # GPU monitoring (NVIDIA, via NVML) — a section at the bottom of the
        # status panel; see core/gpu_monitor.py and status_panel.py.
        self.gpuMonitorCard = SwitchSettingCard(
            FluentIcon.SPEED_HIGH,
            t("gpu_monitoring", "GPU Monitoring"),
            t("gpu_monitoring_hint", "Show NVIDIA GPU sensors in the status panel (needs an NVIDIA GPU and driver)"),
            parent=self.statusPanelGroup
        )
        self.gpuSensorsCard = SettingCard(
            FluentIcon.INFO,
            t("gpu_sensors", "GPU Sensors"),
            t("gpu_sensors_hint", "Choose which GPU readings are shown. Unsupported ones stay hidden."),
            self.statusPanelGroup
        )
        self.gpuSensorsWidget = QWidget(self.gpuSensorsCard)
        self.gpuSensorsLayout = QGridLayout(self.gpuSensorsWidget)
        self.gpuSensorsLayout.setContentsMargins(0, 0, 0, 0)
        self.gpuSensorsLayout.setHorizontalSpacing(10)
        self.gpuSensorsLayout.setVerticalSpacing(4)
        self._gpuSensorChecks = {}
        for index, (key, default) in enumerate(self._GPU_SENSOR_LABELS):
            check = CheckBox(t(f"status_panel_gpu_{key}", default), self.gpuSensorsWidget)
            check.stateChanged.connect(lambda state, k=key: self._onGpuSensorChanged(k, state))
            self._gpuSensorChecks[key] = check
            self.gpuSensorsLayout.addWidget(check, index % 3, index // 3)
        self.gpuSensorsWidget.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Preferred)
        self.gpuSensorsCard.hBoxLayout.addWidget(self.gpuSensorsWidget, 0, Qt.AlignmentFlag.AlignRight)
        self.gpuSensorsCard.hBoxLayout.addSpacing(16)
        self.gpuIntervalCard = SliderLabelCard(
            FluentIcon.HISTORY,
            t("gpu_refresh", "GPU Refresh"),
            250, 5000,
            format_func=lambda v: f"{v} ms",
            description=t("gpu_refresh_hint", "How often the GPU sensors are read"),
            parent=self.statusPanelGroup
        )

        # === Crosshair Settings ===
        self.crosshairGroup = SettingCardGroup(t("crosshair_settings", "Crosshair"), self.scrollWidget)

        self.showCrosshairCard = SwitchSettingCard(
            FluentIcon.ZOOM,
            t("show_crosshair_overlay", "Show Crosshair"),
            "",
            parent=self.crosshairGroup
        )

        self.crosshairStyleCombo = ComboBox()
        self.crosshairStyleCombo.addItems(["Dot", "Cross"])
        self.crosshairStyleCombo.setMinimumWidth(100)
        self.crosshairStyleCard = SettingCard(
            FluentIcon.VIEW,
            t("crosshair_style", "Crosshair Style"),
            "",
            self.crosshairGroup
        )
        self.crosshairStyleCard.hBoxLayout.addWidget(self.crosshairStyleCombo, 0, Qt.AlignmentFlag.AlignRight)
        self.crosshairStyleCard.hBoxLayout.addSpacing(16)

        self.crosshairSizeCard = SliderLabelCard(
            FluentIcon.ZOOM_IN,
            t("crosshair_size", "Crosshair Size"),
            1, 20,
            format_func=lambda v: f"{v}px",
            description="",
            slider_width=160,
            parent=self.crosshairGroup
        )

        self.crosshairColorRCard = SliderLabelCard(
            FluentIcon.BRUSH,
            t("crosshair_color_r", "Red"),
            0, 255,
            format_func=lambda v: str(v),
            description="",
            slider_width=160,
            parent=self.crosshairGroup
        )
        self.crosshairColorGCard = SliderLabelCard(
            FluentIcon.BRUSH,
            t("crosshair_color_g", "Green"),
            0, 255,
            format_func=lambda v: str(v),
            description="",
            slider_width=160,
            parent=self.crosshairGroup
        )
        self.crosshairColorBCard = SliderLabelCard(
            FluentIcon.BRUSH,
            t("crosshair_color_b", "Blue"),
            0, 255,
            format_func=lambda v: str(v),
            description="",
            slider_width=160,
            parent=self.crosshairGroup
        )

        # === Appearance Settings ===
        self.appearanceGroup = SettingCardGroup(t("appearance_options"), self.scrollWidget)

        # Enable Acrylic
        self.enableAcrylicCard = SwitchSettingCard(
            FluentIcon.LAYOUT,
            t("enable_acrylic"),
            t("enable_acrylic_hint"),
            parent=self.appearanceGroup
        )

        # Acrylic Window Alpha
        self.windowAlphaCard = SliderLabelCard(
            FluentIcon.BRUSH,
            t("acrylic_window_alpha"),
            60, 255,
            format_func=lambda v: str(v),
            description="",
            slider_width=200,
            parent=self.appearanceGroup
        )

        # === Web ESP ===
        self.webEspGroup = SettingCardGroup(t("web_esp_settings", "Web ESP"), self.scrollWidget)

        self.webEspEnableCard = SwitchSettingCard(
            FluentIcon.GLOBE,
            t("web_esp_enabled", "Enable Web ESP"),
            t("web_esp_desc", "Stream the ESP overlay to a browser on this PC or any device on your LAN."),
            parent=self.webEspGroup
        )

        self.webEspHttpPortSpin = NoWheelSpinBox()
        self.webEspHttpPortSpin.setRange(1024, 65535)
        self.webEspHttpPortCard = SettingCard(
            FluentIcon.LINK,
            t("web_esp_http_port", "HTTP Port"),
            t("web_esp_http_port_desc", "Port for the overlay web page."),
            self.webEspGroup
        )
        self.webEspHttpPortCard.hBoxLayout.addWidget(self.webEspHttpPortSpin, 0, Qt.AlignmentFlag.AlignRight)
        self.webEspHttpPortCard.hBoxLayout.addSpacing(16)

        self.webEspWsPortSpin = NoWheelSpinBox()
        self.webEspWsPortSpin.setRange(1024, 65535)
        self.webEspWsPortCard = SettingCard(
            FluentIcon.WIFI,
            t("web_esp_ws_port", "WebSocket Port"),
            t("web_esp_ws_port_desc", "Port for the live state stream."),
            self.webEspGroup
        )
        self.webEspWsPortCard.hBoxLayout.addWidget(self.webEspWsPortSpin, 0, Qt.AlignmentFlag.AlignRight)
        self.webEspWsPortCard.hBoxLayout.addSpacing(16)

        # Connect card — URL text, QR code, and an "open in browser" button
        self.webEspConnectCard = SettingCard(
            FluentIcon.QRCODE,
            t("web_esp_connect", "Connect"),
            t("web_esp_connect_desc", "Open this URL on the same PC, or scan the QR from a phone on the same Wi-Fi."),
            self.webEspGroup
        )
        self._webEspConnectWidget = QWidget()
        _cv = QVBoxLayout(self._webEspConnectWidget)
        _cv.setContentsMargins(0, 4, 0, 4)
        _cv.setSpacing(8)
        self.webEspQrLabel = QLabel()
        self.webEspQrLabel.setFixedSize(160, 160)
        self.webEspQrLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.webEspOpenBtn = PushButton(t("web_esp_open", "Open in browser"))
        self.webEspOpenBtn.setMinimumWidth(140)
        _cv.addWidget(self.webEspQrLabel)
        _cv.addWidget(self.webEspOpenBtn)
        self.webEspConnectCard.hBoxLayout.addStretch(1)
        self.webEspConnectCard.hBoxLayout.addWidget(self._webEspConnectWidget)
        self.webEspConnectCard.hBoxLayout.addSpacing(16)


    def _initLayout(self):
        """Lays out all controls"""
        # Display settings
        self.displayGroup.addSettingCard(self.showFovCard)
        self.displayGroup.addSettingCard(self.fovThicknessCard)
        self.displayGroup.addSettingCard(self.showBoxesCard)
        self.displayGroup.addSettingCard(self.boxFullRectCard)
        self.displayGroup.addSettingCard(self.boxThicknessCard)
        self.displayGroup.addSettingCard(self.boxCornerLengthCard)
        self.displayGroup.addSettingCard(self.showConfidenceCard)
        self.displayGroup.addSettingCard(self.showDetectRangeCard)
        self.displayGroup.addSettingCard(self.detectRangeThicknessCard)
        self.displayGroup.addSettingCard(self.showTracerLineCard)
        self.displayGroup.addSettingCard(self.tracerThicknessCard)
        self.displayGroup.addSettingCard(self.showAimPredictionCard)
        self.displayGroup.addSettingCard(self.aimMarkerThicknessCard)
        self.displayGroup.addSettingCard(self.boxThemeCard)
        self.displayGroup.addSettingCard(self.chromaSpeedCard)
        self.addContent(self.displayGroup)

        # Status panel settings
        self.statusPanelGroup.addSettingCard(self.showStatusCard)
        self.statusPanelGroup.addSettingCard(self.statusPanelElementsCard)
        self.statusPanelGroup.addSettingCard(self.gpuMonitorCard)
        self.statusPanelGroup.addSettingCard(self.gpuSensorsCard)
        self.statusPanelGroup.addSettingCard(self.gpuIntervalCard)
        self.addContent(self.statusPanelGroup)

        # Crosshair settings
        self.crosshairGroup.addSettingCard(self.showCrosshairCard)
        self.crosshairGroup.addSettingCard(self.crosshairStyleCard)
        self.crosshairGroup.addSettingCard(self.crosshairSizeCard)
        self.crosshairGroup.addSettingCard(self.crosshairColorRCard)
        self.crosshairGroup.addSettingCard(self.crosshairColorGCard)
        self.crosshairGroup.addSettingCard(self.crosshairColorBCard)
        self.addContent(self.crosshairGroup)

        # Appearance settings
        self.appearanceGroup.addSettingCard(self.enableAcrylicCard)
        self.appearanceGroup.addSettingCard(self.windowAlphaCard)

        self.addContent(self.appearanceGroup)

        # Web ESP overlay settings
        self.webEspGroup.addSettingCard(self.webEspEnableCard)
        self.webEspGroup.addSettingCard(self.webEspHttpPortCard)
        self.webEspGroup.addSettingCard(self.webEspWsPortCard)
        self.webEspGroup.addSettingCard(self.webEspConnectCard)
        self.addContent(self.webEspGroup)

        self.scrollLayout.addStretch(1)

    def _connectSignals(self):
        """Connects signals"""
        # Display settings
        self.showFovCard.checkedChanged.connect(self._onShowFovChanged)
        self.showBoxesCard.checkedChanged.connect(self._onShowBoxesChanged)
        self.boxFullRectCard.checkedChanged.connect(self._onBoxFullRectChanged)
        self.showConfidenceCard.checkedChanged.connect(self._onShowConfidenceChanged)
        self.showStatusCard.checkedChanged.connect(self._onShowStatusChanged)
        self.showDetectRangeCard.checkedChanged.connect(self._onShowDetectRangeChanged)
        self.showTracerLineCard.checkedChanged.connect(self._onShowTracerLineChanged)
        self.showAimPredictionCard.checkedChanged.connect(self._onShowAimPredictionChanged)
        self.boxThemeCombo.currentTextChanged.connect(self._onBoxThemeChanged)
        self.chromaSpeedCard.valueChanged.connect(self._onChromaSpeedChanged)
        self.boxThicknessCard.valueChanged.connect(self._onBoxThicknessChanged)
        for _card, _attr in ((self.fovThicknessCard, 'fov_line_thickness'),
                             (self.detectRangeThicknessCard, 'detect_range_line_thickness'),
                             (self.tracerThicknessCard, 'tracer_line_thickness'),
                             (self.aimMarkerThicknessCard, 'aim_marker_line_thickness')):
            _card.valueChanged.connect(lambda v, a=_attr: self._onLineThicknessChanged(a, v))
        self.boxCornerLengthCard.valueChanged.connect(self._onBoxCornerLengthChanged)
        self.spAutoAimCheck.stateChanged.connect(self._onStatusPanelAutoAimChanged)
        self.spAimToggleCheck.stateChanged.connect(self._onStatusPanelAimToggleChanged)
        self.spPresetConfigCheck.stateChanged.connect(self._onStatusPanelPresetConfigChanged)
        self.spModelCheck.stateChanged.connect(self._onStatusPanelModelChanged)
        self.spMouseMoveCheck.stateChanged.connect(self._onStatusPanelMouseMoveChanged)
        self.spMouseClickCheck.stateChanged.connect(self._onStatusPanelMouseClickChanged)
        self.spScreenshotMethodCheck.stateChanged.connect(self._onStatusPanelScreenshotMethodChanged)
        self.spScreenshotFpsCheck.stateChanged.connect(self._onStatusPanelScreenshotFpsChanged)
        self.spDetectionFpsCheck.stateChanged.connect(self._onStatusPanelDetectionFpsChanged)
        self.gpuMonitorCard.checkedChanged.connect(self._onGpuMonitorChanged)
        self.gpuIntervalCard.valueChanged.connect(self._onGpuIntervalChanged)

        # Crosshair settings
        self.showCrosshairCard.checkedChanged.connect(self._onShowCrosshairChanged)
        self.crosshairStyleCombo.currentTextChanged.connect(self._onCrosshairStyleChanged)
        self.crosshairSizeCard.valueChanged.connect(self._onCrosshairSizeChanged)
        self.crosshairColorRCard.valueChanged.connect(self._onCrosshairColorRChanged)
        self.crosshairColorGCard.valueChanged.connect(self._onCrosshairColorGChanged)
        self.crosshairColorBCard.valueChanged.connect(self._onCrosshairColorBChanged)

        # Appearance settings
        self.enableAcrylicCard.checkedChanged.connect(self._onAcrylicEnabledChanged)
        self.windowAlphaCard.valueChanged.connect(self._onWindowAlphaChanged)
        self.windowAlphaCard.slider.sliderReleased.connect(self._onWindowAlphaCommit)

        # Web ESP overlay
        self.webEspEnableCard.checkedChanged.connect(self._onWebEspEnableChanged)
        self.webEspHttpPortSpin.valueChanged.connect(self._onWebEspHttpPortChanged)
        self.webEspWsPortSpin.valueChanged.connect(self._onWebEspWsPortChanged)
        self.webEspOpenBtn.clicked.connect(self._onWebEspOpen)


    def _loadFromConfig(self):
        """Loads values from Config"""
        if not self._config:
            return

        # Display settings
        self.showFovCard.setChecked(self._config.show_fov)
        self.showBoxesCard.setChecked(self._config.show_boxes)
        self.boxFullRectCard.setChecked(bool(getattr(self._config, 'box_full_rect', False)))
        self.boxThicknessCard.setValue(int(getattr(self._config, 'box_line_thickness', 0) or 0))
        self.boxCornerLengthCard.setValue(int(getattr(self._config, 'box_corner_length_pct', 0) or 0))
        self.boxCornerLengthCard.setVisible(not bool(getattr(self._config, 'box_full_rect', False)))
        self.showConfidenceCard.setChecked(self._config.show_confidence)
        self.showStatusCard.setChecked(self._config.show_status_panel)
        self.showDetectRangeCard.setChecked(self._config.show_detect_range)
        self.showTracerLineCard.setChecked(bool(getattr(self._config, 'show_tracer_line', False)))
        self.showAimPredictionCard.setChecked(bool(getattr(self._config, 'show_aim_prediction_marker', True)))
        self.fovThicknessCard.setValue(int(getattr(self._config, 'fov_line_thickness', 2) or 2))
        self.detectRangeThicknessCard.setValue(int(getattr(self._config, 'detect_range_line_thickness', 1) or 1))
        self.tracerThicknessCard.setValue(int(getattr(self._config, 'tracer_line_thickness', 2) or 2))
        self.aimMarkerThicknessCard.setValue(int(getattr(self._config, 'aim_marker_line_thickness', 2) or 2))
        self._updateLineThicknessVisibility()
        theme_text = str(getattr(self._config, 'box_color_theme', 'default')).capitalize()
        _valid_themes = ("Default", "Cyan", "Red", "Yellow", "White", "Purple")
        self.boxThemeCombo.setCurrentText(theme_text if theme_text in _valid_themes else "Default")
        self.chromaSpeedCard.setValue(int(getattr(self._config, 'chroma_box_speed', 1)))
        self.spAutoAimCheck.setChecked(getattr(self._config, 'status_panel_show_auto_aim', True))
        self.spAimToggleCheck.setChecked(getattr(self._config, 'status_panel_show_aim_toggle', True))
        self.spPresetConfigCheck.setChecked(getattr(self._config, 'status_panel_show_preset_config', True))
        self.spModelCheck.setChecked(getattr(self._config, 'status_panel_show_model', True))
        self.spMouseMoveCheck.setChecked(getattr(self._config, 'status_panel_show_mouse_move', True))
        self.spMouseClickCheck.setChecked(getattr(self._config, 'status_panel_show_mouse_click', True))
        self.spScreenshotMethodCheck.setChecked(getattr(self._config, 'status_panel_show_screenshot_method', True))
        self.spScreenshotFpsCheck.setChecked(getattr(self._config, 'status_panel_show_screenshot_fps', True))
        self.spDetectionFpsCheck.setChecked(getattr(self._config, 'status_panel_show_detection_fps', True))
        gpu_on = bool(getattr(self._config, 'status_panel_show_gpu', False))
        self.gpuMonitorCard.setChecked(gpu_on)
        for key, check in self._gpuSensorChecks.items():
            check.setChecked(bool(getattr(self._config, f'status_panel_gpu_{key}', True)))
        self.gpuIntervalCard.setValue(int(getattr(self._config, 'status_panel_gpu_interval_ms', 1000)))
        self._updateGpuMonitorVisibility(gpu_on)

        # Crosshair settings
        self.showCrosshairCard.setChecked(bool(getattr(self._config, 'show_crosshair', False)))
        style = str(getattr(self._config, 'crosshair_style', 'dot')).capitalize()
        self.crosshairStyleCombo.setCurrentText(style if style in ("Dot", "Cross") else "Dot")
        self.crosshairSizeCard.setValue(int(getattr(self._config, 'crosshair_size', 4)))
        self.crosshairColorRCard.setValue(int(getattr(self._config, 'crosshair_color_r', 255)))
        self.crosshairColorGCard.setValue(int(getattr(self._config, 'crosshair_color_g', 255)))
        self.crosshairColorBCard.setValue(int(getattr(self._config, 'crosshair_color_b', 255)))

        # Appearance settings
        self.enableAcrylicCard.setChecked(self._config.enable_acrylic)
        safe_alpha = max(60, min(255, int(getattr(self._config, 'acrylic_window_alpha', 187))))
        if self._config.acrylic_window_alpha != safe_alpha:
            self._config.acrylic_window_alpha = safe_alpha
        self.windowAlphaCard.setValue(safe_alpha)

        # Web ESP overlay
        web_on = bool(getattr(self._config, 'web_esp_enabled', False))
        self.webEspEnableCard.setChecked(web_on)
        self.webEspHttpPortSpin.setValue(int(getattr(self._config, 'web_esp_http_port', 8080)))
        self.webEspWsPortSpin.setValue(int(getattr(self._config, 'web_esp_ws_port', 8765)))
        self.webEspHttpPortCard.setEnabled(web_on)
        self.webEspWsPortCard.setEnabled(web_on)
        self.webEspConnectCard.setEnabled(web_on)
        self._refreshWebEspConnect()

    # === Callback Functions ===
    def _onShowFovChanged(self, checked):
        if self._config:
            self._config.show_fov = checked
        self._updateLineThicknessVisibility()

    def _onShowBoxesChanged(self, checked):
        if self._config:
            self._config.show_boxes = checked

    def _onBoxFullRectChanged(self, checked):
        if self._config:
            self._config.box_full_rect = bool(checked)
        # Bracket length only applies to corner brackets.
        self.boxCornerLengthCard.setVisible(not bool(checked))

    def _onBoxThicknessChanged(self, value):
        if self._config:
            self._config.box_line_thickness = int(value)

    def _onBoxCornerLengthChanged(self, value):
        if self._config:
            self._config.box_corner_length_pct = int(value)

    def _onShowConfidenceChanged(self, checked):
        if self._config:
            self._config.show_confidence = checked

    def _onShowStatusChanged(self, checked):
        if self._config:
            self._config.show_status_panel = checked

    def _onShowDetectRangeChanged(self, checked):
        if self._config:
            self._config.show_detect_range = checked
        self._updateLineThicknessVisibility()

    def _onShowTracerLineChanged(self, checked):
        if self._config:
            self._config.show_tracer_line = checked
        self._updateLineThicknessVisibility()

    def _onShowAimPredictionChanged(self, checked):
        if self._config:
            self._config.show_aim_prediction_marker = checked
        self._updateLineThicknessVisibility()

    def _onLineThicknessChanged(self, attr, value):
        if self._config:
            setattr(self._config, attr, max(1, min(10, int(value))))

    def _updateLineThicknessVisibility(self):
        """Each thickness card is hidden (not greyed out) while its Show toggle is off."""
        c = self._config
        if not c:
            return
        self.fovThicknessCard.setVisible(bool(getattr(c, 'show_fov', True)))
        self.detectRangeThicknessCard.setVisible(bool(getattr(c, 'show_detect_range', True)))
        self.tracerThicknessCard.setVisible(bool(getattr(c, 'show_tracer_line', False)))
        self.aimMarkerThicknessCard.setVisible(bool(getattr(c, 'show_aim_prediction_marker', True)))

    def _onBoxThemeChanged(self, text):
        if self._config:
            self._config.box_color_theme = str(text).lower()

    def _onChromaSpeedChanged(self, value):
        if self._config:
            self._config.chroma_box_speed = float(value)

    def _onStatusPanelAutoAimChanged(self, state):
        if self._config:
            self._config.status_panel_show_auto_aim = bool(state)

    def _onStatusPanelAimToggleChanged(self, state):
        if self._config:
            self._config.status_panel_show_aim_toggle = bool(state)

    def _onStatusPanelPresetConfigChanged(self, state):
        if self._config:
            self._config.status_panel_show_preset_config = bool(state)

    def _onStatusPanelModelChanged(self, state):
        if self._config:
            self._config.status_panel_show_model = bool(state)

    def _onStatusPanelMouseMoveChanged(self, state):
        if self._config:
            self._config.status_panel_show_mouse_move = bool(state)

    def _onStatusPanelMouseClickChanged(self, state):
        if self._config:
            self._config.status_panel_show_mouse_click = bool(state)

    def _onStatusPanelScreenshotMethodChanged(self, state):
        if self._config:
            self._config.status_panel_show_screenshot_method = bool(state)

    def _onStatusPanelScreenshotFpsChanged(self, state):
        if self._config:
            self._config.status_panel_show_screenshot_fps = bool(state)

    def _onStatusPanelDetectionFpsChanged(self, state):
        if self._config:
            self._config.status_panel_show_detection_fps = bool(state)

    def _onGpuMonitorChanged(self, checked):
        if self._config:
            self._config.status_panel_show_gpu = bool(checked)
        self._updateGpuMonitorVisibility(bool(checked))

    def _onGpuSensorChanged(self, key, state):
        if self._config:
            setattr(self._config, f'status_panel_gpu_{key}', bool(state))

    def _onGpuIntervalChanged(self, value):
        if self._config:
            self._config.status_panel_gpu_interval_ms = int(value)

    def _updateGpuMonitorVisibility(self, enabled):
        """Sensor switches and refresh only matter while GPU monitoring is on;
        a sensor this GPU can't report gets an explanatory tooltip."""
        self.gpuSensorsCard.setVisible(enabled)
        self.gpuIntervalCard.setVisible(enabled)
        hint = self._gpuMonitorHint(enabled)
        if self.gpuMonitorCard.contentLabel.text() != hint:
            self.gpuMonitorCard.contentLabel.setText(hint)
        support = getattr(self._config, 'gpu_sensor_support', None) if self._config else None
        supported = set((support or {}).get('supported') or ())
        known = bool(support) and support.get('status') == 'ok'
        for key, check in self._gpuSensorChecks.items():
            check.setToolTip("" if not known or key in supported
                             else t("gpu_sensor_unsupported", "Not supported on this GPU"))

    def _gpuMonitorHint(self, enabled):
        """Card description; while monitoring is on, says whether NVML works."""
        from core.gpu_monitor import nvml_installed
        if not nvml_installed():
            return t("gpu_monitoring_no_nvml",
                     "NVML module not found: the bundled pynvml (src/python/dependencies) is missing, so reinstall or update Axiom")
        support = getattr(self._config, 'gpu_sensor_support', None) if self._config else None
        status = (support or {}).get('status') if enabled else None
        if status == 'ok':
            return t("gpu_monitoring_ok", "Reading {name}").format(name=support.get('name') or "NVIDIA GPU")
        if status == 'no_gpu':
            return t("gpu_monitoring_no_gpu", "NVML is installed, but no NVIDIA GPU or driver was found")
        return t("gpu_monitoring_hint", "Show NVIDIA GPU sensors in the status panel (needs an NVIDIA GPU and driver)")

    def _onShowCrosshairChanged(self, checked):
        if self._config:
            self._config.show_crosshair = checked

    def _onCrosshairStyleChanged(self, text):
        if self._config:
            self._config.crosshair_style = str(text).lower()

    def _onCrosshairSizeChanged(self, value):
        if self._config:
            self._config.crosshair_size = int(value)

    def _onCrosshairColorRChanged(self, value):
        if self._config:
            self._config.crosshair_color_r = int(value)

    def _onCrosshairColorGChanged(self, value):
        if self._config:
            self._config.crosshair_color_g = int(value)

    def _onCrosshairColorBChanged(self, value):
        if self._config:
            self._config.crosshair_color_b = int(value)

    def _onAcrylicEnabledChanged(self, checked):
        if self._config:
            self._config.enable_acrylic = checked
            self._refreshWindowEffect(apply_styles=True)

    def _onWindowAlphaChanged(self, value):
        if self._config:
            safe_alpha = max(60, min(255, int(value)))
            self._config.acrylic_window_alpha = safe_alpha
            self._scheduleAcrylicRefresh()

    def _onWindowAlphaCommit(self):
        self._refreshWindowEffect()

    # ── Web ESP overlay ──────────────────────────────────
    def _onWebEspEnableChanged(self, checked):
        checked = bool(checked)
        if self._config:
            self._config.web_esp_enabled = checked
        self.webEspHttpPortCard.setEnabled(checked)
        self.webEspWsPortCard.setEnabled(checked)
        self.webEspConnectCard.setEnabled(checked)
        # Start/stop the server live so the URL+QR appear immediately.
        try:
            from core import esp_server
            if checked:
                esp_server.start(self._config)
                # _actual_ws_port is set asynchronously inside the accept-loop's
                # bind-with-fallback, which may still be settling when start()
                # returns — re-check shortly after so the URL/QR reflect the
                # real bound port instead of a stale/guessed one.
                QTimer.singleShot(400, self._refreshWebEspConnect)
            else:
                esp_server.stop()
        except Exception:
            pass
        self._refreshWebEspConnect()

    def _onWebEspHttpPortChanged(self, value):
        if self._config:
            self._config.web_esp_http_port = int(value)
        self._webEspRestartTimer.start()

    def _onWebEspWsPortChanged(self, value):
        if self._config:
            self._config.web_esp_ws_port = int(value)
        self._webEspRestartTimer.start()

    def _restartWebEspIfRunning(self):
        """Web HUD binds its HTTP/WS sockets once in esp_server.start() —
        changing either port while the server is already running doesn't
        rebind on its own, which left the displayed connect URL/QR pointing
        at a port nothing was actually listening on. Restart the server on
        the current (new) ports so it stays in sync, same as toggling the
        enable switch off then on.
        """
        try:
            from core import esp_server
            if esp_server.is_running():
                esp_server.stop()
                esp_server.start(self._config)
                # _actual_ws_port settles asynchronously inside the accept
                # loop's bind-with-fallback — same re-check pattern as
                # _onWebEspEnableChanged.
                QTimer.singleShot(400, self._refreshWebEspConnect)
        except Exception:
            pass
        self._refreshWebEspConnect()

    def _onWebEspOpen(self):
        try:
            from core import esp_server
            url = esp_server.connect_url()
            if url:
                webbrowser.open(url)
        except Exception:
            pass

    def _refreshWebEspConnect(self):
        """Update the connect URL label and QR image from the running server."""
        url = ""
        try:
            from core import esp_server
            if esp_server.is_running():
                url = esp_server.connect_url()
        except Exception:
            url = ""

        if not url:
            self.webEspQrLabel.clear()
            self.webEspOpenBtn.setEnabled(False)
            self._lastQrUrl = ""
            return

        self.webEspOpenBtn.setEnabled(True)
        # Runs on every 1 s refresh; only redraw the QR when the URL changed.
        if url == getattr(self, '_lastQrUrl', ""):
            return
        self._lastQrUrl = url
        # QR code via segno if available; otherwise URL text alone is enough.
        try:
            import io
            import segno
            buf = io.BytesIO()
            segno.make(url, error='m').save(buf, kind='png', scale=4, border=2)
            pix = QPixmap()
            pix.loadFromData(buf.getvalue())
            self.webEspQrLabel.setPixmap(
                pix.scaled(160, 160, Qt.AspectRatioMode.KeepAspectRatio,
                           Qt.TransformationMode.SmoothTransformation))
        except Exception:
            self.webEspQrLabel.setText(t("web_esp_qr_unavailable", "(install 'segno' for QR)"))

    def _scheduleAcrylicRefresh(self):
        if self._acrylicRefreshTimer.isActive():
            self._acrylicRefreshTimer.stop()
        self._acrylicRefreshTimer.start()



    def _refreshWindowEffect(self, apply_styles=False):
        """Notify window to refresh Acrylic effect and styles"""
        if self._acrylicRefreshTimer.isActive():
            self._acrylicRefreshTimer.stop()
        window = self.window()
        if window:
            if hasattr(window, '_applyAcrylicEffect'):
                window._applyAcrylicEffect()
            if apply_styles and hasattr(window, '_applyThemeStyles'):
                window._applyThemeStyles()

    @staticmethod
    def _shortText(key: str, default: str = "") -> str:
        return t(key, default).rstrip(':：').strip()

    def retranslateUi(self):
        """Refreshes translations"""
        super().retranslateUi()

        # Group titles
        self.displayGroup.titleLabel.setText(t("tab_display"))
        self.statusPanelGroup.titleLabel.setText(t("status_panel_settings", "Status Panel"))

        # Display settings
        self.showFovCard.titleLabel.setText(t("show_fov"))
        self.showBoxesCard.titleLabel.setText(t("show_boxes"))
        self.boxFullRectCard.titleLabel.setText("Full Bounding Box")
        self.boxThicknessCard.titleLabel.setText(t("box_line_thickness", "Box Line Thickness"))
        self.boxCornerLengthCard.titleLabel.setText(t("box_corner_length", "Corner Bracket Length"))
        self.boxThicknessCard.contentLabel.setText(t("box_line_thickness_hint", "Thickness of corner brackets and full boxes. Auto = 1-3 px by confidence."))
        self.boxCornerLengthCard.contentLabel.setText(t("box_corner_length_hint", "Bracket arm length as % of the box's shorter side. Auto = 15% (6-24 px)."))
        self.showConfidenceCard.titleLabel.setText(t("show_confidence"))
        self.showStatusCard.titleLabel.setText(t("show_status_panel"))
        self.showDetectRangeCard.titleLabel.setText(t("show_detect_range"))
        self.showTracerLineCard.titleLabel.setText(t("show_tracer_line", "Tracer Line"))
        self.showTracerLineCard.contentLabel.setText(t("show_tracer_line_hint", "Draw a line from screen center to each detected target"))
        for _card, _key, _title, _hkey, _hint in (
            (self.fovThicknessCard, "fov_line_thickness", "FOV Line Thickness", "fov_line_thickness_hint", "Width of the FOV box or ellipse."),
            (self.detectRangeThicknessCard, "detect_range_line_thickness", "Detection Range Line Thickness", "detect_range_line_thickness_hint", "Width of the detection range outline."),
            (self.tracerThicknessCard, "tracer_line_thickness", "Tracer Line Thickness", "tracer_line_thickness_hint", "Width of the line from screen center to each target."),
            (self.aimMarkerThicknessCard, "aim_marker_line_thickness", "Predicted Aim Point Thickness", "aim_marker_line_thickness_hint", "Width of the predicted aim point circle."),
        ):
            _card.titleLabel.setText(t(_key, _title))
            _card.contentLabel.setText(t(_hkey, _hint))
        self.showAimPredictionCard.titleLabel.setText(t("show_aim_prediction_marker", "Predicted Aim Point"))
        self.showAimPredictionCard.contentLabel.setText(t("show_aim_prediction_marker_hint",
            "Draw a second marker at the locked target's post-prediction point (only while Target Tracking's prediction or Kalman is on)"))
        self.boxThemeCard.titleLabel.setText(t("box_color_theme", "Box Color Theme"))
        self.boxThemeCard.contentLabel.setText(t("box_color_theme_hint", "Color preset for detection boxes"))
        self.chromaSpeedCard.titleLabel.setText(t("chroma_box_speed", "Chroma Speed"))
        self.chromaSpeedCard.contentLabel.setText(t("chroma_box_speed_hint", "Rainbow cycle speed for boxes inside FOV. 0 = off."))
        self.statusPanelElementsCard.titleLabel.setText(t("status_panel_elements", "Status Panel Elements"))
        self.statusPanelElementsCard.contentLabel.setText(t("status_panel_elements_hint", "Choose which rows are shown in status panel"))
        self.spAutoAimCheck.setText(self._shortText("aim_status", "Aim Status"))
        self.spAimToggleCheck.setText(t("status_panel_aim_toggle", "Aim Toggle"))
        self.spPresetConfigCheck.setText(t("status_panel_preset_config", "Preset/Config"))
        self.spModelCheck.setText(self._shortText("status_panel_current_model"))
        self.spMouseMoveCheck.setText(self._shortText("mouse_move_method"))
        self.spMouseClickCheck.setText(self._shortText("mouse_click_method"))
        self.spScreenshotMethodCheck.setText(self._shortText("screenshot_method"))
        self.spScreenshotFpsCheck.setText(t("status_panel_screenshot_fps", "Capture FPS"))
        self.spDetectionFpsCheck.setText(t("status_panel_detection_fps", "Inference FPS"))
        self.gpuMonitorCard.titleLabel.setText(t("gpu_monitoring", "GPU Monitoring"))
        self.gpuMonitorCard.contentLabel.setText(self._gpuMonitorHint(self.gpuMonitorCard.isChecked()))
        self.gpuSensorsCard.titleLabel.setText(t("gpu_sensors", "GPU Sensors"))
        self.gpuSensorsCard.contentLabel.setText(t(
            "gpu_sensors_hint", "Choose which GPU readings are shown. Unsupported ones stay hidden."))
        for key, default in self._GPU_SENSOR_LABELS:
            self._gpuSensorChecks[key].setText(t(f"status_panel_gpu_{key}", default))
        self.gpuIntervalCard.titleLabel.setText(t("gpu_refresh", "GPU Refresh"))
        self.gpuIntervalCard.contentLabel.setText(t("gpu_refresh_hint", "How often the GPU sensors are read"))

        # Crosshair settings
        self.crosshairGroup.titleLabel.setText(t("crosshair_settings", "Crosshair"))
        self.showCrosshairCard.titleLabel.setText(t("show_crosshair_overlay", "Show Crosshair"))
        self.crosshairStyleCard.titleLabel.setText(t("crosshair_style", "Crosshair Style"))
        self.crosshairSizeCard.titleLabel.setText(t("crosshair_size", "Crosshair Size"))
        self.crosshairColorRCard.titleLabel.setText(t("crosshair_color_r", "Red"))
        self.crosshairColorGCard.titleLabel.setText(t("crosshair_color_g", "Green"))
        self.crosshairColorBCard.titleLabel.setText(t("crosshair_color_b", "Blue"))

        # 外觀設定
        self.appearanceGroup.titleLabel.setText(t("appearance_options"))
        self.enableAcrylicCard.titleLabel.setText(t("enable_acrylic"))
        self.enableAcrylicCard.contentLabel.setText(t("enable_acrylic_hint"))
        self.windowAlphaCard.titleLabel.setText(t("acrylic_window_alpha"))
        self.windowAlphaCard.contentLabel.setText("")

        # Web HUD overlay settings
        self.webEspGroup.titleLabel.setText(t("web_esp_settings", "Web ESP"))
        self.webEspEnableCard.titleLabel.setText(t("web_esp_enabled", "Enable Web ESP"))
        self.webEspEnableCard.contentLabel.setText(t("web_esp_desc", "Stream the ESP overlay to a browser on this PC or any device on your LAN."))
        self.webEspHttpPortCard.titleLabel.setText(t("web_esp_http_port", "HTTP Port"))
        self.webEspHttpPortCard.contentLabel.setText(t("web_esp_http_port_desc", "Port for the overlay web page."))
        self.webEspWsPortCard.titleLabel.setText(t("web_esp_ws_port", "WebSocket Port"))
        self.webEspWsPortCard.contentLabel.setText(t("web_esp_ws_port_desc", "Port for the live state stream."))
        self.webEspConnectCard.titleLabel.setText(t("web_esp_connect", "Connect"))
        self.webEspConnectCard.contentLabel.setText(t("web_esp_connect_desc", "Open this URL on the same PC, or scan the QR from a phone on the same Wi-Fi."))
        self.webEspOpenBtn.setText(t("web_esp_open", "Open in browser"))
