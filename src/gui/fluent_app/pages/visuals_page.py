# visuals_page.py
"""Visual Settings Page - Display Toggles, Detection Range"""

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QGridLayout, QSizePolicy, QWidget
from qfluentwidgets import (
    SettingCardGroup, SwitchSettingCard, SettingCard,
    FluentIcon, CheckBox, ComboBox
)
from ..components.slider_spin_card import SliderLabelCard

from ..base_page import BasePage
from ..language_manager import t


class VisualsPage(BasePage):
    """Visual Settings Page"""

    def __init__(self, parent=None):
        super().__init__("tab_display", parent)
        self._config = None
        self._acrylicRefreshTimer = QTimer(self)
        self._acrylicRefreshTimer.setSingleShot(True)
        self._acrylicRefreshTimer.setInterval(120)
        self._acrylicRefreshTimer.timeout.connect(self._refreshWindowEffect)
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

        self.spAutoAimCheck = CheckBox(self._shortText("auto_aim"), self.statusPanelElementsWidget)
        self.spModelCheck = CheckBox(self._shortText("status_panel_current_model"), self.statusPanelElementsWidget)
        self.spMouseMoveCheck = CheckBox(self._shortText("mouse_move_method"), self.statusPanelElementsWidget)
        self.spMouseClickCheck = CheckBox(self._shortText("mouse_click_method"), self.statusPanelElementsWidget)
        self.spScreenshotMethodCheck = CheckBox(self._shortText("screenshot_method"), self.statusPanelElementsWidget)
        self.spScreenshotFpsCheck = CheckBox(t("status_panel_screenshot_fps", "Screenshot FPS"), self.statusPanelElementsWidget)
        self.spDetectionFpsCheck = CheckBox(t("status_panel_detection_fps", "Detection FPS"), self.statusPanelElementsWidget)

        self._statusPanelChecks = [
            self.spAutoAimCheck,
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


    def _initLayout(self):
        """Lays out all controls"""
        # Display settings
        self.displayGroup.addSettingCard(self.showFovCard)
        self.displayGroup.addSettingCard(self.showBoxesCard)
        self.displayGroup.addSettingCard(self.showConfidenceCard)
        self.displayGroup.addSettingCard(self.showDetectRangeCard)
        self.displayGroup.addSettingCard(self.showTracerLineCard)
        self.displayGroup.addSettingCard(self.boxThemeCard)
        self.addContent(self.displayGroup)

        # Status panel settings
        self.statusPanelGroup.addSettingCard(self.showStatusCard)
        self.statusPanelGroup.addSettingCard(self.statusPanelElementsCard)
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

        self.scrollLayout.addStretch(1)

    def _connectSignals(self):
        """Connects signals"""
        # Display settings
        self.showFovCard.checkedChanged.connect(self._onShowFovChanged)
        self.showBoxesCard.checkedChanged.connect(self._onShowBoxesChanged)
        self.showConfidenceCard.checkedChanged.connect(self._onShowConfidenceChanged)
        self.showStatusCard.checkedChanged.connect(self._onShowStatusChanged)
        self.showDetectRangeCard.checkedChanged.connect(self._onShowDetectRangeChanged)
        self.showTracerLineCard.checkedChanged.connect(self._onShowTracerLineChanged)
        self.boxThemeCombo.currentTextChanged.connect(self._onBoxThemeChanged)
        self.spAutoAimCheck.stateChanged.connect(self._onStatusPanelAutoAimChanged)
        self.spModelCheck.stateChanged.connect(self._onStatusPanelModelChanged)
        self.spMouseMoveCheck.stateChanged.connect(self._onStatusPanelMouseMoveChanged)
        self.spMouseClickCheck.stateChanged.connect(self._onStatusPanelMouseClickChanged)
        self.spScreenshotMethodCheck.stateChanged.connect(self._onStatusPanelScreenshotMethodChanged)
        self.spScreenshotFpsCheck.stateChanged.connect(self._onStatusPanelScreenshotFpsChanged)
        self.spDetectionFpsCheck.stateChanged.connect(self._onStatusPanelDetectionFpsChanged)

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


    def _loadFromConfig(self):
        """Loads values from Config"""
        if not self._config:
            return

        # Display settings
        self.showFovCard.setChecked(self._config.show_fov)
        self.showBoxesCard.setChecked(self._config.show_boxes)
        self.showConfidenceCard.setChecked(self._config.show_confidence)
        self.showStatusCard.setChecked(self._config.show_status_panel)
        self.showDetectRangeCard.setChecked(self._config.show_detect_range)
        self.showTracerLineCard.setChecked(bool(getattr(self._config, 'show_tracer_line', False)))
        theme_text = str(getattr(self._config, 'box_color_theme', 'default')).capitalize()
        _valid_themes = ("Default", "Cyan", "Red", "Yellow", "White", "Purple")
        self.boxThemeCombo.setCurrentText(theme_text if theme_text in _valid_themes else "Default")
        self.spAutoAimCheck.setChecked(getattr(self._config, 'status_panel_show_auto_aim', True))
        self.spModelCheck.setChecked(getattr(self._config, 'status_panel_show_model', True))
        self.spMouseMoveCheck.setChecked(getattr(self._config, 'status_panel_show_mouse_move', True))
        self.spMouseClickCheck.setChecked(getattr(self._config, 'status_panel_show_mouse_click', True))
        self.spScreenshotMethodCheck.setChecked(getattr(self._config, 'status_panel_show_screenshot_method', True))
        self.spScreenshotFpsCheck.setChecked(getattr(self._config, 'status_panel_show_screenshot_fps', True))
        self.spDetectionFpsCheck.setChecked(getattr(self._config, 'status_panel_show_detection_fps', True))

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

    # === Callback Functions ===
    def _onShowFovChanged(self, checked):
        if self._config:
            self._config.show_fov = checked

    def _onShowBoxesChanged(self, checked):
        if self._config:
            self._config.show_boxes = checked

    def _onShowConfidenceChanged(self, checked):
        if self._config:
            self._config.show_confidence = checked

    def _onShowStatusChanged(self, checked):
        if self._config:
            self._config.show_status_panel = checked

    def _onShowDetectRangeChanged(self, checked):
        if self._config:
            self._config.show_detect_range = checked

    def _onShowTracerLineChanged(self, checked):
        if self._config:
            self._config.show_tracer_line = checked

    def _onBoxThemeChanged(self, text):
        if self._config:
            self._config.box_color_theme = str(text).lower()

    def _onStatusPanelAutoAimChanged(self, state):
        if self._config:
            self._config.status_panel_show_auto_aim = bool(state)

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
        self.showConfidenceCard.titleLabel.setText(t("show_confidence"))
        self.showStatusCard.titleLabel.setText(t("show_status_panel"))
        self.showDetectRangeCard.titleLabel.setText(t("show_detect_range"))
        self.showTracerLineCard.titleLabel.setText(t("show_tracer_line", "Tracer Line"))
        self.showTracerLineCard.contentLabel.setText(t("show_tracer_line_hint", "Draw a line from screen center to each detected target"))
        self.boxThemeCard.titleLabel.setText(t("box_color_theme", "Box Color Theme"))
        self.boxThemeCard.contentLabel.setText(t("box_color_theme_hint", "Color preset for detection boxes"))
        self.statusPanelElementsCard.titleLabel.setText(t("status_panel_elements", "Status Panel Elements"))
        self.statusPanelElementsCard.contentLabel.setText(t("status_panel_elements_hint", "Choose which rows are shown in status panel"))
        self.spAutoAimCheck.setText(self._shortText("auto_aim"))
        self.spModelCheck.setText(self._shortText("status_panel_current_model"))
        self.spMouseMoveCheck.setText(self._shortText("mouse_move_method"))
        self.spMouseClickCheck.setText(self._shortText("mouse_click_method"))
        self.spScreenshotMethodCheck.setText(self._shortText("screenshot_method"))
        self.spScreenshotFpsCheck.setText(t("status_panel_screenshot_fps", "Screenshot FPS"))
        self.spDetectionFpsCheck.setText(t("status_panel_detection_fps", "Detection FPS"))

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
