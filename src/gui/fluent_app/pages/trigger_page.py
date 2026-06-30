# trigger_page.py
"""自動扳機頁面 - 自動射擊設定"""

from PyQt6.QtCore import Qt
from qfluentwidgets import (
    SettingCardGroup, SettingCard, SwitchSettingCard,
    FluentIcon, ComboBox
)
from ..components.slider_spin_card import SliderDoubleSpinCard

from ..base_page import BasePage
from ..language_manager import t


class TriggerPage(BasePage):
    """自動扳機頁面"""

    def __init__(self, parent=None):
        super().__init__("tab_auto_features", parent)
        self._config = None
        self._initWidgets()
        self._initLayout()
        self._connectSignals()

    def setConfig(self, config):
        """設定 Config 實例並載入值"""
        self._config = config
        self._loadFromConfig()

    def _initWidgets(self):
        """初始化所有控制項"""

        # === 自動射擊設定 ===
        self.fireGroup = SettingCardGroup(t("keys_and_auto_fire"), self.scrollWidget)

        # 自動射擊目標
        self.fireTargetCombo = ComboBox()
        self.fireTargetCombo.addItems([t("head"), t("body"), t("both")])
        self.fireTargetCombo.setMinimumWidth(120)
        self.fireTargetCard = SettingCard(
            FluentIcon.PEOPLE,
            t("auto_fire_target"),
            "",
            self.fireGroup
        )
        self.fireTargetCard.hBoxLayout.addWidget(self.fireTargetCombo, 0, Qt.AlignmentFlag.AlignRight)
        self.fireTargetCard.hBoxLayout.addSpacing(16)

        # 持續自動射擊（不需按住按鍵）
        self.alwaysAutoFireCard = SwitchSettingCard(
            FluentIcon.RINGER,
            t("always_auto_fire"),
            "",
            parent=self.fireGroup
        )

        # 滑鼠點擊方式（自動射擊使用的模擬信號）
        self.mouseClickCombo = ComboBox()
        self.mouseClickCombo.addItems(["mouse_event", "sendinput", "ddxoft", "arduino", "makcu", "xbox"])
        self.mouseClickCombo.setMinimumWidth(150)
        self.mouseClickCard = SettingCard(
            FluentIcon.FINGERPRINT,
            t("mouse_click_method"),
            "",
            self.fireGroup
        )
        self.mouseClickCard.hBoxLayout.addWidget(self.mouseClickCombo, 0, Qt.AlignmentFlag.AlignRight)
        self.mouseClickCard.hBoxLayout.addSpacing(16)

        # 開鏡延遲 - 使用 SliderDoubleSpinCard
        self.scopeDelayCard = SliderDoubleSpinCard(
            FluentIcon.HISTORY,
            t("scope_delay"),
            0.0, 2.0,
            decimals=2,
            step=0.01,
            suffix="s",
            description="",
            parent=self.fireGroup
        )

        # 射擊間隔 - 使用 SliderDoubleSpinCard
        self.fireIntervalCard = SliderDoubleSpinCard(
            FluentIcon.SPEED_HIGH,
            t("fire_interval"),
            0.01, 1.0,
            decimals=2,
            step=0.01,
            suffix="s",
            description="",
            parent=self.fireGroup
        )

    def _initLayout(self):
        """排版所有控制項"""
        # 自動射擊設定
        self.fireGroup.addSettingCard(self.fireTargetCard)
        self.fireGroup.addSettingCard(self.alwaysAutoFireCard)
        self.fireGroup.addSettingCard(self.mouseClickCard)
        self.fireGroup.addSettingCard(self.scopeDelayCard)
        self.fireGroup.addSettingCard(self.fireIntervalCard)
        self.addContent(self.fireGroup)

        self.scrollLayout.addStretch(1)

    def _connectSignals(self):
        """連接信號"""
        # 自動射擊設定
        self.fireTargetCombo.currentIndexChanged.connect(self._onFireTargetChanged)
        self.alwaysAutoFireCard.checkedChanged.connect(self._onAlwaysAutoFireChanged)
        self.mouseClickCombo.currentTextChanged.connect(self._onMouseClickChanged)
        self.scopeDelayCard.valueChanged.connect(self._onScopeDelayChanged)
        self.fireIntervalCard.valueChanged.connect(self._onFireIntervalChanged)

    def _loadFromConfig(self):
        """從 Config 載入值"""
        if not self._config:
            return

        # 自動射擊設定
        targets = ["head", "body", "both"]
        if self._config.auto_fire_target_part in targets:
            self.fireTargetCombo.setCurrentIndex(targets.index(self._config.auto_fire_target_part))
        self.alwaysAutoFireCard.setChecked(getattr(self._config, 'always_auto_fire', False))

        # 滑鼠點擊方式
        click_methods = ["mouse_event", "sendinput", "ddxoft", "arduino", "makcu", "xbox"]
        current_click = getattr(self._config, 'mouse_click_method', 'mouse_event')
        if current_click in click_methods:
            self.mouseClickCombo.setCurrentIndex(click_methods.index(current_click))

        # 開鏡延遲 - 使用新組件的 setValue
        self.scopeDelayCard.setValue(self._config.auto_fire_delay)

        # 射擊間隔 - 使用新組件的 setValue
        self.fireIntervalCard.setValue(self._config.auto_fire_interval)

    # === 回調函數 ===
    def _onFireTargetChanged(self, index):
        if self._config:
            targets = ["head", "body", "both"]
            self._config.auto_fire_target_part = targets[index]

    def _onScopeDelayChanged(self, value):
        """開鏡延遲改變"""
        if self._config:
            self._config.auto_fire_delay = value

    def _onAlwaysAutoFireChanged(self, checked):
        if self._config:
            self._config.always_auto_fire = checked
            # 啟用持續自動射擊時，自動關閉 idle detect
            if checked:
                self._config.idle_detect_enabled = False

    def _onMouseClickChanged(self, text):
        if self._config:
            self._config.mouse_click_method = text
            if text == "ddxoft":
                try:
                    from win_utils import ensure_ddxoft_ready
                    ensure_ddxoft_ready()
                except ImportError:
                    pass

    def _onFireIntervalChanged(self, value):
        """射擊間隔改變"""
        if self._config:
            self._config.auto_fire_interval = value

    def retranslateUi(self):
        """刷新翻譯"""
        super().retranslateUi()

        # 群組標題
        self.fireGroup.titleLabel.setText(t("keys_and_auto_fire"))

        # 自動射擊設定
        self.fireTargetCard.titleLabel.setText(t("auto_fire_target"))
        self.alwaysAutoFireCard.titleLabel.setText(t("always_auto_fire"))
        self.mouseClickCard.titleLabel.setText(t("mouse_click_method"))
        self.scopeDelayCard.titleLabel.setText(t("scope_delay"))
        self.fireIntervalCard.titleLabel.setText(t("fire_interval"))

        # 更新 ComboBox 內容
        current_target = self.fireTargetCombo.currentIndex()
        self.fireTargetCombo.clear()
        self.fireTargetCombo.addItems([t("head"), t("body"), t("both")])
        self.fireTargetCombo.setCurrentIndex(current_target)

