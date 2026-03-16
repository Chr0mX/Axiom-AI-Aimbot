# other_page.py
"""其他設定頁面 - 關於資訊"""

import os
from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout
from PyQt6.QtGui import QDesktopServices, QIcon
from qfluentwidgets import (
    SettingCardGroup, SettingCard, SwitchSettingCard,
    PushSettingCard, FluentIcon, PrimaryPushButton,
    PushButton, BodyLabel, ComboBox, HyperlinkCard,
    SubtitleLabel, CaptionLabel, isDarkTheme
)

from ..base_page import BasePage
from ..language_manager import t


class OtherPage(BasePage):
    """其他設定頁面"""
    
    def __init__(self, parent=None):
        super().__init__("tab_program_control", parent)
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

        # === 程式控制 ===
        self.programGroup = SettingCardGroup(t("program_control"), self.scrollWidget)

        # 顯示終端視窗
        self.showConsoleCard = SwitchSettingCard(
            FluentIcon.COMMAND_PROMPT,
            t("show_console"),
            "",
            parent=self.programGroup
        )

        # 離開並儲存
        self.exitSaveBtn = PrimaryPushButton(t("exit_and_save"))
        self.exitSaveCard = SettingCard(
            FluentIcon.POWER_BUTTON,
            t("exit_and_save"),
            "",
            self.programGroup
        )
        self.exitSaveCard.hBoxLayout.addWidget(self.exitSaveBtn, 0, Qt.AlignmentFlag.AlignRight)
        self.exitSaveCard.hBoxLayout.addSpacing(16)

        # === 關於內容（無群組標題）===
        self.aboutTitle = SubtitleLabel(t("about_title"))
        self.aboutSubtitle = CaptionLabel(t("about_subtitle"))
        self.aboutSubtitle.setWordWrap(True)
        self.versionLabel = BodyLabel(t("version_info"))

        # 社群連結
        self.communityLabel = BodyLabel(t("community_links"))
        self.communityLabel.setStyleSheet("font-weight: bold; margin-top: 16px;")

        # 社群按鈕
        self.base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        self.discordBtn = PushButton(t("discord"))
        self._updateDiscordIcon()

        self.githubBtn = PushButton(t("github"))
        self.githubBtn.setIcon(FluentIcon.GITHUB)

        self.donateBtn = PushButton(t("donate"))
        self.donateBtn.setIcon(FluentIcon.HEART)
    
    def _initLayout(self):
        """排版所有控制項"""
        # 程式控制
        self.programGroup.addSettingCard(self.showConsoleCard)
        self.programGroup.addSettingCard(self.exitSaveCard)
        self.addContent(self.programGroup)

        # 關於區塊的內容（無群組標題）
        aboutWidget = QWidget()
        aboutWidget.setStyleSheet("background: transparent;")
        aboutLayout = QVBoxLayout(aboutWidget)
        aboutLayout.setContentsMargins(16, 16, 16, 16)
        aboutLayout.setSpacing(8)
        aboutLayout.addWidget(self.aboutTitle)
        aboutLayout.addWidget(self.aboutSubtitle)
        aboutLayout.addWidget(self.versionLabel)
        aboutLayout.addWidget(self.communityLabel)

        # 社群按鈕區
        btnLayout = QHBoxLayout()
        btnLayout.setSpacing(12)
        btnLayout.addWidget(self.discordBtn)
        btnLayout.addWidget(self.githubBtn)
        btnLayout.addWidget(self.donateBtn)
        btnLayout.addStretch(1)
        aboutLayout.addLayout(btnLayout)

        self.scrollLayout.addWidget(aboutWidget)

        self.scrollLayout.addStretch(1)
    
    def _connectSignals(self):
        """連接信號"""
        # 程式控制
        self.showConsoleCard.checkedChanged.connect(self._onShowConsoleChanged)
        self.exitSaveBtn.clicked.connect(self._onExitSave)

        # 社群按鈕
        self.discordBtn.clicked.connect(lambda: QDesktopServices.openUrl(QUrl("https://discord.gg/h4dEh3b8Bt")))
        self.githubBtn.clicked.connect(lambda: QDesktopServices.openUrl(QUrl("https://github.com/iisHong0w0/Axiom-AI-Aimbot")))
        self.donateBtn.clicked.connect(lambda: QDesktopServices.openUrl(QUrl("https://www.paypal.com/paypalme/iis20160512")))
    
    def _loadFromConfig(self):
        """從 Config 載入值"""
        if not self._config:
            return
        
        self.showConsoleCard.setChecked(self._config.show_console)
    
    # === 回調函數 ===
    def _onShowConsoleChanged(self, checked):
        if self._config:
            self._config.show_console = checked
            # 實際顯示/隱藏終端視窗
            try:
                from win_utils.console import show_console, hide_console
                if checked:
                    show_console()
                else:
                    hide_console()
            except Exception as e:
                print(f"[終端控制] 切換終端視窗失敗: {e}")
    
    def _onExitSave(self):
        """離開並儲存"""
        window = self.window()
        if window:
            # 儲存設定
            from core.config import save_config
            if self._config:
                save_config(self._config)
            # 關閉視窗
            window.close()
    
    def _updateDiscordIcon(self):
        """根據當前主題更新 Discord 圖標顏色"""
        if isDarkTheme():
            icon_file = "discord_white.svg"
        else:
            icon_file = "discord.svg"
        icon_path = os.path.join(self.base_path, "assets", icon_file)
        if os.path.exists(icon_path):
            self.discordBtn.setIcon(QIcon(icon_path))

    def retranslateUi(self):
        """刷新翻譯"""
        super().retranslateUi()

        # 群組標題
        self.programGroup.titleLabel.setText(t("program_control"))

        # 程式控制
        self.showConsoleCard.titleLabel.setText(t("show_console"))
        self.exitSaveCard.titleLabel.setText(t("exit_and_save"))
        self.exitSaveBtn.setText(t("exit_and_save"))

        # 關於內容
        self.aboutTitle.setText(t("about_title"))
        self.aboutSubtitle.setText(t("about_subtitle"))
        self.versionLabel.setText(t("version_info"))
        self.communityLabel.setText(t("community_links"))

        # 更新 Discord 圖標
        self._updateDiscordIcon()
