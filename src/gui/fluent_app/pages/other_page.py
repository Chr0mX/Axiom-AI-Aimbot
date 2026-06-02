# other_page.py
"""其他設定頁面 - 關於資訊"""

import os
import sys
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

        # === Application Settings ===
        self.appSettingsGroup = SettingCardGroup(t("app_settings", "Application"), self.scrollWidget)

        self.languageBtn = PushButton(t("change_language", "Change Language"))
        self.languageBtn.setIcon(FluentIcon.LANGUAGE)
        self.languageCard = SettingCard(
            FluentIcon.LANGUAGE,
            t("language_settings", "Language"),
            "",
            self.appSettingsGroup
        )
        self.languageCard.hBoxLayout.addWidget(self.languageBtn, 0, Qt.AlignmentFlag.AlignRight)
        self.languageCard.hBoxLayout.addSpacing(16)

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

        # === TensorRT Environment ===
        self.trtGroup = SettingCardGroup(t("trt_env", "TensorRT Environment"), self.scrollWidget)

        self.trtRecheckBtn = PushButton(t("trt_recheck", "Re-check"))
        self.trtRecheckBtn.setIcon(FluentIcon.SYNC)
        self.trtStatusCard = SettingCard(
            FluentIcon.IOT,
            t("trt_status", "TensorRT Status"),
            t("trt_checking", "Checking…"),
            self.trtGroup,
        )
        self.trtStatusCard.hBoxLayout.addWidget(self.trtRecheckBtn, 0, Qt.AlignmentFlag.AlignRight)
        self.trtStatusCard.hBoxLayout.addSpacing(16)

        self.trtVersionCard = SettingCard(
            FluentIcon.INFO,
            t("trt_version", "TensorRT Version"),
            "—",
            self.trtGroup,
        )

        self.trtLibsCard = SettingCard(
            FluentIcon.FOLDER,
            t("trt_libs_path", "TensorRT DLL Path"),
            "—",
            self.trtGroup,
        )

        self.trtCacheCard = SettingCard(
            FluentIcon.FOLDER,
            t("trt_cache_path", "Engine Cache Path"),
            "—",
            self.trtGroup,
        )

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
        # Application settings
        self.appSettingsGroup.addSettingCard(self.languageCard)
        self.addContent(self.appSettingsGroup)

        # 程式控制
        self.programGroup.addSettingCard(self.showConsoleCard)
        self.programGroup.addSettingCard(self.exitSaveCard)
        self.addContent(self.programGroup)

        # TensorRT Environment
        self.trtGroup.addSettingCard(self.trtStatusCard)
        self.trtGroup.addSettingCard(self.trtVersionCard)
        self.trtGroup.addSettingCard(self.trtLibsCard)
        self.trtGroup.addSettingCard(self.trtCacheCard)
        self.addContent(self.trtGroup)

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
        # Language
        self.languageBtn.clicked.connect(self._onChangeLanguage)

        # 程式控制
        self.showConsoleCard.checkedChanged.connect(self._onShowConsoleChanged)
        self.exitSaveBtn.clicked.connect(self._onExitSave)

        # TensorRT 環境檢查
        self.trtRecheckBtn.clicked.connect(self._checkTensorRT)
        self._checkTensorRT()

        # 社群按鈕
        self.discordBtn.clicked.connect(lambda: QDesktopServices.openUrl(QUrl("https://discord.gg/h4dEh3b8Bt")))
        self.githubBtn.clicked.connect(lambda: QDesktopServices.openUrl(QUrl("https://github.com/iisHong0w0/Axiom-AI-Aimbot")))
        self.donateBtn.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.abspath(os.path.join(self.base_path, "..", "..", "MVP.html")))))
    
    def _loadFromConfig(self):
        """從 Config 載入值"""
        if not self._config:
            return
        
        self.showConsoleCard.setChecked(self._config.show_console)
    
    # === 回調函數 ===
    def _onChangeLanguage(self):
        try:
            from ..components.language_dialog import LanguageDialog
            from ..language_manager import getLanguageManager
            mgr = getLanguageManager()
            current = mgr.currentLanguage
            dlg = LanguageDialog(current, parent=self)
            dlg.languageChanged.connect(mgr.setLanguage)
            dlg.exec()
        except Exception as e:
            print(f"[Language] Failed to open language dialog: {e}")

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
    
    def _findTrtDllPath(self):
        """Locate the TensorRT inference DLL (nvinfer*) from pip-wheel installs."""
        try:
            import site
            site_dirs = list(site.getsitepackages())
            try:
                site_dirs.append(site.getusersitepackages())
            except (AttributeError, NotImplementedError):
                pass
            for sp in site_dirs:
                trt_libs = os.path.join(sp, "tensorrt_libs")
                if os.path.isdir(trt_libs):
                    for name in os.listdir(trt_libs):
                        low = name.lower()
                        if low.startswith("nvinfer") and low.endswith(".dll"):
                            return os.path.join(trt_libs, name)
                    return trt_libs
        except Exception:
            pass
        return None

    def _checkTensorRT(self):
        """Check whether TensorRT is installed and usable, then update the cards."""
        # ONNX Runtime provider availability
        provider_ok = False
        try:
            import onnxruntime as ort
            provider_ok = "TensorrtExecutionProvider" in ort.get_available_providers()
        except Exception:
            provider_ok = False

        # tensorrt python package + version
        trt_version = None
        try:
            import tensorrt as trt  # noqa: F401
            trt_version = getattr(trt, "__version__", "unknown")
        except Exception:
            trt_version = None

        dll_path = self._findTrtDllPath()
        installed = provider_ok and (dll_path is not None)

        if installed:
            self.trtStatusCard.contentLabel.setText(
                t("trt_installed", "✓ Installed — TensorrtExecutionProvider available"))
            self.trtStatusCard.contentLabel.setStyleSheet("color: #2ecc71;")
        elif provider_ok and dll_path is None:
            self.trtStatusCard.contentLabel.setText(
                t("trt_provider_no_dll",
                  "⚠ Provider present but TensorRT DLLs not found — install tensorrt-cu12 wheels"))
            self.trtStatusCard.contentLabel.setStyleSheet("color: #e67e22;")
        else:
            self.trtStatusCard.contentLabel.setText(
                t("trt_not_installed", "✗ Not installed — falls back to CUDA/CPU"))
            self.trtStatusCard.contentLabel.setStyleSheet("color: #e74c3c;")

        self.trtVersionCard.contentLabel.setText(trt_version or "—")
        self.trtLibsCard.contentLabel.setText(dll_path or t("trt_not_found", "Not found"))

        # Engine cache directory path
        project_root = os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        cache_dir = os.path.join(project_root, "trt_cache")
        self.trtCacheCard.contentLabel.setText(cache_dir)

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
        self.appSettingsGroup.titleLabel.setText(t("app_settings", "Application"))
        self.languageCard.titleLabel.setText(t("language_settings", "Language"))
        self.languageBtn.setText(t("change_language", "Change Language"))
        self.programGroup.titleLabel.setText(t("program_control"))

        # 程式控制
        self.showConsoleCard.titleLabel.setText(t("show_console"))
        self.exitSaveCard.titleLabel.setText(t("exit_and_save"))
        self.exitSaveBtn.setText(t("exit_and_save"))

        # TensorRT 環境
        self.trtGroup.titleLabel.setText(t("trt_env", "TensorRT Environment"))
        self.trtStatusCard.titleLabel.setText(t("trt_status", "TensorRT Status"))
        self.trtVersionCard.titleLabel.setText(t("trt_version", "TensorRT Version"))
        self.trtLibsCard.titleLabel.setText(t("trt_libs_path", "TensorRT DLL Path"))
        self.trtCacheCard.titleLabel.setText(t("trt_cache_path", "Engine Cache Path"))
        self.trtRecheckBtn.setText(t("trt_recheck", "Re-check"))
        self._checkTensorRT()

        # 關於內容
        self.aboutTitle.setText(t("about_title"))
        self.aboutSubtitle.setText(t("about_subtitle"))
        self.versionLabel.setText(t("version_info"))
        self.communityLabel.setText(t("community_links"))

        # 更新 Discord 圖標
        self._updateDiscordIcon()
