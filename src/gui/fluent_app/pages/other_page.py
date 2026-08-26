# other_page.py
"""其他設定頁面 - 關於資訊"""

import os
import sys
from PyQt6.QtCore import Qt, QUrl, QTimer
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QApplication
from PyQt6.QtGui import QDesktopServices, QIcon
from qfluentwidgets import (
    SettingCardGroup, SettingCard, SwitchSettingCard,
    PushSettingCard, FluentIcon, PrimaryPushButton,
    PushButton, BodyLabel, ComboBox, HyperlinkCard,
    SubtitleLabel, CaptionLabel, isDarkTheme
)

from ..base_page import BasePage
from ..language_manager import t
from ..theme_colors import ThemeColors
from ..components.no_wheel_widgets import NoWheelSpinBox
from win_utils.makcu_mouse import makcu_mouse as _makcu_mouse, is_makcu_connected
from version import __version__


class OtherPage(BasePage):
    """其他設定頁面"""
    
    def __init__(self, parent=None):
        super().__init__("tab_program_control", parent)
        self._config = None
        # Debounces a Web Control server restart after a port-field edit —
        # same reasoning as visuals_page.py's Web ESP restart timer: a
        # restart is a real socket rebind, not a cheap config write.
        self._webControlRestartTimer = QTimer(self)
        self._webControlRestartTimer.setSingleShot(True)
        self._webControlRestartTimer.setInterval(600)
        self._webControlRestartTimer.timeout.connect(self._restartWebControlIfRunning)
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

        # === Performance ===
        self.perfGroup = SettingCardGroup(t("performance_settings", "Performance"), self.scrollWidget)

        self.threadPriorityCombo = ComboBox()
        self.threadPriorityCombo.addItems(["Normal", "Above Normal", "High", "Time Critical"])
        self.threadPriorityCombo.setMinimumWidth(150)
        self.threadPriorityCard = SettingCard(
            FluentIcon.SPEED_HIGH,
            t("thread_priority", "Thread Priority"),
            t("thread_priority_desc", "CPU priority for inference, capture and preprocess threads. Takes effect on next inference start."),
            self.perfGroup
        )
        self.threadPriorityCard.hBoxLayout.addWidget(self.threadPriorityCombo, 0, Qt.AlignmentFlag.AlignRight)
        self.threadPriorityCard.hBoxLayout.addSpacing(16)

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

        # === Remote Control (Web Control API) ===
        # A control-plane LAN API — lets a browser on another PC call the
        # same main-function actions the Qt GUI does (see
        # core/app_controller.py). Sibling feature to the Web ESP overlay
        # above; kept in its own group since it's a different concern
        # (control, not just viewing) with its own auth token.
        self.webControlGroup = SettingCardGroup(t("web_control_settings", "Remote Control"), self.scrollWidget)

        self.webControlEnableCard = SwitchSettingCard(
            FluentIcon.GLOBE,
            t("web_control_enabled", "Enable Web Control"),
            t("web_control_desc", "Let a browser on your LAN control main functions (always-aim, status) over a token-authenticated API."),
            parent=self.webControlGroup
        )

        self.webControlPortSpin = NoWheelSpinBox()
        self.webControlPortSpin.setRange(1024, 65535)
        self.webControlPortCard = SettingCard(
            FluentIcon.WIFI,
            t("web_control_port", "Port"),
            t("web_control_port_desc", "Port the control API listens on."),
            self.webControlGroup
        )
        self.webControlPortCard.hBoxLayout.addWidget(self.webControlPortSpin, 0, Qt.AlignmentFlag.AlignRight)
        self.webControlPortCard.hBoxLayout.addSpacing(16)

        self.webControlCopyTokenBtn = PushButton(t("web_control_copy", "Copy"))
        self.webControlCopyTokenBtn.setIcon(FluentIcon.COPY)
        self.webControlRegenTokenBtn = PushButton(t("web_control_regenerate", "Regenerate"))
        self.webControlRegenTokenBtn.setIcon(FluentIcon.SYNC)
        self.webControlTokenCard = SettingCard(
            FluentIcon.CERTIFICATE,
            t("web_control_token", "Access Token"),
            t("web_control_token_desc", "Paste this into the web client so it's allowed to send commands."),
            self.webControlGroup,
        )
        self.webControlTokenCard.hBoxLayout.addWidget(self.webControlCopyTokenBtn, 0, Qt.AlignmentFlag.AlignRight)
        self.webControlTokenCard.hBoxLayout.addSpacing(8)
        self.webControlTokenCard.hBoxLayout.addWidget(self.webControlRegenTokenBtn, 0, Qt.AlignmentFlag.AlignRight)
        self.webControlTokenCard.hBoxLayout.addSpacing(16)

        self.webControlOpenBtn = PushButton(t("web_control_open", "Open in browser"))
        self.webControlConnectCard = SettingCard(
            FluentIcon.LINK,
            t("web_control_connect", "Connect"),
            t("web_control_connect_desc", "Open this URL on the same PC, or on any device on your LAN."),
            self.webControlGroup,
        )
        self.webControlConnectCard.hBoxLayout.addWidget(self.webControlOpenBtn, 0, Qt.AlignmentFlag.AlignRight)
        self.webControlConnectCard.hBoxLayout.addSpacing(16)

        # === Environment — TensorRT ===
        self.trtGroup = SettingCardGroup(t("env_trt", "TensorRT"), self.scrollWidget)

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

        self.trtAppdataCard = SettingCard(
            FluentIcon.FOLDER,
            "AppData Packages Path",
            "—",
            self.trtGroup,
        )

        # === Environment — DirectML ===
        self.dmlGroup = SettingCardGroup(t("env_dml", "DirectML"), self.scrollWidget)

        self.dmlStatusCard = SettingCard(
            FluentIcon.IOT,
            t("dml_status", "DirectML Status"),
            t("dml_checking", "Checking…"),
            self.dmlGroup,
        )

        self.dmlDllCard = SettingCard(
            FluentIcon.FOLDER,
            t("dml_dll_path", "DirectML DLL Path"),
            "—",
            self.dmlGroup,
        )

        self.dmlEmbeddedPathCard = SettingCard(
            FluentIcon.FOLDER,
            t("dml_embedded_path", "Embedded ORT-DirectML Path"),
            "—",
            self.dmlGroup,
        )

        # === MAKCU Hardware ===
        self.makcuHwGroup = SettingCardGroup(t("makcu_hw_info", "MAKCU Hardware"), self.scrollWidget)

        self.makcuHwStatusCard = SettingCard(
            FluentIcon.IOT, t("makcu_hw_status", "Status"), "—", self.makcuHwGroup)

        self.makcuHwPortCard   = SettingCard(FluentIcon.WIFI,            t("makcu_hw_port",   "COM Port"),     "—", self.makcuHwGroup)
        self.makcuHwBaudCard   = SettingCard(FluentIcon.SPEED_HIGH,      t("makcu_hw_baud",   "Baud Rate"),    "—", self.makcuHwGroup)
        self.makcuHwVerCard    = SettingCard(FluentIcon.TAG,             t("makcu_hw_ver",    "Version"),      "—", self.makcuHwGroup)
        self.makcuHwModelCard  = SettingCard(FluentIcon.DEVELOPER_TOOLS, t("makcu_hw_model",  "Model"),        "—", self.makcuHwGroup)
        self.makcuHwVendorCard = SettingCard(FluentIcon.GLOBE,           t("makcu_hw_vendor", "Vendor"),       "—", self.makcuHwGroup)
        self.makcuHwTempCard   = SettingCard(FluentIcon.CALORIES,        t("makcu_hw_temp",   "Temperature"),  "—", self.makcuHwGroup)
        _grey = f"color: {ThemeColors.TEXT_TERTIARY.get()};"
        for _card in (self.makcuHwVerCard, self.makcuHwModelCard,
                      self.makcuHwVendorCard, self.makcuHwTempCard):
            _card.contentLabel.setStyleSheet(_grey)
            _card.titleLabel.setStyleSheet(_grey)

        # === Environment — Python Path ===
        self.pyGroup = SettingCardGroup(t("env_python", "Python Path"), self.scrollWidget)

        self.trtSysPythonCard = SettingCard(
            FluentIcon.COMMAND_PROMPT,
            "System Python",
            "—",
            self.pyGroup,
        )

        self.trtInternalPythonCard = SettingCard(
            FluentIcon.COMMAND_PROMPT,
            "Internal Python",
            "—",
            self.pyGroup,
        )

        # === 關於內容（無群組標題）===
        self.aboutTitle = SubtitleLabel(t("about_title"))
        self.aboutSubtitle = CaptionLabel(t("about_subtitle"))
        self.aboutSubtitle.setWordWrap(True)
        self.versionLabel = BodyLabel(f"{t('version_info')} {__version__}")

        # 社群連結
        self.communityLabel = BodyLabel(t("community_links"))
        self.communityLabel.setStyleSheet("font-weight: bold; margin-top: 16px;")

        # 社群按鈕
        self.base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        self.discordBtn = PushButton(t("discord"))
        self._updateDiscordIcon()

        self.githubBtn = PushButton(t("github"))
        self.githubBtn.setIcon(FluentIcon.GITHUB)

        self.discordOriginalBtn = PushButton(t("discord_original", "Discord (Original)"))
        self._updateDiscordOriginalIcon()

        self.githubOriginalBtn = PushButton(t("github_original", "Github (Original)"))
        self.githubOriginalBtn.setIcon(FluentIcon.GITHUB)

        self.donateBtn = PushButton(t("donate"))
        self.donateBtn.setIcon(FluentIcon.HEART)
    
    def _initLayout(self):
        """排版所有控制項"""
        # Application settings
        self.appSettingsGroup.addSettingCard(self.languageCard)
        self.addContent(self.appSettingsGroup)

        # Performance
        self.perfGroup.addSettingCard(self.threadPriorityCard)
        self.addContent(self.perfGroup)

        # 程式控制
        self.programGroup.addSettingCard(self.showConsoleCard)
        self.programGroup.addSettingCard(self.exitSaveCard)
        self.addContent(self.programGroup)

        # Remote Control
        self.webControlGroup.addSettingCard(self.webControlEnableCard)
        self.webControlGroup.addSettingCard(self.webControlPortCard)
        self.webControlGroup.addSettingCard(self.webControlTokenCard)
        self.webControlGroup.addSettingCard(self.webControlConnectCard)
        self.addContent(self.webControlGroup)

        # Environment — TensorRT
        self.trtGroup.addSettingCard(self.trtStatusCard)
        self.trtGroup.addSettingCard(self.trtVersionCard)
        self.trtGroup.addSettingCard(self.trtLibsCard)
        self.trtGroup.addSettingCard(self.trtCacheCard)
        self.trtGroup.addSettingCard(self.trtAppdataCard)
        self.addContent(self.trtGroup)

        # Environment — DirectML
        self.dmlGroup.addSettingCard(self.dmlStatusCard)
        self.dmlGroup.addSettingCard(self.dmlDllCard)
        self.dmlGroup.addSettingCard(self.dmlEmbeddedPathCard)
        self.addContent(self.dmlGroup)

        # Environment — Python Path
        self.pyGroup.addSettingCard(self.trtSysPythonCard)
        self.pyGroup.addSettingCard(self.trtInternalPythonCard)
        self.addContent(self.pyGroup)

        # MAKCU Hardware
        self.makcuHwGroup.addSettingCard(self.makcuHwStatusCard)
        self.makcuHwGroup.addSettingCard(self.makcuHwPortCard)
        self.makcuHwGroup.addSettingCard(self.makcuHwBaudCard)
        self.makcuHwGroup.addSettingCard(self.makcuHwVerCard)
        self.makcuHwGroup.addSettingCard(self.makcuHwModelCard)
        self.makcuHwGroup.addSettingCard(self.makcuHwVendorCard)
        self.makcuHwGroup.addSettingCard(self.makcuHwTempCard)
        self.addContent(self.makcuHwGroup)

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
        btnLayout.addWidget(self.discordOriginalBtn)
        btnLayout.addWidget(self.githubOriginalBtn)
        btnLayout.addWidget(self.donateBtn)
        btnLayout.addStretch(1)
        aboutLayout.addLayout(btnLayout)

        self.scrollLayout.addWidget(aboutWidget)

        self.scrollLayout.addStretch(1)
    
    def _connectSignals(self):
        """連接信號"""
        # Language
        self.languageBtn.clicked.connect(self._onChangeLanguage)

        # Performance
        self.threadPriorityCombo.currentTextChanged.connect(self._onThreadPriorityChanged)

        # 程式控制
        self.showConsoleCard.checkedChanged.connect(self._onShowConsoleChanged)
        self.exitSaveBtn.clicked.connect(self._onExitSave)

        # Remote Control (Web Control API)
        self.webControlEnableCard.checkedChanged.connect(self._onWebControlEnableChanged)
        self.webControlPortSpin.valueChanged.connect(self._onWebControlPortChanged)
        self.webControlCopyTokenBtn.clicked.connect(self._onWebControlCopyToken)
        self.webControlRegenTokenBtn.clicked.connect(self._onWebControlRegenerateToken)
        self.webControlOpenBtn.clicked.connect(self._onWebControlOpen)

        # TensorRT 環境檢查
        self.trtRecheckBtn.clicked.connect(self._checkTensorRT)
        self._checkTensorRT()

        # MAKCU Hardware — auto-refresh every 3 s
        from PyQt6.QtCore import QTimer as _QTimer
        self._makcuHwTimer = _QTimer(self)
        self._makcuHwTimer.timeout.connect(self._refreshMakcuHwInfo)
        self._makcuHwTimer.start(3000)

        # 社群按鈕
        self.discordBtn.clicked.connect(lambda: QDesktopServices.openUrl(QUrl("https://discord.gg/DpcqaQEj5b")))
        self.githubBtn.clicked.connect(lambda: QDesktopServices.openUrl(QUrl("https://github.com/Chr0mX/Axiom-AI-Aimbot")))
        self.discordOriginalBtn.clicked.connect(lambda: QDesktopServices.openUrl(QUrl("https://discord.gg/h4dEh3b8Bt")))
        self.githubOriginalBtn.clicked.connect(lambda: QDesktopServices.openUrl(QUrl("https://github.com/iisHong0w0/Axiom-AI-Aimbot")))
        self.donateBtn.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.abspath(os.path.join(self.base_path, "..", "MVP.html")))))
    
    def _loadFromConfig(self):
        """從 Config 載入值"""
        if not self._config:
            return

        self.showConsoleCard.setChecked(self._config.show_console)

        _prio_rev = {"normal": "Normal", "above_normal": "Above Normal",
                     "high": "High", "time_critical": "Time Critical"}
        self.threadPriorityCombo.blockSignals(True)
        self.threadPriorityCombo.setCurrentText(
            _prio_rev.get(getattr(self._config, 'thread_priority', 'high'), "High"))
        self.threadPriorityCombo.blockSignals(False)

        web_control_on = bool(getattr(self._config, 'web_control_enabled', False))
        self.webControlEnableCard.setChecked(web_control_on)
        self.webControlPortSpin.blockSignals(True)
        self.webControlPortSpin.setValue(int(getattr(self._config, 'web_control_port', 8090)))
        self.webControlPortSpin.blockSignals(False)
        self.webControlTokenCard.contentLabel.setText(
            getattr(self._config, 'web_control_token', '') or t("web_control_token_none", "(generated on first enable)"))
        self.webControlPortCard.setEnabled(web_control_on)
        self.webControlTokenCard.setEnabled(web_control_on)
        self.webControlConnectCard.setEnabled(web_control_on)
        self._refreshWebControlConnect()

        self._refreshMakcuHwInfo()
    
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

    def _onThreadPriorityChanged(self, text):
        if not self._config:
            return
        _prio_map = {"Normal": "normal", "Above Normal": "above_normal",
                     "High": "high", "Time Critical": "time_critical"}
        self._config.thread_priority = _prio_map.get(text, "high")

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

    # === Remote Control (Web Control API) ===
    def _onWebControlEnableChanged(self, checked):
        if self._config:
            self._config.web_control_enabled = bool(checked)
        self.webControlPortCard.setEnabled(checked)
        self.webControlTokenCard.setEnabled(checked)
        self.webControlConnectCard.setEnabled(checked)
        try:
            from core import web_control_server
            if checked:
                web_control_server.start(self._config)
                # start() auto-generates a token the first time it's empty —
                # reflect whatever it actually ended up with.
                self.webControlTokenCard.contentLabel.setText(
                    getattr(self._config, 'web_control_token', '') or "—")
            else:
                web_control_server.stop()
        except Exception as e:
            print(f"[WebControl] failed to start/stop: {e}")
        self._refreshWebControlConnect()

    def _onWebControlPortChanged(self, value):
        if self._config:
            self._config.web_control_port = int(value)
        self._webControlRestartTimer.start()

    def _restartWebControlIfRunning(self):
        # web_control_server.start() only binds a socket once — changing the
        # port on a live config does nothing on its own until an explicit
        # stop()+start() cycle, same as esp_server.py's own restart-on-
        # port-change path (visuals_page.py's _restartWebEspIfRunning).
        try:
            from core import web_control_server
            if web_control_server.is_running():
                web_control_server.stop()
                web_control_server.start(self._config)
        except Exception as e:
            print(f"[WebControl] failed to restart: {e}")
        self._refreshWebControlConnect()

    def _onWebControlRegenerateToken(self):
        if not self._config:
            return
        import secrets
        self._config.web_control_token = secrets.token_urlsafe(24)
        self.webControlTokenCard.contentLabel.setText(self._config.web_control_token)
        try:
            from core import web_control_server
            if web_control_server.is_running():
                web_control_server.stop()
                web_control_server.start(self._config)
        except Exception as e:
            print(f"[WebControl] failed to apply new token: {e}")

    def _onWebControlCopyToken(self):
        if not self._config:
            return
        QApplication.clipboard().setText(getattr(self._config, 'web_control_token', '') or '')

    def _onWebControlOpen(self):
        try:
            from core import web_control_server
            url = web_control_server.connect_url()
        except Exception:
            url = ""
        if url:
            QDesktopServices.openUrl(QUrl(url))

    def _refreshWebControlConnect(self):
        try:
            from core import web_control_server
            running = web_control_server.is_running()
        except Exception:
            running = False
        self.webControlOpenBtn.setEnabled(running)

    def _refreshMakcuHwInfo(self):
        dash = "—"
        try:
            connected = is_makcu_connected()
        except Exception:
            connected = False

        if connected:
            self.makcuHwStatusCard.contentLabel.setText(t("connected", "Connected"))
            self.makcuHwStatusCard.contentLabel.setStyleSheet(f"color: {ThemeColors.SUCCESS.get()};")
        else:
            self.makcuHwStatusCard.contentLabel.setText(t("disconnected", "Disconnected"))
            self.makcuHwStatusCard.contentLabel.setStyleSheet(f"color: {ThemeColors.ERROR.get()};")

        port = getattr(_makcu_mouse, '_com_port', '') or (
            getattr(self._config, 'makcu_com_port', '') if self._config else '')
        self.makcuHwPortCard.contentLabel.setText(port or dash)

        baud = getattr(_makcu_mouse, '_baud_rate', None)
        if not baud:
            baud = getattr(self._config, 'makcu_baud_rate', 115200) if self._config else 115200
        self.makcuHwBaudCard.contentLabel.setText(f"{int(baud):,}")

        if connected:
            try:
                _makcu_mouse.query_info()
            except Exception:
                pass

        ver  = getattr(_makcu_mouse, 'version_string', '')
        info = getattr(_makcu_mouse, 'device_info', {})
        if not ver:
            ver = info.get('VERSION', dash)
        self.makcuHwVerCard.contentLabel.setText(ver or dash)
        self.makcuHwModelCard.contentLabel.setText(info.get('MODEL', dash))
        self.makcuHwVendorCard.contentLabel.setText(info.get('VENDOR', dash))
        temp = info.get('TEMP', '')
        self.makcuHwTempCard.contentLabel.setText(f"{temp} °C" if temp else dash)

    def _findTrtDllPath(self):
        """Locate the TensorRT inference DLL (nvinfer*) from pip-wheel installs."""
        try:
            # Check AppData AxiomAI path first (primary install location)
            localappdata = os.environ.get("LOCALAPPDATA", "")
            if localappdata:
                trt_libs = os.path.join(localappdata, "AxiomAI", "site-packages", "tensorrt_libs")
                if os.path.isdir(trt_libs):
                    for name in os.listdir(trt_libs):
                        low = name.lower()
                        if low.startswith("nvinfer") and low.endswith(".dll"):
                            return os.path.join(trt_libs, name)
                    return trt_libs

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
        """Check whether TensorRT is installed and usable, then update the cards.

        Despite the name, this also refreshes the DirectML status/DLL-path/
        embedded-path content labels (dmlStatusCard/dmlDllCard/
        dmlEmbeddedPathCard) — the two environment checks share one method.
        """
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
            import tensorrt as _trt
            trt_version = getattr(_trt, "__version__", None)
        except Exception:
            pass

        if not trt_version:
            dll = self._findTrtDllPath()
            if dll and sys.platform == "win32":
                try:
                    import win32api
                    info = win32api.GetFileVersionInfo(dll, "\\")
                    ms, ls = info["FileVersionMS"], info["FileVersionLS"]
                    trt_version = f"{ms >> 16}.{ms & 0xFFFF}.{ls >> 16}.{ls & 0xFFFF}"
                except Exception:
                    pass
            if not trt_version and dll:
                import re as _re
                m = _re.search(r"nvinfer[_-](\d+[\.\d]*)", os.path.basename(dll))
                if m:
                    trt_version = m.group(1)

        dll_path = self._findTrtDllPath()
        installed = provider_ok and (dll_path is not None)

        if installed:
            self.trtStatusCard.contentLabel.setText(
                t("trt_installed", "✓ Installed — TensorrtExecutionProvider available"))
            self.trtStatusCard.contentLabel.setStyleSheet(f"color: {ThemeColors.SUCCESS.get()};")
        elif provider_ok and dll_path is None:
            self.trtStatusCard.contentLabel.setText(
                t("trt_provider_no_dll",
                  "⚠ Provider present but TensorRT DLLs not found — install tensorrt-cu12 wheels"))
            self.trtStatusCard.contentLabel.setStyleSheet(f"color: {ThemeColors.WARNING.get()};")
        else:
            self.trtStatusCard.contentLabel.setText(
                t("trt_not_installed", "✗ Not installed — falls back to CUDA/CPU"))
            self.trtStatusCard.contentLabel.setStyleSheet(f"color: {ThemeColors.ERROR.get()};")

        self.trtVersionCard.contentLabel.setText(trt_version or "—")
        self.trtLibsCard.contentLabel.setText(dll_path or t("trt_not_found", "Not found"))

        # Engine cache directory path
        project_root = os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        cache_dir = os.path.join(project_root, "trt_cache")
        self.trtCacheCard.contentLabel.setText(cache_dir)
        self.trtCacheCard.contentLabel.setStyleSheet(f"color: {ThemeColors.TEXT_PRIMARY.get()};")

        # AppData packages path
        localappdata = os.environ.get("LOCALAPPDATA", "")
        appdata_pkg = os.path.join(localappdata, "AxiomAI", "site-packages") if localappdata else ""
        if appdata_pkg and os.path.isdir(appdata_pkg):
            self.trtAppdataCard.contentLabel.setText(appdata_pkg)
            self.trtAppdataCard.contentLabel.setStyleSheet(f"color: {ThemeColors.SUCCESS.get()};")
        else:
            self.trtAppdataCard.contentLabel.setText(appdata_pkg or "LOCALAPPDATA not set")
            self.trtAppdataCard.contentLabel.setStyleSheet(f"color: {ThemeColors.ERROR.get()};")

        # Compute DML paths early so status block can use them
        src_dir = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))))
        embedded_site = os.path.join(src_dir, "python", "Lib", "site-packages")
        dml_dll = os.path.join(embedded_site, "onnxruntime", "capi", "DirectML.dll")
        dml_installed = os.path.exists(dml_dll)

        # DirectML status
        try:
            import onnxruntime as _ort2
            dml_ok = "DmlExecutionProvider" in _ort2.get_available_providers()
        except Exception:
            dml_ok = False
        configured_backend = os.environ.get("AXIOM_BACKEND", "auto")
        if dml_ok:
            self.dmlStatusCard.contentLabel.setText(
                t("dml_available", "✓ DmlExecutionProvider active"))
            self.dmlStatusCard.contentLabel.setStyleSheet(f"color: {ThemeColors.SUCCESS.get()};")
        elif dml_installed:
            self.dmlStatusCard.contentLabel.setText(
                t("dml_installed_inactive", "✓ Installed — set backend to DirectML to activate"))
            self.dmlStatusCard.contentLabel.setStyleSheet(f"color: {ThemeColors.SUCCESS.get()};")
        elif configured_backend == "directml":
            self.dmlStatusCard.contentLabel.setText(
                t("dml_restart_required", "⚠ Restart required — DirectML takes effect on next launch"))
            self.dmlStatusCard.contentLabel.setStyleSheet(f"color: {ThemeColors.WARNING.get()};")
        else:
            self.dmlStatusCard.contentLabel.setText(
                t("dml_not_available", "✗ Not installed"))
            self.dmlStatusCard.contentLabel.setStyleSheet(f"color: {ThemeColors.ERROR.get()};")

        # DirectML DLL path
        if os.path.exists(dml_dll):
            self.dmlDllCard.contentLabel.setText(dml_dll)
            self.dmlDllCard.contentLabel.setStyleSheet(f"color: {ThemeColors.SUCCESS.get()};")
        else:
            self.dmlDllCard.contentLabel.setText(t("dml_not_found", "Not found"))
            self.dmlDllCard.contentLabel.setStyleSheet(f"color: {ThemeColors.ERROR.get()};")

        # Embedded onnxruntime-directml site-packages path
        if os.path.isdir(embedded_site):
            self.dmlEmbeddedPathCard.contentLabel.setText(embedded_site)
            self.dmlEmbeddedPathCard.contentLabel.setStyleSheet(f"color: {ThemeColors.SUCCESS.get()};")
        else:
            self.dmlEmbeddedPathCard.contentLabel.setText(t("dml_not_found", "Not found"))
            self.dmlEmbeddedPathCard.contentLabel.setStyleSheet(f"color: {ThemeColors.ERROR.get()};")

        # System Python — search Windows PATH, skipping the embedded interpreter
        import shutil as _shutil
        embedded_python_dir = os.path.abspath(os.path.join(project_root, "python"))
        sys_python = None
        for _name in ("python3", "python"):
            _found = _shutil.which(_name)
            if _found and not os.path.abspath(_found).startswith(embedded_python_dir):
                sys_python = _found
                break
        self.trtSysPythonCard.contentLabel.setText(sys_python or "Not found in PATH")
        self.trtSysPythonCard.contentLabel.setStyleSheet(
            f"color: {ThemeColors.SUCCESS.get()};" if sys_python else f"color: {ThemeColors.ERROR.get()};")

        # Internal/Embedded Python (<project_root>/python/python.exe)
        internal_python = os.path.join(project_root, "python", "python.exe")
        if os.path.exists(internal_python):
            self.trtInternalPythonCard.contentLabel.setText(internal_python)
            self.trtInternalPythonCard.contentLabel.setStyleSheet(f"color: {ThemeColors.SUCCESS.get()};")
        else:
            self.trtInternalPythonCard.contentLabel.setText(f"{internal_python}  (not found)")
            self.trtInternalPythonCard.contentLabel.setStyleSheet(f"color: {ThemeColors.WARNING.get()};")

    def _updateDiscordIcon(self):
        """根據當前主題更新 Discord 圖標顏色"""
        if isDarkTheme():
            icon_file = "discord_white.svg"
        else:
            icon_file = "discord.svg"
        icon_path = os.path.join(self.base_path, "assets", icon_file)
        if os.path.exists(icon_path):
            self.discordBtn.setIcon(QIcon(icon_path))

    def _updateDiscordOriginalIcon(self):
        """根據當前主題更新 Discord (Original) 圖標顏色"""
        if isDarkTheme():
            icon_file = "discord_white.svg"
        else:
            icon_file = "discord.svg"
        icon_path = os.path.join(self.base_path, "assets", icon_file)
        if os.path.exists(icon_path):
            self.discordOriginalBtn.setIcon(QIcon(icon_path))

    def retranslateUi(self):
        """刷新翻譯"""
        super().retranslateUi()

        # 群組標題
        self.appSettingsGroup.titleLabel.setText(t("app_settings", "Application"))
        self.languageCard.titleLabel.setText(t("language_settings", "Language"))
        self.languageBtn.setText(t("change_language", "Change Language"))
        self.programGroup.titleLabel.setText(t("program_control"))

        # Performance
        self.perfGroup.titleLabel.setText(t("performance_settings", "Performance"))
        self.threadPriorityCard.titleLabel.setText(t("thread_priority", "Thread Priority"))
        self.threadPriorityCard.contentLabel.setText(
            t("thread_priority_desc", "CPU priority for inference, capture and NDI receive threads. Takes effect on next inference start."))

        # 程式控制
        self.showConsoleCard.titleLabel.setText(t("show_console"))
        self.exitSaveCard.titleLabel.setText(t("exit_and_save"))
        self.exitSaveBtn.setText(t("exit_and_save"))

        # Remote Control (Web Control API)
        self.webControlGroup.titleLabel.setText(t("web_control_settings", "Remote Control"))
        self.webControlEnableCard.titleLabel.setText(t("web_control_enabled", "Enable Web Control"))
        self.webControlEnableCard.contentLabel.setText(
            t("web_control_desc", "Let a browser on your LAN control main functions (always-aim, status) over a token-authenticated API."))
        self.webControlPortCard.titleLabel.setText(t("web_control_port", "Port"))
        self.webControlPortCard.contentLabel.setText(t("web_control_port_desc", "Port the control API listens on."))
        self.webControlCopyTokenBtn.setText(t("web_control_copy", "Copy"))
        self.webControlRegenTokenBtn.setText(t("web_control_regenerate", "Regenerate"))
        self.webControlTokenCard.titleLabel.setText(t("web_control_token", "Access Token"))
        self.webControlTokenCard.contentLabel.setText(
            (getattr(self._config, 'web_control_token', '') if self._config else '')
            or t("web_control_token_none", "(generated on first enable)"))
        self.webControlOpenBtn.setText(t("web_control_open", "Open in browser"))
        self.webControlConnectCard.titleLabel.setText(t("web_control_connect", "Connect"))
        self.webControlConnectCard.contentLabel.setText(
            t("web_control_connect_desc", "Open this URL on the same PC, or on any device on your LAN."))

        # TensorRT 環境
        self.trtGroup.titleLabel.setText(t("trt_env", "TensorRT Environment"))
        self.trtStatusCard.titleLabel.setText(t("trt_status", "TensorRT Status"))
        self.trtVersionCard.titleLabel.setText(t("trt_version", "TensorRT Version"))
        self.trtLibsCard.titleLabel.setText(t("trt_libs_path", "TensorRT DLL Path"))
        self.trtCacheCard.titleLabel.setText(t("trt_cache_path", "Engine Cache Path"))
        self.trtRecheckBtn.setText(t("trt_recheck", "Re-check"))

        # DirectML — title/description labels only; content labels are
        # refreshed by _checkTensorRT() below (it checks DML too despite
        # the name — see that method).
        self.dmlGroup.titleLabel.setText(t("env_dml", "DirectML"))
        self.dmlStatusCard.titleLabel.setText(t("dml_status", "DirectML Status"))
        self.dmlDllCard.titleLabel.setText(t("dml_dll_path", "DirectML DLL Path"))
        self.dmlEmbeddedPathCard.titleLabel.setText(t("dml_embedded_path", "Embedded ORT-DirectML Path"))

        self._checkTensorRT()

        # 關於內容
        self.aboutTitle.setText(t("about_title"))
        self.aboutSubtitle.setText(t("about_subtitle"))
        self.versionLabel.setText(f"{t('version_info')} {__version__}")
        self.communityLabel.setText(t("community_links"))
        self.discordBtn.setText(t("discord"))
        self.githubBtn.setText(t("github"))
        self.discordOriginalBtn.setText(t("discord_original", "Discord (Original)"))
        self.githubOriginalBtn.setText(t("github_original", "Github (Original)"))
        self.donateBtn.setText(t("donate"))

        # MAKCU Hardware
        self.makcuHwGroup.titleLabel.setText(t("makcu_hw_info", "MAKCU Hardware"))
        self.makcuHwStatusCard.titleLabel.setText(t("makcu_hw_status", "Status"))
        self.makcuHwPortCard.titleLabel.setText(t("makcu_hw_port", "COM Port"))
        self.makcuHwBaudCard.titleLabel.setText(t("makcu_hw_baud", "Baud Rate"))
        self.makcuHwVerCard.titleLabel.setText(t("makcu_hw_ver", "Version"))
        self.makcuHwModelCard.titleLabel.setText(t("makcu_hw_model", "Model"))
        self.makcuHwVendorCard.titleLabel.setText(t("makcu_hw_vendor", "Vendor"))
        self.makcuHwTempCard.titleLabel.setText(t("makcu_hw_temp", "Temperature"))

        # 更新 Discord 圖標
        self._updateDiscordIcon()
        self._updateDiscordOriginalIcon()
