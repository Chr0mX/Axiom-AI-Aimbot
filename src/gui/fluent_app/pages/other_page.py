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
from win_utils.makcu_mouse import makcu_mouse as _makcu_mouse, is_makcu_connected


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
        _grey = "color: #888888;"
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

        # MAKCU Hardware — auto-refresh every 3 s
        from PyQt6.QtCore import QTimer as _QTimer
        self._makcuHwTimer = _QTimer(self)
        self._makcuHwTimer.timeout.connect(self._refreshMakcuHwInfo)
        self._makcuHwTimer.start(3000)

        # 社群按鈕
        self.discordBtn.clicked.connect(lambda: QDesktopServices.openUrl(QUrl("https://discord.gg/h4dEh3b8Bt")))
        self.githubBtn.clicked.connect(lambda: QDesktopServices.openUrl(QUrl("https://github.com/iisHong0w0/Axiom-AI-Aimbot")))
        self.donateBtn.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.abspath(os.path.join(self.base_path, "..", "..", "MVP.html")))))
    
    def _loadFromConfig(self):
        """從 Config 載入值"""
        if not self._config:
            return

        self.showConsoleCard.setChecked(self._config.show_console)

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
    
    def _refreshMakcuHwInfo(self):
        dash = "—"
        try:
            connected = is_makcu_connected()
        except Exception:
            connected = False

        if connected:
            self.makcuHwStatusCard.contentLabel.setText(t("connected", "Connected"))
            self.makcuHwStatusCard.contentLabel.setStyleSheet("color: #2ecc71;")
        else:
            self.makcuHwStatusCard.contentLabel.setText(t("disconnected", "Disconnected"))
            self.makcuHwStatusCard.contentLabel.setStyleSheet("color: #e74c3c;")

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

        # AppData packages path
        localappdata = os.environ.get("LOCALAPPDATA", "")
        appdata_pkg = os.path.join(localappdata, "AxiomAI", "site-packages") if localappdata else ""
        if appdata_pkg and os.path.isdir(appdata_pkg):
            self.trtAppdataCard.contentLabel.setText(appdata_pkg)
            self.trtAppdataCard.contentLabel.setStyleSheet("color: #2ecc71;")
        else:
            self.trtAppdataCard.contentLabel.setText(appdata_pkg or "LOCALAPPDATA not set")
            self.trtAppdataCard.contentLabel.setStyleSheet("color: #e74c3c;")

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
            self.dmlStatusCard.contentLabel.setStyleSheet("color: #2ecc71;")
        elif dml_installed:
            self.dmlStatusCard.contentLabel.setText(
                t("dml_installed_inactive", "✓ Installed — set backend to DirectML to activate"))
            self.dmlStatusCard.contentLabel.setStyleSheet("color: #2ecc71;")
        elif configured_backend == "directml":
            self.dmlStatusCard.contentLabel.setText(
                t("dml_restart_required", "⚠ Restart required — DirectML takes effect on next launch"))
            self.dmlStatusCard.contentLabel.setStyleSheet("color: #e67e22;")
        else:
            self.dmlStatusCard.contentLabel.setText(
                t("dml_not_available", "✗ Not installed"))
            self.dmlStatusCard.contentLabel.setStyleSheet("color: #e74c3c;")

        # DirectML DLL path
        if os.path.exists(dml_dll):
            self.dmlDllCard.contentLabel.setText(dml_dll)
            self.dmlDllCard.contentLabel.setStyleSheet("color: #2ecc71;")
        else:
            self.dmlDllCard.contentLabel.setText(t("dml_not_found", "Not found"))
            self.dmlDllCard.contentLabel.setStyleSheet("color: #e74c3c;")

        # Embedded onnxruntime-directml site-packages path
        if os.path.isdir(embedded_site):
            self.dmlEmbeddedPathCard.contentLabel.setText(embedded_site)
            self.dmlEmbeddedPathCard.contentLabel.setStyleSheet("color: #2ecc71;")
        else:
            self.dmlEmbeddedPathCard.contentLabel.setText(t("dml_not_found", "Not found"))
            self.dmlEmbeddedPathCard.contentLabel.setStyleSheet("color: #e74c3c;")

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
            "color: #2ecc71;" if sys_python else "color: #e74c3c;")

        # Internal/Embedded Python (<project_root>/python/python.exe)
        internal_python = os.path.join(project_root, "python", "python.exe")
        if os.path.exists(internal_python):
            self.trtInternalPythonCard.contentLabel.setText(internal_python)
            self.trtInternalPythonCard.contentLabel.setStyleSheet("color: #2ecc71;")
        else:
            self.trtInternalPythonCard.contentLabel.setText(f"{internal_python}  (not found)")
            self.trtInternalPythonCard.contentLabel.setStyleSheet("color: #e67e22;")

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
