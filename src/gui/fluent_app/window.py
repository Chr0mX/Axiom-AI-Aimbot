import os
import sys
from ctypes import byref, c_int, WinDLL
from ctypes.wintypes import DWORD
from PyQt6.QtCore import QUrl, QTimer, Qt
from PyQt6.QtGui import (QIcon, QDesktopServices, QColor, QGuiApplication, QPainter, QPixmap,
                         QLinearGradient, QRadialGradient)
from PyQt6.QtWidgets import QGraphicsOpacityEffect, QLabel
from qfluentwidgets import (FluentWindow, FluentIcon, RoundMenu, Action,
                            Theme, setTheme, setThemeColor, qconfig, InfoBar, InfoBarPosition)
from qfluentwidgets.common.style_sheet import setCustomStyleSheet

from .pages.visuals_page import VisualsPage

from .pages.model_page import ModelPage
from .pages.capture_page import CapturePage
from .pages.inference_page import InferencePage
from .pages.aim_page import AimPage
from .pages.trigger_page import TriggerPage
from .pages.configs_page import ConfigsPage
from .pages.keys_page import KeysPage
from .pages.other_page import OtherPage
from .pages.convert_page import ConvertPage
from .components.language_dialog import LanguageDialog
from .language_manager import getLanguageManager, t
from .theme_colors import ThemeColors
from .components.glass import apply_widget_styles, install_tooltip_style, notify_appearance_changed, refresh_motion
from .components.floating_nav import FloatingNav, NavEntry
from .components.status_pill import StatusPill
from .components.caption_buttons import CaptionIconButton, style_caption_buttons
from .components.motion import Tween, fade_in, DUR, DUR_SLOW
from .components.notify import notify
from core import ui_accent, ui_motion, ui_status
try:                                   # CPU % for the status pill; the pill hides the cell without it
    import psutil as _psutil
except Exception:                      # noqa: BLE001 - optional
    _psutil = None
from core.updater import UpdateChecker, open_update_url
from version import __version__



class AxiomWindow(FluentWindow):
    def __init__(self):
        super().__init__()
        install_tooltip_style()          # once, before the pages exist (an app-level QSS change re-polishes every widget)

        # Config references
        self._config = None
        self._configManager = None       # aim-only Preset manager
        self._fullConfigManager = None   # full Config manager
        self._isApplyingAcrylic = False
        self._pendingAcrylicApply = False
        self._transparencyEnsured = False
        self._previewCollapsed = False
        self._navDock = 'top'
        self._backdropKey = None
        self._backdropPixmap = None

        # Get language manager
        self.langManager = getLanguageManager()

        # Window Setup
        self.setWindowTitle(f"Axiom v{__version__}")
        # 1100x750 was cramped enough to clip/wrap SettingCard titles and
        # descriptions on several pages, so the preferred size is 1280x820 —
        # but never larger than ~90 % of the screen's usable area (820 px is
        # taller than a 1366x768 panel), and never smaller than a size at
        # which the slider cards (flexible slider + fixed-width value box)
        # still have room for their controls.
        self.setMinimumSize(900, 600)
        avail = QGuiApplication.primaryScreen().availableGeometry() if QGuiApplication.primaryScreen() else None
        if avail is not None:
            self.resize(max(900, min(1280, int(avail.width() * 0.9))), max(600, min(820, int(avail.height() * 0.9))))
        else:
            self.resize(1280, 820)

        # Track current theme state
        # Respect the global qfluentwidgets theme (which may have been set by the wizard)
        from qfluentwidgets import isDarkTheme as _curIsDark
        _currently_dark = _curIsDark()
        self._isDarkTheme = _currently_dark

        # Disable automatic following of system theme, force fixed theme
        _forced_theme = Theme.DARK if _currently_dark else Theme.LIGHT
        qconfig.set(qconfig.themeMode, _forced_theme, save=False)
        setTheme(_forced_theme)
        # Propagate the app's own accent to all Fluent widgets
        setThemeColor(QColor(ThemeColors.ACCENT.get()))

        # Force set Windows title bar color, unaffected by system theme
        self._forceWindowsTitleBarColor(isDark=_currently_dark)

        # Disable Mica (use Acrylic instead)
        self.setMicaEffectEnabled(False)

        # Set background to completely transparent for Acrylic blur effect visibility
        self.setCustomBackgroundColor(
            QColor(0, 0, 0, 0),  # light theme - transparent
            QColor(0, 0, 0, 0)   # dark theme - transparent
        )

        # Enable Windows Acrylic translucent blur effect
        self._applyAcrylicEffect()

        # Set Win11 window rounded corners
        self._applyWindowRoundedCorners()

        # Apply custom theme color styles
        self._applyThemeStyles()
        
        # Determine logo paths
        self.base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.logo_black_path = os.path.join(self.base_path, "logo.png")
        
        # Fallback if not found
        if not os.path.exists(self.logo_black_path):
            self.logo_black_path = "src/gui/logo.png"
             
        self.updateLogo()
        
        # Enlarge title bar Logo and text
        from PyQt6.QtGui import QFont
        if hasattr(self, 'titleBar'):
            # Enlarge iconLabel (originally 18x18 -> 32x32)
            self.titleBar.iconLabel.setFixedSize(32, 32)
            # Re-set icon pixmap to larger size
            if os.path.exists(self.logo_black_path):
                self.titleBar.iconLabel.setPixmap(QIcon(self.logo_black_path).pixmap(32, 32))
            
            # Enlarge title text
            title_font = QFont('Segoe UI Variable Display', 14)
            title_font.setWeight(QFont.Weight.DemiBold)
            self.titleBar.titleLabel.setFont(title_font)
            self.titleBar.titleLabel.adjustSize()
            
            # Reduce spacing between icon and title
            self.titleBar.hBoxLayout.setSpacing(6)
            self.titleBar.hBoxLayout.setContentsMargins(16, 0, 6, 0)

        # Setup Pages - using new tab key names

        self.displayInterface = VisualsPage(self)  # tab_display
        self.modelInterface = ModelPage(self)       # tab_model
        self.captureInterface = CapturePage(self)   # tab_capture
        self.inferenceInterface = InferencePage(self)  # tab_inference
        self.aimInterface = AimPage(self)           # tab_aim_control
        self.triggerInterface = TriggerPage(self)   # tab_auto_features
        self.keysInterface = KeysPage(self)         # tab_keys
        self.configInterface = ConfigsPage(self)    # tab_config_management
        self.convertInterface = ConvertPage(self)   # tab_convert
        self.otherInterface = OtherPage(self)       # tab_program_control


        self.displayInterface.setObjectName("displayInterface")
        self.modelInterface.setObjectName("modelInterface")
        self.captureInterface.setObjectName("captureInterface")
        self.inferenceInterface.setObjectName("inferenceInterface")
        self.aimInterface.setObjectName("aimInterface")
        self.triggerInterface.setObjectName("triggerInterface")
        self.keysInterface.setObjectName("keysInterface")
        self.configInterface.setObjectName("configInterface")
        self.convertInterface.setObjectName("convertInterface")
        self.otherInterface.setObjectName("otherInterface")

        self.initNavigation()
        self.initChrome()

        # Re-sync a page's widgets from the live Config every time the user
        # navigates to it (zero-lag — doesn't wait for the timer below) and
        # continuously while they're already looking at one (the timer).
        # Config is a single object shared by reference between this GUI
        # and web_control_server.py's routes running on their own thread —
        # a web-triggered change (toggling a setting from the browser, a
        # completed remote model conversion) writes straight into it, but
        # nothing previously told this window's widgets to reflect that.
        # A first attempt only refreshed on navigation, which missed the
        # common case of both UIs open side by side on the *same* page —
        # continuous polling is the only way to catch that, matching the
        # web client's own 1s poll cadence.
        self.stackedWidget.currentChanged.connect(self._onPageActivated)
        self._configSyncTimer = QTimer(self)
        self._configSyncTimer.timeout.connect(self._onConfigSyncTick)
        self._configSyncTimer.start(1000)

        # A remote conversion (Web Control API's POST /api/control/convert)
        # runs as a subprocess app_controller.py itself owns, with no Qt
        # widget of its own tied to it — nothing previously told this
        # window to reflect one in progress at all, so an operator
        # physically at the machine had no visibility into it short of
        # walking through the web page themselves. Riding the same 1s timer
        # as _onConfigSyncTick above (a second, independent slot — see
        # _onRemoteConvertTick's own docstring for why it isn't folded into
        # that method).
        self._remoteConvertSince = 0
        self._remoteConvertActive = False
        self._configSyncTimer.timeout.connect(self._onRemoteConvertTick)

        # ai_loop.py's hot-swap publishes config.model_load_error when a
        # selected model fails to load (it keeps running the old one); show
        # each distinct failure once instead of leaving it in the log only.
        self._lastShownModelLoadError = ""
        self._configSyncTimer.timeout.connect(self._onModelLoadErrorTick)
        self._configSyncTimer.timeout.connect(self._onStatusPillTick)

        from .components.capture_preview import CapturePreviewPanel
        from PyQt6.QtWidgets import QToolButton

        self._previewArrow = QToolButton(self)
        self._previewArrow.setText("◀")
        self._previewArrow.setFixedWidth(18)
        self._previewArrow.setFixedHeight(60)
        self._previewArrow.setToolTip("Collapse preview")
        self._previewArrow.setStyleSheet(
            "QToolButton { border: none; color: #888; font-size: 10px; }"
            "QToolButton:hover { color: #ccc; }"
        )
        self._previewArrow.clicked.connect(self._togglePreviewCollapse)
        self.widgetLayout.addWidget(self._previewArrow)
        self._previewArrow.hide()

        self.previewPanel = CapturePreviewPanel(None, self)
        self.widgetLayout.addWidget(self.previewPanel)
        self.previewPanel.hide()

        # Stretch: content gets 4x, preview gets 1x; swapped when nav collapses
        self.widgetLayout.setStretchFactor(self.stackedWidget, 4)
        self.widgetLayout.setStretchFactor(self.previewPanel, 1)

        # Connect language change signal
        self.langManager.languageChanged.connect(self._refreshUI)

        # Automatically check for updates
        QTimer.singleShot(2000, self.check_for_updates)

    def check_for_updates(self):
        """Starts a background thread to check for updates"""
        self.update_checker = UpdateChecker()
        self.update_checker.update_available.connect(self.on_update_available)
        # self.update_checker.up_to_date.connect(lambda: print("Alredy latest version")) # Debug
        # self.update_checker.check_failed.connect(lambda e: print(f"Update check failed: {e}"))
        self.update_checker.start()

    def on_update_available(self, new_version, url, body):
        """Displays update notification dialog"""
        from qfluentwidgets import MessageDialog
        
        title = self.langManager.get("update_available_title", "New Version Available")
        content = self.langManager.get("update_available_content", "New version {new_version} detected. Would you like to download it?\n\nCurrent version: v{current_version}")
        content = content.replace("{new_version}", new_version).replace("{current_version}", __version__)

        
        # Simple release notes processing, truncate long parts
        if body:
             # Only show first few lines for preview
             preview = "\n".join(body.split('\n')[:5])
             if len(body.split('\n')) > 5:
                 preview += "\n..."
             content += f"\n\n{preview}"

        w = MessageDialog(title, content, self)
        
        # Set button labels
        w.yesButton.setText(self.langManager.get("update_yes", "Download"))
        w.cancelButton.setText(self.langManager.get("update_no", "Later"))
        
        if w.exec():
            open_update_url(url)


    def _forceWindowsTitleBarColor(self, isDark: bool = False):
        """Force set Windows title bar color, unaffected by system theme

        Parameters
        ----------
        isDark: bool
            True = Dark title bar, False = Light title bar
        """
        if sys.platform != 'win32':
            return

        try:
            # DWMWA_USE_IMMERSIVE_DARK_MODE = 20
            DWMWA_USE_IMMERSIVE_DARK_MODE = 20
            dwmapi = WinDLL("dwmapi")
            hwnd = int(self.winId())
            value = c_int(1 if isDark else 0)
            dwmapi.DwmSetWindowAttribute(hwnd, DWORD(DWMWA_USE_IMMERSIVE_DARK_MODE), byref(value), 4)
        except Exception:
            pass  # Ignore errors, not critical for program functionality

    def _applyWindowRoundedCorners(self):
        """Sets Windows 11 window rounded corners (DWMWA_WINDOW_CORNER_PREFERENCE)
        
        Win11 supports DWM corner options:
        - DWMWCP_DEFAULT (0): Default
        - DWMWCP_DONOTROUND (1): No rounding
        - DWMWCP_ROUND (2): Rounded
        - DWMWCP_ROUNDSMALL (3): 小圓角
        """
        if sys.platform != 'win32':
            return
        try:
            DWMWA_WINDOW_CORNER_PREFERENCE = 33
            DWMWCP_ROUND = c_int(2)  # 大圓角
            dwmapi = WinDLL("dwmapi")
            hwnd = int(self.winId())
            dwmapi.DwmSetWindowAttribute(
                hwnd, DWORD(DWMWA_WINDOW_CORNER_PREFERENCE),
                byref(DWMWCP_ROUND), 4
            )
        except Exception:
            pass  # Win10 不支援，忽略

    def _ensureSystemTransparencyEnabled(self):
        """Turn on Windows' "Transparency effects" setting if it's off.

        DWM's ACCENT_ENABLE_ACRYLICBLURBEHIND composites as a flat, unblurred
        fill (not an error) when this system-wide toggle is disabled — same
        registry key Settings > Personalization > Colors > Transparency
        effects writes. Broadcasting WM_SETTINGCHANGE mirrors what the
        Settings app does so Explorer/DWM notice immediately.
        """
        if sys.platform != 'win32':
            return
        try:
            import winreg
            key_path = r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize"
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path, 0, winreg.KEY_READ | winreg.KEY_WRITE) as key:
                try:
                    value, _ = winreg.QueryValueEx(key, "EnableTransparency")
                except FileNotFoundError:
                    value = 0
                if value:
                    return
                winreg.SetValueEx(key, "EnableTransparency", 0, winreg.REG_DWORD, 1)

            user32 = WinDLL("user32")
            HWND_BROADCAST = 0xFFFF
            WM_SETTINGCHANGE = 0x001A
            SMTO_ABORTIFHUNG = 0x0002
            result = DWORD()
            user32.SendMessageTimeoutW(
                HWND_BROADCAST, WM_SETTINGCHANGE, 0, "ImmersiveColorSet",
                SMTO_ABORTIFHUNG, 1000, byref(result)
            )
        except Exception as e:
            print(f"[Acrylic] Failed to enable system Transparency effects: {e}")

    def _applyAcrylicEffect(self):
        """應用 Windows Acrylic 半透明毛玻璃效果"""
        if sys.platform != 'win32':
            return

        if self._isApplyingAcrylic:
            self._pendingAcrylicApply = True
            return

        self._isApplyingAcrylic = True

        try:
            enable = self._config.enable_acrylic if self._config else True

            if not enable:
                # 如果停用，恢復不透明背景
                bg_light = QColor("#F5F5F5")
                bg_dark = QColor("#1A1A1A")
                self.setCustomBackgroundColor(bg_light, bg_dark)
                return

            if not self._transparencyEnsured:
                self._transparencyEnsured = True
                self._ensureSystemTransparencyEnabled()

            # 啟用時務必設為透明
            self.setCustomBackgroundColor(QColor(0, 0, 0, 0), QColor(0, 0, 0, 0))

            # 從設定獲取不透明度 (使用安全範圍 60-255，避免底層 API 不穩定)
            raw_alpha = self._config.acrylic_window_alpha if self._config else 187
            alpha = max(60, min(255, int(raw_alpha)))
            if self._config and self._config.acrylic_window_alpha != alpha:
                self._config.acrylic_window_alpha = alpha

            alpha_hex = hex(alpha)[2:].upper().zfill(2)

            # Acrylic gradientColor 格式為 RRGGBBAA (hex)
            if self._isDarkTheme:
                gradient_color = f"1A1A1A{alpha_hex}"
            else:
                gradient_color = f"F5F5F5{alpha_hex}"

            self.windowEffect.setAcrylicEffect(
                self.winId(),
                gradientColor=gradient_color,
                enableShadow=True,
                animationId=0
            )
        except Exception as e:
            print(f"[Acrylic] 套用 Acrylic 效果失敗: {e}")
        finally:
            self._isApplyingAcrylic = False
            if self._pendingAcrylicApply:
                self._pendingAcrylicApply = False
                QTimer.singleShot(0, self._applyAcrylicEffect)

    def showEvent(self, event):
        """視窗顯示時重新套用標題列顏色和 Acrylic 效果"""
        super().showEvent(event)
        # 確保視窗完全初始化後再次套用標題列顏色
        self._forceWindowsTitleBarColor(isDark=self._isDarkTheme)
        # 延遲重新套用 Acrylic 效果以確保視窗完全初始化
        QTimer.singleShot(100, self._applyAcrylicEffect)
        # 延遲重新套用圓角
        QTimer.singleShot(100, self._applyWindowRoundedCorners)

    def setConfig(self, config):
        """設定 Config 實例並傳遞給所有頁面"""
        self._config = config
        pages = [
            self.displayInterface, self.modelInterface, self.captureInterface,
            self.inferenceInterface, self.aimInterface, self.triggerInterface,
            self.keysInterface, self.configInterface, self.convertInterface,
            self.otherInterface
        ]
        for page in pages:
            if hasattr(page, 'setConfig'):
                page.setConfig(config)
        self.applyAdvancedSettings()
        
        # 從配置還原主題設定（無條件強制套用，避免狀態不一致）
        saved_dark = getattr(config, 'dark_mode', False)
        if saved_dark:
            setTheme(Theme.DARK)
            qconfig.set(qconfig.themeMode, Theme.DARK, save=False)
            self._isDarkTheme = True
        else:
            setTheme(Theme.LIGHT)
            qconfig.set(qconfig.themeMode, Theme.LIGHT, save=False)
            self._isDarkTheme = False
        self._updateThemeButton()
        ui_motion.set_animations_enabled(bool(getattr(config, 'ui_animations', True)))
        refresh_motion(self)
        ui_accent.set_accent(getattr(config, 'accent_color', ui_accent.DEFAULT_ACCENT))
        setThemeColor(QColor(ThemeColors.ACCENT.get()))
        self._forceWindowsTitleBarColor(isDark=self._isDarkTheme)
        self.updateLogo()
        self._navDock = 'left' if getattr(config, 'nav_dock', 'top') == 'left' else 'top'
        self.layoutChrome()

        # 更新頁面內的主題相關圖標（如 Discord 按鈕）
        if hasattr(self, 'otherInterface') and hasattr(self.otherInterface, '_updateDiscordIcon'):
            self.otherInterface._updateDiscordIcon()

        # 重新套用 Acrylic 效果和主題樣式（使用 config 中的 alpha 值）
        self._applyAcrylicEffect()
        self._applyThemeStyles()
        self._backdropKey = None
        self.update()

        # 確保參數管理頁的左右面板在配置更新後立即重繪磨砂樣式
        if hasattr(self, 'configInterface'):
            self.configInterface._applyPanelStyles()

        if hasattr(self, 'previewPanel'):
            self.previewPanel.setConfig(config)
        # (The Visuals page is not hidden for UVC/NDI/UDP: its detection-box settings are also drawn
        # by screen_capture._draw_detection_overlay() into the preview.)
        self.updatePreviewPanelVisibility()

    def updatePreviewPanelVisibility(self) -> None:
        """Show/hide preview panel based on screenshot method and uvc_show_window toggle."""
        if not hasattr(self, 'previewPanel') or self._config is None:
            return
        method = str(getattr(self._config, 'screenshot_method', 'mss')).lower()
        show = bool(getattr(self._config, 'uvc_show_window', True))
        active = method in ('ndi', 'uvc', 'udp') and show
        self._previewArrow.setVisible(active)
        self._previewArrow.setText("▶" if self._previewCollapsed else "◀")
        self._previewArrow.setToolTip("Show preview" if self._previewCollapsed else "Hide preview")
        if active and self._previewCollapsed:
            self.previewPanel.stop()
            self.previewPanel.hide()
        elif active:
            self.previewPanel.show()
            self.previewPanel.start()
        else:
            self.previewPanel.stop()
            self.previewPanel.hide()

    def _togglePreviewCollapse(self) -> None:
        """Fold the capture preview away (or bring it back); the pages gain the freed width."""
        self._previewCollapsed = not self._previewCollapsed
        self.updatePreviewPanelVisibility()

    def setConfigManager(self, manager):
        """Binds the aim-only Preset ConfigManager instance."""
        self._configManager = manager
        if hasattr(self.configInterface, 'setConfigManager'):
            self.configInterface.setConfigManager(manager)

    def setFullConfigManager(self, manager):
        """Binds the full-config-snapshot ConfigManager instance."""
        self._fullConfigManager = manager
        if hasattr(self.configInterface, 'setFullConfigManager'):
            self.configInterface.setFullConfigManager(manager)

    def _refreshAllPages(self, full: bool = True):
        """刷新所有頁面的設定值

        full=True re-binds every page (setConfig), which also redoes each
        page's device/file work (model folder rescan, port scan, ...) — what
        a finished TensorRT build needs. full=False only copies Config into
        the widgets (refreshFromConfig), for e.g. a loaded preset."""
        if not self._config:
            return
        if full:
            self.setConfig(self._config)
            return
        for i in range(self.stackedWidget.count()):
            self._syncPage(self.stackedWidget.widget(i))

    def _syncPage(self, widget) -> None:
        """Copy the live Config into one page's widgets. Pages with
        device/file work in their full load expose a light
        refreshFromConfig(); the rest are cheap to setConfig() again."""
        if widget is None or self._config is None:
            return
        if hasattr(widget, 'refreshFromConfig'):
            widget.refreshFromConfig()
        elif hasattr(widget, 'setConfig'):
            widget.setConfig(self._config)
        if hasattr(widget, 'applyAdvanced'):          # the global "Show advanced settings" switch
            widget.applyAdvanced()

    def applyAdvancedSettings(self) -> None:
        """The switch changed (here or from Web Control): update every page at once."""
        for i in range(self.stackedWidget.count()):
            page = self.stackedWidget.widget(i)
            if hasattr(page, 'applyAdvanced'):
                page.applyAdvanced()

    def _onPageActivated(self, index):
        """Refreshes just the page becoming visible from the live Config —
        see the connect() call in __init__ for why this exists (a web
        control route can change Config on its own thread with nothing
        else telling this window's widgets to reflect it). Deliberately
        scoped to one page, not all ten (_refreshAllPages() territory) —
        cheap, and the user only ever sees the page they just switched to.
        Zero-lag: this fires the instant navigation happens, rather than
        waiting for _onConfigSyncTick()'s next tick.
        """
        widget = self.stackedWidget.widget(index)
        if hasattr(self, 'floatingNav') and widget is not None:
            self.floatingNav.setCurrent(widget.objectName())     # programmatic switchTo() moves the capsule too
            self._fadeCurrentPage()
        if self._config is None:
            return
        self._syncPage(widget)

    def _onConfigSyncTick(self):
        """Continuously re-syncs the *currently visible* page from the live
        Config, once a second — catches the case _onPageActivated() can't:
        the operator already has this exact page open (in Qt) when a web
        control route changes a field it shows, e.g. both UIs open side by
        side and a toggle flipped from the browser. Skips the refresh
        entirely while the keyboard focus is somewhere inside the current
        page, so an in-progress local edit (typing a value, mid-drag on a
        slider) is never clobbered by a tick landing at the wrong moment —
        the same guard the web client's own poll uses for its fields.
        """
        if self._config is None:
            return
        widget = self.stackedWidget.currentWidget()
        if widget is None or not hasattr(widget, 'setConfig'):
            return
        from PyQt6.QtWidgets import QApplication, QAbstractSlider
        focused = QApplication.focusWidget()
        if focused is not None and widget.isAncestorOf(focused):
            # A slider keeps keyboard focus after every completed drag (Qt
            # doesn't clear it just because the mouse released, or because
            # the page was navigated away from and back) — so the common
            # case of "last thing I touched on this page was a slider" would
            # otherwise wedge this guard on indefinitely, silently dropping
            # every tick's refresh (e.g. a preset loaded from the Web Control
            # client) until something else happens to steal focus. Only a
            # slider actively mid-drag right now (isSliderDown()) is a real
            # in-progress edit worth protecting; merely holding stale focus
            # from an earlier interaction is not.
            if not (isinstance(focused, QAbstractSlider) and not focused.isSliderDown()):
                return
        self._syncPage(widget)

    def _onModelLoadErrorTick(self):
        if self._config is None:
            return
        err = str(getattr(self._config, "model_load_error", "") or "")
        if err == self._lastShownModelLoadError:
            return
        self._lastShownModelLoadError = err
        if err:
            InfoBar.error(
                t("model_load_failed", "Model failed to load"), err,
                duration=10000, isClosable=True,
                position=InfoBarPosition.TOP, parent=self,
            )
            try:
                self._refreshAllPages(full=False)  # selection was reverted — resync combos
            except Exception:
                pass

    def _onRemoteConvertTick(self):
        """Mirrors a TensorRT build triggered remotely via the Web Control
        API's POST /api/control/convert (app_controller.start_conversion())
        into this window — the one thing that request does that nothing
        previously reflected on the Qt side at all, since it's a plain
        background subprocess app_controller.py owns directly, with no Qt
        widget of its own tied to it.

        The instant one starts, switches to the Convert tab so an operator
        physically at the machine sees it live rather than only in the
        browser, then feeds each tick's new log lines into
        ConvertPage.showRemoteConversionState(). On completion, calls
        _refreshAllPages() — the same call a LOCAL Convert-tab build already
        triggers via _onConvertFinished() — so the model/backend change
        _run_conversion_worker() already applied and saved lands in every
        page (including the Model page's own cache-status badges), not just
        the Convert tab: "auto redirect the gui to start the conversion
        process then wait for it to finish then detect the model swap", per
        the explicit request this was built for.

        Kept as its own timeout-connected slot rather than folded into
        _onConfigSyncTick() above: that method's early returns (no config,
        no setConfig() on the current widget, focus inside the current
        page) are specifically about not clobbering an in-progress local
        edit on whichever page happens to be visible — none of that logic
        applies here, and forcibly switching tabs the instant a remote
        build starts is the whole point, not something to guard against.
        A plain Qt signal can have any number of slots connected to it, so
        this runs independently off the exact same 1s timer with no
        interference between the two.

        get_conversion_status() is a cheap lock-guarded dict read that's a
        no-op the overwhelming majority of the time (no build ever in
        progress), so this adds negligible cost to the existing tick.
        """
        if self._config is None:
            return
        try:
            from core import app_controller
        except Exception:
            return

        status = app_controller.get_conversion_status(self._remoteConvertSince)
        self._remoteConvertSince = status["next_since"]

        if not status["running"] and not (status["done"] and self._remoteConvertActive):
            return

        if not self._remoteConvertActive:
            self._remoteConvertActive = True
            self.switchTo(self.convertInterface)

        if hasattr(self.convertInterface, "showRemoteConversionState"):
            try:
                self.convertInterface.showRemoteConversionState(status)
            except Exception:
                pass

        if status["done"]:
            self._remoteConvertActive = False
            self._remoteConvertSince = 0
            self._refreshAllPages()

    def initNavigation(self):
        # Navigation items using translation keys

        
        self.nav_model = self.addSubInterface(self.modelInterface, FluentIcon.ROBOT, t("tab_model"))

        self.nav_capture = self.addSubInterface(self.captureInterface, FluentIcon.CAMERA, t("tab_capture", "Capture"))

        self.nav_inference = self.addSubInterface(self.inferenceInterface, FluentIcon.COMMAND_PROMPT, t("tab_inference"))

        self.nav_aim = self.addSubInterface(self.aimInterface, QIcon(os.path.join(self.base_path, "assets", "aim.svg")), t("tab_aim_control"))
        
        self.nav_trigger = self.addSubInterface(self.triggerInterface, QIcon(os.path.join(self.base_path, "assets", "trigger.svg")), t("tab_auto_features", "Auto-Fire"))

        self.nav_keys = self.addSubInterface(self.keysInterface, QIcon(os.path.join(self.base_path, "assets", "mouse.svg")), t("tab_hardware_output", "Keys & HW"))
        
        self.nav_display = self.addSubInterface(self.displayInterface, QIcon(os.path.join(self.base_path, "assets", "eye.svg")), t("tab_display"))

        self.nav_config = self.addSubInterface(self.configInterface, QIcon(os.path.join(self.base_path, "assets", "save.svg")), t("tab_config_management"))

        self.nav_convert = self.addSubInterface(self.convertInterface, FluentIcon.SPEED_HIGH, t("tab_convert"))

        self.nav_other = self.addSubInterface(self.otherInterface, FluentIcon.APPLICATION, t("tab_program_control"))


    # ------------------------------------------------------------------ glass chrome
    # (page order, icon, i18n key, English default). The vendored navigation panel still registers the
    # pages (addSubInterface) so switchTo()/qrouter keep working, but it is hidden; FloatingNav is the UI.
    _NAV = (
        ("modelInterface", "ROBOT", "tab_model", "Model"),
        ("captureInterface", "CAMERA", "tab_capture", "Capture"),
        ("inferenceInterface", "COMMAND_PROMPT", "tab_inference", "Inference"),
        ("aimInterface", "aim.svg", "tab_aim_control", "Aim"),
        ("triggerInterface", "trigger.svg", "tab_auto_features", "Auto-Fire"),
        ("keysInterface", "mouse.svg", "tab_hardware_output", "Keys & HW"),
        ("displayInterface", "eye.svg", "tab_display", "Visuals"),
        ("configInterface", "save.svg", "tab_config_management", "Configs"),
        ("convertInterface", "SPEED_HIGH", "tab_convert", "Convert"),
        ("otherInterface", "APPLICATION", "tab_program_control", "Other"),
    )

    def _navIcon(self, spec):
        if spec.endswith(".svg"):
            return QIcon(os.path.join(self.base_path, "assets", spec))
        return getattr(FluentIcon, spec)

    def _navLabels(self) -> dict:
        return {route: t(key, default) for route, _icon, key, default in self._NAV}

    def initChrome(self):
        """Hide the vendored navigation panel and build the floating pill nav, title-bar buttons,
        status pill and page-fade wiring."""
        self.navigationInterface.hide()
        self.stackedWidget.setAnimationEnabled(False)          # pages fade in via components.motion instead

        labels = self._navLabels()
        self.floatingNav = FloatingNav(self)
        self.floatingNav.setItems([NavEntry(route, self._navIcon(icon), labels[route])
                                   for route, icon, _k, _d in self._NAV])
        self.floatingNav.routeChanged.connect(self._onNavRoute)
        self.floatingNav.dockRequested.connect(self.setNavDock)

        # title bar: theme / language / overflow, then the vendored min / max / close (restyled)
        self.themeBtn = CaptionIconButton(FluentIcon.QUIET_HOURS, t("theme_toggle", "Toggle theme"), self.titleBar)
        self.themeBtn.clicked.connect(self.toggleTheme)
        self.langBtn = CaptionIconButton(FluentIcon.GLOBE, t("language", "Language"), self.titleBar)
        self.langBtn.clicked.connect(self.showLanguageDialog)
        self.moreBtn = CaptionIconButton(FluentIcon.MORE, t("more", "More"), self.titleBar)
        self.moreBtn.clicked.connect(self._showMoreMenu)
        for i, b in enumerate((self.themeBtn, self.langBtn, self.moreBtn)):
            self.titleBar.buttonLayout.insertWidget(i, b)
        style_caption_buttons(self.titleBar)

        self.statusPill = StatusPill(self)
        self.statusPill.slotClicked.connect(self._onPresetSlot)
        self._retranslateChrome()
        # prime psutil's interval counter (the first reading is always 0)
        if _psutil is not None:
            try:
                _psutil.cpu_percent(interval=None)
            except Exception:      # noqa: BLE001 - cosmetic
                pass
        self.layoutChrome()

    def _retranslateChrome(self):
        self.floatingNav.retranslate(self._navLabels())
        self.themeBtn.setToolTip(t("theme_toggle", "Toggle theme"))
        self.langBtn.setToolTip(t("language", "Language"))
        self.moreBtn.setToolTip(t("more", "More"))
        self.statusPill.setCaptions(
            t("status_pill_cpu", "CPU"), t("status_pill_preset", "PRESET"),
            t("status_pill_empty", "Empty slot - assign a preset to it on the Configs page"))
        self.layoutChrome()

    def _showMoreMenu(self):
        menu = RoundMenu(parent=self)
        discord = Action(QIcon(os.path.join(self.base_path, "assets",
                         "discord_white.svg" if self._isDarkTheme else "discord.svg")), t("discord", "Discord"))
        discord.triggered.connect(lambda: QDesktopServices.openUrl(QUrl("https://discord.gg/DpcqaQEj5b")))
        github = Action(FluentIcon.GITHUB, t("github", "GitHub"))
        github.triggered.connect(lambda: QDesktopServices.openUrl(QUrl("https://github.com/Chr0mX/Axiom-AI-Aimbot")))
        menu.addAction(discord)
        menu.addAction(github)
        menu.exec(self.moreBtn.mapToGlobal(self.moreBtn.rect().bottomLeft()))

    def _updateThemeButton(self):
        if hasattr(self, 'themeBtn'):
            self.themeBtn.setIconSource(FluentIcon.QUIET_HOURS if self._isDarkTheme else FluentIcon.BRIGHTNESS)

    # -- layout of the overlay chrome ---------------------------------------------------------
    def resizeEvent(self, e):
        super().resizeEvent(e)
        # the vendored window assumes a 46 px navigation column; there is none any more
        self.titleBar.move(0, 0)
        self.titleBar.resize(self.width(), self.titleBar.height())
        self._backdropKey = None
        self.layoutChrome()

    def layoutChrome(self):
        if not hasattr(self, 'statusPill'):
            return
        w, h = self.width(), self.height()
        nav = self.floatingNav
        left = 'left' == self._navDock
        nav.setOrientation('left' if left else 'top')
        margins_left = 0
        if left:
            nav.resize(nav.sizeHint())
            nav.move(10, 58)
            margins_left = nav.width() + 20
        else:
            left_reserved = 200 if w >= 1000 else 150
            right_reserved = 3 * 34 + 3 * 46 + 12
            avail = max(200, w - left_reserved - right_reserved)
            full, compact = nav.preferredWidth(False), nav.preferredWidth(True)
            use_compact = full > avail
            nav.setCompact(use_compact)
            nw = min(compact if use_compact else full, avail)
            x = max(left_reserved, min((w - nw) // 2, w - right_reserved - nw))
            nav.setGeometry(x, 1, nw, nav.sizeHint().height())
        self.widgetLayout.setContentsMargins(margins_left, 48, 0, 52)
        pill = self.statusPill
        pill.resize(max(pill.sizeHint().width() + 8, 120), pill.height())
        content_w = w - margins_left
        pill.move(int(margins_left + (content_w - pill.width()) / 2), h - pill.height() - 10)
        nav.raise_()
        pill.raise_()

    # -- page switching -----------------------------------------------------------------------
    def _onNavRoute(self, route: str):
        page = getattr(self, route, None)
        if page is not None:
            self.switchTo(page)

    def _fadeCurrentPage(self):
        page = self.stackedWidget.currentWidget()
        if page is not None:
            fade_in(page, DUR, rise=0)

    def setNavDock(self, dock: str):
        """Dock the navigation ``top`` (title row) or ``left`` (vertical rail) and remember it."""
        dock = 'left' if dock == 'left' else 'top'
        changed = dock != self._navDock
        self._navDock = dock
        if self._config is not None:
            self._config.nav_dock = dock
        self.layoutChrome()
        if changed:
            fade_in(self.floatingNav, DUR)
            self._fadeCurrentPage()

    # -- appearance ---------------------------------------------------------------------------
    def setAccentColor(self, hex_color: str, animate: bool = True):
        """Make ``hex_color`` the accent everywhere (switches, sliders, nav capsule, buttons)."""
        snap = self._beginCrossfade() if animate else None
        stored = ui_accent.set_accent(hex_color)
        if self._config is not None:
            self._config.accent_color = stored
        setThemeColor(QColor(stored))
        self._applyThemeStyles(full=False, controls=True)
        notify_appearance_changed()
        self._backdropKey = None
        self.update()
        self._endCrossfade(snap, DUR)

    def setAnimationsEnabled(self, enabled: bool):
        ui_motion.set_animations_enabled(bool(enabled))
        if self._config is not None:
            self._config.ui_animations = bool(enabled)
        refresh_motion(self)

    def _beginCrossfade(self):
        """Snapshot of the window to fade away after a theme/accent change (None when animations are off)."""
        if not ui_motion.animations_enabled() or not self.isVisible():
            return None
        overlay = QLabel(self)
        overlay.setPixmap(self.grab())
        overlay.setGeometry(self.rect())
        overlay.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        overlay.show()
        overlay.raise_()
        return overlay

    def _endCrossfade(self, overlay, ms: int = DUR_SLOW):
        if overlay is None:
            return
        overlay.raise_()
        effect = QGraphicsOpacityEffect(overlay)
        effect.setOpacity(1.0)
        overlay.setGraphicsEffect(effect)
        tween = Tween(overlay, 1.0)
        tween.changed.connect(effect.setOpacity)
        tween.finished.connect(overlay.deleteLater)
        tween.to(0.0, ms)

    # -- backdrop -----------------------------------------------------------------------------
    def _buildBackdrop(self) -> QPixmap:
        w, h = max(1, self.width()), max(1, self.height())
        dark = self._isDarkTheme
        opaque = not (sys.platform == 'win32' and (self._config is None or getattr(self._config, 'enable_acrylic', True)))
        base_alpha = 255 if opaque else 175
        pm = QPixmap(w, h)
        pm.fill(Qt.GlobalColor.transparent)
        p = QPainter(pm)
        top, bottom = ThemeColors.GLASS_BACKDROP_TOP.qcolor(), ThemeColors.GLASS_BACKDROP_BOTTOM.qcolor()
        top.setAlpha(base_alpha)
        bottom.setAlpha(base_alpha)
        g = QLinearGradient(0, 0, 0, h)
        g.setColorAt(0, top)
        g.setColorAt(1, bottom)
        p.fillRect(pm.rect(), g)
        pal = ui_accent.derive(ui_accent.get_accent(), dark)
        glow = QRadialGradient(w * 0.5, -h * 0.08, w * 0.75)
        glow.setColorAt(0, QColor(*pal.base, 58 if dark else 40))
        glow.setColorAt(1, QColor(*pal.base, 0))
        p.fillRect(pm.rect(), glow)
        glow2 = QRadialGradient(w * 0.95, h * 1.02, w * 0.55)
        glow2.setColorAt(0, QColor(*pal.hi, 34 if dark else 26))
        glow2.setColorAt(1, QColor(*pal.hi, 0))
        p.fillRect(pm.rect(), glow2)
        p.end()
        return pm

    def paintEvent(self, e):
        super().paintEvent(e)
        key = (self.width(), self.height(), self._isDarkTheme, ui_accent.get_accent())
        if key != self._backdropKey or self._backdropPixmap is None:
            self._backdropKey = key
            self._backdropPixmap = self._buildBackdrop()
        p = QPainter(self)
        p.drawPixmap(0, 0, self._backdropPixmap)

    # -- status pill --------------------------------------------------------------------------
    def _onStatusPillTick(self):
        if self._config is None or not hasattr(self, 'statusPill'):
            return
        cpu = None
        if _psutil is not None:
            try:
                cpu = _psutil.cpu_percent(interval=None)
            except Exception:      # noqa: BLE001 - cosmetic
                cpu = None
        cfg = self._config
        self.statusPill.update_state(
            cpu, getattr(cfg, 'preset_slots', []), getattr(cfg, 'active_preset_name', ''),
            getattr(cfg, 'active_config_name', ''), t("status_panel_none", "None"))
        self.layoutChrome()

    def _onPresetSlot(self, index: int):
        cfg = self._config
        if cfg is None:
            return
        name = ui_status.normalize_slots(getattr(cfg, 'preset_slots', []))[index]
        if not name:
            notify(self, "info", t("status_pill_slot_title", "Quick preset {n}").format(n=index + 1),
                   t("status_pill_empty", "Empty slot - assign a preset to it on the Configs page"))
            return
        mgr = self._configManager
        if mgr is None or not mgr.load_config(cfg, name):
            notify(self, "error", t("status_pill_slot_title", "Quick preset {n}").format(n=index + 1),
                   t("status_pill_load_failed", "Could not load preset '{name}'").format(name=name))
            return
        self._refreshAllPages(full=False)
        self._onStatusPillTick()
        notify(self, "success", t("status_pill_slot_title", "Quick preset {n}").format(n=index + 1),
               t("status_pill_loaded", "Loaded '{name}'").format(name=name))

    def updateLogo(self):
        """Update window icon based on theme."""
        if os.path.exists(self.logo_black_path):
            self.setWindowIcon(QIcon(self.logo_black_path))
            # 同步更新 titleBar 的 iconLabel 為放大版
            if hasattr(self, 'titleBar') and hasattr(self.titleBar, 'iconLabel'):
                self.titleBar.iconLabel.setPixmap(QIcon(self.logo_black_path).pixmap(32, 32))

    def _applyThemeStyles(self, full: bool = True, controls: bool = True):
        """Make every container transparent (the window paints the backdrop) and give the stock controls
        the glass look.

        ``full`` redoes the theme-independent work (transparent containers / scroll areas); ``controls``
        redoes the per-widget control stylesheets (needed when the *accent* changes — they carry both the
        light and dark variants, so a plain theme toggle does not need it). A theme toggle only needs the
        title bar text colour: the glass painters read the live theme at paint time.
        """
        from PyQt6.QtGui import QPainter
        from PyQt6.QtWidgets import QAbstractScrollArea
        from qfluentwidgets import SegmentedWidget as _SW

        if full:
            # stackedWidget and its view are the biggest occluding layer
            self.stackedWidget.setStyleSheet("StackedWidget, QFrame { background-color: transparent; border: none; }")
            if hasattr(self.stackedWidget, 'view'):
                self.stackedWidget.view.setStyleSheet("background-color: transparent; border: none;")
            nav_qss = ("NavigationInterface { background-color: transparent; border: none; }"
                       "NavigationPanel { background-color: transparent; }"
                       "NavigationTreeWidget { background-color: transparent; }")
            setCustomStyleSheet(self.navigationInterface, nav_qss, nav_qss)
            for sa in self.findChildren(QAbstractScrollArea):
                sa.setStyleSheet("background-color: transparent; border: none;")
                if sa.viewport():
                    sa.viewport().setStyleSheet("background-color: transparent;")

        if hasattr(self, 'titleBar'):       # transparent title bar, larger title text
            text_color = '#FFFFFF' if self._isDarkTheme else '#1A1A1A'
            qss = (f"FluentTitleBar {{ background-color: transparent; }}"
                   f"#titleLabel {{ font-family: 'Segoe UI Variable Display', 'Segoe UI', 'Microsoft JhengHei', sans-serif;"
                   f" font-size: 16px; font-weight: 600; color: {text_color}; }}")
            setCustomStyleSheet(self.titleBar, qss, qss)

        if controls:
            if hasattr(self, 'configInterface'):
                self.configInterface._applyPanelStyles()
            apply_widget_styles(self)

        # Segmented controls (PID X/Y axis...): the vendored paintEvent hard-codes colours; ours reads the
        # live theme and accent, so it is patched once per widget.
        import types
        from PyQt6.QtCore import QRectF
        from PyQt6.QtGui import QPen

        def glass_seg_paint(seg_self, e):
            from PyQt6.QtWidgets import QWidget as _QW
            _QW.paintEvent(seg_self, e)
            item = seg_self.currentItem()
            if not item:
                return
            painter = QPainter(seg_self)
            painter.setRenderHints(QPainter.RenderHint.Antialiasing)
            pal = ui_accent.derive(ui_accent.get_accent(), self._isDarkTheme)
            rect = QRectF(item.rect().adjusted(1, 2, -1, -2).translated(int(seg_self.slideAni.value()), 0))
            painter.setPen(QPen(QColor(*pal.base, 150), 1))
            painter.setBrush(QColor(*pal.soft))
            painter.drawRoundedRect(rect, rect.height() / 2, rect.height() / 2)

        for seg in self.findChildren(_SW):
            if not getattr(seg, '_glassSegPatched', False):
                seg.paintEvent = types.MethodType(glass_seg_paint, seg)
                seg._glassSegPatched = True

    def toggleTheme(self):
        snap = self._beginCrossfade()
        if self._isDarkTheme:
            setTheme(Theme.LIGHT)
            qconfig.set(qconfig.themeMode, Theme.LIGHT, save=False)
            self._isDarkTheme = False
        else:
            setTheme(Theme.DARK)
            qconfig.set(qconfig.themeMode, Theme.DARK, save=False)
            self._isDarkTheme = True
        self._updateThemeButton()      # the accent is unchanged: setThemeColor() would re-style every widget again

        # 保存主題設定到配置
        if self._config is not None:
            self._config.dark_mode = self._isDarkTheme

        # 強制更新 Windows 標題列顏色
        self._forceWindowsTitleBarColor(isDark=self._isDarkTheme)

        # 重新套用 Acrylic 效果（更新顏色）
        self._applyAcrylicEffect()

        # 重新套用圓角
        self._applyWindowRoundedCorners()

        # 應用新的主題樣式
        self._applyThemeStyles(full=False, controls=False)

        self.updateLogo()
        notify_appearance_changed()
        self._backdropKey = None
        self.update()

        # 更新頁面內的主題相關圖標（如 Discord 按鈕）
        if hasattr(self, 'otherInterface') and hasattr(self.otherInterface, '_updateDiscordIcon'):
            self.otherInterface._updateDiscordIcon()
        self._endCrossfade(snap)

    def showLanguageDialog(self):
        """Show language selection dialog."""
        dialog = LanguageDialog(currentLanguage=self.langManager.currentLanguage, parent=self)
        dialog.languageChanged.connect(self._onLanguageChanged)
        dialog.exec()
    
    def _onLanguageChanged(self, languageCode: str):
        """Handle language change."""
        self.langManager.setLanguage(languageCode)
        print(f"Language changed to: {languageCode}")
    
    def _refreshUI(self):
        """Refresh UI text after language change."""
        self._retranslateChrome()

        # Update all pages
        pages = [
            self.displayInterface, self.modelInterface, self.captureInterface,
            self.inferenceInterface, self.aimInterface, self.triggerInterface,
            self.keysInterface, self.configInterface, self.convertInterface,
            self.otherInterface
        ]

        for page in pages:
            if hasattr(page, 'retranslateUi'):
                page.retranslateUi()

        if hasattr(self, 'previewPanel'):
            self.previewPanel.retranslateUi()

    def closeEvent(self, event):
        """視窗關閉時自動保存配置"""
        if self._config is not None:
            try:
                from core.config import save_config
                save_config(self._config)
            except Exception as e:
                print(f"關閉時保存配置失敗: {e}")
        super().closeEvent(event)
