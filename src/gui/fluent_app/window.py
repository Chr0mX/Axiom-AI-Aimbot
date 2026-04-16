import os
import sys
from ctypes import byref, c_int, WinDLL
from ctypes.wintypes import DWORD
from PyQt6.QtCore import QUrl, QSize, QTimer
from PyQt6.QtGui import QIcon, QDesktopServices, QColor
from PyQt6.QtWidgets import QApplication
from qfluentwidgets import (FluentWindow, NavigationItemPosition, FluentIcon,
                            Theme, setTheme, setThemeColor, qconfig)
from qfluentwidgets.common.style_sheet import setCustomStyleSheet
from qfluentwidgets.components.settings.setting_card import SettingCard

from .pages.visuals_page import VisualsPage

from .pages.aim_page import AimPage
from .pages.trigger_page import TriggerPage
from .pages.configs_page import ConfigsPage
from .pages.keys_page import KeysPage
from .pages.other_page import OtherPage
from .components.language_dialog import LanguageDialog
from .language_manager import getLanguageManager, t
from .theme_colors import ThemeColors, get_color
from .theme_manager import get_theme_manager, apply_theme_to_app
from core.updater import UpdateChecker, open_update_url
from version import __version__



class AxiomWindow(FluentWindow):
    def __init__(self):
        super().__init__()

        # Config references
        self._config = None
        self._configManager = None
        self._isApplyingAcrylic = False
        self._pendingAcrylicApply = False

        # Get language manager
        self.langManager = getLanguageManager()

        # Window Setup
        self.setWindowTitle(f"Axiom v{__version__}")
        self.resize(1100, 750)

        # Track current theme state
        # Respect the global qfluentwidgets theme (which may have been set by the wizard)
        from qfluentwidgets import isDarkTheme as _curIsDark
        _currently_dark = _curIsDark()
        self._isDarkTheme = _currently_dark

        # Disable automatic following of system theme, force fixed theme
        _forced_theme = Theme.DARK if _currently_dark else Theme.LIGHT
        qconfig.set(qconfig.themeMode, _forced_theme, save=False)
        setTheme(_forced_theme)

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
        from PyQt6.QtGui import QPixmap, QFont
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

        # Setup Pages - using new tab key names

        self.displayInterface = VisualsPage(self)  # tab_display
        self.aimInterface = AimPage(self)           # tab_aim_control
        self.triggerInterface = TriggerPage(self)   # tab_auto_features
        self.keysInterface = KeysPage(self)         # tab_keys
        self.configInterface = ConfigsPage(self)    # tab_config_management
        self.otherInterface = OtherPage(self)       # tab_program_control
        

        self.displayInterface.setObjectName("displayInterface")
        self.aimInterface.setObjectName("aimInterface")
        self.triggerInterface.setObjectName("triggerInterface")
        self.keysInterface.setObjectName("keysInterface")
        self.configInterface.setObjectName("configInterface")
        self.otherInterface.setObjectName("otherInterface")

        # 調整導航欄寬度
        self.navigationInterface.setMinimumWidth(40)
        self.navigationInterface.setExpandWidth(150)

        self.initNavigation()
        self.initBottomNavigation()
        
        # 若為深色主題，立即切換圖標為白色版本
        if self._isDarkTheme:
            self.updateIcons()

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
            self.displayInterface, self.aimInterface,
            self.triggerInterface, self.keysInterface, 
            self.configInterface, self.otherInterface
        ]
        for page in pages:
            if hasattr(page, 'setConfig'):
                page.setConfig(config)
        self.updateVisualsVisibilityForScreenshotMethod(getattr(config, 'screenshot_method', 'mss'))
        
        # 從配置還原主題設定（無條件強制套用，避免狀態不一致）
        saved_dark = getattr(config, 'dark_mode', False)
        if saved_dark:
            setTheme(Theme.DARK)
            qconfig.set(qconfig.themeMode, Theme.DARK, save=False)
            self._isDarkTheme = True
            self.themeButton.setIcon(FluentIcon.BRIGHTNESS)
        else:
            setTheme(Theme.LIGHT)
            qconfig.set(qconfig.themeMode, Theme.LIGHT, save=False)
            self._isDarkTheme = False
            self.themeButton.setIcon(FluentIcon.QUIET_HOURS)
        self._forceWindowsTitleBarColor(isDark=self._isDarkTheme)
        self.updateLogo()
        self.updateIcons()

        # 更新頁面內的主題相關圖標（如 Discord 按鈕）
        if hasattr(self, 'otherInterface') and hasattr(self.otherInterface, '_updateDiscordIcon'):
            self.otherInterface._updateDiscordIcon()

        # 重新套用 Acrylic 效果和主題樣式（使用 config 中的 alpha 值）
        self._applyAcrylicEffect()
        self._applyThemeStyles()

        # 確保參數管理頁的左右面板在配置更新後立即重繪磨砂樣式
        if hasattr(self, 'configInterface') and hasattr(self.configInterface, '_applyPanelStyles'):
            self.configInterface._applyPanelStyles()

    def updateVisualsVisibilityForScreenshotMethod(self, screenshot_method: str):
        """UVC 模式下隱藏 visual 頁籤，避免與 UVC 預覽重複顯示。"""
        is_uvc = str(screenshot_method).lower() == 'uvc'
        should_show_visuals = not is_uvc
        if hasattr(self, 'nav_display') and self.nav_display is not None:
            self.nav_display.setVisible(should_show_visuals)

        if is_uvc and self.stackedWidget.currentWidget() is self.displayInterface:
            self.switchTo(self.aimInterface)
    
    def setConfigManager(self, manager):
        """設定 ConfigManager 實例"""
        self._configManager = manager
        if hasattr(self.configInterface, 'setConfigManager'):
            self.configInterface.setConfigManager(manager)
    
    def _refreshAllPages(self):
        """刷新所有頁面的設定值"""
        if self._config:
            self.setConfig(self._config)
        
    def initNavigation(self):
        # Navigation items using translation keys

        
        self.nav_aim = self.addSubInterface(self.aimInterface, QIcon(os.path.join(self.base_path, "assets", "aim.svg")), t("tab_aim_control")) 
        
        self.nav_trigger = self.addSubInterface(self.triggerInterface, QIcon(os.path.join(self.base_path, "assets", "trigger.svg")), t("tab_auto_features"))
        
        self.nav_keys = self.addSubInterface(self.keysInterface, QIcon(os.path.join(self.base_path, "assets", "mouse.svg")), t("tab_keys"))
        
        self.nav_display = self.addSubInterface(self.displayInterface, QIcon(os.path.join(self.base_path, "assets", "eye.svg")), t("tab_display"))

        self.nav_config = self.addSubInterface(self.configInterface, QIcon(os.path.join(self.base_path, "assets", "save.svg")), t("tab_config_management"))
        
        self.nav_other = self.addSubInterface(self.otherInterface, FluentIcon.APPLICATION, t("tab_program_control"))


    def initBottomNavigation(self):
        # Language
        self.languageButton = self.navigationInterface.addItem(
            routeKey="language",
            icon=FluentIcon.GLOBE,
            text=t("language"),
            onClick=self.showLanguageDialog,
            position=NavigationItemPosition.BOTTOM
        )

        # Theme Toggle
        self.themeButton = self.navigationInterface.addItem(
            routeKey="theme_switch",
            icon=FluentIcon.QUIET_HOURS,
            text=t("theme_toggle"),
            onClick=self.toggleTheme,
            position=NavigationItemPosition.BOTTOM
        )
        
        # Discord
        self.discordButton = self.navigationInterface.addItem(
            routeKey="discord",
            icon=QIcon(os.path.join(self.base_path, "assets", "discord.svg")),
            text=t("discord"),
            onClick=lambda: QDesktopServices.openUrl(QUrl("https://discord.gg/h4dEh3b8Bt")),
            position=NavigationItemPosition.BOTTOM
        )
        
        # Github
        self.githubButton = self.navigationInterface.addItem(
            routeKey="github",
            icon=FluentIcon.GITHUB,
            text=t("github"),
            onClick=lambda: QDesktopServices.openUrl(QUrl("https://github.com/iisHong0w0/Axiom-AI-Aimbot")),
            position=NavigationItemPosition.BOTTOM
        )
        
        # Donate
        self.donateButton = self.navigationInterface.addItem(
            routeKey="donate",
            icon=FluentIcon.HEART,
            text=t("donate"),
            onClick=lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.abspath(os.path.join(self.base_path, "..", "MVP.html")))),
            position=NavigationItemPosition.BOTTOM
        )

    def updateLogo(self):
        """Update window icon based on theme."""
        if os.path.exists(self.logo_black_path):
            self.setWindowIcon(QIcon(self.logo_black_path))
            # 同步更新 titleBar 的 iconLabel 為放大版
            if hasattr(self, 'titleBar') and hasattr(self.titleBar, 'iconLabel'):
                self.titleBar.iconLabel.setPixmap(QIcon(self.logo_black_path).pixmap(32, 32))

    def updateIcons(self):
        """Update custom icons based on theme."""
        custom_items = [
            (self.nav_display, "eye.svg"),
            (self.nav_aim, "aim.svg"),
            (self.nav_trigger, "trigger.svg"),
            (self.nav_keys, "mouse.svg"),
            (self.nav_config, "save.svg"),
            (self.discordButton, "discord.svg")
        ]

        for item, filename in custom_items:
            if not item: continue
            
            if self._isDarkTheme:
                icon_path = os.path.join(self.base_path, "assets", filename.replace(".svg", "_white.svg"))
            else:
                icon_path = os.path.join(self.base_path, "assets", filename)
            
            if os.path.exists(icon_path):
                item.setIcon(QIcon(icon_path))

    def _applyThemeStyles(self):
        """應用自定義主題顏色樣式表
        
        所有元件都設為透明背景，統一使用視窗底層的 Acrylic 磨砂效果。
        需要處理多層遮擋：
        1. stackedWidget - FluentWindow 的內容容器（由框架 QSS 上色）
        2. NavigationInterface - 導航列
        3. SettingCard - paintEvent 硬編碼了白色背景
        4. TitleBar - 標題列
        """
        from PyQt6.QtGui import QPainter
        from PyQt6.QtWidgets import QFrame
        
        # --- 1. stackedWidget 設為透明（這是最大的遮擋層）---
        self.stackedWidget.setStyleSheet("""
            StackedWidget, QFrame {
                background-color: transparent;
                border: none;
            }
        """)
        # 內部的 view 也要透明
        if hasattr(self.stackedWidget, 'view'):
            self.stackedWidget.view.setStyleSheet("background-color: transparent; border: none;")
        
        # --- 2. 標題列設為透明 並放大標題文字 ---
        if hasattr(self, 'titleBar'):
            text_color = '#FFFFFF' if self._isDarkTheme else '#1A1A1A'
            setCustomStyleSheet(self.titleBar,
                f"""
                FluentTitleBar {{ background-color: transparent; }}
                #titleLabel {{
                    font-family: 'Segoe UI Variable Display', 'Segoe UI', 'Microsoft JhengHei', sans-serif;
                    font-size: 16px;
                    font-weight: 600;
                    color: {text_color};
                }}
                """,
                f"""
                FluentTitleBar {{ background-color: transparent; }}
                #titleLabel {{
                    font-family: 'Segoe UI Variable Display', 'Segoe UI', 'Microsoft JhengHei', sans-serif;
                    font-size: 16px;
                    font-weight: 600;
                    color: {text_color};
                }}
                """)
        
        # --- 3. 導航列設為透明 ---
        nav_qss = """
        NavigationInterface {
            background-color: transparent;
            border: none;
        }
        NavigationPanel {
            background-color: transparent;
        }
        NavigationTreeWidget {
            background-color: transparent;
        }
        """
        setCustomStyleSheet(self.navigationInterface, nav_qss, nav_qss)
        
        # --- 4. 覆寫所有 SettingCard 的 paintEvent 使其透明 ---
        is_dark = self._isDarkTheme
        
        def transparent_paint_event(card_self, e):
            """透明背景的 paintEvent，只畫一條微弱的邊框"""
            painter = QPainter(card_self)
            painter.setRenderHints(QPainter.RenderHint.Antialiasing)
            if is_dark:
                painter.setBrush(QColor(255, 255, 255, 8))
                painter.setPen(QColor(255, 255, 255, 15))
            else:
                painter.setBrush(QColor(255, 255, 255, 20))
                painter.setPen(QColor(0, 0, 0, 12))
            painter.drawRoundedRect(card_self.rect().adjusted(1, 1, -1, -1), 18, 18)
        
        import types
        for card in self.findChildren(SettingCard):
            card.paintEvent = types.MethodType(transparent_paint_event, card)
        
        # --- 5. 所有 ScrollArea 和 viewport 也設為透明 ---
        from PyQt6.QtWidgets import QAbstractScrollArea
        for sa in self.findChildren(QAbstractScrollArea):
            sa.setStyleSheet("background-color: transparent; border: none;")
            if sa.viewport():
                sa.viewport().setStyleSheet("background-color: transparent;")

        # --- 6. 刷新各頁自定義面板樣式（例如參數頁左右面板） ---
        pages = [
            getattr(self, 'displayInterface', None),
            getattr(self, 'aimInterface', None),
            getattr(self, 'triggerInterface', None),
            getattr(self, 'keysInterface', None),
            getattr(self, 'configInterface', None),
            getattr(self, 'otherInterface', None)
        ]
        for page in pages:
            if page is not None and hasattr(page, '_applyPanelStyles'):
                page._applyPanelStyles()

        # --- 7. 按鈕、下拉框、輸入框、分段控件等子控件半透明 ---
        from qfluentwidgets import (PushButton as _PB, PrimaryPushButton as _PPB,
                                     ComboBox as _CB, SegmentedWidget as _SW,
                                     themeColor)
        from PyQt6.QtWidgets import QAbstractSpinBox
        from PyQt6.QtCore import QRectF, Qt as _Qt

        if is_dark:
            glass      = "rgba(255,255,255,12)"
            glass_h    = "rgba(255,255,255,20)"
            glass_p    = "rgba(255,255,255,6)"
            glass_bdr  = "rgba(255,255,255,15)"
        else:
            glass      = "rgba(255,255,255,40)"
            glass_h    = "rgba(255,255,255,60)"
            glass_p    = "rgba(255,255,255,25)"
            glass_bdr  = "rgba(0,0,0,12)"

        # -- 7a. PushButton（一般按鈕 / 快捷鍵按鈕）--
        btn_qss = f"""
        PushButton {{
            background-color: {glass};
            border: 1px solid {glass_bdr};
        }}
        PushButton:hover {{
            background-color: {glass_h};
        }}
        PushButton:pressed {{
            background-color: {glass_p};
        }}
        """
        for w in self.findChildren(_PB):
            if isinstance(w, _PPB):
                continue
            setCustomStyleSheet(w, btn_qss, btn_qss)

        # -- 7b. PrimaryPushButton（主要按鈕保留強調色但半透明）--
        tc = themeColor()
        pa = 155 if not is_dark else 130
        ppb_qss = f"""
        PrimaryPushButton {{
            background-color: rgba({tc.red()},{tc.green()},{tc.blue()},{pa});
        }}
        PrimaryPushButton:hover {{
            background-color: rgba({tc.red()},{tc.green()},{tc.blue()},{min(255, pa+25)});
        }}
        PrimaryPushButton:pressed {{
            background-color: rgba({tc.red()},{tc.green()},{tc.blue()},{max(0, pa-20)});
        }}
        """
        for w in self.findChildren(_PPB):
            setCustomStyleSheet(w, ppb_qss, ppb_qss)

        # -- 7c. ComboBox（下拉選單）--
        combo_qss = f"""
        ComboBox {{
            background-color: {glass};
            border: 1px solid {glass_bdr};
        }}
        ComboBox:hover {{
            background-color: {glass_h};
        }}
        """
        for w in self.findChildren(_CB):
            setCustomStyleSheet(w, combo_qss, combo_qss)

        # -- 7d. SpinBox / DoubleSpinBox（數值輸入框）--
        spin_qss = f"""
        SpinBox {{
            background-color: {glass};
            border: 1px solid {glass_bdr};
        }}
        SpinBox:hover {{
            background-color: {glass_h};
        }}
        DoubleSpinBox {{
            background-color: {glass};
            border: 1px solid {glass_bdr};
        }}
        DoubleSpinBox:hover {{
            background-color: {glass_h};
        }}
        """
        for w in self.findChildren(QAbstractSpinBox):
            setCustomStyleSheet(w, spin_qss, spin_qss)

        # -- 7e. SegmentedWidget（XY 軸切換等分段控件，paintEvent 硬編碼顏色需覆寫）--
        def transparent_seg_paint(seg_self, e):
            from PyQt6.QtWidgets import QWidget as _QW
            _QW.paintEvent(seg_self, e)
            if not seg_self.currentItem():
                return
            painter = QPainter(seg_self)
            painter.setRenderHints(QPainter.RenderHint.Antialiasing)
            if is_dark:
                painter.setPen(QColor(255, 255, 255, 10))
                painter.setBrush(QColor(255, 255, 255, 12))
            else:
                painter.setPen(QColor(0, 0, 0, 10))
                painter.setBrush(QColor(255, 255, 255, 25))
            item = seg_self.currentItem()
            rect = item.rect().adjusted(1, 1, -1, -1).translated(int(seg_self.slideAni.value()), 0)
            painter.drawRoundedRect(rect, 5, 5)
            # 繪製底部指示條
            painter.setPen(_Qt.PenStyle.NoPen)
            from qfluentwidgets.common.color import autoFallbackThemeColor
            painter.setBrush(autoFallbackThemeColor(seg_self.lightIndicatorColor, seg_self.darkIndicatorColor))
            x = int(seg_self.currentItem().width() / 2 - 8 + seg_self.slideAni.value())
            painter.drawRoundedRect(QRectF(x, seg_self.height() - 3.5, 16, 3), 1.5, 1.5)

        for seg in self.findChildren(_SW):
            seg.paintEvent = types.MethodType(transparent_seg_paint, seg)

    def toggleTheme(self):
        if self._isDarkTheme:
            setTheme(Theme.LIGHT)
            qconfig.set(qconfig.themeMode, Theme.LIGHT, save=False)
            self._isDarkTheme = False
            self.themeButton.setIcon(FluentIcon.QUIET_HOURS)
        else:
            setTheme(Theme.DARK)
            qconfig.set(qconfig.themeMode, Theme.DARK, save=False)
            self._isDarkTheme = True
            self.themeButton.setIcon(FluentIcon.BRIGHTNESS)

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
        self._applyThemeStyles()
        
        self.updateLogo()
        self.updateIcons()

        # 更新頁面內的主題相關圖標（如 Discord 按鈕）
        if hasattr(self, 'otherInterface') and hasattr(self.otherInterface, '_updateDiscordIcon'):
            self.otherInterface._updateDiscordIcon()

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
        # Update navigation items text

        if hasattr(self, 'nav_display'): self.nav_display.setText(t("tab_display"))
        if hasattr(self, 'nav_aim'): self.nav_aim.setText(t("tab_aim_control"))
        if hasattr(self, 'nav_trigger'): self.nav_trigger.setText(t("tab_auto_features"))
        if hasattr(self, 'nav_keys'): self.nav_keys.setText(t("tab_keys"))
        if hasattr(self, 'nav_config'): self.nav_config.setText(t("tab_config_management"))
        if hasattr(self, 'nav_other'): self.nav_other.setText(t("tab_program_control"))
        
        # Update bottom navigation
        self.themeButton.setText(t("theme_toggle"))
        self.languageButton.setText(t("language"))
        self.discordButton.setText(t("discord"))
        self.githubButton.setText(t("github"))
        self.donateButton.setText(t("donate"))
        
        # Update all pages
        pages = [
            self.displayInterface, self.aimInterface,
            self.triggerInterface, self.keysInterface, self.configInterface,
            self.otherInterface
        ]
        
        for page in pages:
            if hasattr(page, 'retranslateUi'):
                page.retranslateUi()

    def closeEvent(self, event):
        """視窗關閉時自動保存配置"""
        if self._config is not None:
            try:
                from core.config import save_config
                save_config(self._config)
            except Exception as e:
                print(f"關閉時保存配置失敗: {e}")
        super().closeEvent(event)
