"""The classic (pre-redesign) window chrome, kept so the interface can be switched back to it.

``AxiomWindow`` mixes this in and calls these methods when ``core.ui_style`` is ``classic``: the
vendored left ``NavigationInterface`` with its bottom action items (language / theme / Discord /
GitHub), the navigation-collapse arrow that also widens the capture preview, and the original
transparent-container theme path (``SettingCard.paintEvent`` is patched to a faint border). The code
is the window as it was before the glass redesign, only renamed with a ``_classic`` prefix where it
would clash with a glass method. Nothing here runs in the glass look.
"""
import os

from PyQt6.QtCore import QUrl
from PyQt6.QtGui import QColor, QDesktopServices, QIcon
from qfluentwidgets import FluentIcon, NavigationItemPosition, Theme, qconfig, setTheme, setThemeColor
from qfluentwidgets.common.style_sheet import setCustomStyleSheet
from qfluentwidgets.components.settings.setting_card import SettingCard

from .language_manager import t
from .theme_colors import ThemeColors


class ClassicWindowMixin:
    def initClassicChrome(self):
        """Left-navigation width, bottom action items and nav-mode wiring (the old ``__init__`` block)."""
        self.navigationInterface.setMinimumWidth(40)
        self.navigationInterface.setExpandWidth(150)
        self.initBottomNavigation()
        self.navigationInterface.displayModeChanged.connect(self._onNavModeChanged)
        if self._isDarkTheme:
            self.updateIcons()

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
            onClick=lambda: QDesktopServices.openUrl(QUrl("https://discord.gg/DpcqaQEj5b")),
            position=NavigationItemPosition.BOTTOM
        )

        # Github
        self.githubButton = self.navigationInterface.addItem(
            routeKey="github",
            icon=FluentIcon.GITHUB,
            text=t("github"),
            onClick=lambda: QDesktopServices.openUrl(QUrl("https://github.com/Chr0mX/Axiom-AI-Aimbot")),
            position=NavigationItemPosition.BOTTOM
        )

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

    def _onNavModeChanged(self, mode) -> None:
        from qfluentwidgets import NavigationDisplayMode
        collapsed = mode in (NavigationDisplayMode.MINIMAL, NavigationDisplayMode.COMPACT)
        if collapsed:
            self.widgetLayout.setStretchFactor(self.stackedWidget, 2)
            self.widgetLayout.setStretchFactor(self.previewPanel, 1)
            self._previewArrow.setText("▶")
            self._previewArrow.setToolTip("Expand navigation")
        else:
            self.widgetLayout.setStretchFactor(self.stackedWidget, 4)
            self.widgetLayout.setStretchFactor(self.previewPanel, 1)
            self._previewArrow.setText("◀")
            self._previewArrow.setToolTip("Collapse navigation")

    def _classicTogglePreviewCollapse(self) -> None:
        """Toggle left navigation collapsed/expanded; preview gains the freed space."""
        self.navigationInterface.toggle()

    def _classicUpdatePreviewPanelVisibility(self) -> None:
        """Show/hide preview panel based on screenshot method and uvc_show_window toggle."""
        if not hasattr(self, 'previewPanel') or self._config is None:
            return
        method = str(getattr(self._config, 'screenshot_method', 'mss')).lower()
        show = bool(getattr(self._config, 'uvc_show_window', True))
        active = method in ('ndi', 'uvc', 'udp') and show
        self._previewArrow.setVisible(active)
        if active:
            self.previewPanel.show()
            self.previewPanel.start()
        else:
            self.previewPanel.stop()
            self.previewPanel.hide()

    def _classicApplyThemeStyles(self):
        """應用自定義主題顏色樣式表
        
        所有元件都設為透明背景，統一使用視窗底層的 Acrylic 磨砂效果。
        需要處理多層遮擋：
        1. stackedWidget - FluentWindow 的內容容器（由框架 QSS 上色）
        2. NavigationInterface - 導航列
        3. SettingCard - paintEvent 硬編碼了白色背景
        4. TitleBar - 標題列
        """
        from PyQt6.QtGui import QPainter
        
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

        # --- 6. 刷新參數頁（Configs）左右面板樣式 ---
        if hasattr(self, 'configInterface'):
            self.configInterface._applyPanelStyles()

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

    def _classicToggleTheme(self):
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
        setThemeColor(QColor(ThemeColors.ACCENT.get()))

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

    def _classicRetranslateNav(self):
        """Navigation and bottom-item texts after a language change."""
        # Update navigation items text

        if hasattr(self, 'nav_display'): self.nav_display.setText(t("tab_display"))
        if hasattr(self, 'nav_model'): self.nav_model.setText(t("tab_model"))
        if hasattr(self, 'nav_capture'): self.nav_capture.setText(t("tab_capture", "Capture"))
        if hasattr(self, 'nav_inference'): self.nav_inference.setText(t("tab_inference"))
        if hasattr(self, 'nav_aim'): self.nav_aim.setText(t("tab_aim_control"))
        if hasattr(self, 'nav_trigger'): self.nav_trigger.setText(t("tab_auto_features", "Auto-Fire"))
        if hasattr(self, 'nav_keys'): self.nav_keys.setText(t("tab_hardware_output", "Keys & HW"))
        if hasattr(self, 'nav_config'): self.nav_config.setText(t("tab_config_management"))
        if hasattr(self, 'nav_convert'): self.nav_convert.setText(t("tab_convert"))
        if hasattr(self, 'nav_other'): self.nav_other.setText(t("tab_program_control"))
        
        # Update bottom navigation
        self.themeButton.setText(t("theme_toggle"))
        self.languageButton.setText(t("language"))
        self.discordButton.setText(t("discord"))
        self.githubButton.setText(t("github"))
