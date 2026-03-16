# configs_page.py
"""參數管理頁面 - 左側列表，右側按鈕"""

import os
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFileDialog,
    QInputDialog, QMessageBox, QSplitter, QFrame
)
from qfluentwidgets import (
    SettingCardGroup, FluentIcon, PrimaryPushButton, PushButton,
    ListWidget, TitleLabel, InfoBar, InfoBarPosition, isDarkTheme,
    qconfig
)

from ..base_page import BasePage
from ..language_manager import t
from ..theme_colors import ThemeColors


class ConfigsPage(BasePage):
    """參數管理頁面 - 左右分割佈局"""
    
    def __init__(self, parent=None):
        super().__init__("tab_config_management", parent)
        self._config = None
        self._configManager = None
        self._initWidgets()
        self._initLayout()
        self._connectSignals()
        
        # 連接主題變更信號
        qconfig.themeChanged.connect(self._applyPanelStyles)
    
    def setConfig(self, config):
        """設定 Config 實例"""
        self._config = config
        self._applyPanelStyles()
    
    def setConfigManager(self, manager):
        """設定 ConfigManager 實例"""
        self._configManager = manager
        self._refreshConfigList()
    
    def _initWidgets(self):
        """初始化所有控制項"""
        
        # === 左側：參數列表 ===
        self.leftPanel = QFrame()
        self.leftPanel.setObjectName("configLeftPanel")
        self.leftLayout = QVBoxLayout(self.leftPanel)
        self.leftLayout.setContentsMargins(16, 16, 16, 16)
        self.leftLayout.setSpacing(12)

        self.listTitle = TitleLabel(t("config_management_features"))
        font = self.listTitle.font()
        font.setPixelSize(18)
        self.listTitle.setFont(font)
        
        self.configList = ListWidget()
        self.configList.setMinimumHeight(400)
        self.configList.setObjectName("configListWidget")
        
        # === 右側：按鈕組 ===
        self.rightPanel = QFrame()
        self.rightPanel.setObjectName("configRightPanel")
        self.rightLayout = QVBoxLayout(self.rightPanel)
        self.rightLayout.setContentsMargins(16, 16, 16, 16)
        self.rightLayout.setSpacing(12)
        
        self.buttonTitle = TitleLabel(t("config_config"))
        font = self.buttonTitle.font()
        font.setPixelSize(18)
        self.buttonTitle.setFont(font)
        
        # 按鈕 - 只有文字，不需要 SettingCard
        self.createBtn = PrimaryPushButton(FluentIcon.ADD, t("create_config"))
        self.loadBtn = PushButton(FluentIcon.DOWNLOAD, t("load_config"))
        self.saveBtn = PushButton(FluentIcon.SAVE, t("save_config"))
        self.deleteBtn = PushButton(FluentIcon.DELETE, t("delete_config"))
        self.renameBtn = PushButton(FluentIcon.EDIT, t("rename_config"))
        self.refreshBtn = PushButton(FluentIcon.SYNC, t("refresh_config"))
        
        # 分隔線
        self.separator = QFrame()
        self.separator.setFrameShape(QFrame.Shape.HLine)
        self.separator.setStyleSheet("background-color: rgba(128, 128, 128, 0.3);")
        self.separator.setFixedHeight(1)
        
        self.importBtn = PushButton(FluentIcon.FOLDER_ADD, t("import_config"))
        self.exportBtn = PushButton(FluentIcon.SHARE, t("export_config"))
        self.openFolderBtn = PushButton(FluentIcon.FOLDER, t("open_config_folder"))
        
        # 設定按鈕最小寬度
        for btn in [self.createBtn, self.loadBtn, self.saveBtn, self.deleteBtn, 
                    self.renameBtn, self.refreshBtn, self.importBtn, self.exportBtn, 
                    self.openFolderBtn]:
            btn.setMinimumWidth(160)
            btn.setMinimumHeight(36)
    
    def _initLayout(self):
        """排版所有控制項 - 左右分割"""
        # 套用面板樣式
        self._applyPanelStyles()

        # 主容器
        self.mainContainer = QWidget()
        self.mainHLayout = QHBoxLayout(self.mainContainer)
        self.mainHLayout.setContentsMargins(0, 0, 0, 0)
        self.mainHLayout.setSpacing(0)
        
        # 左側佈局
        self.leftLayout.addWidget(self.listTitle)
        self.leftLayout.addWidget(self.configList, 1)
        
        # 右側佈局
        self.rightLayout.addWidget(self.buttonTitle)
        self.rightLayout.addWidget(self.createBtn)
        self.rightLayout.addWidget(self.loadBtn)
        self.rightLayout.addWidget(self.saveBtn)
        self.rightLayout.addWidget(self.deleteBtn)
        self.rightLayout.addWidget(self.renameBtn)
        self.rightLayout.addWidget(self.refreshBtn)
        self.rightLayout.addWidget(self.separator)
        self.rightLayout.addWidget(self.importBtn)
        self.rightLayout.addWidget(self.exportBtn)
        self.rightLayout.addWidget(self.openFolderBtn)
        self.rightLayout.addStretch(1)
        
        # 使用 QSplitter 分割左右
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.addWidget(self.leftPanel)
        self.splitter.addWidget(self.rightPanel)
        self.splitter.setSizes([500, 300])  # 左邊稍大
        self.splitter.setStyleSheet("QSplitter::handle { background: transparent; }")

        self.mainHLayout.addWidget(self.splitter)
        self.addContent(self.mainContainer)

        self.scrollLayout.addStretch(1)

    def _applyPanelStyles(self, *_):
        """套用面板邊框樣式 - 根據當前主題動態切換"""
        acrylic_enabled = bool(getattr(self._config, 'enable_acrylic', False))
        element_alpha = int(getattr(self._config, 'acrylic_element_alpha', 25))
        element_alpha = max(0, min(255, element_alpha))
        is_dark = isDarkTheme()

        def rgba(hex_color: str, alpha: int) -> str:
            c = QColor(hex_color)
            a = max(0, min(255, int(alpha)))
            return f"rgba({c.red()}, {c.green()}, {c.blue()}, {a})"

        base_panel_bg = ThemeColors.CARD_BACKGROUND.get()
        base_panel_border = ThemeColors.BORDER_SUBTLE.get()
        base_item_bg = ThemeColors.BACKGROUND_SECONDARY.get()
        base_item_border = ThemeColors.BORDER_SUBTLE.get()
        base_item_hover_bg = ThemeColors.BACKGROUND_HOVER.get()
        base_item_hover_border = ThemeColors.BORDER_DEFAULT.get()
        base_item_selected_bg = ThemeColors.BACKGROUND_PRESSED.get()
        base_item_selected_border = ThemeColors.BORDER_STRONG.get()

        if acrylic_enabled:
            # 磨砂模式：改用低透明度玻璃層，避免大面積白底覆蓋背景
            soft_a = max(8, min(36, element_alpha + 4))
            hover_a = max(14, min(56, element_alpha + 16))
            selected_a = max(24, min(78, element_alpha + 30))

            if is_dark:
                panel_bg = rgba("#FFFFFF", soft_a)
                panel_border = rgba("#FFFFFF", 28)
                item_bg = rgba("#FFFFFF", soft_a + 6)
                item_border = rgba("#FFFFFF", 24)
                item_hover_bg = rgba("#FFFFFF", hover_a)
                item_hover_border = rgba("#FFFFFF", 34)
                item_selected_bg = rgba("#4CC2FF", selected_a)
                item_selected_border = rgba("#4CC2FF", 120)
                separator_color = rgba("#FFFFFF", 32)
            else:
                panel_bg = rgba("#FFFFFF", soft_a + 8)
                panel_border = rgba("#000000", 28)
                item_bg = rgba("#FFFFFF", soft_a + 12)
                item_border = rgba("#000000", 22)
                item_hover_bg = rgba("#FFFFFF", hover_a + 8)
                item_hover_border = rgba("#000000", 30)
                item_selected_bg = rgba("#0078D4", selected_a)
                item_selected_border = rgba("#0078D4", 110)
                separator_color = rgba("#000000", 30)
        else:
            panel_bg = base_panel_bg
            panel_border = base_panel_border
            item_bg = base_item_bg
            item_border = base_item_border
            item_hover_bg = base_item_hover_bg
            item_hover_border = base_item_hover_border
            item_selected_bg = base_item_selected_bg
            item_selected_border = base_item_selected_border
            separator_color = "rgba(128, 128, 128, 0.3)"

        text_color = ThemeColors.TEXT_PRIMARY.get()
        
        panelStyle = f"""
            QFrame#configLeftPanel, QFrame#configRightPanel {{
                background-color: {panel_bg};
                border: 1px solid {panel_border};
                border-radius: 18px;
            }}
        """
        
        listStyle = f"""
            QListWidget#configListWidget {{
                background-color: transparent;
                border: none;
                outline: none;
            }}
            QListWidget#configListWidget::item {{
                background-color: {item_bg};
                border: 1px solid {item_border};
                border-radius: 14px;
                padding: 12px 16px;
                margin: 4px 2px;
                color: {text_color};
            }}
            QListWidget#configListWidget::item:hover {{
                background-color: {item_hover_bg};
                border: 1px solid {item_hover_border};
            }}
            QListWidget#configListWidget::item:selected {{
                background-color: {item_selected_bg};
                border: 2px solid {item_selected_border};
                color: {text_color};
            }}
        """
        
        self.leftPanel.setStyleSheet(panelStyle)
        self.rightPanel.setStyleSheet(panelStyle)
        self.configList.setStyleSheet(listStyle)
        self.separator.setStyleSheet(f"background-color: {separator_color};")
    
    def _connectSignals(self):
        """連接信號"""
        self.createBtn.clicked.connect(self._onCreateConfig)
        self.loadBtn.clicked.connect(self._onLoadConfig)
        self.saveBtn.clicked.connect(self._onSaveConfig)
        self.deleteBtn.clicked.connect(self._onDeleteConfig)
        self.renameBtn.clicked.connect(self._onRenameConfig)
        self.refreshBtn.clicked.connect(self._refreshConfigList)
        self.importBtn.clicked.connect(self._onImportConfig)
        self.exportBtn.clicked.connect(self._onExportConfig)
        self.openFolderBtn.clicked.connect(self._onOpenFolder)
    
    def _refreshConfigList(self):
        """刷新參數列表"""
        self.configList.clear()
        if self._configManager:
            configs = self._configManager.get_config_list()
            for name in configs:
                self.configList.addItem(name)
    
    def _getSelectedConfig(self) -> str:
        """獲取選中的參數名稱"""
        item = self.configList.currentItem()
        return item.text() if item else ""
    
    def _showInfo(self, title: str, content: str, success: bool = True):
        """顯示訊息提示"""
        if success:
            InfoBar.success(
                title=title, content=content,
                orient=Qt.Orientation.Horizontal,
                isClosable=True, position=InfoBarPosition.TOP,
                duration=2000, parent=self
            )
        else:
            InfoBar.error(
                title=title, content=content,
                orient=Qt.Orientation.Horizontal,
                isClosable=True, position=InfoBarPosition.TOP,
                duration=3000, parent=self
            )
    
    # === 回調函數 ===
    def _onCreateConfig(self):
        name, ok = QInputDialog.getText(self, t("create_config"), t("create_config") + ":")
        if ok and name and self._configManager and self._config:
            try:
                self._configManager.save_config(self._config, name)
                self._refreshConfigList()
                self._showInfo(t("config_success"), t("config_saved"))
            except Exception as e:
                self._showInfo(t("config_error"), str(e), False)
    
    def _onLoadConfig(self):
        name = self._getSelectedConfig()
        if not name:
            self._showInfo(t("config_warning"), t("no_selection"), False)
            return
        if self._configManager and self._config:
            try:
                self._configManager.load_config(self._config, name)
                self._showInfo(t("config_success"), t("config_loaded"))
                window = self.window()
                if hasattr(window, '_refreshAllPages'):
                    window._refreshAllPages()
            except Exception as e:
                self._showInfo(t("config_error"), t("config_load_failed"), False)
    
    def _onSaveConfig(self):
        name = self._getSelectedConfig()
        if not name:
            self._showInfo(t("config_warning"), t("no_selection"), False)
            return
        reply = QMessageBox.question(
            self, t("confirm_overwrite"), f"{t('confirm_overwrite')}: {name}?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            if self._configManager and self._config:
                try:
                    self._configManager.save_config(self._config, name)
                    self._showInfo(t("config_success"), t("config_saved"))
                except Exception as e:
                    self._showInfo(t("config_error"), t("config_save_failed"), False)
    
    def _onDeleteConfig(self):
        name = self._getSelectedConfig()
        if not name:
            self._showInfo(t("config_warning"), t("no_selection"), False)
            return
        reply = QMessageBox.question(
            self, t("confirm_delete"), f"{t('confirm_delete')}: {name}?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            if self._configManager:
                try:
                    self._configManager.delete_config(name)
                    self._refreshConfigList()
                    self._showInfo(t("config_success"), t("config_saved"))
                except Exception as e:
                    self._showInfo(t("config_error"), str(e), False)
    
    def _onRenameConfig(self):
        old_name = self._getSelectedConfig()
        if not old_name:
            self._showInfo(t("config_warning"), t("no_selection"), False)
            return
        new_name, ok = QInputDialog.getText(
            self, t("rename_config"), t("rename_config") + ":", text=old_name
        )
        if ok and new_name and new_name != old_name:
            if self._configManager:
                try:
                    self._configManager.rename_config(old_name, new_name)
                    self._refreshConfigList()
                    self._showInfo(t("config_success"), t("config_saved"))
                except Exception as e:
                    self._showInfo(t("config_error"), str(e), False)
    
    def _onImportConfig(self):
        path, _ = QFileDialog.getOpenFileName(self, t("import_config"), "", "JSON Files (*.json)")
        if path and self._configManager:
            try:
                name = self._configManager.import_config(path)
                if name:
                    self._refreshConfigList()
                    self._showInfo(t("config_success"), t("config_loaded"))
                else:
                    self._showInfo(t("config_error"), t("config_load_failed"), False)
            except Exception as e:
                self._showInfo(t("config_error"), str(e), False)
    
    def _onExportConfig(self):
        name = self._getSelectedConfig()
        if not name:
            self._showInfo(t("config_warning"), t("no_selection"), False)
            return
        path, _ = QFileDialog.getSaveFileName(self, t("export_config"), f"{name}.json", "JSON Files (*.json)")
        if path and self._configManager:
            try:
                self._configManager.export_config(name, path)
                self._showInfo(t("config_success"), t("config_saved"))
            except Exception as e:
                self._showInfo(t("config_error"), str(e), False)
    
    def _onOpenFolder(self):
        if self._configManager:
            folder = self._configManager.configs_dir
            if os.path.exists(folder):
                os.startfile(folder)
    
    def retranslateUi(self):
        """刷新翻譯"""
        super().retranslateUi()
        self.listTitle.setText(t("config_management_features"))
        self.buttonTitle.setText(t("config_config"))
        self.createBtn.setText(t("create_config"))
        self.loadBtn.setText(t("load_config"))
        self.saveBtn.setText(t("save_config"))
        self.deleteBtn.setText(t("delete_config"))
        self.renameBtn.setText(t("rename_config"))
        self.refreshBtn.setText(t("refresh_config"))
        self.importBtn.setText(t("import_config"))
        self.exportBtn.setText(t("export_config"))
        self.openFolderBtn.setText(t("open_config_folder"))
