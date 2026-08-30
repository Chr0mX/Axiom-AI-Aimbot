# configs_page.py
"""Configs page — two boxes: Config (full settings snapshot) and Preset (aim-only settings)."""

import os
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFileDialog,
    QInputDialog, QMessageBox, QSplitter, QFrame
)
from qfluentwidgets import (
    FluentIcon, PrimaryPushButton, PushButton,
    ListWidget, TitleLabel, InfoBar, InfoBarPosition, isDarkTheme,
    qconfig
)

from ..base_page import BasePage
from ..language_manager import t
from ..theme_colors import ThemeColors


# Each box (Config / Preset) is identical in behavior — same CRUD, same
# dialogs, same styling — differing only in (a) which ConfigManager
# instance it operates against (aim_only=False for Config, True for
# Preset) and (b) which i18n keys label it. These two dicts are that
# entire difference; _ManagerBox itself has no "which one am I" branches
# anywhere. A tuple value is (key, default) for a key that may not exist
# in every language file yet — see language_manager.t()'s own fallback.
_CONFIG_I18N = dict(
    list_title="full_config_management_features",
    box_title="full_config_actions",
    create="create_full_config",
    load="load_full_config",
    save="save_full_config",
    delete="delete_full_config",
    rename="rename_full_config",
    refresh="refresh_full_config",
    import_="import_full_config",
    export="export_full_config",
    open_folder="open_full_config_folder",
    saved="full_config_saved",
    save_failed="full_config_save_failed",
    loaded="full_config_loaded",
    load_failed="full_config_load_failed",
    deleted="full_config_deleted",
    # These two mirror config_delete_failed/config_rename_failed's own
    # precedent — never added to English_English.json, inline-default only.
    delete_failed=("full_config_delete_failed", "Failed to delete config"),
    rename_failed=("full_config_rename_failed", "Failed to rename config"),
    diff_will_change="full_config_diff_will_change",
)

_PRESET_I18N = dict(
    list_title="config_management_features",
    box_title="config_config",
    create="create_config",
    load="load_config",
    save="save_config",
    delete="delete_config",
    rename="rename_config",
    refresh="refresh_config",
    import_="import_config",
    export="export_config",
    open_folder="open_config_folder",
    saved="config_saved",
    save_failed="config_save_failed",
    loaded="config_loaded",
    load_failed="config_load_failed",
    deleted=("config_deleted", "Preset deleted"),
    delete_failed=("config_delete_failed", "Failed to delete preset"),
    rename_failed=("config_rename_failed", "Failed to rename preset"),
    diff_will_change="preset_diff_will_change",
)


def _t(key_or_tuple) -> str:
    """t() wrapper accepting either a plain i18n key or a (key, default)
    tuple — the latter for keys that may not exist in every language's
    JSON yet (see language_manager.t()'s own English-fallback behavior)."""
    if isinstance(key_or_tuple, tuple):
        return t(*key_or_tuple)
    return t(key_or_tuple)


class _ManagerBox(QFrame):
    """One list+action-buttons box bound to a single ConfigManager
    instance — either the full Config manager or the aim-only Preset
    manager (see the two _*_I18N dicts above for which). ConfigsPage
    instantiates this twice; nothing in here ever needs to know which one
    it is beyond the i18n dict and the ConfigManager it was given.
    """

    def __init__(self, i18n: dict, parent=None):
        super().__init__(parent)
        self._i18n = i18n
        self._config = None
        self._configManager = None

        self._initWidgets()
        self._initLayout()
        self._connectSignals()

    def setConfig(self, config):
        self._config = config

    def setConfigManager(self, manager):
        self._configManager = manager
        self._refreshConfigList()

    def _initWidgets(self):
        self.leftPanel = QFrame()
        self.leftLayout = QVBoxLayout(self.leftPanel)
        self.leftLayout.setContentsMargins(16, 16, 16, 16)
        self.leftLayout.setSpacing(12)

        self.listTitle = TitleLabel(_t(self._i18n["list_title"]))
        font = self.listTitle.font()
        font.setPixelSize(18)
        self.listTitle.setFont(font)

        self.configList = ListWidget()
        self.configList.setMinimumHeight(280)

        self.rightPanel = QFrame()
        self.rightLayout = QVBoxLayout(self.rightPanel)
        self.rightLayout.setContentsMargins(16, 16, 16, 16)
        self.rightLayout.setSpacing(12)

        self.buttonTitle = TitleLabel(_t(self._i18n["box_title"]))
        font = self.buttonTitle.font()
        font.setPixelSize(18)
        self.buttonTitle.setFont(font)

        self.createBtn = PrimaryPushButton(FluentIcon.ADD, _t(self._i18n["create"]))
        self.loadBtn = PushButton(FluentIcon.DOWNLOAD, _t(self._i18n["load"]))
        self.saveBtn = PushButton(FluentIcon.SAVE, _t(self._i18n["save"]))
        self.deleteBtn = PushButton(FluentIcon.DELETE, _t(self._i18n["delete"]))
        self.renameBtn = PushButton(FluentIcon.EDIT, _t(self._i18n["rename"]))
        self.refreshBtn = PushButton(FluentIcon.SYNC, _t(self._i18n["refresh"]))

        # Styled in applyPanelStyles(), called by the owning page right
        # after this widget is constructed — no initial stylesheet needed here.
        self.separator = QFrame()
        self.separator.setFrameShape(QFrame.Shape.HLine)
        self.separator.setFixedHeight(1)

        self.importBtn = PushButton(FluentIcon.FOLDER_ADD, _t(self._i18n["import_"]))
        self.exportBtn = PushButton(FluentIcon.SHARE, _t(self._i18n["export"]))
        self.openFolderBtn = PushButton(FluentIcon.FOLDER, _t(self._i18n["open_folder"]))

        for btn in [self.createBtn, self.loadBtn, self.saveBtn, self.deleteBtn,
                    self.renameBtn, self.refreshBtn, self.importBtn, self.exportBtn,
                    self.openFolderBtn]:
            btn.setMinimumWidth(160)
            btn.setMinimumHeight(36)

    def _initLayout(self):
        self.leftLayout.addWidget(self.listTitle)
        self.leftLayout.addWidget(self.configList, 1)

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

        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.addWidget(self.leftPanel)
        self.splitter.addWidget(self.rightPanel)
        self.splitter.setSizes([500, 300])
        self.splitter.setStyleSheet("QSplitter::handle { background: transparent; }")

        boxLayout = QVBoxLayout(self)
        boxLayout.setContentsMargins(0, 0, 0, 0)
        boxLayout.addWidget(self.splitter)

    def applyPanelStyles(self, config):
        """Mirrors the acrylic/theme-aware styling the page-level version of
        this used to compute once for a single pair of panels — now called
        once per box, both times with the same `config`/theme state, so the
        two boxes always look identical regardless of which manager each
        one is bound to."""
        acrylic_enabled = bool(getattr(config, 'enable_acrylic', False))
        element_alpha = int(getattr(config, 'acrylic_element_alpha', 25))
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
            separator_color = ThemeColors.BORDER_SUBTLE.get()

        text_color = ThemeColors.TEXT_PRIMARY.get()

        panelStyle = f"""
            QFrame {{
                background-color: {panel_bg};
                border: 1px solid {panel_border};
                border-radius: 18px;
            }}
        """
        listStyle = f"""
            QListWidget {{
                background-color: transparent;
                border: none;
                outline: none;
            }}
            QListWidget::item {{
                background-color: {item_bg};
                border: 1px solid {item_border};
                border-radius: 14px;
                padding: 12px 16px;
                margin: 4px 2px;
                color: {text_color};
            }}
            QListWidget::item:hover {{
                background-color: {item_hover_bg};
                border: 1px solid {item_hover_border};
            }}
            QListWidget::item:selected {{
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
        self.configList.clear()
        if self._configManager:
            for name in self._configManager.get_config_list():
                self.configList.addItem(name)

    def _getSelectedConfig(self) -> str:
        item = self.configList.currentItem()
        return item.text() if item else ""

    def _showInfo(self, title: str, content: str, success: bool = True):
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

    def _onCreateConfig(self):
        create_label = _t(self._i18n["create"])
        name, ok = QInputDialog.getText(self, create_label, create_label + ":")
        if ok and name and self._configManager and self._config:
            if name in self._configManager.get_config_list():
                reply = QMessageBox.question(
                    self, t("confirm_overwrite"), f"{t('confirm_overwrite')}: {name}?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                if reply != QMessageBox.StandardButton.Yes:
                    return
            try:
                if self._configManager.save_config(self._config, name):
                    self._refreshConfigList()
                    self._showInfo(t("config_success"), _t(self._i18n["saved"]))
                else:
                    self._showInfo(t("config_error"), _t(self._i18n["save_failed"]), False)
            except Exception as e:
                self._showInfo(t("config_error"), str(e), False)

    def _onLoadConfig(self):
        name = self._getSelectedConfig()
        if not name:
            self._showInfo(t("config_warning"), t("no_selection"), False)
            return
        if not (self._configManager and self._config):
            return

        # Preview what applying this file would actually change before
        # doing it — a dry run via preview_config_changes(), never the
        # live 'config' instance the rest of the app is using. Only prompt
        # when something would genuinely change; identical-to-current (or
        # a preview failure, e.g. a corrupt file — the same failure mode
        # load_config() itself would hit) falls straight through to the
        # load below, matching this button's original no-prompt behavior.
        try:
            changes = self._configManager.preview_config_changes(self._config, name)
        except Exception:
            changes = None

        if changes:
            bullet_lines = "\n".join(f"• {c}" for c in changes)
            reply = QMessageBox.question(
                self, _t(self._i18n["load"]),
                f"{_t(self._i18n['diff_will_change'])}\n\n{bullet_lines}",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        try:
            if self._configManager.load_config(self._config, name):
                self._showInfo(t("config_success"), _t(self._i18n["loaded"]))
                window = self.window()
                if hasattr(window, '_refreshAllPages'):
                    window._refreshAllPages()
            else:
                self._showInfo(t("config_error"), _t(self._i18n["load_failed"]), False)
        except Exception:
            self._showInfo(t("config_error"), _t(self._i18n["load_failed"]), False)

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
                    if self._configManager.save_config(self._config, name):
                        self._showInfo(t("config_success"), _t(self._i18n["saved"]))
                    else:
                        self._showInfo(t("config_error"), _t(self._i18n["save_failed"]), False)
                except Exception:
                    self._showInfo(t("config_error"), _t(self._i18n["save_failed"]), False)

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
                    if self._configManager.delete_config(name):
                        self._refreshConfigList()
                        self._showInfo(t("config_success"), _t(self._i18n["deleted"]))
                    else:
                        self._showInfo(t("config_error"), _t(self._i18n["delete_failed"]), False)
                except Exception as e:
                    self._showInfo(t("config_error"), str(e), False)

    def _onRenameConfig(self):
        old_name = self._getSelectedConfig()
        if not old_name:
            self._showInfo(t("config_warning"), t("no_selection"), False)
            return
        rename_label = _t(self._i18n["rename"])
        new_name, ok = QInputDialog.getText(
            self, rename_label, rename_label + ":", text=old_name
        )
        if ok and new_name and new_name != old_name:
            if self._configManager:
                try:
                    if self._configManager.rename_config(old_name, new_name):
                        self._refreshConfigList()
                        self._showInfo(t("config_success"), _t(self._i18n["saved"]))
                    else:
                        self._showInfo(t("config_error"), _t(self._i18n["rename_failed"]), False)
                except Exception as e:
                    self._showInfo(t("config_error"), str(e), False)

    def _onImportConfig(self):
        import_label = _t(self._i18n["import_"])
        path, _ = QFileDialog.getOpenFileName(self, import_label, "", "JSON Files (*.json)")
        if path and self._configManager:
            try:
                name = self._configManager.import_config(path)
                if name:
                    self._refreshConfigList()
                    self._showInfo(t("config_success"), _t(self._i18n["loaded"]))
                else:
                    self._showInfo(t("config_error"), _t(self._i18n["load_failed"]), False)
            except Exception as e:
                self._showInfo(t("config_error"), str(e), False)

    def _onExportConfig(self):
        name = self._getSelectedConfig()
        if not name:
            self._showInfo(t("config_warning"), t("no_selection"), False)
            return
        export_label = _t(self._i18n["export"])
        path, _ = QFileDialog.getSaveFileName(self, export_label, f"{name}.json", "JSON Files (*.json)")
        if path and self._configManager:
            try:
                self._configManager.export_config(name, path)
                self._showInfo(t("config_success"), _t(self._i18n["saved"]))
            except Exception as e:
                self._showInfo(t("config_error"), str(e), False)

    def _onOpenFolder(self):
        if self._configManager:
            folder = self._configManager.configs_dir
            if os.path.exists(folder):
                os.startfile(folder)

    def retranslateUi(self):
        self.listTitle.setText(_t(self._i18n["list_title"]))
        self.buttonTitle.setText(_t(self._i18n["box_title"]))
        self.createBtn.setText(_t(self._i18n["create"]))
        self.loadBtn.setText(_t(self._i18n["load"]))
        self.saveBtn.setText(_t(self._i18n["save"]))
        self.deleteBtn.setText(_t(self._i18n["delete"]))
        self.renameBtn.setText(_t(self._i18n["rename"]))
        self.refreshBtn.setText(_t(self._i18n["refresh"]))
        self.importBtn.setText(_t(self._i18n["import_"]))
        self.exportBtn.setText(_t(self._i18n["export"]))
        self.openFolderBtn.setText(_t(self._i18n["open_folder"]))


class ConfigsPage(BasePage):
    """The "Configs" nav page — two stacked, always-visible boxes:

    - **Config** (top): full settings snapshots — every Config field,
      the same "everything" a hand-edited config.json would carry. Backed
      by a ConfigManager(configs_dir="configs", aim_only=False).
    - **Preset** (bottom): aim-only settings — lets the user swap
      aiming/tracking/humanization behavior without touching capture,
      model, display, hardware, or any other setting. Backed by the
      existing ConfigManager(configs_dir="presets") (aim_only=True, the
      default).

    Both boxes are plain _ManagerBox instances differing only in which
    ConfigManager and i18n dict they're constructed with.
    """

    def __init__(self, parent=None):
        super().__init__("tab_config_management", parent)
        self._config = None
        self._configManager = None       # aim-only Preset manager
        self._fullConfigManager = None   # full Config manager

        self.configBox = _ManagerBox(_CONFIG_I18N, self)
        self.presetBox = _ManagerBox(_PRESET_I18N, self)

        self.addContent(self.configBox)
        self.addContent(self.presetBox)
        self.scrollLayout.addStretch(1)

        qconfig.themeChanged.connect(self._applyPanelStyles)

    def setConfig(self, config):
        """Binds the live Config instance."""
        self._config = config
        self.configBox.setConfig(config)
        self.presetBox.setConfig(config)
        self._applyPanelStyles()

    def setConfigManager(self, manager):
        """Binds the aim-only Preset ConfigManager instance."""
        self._configManager = manager
        self.presetBox.setConfigManager(manager)

    def setFullConfigManager(self, manager):
        """Binds the full-config-snapshot ConfigManager instance."""
        self._fullConfigManager = manager
        self.configBox.setConfigManager(manager)

    def _applyPanelStyles(self, *_):
        if self._config is None:
            return
        self.configBox.applyPanelStyles(self._config)
        self.presetBox.applyPanelStyles(self._config)

    def retranslateUi(self):
        """Refreshes translated text on both boxes."""
        super().retranslateUi()
        self.configBox.retranslateUi()
        self.presetBox.retranslateUi()
