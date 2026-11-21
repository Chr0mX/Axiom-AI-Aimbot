"""
Base Page for Axiom GUI
Provides a base page class with large title and smooth scroll area.
"""
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QFrame
from qfluentwidgets import SmoothScrollArea, TitleLabel, FluentWindow, SwitchSettingCard, FluentIcon
from .components.advanced import AdvancedVisibility

class BasePage(QWidget):
    """
    Base page with a large title and smooth scroll area.
    Implements the iOS-like 'Large Title' behavior.
    """
    
    def __init__(self, titleKey: str, parent=None):
        super().__init__(parent)
        from .language_manager import t
        self.titleKey = titleKey
        self.pageTitle = t(titleKey)
        # Advanced cards (see components/advanced.py): hidden unless the global switch is on.
        # Before a Config is bound nothing is hidden, so there is no flash of missing settings.
        self._advanced = AdvancedVisibility(self._advancedOn)
        self.advancedSwitchCard = None
        self.setStyleSheet("background: transparent;")
        
        # Main layout
        self.mainLayout = QVBoxLayout(self)
        self.mainLayout.setContentsMargins(0, 0, 0, 0)
        self.mainLayout.setSpacing(0)
        
        # Scroll Area
        self.scrollArea = SmoothScrollArea(self)
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setFrameShape(QFrame.Shape.NoFrame)
        self.scrollArea.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scrollArea.setViewportMargins(0, 0, 0, 0)
        self.scrollArea.setStyleSheet("background: transparent; border: none;")
        
        # Content Widget inside Scroll Area
        self.scrollWidget = QWidget()
        self.scrollWidget.setStyleSheet("background: transparent;")
        self.scrollLayout = QVBoxLayout(self.scrollWidget)
        self.scrollLayout.setContentsMargins(36, 20, 36, 36)
        self.scrollLayout.setSpacing(20)
        
        # Large Title
        self.largeTitleLabel = TitleLabel(self.pageTitle, self.scrollWidget)
        # Apply custom font styling if needed, but TitleLabel is usually good
        # We might want it even bigger/bolder
        font = self.largeTitleLabel.font()
        font.setPixelSize(32)
        font.setBold(True)
        self.largeTitleLabel.setFont(font)
        
        self.scrollLayout.addWidget(self.largeTitleLabel)
        
        self.scrollArea.setWidget(self.scrollWidget)
        self.mainLayout.addWidget(self.scrollArea)
        
        # Connect scroll signal
        self.scrollArea.verticalScrollBar().valueChanged.connect(self._onScroll)
        
    def _onScroll(self, value):
        """
        Handle scroll events to toggle the window title.
        """
        # Threshold to switch title location (approx header height)
        threshold = 50 
        
        window = self.window()
        if isinstance(window, FluentWindow):
            if value > threshold:
                if window.titleBar.titleLabel.text() != self.pageTitle:
                    window.titleBar.titleLabel.setText(self.pageTitle)
            else:
                if window.titleBar.titleLabel.text() != "Axiom":
                    window.titleBar.titleLabel.setText("Axiom")
                    
    # ── global "Show advanced settings" ───────────────────────────────
    def _advancedOn(self) -> bool:
        cfg = getattr(self, '_config', None)
        return True if cfg is None else bool(getattr(cfg, 'show_advanced_settings', True))

    def markAdvanced(self, *widgets) -> None:
        """Cards (or whole groups) that only show while the global switch is on."""
        self._advanced.register(*widgets)

    def markAdvancedAttrs(self, *names: str) -> None:
        """Same, by attribute name; names this page does not have are skipped."""
        self._advanced.register_names(self, names)

    def addAdvancedSwitch(self) -> None:
        """The shared header switch, right under the page title. One persisted setting for the
        whole app: any page's switch (or the Web Control one) changes them all."""
        from .language_manager import t
        card = SwitchSettingCard(
            FluentIcon.SETTING, t("show_advanced_settings", "Show advanced settings"),
            t("show_advanced_settings_desc",
              "Also show fine-tuning options. Hidden options keep their values."),
            parent=self.scrollWidget)
        card.checkedChanged.connect(self._onAdvancedToggled)
        self.scrollLayout.insertWidget(1, card)
        self.advancedSwitchCard = card
        self.applyAdvanced()

    def applyAdvanced(self) -> None:
        """Show/hide the advanced cards and sync the switch to the config (no signal, no write)."""
        card = self.advancedSwitchCard
        if card is not None:
            on = self._advancedOn()
            if bool(card.isChecked()) != on:
                card.blockSignals(True)
                card.setChecked(on)
                card.blockSignals(False)
        self._advanced.apply()

    def _onAdvancedToggled(self, checked: bool) -> None:
        cfg = getattr(self, '_config', None)
        if cfg is not None:
            cfg.show_advanced_settings = bool(checked)
        win = self.window()
        if hasattr(win, 'applyAdvancedSettings'):
            win.applyAdvancedSettings()
        else:
            self.applyAdvanced()

    def addContent(self, widget: QWidget):
        """Add a widget to the scrollable content area."""
        self.scrollLayout.addWidget(widget)

    def addLayout(self, layout):
        """Add a layout to the scrollable content area."""
        self.scrollLayout.addLayout(layout)

    def retranslateUi(self):
        """Update localized text."""
        from .language_manager import t
        self.pageTitle = t(self.titleKey)
        self.largeTitleLabel.setText(self.pageTitle)
        if self.advancedSwitchCard is not None:
            self.advancedSwitchCard.titleLabel.setText(t("show_advanced_settings", "Show advanced settings"))
            self.advancedSwitchCard.contentLabel.setText(t(
                "show_advanced_settings_desc", "Also show fine-tuning options. Hidden options keep their values."))
