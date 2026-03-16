"""
Base Page for Axiom GUI
Provides a base page class with large title and smooth scroll area.
"""
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QFrame
from qfluentwidgets import SmoothScrollArea, TitleLabel, FluentWindow

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
