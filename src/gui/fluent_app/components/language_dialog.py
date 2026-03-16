import os
from PyQt6.QtCore import Qt, pyqtSignal, QSize
from PyQt6.QtWidgets import QVBoxLayout, QHBoxLayout, QWidget, QLabel, QFrame
from PyQt6.QtGui import QFont, QColor, QPixmap, QIcon
from PyQt6.QtSvg import QSvgRenderer
from PyQt6.QtCore import QByteArray
from qfluentwidgets import (MaskDialogBase, PrimaryPushButton, PushButton,
                            RadioButton, ScrollArea, BodyLabel,
                            SubtitleLabel, isDarkTheme)
from ..theme_colors import ThemeColors


# Get flags directory
FLAGS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "flags")

# Language data: (code, name, flag_file)
# Must match files in src/language_data/
LANGUAGES = [
    ("English", "English", "us.svg"),
    ("Chinese", "中文", "tw.svg"),
    ("Japanese", "日本語", "jp.svg"),
    ("Korean", "한국어", "kr.svg"),
    ("German", "Deutsch", "de.svg"),
    ("French", "Français", "fr.svg"),
    ("Spanish", "Español", "es.svg"),
    ("Portuguese", "Português", "br.svg"),
    ("Russian", "Русский", "ru.svg"),
    ("Hindi", "हिन्दी", "in.svg"),
]


class LanguageCard(QFrame):
    """A single language option card with flag and name."""
    
    selected = pyqtSignal(str)  # Emits language code
    
    def __init__(self, code: str, name: str, flag_file: str, parent=None):
        super().__init__(parent)
        self.code = code
        self.setFixedHeight(50)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._applyStyles()
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 8, 16, 8)
        layout.setSpacing(12)
        
        # Radio button
        self.radio = RadioButton(self)
        layout.addWidget(self.radio)
        
        # Flag image
        self.flagLabel = QLabel(self)
        self.flagLabel.setFixedSize(32, 24)
        flag_path = os.path.join(FLAGS_DIR, flag_file)
        if os.path.exists(flag_path):
            pixmap = QPixmap(flag_path)
            if not pixmap.isNull():
                self.flagLabel.setPixmap(pixmap.scaled(32, 24, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
            else:
                self.flagLabel.setText(code[:2].upper())
        else:
            self.flagLabel.setText(code[:2].upper())
        self.flagLabel.setStyleSheet("background: transparent; border-radius: 6px;")
        layout.addWidget(self.flagLabel)
        
        # Language name
        self.nameLabel = BodyLabel(name, self)
        layout.addWidget(self.nameLabel)
        
        layout.addStretch()
        
        # Connect radio button
        self.radio.clicked.connect(lambda: self.selected.emit(self.code))
    
    def _applyStyles(self):
        """應用根據主題動態切換的樣式"""
        # get() 會自動判斷當前主題
        bg = ThemeColors.DIALOG_ITEM_BACKGROUND.get()
        border = ThemeColors.DIALOG_ITEM_BORDER.get()
        hover_bg = ThemeColors.DIALOG_ITEM_HOVER.get()
        hover_border = ThemeColors.BORDER_DEFAULT.get()
        
        self.setStyleSheet(f"""
            LanguageCard {{
                background-color: {bg};
                border-radius: 18px;
                border: 1px solid {border};
            }}
            LanguageCard:hover {{
                background-color: {hover_bg};
                border: 1px solid {hover_border};
            }}
        """)
        
    def setChecked(self, checked: bool):
        self.radio.setChecked(checked)
        
    def isChecked(self) -> bool:
        return self.radio.isChecked()
    
    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        self.radio.setChecked(True)
        self.selected.emit(self.code)


class LanguageDialog(MaskDialogBase):
    """Language selection dialog with flags."""
    
    languageChanged = pyqtSignal(str)  # Emits selected language code
    
    def __init__(self, currentLanguage: str = "English", parent=None):
        super().__init__(parent)
        # Disable shadow effect to prevent QPainter conflicts
        self.widget.setGraphicsEffect(None)
        self.currentLanguage = currentLanguage
        self.selectedLanguage = currentLanguage
        self.languageCards = []
        
        # Main widget
        self.widget = QWidget(self)
        self.widget.setFixedSize(400, 500)
        self._applyWidgetStyles()
        
        # Layout
        self.mainLayout = QVBoxLayout(self.widget)
        self.mainLayout.setContentsMargins(24, 24, 24, 24)
        self.mainLayout.setSpacing(16)
        
        # Title
        self.titleLabel = SubtitleLabel("Select Language", self.widget)
        self.titleLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        titleFont = self.titleLabel.font()
        titleFont.setPixelSize(20)
        titleFont.setBold(True)
        self.titleLabel.setFont(titleFont)
        self.mainLayout.addWidget(self.titleLabel)
        
        # Scroll area for languages
        self.scrollArea = ScrollArea(self.widget)
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setStyleSheet("background: transparent; border: none;")
        
        self.scrollWidget = QWidget()
        self.scrollWidget.setStyleSheet("background: transparent;")
        self.scrollLayout = QVBoxLayout(self.scrollWidget)
        self.scrollLayout.setContentsMargins(0, 0, 8, 0)
        self.scrollLayout.setSpacing(8)
        
        # Add language cards
        for code, name, flag_file in LANGUAGES:
            card = LanguageCard(code, name, flag_file, self.scrollWidget)
            card.setChecked(code == self.currentLanguage)
            card.selected.connect(self._onLanguageSelected)
            self.languageCards.append(card)
            self.scrollLayout.addWidget(card)
        
        self.scrollLayout.addStretch()
        self.scrollArea.setWidget(self.scrollWidget)
        self.mainLayout.addWidget(self.scrollArea, 1)
        
        # Buttons
        self.buttonLayout = QHBoxLayout()
        self.buttonLayout.setSpacing(12)
        
        self.cancelButton = PushButton("Cancel", self.widget)
        self.cancelButton.setFixedWidth(100)
        self.cancelButton.clicked.connect(self.reject)
        
        self.confirmButton = PrimaryPushButton("Confirm", self.widget)
        self.confirmButton.setFixedWidth(100)
        self.confirmButton.clicked.connect(self._onConfirm)
        
        self.buttonLayout.addStretch()
        self.buttonLayout.addWidget(self.cancelButton)
        self.buttonLayout.addWidget(self.confirmButton)
        
        self.mainLayout.addLayout(self.buttonLayout)
        
        # Set mask color (removed shadow effect to avoid QPainter errors)
        self.setMaskColor(QColor(0, 0, 0, 150))
    
    def _applyWidgetStyles(self):
        """應用根據主題動態切換的對話框樣式"""
        # get() 會自動判斷當前主題
        bg = ThemeColors.DIALOG_BACKGROUND.get()
        
        self.widget.setStyleSheet(f"""
            QWidget {{
                background-color: {bg};
                border-radius: 22px;
            }}
        """)
    
    def showEvent(self, e):
        """Override to skip opacity animation that causes QPainter conflicts."""
        # Skip MaskDialogBase's showEvent animation, call QDialog directly
        from PyQt6.QtWidgets import QDialog
        QDialog.showEvent(self, e)
    
    def done(self, code):
        """Override to skip opacity animation that causes QPainter conflicts."""
        # Skip MaskDialogBase's done animation, call QDialog directly
        from PyQt6.QtWidgets import QDialog
        QDialog.done(self, code)
        
    def _onLanguageSelected(self, code: str):
        """Handle language card selection."""
        self.selectedLanguage = code
        for card in self.languageCards:
            card.setChecked(card.code == code)
    
    def _onConfirm(self):
        """Confirm selection and close dialog."""
        if self.selectedLanguage != self.currentLanguage:
            self.languageChanged.emit(self.selectedLanguage)
        self.accept()
