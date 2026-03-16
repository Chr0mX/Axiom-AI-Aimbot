import os
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QTextBrowser, QPushButton, 
                             QHBoxLayout, QLabel, QWidget)
from PyQt6.QtCore import Qt, QSize
from qfluentwidgets import PrimaryPushButton, PushButton, StrongBodyLabel

class DisclaimerDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Axiom - Disclaimer / Terms of Use")
        self.resize(800, 600)
        # self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.Dialog) # Optional: keep frameless if desired
        
        self.setup_ui()
        self.load_disclaimer()
        
        # Simple styling
        self.setStyleSheet("background-color: white; color: black;")
        # If dark mode needed, we might need more logic or just rely on system title bar

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)
        
        # Title
        title_label = StrongBodyLabel("Disclaimer / Terms of Use", self)
        title_label.setStyleSheet("font-size: 24px; font-weight: bold;")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)
        
        # Text Browser
        self.text_browser = QTextBrowser()
        self.text_browser.setOpenExternalLinks(True)
        # Basic clean style
        self.text_browser.setStyleSheet("""
            QTextBrowser {
                border: 1px solid #ccc;
                border-radius: 18px;
                background-color: #f9f9f9;
                padding: 10px;
                font-size: 14px;
                color: #333;
            }
        """)
        layout.addWidget(self.text_browser)
        
        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(20)
        
        self.btn_exit = PushButton("Decline & Exit")
        self.btn_exit.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_exit.clicked.connect(self.reject)
        
        self.btn_agree = PrimaryPushButton("I Agree")
        self.btn_agree.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_agree.clicked.connect(self.accept)
        
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_exit)
        btn_layout.addWidget(self.btn_agree)
        btn_layout.addStretch()
        
        layout.addLayout(btn_layout)

    def load_disclaimer(self):
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(current_dir))
            md_path = os.path.join(project_root, "Disclaimer.md")
            
            if os.path.exists(md_path):
                with open(md_path, "r", encoding="utf-8") as f:
                    content = f.read()
                self.text_browser.setMarkdown(content)
            else:
                self.text_browser.setText("Disclaimer.md not found.")
        except Exception as e:
            self.text_browser.setText(f"Error loading disclaimer: {str(e)}")
