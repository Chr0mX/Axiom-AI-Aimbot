# setup_wizard.py
"""First-run setup wizard
First-Run Setup Wizard — shown only on a fresh install (no config.json found).
Steps: Welcome → Language → Theme → Acrylic → Done
"""

from __future__ import annotations

import os
import platform
import re
import subprocess
import types

from PyQt6.QtCore import Qt, pyqtSignal, QSize
from PyQt6.QtGui import QColor, QFont, QIcon, QPainter, QPainterPath, QPen, QPixmap
from PyQt6.QtWidgets import (
    QAbstractSlider, QDialog, QFrame, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QScrollArea, QSizePolicy, QSlider,
    QStackedWidget, QVBoxLayout, QWidget,
)

try:
    from qfluentwidgets import (
        BodyLabel, CaptionLabel, ComboBox,
        PrimaryPushButton, PushButton, Slider,
        StrongBodyLabel, SwitchButton, TitleLabel,
        setTheme, Theme, qconfig,
    )
    _HAS_FLUENT = True
except ImportError:
    _HAS_FLUENT = False

from .language_manager import getLanguageManager, LANGUAGE_FILE_MAP

# Language list: (code, native_name, flag_svg)
_WIZARD_LANGUAGES = [
    ("English",    "English",    "us.svg"),
    ("Chinese",    "中文",       "tw.svg"),
    ("Japanese",   "日本語",     "jp.svg"),
    ("Korean",     "한국어",     "kr.svg"),
    ("German",     "Deutsch",    "de.svg"),
    ("French",     "Français",   "fr.svg"),
    ("Spanish",    "Español",    "es.svg"),
    ("Portuguese", "Português",  "br.svg"),
    ("Russian",    "Русский",    "ru.svg"),
    ("Hindi",      "हिन्दी",       "in.svg"),
]

_FLAGS_DIR = os.path.join(
    os.path.dirname(__file__), "assets", "flags")

# ──────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────

def _lbl(text: str, size: int = 13, bold: bool = False,
         color: str = "") -> QLabel:
    """Quickly create a QLabel."""
    w = QLabel(text)
    w.setWordWrap(True)
    f = QFont("Segoe UI Variable Display", size)
    if bold:
        f.setWeight(QFont.Weight.DemiBold)
    w.setFont(f)
    if color:
        w.setStyleSheet(f"color: {color};")
    return w


# ──────────────────────────────────────────────────────────
# Step Indicator
# ──────────────────────────────────────────────────────────

class _StepDots(QWidget):
    """Top step dot indicator."""

    def __init__(self, total: int, parent=None):
        super().__init__(parent)
        self._total = total
        self._current = 0
        self.setFixedHeight(20)

    def setCurrent(self, idx: int):
        self._current = idx
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = 5
        gap = 16
        total_w = self._total * 2 * r + (self._total - 1) * gap
        ox = (self.width() - total_w) // 2
        oy = self.height() // 2
        for i in range(self._total):
            cx = ox + i * (2 * r + gap) + r
            if i == self._current:
                p.setBrush(QColor("#0078D4"))
                p.setPen(Qt.PenStyle.NoPen)
                p.drawEllipse(cx - r, oy - r, 2 * r, 2 * r)
            elif i < self._current:
                p.setBrush(QColor("#50A8D4"))
                p.setPen(Qt.PenStyle.NoPen)
                p.drawEllipse(cx - r, oy - r, 2 * r, 2 * r)
            else:
                p.setBrush(QColor("#D0D0D0"))
                p.setPen(Qt.PenStyle.NoPen)
                p.drawEllipse(cx - r, oy - r, 2 * r, 2 * r)


# ──────────────────────────────────────────────────────────
# Language Card (for wizard)
# ──────────────────────────────────────────────────────────

class _WizardLangCard(QFrame):
    """Language option card with flag and native language name."""

    clicked = pyqtSignal(str)  # emits language code e.g. "English"

    def __init__(self, code: str, native_name: str, flag_file: str,
                 parent=None):
        super().__init__(parent)
        self._code = code
        self._selected = False
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedHeight(42)

        ly = QHBoxLayout(self)
        ly.setContentsMargins(14, 4, 14, 4)
        ly.setSpacing(10)

        # Flag icon
        flag_lbl = QLabel()
        flag_lbl.setFixedSize(28, 20)
        flag_path = os.path.join(_FLAGS_DIR, flag_file)
        if os.path.exists(flag_path):
            pix = QPixmap(flag_path)
            if not pix.isNull():
                flag_lbl.setPixmap(pix.scaled(
                    28, 20,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation))
        flag_lbl.setStyleSheet("background: transparent;")
        ly.addWidget(flag_lbl)

        # Language name: English Name — Native Name
        display_text = f"{code}  —  {native_name}" if code != native_name else code
        name_lbl = QLabel(display_text)
        name_lbl.setFont(QFont("Segoe UI Variable Display", 11))
        name_lbl.setStyleSheet("background: transparent;")
        ly.addWidget(name_lbl)

        ly.addStretch()

        self._refreshStyle()

    def setSelected(self, v: bool):
        self._selected = v
        self._refreshStyle()

    def _refreshStyle(self):
        if self._selected:
            self.setStyleSheet("""
                _WizardLangCard, QFrame {
                    border: 2px solid #0078D4;
                    border-radius: 10px;
                    background-color: rgba(0,120,212,0.10);
                }
            """)
        else:
            self.setStyleSheet("""
                _WizardLangCard, QFrame {
                    border: 1px solid #D0D0D0;
                    border-radius: 10px;
                    background-color: rgba(0,0,0,0.03);
                }
                _WizardLangCard:hover, QFrame:hover {
                    border: 1px solid #0078D4;
                    background-color: rgba(0,120,212,0.06);
                }
            """)

    def mousePressEvent(self, _):
        self.clicked.emit(self._code)


# ──────────────────────────────────────────────────────────
# Theme Card
# ──────────────────────────────────────────────────────────

class _ThemeCard(QFrame):
    """Clickable theme option card, uses color blocks for preview rather than emojis."""

    clicked = pyqtSignal(str)   # emits 'light' or 'dark'

    def __init__(self, mode: str, parent=None):
        super().__init__(parent)
        self._mode = mode
        self._selected = False
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedSize(180, 120)
        self.setFrameShape(QFrame.Shape.StyledPanel)

        ly = QVBoxLayout(self)
        ly.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ly.setSpacing(10)

        # 色塊預覽
        preview = QFrame()
        preview.setFixedSize(140, 56)
        if mode == "light":
            preview.setStyleSheet("""
                QFrame {
                    background-color: #F5F5F5;
                    border: 1px solid #D0D0D0;
                    border-radius: 8px;
                }
            """)
            # 內部小裝飾條
            inner_ly = QVBoxLayout(preview)
            inner_ly.setContentsMargins(10, 8, 10, 8)
            inner_ly.setSpacing(4)
            for w_pct in [70, 50, 40]:
                bar = QFrame()
                bar.setFixedHeight(6)
                bar.setFixedWidth(int(120 * w_pct / 100))
                bar.setStyleSheet("background: #C8C8C8; border-radius: 3px;")
                inner_ly.addWidget(bar)
        else:
            preview.setStyleSheet("""
                QFrame {
                    background-color: #1E1E1E;
                    border: 1px solid #444;
                    border-radius: 8px;
                }
            """)
            inner_ly = QVBoxLayout(preview)
            inner_ly.setContentsMargins(10, 8, 10, 8)
            inner_ly.setSpacing(4)
            for w_pct in [70, 50, 40]:
                bar = QFrame()
                bar.setFixedHeight(6)
                bar.setFixedWidth(int(120 * w_pct / 100))
                bar.setStyleSheet("background: #555; border-radius: 3px;")
                inner_ly.addWidget(bar)

        ly.addWidget(preview, 0, Qt.AlignmentFlag.AlignCenter)

        self._text_lbl = QLabel("")
        self._text_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._text_lbl.setFont(QFont("Segoe UI Variable Display", 11,
                                     QFont.Weight.DemiBold))
        ly.addWidget(self._text_lbl)

        self._refreshStyle()

    def setText(self, text: str):
        self._text_lbl.setText(text)

    def setSelected(self, v: bool):
        self._selected = v
        self._refreshStyle()

    def _refreshStyle(self):
        if self._selected:
            self.setStyleSheet("""
                QFrame {
                    border: 2px solid #0078D4;
                    border-radius: 12px;
                    background-color: rgba(0,120,212,0.10);
                }
            """)
        else:
            self.setStyleSheet("""
                QFrame {
                    border: 1px solid #D0D0D0;
                    border-radius: 12px;
                    background-color: rgba(0,0,0,0.03);
                }
                QFrame:hover {
                    border: 1px solid #0078D4;
                    background-color: rgba(0,120,212,0.06);
                }
            """)

    def mousePressEvent(self, _):
        self.clicked.emit(self._mode)


# ──────────────────────────────────────────────────────────
def _detect_gpu_name() -> str:
    try:
        r = subprocess.run(
            ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=5
        )
        if r.returncode == 0:
            name = r.stdout.strip().splitlines()[0]
            if name:
                return name
    except Exception:
        pass
    return ''

def _is_rtx20_or_newer(gpu_name: str) -> bool:
    m = re.search(r'RTX\s*(\d{4})', gpu_name, re.IGNORECASE)
    return bool(m) and int(m.group(1)) >= 2000


# Setup Wizard
# ──────────────────────────────────────────────────────────

class SetupWizard(QDialog):
    """首次啟動 5 步驟設置精靈。"""

    # 完成後通知外部（包含是否為深色主題）
    setupComplete = pyqtSignal()

    # 步驟常數
    STEP_WELCOME      = 0
    STEP_LANGUAGE     = 1
    STEP_THEME        = 2
    STEP_ACRYLIC      = 3
    STEP_PERFORMANCE  = 4
    STEP_DONE         = 5
    TOTAL_STEPS       = 6

    def __init__(self, config, parent=None):
        super().__init__(parent)

        self._config      = config
        self._langManager = getLanguageManager()
        self._isDark      = True    # 精靈自身預覽狀態（預設深色）

        self.setWindowTitle("Axiom – Setup")
        self.setFixedSize(580, 510)
        self.setWindowFlags(Qt.WindowType.Dialog |
                            Qt.WindowType.WindowTitleHint |
                            Qt.WindowType.WindowCloseButtonHint)

        self._buildUI()
        self._applyDarkStyle()
        self._updateTexts()

        # 語言切換時即時刷新精靈文字
        self._langManager.languageChanged.connect(lambda _: self._updateTexts())

    # ── Build UI ──────────────────────────────────────────

    def _buildUI(self):
        main = QVBoxLayout(self)
        main.setContentsMargins(32, 24, 32, 20)
        main.setSpacing(0)

        # 步驟指示器
        self._dots = _StepDots(self.TOTAL_STEPS, self)
        main.addWidget(self._dots)
        main.addSpacing(16)

        # 分頁容器
        self._stack = QStackedWidget(self)
        main.addWidget(self._stack, 1)
        main.addSpacing(20)

        # 底部按鈕列
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        self._btn_skip = PushButton("") if _HAS_FLUENT else QPushButton("")  # type: ignore[assignment]
        self._btn_skip.setFixedWidth(90)
        self._btn_skip.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_skip.clicked.connect(self._onSkip)

        self._btn_back = PushButton("") if _HAS_FLUENT else QPushButton("")  # type: ignore[assignment]
        self._btn_back.setFixedWidth(90)
        self._btn_back.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_back.clicked.connect(self._onBack)
        self._btn_back.setEnabled(False)

        self._btn_next = (PrimaryPushButton("") if _HAS_FLUENT
                         else QPushButton(""))            # type: ignore[assignment]
        self._btn_next.setFixedWidth(110)
        self._btn_next.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_next.clicked.connect(self._onNext)

        btn_row.addWidget(self._btn_skip)
        btn_row.addStretch(1)
        btn_row.addWidget(self._btn_back)
        btn_row.addWidget(self._btn_next)
        main.addLayout(btn_row)

        # 建立各步驟頁面
        self._pages = [
            self._buildWelcomePage(),
            self._buildLanguagePage(),
            self._buildThemePage(),
            self._buildAcrylicPage(),
            self._buildPerformancePage(),
            self._buildDonePage(),
        ]
        for p in self._pages:
            self._stack.addWidget(p)

    # ── Pages ─────────────────────────────────────────────

    def _buildWelcomePage(self) -> QWidget:
        w = QWidget()
        ly = QVBoxLayout(w)
        ly.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ly.setSpacing(14)

        # Logo
        logo_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "logo.png"))
        logo_lbl = QLabel()
        logo_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        if os.path.exists(logo_path):
            pix = QPixmap(logo_path).scaled(
                80, 80,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation)
            logo_lbl.setPixmap(pix)
        else:
            logo_lbl.setText("🎯")
            logo_lbl.setFont(QFont("Segoe UI Emoji", 40))
        ly.addWidget(logo_lbl)

        self._lbl_welcome_title = _lbl("", 22, bold=True)
        self._lbl_welcome_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ly.addWidget(self._lbl_welcome_title)

        self._lbl_welcome_sub = _lbl("", 12)
        self._lbl_welcome_sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ly.addWidget(self._lbl_welcome_sub)

        return w

    def _buildLanguagePage(self) -> QWidget:
        w = QWidget()
        ly = QVBoxLayout(w)
        ly.setSpacing(10)

        self._lbl_lang_title = _lbl("", 18, bold=True)
        self._lbl_lang_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ly.addWidget(self._lbl_lang_title)

        self._lbl_lang_sub = _lbl("", 12)
        self._lbl_lang_sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ly.addWidget(self._lbl_lang_sub)

        ly.addSpacing(6)

        # 可滾動的語言卡片清單
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet(
            "QScrollArea { border: none; background: transparent; }"
            "QWidget#langContainer { background: transparent; }")

        container = QWidget()
        container.setObjectName("langContainer")
        cards_ly = QVBoxLayout(container)
        cards_ly.setAlignment(Qt.AlignmentFlag.AlignTop)
        cards_ly.setContentsMargins(40, 0, 40, 0)
        cards_ly.setSpacing(6)

        cur = self._langManager.currentLanguage
        self._lang_cards: list[_WizardLangCard] = []

        for code, native_name, flag_file in _WIZARD_LANGUAGES:
            card = _WizardLangCard(code, native_name, flag_file)
            card.setSelected(code == cur)
            card.clicked.connect(self._onLangCardClicked)
            cards_ly.addWidget(card)
            self._lang_cards.append(card)

        scroll.setWidget(container)
        ly.addWidget(scroll, 1)

        return w

    def _buildThemePage(self) -> QWidget:
        w = QWidget()
        ly = QVBoxLayout(w)
        ly.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ly.setSpacing(16)

        self._lbl_theme_title = _lbl("", 18, bold=True)
        self._lbl_theme_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ly.addWidget(self._lbl_theme_title)

        self._lbl_theme_sub = _lbl("", 12)
        self._lbl_theme_sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ly.addWidget(self._lbl_theme_sub)

        ly.addSpacing(16)

        cards_row = QHBoxLayout()
        cards_row.setAlignment(Qt.AlignmentFlag.AlignCenter)
        cards_row.setSpacing(24)

        self._card_light = _ThemeCard("light")
        self._card_dark  = _ThemeCard("dark")
        self._card_dark.setSelected(True)    # 預設 dark

        self._card_light.clicked.connect(self._onThemeCardClicked)
        self._card_dark.clicked.connect(self._onThemeCardClicked)

        cards_row.addWidget(self._card_light)
        cards_row.addWidget(self._card_dark)
        ly.addLayout(cards_row)

        return w

    def _buildAcrylicPage(self) -> QWidget:
        w = QWidget()
        ly = QVBoxLayout(w)
        ly.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ly.setSpacing(14)

        self._lbl_acrylic_title = _lbl("", 18, bold=True)
        self._lbl_acrylic_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ly.addWidget(self._lbl_acrylic_title)

        self._lbl_acrylic_sub = _lbl("", 12)
        self._lbl_acrylic_sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ly.addWidget(self._lbl_acrylic_sub)

        ly.addSpacing(20)

        # Enable toggle
        toggle_row = QHBoxLayout()
        toggle_row.setAlignment(Qt.AlignmentFlag.AlignCenter)
        toggle_row.setSpacing(12)

        self._lbl_acrylic_enable = _lbl("", 12)
        self._lbl_acrylic_enable.setMinimumWidth(300)
        if _HAS_FLUENT:
            self._sw_acrylic = SwitchButton()
        else:
            from PyQt6.QtWidgets import QCheckBox
            self._sw_acrylic = QCheckBox()          # type: ignore[assignment]

        self._sw_acrylic.setChecked(self._config.enable_acrylic)
        self._sw_acrylic.checkedChanged.connect(self._onAcrylicToggle)

        toggle_row.addWidget(self._sw_acrylic)
        toggle_row.addWidget(self._lbl_acrylic_enable)
        ly.addLayout(toggle_row)

        # Windows 11 only hint
        self._lbl_acrylic_hint = _lbl("", 9)
        self._lbl_acrylic_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._lbl_acrylic_hint.setStyleSheet("color: #888888;")
        ly.addWidget(self._lbl_acrylic_hint)

        ly.addSpacing(16)

        # Opacity label + slider
        self._lbl_opacity = _lbl("", 11)
        self._lbl_opacity.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ly.addWidget(self._lbl_opacity)

        slider_row = QHBoxLayout()
        slider_row.setAlignment(Qt.AlignmentFlag.AlignCenter)
        slider_row.setSpacing(12)

        if _HAS_FLUENT:
            self._slider_opacity = Slider(Qt.Orientation.Horizontal)
        else:
            self._slider_opacity = QSlider(Qt.Orientation.Horizontal)  # type: ignore[assignment]

        self._slider_opacity.setFixedWidth(260)
        self._slider_opacity.setRange(60, 255)
        self._slider_opacity.setValue(self._config.acrylic_window_alpha)
        self._slider_opacity.valueChanged.connect(self._onOpacityChanged)

        self._lbl_opacity_val = _lbl("", 11, bold=True)
        self._lbl_opacity_val.setFixedWidth(40)
        self._onOpacityChanged(self._config.acrylic_window_alpha)  # init label

        slider_row.addWidget(self._slider_opacity)
        slider_row.addWidget(self._lbl_opacity_val)
        ly.addLayout(slider_row)

        return w

    def _buildPerformancePage(self) -> QWidget:
        w = QWidget()
        ly = QVBoxLayout(w)
        ly.setAlignment(Qt.AlignmentFlag.AlignTop)
        ly.setSpacing(14)

        self._lbl_perf_title = _lbl("", 18, bold=True)
        self._lbl_perf_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ly.addWidget(self._lbl_perf_title)

        self._lbl_perf_sub = _lbl("", 12)
        self._lbl_perf_sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ly.addWidget(self._lbl_perf_sub)

        ly.addSpacing(16)

        # ── Hardware Detection Row ────────────────────────────
        self._detected_gpu = _detect_gpu_name()
        cpu_name = platform.processor() or platform.machine()

        hw_row = QHBoxLayout()
        hw_row.setSpacing(12)
        hw_row.addWidget(_lbl("Hardware", 11, bold=True))
        hw_row.addStretch()
        hw_row.addWidget(_lbl(f"GPU: {self._detected_gpu or 'Not detected'}", 10))
        hw_row.addSpacing(12)
        hw_row.addWidget(_lbl(f"CPU: {cpu_name or 'Unknown'}", 10))
        ly.addLayout(hw_row)

        ly.addSpacing(8)

        # ── Inference Backend ─────────────────────────────────
        backend_row = QHBoxLayout()
        backend_row.setSpacing(12)

        self._lbl_perf_backend = _lbl("Inference Backend", 11, bold=True)
        backend_row.addWidget(self._lbl_perf_backend)
        backend_row.addStretch()

        if _HAS_FLUENT:
            self._combo_backend = ComboBox()
        else:
            from PyQt6.QtWidgets import QComboBox
            self._combo_backend = QComboBox()  # type: ignore[assignment]

        self._combo_backend.addItems(["TensorRT (NVIDIA GPU)", "DirectML (AMD / Intel / NVIDIA)"])

        # Auto-select backend based on detected GPU
        if self._detected_gpu:
            if _is_rtx20_or_newer(self._detected_gpu):
                auto_idx = 0  # TensorRT
                self._config.inference_backend = 'tensorrt'
            else:
                auto_idx = 1  # DirectML
                self._config.inference_backend = 'directml'
        else:
            cur_backend = getattr(self._config, 'inference_backend', 'auto')
            auto_idx = 1 if cur_backend == "directml" else 0

        self._combo_backend.setCurrentIndex(auto_idx)
        self._combo_backend.currentIndexChanged.connect(self._onBackendChanged)
        backend_row.addWidget(self._combo_backend)
        ly.addLayout(backend_row)

        ly.addSpacing(10)

        # ── Model section (TensorRT only) ─────────────────────
        self._model_section = QWidget()
        model_ly = QVBoxLayout(self._model_section)
        model_ly.setContentsMargins(0, 0, 0, 0)
        model_ly.setSpacing(6)

        self._lbl_perf_model = _lbl("", 11, bold=True)
        model_ly.addWidget(self._lbl_perf_model)

        model_row = QHBoxLayout()
        model_row.setSpacing(6)

        if _HAS_FLUENT:
            self._combo_model = ComboBox()
        else:
            from PyQt6.QtWidgets import QComboBox
            self._combo_model = QComboBox()  # type: ignore[assignment]

        self._combo_model.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._combo_model.currentIndexChanged.connect(self._onModelComboChanged)
        model_row.addWidget(self._combo_model, 1)

        self._btn_refresh_model = QPushButton("↻")
        self._btn_refresh_model.setFixedWidth(30)
        self._btn_refresh_model.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_refresh_model.clicked.connect(self._refreshModelCombo)
        model_row.addWidget(self._btn_refresh_model)

        model_ly.addLayout(model_row)
        ly.addWidget(self._model_section)

        # Populate model combo and set initial visibility
        self._refreshModelCombo()
        self._model_section.setVisible(self._combo_backend.currentIndex() == 0)

        return w

    # ── Performance Page Handlers ─────────────────────────

    def _onBackendChanged(self, idx: int) -> None:
        self._config.inference_backend = "directml" if idx == 1 else "tensorrt"
        self._model_section.setVisible(idx == 0)

    def _refreshModelCombo(self) -> None:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))))  # fluent_app→gui→src→project root
        model_dir = os.path.join(project_root, "Model")
        self._combo_model.blockSignals(True)
        self._combo_model.clear()
        try:
            files = sorted(
                f for f in os.listdir(model_dir)
                if f.lower().endswith(".onnx")
            ) if os.path.isdir(model_dir) else []
        except OSError:
            files = []
        if files:
            for f in files:
                self._combo_model.addItem(f, f"Model/{f}")
            cur = getattr(self._config, 'model_path', '')
            idx = self._combo_model.findData(cur)
            self._combo_model.setCurrentIndex(idx if idx >= 0 else 0)
            self._combo_model.setEnabled(True)
            if self._config:
                data = self._combo_model.currentData() or f"Model/{self._combo_model.currentText()}"
                self._config.model_path = data
        else:
            self._combo_model.addItem("— No models found —", "")
            self._combo_model.setEnabled(False)
        self._combo_model.blockSignals(False)

    def _onModelComboChanged(self, _idx: int) -> None:
        data = self._combo_model.currentData()
        if not data:
            text = self._combo_model.currentText()
            if text and not text.startswith('—'):
                data = f"Model/{text}"
        if data and self._config:
            self._config.model_path = data

    def _buildDonePage(self) -> QWidget:
        w = QWidget()
        ly = QVBoxLayout(w)
        ly.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ly.setSpacing(16)

        done_icon = QLabel("✅")
        done_icon.setFont(QFont("Segoe UI Emoji", 42))
        done_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ly.addWidget(done_icon)

        self._lbl_done_title = _lbl("", 20, bold=True)
        self._lbl_done_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ly.addWidget(self._lbl_done_title)

        self._lbl_done_sub = _lbl("", 12)
        self._lbl_done_sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ly.addWidget(self._lbl_done_sub)

        return w

    # ── Navigation ────────────────────────────────────────

    def _onNext(self):
        cur = self._stack.currentIndex()
        if cur == self.STEP_DONE:
            self._finish()
        elif cur < self.STEP_DONE:
            self._goTo(cur + 1)

    def _onBack(self):
        cur = self._stack.currentIndex()
        if cur > 0:
            self._goTo(cur - 1)

    def _onSkip(self):
        """跳過精靈直接使用預設值。"""
        self._finish()

    def _goTo(self, idx: int):
        self._stack.setCurrentIndex(idx)
        self._dots.setCurrent(idx)
        self._btn_back.setEnabled(idx > 0)
        self._updateNextButton(idx)

    def _updateNextButton(self, idx: int):
        lm = self._langManager
        if idx == self.STEP_DONE:
            self._btn_next.setText(lm.get("wizard_finish", "Finish"))
            self._btn_skip.setVisible(False)
        else:
            self._btn_next.setText(lm.get("wizard_next", "Next"))
            self._btn_skip.setVisible(True)

    def _finish(self):
        backend = getattr(self._config, 'inference_backend', 'auto')
        if backend == "tensorrt":
            try:
                import onnxruntime as _ort
                trt_ready = "TensorrtExecutionProvider" in _ort.get_available_providers()
            except Exception:
                trt_ready = False
            if not trt_ready:
                # setup_wizard.py lives at src/gui/fluent_app/ → up 3 = project root
                project_root = os.path.dirname(os.path.dirname(
                    os.path.dirname(os.path.abspath(__file__))))
                bat = os.path.join(project_root, "Install TensorRT.bat")
                if os.path.exists(bat):
                    import subprocess
                    subprocess.Popen([bat], shell=True)
        self.setupComplete.emit()
        self.accept()

    # ── Handlers ─────────────────────────────────────────

    def _onLangCardClicked(self, lang_code: str):
        # 更新卡片選中狀態
        for card in self._lang_cards:
            card.setSelected(card._code == lang_code)
        # 切換語言
        if lang_code in LANGUAGE_FILE_MAP:
            self._langManager.setLanguage(lang_code)

    def _onThemeCardClicked(self, mode: str):
        self._isDark = (mode == "dark")
        self._card_light.setSelected(mode == "light")
        self._card_dark.setSelected(mode == "dark")
        # 即時預覽精靈主題
        if self._isDark:
            self._applyDarkStyle()
        else:
            self._applyLightStyle()

    def _onAcrylicToggle(self, checked: bool):
        self._config.enable_acrylic = checked
        self._slider_opacity.setEnabled(checked)
        self._lbl_opacity.setEnabled(checked)
        self._lbl_opacity_val.setEnabled(checked)

    def _onOpacityChanged(self, val: int):
        self._config.acrylic_window_alpha = val
        pct = round(val / 255 * 100)
        self._lbl_opacity_val.setText(f"{pct}%")

    # ── Text Update (language-aware) ──────────────────────

    def _updateTexts(self):
        lm = self._langManager

        # Welcome
        self._lbl_welcome_title.setText(
            lm.get("wizard_welcome_title", "Welcome to Axiom"))
        self._lbl_welcome_sub.setText(
            lm.get("wizard_welcome_subtitle",
                   "Let's take a moment to complete the basic setup."))

        # Language
        self._lbl_lang_title.setText(
            lm.get("wizard_language_title", "Choose Language"))
        self._lbl_lang_sub.setText(
            lm.get("wizard_language_subtitle",
                   "Select your preferred interface language."))

        # Theme
        self._lbl_theme_title.setText(
            lm.get("wizard_theme_title", "Choose Theme"))
        self._lbl_theme_sub.setText(
            lm.get("wizard_theme_subtitle",
                   "Select the visual style you prefer."))
        self._card_light.setText(lm.get("wizard_theme_light", "Light"))
        self._card_dark.setText(lm.get("wizard_theme_dark", "Dark"))

        # Acrylic
        self._lbl_acrylic_title.setText(
            lm.get("wizard_acrylic_title", "Background Effect"))
        self._lbl_acrylic_sub.setText(
            lm.get("wizard_acrylic_subtitle",
                   "Acrylic blur creates a layered, translucent look."))
        self._lbl_acrylic_enable.setText(
            lm.get("wizard_acrylic_enable", "Enable Acrylic (frosted glass)"))
        self._lbl_acrylic_hint.setText(
            lm.get("enable_acrylic_hint", "Only available on Windows 11"))
        self._lbl_opacity.setText(
            lm.get("wizard_acrylic_opacity", "Window Opacity"))

        # Performance
        self._lbl_perf_title.setText(
            lm.get("wizard_perf_title", "Performance Settings"))
        self._lbl_perf_sub.setText(
            lm.get("wizard_perf_subtitle",
                   "Configure inference and tracking options. These can be changed later in the Aim tab."))
        self._lbl_perf_model.setText(
            lm.get("wizard_perf_model_path", "Model"))

        # Done
        self._lbl_done_title.setText(
            lm.get("wizard_done_title", "All Set!"))
        self._lbl_done_sub.setText(
            lm.get("wizard_done_subtitle",
                   "Setup complete. Enjoy using Axiom!"))

        # Buttons
        self._btn_skip.setText(lm.get("wizard_skip", "Skip"))
        self._btn_back.setText(lm.get("wizard_back", "Back"))
        self._updateNextButton(self._stack.currentIndex())

    # ── Theme Preview ─────────────────────────────────────

    def _applyLightStyle(self):
        self.setStyleSheet("""
            QDialog {
                background-color: #F5F5F5;
            }
            QLabel {
                color: #1A1A1A;
                background: transparent;
            }
            QFrame {
                background: transparent;
            }
        """)

    def _applyDarkStyle(self):
        self.setStyleSheet("""
            QDialog {
                background-color: #1E1E1E;
            }
            QLabel {
                color: #F0F0F0;
                background: transparent;
            }
            QFrame {
                background: transparent;
            }
        """)
        # 更新 theme cards border color for dark bg
        self._card_light._refreshStyle()
        self._card_dark._refreshStyle()

    # ── Public: apply chosen theme to the main app ────────

    def applyChosenTheme(self):
        """在精靈關閉後，將主題套用到 qfluentwidgets 全局。"""
        if not _HAS_FLUENT:
            return
        if self._isDark:
            setTheme(Theme.DARK)
            qconfig.set(qconfig.themeMode, Theme.DARK, save=False)
        else:
            setTheme(Theme.LIGHT)
            qconfig.set(qconfig.themeMode, Theme.LIGHT, save=False)
