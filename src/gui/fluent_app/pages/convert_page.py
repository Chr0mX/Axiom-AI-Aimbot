# convert_page.py
"""TensorRT engine conversion page — convert ONNX models to .engine caches."""

import glob
import os
import sys

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QFileDialog
from qfluentwidgets import (
    SettingCardGroup, SettingCard, SwitchSettingCard, ComboBox,
    FluentIcon, PrimaryPushButton, PushButton, BodyLabel, CaptionLabel,
    TextEdit, IndeterminateProgressBar, InfoBar, InfoBarPosition,
)

from ..base_page import BasePage
from ..language_manager import t


class _ConvertWorker(QThread):
    """Background worker that builds a TensorRT engine without freezing the UI.

    Runs convert_to_engine.py as a subprocess so that C-level TRT/ORT output
    is captured and streamed into the UI log in real time.
    """

    logLine = pyqtSignal(str)
    finishedResult = pyqtSignal(bool, str)  # (success, output_path_or_message)

    def __init__(self, onnx_path: str, cache_dir: str, fp16: bool,
                 workspace_mb: int, method: str,
                 python_exe: str, script_path: str, parent=None):
        super().__init__(parent)
        self._onnx_path = onnx_path
        self._cache_dir = cache_dir
        self._fp16 = fp16
        self._workspace_mb = workspace_mb
        self._method = method
        self._python_exe = python_exe
        self._script_path = script_path

    def run(self) -> None:
        import subprocess
        cmd = [
            self._python_exe, "-u", self._script_path,
            "--model", self._onnx_path,
            "--output", self._cache_dir,
            "--workspace", str(self._workspace_mb),
            "--method", self._method,
        ]
        if not self._fp16:
            cmd.append("--no-fp16")
        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in proc.stdout:
                self.logLine.emit(line.rstrip())
            proc.wait()
            if proc.returncode == 0:
                self.finishedResult.emit(True, self._cache_dir)
            else:
                self.finishedResult.emit(
                    False, f"Process exited with code {proc.returncode} — see log above.")
        except Exception as exc:
            self.finishedResult.emit(False, f"{type(exc).__name__}: {exc}")


class ConvertPage(BasePage):
    """ONNX → TensorRT engine conversion."""

    def __init__(self, parent=None):
        super().__init__("tab_convert", parent)
        self._config = None
        self._worker = None
        # repo root: .../src/gui/fluent_app/pages/convert_page.py → up 4 → src → up 1 → root
        _src_dir = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))))
        self._src_dir = _src_dir
        self._project_root = os.path.dirname(_src_dir)
        self._model_dir = os.path.join(self._project_root, "Model")
        self._cache_dir = os.path.join(self._project_root, "trt_cache")
        self._initWidgets()
        self._initLayout()
        self._connectSignals()
        self._refreshModelList()

    def setConfig(self, config):
        self._config = config
        if config is None:
            return
        self.fp16Card.setChecked(bool(getattr(config, 'trt_fp16_enabled', True)))
        model_path = getattr(config, 'model_path', '')
        model_name = os.path.basename(model_path)
        idx = self.modelCombo.findText(model_name)
        if idx >= 0:
            self.modelCombo.setCurrentIndex(idx)
        elif model_path and os.path.isfile(model_path):
            self.modelCombo.addItem(model_name, userData=model_path)
            self.modelCombo.setCurrentIndex(self.modelCombo.count() - 1)

    def _initWidgets(self):
        # === Conversion settings ===
        self.convertGroup = SettingCardGroup(
            t("trt_convert_settings", "Engine Conversion"), self.scrollWidget)

        # Model selector card (combo + browse)
        self.modelCombo = ComboBox()
        self.modelCombo.setMinimumWidth(240)
        self.browseBtn = PushButton(t("trt_browse", "Browse"))
        self.browseBtn.setIcon(FluentIcon.FOLDER)
        self.modelCard = SettingCard(
            FluentIcon.DOCUMENT,
            t("trt_source_model", "Source ONNX Model"),
            t("trt_source_model_desc", "Select the .onnx model to compile into a TensorRT engine."),
            self.convertGroup,
        )
        self.modelCard.hBoxLayout.addWidget(self.modelCombo, 0, Qt.AlignmentFlag.AlignRight)
        self.modelCard.hBoxLayout.addSpacing(8)
        self.modelCard.hBoxLayout.addWidget(self.browseBtn, 0, Qt.AlignmentFlag.AlignRight)
        self.modelCard.hBoxLayout.addSpacing(16)

        # FP16 toggle
        self.fp16Card = SwitchSettingCard(
            FluentIcon.SPEED_HIGH,
            t("trt_fp16", "FP16 Precision"),
            t("trt_fp16_desc", "Half precision — ~2× faster on RTX GPUs, negligible accuracy loss."),
            parent=self.convertGroup,
        )
        self.fp16Card.setChecked(True)

        # Workspace budget
        self.workspaceCombo = ComboBox()
        self.workspaceCombo.addItems(["1024", "2048", "4096", "8192"])
        self.workspaceCombo.setCurrentText("2048")
        self.workspaceCard = SettingCard(
            FluentIcon.SPEED_MEDIUM,
            t("trt_workspace", "Builder Workspace (MiB)"),
            t("trt_workspace_desc", "GPU memory budget for the build. Increase for larger models."),
            self.convertGroup,
        )
        self.workspaceCard.hBoxLayout.addWidget(self.workspaceCombo, 0, Qt.AlignmentFlag.AlignRight)
        self.workspaceCard.hBoxLayout.addSpacing(16)

        # Build method
        self.methodCombo = ComboBox()
        self.methodCombo.addItems(["ort", "trt"])
        self.methodCombo.setCurrentText("ort")
        self.methodCard = SettingCard(
            FluentIcon.DEVELOPER_TOOLS,
            t("trt_method", "Build Method"),
            t("trt_method_desc", "'ort' matches what the app uses at runtime (recommended). 'trt' uses the TensorRT API directly."),
            self.convertGroup,
        )
        self.methodCard.hBoxLayout.addWidget(self.methodCombo, 0, Qt.AlignmentFlag.AlignRight)
        self.methodCard.hBoxLayout.addSpacing(16)

        # Output dir card with Convert button
        self.convertBtn = PrimaryPushButton(t("trt_convert", "Convert"))
        self.convertBtn.setIcon(FluentIcon.SYNC)
        self.outputCard = SettingCard(
            FluentIcon.FOLDER,
            t("trt_output", "Output Cache Directory"),
            self._cache_dir,
            self.convertGroup,
        )
        self.outputCard.hBoxLayout.addWidget(self.convertBtn, 0, Qt.AlignmentFlag.AlignRight)
        self.outputCard.hBoxLayout.addSpacing(16)

        # === Build log ===
        self.progressBar = IndeterminateProgressBar(self)
        self.progressBar.setVisible(False)

        self.logLabel = BodyLabel(t("trt_build_log", "Build Log"))
        self.logLabel.setStyleSheet("font-weight: bold; margin-top: 8px;")

        self.logView = TextEdit()
        self.logView.setReadOnly(True)
        self.logView.setMinimumHeight(220)
        self.logView.setPlaceholderText(
            t("trt_log_placeholder",
              "Conversion output will appear here. First build can take 1–5 minutes.")
        )

        self.hintLabel = CaptionLabel(
            t("trt_convert_hint",
              "The engine is cached in trt_cache/. On the next app launch, TensorRT "
              "loads it in under a second instead of rebuilding.")
        )
        self.hintLabel.setWordWrap(True)

    def _initLayout(self):
        self.convertGroup.addSettingCard(self.modelCard)
        self.convertGroup.addSettingCard(self.fp16Card)
        self.convertGroup.addSettingCard(self.workspaceCard)
        self.convertGroup.addSettingCard(self.methodCard)
        self.convertGroup.addSettingCard(self.outputCard)
        self.addContent(self.convertGroup)

        logWidget = QWidget()
        logWidget.setStyleSheet("background: transparent;")
        logLayout = QVBoxLayout(logWidget)
        logLayout.setContentsMargins(16, 8, 16, 8)
        logLayout.setSpacing(8)
        logLayout.addWidget(self.progressBar)
        logLayout.addWidget(self.logLabel)
        logLayout.addWidget(self.logView)
        logLayout.addWidget(self.hintLabel)
        self.scrollLayout.addWidget(logWidget)

        self.scrollLayout.addStretch(1)

    def _connectSignals(self):
        self.browseBtn.clicked.connect(self._onBrowse)
        self.convertBtn.clicked.connect(self._onConvert)

    def _refreshModelList(self):
        self.modelCombo.clear()
        if os.path.isdir(self._model_dir):
            for m in sorted(glob.glob(os.path.join(self._model_dir, "*.onnx"))):
                self.modelCombo.addItem(os.path.basename(m), userData=m)

    def _resolveModelPath(self) -> str | None:
        data = self.modelCombo.currentData()
        if data:
            return data
        text = self.modelCombo.currentText().strip()
        if not text:
            return None
        if os.path.isabs(text):
            return text if os.path.isfile(text) else None
        candidate = os.path.join(self._model_dir, text)
        return candidate if os.path.isfile(candidate) else None

    # === Callbacks ===
    def _onBrowse(self):
        start_dir = self._model_dir if os.path.isdir(self._model_dir) else self._project_root
        path, _ = QFileDialog.getOpenFileName(
            self, t("trt_select_onnx", "Select ONNX Model"),
            start_dir, "ONNX Models (*.onnx)")
        if path:
            idx = self.modelCombo.findText(os.path.basename(path))
            if idx >= 0:
                self.modelCombo.setCurrentIndex(idx)
            else:
                self.modelCombo.addItem(os.path.basename(path), userData=path)
                self.modelCombo.setCurrentIndex(self.modelCombo.count() - 1)

    def _onConvert(self):
        if self._worker is not None and self._worker.isRunning():
            return

        onnx_path = self._resolveModelPath()
        if not onnx_path:
            InfoBar.error(
                t("trt_no_model", "No model selected"),
                t("trt_no_model_desc", "Pick a valid .onnx file first."),
                duration=4000, isClosable=True, position=InfoBarPosition.TOP, parent=self,
            )
            return

        os.makedirs(self._cache_dir, exist_ok=True)
        fp16 = self.fp16Card.isChecked()
        try:
            workspace_mb = int(self.workspaceCombo.currentText())
        except ValueError:
            workspace_mb = 2048
        method = self.methodCombo.currentText()

        self.logView.clear()
        self.logView.append(f"→ Converting {os.path.basename(onnx_path)} "
                            f"(fp16={fp16}, workspace={workspace_mb} MiB, method={method})")
        self.progressBar.setVisible(True)
        self.convertBtn.setEnabled(False)
        self.convertBtn.setText(t("trt_converting", "Converting…"))

        python_exe = os.path.join(self._project_root, "python", "python.exe")
        if not os.path.exists(python_exe):
            python_exe = sys.executable
        script_path = os.path.join(self._src_dir, "core", "convert_to_engine.py")

        self._worker = _ConvertWorker(
            onnx_path, self._cache_dir, fp16, workspace_mb, method,
            python_exe, script_path, parent=self)
        self._worker.logLine.connect(self._onLogLine)
        self._worker.finishedResult.connect(self._onConvertFinished)
        self._worker.start()

    def _onLogLine(self, line: str):
        self.logView.append(line)

    def _onConvertFinished(self, success: bool, message: str):
        self.progressBar.setVisible(False)
        self.convertBtn.setEnabled(True)
        self.convertBtn.setText(t("trt_convert", "Convert"))
        if success:
            self.logView.append(f"✓ Done. Engine cache written to: {message}")
            InfoBar.success(
                t("trt_convert_ok", "Conversion complete"),
                message, duration=6000, isClosable=True,
                position=InfoBarPosition.TOP, parent=self,
            )
        else:
            self.logView.append(f"✗ {message}")
            InfoBar.error(
                t("trt_convert_fail", "Conversion failed"),
                message, duration=8000, isClosable=True,
                position=InfoBarPosition.TOP, parent=self,
            )
        self._worker = None

    def retranslateUi(self):
        super().retranslateUi()
        self.convertGroup.titleLabel.setText(t("trt_convert_settings", "Engine Conversion"))
        self.modelCard.titleLabel.setText(t("trt_source_model", "Source ONNX Model"))
        self.browseBtn.setText(t("trt_browse", "Browse"))
        self.fp16Card.titleLabel.setText(t("trt_fp16", "FP16 Precision"))
        self.workspaceCard.titleLabel.setText(t("trt_workspace", "Builder Workspace (MiB)"))
        self.methodCard.titleLabel.setText(t("trt_method", "Build Method"))
        self.outputCard.titleLabel.setText(t("trt_output", "Output Cache Directory"))
        self.logLabel.setText(t("trt_build_log", "Build Log"))
        if not (self._worker and self._worker.isRunning()):
            self.convertBtn.setText(t("trt_convert", "Convert"))
