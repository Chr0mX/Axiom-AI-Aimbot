# convert_page.py
"""TensorRT engine conversion page — convert ONNX models to .engine caches."""

import glob
import os
import sys

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QFileDialog
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
                 workspace_mb: int,
                 python_exe: str, script_path: str, parent=None):
        super().__init__(parent)
        self._onnx_path = onnx_path
        self._cache_dir = cache_dir
        self._fp16 = fp16
        self._workspace_mb = workspace_mb
        self._python_exe = python_exe
        self._script_path = script_path

    def run(self) -> None:
        import subprocess
        model_stem = os.path.splitext(os.path.basename(self._onnx_path))[0]
        cmd = [
            self._python_exe, "-u", self._script_path,
            "--model", self._onnx_path,
            "--output", self._cache_dir,
            "--workspace", str(self._workspace_mb),
            "--method", "ort",
            "--engine-prefix", model_stem,
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
        self._converting_onnx_path = None
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
        model_name = os.path.basename(model_path or "")
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

    def selectModelForConversion(self, model_path: str) -> None:
        """Select `model_path` in the model combo, adding it if it isn't
        already listed. `model_path` may be relative (e.g. "Model/x.onnx",
        as model_page.py stores it) or absolute."""
        abs_path = model_path if os.path.isabs(model_path) else os.path.join(
            self._project_root, model_path)
        name = os.path.basename(abs_path)
        idx = self.modelCombo.findText(name)
        if idx >= 0:
            self.modelCombo.setCurrentIndex(idx)
        else:
            self.modelCombo.addItem(name, userData=abs_path)
            self.modelCombo.setCurrentIndex(self.modelCombo.count() - 1)

    def startConversionFor(self, model_path: str | None = None) -> None:
        """Public entry point for other pages: select `model_path` (if given)
        then start conversion, as if the user clicked Convert. Used by
        ModelPage's model/backend selectors to redirect here automatically
        when the selected model has no cached TensorRT engine yet."""
        if model_path:
            self.selectModelForConversion(model_path)
        self._onConvert()

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
        self.logView.clear()
        self.logView.append(f"→ Converting {os.path.basename(onnx_path)} "
                            f"(fp16={fp16}, workspace={workspace_mb} MiB)")
        self.progressBar.setVisible(True)
        self.convertBtn.setEnabled(False)
        self.convertBtn.setText(t("trt_converting", "Converting…"))
        # Lock model selection for the duration — _onConvertFinished applies
        # self._converting_onnx_path on success, so it must stay in sync with
        # whatever the worker is actually building.
        self.modelCombo.setEnabled(False)
        self.browseBtn.setEnabled(False)
        self._converting_onnx_path = onnx_path

        python_exe = os.path.join(self._project_root, "python", "python.exe")
        if not os.path.exists(python_exe):
            python_exe = sys.executable
        script_path = os.path.join(self._src_dir, "core", "convert_to_engine.py")

        self._worker = _ConvertWorker(
            onnx_path, self._cache_dir, fp16, workspace_mb,
            python_exe, script_path, parent=self)
        self._worker.logLine.connect(self._onLogLine)
        self._worker.finishedResult.connect(self._onConvertFinished)
        if self._config:
            self._config.inference_paused = True
        self._worker.start()

    def _onLogLine(self, line: str):
        self.logView.append(line)

    def _onConvertFinished(self, success: bool, message: str):
        self.progressBar.setVisible(False)
        self.convertBtn.setEnabled(True)
        self.convertBtn.setText(t("trt_convert", "Convert"))
        self.modelCombo.setEnabled(True)
        self.browseBtn.setEnabled(True)
        if self._config:
            self._config.inference_paused = False
        if success:
            self.logView.append(f"✓ Done. Engine cache written to: {message}")
            if self._config is not None:
                self._config.trt_fp16_enabled = self.fp16Card.isChecked()
                # Point the running app at the model we just built an engine
                # for, so the (now cache-hit, near-instant) hot-swap in
                # ai_loop.py picks it up on the next frame instead of the
                # user having to reselect it manually on the Model tab.
                # Store it the same way model_page.py does ("Model/x.onnx")
                # rather than the absolute path _resolveModelPath() returns,
                # so config.json/presets stay portable across machines.
                if self._converting_onnx_path:
                    converted = self._converting_onnx_path
                    if os.path.isabs(converted) and os.path.dirname(converted) == self._model_dir:
                        converted = os.path.join("Model", os.path.basename(converted))
                    self._config.model_path = converted
                try:
                    from core.config import save_config
                    save_config(self._config)
                except Exception:
                    pass
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
        # Re-sync every page (Model tab's combo in particular) to whatever
        # config.model_path actually is now — the just-converted model on
        # success, or unchanged (still the old, still-running model) on
        # failure, since this page's own combo was left on the attempted
        # model either way.
        if self._config is not None:
            try:
                win = self.window()
                if hasattr(win, '_refreshAllPages'):
                    win._refreshAllPages()
            except Exception:
                pass
        self._converting_onnx_path = None
        self._worker = None

    def retranslateUi(self):
        super().retranslateUi()
        self.convertGroup.titleLabel.setText(t("trt_convert_settings", "Engine Conversion"))
        self.modelCard.titleLabel.setText(t("trt_source_model", "Source ONNX Model"))
        self.modelCard.contentLabel.setText(t("trt_source_model_desc", "Select the .onnx model to compile into a TensorRT engine."))
        self.browseBtn.setText(t("trt_browse", "Browse"))
        self.fp16Card.titleLabel.setText(t("trt_fp16", "FP16 Precision"))
        self.fp16Card.contentLabel.setText(t("trt_fp16_desc", "Half precision — ~2× faster on RTX GPUs, negligible accuracy loss."))
        self.workspaceCard.titleLabel.setText(t("trt_workspace", "Builder Workspace (MiB)"))
        self.workspaceCard.contentLabel.setText(t("trt_workspace_desc", "GPU memory budget for the build. Increase for larger models."))
        self.outputCard.titleLabel.setText(t("trt_output", "Output Cache Directory"))
        self.logLabel.setText(t("trt_build_log", "Build Log"))
        self.logView.setPlaceholderText(
            t("trt_log_placeholder",
              "Conversion output will appear here. First build can take 1–5 minutes.")
        )
        self.hintLabel.setText(
            t("trt_convert_hint",
              "The engine is cached in trt_cache/. On the next app launch, TensorRT "
              "loads it in under a second instead of rebuilding.")
        )
        if not (self._worker and self._worker.isRunning()):
            self.convertBtn.setText(t("trt_convert", "Convert"))
