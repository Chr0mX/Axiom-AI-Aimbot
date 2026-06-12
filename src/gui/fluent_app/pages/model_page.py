# model_page.py
"""Model Page — model selection, inference backend, and model inspector."""

import glob
import os
import sys
import subprocess
import threading
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QApplication, QMessageBox
from qfluentwidgets import (
    SettingCardGroup,
    FluentIcon,
    BodyLabel, ComboBox, PrimaryPushButton, SettingCard,
    InfoBar, InfoBarPosition
)

from ..base_page import BasePage
from ..language_manager import t


class ModelPage(BasePage):
    """Model configuration page: model file, inference backend, and live model info."""

    def __init__(self, parent=None):
        super().__init__("tab_model", parent)
        self._config = None
        self._isLoadingConfig = False
        self._trt_installer_launched = False
        self._initWidgets()
        self._initLayout()
        self._connectSignals()

    def setConfig(self, config):
        self._config = config
        self._loadFromConfig()

    # ──────────────────────────────────────────────
    # Widget initialisation
    # ──────────────────────────────────────────────

    def _initWidgets(self):
        self.modelGroup = SettingCardGroup(t("model_settings"), self.scrollWidget)

        self.modelCombo = ComboBox()
        self.modelCombo.setMinimumWidth(200)
        self.modelCard = SettingCard(
            FluentIcon.ROBOT,
            t("model"),
            "",
            self.modelGroup
        )
        self.modelCard.hBoxLayout.addWidget(self.modelCombo, 0, Qt.AlignmentFlag.AlignRight)
        self.modelCard.hBoxLayout.addSpacing(16)

        # Live model info (input size, classes, precision, file size)
        self.modelInfoLabel = BodyLabel("", self.modelGroup)
        self.modelInfoLabel.setWordWrap(True)
        self.modelInfoCard = SettingCard(
            FluentIcon.INFO,
            t("model_info", "Model Info"),
            "",
            self.modelGroup
        )
        self.modelInfoCard.hBoxLayout.addWidget(self.modelInfoLabel, 1, Qt.AlignmentFlag.AlignRight)
        self.modelInfoCard.hBoxLayout.addSpacing(16)

        self.inferenceBackendCombo = ComboBox()
        self.inferenceBackendCombo.addItems(["Auto", "TensorRT", "DirectML", "CPU"])
        self.inferenceBackendCombo.setMinimumWidth(150)
        self.inferenceBackendCard = SettingCard(
            FluentIcon.COMMAND_PROMPT,
            t("inference_backend"),
            "",
            self.modelGroup
        )
        self.inferenceBackendCard.hBoxLayout.addWidget(self.inferenceBackendCombo, 0, Qt.AlignmentFlag.AlignRight)
        self.inferenceBackendCard.hBoxLayout.addSpacing(16)

        self.openModelFolderBtn = PrimaryPushButton(t("open_model_folder"))
        self.openModelFolderCard = SettingCard(
            FluentIcon.FOLDER,
            t("open_model_folder"),
            "",
            self.modelGroup
        )
        self.openModelFolderCard.hBoxLayout.addWidget(self.openModelFolderBtn, 0, Qt.AlignmentFlag.AlignRight)
        self.openModelFolderCard.hBoxLayout.addSpacing(16)

    # ──────────────────────────────────────────────
    # Layout
    # ──────────────────────────────────────────────

    def _initLayout(self):
        self.modelGroup.addSettingCard(self.modelCard)
        self.modelGroup.addSettingCard(self.modelInfoCard)
        self.modelGroup.addSettingCard(self.inferenceBackendCard)
        self.modelGroup.addSettingCard(self.openModelFolderCard)
        self.addContent(self.modelGroup)
        self.scrollLayout.addStretch(1)

    # ──────────────────────────────────────────────
    # Signal connections
    # ──────────────────────────────────────────────

    def _connectSignals(self):
        self.modelCombo.currentTextChanged.connect(self._onModelChanged)
        self.inferenceBackendCombo.currentTextChanged.connect(self._onInferenceBackendChanged)
        self.openModelFolderBtn.clicked.connect(self._openModelFolder)

    # ──────────────────────────────────────────────
    # Config load
    # ──────────────────────────────────────────────

    def _loadFromConfig(self):
        if not self._config:
            return
        self._isLoadingConfig = True
        try:
            self.modelCombo.blockSignals(True)
            self._refreshModelList()
            model_name = os.path.basename(self._config.model_path or "")
            idx = -1
            for i in range(self.modelCombo.count()):
                if self.modelCombo.itemText(i).lower() == model_name.lower():
                    idx = i
                    break
            if idx >= 0:
                self.modelCombo.setCurrentIndex(idx)
            elif self.modelCombo.count() > 0:
                default_name = "ApexLegendsOrbeet_15k.onnx"
                default_idx = -1
                for i in range(self.modelCombo.count()):
                    if self.modelCombo.itemText(i).lower() == default_name.lower():
                        default_idx = i
                        break
                pick = default_idx if default_idx >= 0 else 0
                self.modelCombo.setCurrentIndex(pick)
                if self._config:
                    self._config.model_path = "Model/" + self.modelCombo.itemText(pick)
            self.modelCombo.blockSignals(False)

            backend_map = {
                "auto": "Auto",
                "tensorrt": "TensorRT",
                "cuda": "TensorRT",
                "directml": "DirectML",
                "cpu": "CPU",
            }
            self.inferenceBackendCombo.blockSignals(True)
            backend_text = backend_map.get(
                getattr(self._config, "inference_backend", "auto").lower(), "Auto"
            )
            self.inferenceBackendCombo.setCurrentText(backend_text)
            self.inferenceBackendCombo.blockSignals(False)
            self._updateInferenceBackendSubtitle()
        finally:
            self._isLoadingConfig = False

        # Kick off model inspection after load
        self._updateModelInfo(self._config.model_path)

    # ──────────────────────────────────────────────
    # Model inspector
    # ──────────────────────────────────────────────

    def _updateModelInfo(self, model_path: str) -> None:
        """Inspect the selected model in a background thread and update the info card."""
        if not model_path:
            self.modelInfoLabel.setText(t("model_no_model", "No model selected."))
            return
        self.modelInfoLabel.setText(t("model_inspecting", "Inspecting…"))

        config = self._config

        def _worker():
            try:
                if not os.path.isabs(model_path):
                    src = os.path.dirname(os.path.abspath(__file__))
                    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(src))))
                    full = os.path.join(root, model_path)
                else:
                    full = model_path

                model_stem = os.path.splitext(os.path.basename(full))[0]

                # When TRT is the active provider, look up the cached engine instead
                provider = getattr(config, 'current_provider', '') if config else ''
                if provider == 'TensorrtExecutionProvider':
                    src_dir = os.path.dirname(os.path.abspath(__file__))
                    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(src_dir))))
                    trt_cache = os.path.join(project_root, "trt_cache")
                    engine_files = []
                    if os.path.isdir(trt_cache):
                        import glob as _glob
                        engine_files = _glob.glob(os.path.join(trt_cache, f"{model_stem}*.engine"))
                    if engine_files:
                        engine_file = sorted(engine_files)[-1]
                        engine_name = os.path.basename(engine_file)
                        size_mb = os.path.getsize(engine_file) / (1024 * 1024)
                        precision = "FP16" if "_fp16" in engine_name.lower() else "FP32"
                        text = f"TensorRT Engine  •  {precision}  •  {size_mb:.1f} MiB  •  {engine_name}"
                        QTimer.singleShot(0, lambda t=text: self.modelInfoLabel.setText(t))
                        return

                from model_detect import inspect_model
                info = inspect_model(full)
                parts = [f"Input: {info['input_size']}"]
                if info.get("num_classes"):
                    parts.append(f"Classes: {info['num_classes']}")
                if info.get("precision"):
                    parts.append(f"Precision: {info['precision']}")
                if info.get("file_size"):
                    parts.append(info["file_size"])
                text = "  •  ".join(parts)
            except Exception as exc:
                text = str(exc)[:120]
            QTimer.singleShot(0, lambda t=text: self.modelInfoLabel.setText(t))

        threading.Thread(target=_worker, daemon=True).start()

    # ──────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────

    def _refreshModelList(self):
        self.modelCombo.clear()
        src_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        project_root = os.path.dirname(src_dir)
        model_dir = os.path.join(project_root, "Model")
        if os.path.exists(model_dir):
            models = glob.glob(os.path.join(model_dir, "*.onnx"))
            for m in models:
                self.modelCombo.addItem(os.path.basename(m))

    def _openModelFolder(self):
        src_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        project_root = os.path.dirname(src_dir)
        model_dir = os.path.join(project_root, "Model")
        if os.path.exists(model_dir):
            os.startfile(model_dir)

    def _updateInferenceBackendSubtitle(self):
        provider = getattr(self._config, "current_provider", "Unknown") if self._config else "Unknown"
        self.inferenceBackendCard.contentLabel.setText(
            f"{t('inference_backend_desc')} ({t('inference_backend_current')}: {provider})"
        )

    def _getEmbeddedPythonExe(self) -> str:
        src_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        python_exe = os.path.join(src_dir, "python", "python.exe")
        if os.path.exists(python_exe):
            return python_exe
        return sys.executable

    def _runLocalInstallerScript(self, script_name: str, feature_name: str, capture_output: bool = True) -> bool:
        src_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        script_path = os.path.join(src_dir, script_name)
        if not os.path.exists(script_path):
            QMessageBox.warning(self, f"{feature_name} install failed", f"Missing installer script:\n{script_path}")
            return False
        python_exe = self._getEmbeddedPythonExe()
        install_cmd = [python_exe, script_path]
        try:
            if capture_output:
                result = subprocess.run(install_cmd, check=True, text=True, capture_output=True)
                if result.stdout:
                    print(f"[Dependency][{feature_name}][stdout]\n{result.stdout}")
            else:
                subprocess.run(install_cmd, check=True, text=True)
            return True
        except subprocess.CalledProcessError as exc:
            parts = []
            if getattr(exc, 'stderr', None):
                parts.append(exc.stderr.strip())
            if getattr(exc, 'stdout', None):
                parts.append(exc.stdout.strip())
            error_text = "\n".join(parts) if parts else str(exc)
            QMessageBox.warning(
                self, f"{feature_name} install failed",
                f"Failed command: {' '.join(install_cmd)}\n\n{error_text}\n\nPlease run the installer script manually and try again."
            )
            return False

    def _ensureTrtInstalled(self) -> None:
        self._trt_installer_launched = False
        try:
            import onnxruntime as _ort
            if "TensorrtExecutionProvider" in _ort.get_available_providers():
                return
        except Exception:
            pass
        localappdata = os.environ.get("LOCALAPPDATA", "")
        if localappdata:
            trt_libs = os.path.join(localappdata, "AxiomAI", "site-packages", "tensorrt_libs")
            if os.path.isdir(trt_libs):
                for name in os.listdir(trt_libs):
                    if name.lower().startswith("nvinfer") and name.lower().endswith(".dll"):
                        return
        src_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        project_root = os.path.dirname(src_dir)
        bat_path = os.path.join(project_root, "Install TensorRT.bat")
        if not os.path.exists(bat_path):
            InfoBar.warning("TensorRT installer not found", f"Expected: {bat_path}",
                            duration=6000, isClosable=True, position=InfoBarPosition.TOP, parent=self)
            return
        subprocess.Popen([bat_path], shell=True)
        self._trt_installer_launched = True

    def _startRestartCountdown(self) -> None:
        self._restartCountdown = 5
        self._restartCountdownBar = InfoBar.info(
            t("restart_required", "Restart Required"),
            f"Restarting in {self._restartCountdown}s…",
            duration=-1, isClosable=True, position=InfoBarPosition.TOP, parent=self,
        )
        self._restartTimer = QTimer(self)
        self._restartTimer.timeout.connect(self._onRestartTick)
        self._restartTimer.start(1000)

    def _onRestartTick(self) -> None:
        from qfluentwidgets import BodyLabel as _BL
        self._restartCountdown -= 1
        if self._restartCountdown <= 0:
            self._restartTimer.stop()
            bar = getattr(self, "_restartCountdownBar", None)
            if bar:
                bar.close()
            self._restartApp()
        else:
            bar = getattr(self, "_restartCountdownBar", None)
            if bar:
                for lbl in bar.findChildren(_BL):
                    lbl.setText(f"Restarting in {self._restartCountdown}s…")
                    break

    def _restartApp(self) -> None:
        from core.config import save_config
        if self._config:
            save_config(self._config)
        subprocess.Popen([sys.executable] + sys.argv)
        QApplication.instance().quit()

    # ──────────────────────────────────────────────
    # Callbacks
    # ──────────────────────────────────────────────

    def _onModelChanged(self, text):
        if self._config and text:
            self._config.model_path = os.path.join("Model", text)
            self._updateModelInfo(self._config.model_path)

    def _onInferenceBackendChanged(self, text):
        if not self._config:
            return
        backend_map = {"Auto": "auto", "TensorRT": "tensorrt", "DirectML": "directml", "CPU": "cpu"}
        prev_backend = getattr(self._config, "inference_backend", "auto")
        selected_backend = backend_map.get(text, "auto")
        if prev_backend != selected_backend:
            self._config.inference_backend = selected_backend

        # Auto-enable CUDA IO Binding when TensorRT is selected (config only; widget syncs on next load)
        if selected_backend == "tensorrt" and not self._isLoadingConfig:
            self._config.cuda_io_binding_enabled = True

        if not self._isLoadingConfig and (selected_backend == "directml" or prev_backend == "directml"):
            if selected_backend == "tensorrt":
                self._ensureTrtInstalled()
                if getattr(self, "_trt_installer_launched", False):
                    InfoBar.info("TensorRT Installer Launched",
                                 "Restart the app after installation completes.",
                                 duration=6000, isClosable=True,
                                 position=InfoBarPosition.TOP, parent=self)
                    self._updateInferenceBackendSubtitle()
                    return
            self._startRestartCountdown()
            return
        self._updateInferenceBackendSubtitle()

    # ──────────────────────────────────────────────
    # Retranslate
    # ──────────────────────────────────────────────

    def retranslateUi(self):
        super().retranslateUi()
        self.modelGroup.titleLabel.setText(t("model_settings"))
        self.modelCard.titleLabel.setText(t("model"))
        self.modelInfoCard.titleLabel.setText(t("model_info", "Model Info"))
        self.inferenceBackendCard.titleLabel.setText(t("inference_backend"))
        self._updateInferenceBackendSubtitle()
        self.openModelFolderCard.titleLabel.setText(t("open_model_folder"))
        self.openModelFolderBtn.setText(t("open_model_folder"))
