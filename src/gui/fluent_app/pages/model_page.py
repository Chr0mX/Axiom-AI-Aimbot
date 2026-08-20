# model_page.py
"""Model Page — model selection, inference backend, and model inspector."""

import glob
import json
import os
import subprocess
import sys
from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication, QHBoxLayout, QLabel, QMessageBox, QTextEdit, QVBoxLayout,
)
from qfluentwidgets import (
    CardWidget,
    SettingCardGroup,
    FluentIcon,
    ComboBox, PrimaryPushButton, PushButton, SettingCard,
    InfoBar, InfoBarPosition, SearchLineEdit,
)

from ..base_page import BasePage
from ..language_manager import t


# ──────────────────────────────────────────────
# model_info.json helpers
# ──────────────────────────────────────────────

def _notes_path() -> str:
    src = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    return os.path.join(os.path.dirname(src), "model_info.json")


def _load_notes() -> dict:
    p = _notes_path()
    if os.path.exists(p):
        try:
            with open(p, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_notes(data: dict) -> None:
    with open(_notes_path(), 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _default_template(model_name: str) -> str:
    return (
        f"### Recommend settings for {model_name}\n"
        "**Game Settings**\n"
        "Enter settings here\n\n"
        "**AI Settings**\n"
        "Enter settings here"
    )


# ──────────────────────────────────────────────
# Model notes card widget
# ──────────────────────────────────────────────

class _ModelNotesCard(CardWidget):
    """Per-model notes card with view/edit mode, saved to model_info.json."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_model = ""
        self._notes_data: dict = {}
        self._raw_text = ""

        self._titleLabel = QLabel("Model Notes")
        self._titleLabel.setStyleSheet("font-size: 14px; font-weight: 600;")

        self._editBtn = PushButton("Edit")
        self._editBtn.setFixedWidth(80)
        self._editBtn.clicked.connect(self._onEditSaveClicked)

        headerRow = QHBoxLayout()
        headerRow.addWidget(self._titleLabel)
        headerRow.addStretch(1)
        headerRow.addWidget(self._editBtn)

        self._textEdit = QTextEdit()
        self._textEdit.setReadOnly(True)
        self._textEdit.setFixedHeight(160)
        self._textEdit.setAcceptRichText(False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(10)
        layout.addLayout(headerRow)
        layout.addWidget(self._textEdit)

    def setModel(self, model_name: str) -> None:
        self._current_model = model_name
        self._notes_data = _load_notes()
        text = self._notes_data.get(model_name) if model_name else ""
        if not text:
            text = _default_template(model_name) if model_name else ""
        self._raw_text = text
        self._editBtn.setText("Edit")
        self._showRendered()

    def _showRendered(self) -> None:
        """View mode: render the raw markdown source as formatted text."""
        self._textEdit.setReadOnly(True)
        self._textEdit.setMarkdown(self._raw_text)

    def _showEditable(self) -> None:
        """Edit mode: show the raw markdown source for editing."""
        self._textEdit.setReadOnly(False)
        self._textEdit.setPlainText(self._raw_text)
        self._textEdit.setFocus()

    def _onEditSaveClicked(self) -> None:
        if self._textEdit.isReadOnly():
            self._editBtn.setText("Save")
            self._showEditable()
        else:
            self._raw_text = self._textEdit.toPlainText()
            if self._current_model:
                self._notes_data[self._current_model] = self._raw_text
                _save_notes(self._notes_data)
            self._editBtn.setText("Edit")
            self._showRendered()


class _ModelInspectWorker(QThread):
    """Background worker that inspects a model file and emits the result as a string."""

    resultReady = pyqtSignal(str)

    def __init__(self, inspect_path: str, parent=None):
        super().__init__(parent)
        self._inspect_path = inspect_path

    def run(self) -> None:
        try:
            from model_detect import inspect_model
            info = inspect_model(self._inspect_path)
            parts = []
            if info.get("format"):
                parts.append(info["format"])
            parts.append(f"Input: {info['input_size']}")
            if info.get("num_classes"):
                parts.append(f"Classes: {info['num_classes']}")
            if info.get("precision"):
                parts.append(f"Precision: {info['precision']}")
            if info.get("file_size"):
                parts.append(info["file_size"])
            text = "  •  ".join(parts)
        except BaseException as exc:
            text = str(exc)[:120]
        self.resultReady.emit(text)


class ModelPage(BasePage):
    """Model configuration page: model file, inference backend, and live model info."""

    def __init__(self, parent=None):
        super().__init__("tab_model", parent)
        self._config = None
        self._isLoadingConfig = False
        self._trt_installer_launched = False
        self._inspect_worker = None
        self._all_model_files = []  # master list; modelSearchEdit filters a view of this
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

        # Search box — a plain filter over modelCombo's item list rather than
        # an editable combo, so there's no risk of a stray Enter press adding
        # a phantom "model" that doesn't correspond to a real .onnx file.
        self.modelSearchEdit = SearchLineEdit()
        self.modelSearchEdit.setPlaceholderText(t("model_search_placeholder", "Search models…"))
        self.modelSearchEdit.setMinimumWidth(220)
        self.modelSearchCard = SettingCard(
            FluentIcon.SEARCH,
            t("model_search", "Search Models"),
            t("model_search_desc", "Filter the model list below by filename"),
            self.modelGroup
        )
        self.modelSearchCard.hBoxLayout.addWidget(self.modelSearchEdit, 0, Qt.AlignmentFlag.AlignRight)
        self.modelSearchCard.hBoxLayout.addSpacing(16)

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

        # Live model info — displayed as the card subtitle (contentLabel)
        self.modelInfoCard = SettingCard(
            FluentIcon.INFO,
            t("model_info", "Model Info"),
            t("model_inspecting", "Inspecting…"),
            self.modelGroup,
        )
        self.modelInfoCard.contentLabel.setWordWrap(True)

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

        self.modelNotesCard = _ModelNotesCard(self.scrollWidget)

        # === Model HUD Settings ===
        self.hudModelGroup = SettingCardGroup("Model HUD Settings", self.scrollWidget)

        self.hudGameCombo = ComboBox()
        self.hudGameCombo.setMinimumWidth(200)
        self.hudGameCard = SettingCard(
            FluentIcon.GAME,
            "Game Profile",
            "HUD region coords loaded from game.json",
            self.hudModelGroup,
        )
        self.hudGameCard.hBoxLayout.addWidget(self.hudGameCombo, 0, Qt.AlignmentFlag.AlignRight)
        self.hudGameCard.hBoxLayout.addSpacing(16)

        self.hudModelCombo = ComboBox()
        self.hudModelCombo.setMinimumWidth(200)
        self.hudModelCard = SettingCard(
            FluentIcon.ROBOT,
            "HUD Model",
            "YOLO11n .onnx model from Model_Hud/ for V2 weapon detection",
            self.hudModelGroup,
        )
        self.hudModelCard.hBoxLayout.addWidget(self.hudModelCombo, 0, Qt.AlignmentFlag.AlignRight)
        self.hudModelCard.hBoxLayout.addSpacing(16)

    # ──────────────────────────────────────────────
    # Layout
    # ──────────────────────────────────────────────

    def _initLayout(self):
        self.modelGroup.addSettingCard(self.modelSearchCard)
        self.modelGroup.addSettingCard(self.modelCard)
        self.modelGroup.addSettingCard(self.modelInfoCard)
        self.modelGroup.addSettingCard(self.inferenceBackendCard)
        self.modelGroup.addSettingCard(self.openModelFolderCard)
        self.addContent(self.modelGroup)
        self.scrollLayout.addWidget(self.modelNotesCard)

        self.hudModelGroup.addSettingCard(self.hudGameCard)
        self.hudModelGroup.addSettingCard(self.hudModelCard)
        self.addContent(self.hudModelGroup)

        self.scrollLayout.addStretch(1)

    # ──────────────────────────────────────────────
    # Signal connections
    # ──────────────────────────────────────────────

    def _connectSignals(self):
        self.modelSearchEdit.textChanged.connect(self._onModelSearchChanged)
        self.modelCombo.currentTextChanged.connect(self._onModelChanged)
        self.inferenceBackendCombo.currentTextChanged.connect(self._onInferenceBackendChanged)
        self.openModelFolderBtn.clicked.connect(self._openModelFolder)
        self.hudGameCombo.currentTextChanged.connect(self._onHudGameChanged)
        self.hudModelCombo.currentTextChanged.connect(self._onHudModelChanged)

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

        # Load game profile combo (game.json)
        self.hudGameCombo.blockSignals(True)
        game_data = self._loadGameJson()
        self.hudGameCombo.clear()
        for game_name in game_data:
            self.hudGameCombo.addItem(game_name)
        saved_game = getattr(self._config, 'hud_game', 'Apex Legends')
        game_idx = -1
        for i in range(self.hudGameCombo.count()):
            if self.hudGameCombo.itemText(i) == saved_game:
                game_idx = i
                break
        if game_idx >= 0:
            self.hudGameCombo.setCurrentIndex(game_idx)
        elif self.hudGameCombo.count() > 0:
            self.hudGameCombo.setCurrentIndex(0)
            first_game = self.hudGameCombo.itemText(0)
            if self._config:
                self._config.hud_game = first_game
                self._config.hud_roi_coords = game_data.get(first_game, "")
        self.hudGameCombo.blockSignals(False)

        # Load HUD model combo
        self.hudModelCombo.blockSignals(True)
        self._refreshHudModelList()
        hud_name = os.path.basename(getattr(self._config, 'hud_model_path', '') or "")
        hud_idx = -1
        for i in range(self.hudModelCombo.count()):
            if self.hudModelCombo.itemText(i).lower() == hud_name.lower():
                hud_idx = i
                break
        if hud_idx >= 0:
            self.hudModelCombo.setCurrentIndex(hud_idx)
        elif self.hudModelCombo.count() > 0:
            self.hudModelCombo.setCurrentIndex(0)
            if self._config:
                self._config.hud_model_path = os.path.join("Model_Hud", self.hudModelCombo.itemText(0))
        self.hudModelCombo.blockSignals(False)

        # Kick off model inspection and load notes after config load
        self._updateModelInfo(self._config.model_path)
        self.modelNotesCard.setModel(os.path.basename(self._config.model_path or ""))

    # ──────────────────────────────────────────────
    # Model inspector
    # ──────────────────────────────────────────────

    def _updateModelInfo(self, model_path: str) -> None:
        """Inspect the selected model in a QThread and update the info card via signal."""
        if not model_path:
            self.modelInfoCard.contentLabel.setText(t("model_no_model", "No model selected."))
            return

        # Resolve paths on the main thread so the worker only gets an absolute path
        if not os.path.isabs(model_path):
            _pages = os.path.dirname(os.path.abspath(__file__))
            _src   = os.path.dirname(os.path.dirname(os.path.dirname(_pages)))
            _root  = os.path.dirname(_src)
            full_onnx = os.path.join(_root, model_path)
        else:
            _src   = os.path.dirname(os.path.dirname(os.path.dirname(
                         os.path.dirname(os.path.abspath(__file__)))))
            _root  = os.path.dirname(_src)
            full_onnx = model_path

        # When TRT is active, prefer the matching cached engine file
        inspect_path = full_onnx
        provider = getattr(self._config, 'current_provider', '') if self._config else ''
        if provider == 'TensorrtExecutionProvider':
            trt_cache = os.path.join(_root, "trt_cache")
            if os.path.isdir(trt_cache):
                model_stem = os.path.splitext(os.path.basename(full_onnx))[0]
                engine_files = glob.glob(os.path.join(trt_cache, f"{model_stem}*.engine"))
                if engine_files:
                    inspect_path = sorted(engine_files)[-1]

        self.modelInfoCard.contentLabel.setText(t("model_inspecting", "Inspecting…"))

        # Stop any in-flight worker safely — do NOT use deleteLater; let Python GC own lifetime
        if self._inspect_worker is not None:
            if self._inspect_worker.isRunning():
                self._inspect_worker.quit()
                self._inspect_worker.wait(200)
            self._inspect_worker = None

        self._inspect_worker = _ModelInspectWorker(inspect_path)
        self._inspect_worker.resultReady.connect(self.modelInfoCard.contentLabel.setText)
        self._inspect_worker.finished.connect(self._onInspectWorkerDone)
        self._inspect_worker.start()

    def _onInspectWorkerDone(self) -> None:
        """Clear the worker reference once finished so isRunning() is never called on a dead object."""
        self._inspect_worker = None

    # ──────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────

    def _refreshModelList(self):
        """Rebuild the master model file list and show all of them
        (unfiltered) in modelCombo — callers that need to re-select the
        active/default model (_loadFromConfig()) do that immediately after
        this runs. _onModelSearchChanged() is the only thing that narrows
        the combo to a subset of self._all_model_files afterward."""
        src_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        project_root = os.path.dirname(src_dir)
        model_dir = os.path.join(project_root, "Model")
        self._all_model_files = []
        if os.path.exists(model_dir):
            self._all_model_files = sorted(
                os.path.basename(m) for m in glob.glob(os.path.join(model_dir, "*.onnx")))
        self.modelCombo.clear()
        for name in self._all_model_files:
            self.modelCombo.addItem(name)

    def _onModelSearchChanged(self, query: str):
        """Live-filter modelCombo to entries matching `query` (case-
        insensitive substring) as the user types — but never hide the
        currently active model. Search only needs to help find something
        else to switch *to*; it must never make the combo appear to show
        the wrong "currently active" model just because the search text
        doesn't happen to match it."""
        if not self._all_model_files:
            return
        query = (query or '').strip().lower()
        current_name = os.path.basename(getattr(self._config, 'model_path', '') or '') if self._config else ''

        self.modelCombo.blockSignals(True)
        self.modelCombo.clear()
        matched_current = not current_name
        for name in self._all_model_files:
            if not query or query in name.lower():
                self.modelCombo.addItem(name)
                if name.lower() == current_name.lower():
                    matched_current = True
        if not matched_current and current_name:
            self.modelCombo.addItem(current_name)
        idx = self.modelCombo.findText(current_name)
        if idx >= 0:
            self.modelCombo.setCurrentIndex(idx)
        self.modelCombo.blockSignals(False)

    def _refreshHudModelList(self):
        self.hudModelCombo.clear()
        src_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        project_root = os.path.dirname(src_dir)
        model_dir = os.path.join(project_root, "Model_Hud")
        if os.path.exists(model_dir):
            models = glob.glob(os.path.join(model_dir, "*.onnx"))
            for m in sorted(models):
                self.hudModelCombo.addItem(os.path.basename(m))

    def _loadGameJson(self) -> dict:
        """Read game.json from project root. Returns {} on missing/invalid file."""
        src_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        project_root = os.path.dirname(src_dir)
        game_json_path = os.path.join(project_root, "game.json")
        try:
            with open(game_json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return {k: v for k, v in data.items() if isinstance(k, str)}
        except Exception:
            return {}

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
        if not self._config or not text:
            return
        new_path = os.path.join("Model", text)
        if self._redirectToConvertIfNeeded(new_path):
            # Engine missing under the active TensorRT backend — conversion
            # was just kicked off on the Convert tab instead of applying this
            # model_path here, which would otherwise hang the running aim
            # loop's hot-swap doing a blocking 1-5 min build. The Convert
            # tab sets model_path itself once the build finishes.
            return
        self._config.model_path = new_path
        self._updateModelInfo(self._config.model_path)
        self.modelNotesCard.setModel(os.path.basename(text))

    def _needsTrtConversion(self, model_path: str) -> bool:
        if not self._config or not model_path:
            return False
        try:
            from core.session_utils import needs_trt_build
            return needs_trt_build(self._config, model_path)
        except Exception:
            return False

    def _redirectToConvertIfNeeded(self, model_path: str) -> bool:
        """If `model_path` would need a fresh (1-5 min) TensorRT build under
        the currently active backend, send the user to the Convert tab and
        start that build there (background thread, progress bar) instead of
        ever letting config.model_path reach a combination that would hang
        the running aim loop's hot-swap. Returns True if it redirected."""
        if not self._needsTrtConversion(model_path):
            return False
        InfoBar.warning(
            t("model_trt_engine_missing", "TensorRT engine not built yet"),
            t("model_trt_engine_missing_desc",
              "Redirecting to the Convert tab to build it — this can take 1-5 minutes. "
              "The model will switch over automatically once it's done."),
            duration=6000, isClosable=True, position=InfoBarPosition.TOP, parent=self,
        )
        try:
            win = self.window()
            convert_page = getattr(win, 'convertInterface', None)
            if convert_page is not None:
                if hasattr(win, 'switchTo'):
                    win.switchTo(convert_page)
                if hasattr(convert_page, 'startConversionFor'):
                    convert_page.startConversionFor(model_path)
        except Exception:
            pass
        return True

    def _onHudGameChanged(self, text):
        if not self._config or not text or self._isLoadingConfig:
            return
        self._config.hud_game = text
        game_data = self._loadGameJson()
        coords = game_data.get(text, "")
        self._config.hud_roi_coords = coords
        self.hudGameCard.contentLabel.setText(coords or "No coords configured")

    def _onHudModelChanged(self, text):
        if self._config and text:
            self._config.hud_model_path = os.path.join("Model_Hud", text)

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

        # Backend switched to (or stayed on) something that may now prefer
        # TensorRT as its effective provider (selected_backend itself, or
        # "auto" resolving to it) without going through the DirectML restart
        # above — this is the live hot-swap path, so redirect to convert the
        # current model instead of letting ai_loop.py's hot-swap hang on it.
        if not self._isLoadingConfig and prev_backend != selected_backend:
            self._redirectToConvertIfNeeded(self._config.model_path)
        self._updateInferenceBackendSubtitle()

    # ──────────────────────────────────────────────
    # Retranslate
    # ──────────────────────────────────────────────

    def retranslateUi(self):
        super().retranslateUi()
        self.modelGroup.titleLabel.setText(t("model_settings"))
        self.modelSearchCard.titleLabel.setText(t("model_search", "Search Models"))
        self.modelSearchCard.contentLabel.setText(t("model_search_desc", "Filter the model list below by filename"))
        self.modelSearchEdit.setPlaceholderText(t("model_search_placeholder", "Search models…"))
        self.modelCard.titleLabel.setText(t("model"))
        self.modelInfoCard.titleLabel.setText(t("model_info", "Model Info"))
        self.inferenceBackendCard.titleLabel.setText(t("inference_backend"))
        self._updateInferenceBackendSubtitle()
        self.openModelFolderCard.titleLabel.setText(t("open_model_folder"))
        self.openModelFolderBtn.setText(t("open_model_folder"))
        self.hudModelGroup.titleLabel.setText("Model HUD Settings")
        self.hudGameCard.titleLabel.setText("Game Profile")
        self.hudModelCard.titleLabel.setText("HUD Model")
