# capture_page.py
"""Capture Page — Screenshot Method, UVC, NDI, Preview settings"""

import os
import sys
import subprocess
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QMessageBox
from qfluentwidgets import (
    SettingCardGroup, SwitchSettingCard, FluentIcon,
    ComboBox, PushButton, SettingCard, BodyLabel,
)
from ..components.slider_spin_card import SliderSpinCard
from ..base_page import BasePage
from ..language_manager import t


class CapturePage(BasePage):
    """Capture Settings Page — Screenshot Method, UVC, NDI, Preview"""

    def __init__(self, parent=None):
        super().__init__("tab_capture", parent)
        self._config = None
        self._isLoadingConfig = False
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
        # === Capture Method ===
        self.captureGroup = SettingCardGroup(t("capture_method_group", "Capture"), self.scrollWidget)

        self.screenshotMethodCombo = ComboBox()
        self.screenshotMethodCombo.addItems(["mss", "dxcam", "uvc", "ndi"])
        self.screenshotMethodCombo.setMinimumWidth(150)
        self.screenshotMethodCard = SettingCard(
            FluentIcon.CAMERA,
            t("screenshot_method"),
            "",
            self.captureGroup
        )
        self.screenshotMethodCard.hBoxLayout.addWidget(self.screenshotMethodCombo, 0, Qt.AlignmentFlag.AlignRight)
        self.screenshotMethodCard.hBoxLayout.addSpacing(16)

        self.screenshotIntervalCard = SliderSpinCard(
            FluentIcon.CAMERA,
            t("screenshot_interval"),
            1, 100,
            suffix="ms",
            description="",
            parent=self.captureGroup
        )


        # === UVC Camera ===
        self.uvcGroup = SettingCardGroup("UVC Camera", self.scrollWidget)

        self.uvcDeviceCard = SliderSpinCard(
            FluentIcon.CAMERA,
            "UVC Device Index",
            0, 16,
            suffix="",
            description="",
            parent=self.uvcGroup
        )

        self.uvcResolutionCombo = ComboBox()
        self.uvcResolutionCombo.setMinimumWidth(180)
        self.uvcResolutionCard = SettingCard(
            FluentIcon.FULL_SCREEN,
            "UVC Resolution",
            "Auto-detect supported resolutions",
            self.uvcGroup
        )
        self.uvcResolutionCard.hBoxLayout.addWidget(self.uvcResolutionCombo, 0, Qt.AlignmentFlag.AlignRight)
        self.uvcResolutionCard.hBoxLayout.addSpacing(16)

        self.uvcRefreshResolutionBtn = PushButton(t("refresh"))
        self.uvcRefreshResolutionBtn.setFixedWidth(80)
        self.uvcRefreshResolutionCard = SettingCard(
            FluentIcon.SYNC,
            "Refresh UVC Resolution List",
            "",
            self.uvcGroup
        )
        self.uvcRefreshResolutionCard.hBoxLayout.addWidget(self.uvcRefreshResolutionBtn, 0, Qt.AlignmentFlag.AlignRight)
        self.uvcRefreshResolutionCard.hBoxLayout.addSpacing(16)

        self.uvcWidthCard = SliderSpinCard(
            FluentIcon.FULL_SCREEN,
            "UVC Width",
            320, 7680,
            suffix="px",
            description="",
            parent=self.uvcGroup
        )
        self.uvcHeightCard = SliderSpinCard(
            FluentIcon.FULL_SCREEN,
            "UVC Height",
            240, 4320,
            suffix="px",
            description="",
            parent=self.uvcGroup
        )
        self.uvcWidthCard.setVisible(False)
        self.uvcHeightCard.setVisible(False)

        self.uvcFpsCombo = ComboBox()
        self.uvcFpsCombo.addItems(["24", "30", "60", "90", "120", "144", "240"])
        self.uvcFpsCombo.setCurrentText("60")
        self.uvcFpsCombo.setMinimumWidth(120)
        self.uvcFpsCard = SettingCard(
            FluentIcon.SPEED_MEDIUM,
            "UVC FPS",
            "",
            self.uvcGroup
        )
        self.uvcFpsCard.hBoxLayout.addWidget(self.uvcFpsCombo, 0, Qt.AlignmentFlag.AlignRight)
        self.uvcFpsCard.hBoxLayout.addSpacing(16)

        self.uvcCaptureMethodCombo = ComboBox()
        self.uvcCaptureMethodCombo.addItems(["msmf", "dshow", "auto", "any"])
        self.uvcCaptureMethodCombo.setMinimumWidth(140)
        self.uvcCaptureMethodCard = SettingCard(
            FluentIcon.CAMERA,
            "UVC Capture Method",
            "msmf recommended for 1080p60 on Windows 10/11",
            self.uvcGroup
        )
        self.uvcCaptureMethodCard.hBoxLayout.addWidget(self.uvcCaptureMethodCombo, 0, Qt.AlignmentFlag.AlignRight)
        self.uvcCaptureMethodCard.hBoxLayout.addSpacing(16)

        self.uvcHwInfoLabel = BodyLabel("—")
        self.uvcQueryBtn = PushButton("Query Device")
        self.uvcQueryBtn.setFixedWidth(110)
        self.uvcHwInfoCard = SettingCard(
            FluentIcon.INFO,
            "Device Resolution & FPS",
            "Actual values reported by the driver",
            self.uvcGroup
        )
        self.uvcHwInfoCard.hBoxLayout.addWidget(self.uvcHwInfoLabel, 0, Qt.AlignmentFlag.AlignRight)
        self.uvcHwInfoCard.hBoxLayout.addWidget(self.uvcQueryBtn, 0, Qt.AlignmentFlag.AlignRight)
        self.uvcHwInfoCard.hBoxLayout.addSpacing(16)

        # === NDI ===
        self.ndiGroup = SettingCardGroup("NDI", self.scrollWidget)

        self.ndiSourceCombo = ComboBox()
        self.ndiSourceCombo.setMinimumWidth(360)
        self.ndiSourceCard = SettingCard(
            FluentIcon.CAMERA,
            "NDI Stream",
            "Select the NDI source to capture",
            self.ndiGroup
        )
        self.ndiSourceCard.hBoxLayout.addWidget(self.ndiSourceCombo, 0, Qt.AlignmentFlag.AlignRight)
        self.ndiSourceCard.hBoxLayout.addSpacing(16)

        self.ndiRefreshBtn = PushButton(t("refresh"))
        self.ndiRefreshBtn.setFixedWidth(80)
        self.ndiRefreshCard = SettingCard(
            FluentIcon.SYNC,
            "Refresh NDI Streams",
            "",
            self.ndiGroup
        )
        self.ndiRefreshCard.hBoxLayout.addWidget(self.ndiRefreshBtn, 0, Qt.AlignmentFlag.AlignRight)
        self.ndiRefreshCard.hBoxLayout.addSpacing(16)

        self.ndiBandwidthCombo = ComboBox()
        self.ndiBandwidthCombo.addItems(["Highest", "Lowest"])
        self.ndiBandwidthCombo.setMinimumWidth(120)
        self.ndiBandwidthCard = SettingCard(
            FluentIcon.SPEED_HIGH,
            "NDI Bandwidth",
            "Receive bandwidth for the NDI stream",
            self.ndiGroup
        )
        self.ndiBandwidthCard.hBoxLayout.addWidget(self.ndiBandwidthCombo, 0, Qt.AlignmentFlag.AlignRight)
        self.ndiBandwidthCard.hBoxLayout.addSpacing(16)

        self.ndiHwInfoLabel = BodyLabel("—")
        self.ndiRefreshInfoBtn = PushButton("Refresh Info")
        self.ndiRefreshInfoBtn.setFixedWidth(100)
        self.ndiHwInfoCard = SettingCard(
            FluentIcon.INFO,
            "Stream Resolution & FPS",
            "Actual values from the active NDI source",
            self.ndiGroup
        )
        self.ndiHwInfoCard.hBoxLayout.addWidget(self.ndiHwInfoLabel, 0, Qt.AlignmentFlag.AlignRight)
        self.ndiHwInfoCard.hBoxLayout.addWidget(self.ndiRefreshInfoBtn, 0, Qt.AlignmentFlag.AlignRight)
        self.ndiHwInfoCard.hBoxLayout.addSpacing(16)

        # === Preview ===
        self.previewGroup = SettingCardGroup(t("preview_group", "Preview"), self.scrollWidget)

        self.uvcPreviewCard = SwitchSettingCard(
            FluentIcon.VIEW,
            "Capture Preview Window",
            "",
            parent=self.previewGroup
        )

        self.previewCropCard = SwitchSettingCard(
            FluentIcon.ZOOM_IN,
            t("preview_crop_label"),
            t("preview_crop_desc"),
            parent=self.previewGroup
        )

        self.uvcPreviewScaleCombo = ComboBox()
        self.uvcPreviewScaleCombo.addItems(["scale_to_fit", "scale_to_canvas", "fit_to_screen"])
        self.uvcPreviewScaleCombo.setMinimumWidth(170)
        self.uvcPreviewScaleCard = SettingCard(
            FluentIcon.FULL_SCREEN,
            "Capture Preview Scale Mode",
            "",
            self.previewGroup
        )
        self.uvcPreviewScaleCard.hBoxLayout.addWidget(self.uvcPreviewScaleCombo, 0, Qt.AlignmentFlag.AlignRight)
        self.uvcPreviewScaleCard.hBoxLayout.addSpacing(16)

    # ──────────────────────────────────────────────
    # Layout
    # ──────────────────────────────────────────────

    def _initLayout(self):
        self.captureGroup.addSettingCard(self.screenshotMethodCard)
        self.captureGroup.addSettingCard(self.screenshotIntervalCard)
        self.addContent(self.captureGroup)

        self.uvcGroup.addSettingCard(self.uvcDeviceCard)
        self.uvcGroup.addSettingCard(self.uvcResolutionCard)
        self.uvcGroup.addSettingCard(self.uvcRefreshResolutionCard)
        self.uvcGroup.addSettingCard(self.uvcWidthCard)
        self.uvcGroup.addSettingCard(self.uvcHeightCard)
        self.uvcGroup.addSettingCard(self.uvcFpsCard)
        self.uvcGroup.addSettingCard(self.uvcCaptureMethodCard)
        self.uvcGroup.addSettingCard(self.uvcHwInfoCard)
        self.addContent(self.uvcGroup)
        self.uvcGroup.setVisible(False)

        self.ndiGroup.addSettingCard(self.ndiSourceCard)
        self.ndiGroup.addSettingCard(self.ndiRefreshCard)
        self.ndiGroup.addSettingCard(self.ndiBandwidthCard)
        self.ndiGroup.addSettingCard(self.ndiHwInfoCard)
        self.addContent(self.ndiGroup)
        self.ndiGroup.setVisible(False)

        self.previewGroup.addSettingCard(self.uvcPreviewCard)
        self.previewGroup.addSettingCard(self.previewCropCard)
        self.previewGroup.addSettingCard(self.uvcPreviewScaleCard)
        self.addContent(self.previewGroup)
        self.previewGroup.setVisible(False)

        self.scrollLayout.addStretch(1)

    # ──────────────────────────────────────────────
    # Signal connections
    # ──────────────────────────────────────────────

    def _connectSignals(self):
        self.screenshotMethodCombo.currentTextChanged.connect(self._onScreenshotMethodChanged)
        self.screenshotIntervalCard.valueChanged.connect(self._onScreenshotIntervalChanged)
        self.uvcDeviceCard.valueChanged.connect(self._onUvcDeviceChanged)
        self.uvcResolutionCombo.currentTextChanged.connect(self._onUvcResolutionChanged)
        self.uvcRefreshResolutionBtn.clicked.connect(self._refreshUvcResolutions)
        self.uvcFpsCombo.currentTextChanged.connect(self._onUvcFpsChanged)
        self.uvcCaptureMethodCombo.currentTextChanged.connect(self._onUvcCaptureMethodChanged)
        self.uvcQueryBtn.clicked.connect(self._queryUvcHwInfo)
        self.ndiRefreshInfoBtn.clicked.connect(self._refreshNdiHwInfo)
        self.uvcPreviewCard.checkedChanged.connect(self._onUvcPreviewChanged)
        self.previewCropCard.checkedChanged.connect(self._onPreviewCropChanged)
        self.uvcPreviewScaleCombo.currentTextChanged.connect(self._onUvcPreviewScaleModeChanged)
        self.ndiSourceCombo.currentTextChanged.connect(self._onNdiSourceChanged)
        self.ndiRefreshBtn.clicked.connect(self._refreshNdiSources)
        self.ndiBandwidthCombo.currentTextChanged.connect(self._onNdiBandwidthChanged)

    # ──────────────────────────────────────────────
    # Config load
    # ──────────────────────────────────────────────

    def _loadFromConfig(self):
        if not self._config:
            return
        self._isLoadingConfig = True
        try:
            screenshot_methods = ["mss", "dxcam", "uvc", "ndi"]
            screenshot_method = getattr(self._config, 'screenshot_method', 'mss')
            if screenshot_method in screenshot_methods:
                self.screenshotMethodCombo.setCurrentIndex(screenshot_methods.index(screenshot_method))

            screenshot_interval_ms = int(
                getattr(self._config, 'screenshot_interval',
                        getattr(self._config, 'detect_interval', 0.01)) * 1000
            )
            self.screenshotIntervalCard.setValue(screenshot_interval_ms)

            self.uvcDeviceCard.setValue(int(getattr(self._config, 'uvc_device_index', 0)))
            self.uvcCaptureMethodCombo.setCurrentText(str(getattr(self._config, 'uvc_capture_method', 'msmf')))
            resolution_text = str(getattr(self._config, 'uvc_resolution',
                f"{getattr(self._config, 'uvc_width', self._config.width)}x{getattr(self._config, 'uvc_height', self._config.height)}"))
            if screenshot_method == 'uvc':
                self._refreshUvcResolutions()
                idx = self.uvcResolutionCombo.findText(resolution_text)
                if idx < 0:
                    self.uvcResolutionCombo.addItem(resolution_text)
                    idx = self.uvcResolutionCombo.findText(resolution_text)
                if idx >= 0:
                    self.uvcResolutionCombo.setCurrentIndex(idx)
            else:
                self.uvcResolutionCombo.blockSignals(True)
                self.uvcResolutionCombo.clear()
                self.uvcResolutionCombo.addItem(resolution_text)
                self.uvcResolutionCombo.blockSignals(False)
            self.uvcFpsCombo.setCurrentText(str(int(getattr(self._config, 'uvc_fps', 60))))
            self.uvcPreviewCard.setChecked(bool(getattr(self._config, 'uvc_show_window', True)))
            self.previewCropCard.setChecked(bool(getattr(self._config, 'preview_crop_to_detection', False)))
            self.uvcPreviewScaleCombo.setCurrentText(str(getattr(self._config, 'uvc_preview_scale_mode', 'scale_to_fit')))

            ndi_source = str(getattr(self._config, 'ndi_source_name', '')).strip()
            if screenshot_method == 'ndi':
                self._refreshNdiSources()
                if ndi_source:
                    idx = self.ndiSourceCombo.findText(ndi_source)
                    if idx < 0:
                        self.ndiSourceCombo.addItem(ndi_source)
                        idx = self.ndiSourceCombo.findText(ndi_source)
                    if idx >= 0:
                        self.ndiSourceCombo.setCurrentIndex(idx)
            else:
                self.ndiSourceCombo.blockSignals(True)
                self.ndiSourceCombo.clear()
                if ndi_source:
                    self.ndiSourceCombo.addItem(ndi_source)
                self.ndiSourceCombo.blockSignals(False)
            ndi_bw = str(getattr(self._config, 'ndi_bandwidth', 'highest')).capitalize()
            self.ndiBandwidthCombo.setCurrentText(ndi_bw if ndi_bw in ("Highest", "Lowest") else "Highest")

            self._updateCaptureControlsVisibility(screenshot_method)
        finally:
            self._isLoadingConfig = False

    # ──────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────

    def _updateCaptureControlsVisibility(self, screenshot_method):
        is_uvc = (screenshot_method == "uvc")
        is_ndi = (screenshot_method == "ndi")
        self.uvcGroup.setVisible(is_uvc)
        self.ndiGroup.setVisible(is_ndi)
        self.previewGroup.setVisible(is_uvc or is_ndi)
        self._notifyInferenceFovFollow(screenshot_method)

    def _notifyInferenceFovFollow(self, method):
        try:
            win = self.window()
            if hasattr(win, 'inferenceInterface'):
                win.inferenceInterface._applyScreenshotMethodEffect(method)
        except Exception:
            pass

    def _refreshUvcResolutions(self):
        if not self._config:
            return
        try:
            from core.screen_capture import list_supported_uvc_resolutions
            resolutions = list_supported_uvc_resolutions(
                int(getattr(self._config, 'uvc_device_index', 0)),
                str(getattr(self._config, 'uvc_capture_method', 'msmf')),
            )
        except Exception:
            resolutions = []
        current_text = self.uvcResolutionCombo.currentText().strip()
        self.uvcResolutionCombo.blockSignals(True)
        self.uvcResolutionCombo.clear()
        if resolutions:
            for width, height in resolutions:
                self.uvcResolutionCombo.addItem(f"{width}x{height}")
        else:
            fallback = f"{int(getattr(self._config, 'uvc_width', 1920))}x{int(getattr(self._config, 'uvc_height', 1080))}"
            self.uvcResolutionCombo.addItem(fallback)
        if current_text:
            idx = self.uvcResolutionCombo.findText(current_text)
            if idx >= 0:
                self.uvcResolutionCombo.setCurrentIndex(idx)
        self.uvcResolutionCombo.blockSignals(False)

    def _refreshUvcFps(self):
        if not self._config:
            return
        try:
            from core.screen_capture import list_supported_uvc_fps
            w = int(getattr(self._config, 'uvc_width', 1920))
            h = int(getattr(self._config, 'uvc_height', 1080))
            fps_list = list_supported_uvc_fps(
                int(getattr(self._config, 'uvc_device_index', 0)),
                w, h,
                str(getattr(self._config, 'uvc_capture_method', 'msmf')),
            )
        except Exception:
            fps_list = [24, 30, 60, 90, 120, 144, 240]
        current_fps = self.uvcFpsCombo.currentText()
        self.uvcFpsCombo.blockSignals(True)
        self.uvcFpsCombo.clear()
        for fps in fps_list:
            self.uvcFpsCombo.addItem(str(fps))
        idx = self.uvcFpsCombo.findText(current_fps)
        if idx >= 0:
            self.uvcFpsCombo.setCurrentIndex(idx)
        self.uvcFpsCombo.blockSignals(False)

    def _refreshNdiSources(self):
        if not self._config:
            return
        try:
            from core.screen_capture import list_available_ndi_source_details
            source_details = list_available_ndi_source_details()
        except Exception:
            source_details = []
        current_name = self.ndiSourceCombo.currentData()
        if not isinstance(current_name, str):
            current_name = self.ndiSourceCombo.currentText().strip()
        configured = str(getattr(self._config, 'ndi_source_name', '')).strip()
        self.ndiSourceCombo.blockSignals(True)
        self.ndiSourceCombo.clear()
        known_names: set = set()
        for detail in source_details:
            name = str(detail.get('name', '')).strip()
            if not name:
                continue
            label = str(detail.get('label', '')).strip() or name
            self.ndiSourceCombo.addItem(label, name)
            known_names.add(name)
        if configured and configured not in known_names:
            self.ndiSourceCombo.addItem(f"{configured} (Unknown @ Unknown fps)", configured)
        fallback_name = configured or (current_name if isinstance(current_name, str) else '')
        if fallback_name:
            for i in range(self.ndiSourceCombo.count()):
                data = self.ndiSourceCombo.itemData(i)
                if isinstance(data, str) and data == fallback_name:
                    self.ndiSourceCombo.setCurrentIndex(i)
                    break
        self.ndiSourceCombo.blockSignals(False)

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

    # ──────────────────────────────────────────────
    # Callbacks
    # ──────────────────────────────────────────────

    def _onScreenshotMethodChanged(self, text):
        if self._config:
            self._config.screenshot_method = text
        if str(text).strip().lower() == 'uvc' and not self._isLoadingConfig:
            self._refreshUvcResolutions()
        if str(text).strip().lower() == "ndi" and not self._isLoadingConfig:
            has_ran = bool(getattr(self._config, "ndi_installer_ran_once", False))
            if not has_ran:
                if self._runLocalInstallerScript("install_cyndilib.py", "NDI"):
                    self._config.ndi_installer_ran_once = True
        self._updateCaptureControlsVisibility(text)
        main_window = self.window()
        if main_window and hasattr(main_window, 'updateVisualsVisibilityForScreenshotMethod'):
            main_window.updateVisualsVisibilityForScreenshotMethod(text)

    def _onScreenshotIntervalChanged(self, value):
        if self._config:
            self._config.screenshot_interval = value / 1000.0

    def _onUvcDeviceChanged(self, value):
        if self._config:
            self._config.uvc_device_index = int(value)
        self._refreshUvcResolutions()
        self._refreshUvcFps()
        QTimer.singleShot(400, self._queryUvcHwInfo)

    def _onUvcResolutionChanged(self, value):
        if self._config:
            text = str(value).strip().lower()
            if 'x' not in text:
                return
            width_str, height_str = text.split('x', 1)
            try:
                self._config.uvc_width = int(width_str)
                self._config.uvc_height = int(height_str)
                self._config.uvc_resolution = f"{self._config.uvc_width}x{self._config.uvc_height}"
            except ValueError:
                return
        self._refreshUvcFps()
        QTimer.singleShot(400, self._queryUvcHwInfo)

    def _onUvcFpsChanged(self, value):
        if self._config:
            try:
                self._config.uvc_fps = int(value)
            except (ValueError, TypeError):
                pass

    def _onUvcCaptureMethodChanged(self, text):
        if self._config:
            self._config.uvc_capture_method = str(text)
        self._refreshUvcResolutions()
        QTimer.singleShot(400, self._queryUvcHwInfo)

    def _onUvcPreviewChanged(self, checked):
        if self._config:
            self._config.uvc_show_window = bool(checked)

    def _onPreviewCropChanged(self, checked):
        if self._config:
            self._config.preview_crop_to_detection = bool(checked)

    def _onUvcPreviewScaleModeChanged(self, text):
        if self._config:
            self._config.uvc_preview_scale_mode = str(text)

    def _onNdiBandwidthChanged(self, text):
        if self._config:
            self._config.ndi_bandwidth = str(text).lower()

    def _onNdiSourceChanged(self, text):
        if not self._config:
            return
        source_name = self.ndiSourceCombo.currentData()
        if not isinstance(source_name, str) or not source_name.strip():
            source_name = str(text).strip()
        self._config.ndi_source_name = source_name.strip()

    def _queryUvcHwInfo(self):
        """Open a temporary VideoCapture to read the driver's actual resolution and FPS."""
        if not self._config:
            self.uvcHwInfoLabel.setText("—")
            return
        try:
            import cv2
            idx = int(getattr(self._config, 'uvc_device_index', 0))
            method_str = str(getattr(self._config, 'uvc_capture_method', 'msmf')).lower()
            backend_map = {'msmf': cv2.CAP_MSMF, 'dshow': cv2.CAP_DSHOW}
            backend = backend_map.get(method_str, cv2.CAP_ANY)
            # Apply requested settings so the driver reports accurate values
            w_req = int(getattr(self._config, 'uvc_width', 1920))
            h_req = int(getattr(self._config, 'uvc_height', 1080))
            fps_req = int(getattr(self._config, 'uvc_fps', 60))
            cap = cv2.VideoCapture(idx, backend)
            if not cap.isOpened():
                self.uvcHwInfoLabel.setText("—  (device not available)")
                return
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w_req)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h_req)
            cap.set(cv2.CAP_PROP_FPS, fps_req)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            fps_str = f"{fps:.1f}" if fps > 0 else "?"
            self.uvcHwInfoLabel.setText(f"{w} × {h} @ {fps_str} fps")
        except Exception as exc:
            self.uvcHwInfoLabel.setText(f"—  ({exc})")

    def _refreshNdiHwInfo(self):
        """Read NDI stream resolution/FPS from config fields written by NDICapture."""
        if not self._config:
            self.ndiHwInfoLabel.setText("—")
            return
        w = int(getattr(self._config, 'ndi_width', 0) or 0)
        h = int(getattr(self._config, 'ndi_height', 0) or 0)
        fps = float(getattr(self._config, 'source_nominal_fps', 0.0) or 0.0)
        if w > 0 and h > 0:
            fps_str = f"{fps:.1f}" if fps > 0 else "?"
            self.ndiHwInfoLabel.setText(f"{w} × {h} @ {fps_str} fps")
        else:
            self.ndiHwInfoLabel.setText("—  (connect source to see info)")

    # ──────────────────────────────────────────────
    # Retranslate
    # ──────────────────────────────────────────────

    def retranslateUi(self):
        super().retranslateUi()
        self.captureGroup.titleLabel.setText(t("capture_method_group", "Capture"))
        self.screenshotMethodCard.titleLabel.setText(t("screenshot_method"))
        self.screenshotIntervalCard.titleLabel.setText(t("screenshot_interval"))
        self.uvcDeviceCard.titleLabel.setText("UVC Device Index")
        self.uvcResolutionCard.titleLabel.setText("UVC Resolution")
        self.uvcRefreshResolutionCard.titleLabel.setText("Refresh UVC Resolution List")
        self.uvcRefreshResolutionBtn.setText(t("refresh"))
        self.uvcFpsCard.titleLabel.setText("UVC FPS")  # type: ignore[attr-defined]
        self.uvcCaptureMethodCard.titleLabel.setText("UVC Capture Method")
        self.uvcHwInfoCard.titleLabel.setText("Device Resolution & FPS")
        self.uvcQueryBtn.setText("Query Device")
        self.ndiSourceCard.titleLabel.setText("NDI Stream")
        self.ndiRefreshCard.titleLabel.setText("Refresh NDI Streams")
        self.ndiRefreshBtn.setText(t("refresh"))
        self.ndiBandwidthCard.titleLabel.setText("NDI Bandwidth")
        self.ndiHwInfoCard.titleLabel.setText("Stream Resolution & FPS")
        self.ndiRefreshInfoBtn.setText("Refresh Info")
        self.previewGroup.titleLabel.setText(t("preview_group", "Preview"))
        self.uvcPreviewCard.titleLabel.setText("Capture Preview Window")
        self.previewCropCard.titleLabel.setText(t("preview_crop_label"))
        self.previewCropCard.contentLabel.setText(t("preview_crop_desc"))
        self.uvcPreviewScaleCard.titleLabel.setText("Capture Preview Scale Mode")
