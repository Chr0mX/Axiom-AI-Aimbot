# capture_page.py
"""Capture Page — Screenshot Method, UVC, NDI, UDP, Preview settings"""

import os
import socket
import sys
import subprocess
import time
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QMessageBox
from qfluentwidgets import (
    SettingCardGroup, SwitchSettingCard, FluentIcon,
    ComboBox, PushButton, SettingCard, BodyLabel, SegmentedWidget,
)
from ..components.slider_spin_card import SliderSpinCard
from ..base_page import BasePage
from ..language_manager import t


def _get_local_ips() -> list[str]:
    """Return non-loopback IPv4 addresses for this machine."""
    ips: list[str] = []
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None):
            if info[0] == socket.AF_INET:
                ip = info[4][0]
                if ip and not ip.startswith('127.'):
                    if ip not in ips:
                        ips.append(ip)
    except Exception:
        pass
    if not ips:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            if ip and not ip.startswith('127.'):
                ips.append(ip)
        except Exception:
            pass
    return ips


class CapturePage(BasePage):
    """Capture Settings Page — Screenshot Method, UVC, NDI, Preview"""

    def __init__(self, parent=None):
        super().__init__("tab_capture", parent)
        self._config = None
        self._isLoadingConfig = False
        self._last_probe_time: float = 0.0
        self._initWidgets()
        self._initLayout()
        self._connectSignals()

    def setConfig(self, config):
        self._config = config
        self._loadFromConfig()

    def showEvent(self, event):
        super().showEvent(event)
        if (self._config
                and str(getattr(self._config, 'screenshot_method', '')).lower() == 'uvc'
                and (time.time() - self._last_probe_time) > 8):
            self._refreshUvcResolutions()
            self._refreshUvcFps()
            self._last_probe_time = time.time()

    # ──────────────────────────────────────────────
    # Widget initialisation
    # ──────────────────────────────────────────────

    def _initWidgets(self):
        # === Capture Method ===
        self.captureGroup = SettingCardGroup(t("capture_method_group", "Capture"), self.scrollWidget)

        self.screenshotMethodCombo = ComboBox()
        self.screenshotMethodCombo.addItems(["mss", "dxcam", "uvc", "ndi", "udp"])
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

        # === UDP Stream ===
        self.udpGroup = SettingCardGroup("UDP Stream", self.scrollWidget)

        local_ips = _get_local_ips()
        system_ip_text = ", ".join(local_ips) if local_ips else "—"
        self.udpSystemIpCard = SettingCard(
            FluentIcon.WIFI,
            "System IP Address",
            f"Stream to: {system_ip_text}",
            self.udpGroup
        )

        self.udpBindIpCombo = ComboBox()
        bind_ip_options = ["0.0.0.0"] + local_ips
        self.udpBindIpCombo.addItems(bind_ip_options)
        self.udpBindIpCombo.setMinimumWidth(160)
        self.udpBindIpCard = SettingCard(
            FluentIcon.GLOBE,
            "Bind IP",
            "Listen on a specific interface, or 0.0.0.0 for all",
            self.udpGroup
        )
        self.udpBindIpCard.hBoxLayout.addWidget(self.udpBindIpCombo, 0, Qt.AlignmentFlag.AlignRight)
        self.udpBindIpCard.hBoxLayout.addSpacing(16)

        self.udpPortCard = SliderSpinCard(
            FluentIcon.CONNECT,
            "UDP Port",
            1, 65535,
            suffix="",
            description="",
            parent=self.udpGroup
        )

        self.udpRefreshBtn = PushButton(t("refresh"))
        self.udpRefreshBtn.setFixedWidth(80)
        self.udpRefreshCard = SettingCard(
            FluentIcon.SYNC,
            "Restart Receiver",
            "Stop and re-bind the UDP socket",
            self.udpGroup
        )
        self.udpRefreshCard.hBoxLayout.addWidget(self.udpRefreshBtn, 0, Qt.AlignmentFlag.AlignRight)
        self.udpRefreshCard.hBoxLayout.addSpacing(16)

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

        self.uvcAlwaysOnTopCard = SwitchSettingCard(
            FluentIcon.PIN,
            "Always On Top",
            "",
            parent=self.previewGroup
        )

        self.previewFpsCapSegment = SegmentedWidget()
        self.previewFpsCapSegment.addItem(routeKey='uncapped', text="None")
        self.previewFpsCapSegment.addItem(routeKey='30',       text="30 FPS")
        self.previewFpsCapSegment.addItem(routeKey='60',       text="60 FPS")
        self.previewFpsCapSegment.setCurrentItem('uncapped')
        self.previewFpsCapCard = SettingCard(
            FluentIcon.SPEED_HIGH,
            "Preview FPS Cap",
            "",
            self.previewGroup
        )
        self.previewFpsCapCard.hBoxLayout.addWidget(self.previewFpsCapSegment, 0, Qt.AlignmentFlag.AlignRight)
        self.previewFpsCapCard.hBoxLayout.addSpacing(16)

        # === Inferred Text (OCR) ===
        self.ocrGroup = SettingCardGroup(t("ocr_inferred_text", "Inferred Text"), self.scrollWidget)

        self.ocrScanBtn = PushButton("Scan Full Screen")
        self.ocrScanBtn.setFixedWidth(140)
        self.ocrScanCard = SettingCard(
            FluentIcon.SEARCH,
            "Full Screen Scan",
            "Run OCR on the full 1920×1080 frame once to find where text lives",
            self.ocrGroup
        )
        self.ocrScanCard.hBoxLayout.addWidget(self.ocrScanBtn, 0, Qt.AlignmentFlag.AlignRight)
        self.ocrScanCard.hBoxLayout.addSpacing(16)

        self.ocrResultLabel = BodyLabel("—")
        self.ocrResultLabel.setWordWrap(True)
        self.ocrResultCard = SettingCard(
            FluentIcon.DOCUMENT,
            t("ocr_result_title", "OCR Result"),
            "",
            self.ocrGroup
        )
        self.ocrResultCard.hBoxLayout.addWidget(self.ocrResultLabel, 1, Qt.AlignmentFlag.AlignRight)
        self.ocrResultCard.hBoxLayout.addSpacing(16)

        self._ocrRefreshTimer = QTimer(self)
        self._ocrRefreshTimer.setInterval(500)
        self._ocrRefreshTimer.timeout.connect(self._refreshOcrDisplay)
        self._ocrRefreshTimer.start()

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

        self.udpGroup.addSettingCard(self.udpSystemIpCard)
        self.udpGroup.addSettingCard(self.udpBindIpCard)
        self.udpGroup.addSettingCard(self.udpPortCard)
        self.udpGroup.addSettingCard(self.udpRefreshCard)
        self.addContent(self.udpGroup)
        self.udpGroup.setVisible(False)

        self.previewGroup.addSettingCard(self.uvcPreviewCard)
        self.previewGroup.addSettingCard(self.previewCropCard)
        self.previewGroup.addSettingCard(self.uvcPreviewScaleCard)
        self.previewGroup.addSettingCard(self.uvcAlwaysOnTopCard)
        self.previewGroup.addSettingCard(self.previewFpsCapCard)
        self.addContent(self.previewGroup)
        self.previewGroup.setVisible(False)

        self.ocrGroup.addSettingCard(self.ocrScanCard)
        self.ocrGroup.addSettingCard(self.ocrResultCard)
        self.addContent(self.ocrGroup)

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
        self.uvcAlwaysOnTopCard.checkedChanged.connect(self._onAlwaysOnTopChanged)
        self.previewFpsCapSegment.currentItemChanged.connect(self._onPreviewFpsCapChanged)
        self.ndiSourceCombo.currentTextChanged.connect(self._onNdiSourceChanged)
        self.ndiRefreshBtn.clicked.connect(self._refreshNdiSources)
        self.ndiBandwidthCombo.currentTextChanged.connect(self._onNdiBandwidthChanged)
        self.udpBindIpCombo.currentTextChanged.connect(self._onUdpBindIpChanged)
        self.udpPortCard.valueChanged.connect(self._onUdpPortChanged)
        self.udpRefreshBtn.clicked.connect(self._onUdpRefreshClicked)
        self.ocrScanBtn.clicked.connect(self._onOcrScanClicked)

    # ──────────────────────────────────────────────
    # Config load
    # ──────────────────────────────────────────────

    def _loadFromConfig(self):
        if not self._config:
            return
        self._isLoadingConfig = True
        try:
            screenshot_methods = ["mss", "dxcam", "uvc", "ndi", "udp"]
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
            resolution_text = (
                f"{getattr(self._config, 'uvc_width', self._config.width)}"
                f"x{getattr(self._config, 'uvc_height', self._config.height)}")
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
            self.uvcAlwaysOnTopCard.setChecked(bool(getattr(self._config, 'uvc_always_on_top', True)))
            _cap_key = {0: 'uncapped', 30: '30', 60: '60'}.get(
                getattr(self._config, 'preview_fps_cap', 0), 'uncapped'
            )
            self.previewFpsCapSegment.setCurrentItem(_cap_key)

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

            udp_bind_ip = str(getattr(self._config, 'udp_bind_ip', '0.0.0.0'))
            idx = self.udpBindIpCombo.findText(udp_bind_ip)
            if idx >= 0:
                self.udpBindIpCombo.setCurrentIndex(idx)
            self.udpPortCard.setValue(int(getattr(self._config, 'udp_bind_port', 5600)))

            self._updateCaptureControlsVisibility(screenshot_method)
        finally:
            self._isLoadingConfig = False

    # ──────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────

    def _updateCaptureControlsVisibility(self, screenshot_method):
        is_uvc = (screenshot_method == "uvc")
        is_ndi = (screenshot_method == "ndi")
        is_udp = (screenshot_method == "udp")
        self.uvcGroup.setVisible(is_uvc)
        self.ndiGroup.setVisible(is_ndi)
        self.udpGroup.setVisible(is_udp)
        self.previewGroup.setVisible(is_uvc or is_ndi or is_udp)
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
        device = int(getattr(self._config, 'uvc_device_index', 0))
        method = str(getattr(self._config, 'uvc_capture_method', 'msmf'))
        print(f"[Capture][UVC] Refreshing supported resolutions (device={device}, method={method})...")
        try:
            from core.screen_capture import list_supported_uvc_resolutions
            resolutions = list_supported_uvc_resolutions(device, method)
        except Exception as exc:
            print(f"[Capture][UVC] Resolution probe failed: {exc}")
            resolutions = []
        print(f"[Capture][UVC] Found {len(resolutions)} supported resolution(s): {resolutions}")
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
        w = int(getattr(self._config, 'uvc_width', 1920))
        h = int(getattr(self._config, 'uvc_height', 1080))
        device = int(getattr(self._config, 'uvc_device_index', 0))
        method = str(getattr(self._config, 'uvc_capture_method', 'msmf'))
        print(f"[Capture][UVC] Refreshing supported FPS (device={device}, {w}x{h}, method={method})...")
        try:
            from core.screen_capture import list_supported_uvc_fps
            fps_list = list_supported_uvc_fps(device, w, h, method)
        except Exception as exc:
            print(f"[Capture][UVC] FPS probe failed: {exc}")
            fps_list = [24, 30, 60, 90, 120, 144, 240]
        print(f"[Capture][UVC] Supported FPS: {fps_list}")
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
        print("[Capture][NDI] Refreshing available NDI sources...")
        try:
            from core.screen_capture import list_available_ndi_source_details
            source_details = list_available_ndi_source_details()
        except Exception as exc:
            print(f"[Capture][NDI] Source discovery failed: {exc}")
            source_details = []
        print(f"[Capture][NDI] Found {len(source_details)} source(s): "
              f"{[d.get('name', '') for d in source_details]}")
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

        # After repopulating, grab whichever source is now selected and
        # force a reconnect so the backend switches immediately (~0.5 s).
        selected_name = self.ndiSourceCombo.currentData()
        if not isinstance(selected_name, str) or not selected_name.strip():
            selected_name = self.ndiSourceCombo.currentText().strip()
        if selected_name:
            self._config.ndi_source_name = selected_name
            self._config.ndi_force_reconnect = True
            print(f"[Capture][NDI] Reconnecting to '{selected_name}'...")

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
        method = str(getattr(self._config, 'screenshot_method', '?')) if self._config else '?'
        print(f"[Capture][Preview] Preview window {'ENABLED' if checked else 'DISABLED'} "
              f"(method={method}); applies on next capture re-init (~0.5s).")
        win = self.window()
        if hasattr(win, 'updatePreviewPanelVisibility'):
            win.updatePreviewPanelVisibility()

    def _onPreviewCropChanged(self, checked):
        if self._config:
            self._config.preview_crop_to_detection = bool(checked)
        print(f"[Capture][Preview] Crop-to-detection {'ENABLED' if checked else 'DISABLED'}; "
              f"preview window resizes live.")

    def _onUvcPreviewScaleModeChanged(self, text):
        if self._config:
            self._config.uvc_preview_scale_mode = str(text)
        print(f"[Capture][Preview] Scale mode set to '{text}'.")

    def _onAlwaysOnTopChanged(self, checked):
        if self._config:
            self._config.uvc_always_on_top = bool(checked)

    def _onPreviewFpsCapChanged(self, routeKey: str):
        cap = {'uncapped': 0, '30': 30, '60': 60}.get(routeKey, 0)
        if self._config:
            self._config.preview_fps_cap = cap
        print(f"[Capture][Preview] FPS cap set to {cap or 'uncapped'}.")
        win = self.window()
        if hasattr(win, 'previewPanel'):
            win.previewPanel.applyFpsCap()

    def _onNdiBandwidthChanged(self, text):
        if self._config:
            self._config.ndi_bandwidth = str(text).lower()

    def _onNdiSourceChanged(self, text):
        if not self._config or self._isLoadingConfig:
            return
        source_name = self.ndiSourceCombo.currentData()
        if not isinstance(source_name, str) or not source_name.strip():
            return
        source_name = source_name.strip()
        if not source_name:
            return
        old_name = str(getattr(self._config, 'ndi_source_name', '')).strip()
        self._config.ndi_source_name = source_name
        if source_name != old_name:
            self._config.ndi_force_reconnect = True
            print(f"[Capture][NDI] Source changed to '{source_name}' — reconnecting...")

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

    def _onUdpBindIpChanged(self, text):
        if self._isLoadingConfig or not self._config:
            return
        self._config.udp_bind_ip = str(text)

    def _onUdpPortChanged(self, value):
        if self._isLoadingConfig or not self._config:
            return
        self._config.udp_bind_port = int(value)

    def _onUdpRefreshClicked(self):
        if self._config:
            self._config.udp_force_restart = True
            print('[Capture][UDP] Restart requested — receiver will reinitialize within ~0.5s.')

    def _onOcrScanClicked(self):
        from core.ocr_inference import trigger_full_scan
        self.ocrResultLabel.setText("Scanning...")
        trigger_full_scan()

    def _refreshOcrDisplay(self):
        from core.ocr_inference import get_ocr_results
        if not (self._config and getattr(self._config, 'ocr_enabled', False)):
            if self.ocrResultLabel.text() not in ("—", "Scanning..."):
                pass  # keep last scan result visible even when OCR is toggled off
            return
        lines = get_ocr_results()
        if lines:
            self.ocrResultLabel.setText("\n".join(lines))
        elif self.ocrResultLabel.text() == "Scanning...":
            pass  # wait for result to come back
        else:
            self.ocrResultLabel.setText("—")

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
        self.uvcAlwaysOnTopCard.titleLabel.setText("Always On Top")
        self.previewFpsCapCard.titleLabel.setText("Preview FPS Cap")
        self.ocrGroup.titleLabel.setText(t("ocr_inferred_text", "Inferred Text"))
        self.ocrScanCard.titleLabel.setText("Full Screen Scan")
        self.ocrScanBtn.setText("Scan Full Screen")
        self.ocrResultCard.titleLabel.setText(t("ocr_result_title", "OCR Result"))
