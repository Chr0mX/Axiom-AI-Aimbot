# capture_page.py
"""Capture Page — Screenshot Method, UVC, NDI, UDP, Preview settings"""

import os
import socket
import sys
import subprocess
import time
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel, QMessageBox, QSizePolicy, QVBoxLayout
from qfluentwidgets import (
    SettingCardGroup, SwitchSettingCard, FluentIcon,
    ComboBox, PushButton, SettingCard, BodyLabel, SegmentedWidget,
    CaptionLabel, SwitchButton, LineEdit,
)
from ..components.slider_spin_card import SliderSpinCard
from ..base_page import BasePage
from ..language_manager import t
from ..theme_colors import ThemeColors


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


class _UvcProbeWorker(QThread):
    """Enumerates supported UVC resolutions/FPS off the Qt main thread.

    list_supported_uvc_resolutions()/list_supported_uvc_fps() (screen_capture.py)
    open a cv2.VideoCapture and cycle many set()/get() calls — each a real
    driver round-trip. Running that synchronously on the GUI thread (the old
    behavior) freezes the UI for the duration. A combined worker (not two
    separate ones) also avoids two near-simultaneous competing opens against
    the same device index.
    """

    resultReady = pyqtSignal(int, list, list, list)  # (generation, resolutions, fps_list, device_names)

    def __init__(self, generation, device, method, width, height, parent=None):
        super().__init__(parent)
        self._generation = generation
        self._device = device
        self._method = method
        self._width = width
        self._height = height

    def run(self):
        from core.screen_capture import (
            list_supported_uvc_resolutions, list_supported_uvc_fps, list_uvc_device_names,
        )
        # Belt-and-suspenders: screen_capture.py's probe helpers already
        # guard their own cv2 calls, but an unhandled exception escaping a
        # QThread.run() can still take the whole app down (observed on some
        # driver stacks where cv2.VideoCapture.isOpened() reports True on a
        # handle that's actually broken). Never let this thread die loudly.
        try:
            resolutions = list_supported_uvc_resolutions(self._device, self._method)
        except Exception:
            resolutions = []
        try:
            # FPS is probed at the caller's configured resolution (matching
            # the original _refreshUvcFps behavior), not whatever the
            # resolution enumeration happened to find first.
            fps_list = list_supported_uvc_fps(self._device, self._width, self._height, self._method)
        except Exception:
            fps_list = []
        try:
            device_names = list_uvc_device_names()
        except Exception:
            device_names = []
        self.resultReady.emit(self._generation, resolutions, fps_list, device_names)


class CapturePage(BasePage):
    """Capture Settings Page — Screenshot Method, UVC, NDI, Preview"""

    def __init__(self, parent=None):
        super().__init__("tab_capture", parent)
        self._config = None
        self._isLoadingConfig = False
        self._scan_started: float = 0.0
        self._uvc_probe_generation = 0
        self._uvc_probe_worker = None
        self._initWidgets()
        self._initLayout()
        self._connectSignals()

    def setConfig(self, config):
        self._config = config
        self._loadFromConfig()

    def showEvent(self, event):
        super().showEvent(event)
        if self._config and str(getattr(self._config, 'screenshot_method', '')).lower() == 'uvc':
            # Cheap config read after the uvc_actual_* fix below — no device
            # I/O, so no throttle needed. Resolution/FPS support lists don't
            # change for a stable device+driver, so unlike the hw-info
            # readout, those aren't worth re-enumerating just from switching
            # back to this tab; _loadFromConfig() and the explicit change
            # handlers already cover the cases where they'd actually differ.
            self._queryUvcHwInfo()

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
        # Field order mirrors OBS's V4L2/DirectShow source properties panel:
        # capture method, device, resolution, FPS, video format, then a
        # read-only "active values" readout with its own refresh controls.
        self.uvcGroup = SettingCardGroup("UVC Camera", self.scrollWidget)

        self.uvcCaptureMethodCombo = ComboBox()
        self.uvcCaptureMethodCombo.addItems(["msmf", "dshow", "ffmpeg"])
        self.uvcCaptureMethodCombo.setMinimumWidth(140)
        self.uvcCaptureMethodCard = SettingCard(
            FluentIcon.CAMERA,
            "UVC Capture Method",
            "msmf recommended for 1080p60 on Windows 10/11. If MJPEG "
            "negotiation fails despite the device working fine in other "
            "DirectShow apps (e.g. OBS), try 'dshow' instead. 'ffmpeg' uses "
            "an external ffmpeg.exe subprocess instead of OpenCV — see the "
            "FFmpeg options below.",
            self.uvcGroup
        )
        self.uvcCaptureMethodCard.hBoxLayout.addWidget(self.uvcCaptureMethodCombo, 0, Qt.AlignmentFlag.AlignRight)
        self.uvcCaptureMethodCard.hBoxLayout.addSpacing(16)

        self.uvcDeviceCombo = ComboBox()
        self.uvcDeviceCombo.setMinimumWidth(260)
        self.uvcDeviceCard = SettingCard(
            FluentIcon.CAMERA,
            "Device",
            "Select the UVC capture device",
            self.uvcGroup
        )
        self.uvcDeviceCard.hBoxLayout.addWidget(self.uvcDeviceCombo, 0, Qt.AlignmentFlag.AlignRight)
        self.uvcDeviceCard.hBoxLayout.addSpacing(16)

        self.uvcResolutionCombo = ComboBox()
        self.uvcResolutionCombo.setMinimumWidth(180)
        self.uvcResolutionCard = SettingCard(
            FluentIcon.FULL_SCREEN,
            "Resolution",
            "Auto-detect supported resolutions",
            self.uvcGroup
        )
        self.uvcResolutionCard.hBoxLayout.addWidget(self.uvcResolutionCombo, 0, Qt.AlignmentFlag.AlignRight)
        self.uvcResolutionCard.hBoxLayout.addSpacing(16)

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
        self.uvcFpsCombo.addItems(["30", "60", "120", "144", "165", "240"])
        self.uvcFpsCombo.setCurrentText("60")
        self.uvcFpsCombo.setMinimumWidth(120)
        self.uvcFpsCard = SettingCard(
            FluentIcon.SPEED_MEDIUM,
            "FPS",
            "",
            self.uvcGroup
        )
        self.uvcFpsCard.hBoxLayout.addWidget(self.uvcFpsCombo, 0, Qt.AlignmentFlag.AlignRight)
        self.uvcFpsCard.hBoxLayout.addSpacing(16)

        self.uvcVideoFormatCombo = ComboBox()
        self.uvcVideoFormatCombo.addItems(["MJPEG", "YUY2", "NV12", "YUV420P"])
        self.uvcVideoFormatCombo.setMinimumWidth(120)
        self.uvcVideoFormatCard = SettingCard(
            FluentIcon.VIDEO,
            "Video Format",
            "MJPEG (compressed) recommended for 1080p60+; YUY2/NV12/YUV420P "
            "are raw and need much more USB bandwidth at the same "
            "resolution/FPS.",
            self.uvcGroup
        )
        self.uvcVideoFormatCard.hBoxLayout.addWidget(self.uvcVideoFormatCombo, 0, Qt.AlignmentFlag.AlignRight)
        self.uvcVideoFormatCard.hBoxLayout.addSpacing(16)

        # Applies to every UVC capture method (not just ffmpeg) — see
        # uvc_crop_mode's docstring in config.py for the dshow/msmf-specific
        # caveat (freezes the crop rect only; no throughput/CPU benefit there
        # since that capture path is already in-process with no pipe).
        self.uvcCropModeCombo = ComboBox()
        self.uvcCropModeCombo.addItems(["Dynamic", "Fixed (centered)"])
        self.uvcCropModeCombo.setMinimumWidth(160)
        self.uvcCropModeCard = SettingCard(
            FluentIcon.ZOOM_IN,
            "Crop Mode",
            "Dynamic: Axiom crops per-frame to the live Detection Range. "
            "Fixed: the crop rectangle is frozen (centered) at capture-start "
            "instead — a Detection Range change then needs a capture restart "
            "to take effect. With 'ffmpeg' capture method, the crop also "
            "happens inside ffmpeg itself before the frame is piped back, "
            "so far less data crosses the subprocess pipe; with 'dshow'/"
            "'msmf' this only freezes which region is used, no throughput "
            "difference.",
            self.uvcGroup
        )
        self.uvcCropModeCard.hBoxLayout.addWidget(self.uvcCropModeCombo, 0, Qt.AlignmentFlag.AlignRight)
        self.uvcCropModeCard.hBoxLayout.addSpacing(16)

        # === FFmpeg mode options (only relevant/visible when Capture Method == 'ffmpeg') ===
        self.uvcFfmpegPathEdit = LineEdit()
        self.uvcFfmpegPathEdit.setPlaceholderText("auto-detect (bundled ffmpeg/ or system PATH)")
        self.uvcFfmpegPathEdit.setMinimumWidth(260)
        self.uvcFfmpegPathCard = SettingCard(
            FluentIcon.FOLDER,
            "FFmpeg Path",
            "Optional override — path to ffmpeg.exe. Leave blank to "
            "auto-detect (src/ffmpeg/ffmpeg.exe, then system PATH). "
            "Get an LGPL build from ffmpeg.org's build list if needed.",
            self.uvcGroup
        )
        self.uvcFfmpegPathCard.hBoxLayout.addWidget(self.uvcFfmpegPathEdit, 0, Qt.AlignmentFlag.AlignRight)
        self.uvcFfmpegPathCard.hBoxLayout.addSpacing(16)
        self.uvcFfmpegPathCard.setVisible(False)

        self.uvcHwInfoLabel = BodyLabel("—")
        self.uvcQueryBtn = PushButton("Query")
        self.uvcQueryBtn.setFixedWidth(90)
        self.uvcHwInfoCard = SettingCard(
            FluentIcon.INFO,
            "Device Resolution & FPS",
            "Actual values reported by the driver",
            self.uvcGroup
        )
        self.uvcHwInfoCard.hBoxLayout.addWidget(self.uvcHwInfoLabel, 0, Qt.AlignmentFlag.AlignRight)
        self.uvcHwInfoCard.hBoxLayout.addWidget(self.uvcQueryBtn, 0, Qt.AlignmentFlag.AlignRight)
        self.uvcHwInfoCard.hBoxLayout.addSpacing(16)

        self.uvcRefreshResolutionBtn = PushButton("Refresh Device")
        self.uvcRefreshResolutionBtn.setFixedWidth(110)
        self.uvcRefreshResolutionCard = SettingCard(
            FluentIcon.SYNC,
            "Refresh Device",
            "Re-scan connected UVC devices and their supported resolutions/FPS",
            self.uvcGroup
        )
        self.uvcRefreshResolutionCard.hBoxLayout.addWidget(self.uvcRefreshResolutionBtn, 0, Qt.AlignmentFlag.AlignRight)
        self.uvcRefreshResolutionCard.hBoxLayout.addSpacing(16)

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

        # === Active Weapon (2nd Inference) ===
        self.ocrGroup = SettingCardGroup(t("ocr_inferred_text", "Active Weapon"), self.scrollWidget)

        self.ocrFpsCard = SliderSpinCard(
            FluentIcon.SPEED_HIGH,
            "OCR Capture FPS",
            1, 10,
            suffix=" FPS",
            description="How often OCR runs — lower values reduce CPU load",
            parent=self.ocrGroup
        )

        self.ocrScanBtn = PushButton("Scan ROI")
        self.ocrScanBtn.setFixedWidth(140)
        self.ocrScanCard = SettingCard(
            FluentIcon.SEARCH,
            "Scan ROI",
            "Run OCR on the fixed region once and update the preview below",
            self.ocrGroup
        )
        self.ocrScanCard.hBoxLayout.addWidget(self.ocrScanBtn, 0, Qt.AlignmentFlag.AlignRight)
        self.ocrScanCard.hBoxLayout.addSpacing(16)

        self.ocrResultLabel = BodyLabel("—")
        self.ocrResultLabel.setWordWrap(True)
        self.ocrResultCard = SettingCard(
            FluentIcon.DOCUMENT,
            t("ocr_result_title", "Detected"),
            "",
            self.ocrGroup
        )
        self.ocrResultCard.hBoxLayout.addWidget(self.ocrResultLabel, 1, Qt.AlignmentFlag.AlignRight)
        self.ocrResultCard.hBoxLayout.addSpacing(16)

        self.ocrRoiFrame = QFrame(self)
        self.ocrRoiFrame.setObjectName("ocrRoiFrame")
        self.ocrRoiFrame.setStyleSheet(
            "#ocrRoiFrame { background: rgba(255,255,255,0.04); border-radius: 8px; }"
        )
        _roi_vbox = QVBoxLayout(self.ocrRoiFrame)
        _roi_vbox.setContentsMargins(16, 8, 16, 8)
        _roi_vbox.setSpacing(6)

        _roi_hdr = QHBoxLayout()
        _roi_hdr.addWidget(CaptionLabel("ROI Preview"))
        _roi_hdr.addStretch(1)
        self.ocrLiveToggle = SwitchButton(self.ocrRoiFrame)
        self.ocrLiveToggle.setText("Live")
        self.ocrLiveToggle.setChecked(False)
        _roi_hdr.addWidget(self.ocrLiveToggle)
        _roi_vbox.addLayout(_roi_hdr)

        self.ocrRoiLabel = QLabel()
        self.ocrRoiLabel.setFixedHeight(58)
        self.ocrRoiLabel.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.ocrRoiLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.ocrRoiLabel.setStyleSheet(
            f"background: {ThemeColors.CARD_BACKGROUND.get()}; "
            f"border: 1px solid {ThemeColors.CARD_BORDER.get()}; "
            f"border-radius: 8px;"
        )
        _roi_vbox.addWidget(self.ocrRoiLabel)

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

        self.uvcGroup.addSettingCard(self.uvcCaptureMethodCard)
        self.uvcGroup.addSettingCard(self.uvcDeviceCard)
        self.uvcGroup.addSettingCard(self.uvcResolutionCard)
        self.uvcGroup.addSettingCard(self.uvcWidthCard)
        self.uvcGroup.addSettingCard(self.uvcHeightCard)
        self.uvcGroup.addSettingCard(self.uvcFpsCard)
        self.uvcGroup.addSettingCard(self.uvcVideoFormatCard)
        self.uvcGroup.addSettingCard(self.uvcCropModeCard)
        self.uvcGroup.addSettingCard(self.uvcFfmpegPathCard)
        self.uvcGroup.addSettingCard(self.uvcHwInfoCard)
        self.uvcGroup.addSettingCard(self.uvcRefreshResolutionCard)
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

        self.ocrGroup.addSettingCard(self.ocrFpsCard)
        self.ocrGroup.addSettingCard(self.ocrScanCard)
        self.ocrGroup.addSettingCard(self.ocrResultCard)
        self.addContent(self.ocrGroup)
        self.addContent(self.ocrRoiFrame)

        self.scrollLayout.addStretch(1)

    # ──────────────────────────────────────────────
    # Signal connections
    # ──────────────────────────────────────────────

    def _connectSignals(self):
        self.screenshotMethodCombo.currentTextChanged.connect(self._onScreenshotMethodChanged)
        self.screenshotIntervalCard.valueChanged.connect(self._onScreenshotIntervalChanged)
        self.uvcDeviceCombo.currentTextChanged.connect(self._onUvcDeviceChanged)
        self.uvcResolutionCombo.currentTextChanged.connect(self._onUvcResolutionChanged)
        self.uvcRefreshResolutionBtn.clicked.connect(self._startUvcProbe)
        self.uvcFpsCombo.currentTextChanged.connect(self._onUvcFpsChanged)
        self.uvcCaptureMethodCombo.currentTextChanged.connect(self._onUvcCaptureMethodChanged)
        self.uvcVideoFormatCombo.currentTextChanged.connect(self._onUvcVideoFormatChanged)
        self.uvcFfmpegPathEdit.editingFinished.connect(self._onUvcFfmpegPathChanged)
        self.uvcCropModeCombo.currentTextChanged.connect(self._onUvcCropModeChanged)
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
        self.ocrFpsCard.valueChanged.connect(self._onOcrFpsChanged)
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

            _device_index = int(getattr(self._config, 'uvc_device_index', 0))
            # Seed synchronously with the configured value so the combo
            # shows something immediately; _startUvcProbeDelayed() (below,
            # for screenshot_method == 'uvc') enriches it with the full
            # enumerated device list in the background and preserves this
            # selection once it arrives (see _onUvcProbeResult).
            self.uvcDeviceCombo.blockSignals(True)
            self.uvcDeviceCombo.clear()
            self.uvcDeviceCombo.addItem(f"Device {_device_index}", userData=_device_index)
            self.uvcDeviceCombo.blockSignals(False)
            _capture_method = str(getattr(self._config, 'uvc_capture_method', 'msmf'))
            self.uvcCaptureMethodCombo.setCurrentText(_capture_method)
            self.uvcVideoFormatCombo.setCurrentText(str(getattr(self._config, 'uvc_video_format', 'mjpeg')).upper())
            self.uvcFfmpegPathEdit.setText(str(getattr(self._config, 'uvc_ffmpeg_path', '') or ''))
            self.uvcCropModeCombo.setCurrentText(
                'Fixed (centered)' if str(getattr(self._config, 'uvc_crop_mode', 'dynamic')).lower() == 'fixed'
                else 'Dynamic'
            )
            self._updateFfmpegControlsVisibility(_capture_method)
            resolution_text = (
                f"{getattr(self._config, 'uvc_width', self._config.width)}"
                f"x{getattr(self._config, 'uvc_height', self._config.height)}")
            if screenshot_method == 'uvc':
                # Seed synchronously with the configured value so the combo
                # shows something immediately; _startUvcProbeDelayed()
                # enriches it with the full supported list in the background
                # (after giving the live backend's own reinit a head start —
                # see its docstring) and preserves this selection (via
                # currentText()) once it arrives.
                self.uvcResolutionCombo.blockSignals(True)
                self.uvcResolutionCombo.clear()
                self.uvcResolutionCombo.addItem(resolution_text)
                self.uvcResolutionCombo.blockSignals(False)
                self._startUvcProbeDelayed()
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

            self.ocrFpsCard.setValue(int(getattr(self._config, 'second_inference_fps', 2)))

            self._updateCaptureControlsVisibility(screenshot_method)
            self._syncRoiLabelSize()
        finally:
            self._isLoadingConfig = False

    # ──────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────

    def _syncRoiLabelSize(self):
        mode = getattr(self._config, 'second_inference_mode', 'off') if self._config else 'off'
        if mode == 'v2_onnx':
            from core.hud_inference import _parse_roi, _HUD_ROI_DEFAULT_STR
            coords = getattr(self._config, 'hud_roi_coords', _HUD_ROI_DEFAULT_STR) or _HUD_ROI_DEFAULT_STR
            r = _parse_roi(coords) or _parse_roi(_HUD_ROI_DEFAULT_STR)
            self.ocrRoiLabel.setFixedHeight(r["height"] if r else 88)
        else:
            self.ocrRoiLabel.setFixedHeight(58)

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

    def _startUvcProbeDelayed(self, delay_ms: int = 1500):
        """Schedule _startUvcProbe() after a delay instead of firing it immediately.

        Every caller of this (switching TO uvc, loading a config with uvc
        active, changing device/resolution/capture-method) also changes a
        field _uvc_signature() watches, which makes the AI loop's
        _capture_worker (polling every ~0.5s) reinitialize UVCCapture and
        open ITS OWN handle to the same device around the same moment. Two
        cv2.VideoCapture opens racing for the same device at the same time
        is exactly the contention this session's earlier UVC fix was meant
        to avoid — starting the probe immediately reintroduced it at the
        single most common trigger (switching to UVC in the first place):
        observed as a many-second stall before UVCCapture.__init__()
        completes, the driver failing to negotiate MJPEG (logged as "FOURCC
        MJPG not accepted"), and even a spurious extra reinit once the
        contention resolves. Delaying the probe's own competing handle-open
        past the live backend's ~0.5s reinit window fixes the worst of it;
        this doesn't apply to the explicit "Refresh" button, which fires on
        a presumably-already-stable backend.
        """
        QTimer.singleShot(delay_ms, self._startUvcProbe)

    def _startUvcProbe(self):
        """Kick off a background enumeration of supported UVC resolutions/FPS.

        Replaces the old synchronous _refreshUvcResolutions()/_refreshUvcFps()
        pair — both opened a competing cv2.VideoCapture to the live device on
        the Qt main thread, freezing the UI and risking driver-level
        contention with the AI loop's live capture handle. This still opens
        a second handle (unavoidable — enumeration must try many
        resolutions/FPS values, not just read the current one), but now off
        the GUI thread, and superseded results are discarded via the
        generation counter rather than racing to update the combos.
        """
        if not self._config:
            return
        # A probe scheduled via _startUvcProbeDelayed() can still fire after
        # the user has since switched away from 'uvc' (e.g. to 'udp') — the
        # QTimer doesn't know the method changed in the meantime. Check the
        # CURRENT config here (not at schedule time) so a stale timer is a
        # no-op instead of opening a competing device handle for a backend
        # that isn't even active anymore.
        if str(getattr(self._config, 'screenshot_method', '')).lower() != 'uvc':
            return
        # Never let two probes race the same device concurrently. This isn't
        # just about wasted work — cv2.VideoCapture(DSHOW) isn't safe to open
        # from two threads at once (DirectShow/COM apartment-threading), and
        # on some driver stacks concurrent opens have caused a native access
        # violation that crashes the whole process, bypassing every Python
        # try/except in the probe path. Reassigning self._uvc_probe_worker
        # here does NOT stop an in-flight worker's run() — so skip starting
        # a new one outright rather than relying on the generation counter
        # (which only discards a stale *result*, not a stale *in-flight
        # open*).
        if self._uvc_probe_worker is not None and self._uvc_probe_worker.isRunning():
            return
        self._uvc_probe_generation += 1
        generation = self._uvc_probe_generation
        device = int(getattr(self._config, 'uvc_device_index', 0))
        method = str(getattr(self._config, 'uvc_capture_method', 'msmf'))
        width = int(getattr(self._config, 'uvc_width', 1920))
        height = int(getattr(self._config, 'uvc_height', 1080))
        print(f"[Capture][UVC] Probing supported resolutions/FPS (device={device}, method={method})...")
        self._uvc_probe_worker = _UvcProbeWorker(generation, device, method, width, height, parent=self)
        self._uvc_probe_worker.resultReady.connect(self._onUvcProbeResult)
        self._uvc_probe_worker.start()

    def _onUvcProbeResult(self, generation, resolutions, fps_list, device_names=None):
        if generation != self._uvc_probe_generation:
            return  # superseded by a newer probe (device/resolution changed again)
        print(f"[Capture][UVC] Found {len(resolutions)} supported resolution(s): {resolutions}")
        print(f"[Capture][UVC] Supported FPS: {fps_list}")

        device_names = device_names or []
        configured_index = int(getattr(self._config, 'uvc_device_index', 0)) if self._config else 0
        self.uvcDeviceCombo.blockSignals(True)
        self.uvcDeviceCombo.clear()
        if device_names:
            for i, name in enumerate(device_names):
                self.uvcDeviceCombo.addItem(name, userData=i)
        else:
            # No enumeration available (pygrabber/comtypes not installed) —
            # fall back to plain numeric slots so uvc_device_index can still
            # be picked for dshow/msmf/any.
            for i in range(8):
                self.uvcDeviceCombo.addItem(f"Device {i}", userData=i)
        select_idx = -1
        for i in range(self.uvcDeviceCombo.count()):
            if self.uvcDeviceCombo.itemData(i) == configured_index:
                select_idx = i
                break
        if select_idx < 0 and configured_index >= self.uvcDeviceCombo.count():
            self.uvcDeviceCombo.addItem(f"Device {configured_index}", userData=configured_index)
            select_idx = self.uvcDeviceCombo.count() - 1
        if select_idx >= 0:
            self.uvcDeviceCombo.setCurrentIndex(select_idx)
        self.uvcDeviceCombo.blockSignals(False)

        current_text = self.uvcResolutionCombo.currentText().strip()
        self.uvcResolutionCombo.blockSignals(True)
        self.uvcResolutionCombo.clear()
        if resolutions:
            for width, height in resolutions:
                self.uvcResolutionCombo.addItem(f"{width}x{height}")
        else:
            # Enumeration found nothing (device unreachable, or the driver
            # didn't accept any of the probed resolutions) — offer the
            # common presets plus the currently configured value so the
            # combo isn't left with just one arbitrary entry.
            fallback = f"{int(getattr(self._config, 'uvc_width', 1920))}x{int(getattr(self._config, 'uvc_height', 1080))}"
            for preset in ("1280x720", "1920x1080", "2560x1440"):
                self.uvcResolutionCombo.addItem(preset)
            if self.uvcResolutionCombo.findText(fallback) < 0:
                self.uvcResolutionCombo.addItem(fallback)
        # Whatever the probe/fallback found, it's still a guess (driver
        # capability enumeration can be unreliable — e.g. virtual cameras
        # like "OBS Virtual Camera" accept any requested cv2.set() value
        # without validating it, so the cv2 fallback path can report
        # candidates that were never real). The live backend's own already-
        # negotiated resolution (uvc_actual_width/height, published by
        # UVCCapture.__init__ from its actual open handle) is ground truth —
        # always make sure it's a selectable option, even if the probe
        # missed it entirely.
        actual_w = int(getattr(self._config, 'uvc_actual_width', 0) or 0)
        actual_h = int(getattr(self._config, 'uvc_actual_height', 0) or 0)
        if actual_w > 0 and actual_h > 0:
            actual_text = f"{actual_w}x{actual_h}"
            if self.uvcResolutionCombo.findText(actual_text) < 0:
                self.uvcResolutionCombo.addItem(actual_text)
            if not current_text:
                current_text = actual_text
        if current_text:
            idx = self.uvcResolutionCombo.findText(current_text)
            if idx >= 0:
                self.uvcResolutionCombo.setCurrentIndex(idx)
        self.uvcResolutionCombo.blockSignals(False)

        fps_list = list(fps_list) if fps_list else [30, 60, 120, 144, 165, 240]
        # Same reasoning as the resolution list above — always include the
        # live backend's actually-negotiated FPS (uvc_actual_fps) so the
        # combo can't miss the one value we already know for certain works.
        actual_fps = int(round(float(getattr(self._config, 'uvc_actual_fps', 0) or 0)))
        if actual_fps > 0 and actual_fps not in fps_list:
            fps_list.append(actual_fps)
            fps_list.sort()
        current_fps = self.uvcFpsCombo.currentText()
        if not current_fps and actual_fps > 0:
            current_fps = str(actual_fps)
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
            self._startUvcProbeDelayed()
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

    def _onUvcDeviceChanged(self, text):
        if self._config:
            data = self.uvcDeviceCombo.currentData()
            self._config.uvc_device_index = int(data) if data is not None else 0
        self._startUvcProbeDelayed()
        # config.uvc_actual_* only refreshes once the AI loop's live backend
        # hot-swaps to the new device (ai_loop.py's _capture_worker polls for
        # config changes every 0.5s — see reinitialize_if_method_changed()),
        # so give that a moment before reading it; this delay is now purely
        # about staleness, not blocking, since _queryUvcHwInfo() no longer
        # does any device I/O of its own.
        QTimer.singleShot(700, self._queryUvcHwInfo)

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
        self._startUvcProbeDelayed()
        QTimer.singleShot(700, self._queryUvcHwInfo)

    def _onUvcFpsChanged(self, value):
        if self._config:
            try:
                self._config.uvc_fps = int(value)
            except (ValueError, TypeError):
                pass

    def _onUvcCaptureMethodChanged(self, text):
        if self._config:
            self._config.uvc_capture_method = str(text)
        self._updateFfmpegControlsVisibility(text)
        self._startUvcProbeDelayed()
        QTimer.singleShot(700, self._queryUvcHwInfo)

    def _onUvcVideoFormatChanged(self, text):
        if self._config:
            self._config.uvc_video_format = str(text).strip().lower()
        self._startUvcProbeDelayed()
        QTimer.singleShot(700, self._queryUvcHwInfo)

    def _onUvcFfmpegPathChanged(self):
        if self._config:
            self._config.uvc_ffmpeg_path = str(self.uvcFfmpegPathEdit.text()).strip()

    def _onUvcCropModeChanged(self, text):
        if self._config:
            self._config.uvc_crop_mode = 'fixed' if text.strip().lower().startswith('fixed') else 'dynamic'

    def _updateFfmpegControlsVisibility(self, capture_method_text: str):
        # uvcCropModeCard applies to every capture method and stays visible;
        # only the ffmpeg.exe path override is ffmpeg-specific.
        is_ffmpeg = str(capture_method_text).strip().lower() == 'ffmpeg'
        self.uvcFfmpegPathCard.setVisible(is_ffmpeg)

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
        """Read the live UVC device's actual resolution/FPS.

        Reads config.uvc_actual_* (published by UVCCapture.__init__ from its
        own already-open handle) instead of opening a second competing
        cv2.VideoCapture to the same device index — most UVC/webcam drivers
        don't handle two simultaneous open handles gracefully, and a second
        handle opened while the AI loop's live capture is actively streaming
        can stall/corrupt frames on that live handle (this was the root
        cause of erratic aim while adjusting UVC settings).
        """
        if not self._config:
            self.uvcHwInfoLabel.setText("—")
            return
        w = int(getattr(self._config, 'uvc_actual_width', 0) or 0)
        h = int(getattr(self._config, 'uvc_actual_height', 0) or 0)
        fps = float(getattr(self._config, 'uvc_actual_fps', 0.0) or 0.0)
        if w <= 0 or h <= 0:
            self.uvcHwInfoLabel.setText("—  (device not available)")
            return
        fps_str = f"{fps:.1f}" if fps > 0 else "?"
        self.uvcHwInfoLabel.setText(f"{w} × {h} @ {fps_str} fps")

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

    def _onOcrFpsChanged(self, value: int):
        if self._isLoadingConfig or not self._config:
            return
        self._config.second_inference_fps = max(1, min(10, value))

    def _onOcrScanClicked(self):
        self.ocrResultLabel.setText("Scanning...")
        self._scan_started = time.monotonic()
        mode = getattr(self._config, 'second_inference_mode', 'off') if self._config else 'off'
        if mode == 'v1_ocr':
            from core.ocr_inference import trigger_scan
            trigger_scan()
        elif mode == 'v2_onnx':
            from core.hud_inference import trigger_hud_scan
            trigger_hud_scan()
        QTimer.singleShot(1500, self._updateRoiPreview)

    _box_logged: bool = False
    _last_roi_render_sig: tuple | None = None

    @staticmethod
    def _draw_hud_boxes(roi_rgb: "np.ndarray", boxes: list) -> "np.ndarray":
        if not boxes:
            return roi_rgb
        try:
            import cv2
            from core.hud_inference import get_hud_model_size
            inp_w, inp_h = get_hud_model_size()
            roi_h, roi_w = roi_rgb.shape[:2]
            scale = min(inp_w / roi_w, inp_h / roi_h)
            pad_x = (inp_w - roi_w * scale) / 2
            pad_y = (inp_h - roi_h * scale) / 2
            for (x1m, y1m, x2m, y2m, _cid, score) in boxes:
                px1 = int((x1m - pad_x) / scale)
                py1 = int((y1m - pad_y) / scale)
                px2 = int((x2m - pad_x) / scale)
                py2 = int((y2m - pad_y) / scale)
                below = score < 0
                color = (255, 140, 0) if below else (0, 255, 80)  # orange hint / green confirmed
                label = f"~{abs(score):.0%}" if below else f"{score:.0%}"
                print(f"[HUD box] model({inp_w}×{inp_h}) scale={scale:.3f} pad=({pad_x:.1f},{pad_y:.1f}) "
                      f"model_coords=({x1m:.0f},{y1m:.0f},{x2m:.0f},{y2m:.0f}) "
                      f"roi_px=({px1},{py1},{px2},{py2}) {'HINT' if below else 'DETECT'}")
                cv2.rectangle(roi_rgb, (px1, py1), (px2, py2), color, 1)
                cv2.putText(roi_rgb, label, (max(px1, 0), max(py1 - 2, 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        except Exception as exc:
            print(f"[HUD box] draw error: {exc}")
        return roi_rgb

    def _updateRoiPreview(self):
        mode = getattr(self._config, 'second_inference_mode', 'off') if self._config else 'off'
        if mode == 'v1_ocr':
            from core.ocr_inference import get_roi_image
            roi = get_roi_image()
            boxes = []
        elif mode == 'v2_onnx':
            from core.hud_inference import get_hud_roi_image, get_hud_boxes, _parse_roi, _HUD_ROI_DEFAULT_STR
            roi = get_hud_roi_image()
            boxes = get_hud_boxes()
            coords_str = getattr(self._config, 'hud_roi_coords', _HUD_ROI_DEFAULT_STR) or _HUD_ROI_DEFAULT_STR
            _r = _parse_roi(coords_str) or _parse_roi(_HUD_ROI_DEFAULT_STR)
            self.ocrRoiLabel.setFixedHeight((_r["height"] if _r else 88))
        else:
            return
        if roi is None:
            return
        if roi.ndim != 3 or roi.shape[2] < 3:
            return

        # Skip redraw when ROI and boxes haven't changed
        sig = (id(roi), len(boxes), tuple(round(b[5], 3) for b in boxes))
        if sig == self._last_roi_render_sig:
            return
        self._last_roi_render_sig = sig

        h, w = roi.shape[:2]
        roi_rgb = roi[:, :, :3][:, :, ::-1].copy()  # BGR(A) → RGB
        if boxes:
            roi_rgb = self._draw_hud_boxes(roi_rgb, boxes)
        buf = roi_rgb.tobytes()
        qimg = QImage(buf, w, h, w * 3, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        self.ocrRoiLabel.setPixmap(
            pix.scaled(self.ocrRoiLabel.width(), self.ocrRoiLabel.height(),
                       Qt.AspectRatioMode.KeepAspectRatio,
                       Qt.TransformationMode.SmoothTransformation)
        )

    def _refreshOcrDisplay(self):
        mode = getattr(self._config, 'second_inference_mode', 'off') if self._config else 'off'
        if mode == 'v1_ocr':
            from core.ocr_inference import get_ocr_results
            lines = get_ocr_results()
        elif mode == 'v2_onnx':
            from core.hud_inference import get_hud_results
            lines = get_hud_results()
        else:
            if self.ocrResultLabel.text() not in ("—", "Scanning..."):
                pass
            return
        if lines:
            self.ocrResultLabel.setText("\n".join(lines))
        elif self.ocrResultLabel.text() == "Scanning...":
            elapsed = time.monotonic() - getattr(self, '_scan_started', 0)
            if elapsed > 5:
                self.ocrResultLabel.setText("No result — check console for errors")
        else:
            self.ocrResultLabel.setText("—")

        if getattr(self, 'ocrLiveToggle', None) and self.ocrLiveToggle.isChecked():
            self._updateRoiPreview()

    # ──────────────────────────────────────────────
    # Retranslate
    # ──────────────────────────────────────────────

    def retranslateUi(self):
        super().retranslateUi()
        self.captureGroup.titleLabel.setText(t("capture_method_group", "Capture"))
        self.screenshotMethodCard.titleLabel.setText(t("screenshot_method"))
        self.screenshotIntervalCard.titleLabel.setText(t("screenshot_interval"))
        self.uvcGroup.titleLabel.setText("UVC Camera")
        self.uvcCaptureMethodCard.titleLabel.setText("UVC Capture Method")
        self.uvcCaptureMethodCard.contentLabel.setText(
            "msmf recommended for 1080p60 on Windows 10/11. If MJPEG "
            "negotiation fails despite the device working fine in other "
            "DirectShow apps (e.g. OBS), try 'dshow' instead. 'ffmpeg' uses "
            "an external ffmpeg.exe subprocess instead of OpenCV — see the "
            "FFmpeg options below."
        )
        self.uvcDeviceCard.titleLabel.setText("Device")
        self.uvcDeviceCard.contentLabel.setText("Select the UVC capture device")
        self.uvcResolutionCard.titleLabel.setText("Resolution")
        self.uvcResolutionCard.contentLabel.setText("Auto-detect supported resolutions")
        self.uvcWidthCard.titleLabel.setText("UVC Width")
        self.uvcHeightCard.titleLabel.setText("UVC Height")
        self.uvcFpsCard.titleLabel.setText("FPS")  # type: ignore[attr-defined]
        self.uvcVideoFormatCard.titleLabel.setText("Video Format")
        self.uvcVideoFormatCard.contentLabel.setText(
            "MJPEG (compressed) recommended for 1080p60+; YUY2/NV12/YUV420P "
            "are raw and need much more USB bandwidth at the same "
            "resolution/FPS."
        )
        self.uvcFfmpegPathCard.titleLabel.setText("FFmpeg Path")
        self.uvcFfmpegPathCard.contentLabel.setText(
            "Optional override — path to ffmpeg.exe. Leave blank to "
            "auto-detect (src/ffmpeg/ffmpeg.exe, then system PATH). "
            "Get an LGPL build from ffmpeg.org's build list if needed."
        )
        self.uvcCropModeCard.titleLabel.setText("Crop Mode")
        self.uvcCropModeCard.contentLabel.setText(
            "Dynamic: Axiom crops per-frame to the live Detection Range. "
            "Fixed: the crop rectangle is frozen (centered) at capture-start "
            "instead — a Detection Range change then needs a capture restart "
            "to take effect. With 'ffmpeg' capture method, the crop also "
            "happens inside ffmpeg itself before the frame is piped back, "
            "so far less data crosses the subprocess pipe; with 'dshow'/"
            "'msmf' this only freezes which region is used, no throughput "
            "difference."
        )
        self.uvcHwInfoCard.titleLabel.setText("Device Resolution & FPS")
        self.uvcHwInfoCard.contentLabel.setText("Actual values reported by the driver")
        self.uvcQueryBtn.setText("Query")
        self.uvcRefreshResolutionCard.titleLabel.setText("Refresh Device")
        self.uvcRefreshResolutionCard.contentLabel.setText(
            "Re-scan connected UVC devices and their supported resolutions/FPS"
        )
        self.uvcRefreshResolutionBtn.setText("Refresh Device")
        self.ndiGroup.titleLabel.setText("NDI")
        self.ndiSourceCard.titleLabel.setText("NDI Stream")
        self.ndiSourceCard.contentLabel.setText("Select the NDI source to capture")
        self.ndiRefreshCard.titleLabel.setText("Refresh NDI Streams")
        self.ndiRefreshBtn.setText(t("refresh"))
        self.ndiBandwidthCard.titleLabel.setText("NDI Bandwidth")
        self.ndiBandwidthCard.contentLabel.setText("Receive bandwidth for the NDI stream")
        self.ndiHwInfoCard.titleLabel.setText("Stream Resolution & FPS")
        self.ndiHwInfoCard.contentLabel.setText("Actual values from the active NDI source")
        self.ndiRefreshInfoBtn.setText("Refresh Info")
        self.udpGroup.titleLabel.setText("UDP Stream")
        self.udpSystemIpCard.titleLabel.setText("System IP Address")
        _local_ips = _get_local_ips()
        self.udpSystemIpCard.contentLabel.setText(
            f"Stream to: {', '.join(_local_ips) if _local_ips else '—'}"
        )
        self.udpBindIpCard.titleLabel.setText("Bind IP")
        self.udpBindIpCard.contentLabel.setText("Listen on a specific interface, or 0.0.0.0 for all")
        self.udpPortCard.titleLabel.setText("UDP Port")
        self.udpRefreshCard.titleLabel.setText("Restart Receiver")
        self.udpRefreshCard.contentLabel.setText("Stop and re-bind the UDP socket")
        self.udpRefreshBtn.setText(t("refresh"))
        self.previewGroup.titleLabel.setText(t("preview_group", "Preview"))
        self.uvcPreviewCard.titleLabel.setText("Capture Preview Window")
        self.previewCropCard.titleLabel.setText(t("preview_crop_label"))
        self.previewCropCard.contentLabel.setText(t("preview_crop_desc"))
        self.uvcPreviewScaleCard.titleLabel.setText("Capture Preview Scale Mode")
        self.uvcAlwaysOnTopCard.titleLabel.setText("Always On Top")
        self.previewFpsCapCard.titleLabel.setText("Preview FPS Cap")
        self.ocrGroup.titleLabel.setText(t("ocr_inferred_text", "Active Weapon"))
        self.ocrRoiLabel.setStyleSheet(
            f"background: {ThemeColors.CARD_BACKGROUND.get()}; "
            f"border: 1px solid {ThemeColors.CARD_BORDER.get()}; "
            f"border-radius: 8px;"
        )
        self.ocrFpsCard.titleLabel.setText("OCR Capture FPS")
        self.ocrScanCard.titleLabel.setText("Scan ROI")
        self.ocrScanCard.contentLabel.setText("Run OCR on the fixed region once and update the preview below")
        self.ocrScanBtn.setText("Scan ROI")
        self.ocrResultCard.titleLabel.setText(t("ocr_result_title", "Detected"))
