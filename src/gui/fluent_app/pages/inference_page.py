# inference_page.py
"""Inference Page - FOV, Detection, General Parameters, Inference Performance"""

from PyQt6.QtCore import Qt
from qfluentwidgets import (
    SettingCardGroup, SwitchSettingCard,
    FluentIcon, SettingCard,
)
from ..components.slider_spin_card import SliderSpinCard, SliderLabelCard

from ..base_page import BasePage
from ..language_manager import t


class InferencePage(BasePage):
    """Inference Settings Page — FOV, Detection, General Parameters, Inference Performance"""

    def __init__(self, parent=None):
        super().__init__("tab_inference", parent)
        self._config = None
        self._isLoadingConfig = False
        self._initWidgets()
        self._initLayout()
        self._connectSignals()

    def setConfig(self, config):
        self._config = config
        if hasattr(self, 'detectRangeCard') and self._config:
            max_h = max(1080, self._config.height)
            self.detectRangeCard.slider.setMaximum(max_h)
            self.detectRangeCard.spinBox.setMaximum(max_h)
        self._loadFromConfig()

    def showEvent(self, event):
        super().showEvent(event)
        if self._config and hasattr(self, 'idleDetectEnableCard'):
            self.idleDetectEnableCard.setChecked(
                getattr(self._config, 'idle_detect_enabled', True)
            )

    # ──────────────────────────────────────────────
    # Widget initialisation
    # ──────────────────────────────────────────────

    def _initWidgets(self):
        # === FOV & Detection Range ===
        self.fovGroup = SettingCardGroup(t("fov_and_detect_range"), self.scrollWidget)

        self.fovCard = SliderSpinCard(
            FluentIcon.ZOOM,
            t("fov_size"),
            50, 500,
            description="",
            parent=self.fovGroup
        )

        self.fovFollowCard = SwitchSettingCard(
            FluentIcon.MOVE,
            t("fov_follow_mouse"),
            "",
            parent=self.fovGroup
        )

        self.fovCircleCard = SwitchSettingCard(
            FluentIcon.REMOVE,
            t("fov_circle_filter", "Circular FOV Filter"),
            t("fov_circle_filter_desc", "Only track targets inside the FOV circle, not the full square region"),
            parent=self.fovGroup
        )

        self.detectRangeCard = SliderSpinCard(
            FluentIcon.FULL_SCREEN,
            t("detect_range_size"),
            100, 1080,
            description=t("detect_range_note"),
            parent=self.fovGroup
        )

        # === General Parameters ===
        self.generalGroup = SettingCardGroup(t("general_params"), self.scrollWidget)

        self.detectIntervalCard = SliderSpinCard(
            FluentIcon.SPEED_HIGH,
            t("detect_interval"),
            1, 100,
            suffix="ms",
            description="",
            parent=self.generalGroup
        )

        self.confidenceCard = SliderSpinCard(
            FluentIcon.CERTIFICATE,
            t("min_confidence"),
            1, 100,
            suffix="%",
            description="",
            parent=self.generalGroup
        )

        self.semanticFilterCard = SwitchSettingCard(
            FluentIcon.FILTER,
            t("semantic_filter_enabled", "Semantic FP Filter"),
            t("semantic_filter_desc", "Discard trees, vehicles, and HUD elements by class name and geometry"),
            parent=self.generalGroup
        )

        self.alwaysAimCard = SwitchSettingCard(
            FluentIcon.FINGERPRINT,
            t("always_aim"),
            "",
            parent=self.generalGroup
        )

        self.keepDetectingCard = SwitchSettingCard(
            FluentIcon.UPDATE,
            t("keep_detecting"),
            "",
            parent=self.generalGroup
        )

        self.idleDetectEnableCard = SwitchSettingCard(
            FluentIcon.SPEED_MEDIUM,
            t("idle_detect_enabled"),
            "",
            parent=self.generalGroup
        )

        self.idleDetectIntervalCard = SliderSpinCard(
            FluentIcon.SPEED_MEDIUM,
            t("idle_detect_interval"),
            5, 500,
            suffix="ms",
            description="",
            parent=self.generalGroup
        )

        self.singleTargetCard = SwitchSettingCard(
            FluentIcon.PEOPLE,
            t("single_target_mode"),
            "",
            parent=self.generalGroup
        )

        # === Inference Performance ===
        self.inferPerfGroup = SettingCardGroup(t("inference_performance", "Inference Performance"), self.scrollWidget)

        self.skipLetterboxCard = SwitchSettingCard(
            FluentIcon.SPEED_HIGH,
            t("skip_letterbox_label"),
            t("skip_letterbox_desc"),
            parent=self.inferPerfGroup
        )

        self.cudaIoBindingCard = SwitchSettingCard(
            FluentIcon.COPY,
            t("cuda_io_binding", "CUDA IO Binding"),
            t("cuda_io_binding_desc", "Zero-copy GPU inference. Effective only with CUDA or TensorRT backend."),
            parent=self.inferPerfGroup
        )

        self.frameSkipCard = SwitchSettingCard(
            FluentIcon.SPEED_MEDIUM,
            t("frame_skip_enabled", "Frame Skip Gate"),
            t("frame_skip_desc", "Skip inference when the capture region hasn't changed significantly."),
            parent=self.inferPerfGroup
        )

        self.frameSkipThresholdCard = SliderLabelCard(
            FluentIcon.ALIGNMENT,
            t("frame_skip_threshold", "Skip Threshold"),
            5, 100,
            format_func=lambda v: f"{v / 10:.1f}",
            description=t("frame_skip_threshold_desc", "Avg pixel diff below this value triggers skip (higher = more skipping)"),
            slider_width=160,
            parent=self.inferPerfGroup
        )

    # ──────────────────────────────────────────────
    # Layout
    # ──────────────────────────────────────────────

    def _initLayout(self):
        self.fovGroup.addSettingCard(self.fovCard)
        self.fovGroup.addSettingCard(self.fovFollowCard)
        self.fovGroup.addSettingCard(self.fovCircleCard)
        self.fovGroup.addSettingCard(self.detectRangeCard)
        self.addContent(self.fovGroup)

        self.generalGroup.addSettingCard(self.detectIntervalCard)
        self.generalGroup.addSettingCard(self.confidenceCard)
        self.generalGroup.addSettingCard(self.semanticFilterCard)
        self.generalGroup.addSettingCard(self.alwaysAimCard)
        self.generalGroup.addSettingCard(self.keepDetectingCard)
        self.generalGroup.addSettingCard(self.idleDetectEnableCard)
        self.generalGroup.addSettingCard(self.idleDetectIntervalCard)
        self.generalGroup.addSettingCard(self.singleTargetCard)
        self.addContent(self.generalGroup)

        self.inferPerfGroup.addSettingCard(self.skipLetterboxCard)
        self.inferPerfGroup.addSettingCard(self.cudaIoBindingCard)
        self.inferPerfGroup.addSettingCard(self.frameSkipCard)
        self.inferPerfGroup.addSettingCard(self.frameSkipThresholdCard)
        self.addContent(self.inferPerfGroup)

        self.scrollLayout.addStretch(1)

    # ──────────────────────────────────────────────
    # Signal connections
    # ──────────────────────────────────────────────

    def _connectSignals(self):
        self.fovCard.valueChanged.connect(self._onFovChanged)
        self.fovFollowCard.checkedChanged.connect(self._onFovFollowChanged)
        self.fovCircleCard.checkedChanged.connect(self._onFovCircleChanged)
        self.detectRangeCard.valueChanged.connect(self._onDetectRangeChanged)

        self.detectIntervalCard.valueChanged.connect(self._onDetectIntervalChanged)
        self.confidenceCard.valueChanged.connect(self._onConfidenceChanged)
        self.semanticFilterCard.checkedChanged.connect(self._onSemanticFilterChanged)
        self.alwaysAimCard.checkedChanged.connect(self._onAlwaysAimChanged)
        self.keepDetectingCard.checkedChanged.connect(self._onKeepDetectingChanged)
        self.idleDetectEnableCard.checkedChanged.connect(self._onIdleDetectEnableChanged)
        self.idleDetectIntervalCard.valueChanged.connect(self._onIdleDetectIntervalChanged)
        self.singleTargetCard.checkedChanged.connect(self._onSingleTargetChanged)

        self.skipLetterboxCard.checkedChanged.connect(self._onSkipLetterboxChanged)
        self.cudaIoBindingCard.checkedChanged.connect(self._onCudaIoBindingChanged)
        self.frameSkipCard.checkedChanged.connect(self._onFrameSkipChanged)
        self.frameSkipThresholdCard.valueChanged.connect(self._onFrameSkipThresholdChanged)

    # ──────────────────────────────────────────────
    # Config load
    # ──────────────────────────────────────────────

    def _loadFromConfig(self):
        if not self._config:
            return
        self._isLoadingConfig = True
        try:
            self.fovCard.setValue(self._config.fov_size)
            self.fovFollowCard.setChecked(self._config.fov_follow_mouse)
            self.fovCircleCard.setChecked(bool(getattr(self._config, 'fov_circle_filter_enabled', False)))
            self.detectRangeCard.setValue(self._config.detect_range_size)

            interval_ms = int(self._config.detect_interval * 1000)
            self.detectIntervalCard.setValue(interval_ms)
            confidence_pct = int(self._config.min_confidence * 100)
            self.confidenceCard.setValue(confidence_pct)
            self.semanticFilterCard.setChecked(bool(getattr(self._config, 'detect_semantic_filter_enabled', False)))

            self.alwaysAimCard.setChecked(getattr(self._config, 'always_aim', False))
            self.keepDetectingCard.setChecked(getattr(self._config, 'keep_detecting', False))
            self.idleDetectEnableCard.setChecked(getattr(self._config, 'idle_detect_enabled', True))
            idle_ms = int(getattr(self._config, 'idle_detect_interval', 0.05) * 1000)
            self.idleDetectIntervalCard.setValue(max(5, min(500, idle_ms)))
            self.singleTargetCard.setChecked(getattr(self._config, 'single_target_mode', False))

            self.skipLetterboxCard.setChecked(bool(getattr(self._config, 'skip_letterbox', False)))
            self.cudaIoBindingCard.setChecked(bool(getattr(self._config, 'cuda_io_binding_enabled', False)))
            self.frameSkipCard.setChecked(bool(getattr(self._config, 'frame_skip_enabled', False)))
            self.frameSkipThresholdCard.setValue(int(getattr(self._config, 'frame_skip_threshold', 2.0) * 10))

            # Apply initial screenshot-method effect on fov_follow visibility
            method = getattr(self._config, 'screenshot_method', 'mss')
            self._applyScreenshotMethodEffect(method)
        finally:
            self._isLoadingConfig = False

        self._updateFovCircleVisibility()
        self._updateProviderDependentWidgets()

    # ──────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────

    def _applyScreenshotMethodEffect(self, method: str):
        """Called by capture page (and on load) to sync fov_follow visibility."""
        is_external = method in ('uvc', 'ndi')
        self.fovFollowCard.setVisible(not is_external)
        if is_external and self._config:
            self._config.fov_follow_mouse = False
            self.fovFollowCard.setChecked(False)

    def _updateFovCircleVisibility(self):
        """Hide Skip Letterbox Padding when Circular FOV is active (they conflict)."""
        circle_on = bool(self.fovCircleCard.isChecked())
        self.skipLetterboxCard.setVisible(not circle_on)
        if circle_on and self._config:
            self._config.skip_letterbox = False
            self.skipLetterboxCard.setChecked(False)

    def _updateProviderDependentWidgets(self):
        """Show/hide and auto-configure CUDA IO Binding based on the active ORT provider."""
        if not self._config:
            return
        provider = getattr(self._config, 'current_provider', '')
        is_trt = provider == 'TensorrtExecutionProvider'
        is_cuda = provider == 'CUDAExecutionProvider'
        hide = not (is_trt or is_cuda)
        self.cudaIoBindingCard.setVisible(not hide)
        if is_trt and not bool(getattr(self._config, 'cuda_io_binding_enabled', False)):
            self._config.cuda_io_binding_enabled = True
            self.cudaIoBindingCard.setChecked(True)

    def _notifyKeysPageVisibility(self):
        """Tell the keys page to refresh MAKCU card visibility."""
        try:
            win = self.window()
            if hasattr(win, 'keysInterface') and hasattr(win.keysInterface, '_refreshMakcuVisibility'):
                win.keysInterface._refreshMakcuVisibility()
        except Exception:
            pass

    # ──────────────────────────────────────────────
    # Callbacks
    # ──────────────────────────────────────────────

    def _onFovChanged(self, value):
        if self._config:
            self._config.fov_size = value

    def _onFovFollowChanged(self, checked):
        if self._config:
            self._config.fov_follow_mouse = checked

    def _onFovCircleChanged(self, checked):
        if self._config:
            self._config.fov_circle_filter_enabled = bool(checked)
        self._updateFovCircleVisibility()

    def _onDetectRangeChanged(self, value):
        if self._config:
            self._config.detect_range_size = value

    def _onDetectIntervalChanged(self, value):
        if self._config:
            self._config.detect_interval = value / 1000.0
            if getattr(self._config, 'auto_match_fps', False):
                self._config.screenshot_interval = self._config.detect_interval
                try:
                    cap = self.window().captureInterface
                    cap.screenshotIntervalCard.setValue(value)
                except Exception:
                    pass

    def _onConfidenceChanged(self, value):
        if self._config:
            self._config.min_confidence = value / 100.0

    def _onSemanticFilterChanged(self, checked):
        if self._config:
            self._config.detect_semantic_filter_enabled = bool(checked)

    def _onAlwaysAimChanged(self, checked):
        if self._config:
            self._config.always_aim = checked
            if checked:
                self._config.idle_detect_enabled = False
                self.idleDetectEnableCard.setChecked(False)
        self._notifyKeysPageVisibility()

    def _onKeepDetectingChanged(self, checked):
        if self._config:
            self._config.keep_detecting = checked
        self._notifyKeysPageVisibility()

    def _onIdleDetectEnableChanged(self, checked):
        if self._config:
            self._config.idle_detect_enabled = checked

    def _onIdleDetectIntervalChanged(self, value):
        if self._config:
            self._config.idle_detect_interval = value / 1000.0

    def _onSingleTargetChanged(self, checked):
        if self._config:
            self._config.single_target_mode = checked

    def _onSkipLetterboxChanged(self, checked):
        if self._config:
            self._config.skip_letterbox = bool(checked)

    def _onCudaIoBindingChanged(self, checked):
        if self._config:
            self._config.cuda_io_binding_enabled = bool(checked)

    def _onFrameSkipChanged(self, checked):
        if self._config:
            self._config.frame_skip_enabled = bool(checked)

    def _onFrameSkipThresholdChanged(self, value):
        if self._config:
            self._config.frame_skip_threshold = value / 10.0

    # ──────────────────────────────────────────────
    # Retranslate
    # ──────────────────────────────────────────────

    def retranslateUi(self):
        super().retranslateUi()

        self.fovGroup.titleLabel.setText(t("fov_and_detect_range"))
        self.generalGroup.titleLabel.setText(t("general_params"))
        self.inferPerfGroup.titleLabel.setText(t("inference_performance", "Inference Performance"))

        self.fovCard.titleLabel.setText(t("fov_size"))
        self.fovFollowCard.titleLabel.setText(t("fov_follow_mouse"))
        self.detectRangeCard.titleLabel.setText(t("detect_range_size"))
        self.detectRangeCard.contentLabel.setText(t("detect_range_note"))

        self.detectIntervalCard.titleLabel.setText(t("detect_interval"))
        self.confidenceCard.titleLabel.setText(t("min_confidence"))
        self.semanticFilterCard.titleLabel.setText(t("semantic_filter_enabled", "Semantic FP Filter"))
        self.alwaysAimCard.titleLabel.setText(t("always_aim"))
        self.keepDetectingCard.titleLabel.setText(t("keep_detecting"))
        self.idleDetectEnableCard.titleLabel.setText(t("idle_detect_enabled"))
        self.idleDetectIntervalCard.titleLabel.setText(t("idle_detect_interval"))
        self.singleTargetCard.titleLabel.setText(t("single_target_mode"))

        self.skipLetterboxCard.titleLabel.setText(t("skip_letterbox_label"))
        self.skipLetterboxCard.contentLabel.setText(t("skip_letterbox_desc"))
        self.frameSkipCard.titleLabel.setText(t("frame_skip_enabled", "Frame Skip Gate"))
        self.frameSkipCard.contentLabel.setText(t("frame_skip_desc", "Skip inference when the capture region hasn't changed significantly."))
        self.frameSkipThresholdCard.titleLabel.setText(t("frame_skip_threshold", "Skip Threshold"))
