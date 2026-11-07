# inference_page.py
"""Inference Page - FOV, Detection, General Parameters, Inference Performance"""

from PyQt6.QtCore import Qt
from qfluentwidgets import (
    SettingCardGroup, SwitchSettingCard,
    FluentIcon, SettingCard, SegmentedWidget, ComboBox,
)
from ..components.slider_spin_card import SliderSpinCard, SliderLabelCard, SliderDoubleSpinCard

from ..base_page import BasePage
from ..language_manager import t


_FOV_MODES = ("fixed", "adaptive", "dynamic")


def _tip(card, text):
    """Description line plus a hover tooltip carrying the same explanation."""
    card.setToolTip(text)


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

        # Independent from fovCard/fov_size — equal by default (a square,
        # matching pre-rectangle-support behavior), but can diverge to make
        # the FOV a rectangle (or, with fovCircleCard on, an ellipse). See
        # ai_loop_utils.filter_boxes_by_fov().
        self.fovHeightCard = SliderSpinCard(
            FluentIcon.ZOOM,
            t("fov_height", "FOV Height"),
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
            t("fov_circle_filter_desc", "Only track targets inside the FOV ellipse — a circle when Width equals Height — instead of the full rectangular region"),
            parent=self.fovGroup
        )

        # Shrinks the FOV to fovMinSizeCard's percentage (both axes, so a
        # rectangle/ellipse keeps its aspect ratio) while a target is
        # locked, so a random target near the edge of the normally-larger
        # FOV can't steal the lock mid-engagement. The two value cards are
        # only meaningful while this is on — see _updateFovReduceVisibility().
        self.fovShrinkGroup = SettingCardGroup(t("fov_shrinking", "FOV Shrinking"), self.scrollWidget)
        self.advFovGroup = SettingCardGroup(t("advanced_fov", "Advanced FOV"), self.scrollWidget)
        self.acqTrackGroup = SettingCardGroup(t("target_acq_tracking", "Target Acquisition & Tracking"), self.scrollWidget)

        self.fovReduceEnableCard = SwitchSettingCard(
            FluentIcon.ZOOM,
            t("fov_reduce_on_target", "Reduce FOV on Active Target"),
            t("fov_reduce_on_target_desc", "Temporarily shrink the FOV while a target is locked, so a random target near the edge of the FOV can't steal the lock mid-engagement."),
            parent=self.fovShrinkGroup
        )

        self.fovModeCombo = ComboBox()
        self.fovModeCombo.addItems([
            t("fov_mode_fixed", "Fixed"),
            t("fov_mode_adaptive", "Adaptive"),
            t("fov_mode_dynamic", "Dynamic"),
        ])
        self.fovModeCombo.setMinimumWidth(130)
        self.fovModeCard = SettingCard(
            FluentIcon.ZOOM,
            t("fov_shrink_mode", "Shrink Mode"),
            self._fovModeDesc(),
            self.fovShrinkGroup
        )
        self.fovModeCard.hBoxLayout.addWidget(self.fovModeCombo, 0)
        self.fovModeCard.hBoxLayout.addSpacing(16)

        self.fovDynSafeCard = SwitchSettingCard(
            FluentIcon.ZOOM,
            t("fov_dynamic_safeguards", "Dynamic Safeguards"),
            t("fov_dynamic_safeguards_desc", "Dynamic mode only. Never shrink past what the target's distance from the FOV edge allows, and don't trust speed/direction until the lock has settled (about 0.5 s)."),
            parent=self.fovShrinkGroup
        )

        self.fovMinSizeCard = SliderSpinCard(
            FluentIcon.ZOOM,
            t("fov_min_size_pct", "FOV Min Size"),
            1, 100,
            suffix="%",
            description=t("fov_min_size_pct_desc", "How much to shrink the FOV to, as a percentage of FOV Size/Height, while a target is locked"),
            parent=self.fovShrinkGroup
        )

        self.fovMinSizeDurationCard = SliderDoubleSpinCard(
            FluentIcon.HISTORY,
            t("fov_min_size_duration", "FOV Min Size Duration"),
            0.0, 10.0,
            decimals=1,
            step=0.1,
            suffix="s",
            description=t("fov_min_size_duration_desc", "How long it takes the FOV to gradually shrink down to FOV Min Size after a target is acquired, then holds there for the rest of the engagement (0 = shrinks instantly)"),
            parent=self.fovShrinkGroup
        )

        # --- Advanced FOV ---
        self.fovExpandDelayCard = SliderSpinCard(
            FluentIcon.HISTORY,
            t("fov_expand_delay", "FOV Expansion Delay"),
            0, 2000, suffix="ms",
            description=t("fov_expand_delay_desc", "After the lock is released, wait this long before the FOV starts growing back. Controls WHEN recovery begins."),
            parent=self.advFovGroup
        )
        self.fovRecoveryCard = SliderSpinCard(
            FluentIcon.SPEED_HIGH,
            t("fov_recovery_duration", "FOV Recovery Duration"),
            0, 3000, suffix="ms",
            description=t("fov_recovery_duration_desc", "How long the FOV takes to grow from its minimum back to full size (also the speed limit for Adaptive/Dynamic expansion). Controls HOW FAST it recovers. 0 = instant."),
            parent=self.advFovGroup
        )
        self.minLockCard = SliderSpinCard(
            FluentIcon.PIN,
            t("min_lock_duration", "Minimum Target Lock Duration"),
            0, 2000, suffix="ms",
            description=t("min_lock_duration_desc", "A lock can't be swapped for another target sooner than this after it was acquired. Needs Sticky Target Lock (Aim page). Never delays releasing a target that is really gone."),
            parent=self.advFovGroup
        )

        # --- Target Acquisition & Tracking ---
        # Acquisition Confidence is the existing "Min Confidence" card below
        # (General Parameters); it is not duplicated here.
        self.stickyHintCard = SettingCard(
            FluentIcon.INFO,
            t("tracking_needs_sticky", "Needs Sticky Target Lock"),
            t("tracking_needs_sticky_desc", "Turn on Sticky Target Lock (Aim page → Target Tracking) to use these settings. Lock Persistence is set there too."),
            self.acqTrackGroup
        )
        self.trackConfCard = SliderSpinCard(
            FluentIcon.CERTIFICATE,
            t("tracking_confidence", "Tracking Confidence"),
            0, 100, suffix="%",
            description=t("tracking_confidence_desc", "Minimum confidence to KEEP an existing lock. A new target still needs Min Confidence (General Parameters), so a lock survives brief confidence dips. 0 = same as Min Confidence."),
            parent=self.acqTrackGroup
        )
        self.switchThresholdCard = SliderSpinCard(
            FluentIcon.SYNC,
            t("target_switch_threshold", "Target Switch Threshold"),
            0, 100, suffix="%",
            description=t("target_switch_threshold_desc", "How much better (as a % of the priority score) another target must be before it replaces the current one. 100% = never switch while the current target is visible."),
            parent=self.acqTrackGroup
        )
        self.reacquireRadiusCard = SliderSpinCard(
            FluentIcon.ZOOM_IN,
            t("reacquire_radius", "Reacquisition Radius"),
            0, 1000, suffix="px",
            description=t("reacquire_radius_desc", "A target reappearing within this distance of where the lost one was last seen (extrapolated by its velocity) is treated as the same target."),
            parent=self.acqTrackGroup
        )
        self.lostTimeoutCard = SliderSpinCard(
            FluentIcon.HISTORY,
            t("lost_target_timeout", "Lost Target Timeout"),
            0, 10000, suffix="ms",
            description=t("lost_target_timeout_desc", "Longest time without a valid detection before the lock is released and the target forgotten. Aim is held only for Lock Persistence; the target stays reacquirable until this elapses."),
            parent=self.acqTrackGroup
        )

        self.detectRangeCard = SliderSpinCard(
            FluentIcon.FULL_SCREEN,
            t("detect_range_size"),
            100, 1080,
            description=t("detect_range_note"),
            parent=self.fovGroup
        )

        for _c in (self.fovReduceEnableCard, self.fovModeCard, self.fovDynSafeCard, self.fovMinSizeCard, self.fovMinSizeDurationCard,
                   self.fovExpandDelayCard, self.fovRecoveryCard, self.minLockCard, self.trackConfCard,
                   self.switchThresholdCard, self.reacquireRadiusCard, self.lostTimeoutCard, self.stickyHintCard):
            _tip(_c, _c.contentLabel.text())

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

        self.secondInferSegment = SegmentedWidget()
        self.secondInferSegment.addItem("off",     t("weapon_detect_off", "Off"))
        self.secondInferSegment.addItem("v1_ocr",  t("weapon_detect_v1_ocr", "V1 OCR"))
        self.secondInferSegment.addItem("v2_onnx", t("weapon_detect_v2_onnx", "V2 ONNX"))
        self.secondInferCard = SettingCard(
            FluentIcon.DOCUMENT,
            t("weapon_detection_method", "Weapon Detection Method"),
            t("weapon_detection_method_desc", "Off / OCR (PaddleOCR) / ONNX (HUD weapon detector)"),
            self.generalGroup,
        )
        self.secondInferCard.hBoxLayout.addWidget(self.secondInferSegment, 0, Qt.AlignmentFlag.AlignRight)
        self.secondInferCard.hBoxLayout.addSpacing(16)

        self.hudConfidenceCard = SliderDoubleSpinCard(
            FluentIcon.FILTER,
            t("weapon_detection_threshold", "Weapon Detection Threshold"),
            0.01, 1.0,
            decimals=2,
            step=0.05,
            description=t("weapon_detection_threshold_desc", "Minimum confidence for ONNX weapon detections"),
            parent=self.generalGroup,
        )

        # === Inference Performance ===
        self.inferPerfGroup = SettingCardGroup(t("inference_performance", "Inference Performance"), self.scrollWidget)

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
        self.fovGroup.addSettingCard(self.fovHeightCard)
        self.fovGroup.addSettingCard(self.fovFollowCard)
        self.fovGroup.addSettingCard(self.fovCircleCard)
        self.fovGroup.addSettingCard(self.detectRangeCard)
        self.addContent(self.fovGroup)

        self.fovShrinkGroup.addSettingCard(self.fovReduceEnableCard)
        self.fovShrinkGroup.addSettingCard(self.fovModeCard)
        self.fovShrinkGroup.addSettingCard(self.fovDynSafeCard)
        self.fovShrinkGroup.addSettingCard(self.fovMinSizeCard)
        self.fovShrinkGroup.addSettingCard(self.fovMinSizeDurationCard)
        self.addContent(self.fovShrinkGroup)

        self.advFovGroup.addSettingCard(self.fovExpandDelayCard)
        self.advFovGroup.addSettingCard(self.fovRecoveryCard)
        self.advFovGroup.addSettingCard(self.minLockCard)
        self.addContent(self.advFovGroup)

        self.acqTrackGroup.addSettingCard(self.stickyHintCard)
        self.acqTrackGroup.addSettingCard(self.trackConfCard)
        self.acqTrackGroup.addSettingCard(self.switchThresholdCard)
        self.acqTrackGroup.addSettingCard(self.reacquireRadiusCard)
        self.acqTrackGroup.addSettingCard(self.lostTimeoutCard)
        self.addContent(self.acqTrackGroup)

        self.generalGroup.addSettingCard(self.detectIntervalCard)
        self.generalGroup.addSettingCard(self.confidenceCard)
        self.generalGroup.addSettingCard(self.semanticFilterCard)
        self.generalGroup.addSettingCard(self.idleDetectEnableCard)
        self.generalGroup.addSettingCard(self.idleDetectIntervalCard)
        self.generalGroup.addSettingCard(self.singleTargetCard)
        self.generalGroup.addSettingCard(self.secondInferCard)
        self.generalGroup.addSettingCard(self.hudConfidenceCard)
        self.addContent(self.generalGroup)

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
        self.fovHeightCard.valueChanged.connect(self._onFovHeightChanged)
        self.fovFollowCard.checkedChanged.connect(self._onFovFollowChanged)
        self.fovCircleCard.checkedChanged.connect(self._onFovCircleChanged)
        self.fovReduceEnableCard.checkedChanged.connect(self._onFovReduceEnableChanged)
        self.fovMinSizeCard.valueChanged.connect(self._onFovMinSizeChanged)
        self.fovMinSizeDurationCard.valueChanged.connect(self._onFovMinSizeDurationChanged)
        self.fovModeCombo.currentIndexChanged.connect(self._onFovModeChanged)
        self.fovDynSafeCard.checkedChanged.connect(lambda v: self._setCfg('fov_dynamic_safeguards', bool(v)))
        self.fovExpandDelayCard.valueChanged.connect(lambda v: self._setCfg('fov_expand_delay_ms', int(v)))
        self.fovRecoveryCard.valueChanged.connect(lambda v: self._setCfg('fov_recovery_duration_ms', int(v)))
        self.minLockCard.valueChanged.connect(lambda v: self._setCfg('min_lock_duration_ms', int(v)))
        self.trackConfCard.valueChanged.connect(lambda v: self._setCfg('tracking_confidence', v / 100.0))
        self.switchThresholdCard.valueChanged.connect(lambda v: self._setCfg('target_switch_threshold', v / 100.0))
        self.reacquireRadiusCard.valueChanged.connect(lambda v: self._setCfg('reacquire_radius_px', int(v)))
        self.lostTimeoutCard.valueChanged.connect(lambda v: self._setCfg('lost_target_timeout_ms', int(v)))
        self.detectRangeCard.valueChanged.connect(self._onDetectRangeChanged)

        self.detectIntervalCard.valueChanged.connect(self._onDetectIntervalChanged)
        self.confidenceCard.valueChanged.connect(self._onConfidenceChanged)
        self.semanticFilterCard.checkedChanged.connect(self._onSemanticFilterChanged)
        self.idleDetectEnableCard.checkedChanged.connect(self._onIdleDetectEnableChanged)
        self.idleDetectIntervalCard.valueChanged.connect(self._onIdleDetectIntervalChanged)
        self.singleTargetCard.checkedChanged.connect(self._onSingleTargetChanged)

        self.cudaIoBindingCard.checkedChanged.connect(self._onCudaIoBindingChanged)
        self.frameSkipCard.checkedChanged.connect(self._onFrameSkipChanged)
        self.frameSkipThresholdCard.valueChanged.connect(self._onFrameSkipThresholdChanged)

        self.secondInferSegment.currentItemChanged.connect(self._onSecondInferChanged)
        self.hudConfidenceCard.valueChanged.connect(self._onHudConfidenceChanged)

    # ──────────────────────────────────────────────
    # Config load
    # ──────────────────────────────────────────────

    def _loadFromConfig(self):
        if not self._config:
            return
        self._isLoadingConfig = True
        try:
            self.fovCard.setValue(self._config.fov_size)
            self.fovHeightCard.setValue(int(getattr(self._config, 'fov_height', self._config.fov_size)))
            self.fovFollowCard.setChecked(self._config.fov_follow_mouse)
            self.fovCircleCard.setChecked(bool(getattr(self._config, 'fov_circle_filter_enabled', False)))
            fov_reduce_on = bool(getattr(self._config, 'fov_reduce_on_target_enabled', False))
            self.fovReduceEnableCard.setChecked(fov_reduce_on)
            self.fovMinSizeCard.setValue(int(getattr(self._config, 'fov_min_size_pct', 50.0)))
            self.fovMinSizeDurationCard.setValue(float(getattr(self._config, 'fov_min_size_duration', 1.0)))
            mode = str(getattr(self._config, 'fov_shrink_mode', 'adaptive'))
            self.fovModeCombo.setCurrentIndex(_FOV_MODES.index(mode) if mode in _FOV_MODES else 1)
            self.fovDynSafeCard.setChecked(bool(getattr(self._config, 'fov_dynamic_safeguards', False)))
            self.fovExpandDelayCard.setValue(int(getattr(self._config, 'fov_expand_delay_ms', 200)))
            self.fovRecoveryCard.setValue(int(getattr(self._config, 'fov_recovery_duration_ms', 400)))
            self.minLockCard.setValue(int(getattr(self._config, 'min_lock_duration_ms', 150)))
            self.trackConfCard.setValue(int(round(float(getattr(self._config, 'tracking_confidence', 0.0)) * 100)))
            self.switchThresholdCard.setValue(int(round(float(getattr(self._config, 'target_switch_threshold', 0.35)) * 100)))
            self.reacquireRadiusCard.setValue(int(getattr(self._config, 'reacquire_radius_px', 150)))
            self.lostTimeoutCard.setValue(int(getattr(self._config, 'lost_target_timeout_ms', 1500)))
            self._updateFovReduceVisibility(fov_reduce_on)
            self.detectRangeCard.setValue(self._config.detect_range_size)

            interval_ms = int(round(self._config.detect_interval * 1000))
            self.detectIntervalCard.setValue(interval_ms)
            confidence_pct = int(round(self._config.min_confidence * 100))
            self.confidenceCard.setValue(confidence_pct)
            self.semanticFilterCard.setChecked(bool(getattr(self._config, 'detect_semantic_filter_enabled', False)))

            self.idleDetectEnableCard.setChecked(getattr(self._config, 'idle_detect_enabled', True))
            idle_ms = int(round(getattr(self._config, 'idle_detect_interval', 0.05) * 1000))
            self.idleDetectIntervalCard.setValue(max(5, min(500, idle_ms)))
            self.singleTargetCard.setChecked(getattr(self._config, 'single_target_mode', False))

            self.cudaIoBindingCard.setChecked(bool(getattr(self._config, 'cuda_io_binding_enabled', False)))
            self.frameSkipCard.setChecked(bool(getattr(self._config, 'frame_skip_enabled', False)))
            self.frameSkipThresholdCard.setValue(int(round(getattr(self._config, 'frame_skip_threshold', 2.0) * 10)))

            mode = getattr(self._config, 'second_inference_mode', 'off')
            self.secondInferSegment.setCurrentItem(mode if mode in ("off", "v1_ocr", "v2_onnx") else "off")
            self.hudConfidenceCard.setValue(float(getattr(self._config, 'hud_confidence', 0.10)))
            self._updateHudConfidenceVisibility(mode)

            # Apply initial screenshot-method effect on fov_follow visibility
            method = getattr(self._config, 'screenshot_method', 'mss')
            self._applyScreenshotMethodEffect(method)
        finally:
            self._isLoadingConfig = False

        self._updateProviderDependentWidgets()

    # ──────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────

    def _updateFovReduceVisibility(self, enabled: bool):
        """Hide the min-size % / duration cards while the toggle is off —
        per the request, not just greyed out, to keep the FOV group
        uncluttered when the feature isn't in use."""
        self.fovModeCard.setVisible(enabled)
        self.fovDynSafeCard.setVisible(enabled and str(getattr(self._config, 'fov_shrink_mode', 'adaptive')) == 'dynamic')
        self.fovMinSizeCard.setVisible(enabled)
        self.fovMinSizeDurationCard.setVisible(enabled)
        # Advanced FOV only matters while the shrinking feature is on; the
        # minimum lock duration and the tracking group depend on Sticky Lock.
        self.fovExpandDelayCard.setVisible(enabled)
        self.fovRecoveryCard.setVisible(enabled)
        self._updateTrackingVisibility()

    def _updateTrackingVisibility(self):
        sticky = bool(getattr(self._config, 'sticky_lock_enabled', False)) if self._config else False
        self.minLockCard.setVisible(sticky)
        fov_on = bool(getattr(self._config, 'fov_reduce_on_target_enabled', False)) if self._config else False
        self.advFovGroup.setVisible(fov_on or sticky)   # hide the group when none of its cards apply
        self.stickyHintCard.setVisible(not sticky)
        for card in (self.trackConfCard, self.switchThresholdCard,
                     self.reacquireRadiusCard, self.lostTimeoutCard):
            card.setVisible(sticky)

    @staticmethod
    def _fovModeDesc():
        return t(
            "fov_shrink_mode_desc",
            "Fixed: always shrink to FOV Min Size. Adaptive: shrink while the target is steady, "
            "confident and near the centre; grow when it nears the edge or tracking gets shaky. "
            "Dynamic: shrink while steady; grow on fast movement, sudden direction changes or "
            "unstable detections.")

    def _setCfg(self, name, value):
        if self._config and not self._isLoadingConfig:
            setattr(self._config, name, value)

    def _onFovModeChanged(self, index):
        if self._config and not self._isLoadingConfig and 0 <= index < len(_FOV_MODES):
            self._config.fov_shrink_mode = _FOV_MODES[index]
            self.fovDynSafeCard.setVisible(bool(getattr(self._config, 'fov_reduce_on_target_enabled', False))
                                           and _FOV_MODES[index] == 'dynamic')

    def _applyScreenshotMethodEffect(self, method: str):
        """Called by capture page (and on load) to sync fov_follow visibility."""
        # udp is the same class of external-source capture as uvc/ndi — the
        # desktop cursor position is meaningless against any of these feeds,
        # so fov_follow_mouse must be force-disabled for all three, not just
        # uvc/ndi.
        is_external = method in ('uvc', 'ndi', 'udp')
        self.fovFollowCard.setVisible(not is_external)
        if is_external and self._config:
            self._config.fov_follow_mouse = False
            self.fovFollowCard.setChecked(False)

    def _updateProviderDependentWidgets(self):
        """Show/hide and auto-configure CUDA IO Binding based on the active ORT provider."""
        if not self._config:
            return
        provider = getattr(self._config, 'current_provider', '')
        backend  = str(getattr(self._config, 'inference_backend', '')).lower()
        is_trt   = provider == 'TensorrtExecutionProvider' or backend == 'tensorrt'
        is_cuda  = provider == 'CUDAExecutionProvider'     or backend == 'cuda'
        hide = not (is_trt or is_cuda)
        self.cudaIoBindingCard.setVisible(not hide)
        if is_trt and not bool(getattr(self._config, 'cuda_io_binding_enabled', False)):
            self._config.cuda_io_binding_enabled = True
            self.cudaIoBindingCard.setChecked(True)

    # ──────────────────────────────────────────────
    # Callbacks
    # ──────────────────────────────────────────────

    def _onFovChanged(self, value):
        if self._config:
            self._config.fov_size = value

    def _onFovHeightChanged(self, value):
        if self._config:
            self._config.fov_height = value

    def _onFovFollowChanged(self, checked):
        if self._config:
            self._config.fov_follow_mouse = checked

    def _onFovCircleChanged(self, checked):
        if self._config:
            self._config.fov_circle_filter_enabled = bool(checked)

    def _onFovReduceEnableChanged(self, checked):
        if self._config:
            self._config.fov_reduce_on_target_enabled = bool(checked)
        self._updateFovReduceVisibility(bool(checked))

    def _onFovMinSizeChanged(self, value):
        if self._config:
            self._config.fov_min_size_pct = float(value)

    def _onFovMinSizeDurationChanged(self, value):
        if self._config:
            self._config.fov_min_size_duration = float(value)

    def _onDetectRangeChanged(self, value):
        if self._config:
            self._config.detect_range_size = value

    def _onDetectIntervalChanged(self, value):
        if self._config:
            self._config.detect_interval = value / 1000.0

    def _onConfidenceChanged(self, value):
        if self._config:
            self._config.min_confidence = value / 100.0

    def _onSemanticFilterChanged(self, checked):
        if self._config:
            self._config.detect_semantic_filter_enabled = bool(checked)

    def _onIdleDetectEnableChanged(self, checked):
        if self._config:
            self._config.idle_detect_enabled = checked

    def _onIdleDetectIntervalChanged(self, value):
        if self._config:
            self._config.idle_detect_interval = value / 1000.0

    def _onSingleTargetChanged(self, checked):
        if self._config:
            self._config.single_target_mode = checked

    def _onCudaIoBindingChanged(self, checked):
        if self._config:
            self._config.cuda_io_binding_enabled = bool(checked)

    def _onFrameSkipChanged(self, checked):
        if self._config:
            self._config.frame_skip_enabled = bool(checked)

    def _onFrameSkipThresholdChanged(self, value):
        if self._config:
            self._config.frame_skip_threshold = value / 10.0

    def _onSecondInferChanged(self, key: str):
        if self._config:
            self._config.second_inference_mode = key
        self._updateHudConfidenceVisibility(key)

    def _updateHudConfidenceVisibility(self, mode: str):
        # OCR mode has no numeric detection threshold to tune — only the
        # ONNX HUD weapon detector uses hud_confidence.
        self.hudConfidenceCard.setVisible(mode != "v1_ocr")

    def _onHudConfidenceChanged(self, value: float):
        if self._config:
            self._config.hud_confidence = float(value)

    # ──────────────────────────────────────────────
    # Retranslate
    # ──────────────────────────────────────────────

    def retranslateUi(self):
        super().retranslateUi()

        self.fovGroup.titleLabel.setText(t("fov_and_detect_range"))
        self.generalGroup.titleLabel.setText(t("general_params"))
        self.inferPerfGroup.titleLabel.setText(t("inference_performance", "Inference Performance"))

        self.fovCard.titleLabel.setText(t("fov_size"))
        self.fovHeightCard.titleLabel.setText(t("fov_height", "FOV Height"))
        self.fovFollowCard.titleLabel.setText(t("fov_follow_mouse"))
        self.fovCircleCard.titleLabel.setText(t("fov_circle_filter", "Circular FOV Filter"))
        self.fovCircleCard.contentLabel.setText(t("fov_circle_filter_desc", "Only track targets inside the FOV ellipse — a circle when Width equals Height — instead of the full rectangular region"))
        self.fovReduceEnableCard.titleLabel.setText(t("fov_reduce_on_target", "Reduce FOV on Active Target"))
        self.fovReduceEnableCard.contentLabel.setText(t("fov_reduce_on_target_desc", "Temporarily shrink the FOV while a target is locked, so a random target near the edge of the FOV can't steal the lock mid-engagement."))
        self.fovMinSizeCard.titleLabel.setText(t("fov_min_size_pct", "FOV Min Size"))
        self.fovMinSizeCard.contentLabel.setText(t("fov_min_size_pct_desc", "How much to shrink the FOV to, as a percentage of FOV Size/Height, while a target is locked"))
        self.fovMinSizeDurationCard.titleLabel.setText(t("fov_min_size_duration", "FOV Min Size Duration"))
        self.fovMinSizeDurationCard.contentLabel.setText(t("fov_min_size_duration_desc", "How long it takes the FOV to gradually shrink down to FOV Min Size after a target is acquired, then holds there for the rest of the engagement (0 = shrinks instantly)"))
        self.detectRangeCard.titleLabel.setText(t("detect_range_size"))
        self.detectRangeCard.contentLabel.setText(t("detect_range_note"))

        self.fovShrinkGroup.titleLabel.setText(t("fov_shrinking", "FOV Shrinking"))
        self.advFovGroup.titleLabel.setText(t("advanced_fov", "Advanced FOV"))
        self.acqTrackGroup.titleLabel.setText(t("target_acq_tracking", "Target Acquisition & Tracking"))
        self.fovModeCard.titleLabel.setText(t("fov_shrink_mode", "Shrink Mode"))
        self.fovModeCard.contentLabel.setText(self._fovModeDesc())
        self.fovDynSafeCard.titleLabel.setText(t("fov_dynamic_safeguards", "Dynamic Safeguards"))
        self.fovDynSafeCard.contentLabel.setText(t("fov_dynamic_safeguards_desc", "Dynamic mode only. Never shrink past what the target's distance from the FOV edge allows, and don't trust speed/direction until the lock has settled (about 0.5 s)."))
        for i, (key, default) in enumerate((("fov_mode_fixed", "Fixed"), ("fov_mode_adaptive", "Adaptive"), ("fov_mode_dynamic", "Dynamic"))):
            self.fovModeCombo.setItemText(i, t(key, default))
        for card, key, title, desc in (
            (self.fovExpandDelayCard, "fov_expand_delay", "FOV Expansion Delay", "After the lock is released, wait this long before the FOV starts growing back. Controls WHEN recovery begins."),
            (self.fovRecoveryCard, "fov_recovery_duration", "FOV Recovery Duration", "How long the FOV takes to grow from its minimum back to full size (also the speed limit for Adaptive/Dynamic expansion). Controls HOW FAST it recovers. 0 = instant."),
            (self.minLockCard, "min_lock_duration", "Minimum Target Lock Duration", "A lock can't be swapped for another target sooner than this after it was acquired. Needs Sticky Target Lock (Aim page). Never delays releasing a target that is really gone."),
            (self.stickyHintCard, "tracking_needs_sticky", "Needs Sticky Target Lock", "Turn on Sticky Target Lock (Aim page → Target Tracking) to use these settings. Lock Persistence is set there too."),
            (self.trackConfCard, "tracking_confidence", "Tracking Confidence", "Minimum confidence to KEEP an existing lock. A new target still needs Min Confidence (General Parameters), so a lock survives brief confidence dips. 0 = same as Min Confidence."),
            (self.switchThresholdCard, "target_switch_threshold", "Target Switch Threshold", "How much better (as a % of the priority score) another target must be before it replaces the current one. 100% = never switch while the current target is visible."),
            (self.reacquireRadiusCard, "reacquire_radius", "Reacquisition Radius", "A target reappearing within this distance of where the lost one was last seen (extrapolated by its velocity) is treated as the same target."),
            (self.lostTimeoutCard, "lost_target_timeout", "Lost Target Timeout", "Longest time without a valid detection before the lock is released and the target forgotten. Aim is held only for Lock Persistence; the target stays reacquirable until this elapses."),
        ):
            card.titleLabel.setText(t(key, title))
            card.contentLabel.setText(t(key + "_desc" if key != "tracking_needs_sticky" else "tracking_needs_sticky_desc", desc))
            _tip(card, card.contentLabel.text())

        self.detectIntervalCard.titleLabel.setText(t("detect_interval"))
        self.confidenceCard.titleLabel.setText(t("min_confidence"))
        self.semanticFilterCard.titleLabel.setText(t("semantic_filter_enabled", "Semantic FP Filter"))
        self.semanticFilterCard.contentLabel.setText(t("semantic_filter_desc", "Discard trees, vehicles, and HUD elements by class name and geometry"))
        self.idleDetectEnableCard.titleLabel.setText(t("idle_detect_enabled"))
        self.idleDetectIntervalCard.titleLabel.setText(t("idle_detect_interval"))
        self.singleTargetCard.titleLabel.setText(t("single_target_mode"))
        self.secondInferCard.titleLabel.setText(t("weapon_detection_method", "Weapon Detection Method"))
        self.secondInferCard.contentLabel.setText(t("weapon_detection_method_desc", "Off / OCR (PaddleOCR) / ONNX (HUD weapon detector)"))
        self.secondInferSegment.setItemText("off", t("weapon_detect_off", "Off"))
        self.secondInferSegment.setItemText("v1_ocr", t("weapon_detect_v1_ocr", "V1 OCR"))
        self.secondInferSegment.setItemText("v2_onnx", t("weapon_detect_v2_onnx", "V2 ONNX"))
        self.hudConfidenceCard.titleLabel.setText(t("weapon_detection_threshold", "Weapon Detection Threshold"))
        self.hudConfidenceCard.contentLabel.setText(t("weapon_detection_threshold_desc", "Minimum confidence for ONNX weapon detections"))

        self.cudaIoBindingCard.titleLabel.setText(t("cuda_io_binding", "CUDA IO Binding"))
        self.cudaIoBindingCard.contentLabel.setText(t("cuda_io_binding_desc", "Zero-copy GPU inference. Effective only with CUDA or TensorRT backend."))
        self.frameSkipCard.titleLabel.setText(t("frame_skip_enabled", "Frame Skip Gate"))
        self.frameSkipCard.contentLabel.setText(t("frame_skip_desc", "Skip inference when the capture region hasn't changed significantly."))
        self.frameSkipThresholdCard.titleLabel.setText(t("frame_skip_threshold", "Skip Threshold"))
        self.frameSkipThresholdCard.contentLabel.setText(t("frame_skip_threshold_desc", "Avg pixel diff below this value triggers skip (higher = more skipping)"))
