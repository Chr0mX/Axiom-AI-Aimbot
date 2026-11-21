# inference_page.py
"""Inference Page - FOV, Detection, General Parameters, Inference Performance"""

from PyQt6.QtCore import Qt
from core.config import EXTERNAL_CAPTURE_METHODS
from core.detect_mode import MODES as _DETECT_MODES, apply_mode, mode_from_config, reduced_rate_ignored
from qfluentwidgets import (
    SettingCardGroup, SwitchSettingCard,
    FluentIcon, SettingCard, SegmentedWidget, ComboBox,
)
from ..components.slider_spin_card import SliderSpinCard, SliderLabelCard, SliderDoubleSpinCard

from ..base_page import BasePage
from ..language_manager import t


_FOV_MODES = ("fixed", "adaptive", "dynamic", "anchor")

# "When Not Aiming" (core/detect_mode.py): labels in MODES order.
_DETECT_MODE_LABELS = ("Full speed", "Reduced speed", "Stop detecting")
_DETECT_MODE_DESC = ("What detection does while the aim key is not held. Full speed: keep detecting. "
                     "Reduced speed: detect only at the interval below to save resources. "
                     "Stop detecting: nothing runs until you aim.")
_DETECT_MODE_IGNORED = "(Reduced speed is ignored while Always Aim or Always Auto-Fire is on.)"
# "disabled" (the master switch covers it) and "linear" (= Velocity with Acceleration 0) are retired;
# config.migrate_anchor_settings() turns old values into these on load.
_ANCHOR_MODES = ("velocity_accel", "smoothed", "adaptive")


# Anchor FOV (core/anchor_fov.py) cards: one table drives construction, wiring, loading,
# visibility and retranslation.  kind: combo / slider / switch.  For sliders `scale` converts
# the slider integer to the config value (cfg = v / scale); `ints` keeps config ints.
_AFOV_GROUPS = (
    ("afovAnchorGroup", "afov_group_anchor", "Anchor FOV: Anchor"),
    ("afovMoveGroup", "afov_group_move", "Anchor FOV: Movement"),
    ("afovPredGroup", "afov_group_pred", "Anchor FOV: Look-ahead"),
    ("afovLostGroup", "afov_group_lost", "Anchor FOV: Lost Anchor"),
    ("afovRadiusGroup", "afov_group_radius", "Anchor FOV: Size"),
)
_AFOV_CARDS = (
    # attr, group, kind, config field, i18n key, title, description, extra
    ("afovSourceCard", "afovAnchorGroup", "combo", "afov_source", "afov_source", "Anchor Source",
     "Detection: the detected object that is the current target. Tracker: the tracking system's position for the lock (smoother, survives short detection gaps). Manual: a fixed point.",
     (("detection", "afov_source_detection", "Detection"), ("tracker", "afov_source_tracker", "Tracker"),
      ("manual", "afov_source_manual", "Manual"))),
    ("afovPositionCard", "afovAnchorGroup", "combo", "afov_position", "afov_position", "Anchor Position",
     "Where on the box the anchor sits; fine-tune it with the offsets below.",
     (("center", "afov_pos_center", "Bounding Box Centre"), ("top", "afov_pos_top", "Top Centre"),
      ("bottom", "afov_pos_bottom", "Bottom Centre"))),
    ("afovOffsetXCard", "afovAnchorGroup", "slider", "afov_offset_x", "afov_offset_x", "Offset X",
     "Horizontal shift of the anchor, in pixels. With the Manual source: the distance from the crosshair.", (-1000, 1000, "px", 1)),
    ("afovOffsetYCard", "afovAnchorGroup", "slider", "afov_offset_y", "afov_offset_y", "Offset Y",
     "Vertical shift of the anchor, in pixels (positive = down). With the Manual source: the distance from the crosshair.", (-1000, 1000, "px", 1)),
    ("afovFollowCard", "afovMoveGroup", "slider", "afov_follow_speed", "afov_follow_speed", "Follow Speed",
     "How quickly the FOV centre closes in on the anchor (per second). Higher = tighter, lower = floatier.",
     (1, 100, "/s", 1)),
    ("afovSmoothCard", "afovMoveGroup", "slider", "afov_smoothing", "afov_smoothing", "Smoothing",
     "Low-pass on the anchor before the FOV follows it, to remove detection jitter. 0 = off.", (0, 100, "%", 100)),
    ("afovMaxSpeedCard", "afovMoveGroup", "slider", "afov_max_speed", "afov_max_speed", "Maximum Movement Speed",
     "Fastest the FOV centre may travel, in pixels per second (0 = unlimited).", (0, 20000, "px/s", 1)),
    ("afovPredictCard", "afovPredGroup", "switch", "afov_predict_enabled", "afov_predict_enabled", "Enable Look-ahead",
     "Lead the anchor by the tracker's velocity so the FOV arrives where the target is going. Off by default; fades out by itself when confidence drops.",
     None),
    ("afovPredTimeCard", "afovPredGroup", "slider", "afov_predict_time_ms", "afov_predict_time", "Look-ahead Time",
     "How far ahead to look.", (0, 500, "ms", 1)),
    ("afovLostCard", "afovLostGroup", "combo", "afov_lost_behavior", "afov_lost_behavior", "Lost Target Behaviour",
     "Hold: keep the last FOV position. Return to Centre: glide back to the crosshair. Search: stay and expand the FOV so a new anchor can be found.",
     (("hold", "afov_lost_hold", "Hold"), ("return", "afov_lost_return", "Return to Centre"),
      ("search", "afov_lost_search", "Search (Expand)"))),
    ("afovLostDelayCard", "afovLostGroup", "slider", "afov_lost_delay_ms", "afov_lost_delay", "Lost Delay",
     "The anchor must be gone this long before the behaviour above starts (brief detection gaps are ridden out).",
     (0, 2000, "ms", 1)),
    ("afovReacquireCard", "afovLostGroup", "combo", "afov_reacquire", "afov_reacquire", "Reacquisition Behaviour",
     "Smooth: glide to the new anchor at the normal speed. Fast: catch up quickly (still continuous, never a jump).",
     (("smooth", "afov_reacq_smooth", "Smooth"), ("fast", "afov_reacq_fast", "Fast Catch-up"))),
    ("afovMaxRadiusCard", "afovRadiusGroup", "slider", "afov_max_radius", "afov_max_radius", "Maximum Radius",
     "The most the FOV radius can grow to with Dynamic Radius or Search. The normal size is FOV Width \u00f7 2.", (1, 1000, "px", 1)),
    ("afovDynRadiusCard", "afovRadiusGroup", "switch", "afov_dynamic_radius", "afov_dynamic_radius", "Dynamic Radius",
     "Widen the FOV (up to the maximum radius) for fast or low-confidence targets.", None),
    ("afovFitCard", "afovRadiusGroup", "switch", "afov_fit_box_enabled", "afov_fit_box", "Fit FOV to Target Box While Aiming",
     "While aim is active and a target is locked, shrink the FOV to the target's detection box (never above FOV Width/Height); "
     "it grows back to the normal size when aim is released. Needs the Detection or Tracker source.", None),
    ("afovFitPadCard", "afovRadiusGroup", "slider", "afov_fit_padding_pct", "afov_fit_padding", "Fit Padding",
     "Extra room around the box, as a percentage of its size, so a moving target stays inside the FOV.", (0, 200, "%", 1)),
)

# (i18n key, English title, English description) per Anchor Prediction card —
# one table used both when the cards are built and by retranslateUi().
_ANCHOR_TEXT = {
    "anchorEnableCard": ("anchor_enabled", "Enable Box Prediction",
                         "Predict where the locked target's box will be a moment from now, and let that nudge the Adaptive/Dynamic FOV and help re-find the lock. Off = everything behaves exactly as before. The tuning below is shared with the Aim page's Anchor prediction method, which works without this switch."),
    "anchorModeCard": ("anchor_mode", "Box Prediction Mode",
                       "Velocity: the current velocity, plus acceleration by the Acceleration setting (0% = velocity only). Smoothed: smooths velocity and acceleration to cut jitter. Adaptive: smoothed, and the lead follows confidence, stability and direction changes."),
    "anchorHorizonCard": ("anchor_horizon", "Box Prediction Horizon",
                          "How far ahead to predict the box"),
    "anchorStrengthCard": ("anchor_strength", "Box Prediction Strength",
                           "Scales the whole predicted movement (100% = full)"),
    "anchorSmoothingCard": ("anchor_velocity_smoothing", "Velocity Smoothing",
                            "Higher = steadier but slower to react (Smoothed / Adaptive modes)"),
    "anchorAccelCard": ("anchor_accel_comp", "Acceleration Compensation",
                        "How much of the estimated acceleration is applied. 0 = velocity only"),
    "anchorConfCard": ("anchor_confidence_threshold", "Confidence Threshold",
                       "Below this prediction confidence nothing is extrapolated"),
    "anchorHistoryCard": ("anchor_history_len", "Prediction History Length",
                          "How many recent detections the motion estimate uses (acceleration needs 4 or more)"),
    "anchorMaxDistCard": ("anchor_max_distance_pct", "Maximum Prediction Distance",
                          "Longest predicted movement, as a % of the FOV size. 0 = no limit"),
    "anchorInflAdaptiveCard": ("anchor_fov_influence_adaptive", "Prediction Influence on Adaptive FOV",
                               "How strongly the prediction opens the Adaptive FOV early for a target heading for its edge. 0 = ignore the prediction"),
    "anchorInflDynamicCard": ("anchor_fov_influence_dynamic", "Prediction Influence on Dynamic FOV",
                              "How strongly the predicted speed/direction change feeds the Dynamic FOV. 0 = ignore the prediction"),
}


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
        self._markAdvancedCards()
        self.addAdvancedSwitch()

    def setConfig(self, config):
        self._config = config
        if hasattr(self, 'detectRangeCard') and self._config:
            max_h = max(1080, self._config.height)
            self.detectRangeCard.slider.setMaximum(max_h)
            self.detectRangeCard.spinBox.setMaximum(max_h)
        self._loadFromConfig()

    def showEvent(self, event):
        super().showEvent(event)
        if self._config and hasattr(self, 'detectModeCombo'):
            self._syncDetectModeCombo()

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
            t("fov_mode_anchor", "Anchor"),
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
            description=t("fov_min_size_pct_desc", "How much to shrink the FOV to, as a percentage of FOV Width/Height, while a target is locked"),
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

        # --- Anchor Prediction (core/anchor_prediction.py) ---
        self.anchorGroup = SettingCardGroup(t("anchor_prediction", "Anchor Box Prediction"), self.scrollWidget)

        def _at(attr, field):
            return t(_ANCHOR_TEXT[attr][0], _ANCHOR_TEXT[attr][1]) if field == 1 else \
                t(_ANCHOR_TEXT[attr][0] + "_desc", _ANCHOR_TEXT[attr][2])

        self.anchorEnableCard = SwitchSettingCard(
            FluentIcon.ROBOT, _at("anchorEnableCard", 1), _at("anchorEnableCard", 2), parent=self.anchorGroup)
        self.anchorModeCombo = ComboBox()
        self.anchorModeCombo.addItems([
            t("anchor_mode_velocity", "Velocity"),
            t("anchor_mode_smoothed", "Smoothed"),
            t("anchor_mode_adaptive", "Adaptive"),
        ])
        self.anchorModeCombo.setMinimumWidth(180)
        self.anchorModeCard = SettingCard(
            FluentIcon.ROBOT, _at("anchorModeCard", 1), _at("anchorModeCard", 2), self.anchorGroup)
        self.anchorModeCard.hBoxLayout.addWidget(self.anchorModeCombo, 0)
        self.anchorModeCard.hBoxLayout.addSpacing(16)
        self.anchorHorizonCard = SliderSpinCard(
            FluentIcon.HISTORY, _at("anchorHorizonCard", 1), 20, 500, suffix="ms",
            description=_at("anchorHorizonCard", 2), parent=self.anchorGroup)
        self.anchorStrengthCard = SliderSpinCard(
            FluentIcon.SPEED_HIGH, _at("anchorStrengthCard", 1), 0, 150, suffix="%",
            description=_at("anchorStrengthCard", 2), parent=self.anchorGroup)
        self.anchorSmoothingCard = SliderSpinCard(
            FluentIcon.SYNC, _at("anchorSmoothingCard", 1), 0, 100, suffix="%",
            description=_at("anchorSmoothingCard", 2), parent=self.anchorGroup)
        self.anchorAccelCard = SliderSpinCard(
            FluentIcon.SPEED_HIGH, _at("anchorAccelCard", 1), 0, 100, suffix="%",
            description=_at("anchorAccelCard", 2), parent=self.anchorGroup)
        self.anchorConfCard = SliderSpinCard(
            FluentIcon.CERTIFICATE, _at("anchorConfCard", 1), 0, 100, suffix="%",
            description=_at("anchorConfCard", 2), parent=self.anchorGroup)
        self.anchorHistoryCard = SliderSpinCard(
            FluentIcon.HISTORY, _at("anchorHistoryCard", 1), 4, 12,
            description=_at("anchorHistoryCard", 2), parent=self.anchorGroup)
        self.anchorMaxDistCard = SliderSpinCard(
            FluentIcon.ZOOM, _at("anchorMaxDistCard", 1), 0, 100, suffix="%",
            description=_at("anchorMaxDistCard", 2), parent=self.anchorGroup)
        self.anchorInflAdaptiveCard = SliderSpinCard(
            FluentIcon.ZOOM, _at("anchorInflAdaptiveCard", 1), 0, 100, suffix="%",
            description=_at("anchorInflAdaptiveCard", 2), parent=self.anchorGroup)
        self.anchorInflDynamicCard = SliderSpinCard(
            FluentIcon.ZOOM, _at("anchorInflDynamicCard", 1), 0, 100, suffix="%",
            description=_at("anchorInflDynamicCard", 2), parent=self.anchorGroup)

        # --- Anchor FOV (core/anchor_fov.py): built from _AFOV_GROUPS / _AFOV_CARDS ---
        for gattr, gkey, gtitle in _AFOV_GROUPS:
            setattr(self, gattr, SettingCardGroup(t(gkey, gtitle), self.scrollWidget))
        for attr, gattr, kind, field, key, title, desc, extra in _AFOV_CARDS:
            group = getattr(self, gattr)
            if kind == 'combo':
                combo = ComboBox()
                combo.addItems([t(k, d) for _v, k, d in extra])
                combo.setMinimumWidth(180)
                card = SettingCard(FluentIcon.ZOOM, t(key, title), t(key + "_desc", desc), group)
                card.hBoxLayout.addWidget(combo, 0)
                card.hBoxLayout.addSpacing(16)
                card.combo = combo
            elif kind == 'switch':
                card = SwitchSettingCard(FluentIcon.ZOOM, t(key, title), t(key + "_desc", desc), parent=group)
            else:
                lo, hi, suffix, _scale = extra
                card = SliderSpinCard(FluentIcon.ZOOM, t(key, title), lo, hi, suffix=suffix,
                                      description=t(key + "_desc", desc), parent=group)
            setattr(self, attr, card)

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
        self.fovRetentionCard = SwitchSettingCard(
            FluentIcon.PIN,
            t("fov_retention", "Keep Lock Outside Shrunk FOV"),
            t("fov_retention_desc", "Keep tracking a locked target that drifts outside the shrunk FOV (no aiming there) instead of dropping it. Needs Sticky Target Lock. Rivals can't take the lock meanwhile."),
            parent=self.advFovGroup
        )
        self.fovRetentionPctCard = SliderSpinCard(
            FluentIcon.ZOOM,
            t("fov_retention_pct", "Retention FOV Size"),
            100, 400, suffix="%",
            description=t("fov_retention_pct_desc", "How far outside the shrunk FOV a lock is kept, as % of FOV Width/Height. 100% = the full original FOV."),
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
                   self.fovExpandDelayCard, self.fovRecoveryCard, self.fovRetentionCard, self.fovRetentionPctCard, self.minLockCard, self.trackConfCard,
                   self.switchThresholdCard, self.reacquireRadiusCard, self.lostTimeoutCard, self.stickyHintCard):
            _tip(_c, _c.contentLabel.text())

        # === Detection Settings ===
        self.generalGroup = SettingCardGroup(t("inference_general_params", "Detection Settings"), self.scrollWidget)

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

        # One choice instead of the former idle switch + the web-only keep-detecting switch
        # (core/detect_mode.py): full speed / reduced speed / stop while the aim key is not held.
        self.detectModeCombo = ComboBox()
        self.detectModeCombo.addItems(_DETECT_MODE_LABELS)
        self.detectModeCombo.setMinimumWidth(170)
        self.detectModeCard = SettingCard(
            FluentIcon.SPEED_MEDIUM,
            t("detect_mode", "When Not Aiming"),
            t("detect_mode_desc", _DETECT_MODE_DESC),
            self.generalGroup
        )
        self.detectModeCard.hBoxLayout.addWidget(self.detectModeCombo, 0)
        self.detectModeCard.hBoxLayout.addSpacing(16)

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

        for gattr, _gk, _gt in _AFOV_GROUPS:
            group = getattr(self, gattr)
            for attr, cg, *_rest in _AFOV_CARDS:
                if cg == gattr:
                    group.addSettingCard(getattr(self, attr))
            self.addContent(group)

        for card in (self.anchorEnableCard, self.anchorModeCard, self.anchorHorizonCard, self.anchorStrengthCard,
                     self.anchorSmoothingCard, self.anchorAccelCard, self.anchorConfCard, self.anchorHistoryCard,
                     self.anchorMaxDistCard, self.anchorInflAdaptiveCard, self.anchorInflDynamicCard):
            self.anchorGroup.addSettingCard(card)
        self.addContent(self.anchorGroup)

        self.advFovGroup.addSettingCard(self.fovExpandDelayCard)
        self.advFovGroup.addSettingCard(self.fovRecoveryCard)
        self.advFovGroup.addSettingCard(self.fovRetentionCard)
        self.advFovGroup.addSettingCard(self.fovRetentionPctCard)
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
        self.generalGroup.addSettingCard(self.detectModeCard)
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

    def _markAdvancedCards(self):
        """Fine-tuning cards that only show with the global "Show advanced settings" switch
        (components/advanced.py). Their values are never touched; the page's own visibility
        rules keep working underneath."""
        self.markAdvancedAttrs(*"""
anchorStrengthCard anchorSmoothingCard anchorAccelCard anchorConfCard anchorHistoryCard
            anchorMaxDistCard anchorInflAdaptiveCard anchorInflDynamicCard
            advFovGroup acqTrackGroup inferPerfGroup
            afovSmoothCard afovMaxSpeedCard afovLostDelayCard afovReacquireCard afovDynRadiusCard afovMaxRadiusCard
            """.split())

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
        for attr, _g, kind, field, _k, _t, _d, extra in _AFOV_CARDS:
            card = getattr(self, attr)
            if kind == 'combo':
                card.combo.currentIndexChanged.connect(
                    lambda i, f=field, vals=[v for v, _a, _b in extra]: self._onAfovComboChanged(f, vals, i))
            elif kind == 'switch':
                card.checkedChanged.connect(lambda v, f=field: self._onAfovSwitchChanged(f, bool(v)))
            else:
                scale = extra[3]
                card.valueChanged.connect(
                    lambda v, f=field, sc=scale: self._setCfg(f, (v / sc) if sc != 1 else v))
        self.anchorEnableCard.checkedChanged.connect(self._onAnchorEnableChanged)
        self.anchorModeCombo.currentIndexChanged.connect(self._onAnchorModeChanged)
        self.anchorHorizonCard.valueChanged.connect(lambda v: self._setCfg('anchor_horizon_ms', float(v)))
        self.anchorStrengthCard.valueChanged.connect(lambda v: self._setCfg('anchor_strength', v / 100.0))
        self.anchorSmoothingCard.valueChanged.connect(lambda v: self._setCfg('anchor_velocity_smoothing', v / 100.0))
        self.anchorAccelCard.valueChanged.connect(lambda v: self._setCfg('anchor_accel_comp', v / 100.0))
        self.anchorConfCard.valueChanged.connect(lambda v: self._setCfg('anchor_confidence_threshold', v / 100.0))
        self.anchorHistoryCard.valueChanged.connect(lambda v: self._setCfg('anchor_history_len', int(v)))
        self.anchorMaxDistCard.valueChanged.connect(lambda v: self._setCfg('anchor_max_distance_pct', float(v)))
        self.anchorInflAdaptiveCard.valueChanged.connect(lambda v: self._setCfg('anchor_fov_influence_adaptive', v / 100.0))
        self.anchorInflDynamicCard.valueChanged.connect(lambda v: self._setCfg('anchor_fov_influence_dynamic', v / 100.0))
        self.fovExpandDelayCard.valueChanged.connect(lambda v: self._setCfg('fov_expand_delay_ms', int(v)))
        self.fovRecoveryCard.valueChanged.connect(lambda v: self._setCfg('fov_recovery_duration_ms', int(v)))
        self.fovRetentionCard.checkedChanged.connect(self._onFovRetentionChanged)
        self.fovRetentionPctCard.valueChanged.connect(lambda v: self._setCfg('fov_retention_pct', float(v)))
        self.minLockCard.valueChanged.connect(lambda v: self._setCfg('min_lock_duration_ms', int(v)))
        self.trackConfCard.valueChanged.connect(lambda v: self._setCfg('tracking_confidence', v / 100.0))
        self.switchThresholdCard.valueChanged.connect(lambda v: self._setCfg('target_switch_threshold', v / 100.0))
        self.reacquireRadiusCard.valueChanged.connect(lambda v: self._setCfg('reacquire_radius_px', int(v)))
        self.lostTimeoutCard.valueChanged.connect(lambda v: self._setCfg('lost_target_timeout_ms', int(v)))
        self.detectRangeCard.valueChanged.connect(self._onDetectRangeChanged)

        self.detectIntervalCard.valueChanged.connect(self._onDetectIntervalChanged)
        self.confidenceCard.valueChanged.connect(self._onConfidenceChanged)
        self.semanticFilterCard.checkedChanged.connect(self._onSemanticFilterChanged)
        self.detectModeCombo.currentIndexChanged.connect(self._onDetectModeChanged)
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
            for attr, _g, kind, field, _k, _t, _d, extra in _AFOV_CARDS:
                card, val = getattr(self, attr), getattr(self._config, field, None)
                if kind == 'combo':
                    vals = [v for v, _a, _b in extra]
                    card.combo.setCurrentIndex(vals.index(val) if val in vals else 0)
                elif kind == 'switch':
                    card.setChecked(bool(val))
                else:
                    sc = extra[3]
                    card.setValue(int(round(float(val if val is not None else 0) * sc)))
            self.anchorEnableCard.setChecked(bool(getattr(self._config, 'anchor_enabled', False)))
            amode = str(getattr(self._config, 'anchor_mode', 'adaptive'))
            self.anchorModeCombo.setCurrentIndex(_ANCHOR_MODES.index(amode) if amode in _ANCHOR_MODES else 2)
            self.anchorHorizonCard.setValue(int(round(float(getattr(self._config, 'anchor_horizon_ms', 100.0)))))
            self.anchorStrengthCard.setValue(int(round(float(getattr(self._config, 'anchor_strength', 1.0)) * 100)))
            self.anchorSmoothingCard.setValue(int(round(float(getattr(self._config, 'anchor_velocity_smoothing', 0.4)) * 100)))
            self.anchorAccelCard.setValue(int(round(float(getattr(self._config, 'anchor_accel_comp', 0.35)) * 100)))
            self.anchorConfCard.setValue(int(round(float(getattr(self._config, 'anchor_confidence_threshold', 0.35)) * 100)))
            self.anchorHistoryCard.setValue(int(getattr(self._config, 'anchor_history_len', 6)))
            self.anchorMaxDistCard.setValue(int(round(float(getattr(self._config, 'anchor_max_distance_pct', 25.0)))))
            self.anchorInflAdaptiveCard.setValue(int(round(float(getattr(self._config, 'anchor_fov_influence_adaptive', 0.5)) * 100)))
            self.anchorInflDynamicCard.setValue(int(round(float(getattr(self._config, 'anchor_fov_influence_dynamic', 0.5)) * 100)))
            self.fovExpandDelayCard.setValue(int(getattr(self._config, 'fov_expand_delay_ms', 200)))
            self.fovRecoveryCard.setValue(int(getattr(self._config, 'fov_recovery_duration_ms', 400)))
            self.fovRetentionCard.setChecked(bool(getattr(self._config, 'fov_retention_enabled', False)))
            self.fovRetentionPctCard.setValue(int(round(float(getattr(self._config, 'fov_retention_pct', 100.0)))))
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

            self._syncDetectModeCombo()
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
        # Anchor FOV owns the FOV centre and radius, so the shrink-only cards (and the
        # "Reduce FOV" switch that enables them) apply to Fixed/Adaptive/Dynamic only.
        # The mode picker itself is always available.
        mode = str(getattr(self._config, 'fov_shrink_mode', 'adaptive')) if self._config else 'adaptive'
        shrink = enabled and mode != 'anchor'
        self.fovReduceEnableCard.setVisible(mode != 'anchor')
        self.fovModeCard.setVisible(True)
        self.fovDynSafeCard.setVisible(shrink and mode == 'dynamic')
        self.fovMinSizeCard.setVisible(shrink)
        self.fovMinSizeDurationCard.setVisible(shrink)
        # Advanced FOV only matters while the shrinking feature is on; the
        # minimum lock duration and the tracking group depend on Sticky Lock.
        self.fovExpandDelayCard.setVisible(shrink)
        self.fovRecoveryCard.setVisible(shrink)
        self._updateTrackingVisibility()
        self._updateAnchorVisibility()
        self._updateAnchorFovVisibility()

    def _updateAnchorFovVisibility(self):
        """Anchor FOV groups appear only in Anchor mode (hidden, not greyed out); the position
        and offsets need a box source, Manual X/Y the manual one, the prediction sliders need
        prediction on."""
        if not self._config:
            return
        anchor = str(getattr(self._config, 'fov_shrink_mode', 'adaptive')) == 'anchor'
        manual = str(getattr(self._config, 'afov_source', 'detection')) == 'manual'
        pred = bool(getattr(self._config, 'afov_predict_enabled', False))
        grows = bool(getattr(self._config, 'afov_dynamic_radius', False)) or \
            str(getattr(self._config, 'afov_lost_behavior', 'hold')) == 'search'
        for gattr, _k, _t in _AFOV_GROUPS:
            getattr(self, gattr).setVisible(anchor)
        # Manual is a point from the crosshair, so a box position means nothing; the offsets
        # apply to every source. Maximum Radius is only headroom for Dynamic Radius / Search.
        self.afovPositionCard.setVisible(anchor and not manual)
        self.afovPredTimeCard.setVisible(anchor and pred)
        self.afovMaxRadiusCard.setVisible(anchor and grows)
        # Fitting needs a box to fit to; Manual has none. The padding only matters while it is on.
        fit_on = bool(getattr(self._config, 'afov_fit_box_enabled', False))
        self.afovFitCard.setVisible(anchor and not manual)
        self.afovFitPadCard.setVisible(anchor and not manual and fit_on)

    def _onAfovComboChanged(self, field, values, index):
        if self._config and not self._isLoadingConfig and 0 <= index < len(values):
            setattr(self._config, field, values[index])
        self._updateAnchorFovVisibility()

    def _onAfovSwitchChanged(self, field, checked):
        self._setCfg(field, checked)
        self._updateAnchorFovVisibility()

    def _updateAnchorVisibility(self):
        """Anchor settings appear only while they act (hidden, not greyed out): the estimator
        tuning (shared with the Aim page's Anchor prediction method, which works without the
        master switch) while Anchor Prediction is on or that method is selected; the horizon and
        the FOV influence only with the master switch; smoothing only where the mode uses it; and
        only the influence of the current shrink mode (Fixed and Anchor FOV have none)."""
        if not self._config:
            return
        on = bool(getattr(self._config, 'anchor_enabled', False))
        mode = str(getattr(self._config, 'anchor_mode', 'adaptive'))
        fov_on = bool(getattr(self._config, 'fov_reduce_on_target_enabled', False))
        shared = on or (bool(getattr(self._config, 'prediction_enabled', False))
                        and str(getattr(self._config, 'prediction_method', 'velocity')) == 'anchor')
        self.anchorModeCard.setVisible(shared)
        for card in (self.anchorStrengthCard, self.anchorConfCard, self.anchorHistoryCard,
                     self.anchorMaxDistCard, self.anchorAccelCard):
            card.setVisible(shared)
        self.anchorHorizonCard.setVisible(on)
        self.anchorSmoothingCard.setVisible(shared and mode in ('smoothed', 'adaptive'))
        shrink = str(getattr(self._config, 'fov_shrink_mode', 'adaptive'))
        self.anchorInflAdaptiveCard.setVisible(on and fov_on and shrink == 'adaptive')
        self.anchorInflDynamicCard.setVisible(on and fov_on and shrink == 'dynamic')

    def _onAnchorEnableChanged(self, checked):
        self._setCfg('anchor_enabled', bool(checked))
        self._updateAnchorVisibility()

    def _onAnchorModeChanged(self, index):
        if self._config and not self._isLoadingConfig and 0 <= index < len(_ANCHOR_MODES):
            self._config.anchor_mode = _ANCHOR_MODES[index]
        self._updateAnchorVisibility()

    def _updateTrackingVisibility(self):
        sticky = bool(getattr(self._config, 'sticky_lock_enabled', False)) if self._config else False
        self.minLockCard.setVisible(sticky)
        fov_on_ = bool(getattr(self._config, 'fov_reduce_on_target_enabled', False)) if self._config else False
        fov_on_ = fov_on_ and str(getattr(self._config, 'fov_shrink_mode', 'adaptive')) != 'anchor'
        ret_on = bool(getattr(self._config, 'fov_retention_enabled', False)) if self._config else False
        self.fovRetentionCard.setVisible(sticky and fov_on_)
        self.fovRetentionPctCard.setVisible(sticky and fov_on_ and ret_on)
        fov_on = fov_on_
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
            "unstable detections. These three need Reduce FOV on Active Target. "
            "Anchor: the FOV centre follows an anchor and its radius is set independently; "
            "it works on its own.")

    def _setCfg(self, name, value):
        if self._config and not self._isLoadingConfig:
            setattr(self._config, name, value)

    def _onFovRetentionChanged(self, checked):
        self._setCfg('fov_retention_enabled', bool(checked))
        self._updateTrackingVisibility()

    def _onFovModeChanged(self, index):
        if self._config and not self._isLoadingConfig and 0 <= index < len(_FOV_MODES):
            self._config.fov_shrink_mode = _FOV_MODES[index]
            self._updateFovReduceVisibility(bool(getattr(self._config, 'fov_reduce_on_target_enabled', False)))

    def _applyScreenshotMethodEffect(self, method: str):
        """Called by capture page (and on load) to sync fov_follow visibility."""
        # udp is the same class of external-source capture as uvc/ndi — the
        # desktop cursor position is meaningless against any of these feeds,
        # so fov_follow_mouse must be force-disabled for all three, not just
        # uvc/ndi.
        # Only the card is hidden: the saved setting is never rewritten (the runtime ignores it
        # for external feeds via ai_loop_utils.fov_follow_effective), so switching back to
        # mss/dxcam restores the user's choice.
        is_external = method in EXTERNAL_CAPTURE_METHODS
        self.fovFollowCard.setVisible(not is_external)

    def _updateProviderDependentWidgets(self):
        """Show/hide and auto-configure CUDA IO Binding based on the active ORT provider."""
        if not self._config:
            return
        from core.ai_loop_utils import cuda_io_binding_switch_applies
        # Under TensorRT IO binding is always on (ai_loop_utils.cuda_io_binding_effective), so the
        # switch is only meaningful — and only shown — for the plain CUDA provider. The saved
        # value is never rewritten from here.
        self.cudaIoBindingCard.setVisible(cuda_io_binding_switch_applies(self._config))

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

    def _syncDetectModeCombo(self):
        """Show the saved choice (no write) and the card that only the reduced rate needs."""
        if not self._config:
            return
        mode = mode_from_config(self._config)
        self.detectModeCombo.blockSignals(True)
        self.detectModeCombo.setCurrentIndex(_DETECT_MODES.index(mode))
        self.detectModeCombo.blockSignals(False)
        self.idleDetectIntervalCard.setVisible(mode == 'reduced')
        # The reduced rate cannot apply under Always Aim / Always Auto-Fire: say so, don't rewrite it.
        note = _DETECT_MODE_IGNORED if reduced_rate_ignored(self._config) else ""
        self.detectModeCard.contentLabel.setText(t("detect_mode_desc", _DETECT_MODE_DESC) + (" " + note if note else ""))

    def _onDetectModeChanged(self, index):
        if self._config and not self._isLoadingConfig and 0 <= index < len(_DETECT_MODES):
            apply_mode(self._config, _DETECT_MODES[index])
            self._syncDetectModeCombo()

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
        # Only the ONNX HUD weapon detector has a detection threshold (OCR has none, and
        # "off" runs nothing) — the web client follows the same rule.
        self.hudConfidenceCard.setVisible(mode == "v2_onnx")

    def _onHudConfidenceChanged(self, value: float):
        if self._config:
            self._config.hud_confidence = float(value)

    # ──────────────────────────────────────────────
    # Retranslate
    # ──────────────────────────────────────────────

    def retranslateUi(self):
        super().retranslateUi()

        self.fovGroup.titleLabel.setText(t("fov_and_detect_range"))
        self.generalGroup.titleLabel.setText(t("inference_general_params", "Detection Settings"))
        self.inferPerfGroup.titleLabel.setText(t("inference_performance", "Inference Performance"))

        self.fovCard.titleLabel.setText(t("fov_size"))
        self.fovHeightCard.titleLabel.setText(t("fov_height", "FOV Height"))
        self.fovFollowCard.titleLabel.setText(t("fov_follow_mouse"))
        self.fovCircleCard.titleLabel.setText(t("fov_circle_filter", "Circular FOV Filter"))
        self.fovCircleCard.contentLabel.setText(t("fov_circle_filter_desc", "Only track targets inside the FOV ellipse — a circle when Width equals Height — instead of the full rectangular region"))
        self.fovReduceEnableCard.titleLabel.setText(t("fov_reduce_on_target", "Reduce FOV on Active Target"))
        self.fovReduceEnableCard.contentLabel.setText(t("fov_reduce_on_target_desc", "Temporarily shrink the FOV while a target is locked, so a random target near the edge of the FOV can't steal the lock mid-engagement."))
        self.fovMinSizeCard.titleLabel.setText(t("fov_min_size_pct", "FOV Min Size"))
        self.fovMinSizeCard.contentLabel.setText(t("fov_min_size_pct_desc", "How much to shrink the FOV to, as a percentage of FOV Width/Height, while a target is locked"))
        self.fovMinSizeDurationCard.titleLabel.setText(t("fov_min_size_duration", "FOV Min Size Duration"))
        self.fovMinSizeDurationCard.contentLabel.setText(t("fov_min_size_duration_desc", "How long it takes the FOV to gradually shrink down to FOV Min Size after a target is acquired, then holds there for the rest of the engagement (0 = shrinks instantly)"))
        self.detectRangeCard.titleLabel.setText(t("detect_range_size"))
        self.detectRangeCard.contentLabel.setText(t("detect_range_note"))

        self.fovShrinkGroup.titleLabel.setText(t("fov_shrinking", "FOV Shrinking"))
        self.anchorGroup.titleLabel.setText(t("anchor_prediction", "Anchor Box Prediction"))
        for attr, (key, title, desc) in _ANCHOR_TEXT.items():
            card = getattr(self, attr)
            card.titleLabel.setText(t(key, title))
            card.contentLabel.setText(t(key + "_desc", desc))
        for i, (key, default) in enumerate((("anchor_mode_velocity", "Velocity"),
                                            ("anchor_mode_smoothed", "Smoothed"), ("anchor_mode_adaptive", "Adaptive"))):
            self.anchorModeCombo.setItemText(i, t(key, default))
        self.advFovGroup.titleLabel.setText(t("advanced_fov", "Advanced FOV"))
        self.acqTrackGroup.titleLabel.setText(t("target_acq_tracking", "Target Acquisition & Tracking"))
        self.fovModeCard.titleLabel.setText(t("fov_shrink_mode", "Shrink Mode"))
        self.fovModeCard.contentLabel.setText(self._fovModeDesc())
        self.fovRetentionCard.titleLabel.setText(t("fov_retention", "Keep Lock Outside Shrunk FOV"))
        self.fovRetentionCard.contentLabel.setText(t("fov_retention_desc", "Keep tracking a locked target that drifts outside the shrunk FOV (no aiming there) instead of dropping it. Needs Sticky Target Lock. Rivals can't take the lock meanwhile."))
        self.fovRetentionPctCard.titleLabel.setText(t("fov_retention_pct", "Retention FOV Size"))
        self.fovRetentionPctCard.contentLabel.setText(t("fov_retention_pct_desc", "How far outside the shrunk FOV a lock is kept, as % of FOV Width/Height. 100% = the full original FOV."))
        self.fovDynSafeCard.titleLabel.setText(t("fov_dynamic_safeguards", "Dynamic Safeguards"))
        self.fovDynSafeCard.contentLabel.setText(t("fov_dynamic_safeguards_desc", "Dynamic mode only. Never shrink past what the target's distance from the FOV edge allows, and don't trust speed/direction until the lock has settled (about 0.5 s)."))
        for gattr, gkey, gtitle in _AFOV_GROUPS:
            getattr(self, gattr).titleLabel.setText(t(gkey, gtitle))
        for attr, _g, kind, _f, key, title, desc, extra in _AFOV_CARDS:
            card = getattr(self, attr)
            card.titleLabel.setText(t(key, title))
            card.contentLabel.setText(t(key + "_desc", desc))
            if kind == 'combo':
                for i, (_v, ik, idef) in enumerate(extra):
                    card.combo.setItemText(i, t(ik, idef))
        for i, (key, default) in enumerate((("fov_mode_fixed", "Fixed"), ("fov_mode_adaptive", "Adaptive"), ("fov_mode_dynamic", "Dynamic"), ("fov_mode_anchor", "Anchor"))):
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
        self.detectModeCard.titleLabel.setText(t("detect_mode", "When Not Aiming"))
        for i, (key, default) in enumerate(zip(("detect_mode_full", "detect_mode_reduced", "detect_mode_stop"),
                                               _DETECT_MODE_LABELS)):
            self.detectModeCombo.setItemText(i, t(key, default))
        self._syncDetectModeCombo()
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
