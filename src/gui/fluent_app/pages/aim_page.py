# aim_page.py
"""Aim Assist Page - PID, Smart Jitter, Target Priority, Target Tracking"""

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QStackedWidget
from qfluentwidgets import (
    SettingCardGroup, SwitchSettingCard,
    FluentIcon,
    ComboBox, SettingCard,
    SegmentedWidget,
)
from ..components.no_wheel_widgets import NoWheelDoubleSpinBox as DoubleSpinBox
from ..components.slider_spin_card import SliderLabelCard

from ..base_page import BasePage
from ..language_manager import t


class AimPage(BasePage):
    """Aim Assist Settings Page"""

    def __init__(self, parent=None):
        super().__init__("tab_aim_control", parent)
        self._config = None
        self._isLoadingConfig = False
        self._initWidgets()
        self._initLayout()
        self._connectSignals()

    def setConfig(self, config):
        """Sets Config instance and loads values"""
        self._config = config
        self._loadFromConfig()

    def _initWidgets(self):
        """Initializes all controls"""

        # === PID Parameters ===
        self.pidGroup = SettingCardGroup(t("aim_speed_pid"), self.scrollWidget)

        self.pidAxisPivot = SegmentedWidget()
        self.pidAxisPivot.addItem(routeKey='x', text=t("horizontal_x"))
        self.pidAxisPivot.addItem(routeKey='y', text=t("vertical_y"))
        self.pidAxisPivot.setCurrentItem('x')
        self.pidAxisPivot.currentItemChanged.connect(self._onPidAxisChanged)

        self.pidStackedWidget = QStackedWidget()

        self.pidPxCard = SliderLabelCard(
            FluentIcon.SPEED_HIGH,
            t("reaction_speed_p"),
            0, 100,
            format_func=lambda v: f"{v/100:.2f}",
            parent=self.pidGroup
        )

        self.pidIxCard = SliderLabelCard(
            FluentIcon.SYNC,
            t("error_correction_i"),
            0, 100,
            format_func=lambda v: f"{v/100:.2f}",
            parent=self.pidGroup
        )

        self.pidDxCard = SliderLabelCard(
            FluentIcon.ALIGNMENT,
            t("stability_suppression_d"),
            0, 100,
            format_func=lambda v: f"{v/100:.2f}",
            parent=self.pidGroup
        )

        self.pidPyCard = SliderLabelCard(
            FluentIcon.SPEED_HIGH,
            t("reaction_speed_p"),
            0, 100,
            format_func=lambda v: f"{v/100:.2f}",
            parent=self.pidGroup
        )

        self.pidIyCard = SliderLabelCard(
            FluentIcon.SYNC,
            t("error_correction_i"),
            0, 100,
            format_func=lambda v: f"{v/100:.2f}",
            parent=self.pidGroup
        )

        self.pidDyCard = SliderLabelCard(
            FluentIcon.ALIGNMENT,
            t("stability_suppression_d"),
            0, 100,
            format_func=lambda v: f"{v/100:.2f}",
            parent=self.pidGroup
        )

        self.pidYReduceEnableCard = SwitchSettingCard(
            FluentIcon.CARE_UP_SOLID,
            t("aim_y_reduce_enable"),
            "",
            parent=self.pidGroup
        )

        self.pidYReduceDelayCard = SliderLabelCard(
            FluentIcon.STOP_WATCH,
            t("aim_y_reduce_delay"),
            0, 500,
            format_func=lambda v: f"{v/100:.2f} s",
            parent=self.pidGroup
        )

        # === Anti-Detection (Smart Jitter only) ===
        self.antiDetectionGroup = SettingCardGroup(t("anti_detection", "Anti-Detection"), self.scrollWidget)

        self.smartJitterEnableCard = SwitchSettingCard(
            FluentIcon.MOVE,
            t("smart_jitter_label", "Smart Jitter"),
            t("smart_jitter_desc", "Add jitter when target box is small (far targets). Fires while shooting."),
            parent=self.antiDetectionGroup
        )

        self.smartJitterLmbCard = SwitchSettingCard(
            FluentIcon.FINGERPRINT,
            t("smart_jitter_lmb_label", "Only While Shooting (LMB Held)"),
            t("smart_jitter_lmb_desc", "Jitter only fires when an aim key is held"),
            parent=self.antiDetectionGroup
        )

        self.smartJitterStrengthSpin = DoubleSpinBox()
        self.smartJitterStrengthSpin.setRange(0.1, 200.0)
        self.smartJitterStrengthSpin.setSingleStep(0.5)
        self.smartJitterStrengthSpin.setDecimals(1)
        self.smartJitterStrengthSpin.setSuffix(" px")
        self.smartJitterStrengthSpin.setMinimumWidth(110)
        self.smartJitterLevelCard = SettingCard(
            FluentIcon.SPEED_HIGH,
            t("smart_jitter_level_label", "Jitter Strength"),
            t("smart_jitter_level_desc", "Max pixel offset radius applied per frame while jitter fires"),
            self.antiDetectionGroup
        )
        self.smartJitterLevelCard.hBoxLayout.addWidget(self.smartJitterStrengthSpin, 0)
        self.smartJitterLevelCard.hBoxLayout.addSpacing(16)

        self.smartJitterThreshCard = SliderLabelCard(
            FluentIcon.ZOOM_OUT,
            t("smart_jitter_threshold_label", "Box Size Threshold"),
            1, 50,
            format_func=lambda v: f"{v}%",
            description=t("smart_jitter_threshold_desc", "Jitter fires when box height < this % of detection range"),
            slider_width=160,
            parent=self.antiDetectionGroup
        )

        # === Target Priority ===
        self.targetPriorityGroup = SettingCardGroup(t("target_priority", "Target Priority"), self.scrollWidget)

        self.targetPriorityModeCombo = ComboBox()
        self.targetPriorityModeCombo.addItems(["Distance", "Confidence", "Composite"])
        self.targetPriorityModeCombo.setMinimumWidth(130)
        self.targetPriorityModeCard = SettingCard(
            FluentIcon.PEOPLE,
            t("target_priority_mode", "Priority Mode"),
            t("target_priority_mode_desc", "How to select the best target"),
            self.targetPriorityGroup
        )
        self.targetPriorityModeCard.hBoxLayout.addWidget(self.targetPriorityModeCombo, 0)
        self.targetPriorityModeCard.hBoxLayout.addSpacing(16)

        self.targetPriorityWeightCard = SliderLabelCard(
            FluentIcon.CERTIFICATE,
            t("target_priority_confidence_weight", "Confidence Weight"),
            0, 100,
            format_func=lambda v: f"{v}%",
            description=t("target_priority_weight_desc", "Used in Composite mode only"),
            slider_width=160,
            parent=self.targetPriorityGroup
        )

        # === Target Tracking ===
        self.trackingGroup = SettingCardGroup(t("target_tracking", "Target Tracking"), self.scrollWidget)

        self.emaEnableCard = SwitchSettingCard(
            FluentIcon.SPEED_MEDIUM,
            t("ema_enabled", "EMA Smoothing"),
            t("ema_desc", "Exponential moving average on aim-point coordinates before PID. Reduces jitter."),
            parent=self.trackingGroup
        )

        self.emaAlphaCard = SliderLabelCard(
            FluentIcon.MIX_VOLUMES,
            t("ema_alpha", "EMA Alpha"),
            30, 100,
            format_func=lambda v: f"{v / 100:.2f}",
            description=t("ema_alpha_desc", "1.0 = raw (no smoothing), 0.30 = heavy smoothing"),
            slider_width=160,
            parent=self.trackingGroup
        )

        self.predictionEnableCard = SwitchSettingCard(
            FluentIcon.RINGER,
            t("prediction_enabled", "Velocity Prediction"),
            t("prediction_desc", "Extrapolate target position forward by the prediction horizon."),
            parent=self.trackingGroup
        )

        self.predictionHorizonCard = SliderLabelCard(
            FluentIcon.HISTORY,
            t("prediction_horizon", "Prediction Horizon"),
            5, 50,
            format_func=lambda v: f"{v} ms",
            label_width=55,
            parent=self.trackingGroup
        )

        self.predictionMaxVelCard = SliderLabelCard(
            FluentIcon.SPEED_HIGH,
            t("prediction_max_velocity", "Max Velocity Cap"),
            300, 3000,
            format_func=lambda v: f"{v} px/s",
            label_width=70,
            description=t("prediction_max_vel_desc", "Velocity spikes above this are treated as detection jumps and reset prediction"),
            parent=self.trackingGroup
        )

        self.predictionHistoryCard = SliderLabelCard(
            FluentIcon.HISTORY,
            t("prediction_history", "History Frames"),
            2, 6,
            format_func=lambda v: str(v),
            parent=self.trackingGroup
        )

        self.stickyLockCard = SwitchSettingCard(
            FluentIcon.PIN,
            t("sticky_lock_enabled", "Sticky Target Lock"),
            t("sticky_lock_desc", "Lock onto a target and hold aim across short detection gaps."),
            parent=self.trackingGroup
        )

        self.lockDecayCard = SliderLabelCard(
            FluentIcon.HISTORY,
            t("lock_decay_frames", "Lock Decay Frames"),
            3, 60,
            format_func=lambda v: f"{v} fr",
            description=t("lock_decay_desc", "Frames to hold aim after target is lost before releasing the lock"),
            label_width=55,
            parent=self.trackingGroup
        )

        self.lockIouCard = SliderLabelCard(
            FluentIcon.ZOOM_IN,
            t("lock_iou_threshold", "IoU Match Threshold"),
            10, 70,
            format_func=lambda v: f"{v / 100:.2f}",
            description=t("lock_iou_desc", "Minimum overlap required to match the same target across frames"),
            slider_width=160,
            parent=self.trackingGroup
        )

        self.kalmanEnableCard = SwitchSettingCard(
            FluentIcon.SPEED_HIGH,
            t("kalman_enabled_label", "Kalman Filter"),
            t("kalman_enabled_desc", "2D Kalman filter for aim-point smoothing. Mutually exclusive with EMA."),
            parent=self.trackingGroup
        )

        self.kalmanProcessNoiseCard = SliderLabelCard(
            FluentIcon.MOVE,
            t("kalman_process_noise_label", "Process Noise"),
            1, 100,
            format_func=lambda v: f"{v / 100:.2f}",
            description=t("kalman_noise_desc", "Lower = smoother but slower to react"),
            slider_width=160,
            parent=self.trackingGroup
        )

        self.kalmanMeasNoiseCard = SliderLabelCard(
            FluentIcon.ALIGNMENT,
            t("kalman_meas_noise_label", "Measurement Noise"),
            1, 100,
            format_func=lambda v: f"{v / 100:.2f}",
            description=t("kalman_noise_desc", "Lower = reacts faster but noisier"),
            slider_width=160,
            parent=self.trackingGroup
        )

    def _initLayout(self):
        """Layout all controls"""
        # PID - tabbed X/Y layout
        pivotWidget = QWidget()
        pivotLayout = QHBoxLayout(pivotWidget)
        pivotLayout.setContentsMargins(16, 8, 16, 8)
        pivotLayout.addWidget(self.pidAxisPivot)
        pivotLayout.addStretch(1)

        self.pidXPage = QWidget()
        xPageLayout = QVBoxLayout(self.pidXPage)
        xPageLayout.setContentsMargins(0, 0, 0, 0)
        xPageLayout.setSpacing(0)
        xPageLayout.addWidget(self.pidPxCard)
        xPageLayout.addWidget(self.pidIxCard)
        xPageLayout.addWidget(self.pidDxCard)

        self.pidYPage = QWidget()
        yPageLayout = QVBoxLayout(self.pidYPage)
        yPageLayout.setContentsMargins(0, 0, 0, 0)
        yPageLayout.setSpacing(0)
        yPageLayout.addWidget(self.pidPyCard)
        yPageLayout.addWidget(self.pidIyCard)
        yPageLayout.addWidget(self.pidDyCard)
        yPageLayout.addWidget(self.pidYReduceEnableCard)
        yPageLayout.addWidget(self.pidYReduceDelayCard)

        self.pidStackedWidget.addWidget(self.pidXPage)
        self.pidStackedWidget.addWidget(self.pidYPage)

        self.pidGroup.vBoxLayout.addWidget(pivotWidget)
        self.pidGroup.vBoxLayout.addWidget(self.pidStackedWidget)
        self.addContent(self.pidGroup)

        # Anti-Detection (Smart Jitter)
        self.antiDetectionGroup.addSettingCard(self.smartJitterEnableCard)
        self.antiDetectionGroup.addSettingCard(self.smartJitterLmbCard)
        self.antiDetectionGroup.addSettingCard(self.smartJitterLevelCard)
        self.antiDetectionGroup.addSettingCard(self.smartJitterThreshCard)
        self.addContent(self.antiDetectionGroup)

        # Target Priority
        self.targetPriorityGroup.addSettingCard(self.targetPriorityModeCard)
        self.targetPriorityGroup.addSettingCard(self.targetPriorityWeightCard)
        self.addContent(self.targetPriorityGroup)

        # Target Tracking
        self.trackingGroup.addSettingCard(self.emaEnableCard)
        self.trackingGroup.addSettingCard(self.emaAlphaCard)
        self.trackingGroup.addSettingCard(self.predictionEnableCard)
        self.trackingGroup.addSettingCard(self.predictionHorizonCard)
        self.trackingGroup.addSettingCard(self.predictionMaxVelCard)
        self.trackingGroup.addSettingCard(self.predictionHistoryCard)
        self.trackingGroup.addSettingCard(self.stickyLockCard)
        self.trackingGroup.addSettingCard(self.lockDecayCard)
        self.trackingGroup.addSettingCard(self.lockIouCard)
        self.trackingGroup.addSettingCard(self.kalmanEnableCard)
        self.trackingGroup.addSettingCard(self.kalmanProcessNoiseCard)
        self.trackingGroup.addSettingCard(self.kalmanMeasNoiseCard)
        self.addContent(self.trackingGroup)

        self.scrollLayout.addStretch(1)

    def _connectSignals(self):
        """Connect signals"""
        # PID
        self.pidPxCard.valueChanged.connect(lambda v: self._onPidChanged('pid_kp_x', v))
        self.pidIxCard.valueChanged.connect(lambda v: self._onPidChanged('pid_ki_x', v))
        self.pidDxCard.valueChanged.connect(lambda v: self._onPidChanged('pid_kd_x', v))
        self.pidPyCard.valueChanged.connect(lambda v: self._onPidChanged('pid_kp_y', v))
        self.pidIyCard.valueChanged.connect(lambda v: self._onPidChanged('pid_ki_y', v))
        self.pidDyCard.valueChanged.connect(lambda v: self._onPidChanged('pid_kd_y', v))
        self.pidYReduceEnableCard.checkedChanged.connect(lambda checked: self._onPidChanged('aim_y_reduce_enabled', checked, is_bool=True))
        self.pidYReduceDelayCard.valueChanged.connect(lambda v: self._onPidChanged('aim_y_reduce_delay', v))

        # Smart Jitter
        self.smartJitterEnableCard.checkedChanged.connect(self._onSmartJitterEnableChanged)
        self.smartJitterLmbCard.checkedChanged.connect(self._onSmartJitterLmbChanged)
        self.smartJitterStrengthSpin.valueChanged.connect(self._onSmartJitterStrengthChanged)
        self.smartJitterThreshCard.valueChanged.connect(self._onSmartJitterThreshChanged)

        # Target Priority
        self.targetPriorityModeCombo.currentTextChanged.connect(self._onTargetPriorityModeChanged)
        self.targetPriorityWeightCard.valueChanged.connect(self._onTargetPriorityWeightChanged)

        # Target Tracking
        self.emaEnableCard.checkedChanged.connect(self._onEmaEnableChanged)
        self.emaAlphaCard.valueChanged.connect(self._onEmaAlphaChanged)
        self.predictionEnableCard.checkedChanged.connect(self._onPredictionEnableChanged)
        self.predictionHorizonCard.valueChanged.connect(self._onPredictionHorizonChanged)
        self.predictionMaxVelCard.valueChanged.connect(self._onPredictionMaxVelChanged)
        self.predictionHistoryCard.valueChanged.connect(self._onPredictionHistoryChanged)
        self.stickyLockCard.checkedChanged.connect(self._onStickyLockChanged)
        self.lockDecayCard.valueChanged.connect(self._onLockDecayChanged)
        self.lockIouCard.valueChanged.connect(self._onLockIouChanged)
        self.kalmanEnableCard.checkedChanged.connect(self._onKalmanEnableChanged)
        self.kalmanProcessNoiseCard.valueChanged.connect(self._onKalmanProcessNoiseChanged)
        self.kalmanMeasNoiseCard.valueChanged.connect(self._onKalmanMeasNoiseChanged)

    def _loadFromConfig(self):
        """Load values from Config"""
        if not self._config:
            return
        self._isLoadingConfig = True
        try:
            # PID
            self.pidPxCard.setValue(int(self._config.pid_kp_x * 100))
            self.pidIxCard.setValue(int(self._config.pid_ki_x * 100))
            self.pidDxCard.setValue(int(self._config.pid_kd_x * 100))
            self.pidPyCard.setValue(int(self._config.pid_kp_y * 100))
            self.pidIyCard.setValue(int(self._config.pid_ki_y * 100))
            self.pidDyCard.setValue(int(self._config.pid_kd_y * 100))
            self.pidYReduceEnableCard.setChecked(getattr(self._config, 'aim_y_reduce_enabled', False))
            self.pidYReduceDelayCard.setValue(int(getattr(self._config, 'aim_y_reduce_delay', 0.6) * 100))

            # Smart Jitter
            sj_on = bool(getattr(self._config, 'smart_jitter_enabled', False))
            self.smartJitterEnableCard.setChecked(sj_on)
            self.smartJitterLmbCard.setChecked(bool(getattr(self._config, 'smart_jitter_lmb_gate', True)))
            self.smartJitterStrengthSpin.setValue(float(getattr(self._config, 'smart_jitter_strength', 6.0)))
            self.smartJitterThreshCard.setValue(int(getattr(self._config, 'smart_jitter_box_threshold_pct', 15.0)))
            self.smartJitterLmbCard.setEnabled(sj_on)
            self.smartJitterLevelCard.setEnabled(sj_on)
            self.smartJitterThreshCard.setEnabled(sj_on)

            # Target Priority
            mode_map = {"distance": "Distance", "confidence": "Confidence", "composite": "Composite"}
            mode_text = mode_map.get(str(getattr(self._config, 'target_priority_mode', 'distance')), "Distance")
            self.targetPriorityModeCombo.setCurrentText(mode_text)
            self.targetPriorityWeightCard.setValue(int(getattr(self._config, 'target_priority_confidence_weight', 0.5) * 100))

            # Target Tracking
            self.emaEnableCard.setChecked(bool(getattr(self._config, 'ema_enabled', False)))
            self.emaAlphaCard.setValue(int(getattr(self._config, 'ema_alpha', 0.7) * 100))
            self.predictionEnableCard.setChecked(bool(getattr(self._config, 'prediction_enabled', False)))
            self.predictionHorizonCard.setValue(int(getattr(self._config, 'prediction_horizon_ms', 10.0)))
            self.predictionMaxVelCard.setValue(int(getattr(self._config, 'prediction_max_velocity', 1200.0)))
            self.predictionHistoryCard.setValue(int(getattr(self._config, 'prediction_history_len', 3)))
            self.stickyLockCard.setChecked(bool(getattr(self._config, 'sticky_lock_enabled', False)))
            self.lockDecayCard.setValue(int(getattr(self._config, 'lock_decay_frames', 15)))
            self.lockIouCard.setValue(int(getattr(self._config, 'lock_iou_threshold', 0.3) * 100))

            kalman_on = bool(getattr(self._config, 'kalman_enabled', False))
            self.kalmanEnableCard.setChecked(kalman_on)
            self.kalmanProcessNoiseCard.setValue(int(getattr(self._config, 'kalman_process_noise', 0.01) * 100))
            self.kalmanMeasNoiseCard.setValue(int(getattr(self._config, 'kalman_measurement_noise', 0.1) * 100))
            self.kalmanProcessNoiseCard.setEnabled(kalman_on)
            self.kalmanMeasNoiseCard.setEnabled(kalman_on)
            self.emaEnableCard.setEnabled(not kalman_on)
            self.emaAlphaCard.setEnabled(not kalman_on and bool(getattr(self._config, 'ema_enabled', False)))
            if kalman_on:
                self.emaEnableCard.setChecked(False)
                if self._config:
                    self._config.ema_enabled = False
        finally:
            self._isLoadingConfig = False

    # === PID Callbacks ===

    def _onPidAxisChanged(self, routeKey: str):
        self.pidStackedWidget.setCurrentIndex(0 if routeKey == 'x' else 1)

    def _onPidChanged(self, attr, value, is_bool=False):
        if self._config:
            if is_bool:
                setattr(self._config, attr, value)
            else:
                setattr(self._config, attr, value / 100.0)

    # === Smart Jitter Callbacks ===

    def _onSmartJitterEnableChanged(self, checked):
        if self._config:
            self._config.smart_jitter_enabled = bool(checked)
        self.smartJitterLmbCard.setEnabled(bool(checked))
        self.smartJitterLevelCard.setEnabled(bool(checked))
        self.smartJitterThreshCard.setEnabled(bool(checked))

    def _onSmartJitterLmbChanged(self, checked):
        if self._config:
            self._config.smart_jitter_lmb_gate = bool(checked)

    def _onSmartJitterStrengthChanged(self, value):
        if self._config:
            self._config.smart_jitter_strength = float(value)

    def _onSmartJitterThreshChanged(self, value):
        if self._config:
            self._config.smart_jitter_box_threshold_pct = float(value)

    # === Target Priority Callbacks ===

    def _onTargetPriorityModeChanged(self, text):
        if self._config:
            self._config.target_priority_mode = str(text).lower()

    def _onTargetPriorityWeightChanged(self, value):
        if self._config:
            self._config.target_priority_confidence_weight = value / 100.0

    # === Target Tracking Callbacks ===

    def _onEmaEnableChanged(self, checked):
        if self._config:
            self._config.ema_enabled = bool(checked)
        self.kalmanEnableCard.setEnabled(not checked)
        if checked:
            self.kalmanEnableCard.setChecked(False)
            if self._config:
                self._config.kalman_enabled = False
            self.kalmanProcessNoiseCard.setEnabled(False)
            self.kalmanMeasNoiseCard.setEnabled(False)
        self.emaAlphaCard.setEnabled(bool(checked))

    def _onEmaAlphaChanged(self, value):
        if self._config:
            self._config.ema_alpha = value / 100.0

    def _onPredictionEnableChanged(self, checked):
        if self._config:
            self._config.prediction_enabled = bool(checked)

    def _onPredictionHorizonChanged(self, value):
        if self._config:
            self._config.prediction_horizon_ms = float(value)

    def _onPredictionMaxVelChanged(self, value):
        if self._config:
            self._config.prediction_max_velocity = float(value)

    def _onPredictionHistoryChanged(self, value):
        if self._config:
            self._config.prediction_history_len = int(value)

    def _onStickyLockChanged(self, checked):
        if self._config:
            self._config.sticky_lock_enabled = bool(checked)

    def _onLockDecayChanged(self, value):
        if self._config:
            self._config.lock_decay_frames = int(value)

    def _onLockIouChanged(self, value):
        if self._config:
            self._config.lock_iou_threshold = value / 100.0

    def _onKalmanEnableChanged(self, checked):
        if self._config:
            self._config.kalman_enabled = bool(checked)
        self.emaEnableCard.setEnabled(not checked)
        if checked:
            self.emaEnableCard.setChecked(False)
            if self._config:
                self._config.ema_enabled = False
            self.emaAlphaCard.setEnabled(False)
        self.kalmanProcessNoiseCard.setEnabled(bool(checked))
        self.kalmanMeasNoiseCard.setEnabled(bool(checked))

    def _onKalmanProcessNoiseChanged(self, value):
        if self._config:
            self._config.kalman_process_noise = value / 100.0

    def _onKalmanMeasNoiseChanged(self, value):
        if self._config:
            self._config.kalman_measurement_noise = value / 100.0

    def retranslateUi(self):
        """Refresh translations"""
        super().retranslateUi()

        self.pidGroup.titleLabel.setText(t("aim_speed_pid"))
        self.pidAxisPivot.setItemText('x', t("horizontal_x"))
        self.pidAxisPivot.setItemText('y', t("vertical_y"))
        self.pidPxCard.titleLabel.setText(t("reaction_speed_p"))
        self.pidIxCard.titleLabel.setText(t("error_correction_i"))
        self.pidDxCard.titleLabel.setText(t("stability_suppression_d"))
        self.pidPyCard.titleLabel.setText(t("reaction_speed_p"))
        self.pidIyCard.titleLabel.setText(t("error_correction_i"))
        self.pidDyCard.titleLabel.setText(t("stability_suppression_d"))
        self.pidYReduceEnableCard.titleLabel.setText(t("aim_y_reduce_enable"))
        self.pidYReduceDelayCard.titleLabel.setText(t("aim_y_reduce_delay"))

        self.antiDetectionGroup.titleLabel.setText(t("anti_detection", "Anti-Detection"))
        self.smartJitterEnableCard.titleLabel.setText(t("smart_jitter_label", "Smart Jitter"))
        self.smartJitterLmbCard.titleLabel.setText(t("smart_jitter_lmb_label", "Only While Shooting (LMB)"))
        self.smartJitterLevelCard.titleLabel.setText(t("smart_jitter_level_label", "Jitter Strength"))
        self.smartJitterThreshCard.titleLabel.setText(t("smart_jitter_threshold_label", "Box Size Threshold"))

        self.targetPriorityGroup.titleLabel.setText(t("target_priority", "Target Priority"))
        self.targetPriorityModeCard.titleLabel.setText(t("target_priority_mode", "Priority Mode"))
        self.targetPriorityWeightCard.titleLabel.setText(t("target_priority_confidence_weight", "Confidence Weight"))

        self.trackingGroup.titleLabel.setText(t("target_tracking", "Target Tracking"))
        self.emaEnableCard.titleLabel.setText(t("ema_enabled", "EMA Smoothing"))
        self.emaAlphaCard.titleLabel.setText(t("ema_alpha", "EMA Alpha"))
        self.kalmanEnableCard.titleLabel.setText(t("kalman_enabled_label", "Kalman Filter"))
        self.kalmanProcessNoiseCard.titleLabel.setText(t("kalman_process_noise_label", "Process Noise"))
        self.kalmanMeasNoiseCard.titleLabel.setText(t("kalman_meas_noise_label", "Measurement Noise"))
