# aim_page.py
"""Aim Assist Page - Move Method, Arduino, Xbox, PID, Target Priority, Target Tracking"""

import os
import re
import sys
import subprocess
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QStackedWidget, QMessageBox,
)
from PyQt6.QtGui import QDesktopServices, QPainter, QColor, QPen
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from qfluentwidgets import (
    SettingCardGroup, SwitchSettingCard,
    FluentIcon,
    ComboBox, SettingCard,
    SegmentedWidget,
    BodyLabel, PushButton, CheckBox,
)
from ..components.slider_spin_card import SliderLabelCard, SliderSpinCard

from ..base_page import BasePage
from ..language_manager import t
from ..theme_colors import ThemeColors


class _AimPointPreview(QWidget):
    """Live canvas: simulated ESP box with head/body zones and aim-point X mark."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(200)
        self._aim_part = "center"
        self._head_h = 0.20   # fraction 0–1
        self._head_w = 0.38   # fraction 0–1
        self._body_w = 0.87   # fraction 0–1
        self._custom_y = 0.30  # fraction 0–1

    def setParams(self, aim_part: str, head_h_pct: float,
                  head_w_pct: float = None, body_w_pct: float = None,
                  custom_y_pct: float = None):
        self._aim_part = aim_part
        self._head_h = max(0.05, min(0.95, head_h_pct / 100.0))
        if head_w_pct is not None:
            self._head_w = max(0.05, min(1.0, head_w_pct / 100.0))
        if body_w_pct is not None:
            self._body_w = max(0.05, min(1.0, body_w_pct / 100.0))
        if custom_y_pct is not None:
            self._custom_y = max(0.0, min(1.0, custom_y_pct / 100.0))
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        W, H = self.width(), self.height()

        # background
        p.fillRect(0, 0, W, H, QColor(15, 15, 20))

        # full bounding box geometry (tall, centred)
        bw = max(40, int(W * 0.28))
        bh = int(H * 0.82)
        bx = (W - bw) // 2
        by = (H - bh) // 2
        head_px = int(bh * self._head_h)

        # narrowed zone widths
        hw = max(4, int(bw * self._head_w))
        ow = max(4, int(bw * self._body_w))
        hx = bx + (bw - hw) // 2   # head zone x
        ox = bx + (bw - ow) // 2   # body zone x

        # zone fills (show actual width)
        p.fillRect(hx, by,           hw, head_px,      QColor(80,  120, 220, 55))
        p.fillRect(ox, by + head_px, ow, bh - head_px, QColor(200, 120,  40, 45))

        # full corner-box (ESP style)
        cl = max(5, min(bw, bh) // 6)
        p.setPen(QPen(QColor(0, 220, 140), 2))
        for sx, sy, dx, dy in ((bx, by, 1, 1), (bx+bw, by, -1, 1),
                                (bx, by+bh, 1, -1), (bx+bw, by+bh, -1, -1)):
            p.drawLine(sx, sy + dy*cl, sx, sy)
            p.drawLine(sx, sy, sx + dx*cl, sy)

        # head / body divider
        p.setPen(QPen(QColor(140, 140, 200, 80), 1, Qt.PenStyle.DashLine))
        p.drawLine(bx, by + head_px, bx + bw, by + head_px)

        # aim-point position
        ax = bx + bw // 2
        if self._aim_part == "head":
            ay = by + head_px // 2
        elif self._aim_part == "body":
            ay = by + head_px + (bh - head_px) // 2
        elif self._aim_part in ("custom", "center"):
            ay = by + int(bh * self._custom_y)

        # red X crosshair
        r = 6
        p.setPen(QPen(QColor(255, 50, 50), 2))
        p.drawLine(ax - r, ay - r, ax + r, ay + r)
        p.drawLine(ax + r, ay - r, ax - r, ay + r)

        # zone labels to the right of the box
        fnt = p.font()
        fnt.setPointSize(8)
        p.setFont(fnt)
        lx = bx + bw + 8
        p.setPen(QColor(100, 150, 230, 200))
        p.drawText(lx, by + head_px // 2 + 4, "Head")
        p.setPen(QColor(220, 140, 60, 200))
        p.drawText(lx, by + head_px + (bh - head_px) // 2 + 4, "Body")

        # mode label bottom-left
        fnt.setPointSize(7)
        p.setFont(fnt)
        p.setPen(QColor(160, 160, 160, 130))
        label = {"head": "Head aim", "body": "Body aim",
                 "custom": f"Custom ({int(self._custom_y*100)}%)",
                 "center": f"Smart @ {int(self._custom_y*100)}%"}.get(self._aim_part, self._aim_part)
        p.drawText(6, H - 5, label)

        p.end()


class _ArduinoConnectWorker(QThread):
    """Runs connect_arduino() off the GUI thread.

    connect_arduino() sleeps 2s (outside its own lock) waiting for the
    Leonardo's auto-restart-on-connect — safe to call from a background
    thread, the problem was purely that "Connect Arduino" invoked it
    synchronously on the GUI thread, freezing the whole UI for those 2s
    on every click.
    """

    finishedResult = pyqtSignal(bool)  # ok

    def __init__(self, com_port: str, parent=None):
        super().__init__(parent)
        self._com_port = com_port

    def run(self) -> None:
        try:
            from win_utils import connect_arduino
            ok = connect_arduino(self._com_port)
        except Exception:
            ok = False
        self.finishedResult.emit(ok)


class _TargetClassSelector(QWidget):
    """Dynamic multi-select checkbox list for `config.aim_target_class_ids`.

    Rebuilt from scratch whenever the loaded model's own class-name dict
    (`config._detect_class_names`, populated by
    `detection_semantics.sync_detection_class_names_from_backend()` on each
    model load/hot-swap) changes — the valid class IDs and names are
    entirely model-specific, so this can't be a static, hand-authored list
    of checkboxes the way every other card on this page is.

    One `SettingCard` row per class (class name as title, an empty-text
    `CheckBox` right-aligned in its control column via `hBoxLayout`) —
    reuses the exact component/layout every other row on this page already
    uses, rather than a hand-rolled left-aligned checkbox stack, so spacing
    and alignment matches the rest of Target Area Settings instead of
    looking like a different control.

    Hidden entirely for a single-class (or nameless) model — there's
    nothing meaningful to multi-select when every detection is already the
    only class there is; the page hides this widget's own header card to
    match (see AimPage._loadFromConfig()), same "hidden rather than greyed
    out" precedent as the Humanization sub-sliders.
    """

    selectionChanged = pyqtSignal(list)  # emitted with the new aim_target_class_ids

    def __init__(self, parent=None):
        super().__init__(parent)
        self._vbox = QVBoxLayout(self)
        self._vbox.setContentsMargins(0, 0, 0, 0)
        self._vbox.setSpacing(0)
        self._rows: dict[int, SettingCard] = {}
        self._checks: dict[int, CheckBox] = {}
        self._names_key: tuple | None = None
        self.setVisible(False)

    def refresh(self, class_names: dict | None, selected_ids: list) -> None:
        """Rebuild (only if the class set actually changed) and resync
        checked state from `selected_ids` (an empty list means every class
        is checked — see aim_target_class_ids's own "no restriction"
        semantics). Returns True while >= 2 classes are available (i.e.
        whether this widget — and the caller's header card — should show
        at all), False otherwise."""
        names = class_names or {}
        key = tuple(sorted((int(k), str(v)) for k, v in names.items()))
        selected = {int(i) for i in (selected_ids or [])}

        if key == self._names_key:
            # Same class set as last time (the common case — this runs on
            # every ~1s live-config-sync tick) — just resync checked state,
            # e.g. after a preset/config load, without tearing down widgets.
            # Re-assert visibility every tick too (cheap, idempotent) rather
            # than trusting a single setVisible() call from the last rebuild
            # survives indefinitely — removes any chance of this widget
            # getting stuck hidden across ticks for a reason unrelated to
            # the class set itself.
            has_classes = len(names) >= 2
            self.setVisible(has_classes)
            for cid, box in self._checks.items():
                box.blockSignals(True)
                box.setChecked((not selected) or cid in selected)
                box.blockSignals(False)
            return has_classes

        self._names_key = key
        for row in self._rows.values():
            row.setParent(None)
            row.deleteLater()
        self._rows.clear()
        self._checks.clear()

        if len(names) < 2:
            # A single-class (or nameless) model has nothing to multi-select.
            self.setVisible(False)
            return False

        self.setVisible(True)
        for cid in sorted(names.keys()):
            row = SettingCard(FluentIcon.PEOPLE, f"{names[cid]}  (#{cid})", "", self)
            # SettingCard defaults to a 50px fixed height for a no-content
            # row (70px only when a description is set) — noticeably
            # thinner than every other row on this page, which all have a
            # description. There's nothing meaningful to put in this row's
            # content, so raise the fixed height directly instead.
            row.setFixedHeight(60)
            box = CheckBox("", row)
            box.setChecked((not selected) or cid in selected)
            box.stateChanged.connect(self._onCheckChanged)
            row.hBoxLayout.addWidget(box, 0, Qt.AlignmentFlag.AlignRight)
            row.hBoxLayout.addSpacing(16)
            self._vbox.addWidget(row)
            self._rows[cid] = row
            self._checks[cid] = box
        return True

    def _onCheckChanged(self, _state) -> None:
        all_ids = sorted(self._checks.keys())
        checked = sorted(cid for cid, box in self._checks.items() if box.isChecked())
        if not checked:
            # Refuse to let every class be unchecked — that would silently
            # mean "no valid targets at all" with no visual indicator why
            # detection stopped working. Re-check whichever box the user
            # just unchecked instead.
            sender = self.sender()
            if sender is not None:
                sender.blockSignals(True)
                sender.setChecked(True)
                sender.blockSignals(False)
            return
        # All classes checked is the canonical "no restriction" empty list —
        # keeps a model swap to one with MORE classes than were enumerated
        # here from silently excluding a class the user never saw.
        self.selectionChanged.emit([] if len(checked) == len(all_ids) else checked)


class AimPage(BasePage):
    """Aim Assist Settings Page"""

    def __init__(self, parent=None):
        super().__init__("tab_aim_control", parent)
        self._config = None
        self._isLoadingConfig = False
        self._isArduinoConnected = False
        self._arduinoConnectWorker: _ArduinoConnectWorker | None = None
        self._isXboxConnected = False
        self._initWidgets()
        self._initLayout()
        self._connectSignals()

    def setConfig(self, config):
        """Sets Config instance and loads values"""
        self._config = config
        self._loadFromConfig()

    def _initWidgets(self):
        """Initializes all controls"""

        # === General (Aim Part + Move Method) ===
        self.generalGroup = SettingCardGroup(t("general_params"), self.scrollWidget)

        self.aimPartCombo = ComboBox()
        self.aimPartCombo.addItems([t("head"), t("body"), t("center", "Smart (Center-mass)"), t("custom", "Custom")])
        self.aimPartCombo.setMinimumWidth(120)
        self.aimPartCard = SettingCard(
            FluentIcon.PEOPLE,
            t("aim_part"),
            "",
            self.generalGroup
        )
        self.aimPartCard.hBoxLayout.addWidget(self.aimPartCombo, 0, Qt.AlignmentFlag.AlignRight)
        self.aimPartCard.hBoxLayout.addSpacing(16)

        self.mouseMoveCombo = ComboBox()
        self.mouseMoveCombo.addItems(["ddxoft", "mouse_event", "sendinput", "arduino", "makcu", "xbox"])
        self.mouseMoveCombo.setMinimumWidth(150)
        self.mouseMoveCard = SettingCard(
            FluentIcon.FINGERPRINT,
            t("mouse_move_method"),
            "",
            self.generalGroup
        )
        self.mouseMoveCard.hBoxLayout.addWidget(self.mouseMoveCombo, 0, Qt.AlignmentFlag.AlignRight)
        self.mouseMoveCard.hBoxLayout.addSpacing(16)

        # === Arduino Settings ===
        self.arduinoGroup = SettingCardGroup("Arduino", self.scrollWidget)

        self.comPortCombo = ComboBox()
        self.comPortCombo.setMinimumWidth(120)
        self.comPortCombo.addItem(t("no_com_port"))
        self._refreshComPorts()

        self.comRefreshBtn = PushButton(t("refresh"))
        self.comRefreshBtn.setFixedWidth(80)

        self.comPortCard = SettingCard(
            FluentIcon.CONNECT,
            t("arduino_com_port"),
            "",
            self.arduinoGroup
        )
        self.comPortCard.hBoxLayout.addWidget(self.comPortCombo, 0, Qt.AlignmentFlag.AlignRight)
        self.comPortCard.hBoxLayout.addWidget(self.comRefreshBtn, 0, Qt.AlignmentFlag.AlignRight)
        self.comPortCard.hBoxLayout.addSpacing(16)

        self.arduinoBaudCombo = ComboBox()
        self.arduinoBaudCombo.addItems(["115200", "500000", "1000000", "2000000", "4000000"])
        self.arduinoBaudCombo.setMinimumWidth(120)
        self.arduinoBaudCard = SettingCard(
            FluentIcon.SPEED_HIGH,
            t("arduino_baud_rate", "Baud Rate"),
            t("arduino_baud_rate_desc", "⚠ Must match the baud rate in your Arduino sketch"),
            self.arduinoGroup
        )
        self.arduinoBaudCard.hBoxLayout.addWidget(self.arduinoBaudCombo, 0, Qt.AlignmentFlag.AlignRight)
        self.arduinoBaudCard.hBoxLayout.addSpacing(16)

        self.connectionLabel = BodyLabel(t("disconnected"))
        self.connectionLabel.setStyleSheet(f"color: {ThemeColors.ERROR.get()}; font-weight: bold;")
        self.connectionCard = SettingCard(
            FluentIcon.WIFI,
            t("connected") + " / " + t("disconnected"),
            "",
            self.arduinoGroup
        )
        self.connectionCard.hBoxLayout.addWidget(self.connectionLabel, 0, Qt.AlignmentFlag.AlignRight)
        self.connectionCard.hBoxLayout.addSpacing(16)

        self.arduinoConnectBtn = PushButton(t("arduino_connect"))
        self.arduinoConnectBtn.setFixedWidth(120)
        self.arduinoConnectCard = SettingCard(
            FluentIcon.LINK,
            t("arduino_connect"),
            t("arduino_connect_desc"),
            self.arduinoGroup
        )
        self.arduinoConnectCard.hBoxLayout.addWidget(self.arduinoConnectBtn, 0, Qt.AlignmentFlag.AlignRight)
        self.arduinoConnectCard.hBoxLayout.addSpacing(16)

        self.guideBtn = PushButton(t("arduino_guide"))
        self.guideCard = SettingCard(
            FluentIcon.BOOK_SHELF,
            t("arduino_guide"),
            "",
            self.arduinoGroup
        )
        self.guideCard.hBoxLayout.addWidget(self.guideBtn, 0, Qt.AlignmentFlag.AlignRight)
        self.guideCard.hBoxLayout.addSpacing(16)

        self.spoofBtn = PushButton(t("spoof_device"))
        self.spoofCard = SettingCard(
            FluentIcon.VPN,
            t("spoof_device"),
            "",
            self.arduinoGroup
        )
        self.spoofCard.hBoxLayout.addWidget(self.spoofBtn, 0, Qt.AlignmentFlag.AlignRight)
        self.spoofCard.hBoxLayout.addSpacing(16)

        self.verifySpoofBtn = PushButton(t("verify_spoof"))
        self.verifySpoofCard = SettingCard(
            FluentIcon.ACCEPT,
            t("verify_spoof"),
            "",
            self.arduinoGroup
        )
        self.verifySpoofCard.hBoxLayout.addWidget(self.verifySpoofBtn, 0, Qt.AlignmentFlag.AlignRight)
        self.verifySpoofCard.hBoxLayout.addSpacing(16)

        self.testHeartBtn = PushButton(t("test_move_heart"))
        self.testHeartCard = SettingCard(
            FluentIcon.HEART,
            t("test_move_heart"),
            "",
            self.arduinoGroup
        )
        self.testHeartCard.hBoxLayout.addWidget(self.testHeartBtn, 0, Qt.AlignmentFlag.AlignRight)
        self.testHeartCard.hBoxLayout.addSpacing(16)

        # === Xbox 360 Controller ===
        self.xboxGroup = SettingCardGroup("Xbox 360 Controller", self.scrollWidget)

        self.xboxSensitivityCard = SliderSpinCard(
            FluentIcon.SPEED_HIGH,
            t("xbox_sensitivity"),
            10, 500,
            suffix="%",
            description="",
            parent=self.xboxGroup
        )

        self.xboxDeadzoneCard = SliderSpinCard(
            FluentIcon.REMOVE,
            t("xbox_deadzone"),
            0, 50,
            suffix="%",
            description="",
            parent=self.xboxGroup
        )

        self.xboxConnectionLabel = BodyLabel(t("disconnected"))
        self.xboxConnectionLabel.setStyleSheet(f"color: {ThemeColors.ERROR.get()}; font-weight: bold;")
        self.xboxConnectionCard = SettingCard(
            FluentIcon.GAME,
            t("connected") + " / " + t("disconnected"),
            "",
            self.xboxGroup
        )
        self.xboxConnectionCard.hBoxLayout.addWidget(self.xboxConnectionLabel, 0, Qt.AlignmentFlag.AlignRight)
        self.xboxConnectionCard.hBoxLayout.addSpacing(16)

        self.xboxConnectBtn = PushButton(t("xbox_connect"))
        self.xboxConnectBtn.setFixedWidth(120)
        self.xboxConnectCard = SettingCard(
            FluentIcon.WIFI,
            t("xbox_connect"),
            t("xbox_connect_desc"),
            self.xboxGroup
        )
        self.xboxConnectCard.hBoxLayout.addWidget(self.xboxConnectBtn, 0, Qt.AlignmentFlag.AlignRight)
        self.xboxConnectCard.hBoxLayout.addSpacing(16)

        # === PID Parameters ===
        self.pidGroup = SettingCardGroup(t("aim_speed_pid"), self.scrollWidget)
        self.yReduceGroup = SettingCardGroup("Y-Axis Recoil Suppression", self.scrollWidget)

        # Nothing downstream actually clamps Kp — the P sliders just cap their
        # travel at the proven-stable 0.0-0.5 band by default. This toggle
        # re-maps that same slider travel to the full 0.0-1.0 range instead.
        self.pidUnsafeCard = SwitchSettingCard(
            FluentIcon.INFO,
            t("pid_unsafe_mode", "Unsafe Mode"),
            t("pid_unsafe_mode_desc", "Let the P (reaction speed) sliders be dragged past the proven-stable 0.50 cap, up to 1.00. Higher values can cause oscillation/overshoot — tune carefully."),
            parent=self.pidGroup
        )

        self.pidAxisPivot = SegmentedWidget()
        self.pidAxisPivot.addItem(routeKey='x', text=t("horizontal_x"))
        self.pidAxisPivot.addItem(routeKey='y', text=t("vertical_y"))
        self.pidAxisPivot.setCurrentItem('x')
        self.pidAxisPivot.currentItemChanged.connect(self._onPidAxisChanged)

        self.pidStackedWidget = QStackedWidget()

        # Kp slider travel 0–100 maps to config 0.0–0.5 (the proven-stable band) by
        # default, or 0.0–1.0 with Unsafe Mode on — see _onPidChanged / _loadFromConfig
        # / _onPidUnsafeChanged, which swap the /200 vs /100 divisor and format_func
        # together so the displayed label always matches the effective gain.
        self.pidPxCard = SliderLabelCard(
            FluentIcon.SPEED_HIGH,
            t("reaction_speed_p"),
            0, 100,
            format_func=lambda v: f"{v/200:.2f}",
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
            format_func=lambda v: f"{v/200:.2f}",
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

        # Hard per-frame pixel cap on PID output, applied to both axes after
        # sub-pixel carry. Defaults to 85px (non-zero) even with no GUI
        # control previously exposing it — surfaced here so it's visible/
        # adjustable instead of silently capping every correction.
        self.maxMovePerFrameCard = SliderLabelCard(
            FluentIcon.CARE_DOWN_SOLID,
            "Max Move Per Frame",
            0, 500,
            format_func=lambda v: "Off" if v == 0 else f"{v} px",
            description="Hard cap on PID output per frame, both axes (0 = off)",
            parent=self.pidGroup
        )

        self.pidYReduceEnableCard = SwitchSettingCard(
            FluentIcon.CARE_UP_SOLID,
            t("aim_y_reduce_enable"),
            "",
            parent=self.yReduceGroup
        )

        self.pidYReduceDelayCard = SliderLabelCard(
            FluentIcon.STOP_WATCH,
            t("aim_y_reduce_delay"),
            0, 500,
            format_func=lambda v: f"{v/100:.2f} s",
            parent=self.yReduceGroup
        )

        self.pidYReduceFloorCard = SliderLabelCard(
            FluentIcon.CARE_DOWN_SOLID,
            "Y Floor",
            0, 100,
            format_func=lambda v: f"{v/100:.2f}",
            description="Min Y multiplier after ramp — 0.00 = full cut, 1.00 = no suppression",
            parent=self.yReduceGroup
        )

        self.pidYReduceRampCard = SliderLabelCard(
            FluentIcon.STOP_WATCH,
            "Y Ramp Window",
            0, 200,
            format_func=lambda v: f"{v/100:.2f} s",
            description="Time to fade 1.0 → floor after delay (0 = instant cut)",
            parent=self.yReduceGroup
        )

        self.pidYReduceSettleCard = SliderLabelCard(
            FluentIcon.ALIGNMENT,
            "Y Settle Threshold",
            0, 50,
            format_func=lambda v: "Off" if v == 0 else f"{v} px",
            description="Skip suppression while vertical error > this — waits until aim is settled (0 = off)",
            parent=self.yReduceGroup
        )

        self.pidYReduceVelCard = SliderLabelCard(
            FluentIcon.SPEED_MEDIUM,
            "Y Velocity Restore",
            0, 500,
            format_func=lambda v: "Off" if v == 0 else f"{v} px/s",
            description="Restore full Y tracking if target moves vertically faster than this (0 = off)",
            parent=self.yReduceGroup
        )

        # === Humanization ===
        # Post-processing layer applied to the final PID dx/dy, right before mouse
        # injection — see ai_aiming.py's apply_humanization() call site.
        self.humanizationGroup = SettingCardGroup(t("humanization", "Humanization"), self.scrollWidget)

        self.humanizationEnableCard = SwitchSettingCard(
            FluentIcon.PEOPLE,
            t("humanization_enabled", "Humanization"),
            t("humanization_desc", "Perturb the final mouse output to look less robotic. Operates only on dx/dy — never touches detection or PID state."),
            parent=self.humanizationGroup
        )

        self.humanizationIntensityCard = SliderLabelCard(
            FluentIcon.SPEED_HIGH,
            t("humanization_intensity", "Intensity"),
            0, 100,
            format_func=lambda v: f"{v}%",
            description=t("humanization_intensity_desc", "0% = robotic precision, 100% = fully human-like. Scales every effect below."),
            slider_width=160,
            parent=self.humanizationGroup
        )

        self.humanizationMicroJitterCard = SwitchSettingCard(
            FluentIcon.MOVE,
            t("humanization_micro_jitter", "Micro-Jitter"),
            t("humanization_micro_jitter_desc", "Small zero-mean noise added to every move, scaled by movement size."),
            parent=self.humanizationGroup
        )

        self.humanizationMotionVariationCard = SwitchSettingCard(
            FluentIcon.SYNC,
            t("humanization_motion_variation", "Motion Variation"),
            t("humanization_motion_variation_desc", "Randomize output scale slightly each frame (mean-preserving, no drift)."),
            parent=self.humanizationGroup
        )

        self.humanizationSpeedShapingCard = SwitchSettingCard(
            FluentIcon.ZOOM_IN,
            t("humanization_speed_shaping", "Speed Shaping"),
            t("humanization_speed_shaping_desc", "Compress small corrections and pass through large movements unmodified, like human fine-motor control."),
            parent=self.humanizationGroup
        )

        self.humanizationMicroStutterCard = SwitchSettingCard(
            FluentIcon.PAUSE_BOLD,
            t("humanization_micro_stutter", "Micro-Stutter"),
            t("humanization_micro_stutter_desc", "Occasional brief magnitude reduction, modeling muscle hesitation before committing to a move."),
            parent=self.humanizationGroup
        )

        self.humanizationReactionVariabilityCard = SwitchSettingCard(
            FluentIcon.PAUSE_BOLD,
            t("humanization_reaction_variability", "Reaction Variability"),
            t("humanization_reaction_variability_desc", "Occasionally skip a frame's mouse injection to simulate human micro-hesitation. Adds real per-frame latency — off by default."),
            parent=self.humanizationGroup
        )

        # Fine-tuning sliders for the sub-parameters behind each feature above.
        # Each slider is scaled like the PID Kp sliders (int travel / divisor ->
        # float config value) — see _loadHumanizationFromConfig() / the matching
        # _onHumanization*Changed() handler for each divisor.
        self.humanizationJitterBaseCard = SliderLabelCard(
            FluentIcon.MOVE,
            t("humanization_jitter_base", "Jitter Base"),
            0, 200,
            format_func=lambda v: f"{v/100:.2f} px",
            description=t("humanization_jitter_base_desc", "Minimum jitter amplitude added every frame, in pixels."),
            slider_width=160,
            parent=self.humanizationGroup
        )

        self.humanizationJitterScaleCard = SliderLabelCard(
            FluentIcon.MOVE,
            t("humanization_jitter_scale", "Jitter Scale"),
            0, 200,
            format_func=lambda v: f"{v/10:.1f}%",
            description=t("humanization_jitter_scale_desc", "Extra jitter added per pixel of movement, as % of movement size."),
            slider_width=160,
            parent=self.humanizationGroup
        )

        self.humanizationMicroJitterIdleCard = SwitchSettingCard(
            FluentIcon.MOVE,
            t("humanization_micro_jitter_idle", "Apply While Aiming Idle"),
            t("humanization_micro_jitter_idle_desc", "Also apply Micro-Jitter's tremor while the aim key is held but no target is locked, instead of holding perfectly still."),
            parent=self.humanizationGroup
        )

        self.humanizationMotionVariationRangeCard = SliderLabelCard(
            FluentIcon.SYNC,
            t("humanization_motion_variation_range", "Variation Range"),
            0, 200,
            format_func=lambda v: f"±{v/10:.1f}%",
            description=t("humanization_motion_variation_range_desc", "Random output-scale range applied each frame (mean-preserving)."),
            slider_width=160,
            parent=self.humanizationGroup
        )

        self.humanizationSpeedShapingLowCard = SliderLabelCard(
            FluentIcon.ZOOM_IN,
            t("humanization_speed_shaping_low", "Fine-Control Threshold"),
            0, 100,
            format_func=lambda v: f"{v/5:.1f} px",
            description=t("humanization_speed_shaping_low_desc", "Movements below this size are compressed by the Low-Speed Factor."),
            slider_width=160,
            parent=self.humanizationGroup
        )

        self.humanizationSpeedShapingHighCard = SliderLabelCard(
            FluentIcon.ZOOM_OUT,
            t("humanization_speed_shaping_high", "Full-Speed Threshold"),
            0, 100,
            format_func=lambda v: f"{v/2:.1f} px",
            description=t("humanization_speed_shaping_high_desc", "Movements above this size pass through unmodified."),
            slider_width=160,
            parent=self.humanizationGroup
        )

        self.humanizationSpeedShapingLowFactorCard = SliderLabelCard(
            FluentIcon.ZOOM,
            t("humanization_speed_shaping_low_factor", "Low-Speed Factor"),
            0, 100,
            format_func=lambda v: f"{v/100:.2f}",
            description=t("humanization_speed_shaping_low_factor_desc", "Magnitude scale applied to movements below the Fine-Control Threshold."),
            slider_width=160,
            parent=self.humanizationGroup
        )

        self.humanizationStutterProbCard = SliderLabelCard(
            FluentIcon.PAUSE_BOLD,
            t("humanization_stutter_prob", "Stutter Chance"),
            0, 200,
            format_func=lambda v: f"{v/10:.1f}%",
            description=t("humanization_stutter_prob_desc", "Probability per frame of a brief magnitude reduction."),
            slider_width=160,
            parent=self.humanizationGroup
        )

        self.humanizationStutterMinCard = SliderLabelCard(
            FluentIcon.PAUSE_BOLD,
            t("humanization_stutter_min", "Stutter Min"),
            0, 100,
            format_func=lambda v: f"{v/100:.2f}",
            description=t("humanization_stutter_min_desc", "Lower bound of the stutter magnitude factor."),
            slider_width=160,
            parent=self.humanizationGroup
        )

        self.humanizationStutterMaxCard = SliderLabelCard(
            FluentIcon.PAUSE_BOLD,
            t("humanization_stutter_max", "Stutter Max"),
            0, 100,
            format_func=lambda v: f"{v/100:.2f}",
            description=t("humanization_stutter_max_desc", "Upper bound of the stutter magnitude factor."),
            slider_width=160,
            parent=self.humanizationGroup
        )

        self.humanizationReactionSkipProbCard = SliderLabelCard(
            FluentIcon.PAUSE_BOLD,
            t("humanization_reaction_skip_prob", "Skip Chance"),
            0, 100,
            format_func=lambda v: f"{v/10:.1f}%",
            description=t("humanization_reaction_skip_prob_desc", "Probability per frame of skipping the mouse injection entirely."),
            slider_width=160,
            parent=self.humanizationGroup
        )

        self.humanizationResetBtn = PushButton(t("humanization_reset", "Reset to Defaults"))
        self.humanizationResetBtn.setFixedWidth(160)
        self.humanizationResetCard = SettingCard(
            FluentIcon.ROTATE,
            t("humanization_reset", "Reset to Defaults"),
            t("humanization_reset_desc", "Reset the whole Humanization section — Intensity, every feature toggle, and every fine-tuning slider above — back to default values."),
            self.humanizationGroup
        )
        self.humanizationResetCard.hBoxLayout.addWidget(self.humanizationResetBtn, 0)
        self.humanizationResetCard.hBoxLayout.addSpacing(16)

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
            t("kalman_enabled_desc", "2D Kalman filter for aim-point smoothing."),
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

        self.camMotionCompCard = SwitchSettingCard(
            FluentIcon.GLOBE,
            t("cam_motion_comp_enabled", "Camera Motion Compensation"),
            t("cam_motion_comp_desc", "Subtract per-frame global scene shift (phase correlation) from aim error to cancel camera shake."),
            parent=self.trackingGroup
        )

        self.camMotionCompSizeSegment = SegmentedWidget()
        for _lbl, _key in [("128", "128"), ("256", "256")]:
            self.camMotionCompSizeSegment.addItem(routeKey=_key, text=_lbl)
        self.camMotionCompSizeSegment.setCurrentItem("128")
        self.camMotionCompSizeCard = SettingCard(
            FluentIcon.ZOOM_IN,
            t("cam_motion_comp_size", "Compensation Resolution"),
            t("cam_motion_comp_size_desc", "128 = ~0.2 ms (recommended), 256 = ~0.5 ms (more precise)"),
            self.trackingGroup
        )
        self.camMotionCompSizeCard.hBoxLayout.addWidget(self.camMotionCompSizeSegment, 0, Qt.AlignmentFlag.AlignRight)
        self.camMotionCompSizeCard.hBoxLayout.addSpacing(16)

        # === Target Area (shared by aim-point calculation and auto-fire hit zone) ===
        self.targetAreaGroup = SettingCardGroup(t("target_area_settings"), self.scrollWidget)
        self.aimPreview = _AimPointPreview(self.targetAreaGroup)

        self.customYCard = SliderLabelCard(
            FluentIcon.MOVE,
            t("aim_custom_y_pct", "Custom Aim Y Position (%)"),
            0, 100,
            format_func=lambda v: f"{v}%",
            description=t("aim_custom_y_desc", "0% = top of box, 100% = bottom. ~20% = head, ~60% = body."),
            slider_width=200,
            parent=self.targetAreaGroup
        )

        self.headWidthCard = SliderLabelCard(
            FluentIcon.CONSTRACT,
            t("head_width_ratio"),
            10, 100,
            format_func=lambda v: f"{v}%",
            slider_width=200,
            parent=self.targetAreaGroup
        )

        self.headHeightCard = SliderLabelCard(
            FluentIcon.FIT_PAGE,
            t("head_height_ratio"),
            10, 100,
            format_func=lambda v: f"{v}%",
            description=t("body_height_note"),
            slider_width=200,
            parent=self.targetAreaGroup
        )

        self.bodyWidthCard = SliderLabelCard(
            FluentIcon.CONSTRACT,
            t("body_width_ratio"),
            10, 100,
            format_func=lambda v: f"{v}%",
            slider_width=200,
            parent=self.targetAreaGroup
        )

        self.adaptiveRatioCard = SwitchSettingCard(
            FluentIcon.FIT_PAGE,
            t("aim_adaptive_ratio_enabled", "Distance-Adaptive Ratio"),
            t("aim_adaptive_ratio_desc", "Scale head ratio inversely with box size — keeps head aim accurate from close to long range."),
            parent=self.targetAreaGroup
        )

        self.adaptiveRatioRefHCard = SliderLabelCard(
            FluentIcon.ZOOM_IN,
            t("aim_adaptive_ratio_ref_h", "Reference Box Height"),
            20, 200,
            format_func=lambda v: f"{v} px",
            description=t("aim_adaptive_ratio_ref_h_desc", "Box height (px) where head ratio is nominal. Match to your typical close-range target."),
            slider_width=180,
            parent=self.targetAreaGroup
        )

        self.postureAwareCard = SwitchSettingCard(
            FluentIcon.PEOPLE,
            t("aim_posture_aware_enabled", "Posture-Aware Targeting"),
            t("aim_posture_aware_desc", "Fall back to center-mass when box is wider than tall (crouch / slide / prone)."),
            parent=self.targetAreaGroup
        )

        self.crouchAspectCard = SliderLabelCard(
            FluentIcon.CONSTRACT,
            t("aim_crouch_aspect_threshold", "Crouch Aspect Threshold"),
            80, 200,
            format_func=lambda v: f"{v / 100:.1f}×",
            description=t("aim_crouch_aspect_desc", "box_w / box_h above which player is treated as crouching. Default 1.2×."),
            slider_width=180,
            parent=self.targetAreaGroup
        )

        # Target class multi-select — header card + the dynamic checkbox
        # list itself. Both start hidden; _loadFromConfig() shows them only
        # once the loaded model actually reports >= 2 class names (see
        # _TargetClassSelector.refresh()'s return value).
        self.targetClassCard = SettingCard(
            FluentIcon.PEOPLE,
            t("aim_target_class_ids", "Target Classes"),
            t("aim_target_class_ids_desc",
              "Choose which detected classes count as a valid aim target (e.g. never aim at a teammate class). All checked = no restriction."),
            parent=self.targetAreaGroup
        )
        self.targetClassCard.setVisible(False)
        self.targetClassSelector = _TargetClassSelector(self.targetAreaGroup)

    def _initLayout(self):
        """Layout all controls"""
        # General
        self.generalGroup.addSettingCard(self.aimPartCard)
        self.generalGroup.addSettingCard(self.mouseMoveCard)
        self.addContent(self.generalGroup)

        # Arduino
        self.arduinoGroup.addSettingCard(self.comPortCard)
        self.arduinoGroup.addSettingCard(self.arduinoBaudCard)
        self.arduinoGroup.addSettingCard(self.connectionCard)
        self.arduinoGroup.addSettingCard(self.arduinoConnectCard)
        self.arduinoGroup.addSettingCard(self.guideCard)
        self.arduinoGroup.addSettingCard(self.spoofCard)
        self.arduinoGroup.addSettingCard(self.verifySpoofCard)
        self.arduinoGroup.addSettingCard(self.testHeartCard)
        self.addContent(self.arduinoGroup)
        self.arduinoGroup.setVisible(False)

        # Xbox
        self.xboxGroup.addSettingCard(self.xboxSensitivityCard)
        self.xboxGroup.addSettingCard(self.xboxDeadzoneCard)
        self.xboxGroup.addSettingCard(self.xboxConnectionCard)
        self.xboxGroup.addSettingCard(self.xboxConnectCard)
        self.addContent(self.xboxGroup)
        self.xboxGroup.setVisible(False)

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
        xPageLayout.addStretch(1)

        self.pidYPage = QWidget()
        yPageLayout = QVBoxLayout(self.pidYPage)
        yPageLayout.setContentsMargins(0, 0, 0, 0)
        yPageLayout.setSpacing(0)
        yPageLayout.addWidget(self.pidPyCard)
        yPageLayout.addWidget(self.pidIyCard)
        yPageLayout.addWidget(self.pidDyCard)
        yPageLayout.addStretch(1)

        self.pidStackedWidget.addWidget(self.pidXPage)
        self.pidStackedWidget.addWidget(self.pidYPage)

        self.pidGroup.addSettingCard(self.pidUnsafeCard)
        self.pidGroup.vBoxLayout.addWidget(pivotWidget)
        self.pidGroup.vBoxLayout.addWidget(self.pidStackedWidget)
        self.pidGroup.addSettingCard(self.maxMovePerFrameCard)
        self.addContent(self.pidGroup)

        # Y-Axis Recoil Suppression (separate group so X tab has no height gap)
        self.yReduceGroup.addSettingCard(self.pidYReduceEnableCard)
        self.yReduceGroup.addSettingCard(self.pidYReduceDelayCard)
        self.yReduceGroup.addSettingCard(self.pidYReduceFloorCard)
        self.yReduceGroup.addSettingCard(self.pidYReduceRampCard)
        self.yReduceGroup.addSettingCard(self.pidYReduceSettleCard)
        self.yReduceGroup.addSettingCard(self.pidYReduceVelCard)
        self.addContent(self.yReduceGroup)

        # Humanization
        self.humanizationGroup.addSettingCard(self.humanizationEnableCard)
        self.humanizationGroup.addSettingCard(self.humanizationIntensityCard)
        self.humanizationGroup.addSettingCard(self.humanizationMicroJitterCard)
        self.humanizationGroup.addSettingCard(self.humanizationJitterBaseCard)
        self.humanizationGroup.addSettingCard(self.humanizationJitterScaleCard)
        self.humanizationGroup.addSettingCard(self.humanizationMicroJitterIdleCard)
        self.humanizationGroup.addSettingCard(self.humanizationMotionVariationCard)
        self.humanizationGroup.addSettingCard(self.humanizationMotionVariationRangeCard)
        self.humanizationGroup.addSettingCard(self.humanizationSpeedShapingCard)
        self.humanizationGroup.addSettingCard(self.humanizationSpeedShapingLowCard)
        self.humanizationGroup.addSettingCard(self.humanizationSpeedShapingHighCard)
        self.humanizationGroup.addSettingCard(self.humanizationSpeedShapingLowFactorCard)
        self.humanizationGroup.addSettingCard(self.humanizationMicroStutterCard)
        self.humanizationGroup.addSettingCard(self.humanizationStutterProbCard)
        self.humanizationGroup.addSettingCard(self.humanizationStutterMinCard)
        self.humanizationGroup.addSettingCard(self.humanizationStutterMaxCard)
        self.humanizationGroup.addSettingCard(self.humanizationReactionVariabilityCard)
        self.humanizationGroup.addSettingCard(self.humanizationReactionSkipProbCard)
        self.humanizationGroup.addSettingCard(self.humanizationResetCard)
        self.addContent(self.humanizationGroup)

        # Target Priority
        self.targetPriorityGroup.addSettingCard(self.targetPriorityModeCard)
        self.targetPriorityGroup.addSettingCard(self.targetPriorityWeightCard)
        self.addContent(self.targetPriorityGroup)

        # Target Tracking
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
        self.trackingGroup.addSettingCard(self.camMotionCompCard)
        self.trackingGroup.addSettingCard(self.camMotionCompSizeCard)
        self.addContent(self.trackingGroup)

        # Target Area (shared aim-point geometry + auto-fire hit zone)
        self.targetAreaGroup.addSettingCard(self.aimPreview)
        self.targetAreaGroup.addSettingCard(self.customYCard)
        self.targetAreaGroup.addSettingCard(self.headWidthCard)
        self.targetAreaGroup.addSettingCard(self.headHeightCard)
        self.targetAreaGroup.addSettingCard(self.bodyWidthCard)
        self.targetAreaGroup.addSettingCard(self.adaptiveRatioCard)
        self.targetAreaGroup.addSettingCard(self.adaptiveRatioRefHCard)
        self.targetAreaGroup.addSettingCard(self.postureAwareCard)
        self.targetAreaGroup.addSettingCard(self.crouchAspectCard)
        self.targetAreaGroup.addSettingCard(self.targetClassCard)
        self.targetAreaGroup.addSettingCard(self.targetClassSelector)
        self.addContent(self.targetAreaGroup)

        self.scrollLayout.addStretch(1)

    def _connectSignals(self):
        """Connect signals"""
        # General
        self.aimPartCombo.currentIndexChanged.connect(self._onAimPartChanged)
        self.mouseMoveCombo.currentTextChanged.connect(self._onMouseMoveChanged)

        # Arduino
        self.comRefreshBtn.clicked.connect(self._refreshComPorts)
        self.comPortCombo.currentTextChanged.connect(self._onComPortChanged)
        self.arduinoConnectBtn.clicked.connect(self._onArduinoConnectToggle)
        self.guideBtn.clicked.connect(self._onOpenGuide)
        self.spoofBtn.clicked.connect(self._onSpoofDevice)
        self.verifySpoofBtn.clicked.connect(self._onVerifySpoof)
        self.testHeartBtn.clicked.connect(self._onTestHeart)
        self.arduinoBaudCombo.currentTextChanged.connect(self._onArduinoBaudChanged)

        # Xbox
        self.xboxSensitivityCard.valueChanged.connect(self._onXboxSensitivityChanged)
        self.xboxDeadzoneCard.valueChanged.connect(self._onXboxDeadzoneChanged)
        self.xboxConnectBtn.clicked.connect(self._onXboxConnectToggle)

        # PID
        self.pidUnsafeCard.checkedChanged.connect(self._onPidUnsafeChanged)
        self.pidPxCard.valueChanged.connect(lambda v: self._onPidChanged('pid_kp_x', v))
        self.pidIxCard.valueChanged.connect(lambda v: self._onPidChanged('pid_ki_x', v))
        self.pidDxCard.valueChanged.connect(lambda v: self._onPidChanged('pid_kd_x', v))
        self.pidPyCard.valueChanged.connect(lambda v: self._onPidChanged('pid_kp_y', v))
        self.pidIyCard.valueChanged.connect(lambda v: self._onPidChanged('pid_ki_y', v))
        self.pidDyCard.valueChanged.connect(lambda v: self._onPidChanged('pid_kd_y', v))
        self.maxMovePerFrameCard.valueChanged.connect(
            lambda v: setattr(self._config, 'max_move_per_frame_px', float(v)) if self._config else None)
        self.pidYReduceEnableCard.checkedChanged.connect(lambda checked: self._onPidChanged('aim_y_reduce_enabled', checked, is_bool=True))
        self.pidYReduceDelayCard.valueChanged.connect(lambda v: self._onPidChanged('aim_y_reduce_delay', v))
        self.pidYReduceFloorCard.valueChanged.connect(lambda v: self._onPidChanged('aim_y_reduce_floor', v))
        self.pidYReduceRampCard.valueChanged.connect(lambda v: self._onPidChanged('aim_y_reduce_ramp', v))
        self.pidYReduceSettleCard.valueChanged.connect(
            lambda v: setattr(self._config, 'aim_y_reduce_settle_px', float(v)) if self._config else None)
        self.pidYReduceVelCard.valueChanged.connect(
            lambda v: setattr(self._config, 'aim_y_vel_restore_px_s', float(v)) if self._config else None)

        # Humanization
        self.humanizationEnableCard.checkedChanged.connect(self._onHumanizationEnableChanged)
        self.humanizationIntensityCard.valueChanged.connect(self._onHumanizationIntensityChanged)
        self.humanizationMicroJitterCard.checkedChanged.connect(self._onHumanizationMicroJitterChanged)
        self.humanizationMotionVariationCard.checkedChanged.connect(self._onHumanizationMotionVariationChanged)
        self.humanizationSpeedShapingCard.checkedChanged.connect(self._onHumanizationSpeedShapingChanged)
        self.humanizationMicroStutterCard.checkedChanged.connect(self._onHumanizationMicroStutterChanged)
        self.humanizationReactionVariabilityCard.checkedChanged.connect(self._onHumanizationReactionVariabilityChanged)
        self.humanizationJitterBaseCard.valueChanged.connect(self._onHumanizationJitterBaseChanged)
        self.humanizationJitterScaleCard.valueChanged.connect(self._onHumanizationJitterScaleChanged)
        self.humanizationMicroJitterIdleCard.checkedChanged.connect(self._onHumanizationMicroJitterIdleChanged)
        self.humanizationMotionVariationRangeCard.valueChanged.connect(self._onHumanizationMotionVariationRangeChanged)
        self.humanizationSpeedShapingLowCard.valueChanged.connect(self._onHumanizationSpeedShapingLowChanged)
        self.humanizationSpeedShapingHighCard.valueChanged.connect(self._onHumanizationSpeedShapingHighChanged)
        self.humanizationSpeedShapingLowFactorCard.valueChanged.connect(self._onHumanizationSpeedShapingLowFactorChanged)
        self.humanizationStutterProbCard.valueChanged.connect(self._onHumanizationStutterProbChanged)
        self.humanizationStutterMinCard.valueChanged.connect(self._onHumanizationStutterMinChanged)
        self.humanizationStutterMaxCard.valueChanged.connect(self._onHumanizationStutterMaxChanged)
        self.humanizationReactionSkipProbCard.valueChanged.connect(self._onHumanizationReactionSkipProbChanged)
        self.humanizationResetBtn.clicked.connect(self._onHumanizationResetClicked)

        # Target Priority
        self.targetPriorityModeCombo.currentTextChanged.connect(self._onTargetPriorityModeChanged)
        self.targetPriorityWeightCard.valueChanged.connect(self._onTargetPriorityWeightChanged)

        # Target Tracking
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
        self.camMotionCompCard.checkedChanged.connect(self._onCamMotionCompChanged)
        self.camMotionCompSizeSegment.currentItemChanged.connect(self._onCamMotionCompSizeChanged)

        # Target Area
        self.headWidthCard.valueChanged.connect(self._onHeadWidthChanged)
        self.headHeightCard.valueChanged.connect(self._onHeadHeightChanged)
        self.bodyWidthCard.valueChanged.connect(self._onBodyWidthChanged)
        self.customYCard.valueChanged.connect(self._onCustomYChanged)
        self.adaptiveRatioCard.checkedChanged.connect(self._onAdaptiveRatioChanged)
        self.adaptiveRatioRefHCard.valueChanged.connect(self._onAdaptiveRatioRefHChanged)
        self.postureAwareCard.checkedChanged.connect(self._onPostureAwareChanged)
        self.crouchAspectCard.valueChanged.connect(self._onCrouchAspectChanged)
        self.targetClassSelector.selectionChanged.connect(self._onTargetClassSelectionChanged)

    def _loadFromConfig(self):
        """Load values from Config"""
        if not self._config:
            return
        self._isLoadingConfig = True
        try:
            # General
            aim_parts = ["head", "body", "center", "custom"]
            part = self._config.aim_part if self._config.aim_part in aim_parts else "center"
            self.aimPartCombo.setCurrentIndex(aim_parts.index(part))
            self._updateTargetAreaVisibility(part)

            mouse_methods = ["ddxoft", "mouse_event", "sendinput", "arduino", "makcu", "xbox"]
            if self._config.mouse_move_method in mouse_methods:
                self.mouseMoveCombo.setCurrentIndex(mouse_methods.index(self._config.mouse_move_method))

            self._updateMethodGroupVisibility(self._config.mouse_move_method)

            # Arduino
            if self._config.arduino_com_port:
                idx = self.comPortCombo.findText(self._config.arduino_com_port)
                if idx >= 0:
                    self.comPortCombo.setCurrentIndex(idx)
                elif self.comPortCombo.count() > 1:
                    self.comPortCombo.setCurrentIndex(1)
            elif self.comPortCombo.count() > 1:
                self.comPortCombo.setCurrentIndex(1)

            arduino_baud = str(getattr(self._config, 'arduino_baud_rate', 115200))
            if self.arduinoBaudCombo.findText(arduino_baud) < 0:
                arduino_baud = "115200"
            self.arduinoBaudCombo.setCurrentText(arduino_baud)
            self._updateArduinoConnectionStatus()

            # Xbox
            self.xboxSensitivityCard.setValue(int(getattr(self._config, 'xbox_sensitivity', 1.0) * 100))
            self.xboxDeadzoneCard.setValue(int(getattr(self._config, 'xbox_deadzone', 0.05) * 100))
            self._updateXboxConnectionStatus()

            # PID — Kp sliders are scaled x200 normally (slider 0–100 → config
            # 0.0–0.5) or x100 with Unsafe Mode on (→ config 0.0–1.0); clamped
            # to the slider max so a saved value keeps its exact effective gain.
            self.pidUnsafeCard.setChecked(bool(getattr(self._config, 'pid_unsafe_mode', False)))
            self._applyPidKpFormat(self.pidUnsafeCard.isChecked())
            kp_divisor = self._pidKpDivisor()
            self.pidPxCard.setValue(min(100, int(self._config.pid_kp_x * kp_divisor)))
            self.pidIxCard.setValue(int(self._config.pid_ki_x * 100))
            self.pidDxCard.setValue(int(self._config.pid_kd_x * 100))
            self.pidPyCard.setValue(min(100, int(self._config.pid_kp_y * kp_divisor)))
            self.pidIyCard.setValue(int(self._config.pid_ki_y * 100))
            self.pidDyCard.setValue(int(self._config.pid_kd_y * 100))
            self.maxMovePerFrameCard.setValue(min(500, max(0, int(getattr(self._config, 'max_move_per_frame_px', 85.0)))))
            self.pidYReduceEnableCard.setChecked(getattr(self._config, 'aim_y_reduce_enabled', False))
            self.pidYReduceDelayCard.setValue(int(getattr(self._config, 'aim_y_reduce_delay', 0.6) * 100))
            self.pidYReduceFloorCard.setValue(int(getattr(self._config, 'aim_y_reduce_floor', 0.0) * 100))
            self.pidYReduceRampCard.setValue(int(getattr(self._config, 'aim_y_reduce_ramp', 0.0) * 100))
            self.pidYReduceSettleCard.setValue(int(getattr(self._config, 'aim_y_reduce_settle_px', 0.0)))
            self.pidYReduceVelCard.setValue(int(getattr(self._config, 'aim_y_vel_restore_px_s', 0.0)))

            # Humanization
            self._loadHumanizationFromConfig()

            # Target Priority
            mode_map = {"distance": "Distance", "confidence": "Confidence", "composite": "Composite"}
            mode_text = mode_map.get(str(getattr(self._config, 'target_priority_mode', 'distance')), "Distance")
            self.targetPriorityModeCombo.setCurrentText(mode_text)
            self.targetPriorityWeightCard.setValue(int(getattr(self._config, 'target_priority_confidence_weight', 0.5) * 100))

            # Target Tracking
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

            cmc_on = bool(getattr(self._config, 'cam_motion_comp_enabled', False))
            self.camMotionCompCard.setChecked(cmc_on)
            cmc_size = str(getattr(self._config, 'cam_motion_comp_size', 128))
            if cmc_size not in ("128", "256"):
                cmc_size = "128"
            self.camMotionCompSizeSegment.setCurrentItem(cmc_size)
            self.camMotionCompSizeCard.setEnabled(cmc_on)

            # Target Area
            self.customYCard.setValue(int(getattr(self._config, 'aim_custom_y_pct', 30.0)))
            self.headWidthCard.setValue(int(self._config.head_width_ratio * 100))
            self.headHeightCard.setValue(int(self._config.head_height_ratio * 100))
            self.bodyWidthCard.setValue(int(self._config.body_width_ratio * 100))
            adaptive_on = bool(getattr(self._config, 'aim_adaptive_ratio_enabled', False))
            self.adaptiveRatioCard.setChecked(adaptive_on)
            self.adaptiveRatioRefHCard.setValue(int(getattr(self._config, 'aim_adaptive_ratio_ref_h', 80.0)))
            self.adaptiveRatioRefHCard.setEnabled(adaptive_on)
            posture_on = bool(getattr(self._config, 'aim_posture_aware_enabled', False))
            self.postureAwareCard.setChecked(posture_on)
            self.crouchAspectCard.setValue(int(getattr(self._config, 'aim_crouch_aspect_threshold', 1.2) * 100))
            self.crouchAspectCard.setEnabled(posture_on)

            # Target class multi-select — class names are entirely
            # model-specific (config._detect_class_names, runtime-only,
            # populated on model load/hot-swap), so both the header card and
            # the checkbox list itself only show once the loaded model
            # actually reports >= 2 classes.
            has_classes = self.targetClassSelector.refresh(
                getattr(self._config, '_detect_class_names', None),
                getattr(self._config, 'aim_target_class_ids', []))
            self.targetClassCard.setVisible(has_classes)
        finally:
            self._isLoadingConfig = False

    # ── Helpers ──────────────────────────────────

    @staticmethod
    def _sorted_com_ports(ports):
        def _num(p):
            m = re.search(r'(\d+)$', p.device)
            return int(m.group(1)) if m else 0
        return sorted(ports, key=_num, reverse=True)

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

    def _refreshComPorts(self):
        self.comPortCombo.clear()
        self.comPortCombo.addItem(t("no_com_port"))
        try:
            import serial.tools.list_ports
            for port in self._sorted_com_ports(serial.tools.list_ports.comports()):
                self.comPortCombo.addItem(port.device)
        except ImportError:
            pass

    def _updateMethodGroupVisibility(self, method):
        self.arduinoGroup.setVisible(method == "arduino")
        self.xboxGroup.setVisible(method == "xbox")

    # ── General Callbacks ────────────────────────

    def _onAimPartChanged(self, index):
        parts = ["head", "body", "center", "custom"]
        if self._config and 0 <= index < len(parts):
            self._config.aim_part = parts[index]
        self._updateTargetAreaVisibility(parts[index] if 0 <= index < len(parts) else "head")

    def _updateTargetAreaVisibility(self, aim_part):
        is_smart = aim_part == "center"
        is_custom = aim_part == "custom"
        is_head_or_body = aim_part in ("head", "body")
        uses_custom_y = is_smart or is_custom
        if is_smart:
            suffix = t("aim_smart_mode_note", " — Smart + Custom Y")
        elif is_custom:
            suffix = t("aim_custom_mode_note", " — Custom Y mode")
        else:
            suffix = t("aim_head_body_mode_note", " — Head/Body mode")
        self.targetAreaGroup.titleLabel.setText(t("target_area_settings") + suffix)
        # head_height_ratio and the adaptive-ratio scale it feeds are only
        # consumed by calculate_aim_target()'s Head/Body branches — Smart and
        # Custom modes use aim_custom_y_pct instead, so these do nothing there.
        for card in [self.headHeightCard, self.adaptiveRatioCard, self.adaptiveRatioRefHCard]:
            card.setEnabled(is_head_or_body)
        # head_width_ratio/body_width_ratio are consumed only by auto_fire.py's
        # hit-zone geometry, entirely independent of aim_part — always tunable.
        self.headWidthCard.setEnabled(True)
        self.bodyWidthCard.setEnabled(True)
        # Posture-aware's crouch check runs before the aim_part switch inside
        # calculate_aim_target() and applies to every mode, not just Smart —
        # always tunable so it can't get silently stuck on/off from another tab.
        self.postureAwareCard.setEnabled(True)
        self.crouchAspectCard.setEnabled(True)
        self.customYCard.setEnabled(uses_custom_y)
        head_h = self.headHeightCard.value() if hasattr(self.headHeightCard, 'value') else 20
        head_w = self.headWidthCard.value() if hasattr(self.headWidthCard, 'value') else 38
        body_w = self.bodyWidthCard.value() if hasattr(self.bodyWidthCard, 'value') else 87
        custom_y = self.customYCard.value() if hasattr(self.customYCard, 'value') else 30
        self.aimPreview.setParams(aim_part, head_h, head_w, body_w, custom_y)

    def _onMouseMoveChanged(self, text):
        if self._config:
            self._config.mouse_move_method = text
            if text == "makcu":
                self._config.mouse_click_method = "makcu"
            if text == "ddxoft":
                try:
                    from win_utils import ensure_ddxoft_ready
                    ensure_ddxoft_ready()
                except ImportError:
                    pass
        self._updateMethodGroupVisibility(text)
        # Notify keys page to update MAKCU group visibility
        try:
            win = self.window()
            if hasattr(win, 'keysInterface'):
                win.keysInterface._updateMakcuVisibility()
        except Exception:
            pass

    # ── Arduino Callbacks ────────────────────────

    def _onComPortChanged(self, text):
        if self._config and text != t("no_com_port"):
            self._config.arduino_com_port = text

    def _onArduinoBaudChanged(self, text):
        if self._config and not self._isLoadingConfig:
            try:
                self._config.arduino_baud_rate = int(text)
            except ValueError:
                pass

    def _onArduinoConnectToggle(self):
        try:
            from win_utils import is_arduino_connected, disconnect_arduino
            if is_arduino_connected():
                disconnect_arduino()
                self._updateArduinoConnectionStatus()
            else:
                com_port = self.comPortCombo.currentText()
                if not com_port or com_port == t("no_com_port"):
                    QMessageBox.warning(self, t("config_error"), t("no_com_port"))
                    return
                self._startArduinoConnect(com_port)
        except ImportError:
            QMessageBox.warning(self, t("config_error"), "pyserial not installed.\npip install pyserial")

    def _startArduinoConnect(self, com_port: str) -> None:
        """Run connect_arduino() on a background thread instead of blocking
        the GUI thread for its 2s post-connect wait (Leonardo auto-restart)."""
        if self._arduinoConnectWorker is not None and self._arduinoConnectWorker.isRunning():
            return  # a connect attempt is already in flight
        self.arduinoConnectBtn.setEnabled(False)
        self.connectionLabel.setText(t("connecting", "Connecting..."))
        self._arduinoConnectWorker = _ArduinoConnectWorker(com_port, parent=self)
        self._arduinoConnectWorker.finishedResult.connect(
            lambda ok, port=com_port: self._onArduinoConnectFinished(ok, port))
        self._arduinoConnectWorker.start()

    def _onArduinoConnectFinished(self, ok: bool, com_port: str) -> None:
        self.arduinoConnectBtn.setEnabled(True)
        if not ok:
            QMessageBox.warning(self, t("config_error"),
                                f"Arduino {t('disconnected')}: {com_port}")
        self._updateArduinoConnectionStatus()

    def _updateArduinoConnectionStatus(self):
        try:
            from win_utils import is_arduino_connected
            if is_arduino_connected():
                self._isArduinoConnected = True
                self.connectionLabel.setText(t("connected"))
                self.connectionLabel.setStyleSheet(f"color: {ThemeColors.SUCCESS.get()}; font-weight: bold;")
                self.arduinoConnectBtn.setText(t("arduino_disconnect"))
            else:
                self._isArduinoConnected = False
                self.connectionLabel.setText(t("disconnected"))
                self.connectionLabel.setStyleSheet(f"color: {ThemeColors.ERROR.get()}; font-weight: bold;")
                self.arduinoConnectBtn.setText(t("arduino_connect"))
        except ImportError:
            self.connectionLabel.setText("pyserial N/A")
            self.connectionLabel.setStyleSheet(f"color: {ThemeColors.ERROR.get()}; font-weight: bold;")

    def _onOpenGuide(self):
        from PyQt6.QtCore import QUrl
        guide_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
            "Arduino_User_Guide.html"
        )
        if os.path.exists(guide_path):
            QDesktopServices.openUrl(QUrl.fromLocalFile(guide_path))

    def _onSpoofDevice(self):
        reply = QMessageBox.question(
            self, t("spoof_confirm_title"),
            t("spoof_confirm_msg").replace("\\n", "\n"),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            try:
                from win_utils.arduino_spoofer import spoof_arduino_board
                success, boards_path = spoof_arduino_board()
                if success:
                    QMessageBox.information(self, t("spoof_success_title"),
                                            t("spoof_success_msg").replace("\\n", "\n"))
                else:
                    QMessageBox.warning(self, t("spoof_error_title"),
                                        f"Spoof operation returned unsuccessful.\nFile: {boards_path}")
            except FileNotFoundError as e:
                QMessageBox.warning(self, t("spoof_error_title"), str(e))
            except Exception as e:
                QMessageBox.critical(self, t("spoof_error_title"), f"Error: {e}")

    def _onVerifySpoof(self):
        try:
            from win_utils.arduino_spoofer import verify_spoof
            specific_port = None
            if self._config and self._config.arduino_com_port:
                specific_port = self._config.arduino_com_port
            is_spoofed, message = verify_spoof(specific_port)
            if is_spoofed:
                QMessageBox.information(self, t("verify_success_title"), message)
            else:
                QMessageBox.warning(self, t("verify_fail_title"), message)
        except Exception as e:
            QMessageBox.critical(self, t("verify_fail_title"), f"Error: {e}")

    def _onTestHeart(self):
        import math as _math
        import threading as _threading
        import time as _time
        reply = QMessageBox.question(
            self, t("test_heart_confirm_title"),
            t("test_heart_confirm_msg").replace("\\n", "\n"),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            from win_utils.arduino_mouse import arduino_mouse
            if not arduino_mouse.is_connected():
                com_port = self._config.arduino_com_port if self._config else ""
                if not com_port:
                    QMessageBox.warning(self, t("test_heart_confirm_title"),
                                        "Arduino not connected. Please set COM port first.")
                    return
                if not arduino_mouse.connect(com_port):
                    QMessageBox.warning(self, t("test_heart_confirm_title"),
                                        f"Failed to connect to {com_port}.")
                    return

            def _draw_heart():
                num_steps = 120
                scale = 3.0
                points = []
                for i in range(num_steps + 1):
                    angle = 2 * _math.pi * i / num_steps
                    x = 16 * (_math.sin(angle) ** 3)
                    y = -(13 * _math.cos(angle) - 5 * _math.cos(2 * angle)
                           - 2 * _math.cos(3 * angle) - _math.cos(4 * angle))
                    points.append((x * scale, y * scale))
                for i in range(1, len(points)):
                    dx = int(round(points[i][0] - points[i - 1][0]))
                    dy = int(round(points[i][1] - points[i - 1][1]))
                    if dx != 0 or dy != 0:
                        arduino_mouse.move(dx, dy)
                    _time.sleep(0.015)

            _threading.Thread(target=_draw_heart, daemon=True).start()

    # ── Xbox Callbacks ───────────────────────────

    def _onXboxSensitivityChanged(self, value):
        if self._config:
            self._config.xbox_sensitivity = value / 100.0
            try:
                from win_utils import set_xbox_sensitivity
                set_xbox_sensitivity(value / 100.0)
            except ImportError:
                pass

    def _onXboxDeadzoneChanged(self, value):
        if self._config:
            self._config.xbox_deadzone = value / 100.0
            try:
                from win_utils import set_xbox_deadzone
                set_xbox_deadzone(value / 100.0)
            except ImportError:
                pass

    def _onXboxConnectToggle(self):
        try:
            from win_utils import is_xbox_connected, connect_xbox, disconnect_xbox
            if is_xbox_connected():
                disconnect_xbox()
            else:
                connect_xbox()
            self._updateXboxConnectionStatus()
        except ImportError:
            QMessageBox.warning(self, t("config_error"),
                                 "vgamepad not installed.\npip install vgamepad\nInstall ViGEmBus driver.")

    def _updateXboxConnectionStatus(self):
        try:
            from win_utils import is_xbox_connected, is_xbox_available
            if not is_xbox_available():
                self.xboxConnectionLabel.setText("vgamepad " + t("disconnected"))
                self.xboxConnectionLabel.setStyleSheet(f"color: {ThemeColors.ERROR.get()}; font-weight: bold;")
                self.xboxConnectBtn.setText(t("xbox_connect"))
                return
            if is_xbox_connected():
                self._isXboxConnected = True
                self.xboxConnectionLabel.setText(t("connected"))
                self.xboxConnectionLabel.setStyleSheet(f"color: {ThemeColors.SUCCESS.get()}; font-weight: bold;")
                self.xboxConnectBtn.setText(t("xbox_disconnect"))
            else:
                self._isXboxConnected = False
                self.xboxConnectionLabel.setText(t("disconnected"))
                self.xboxConnectionLabel.setStyleSheet(f"color: {ThemeColors.ERROR.get()}; font-weight: bold;")
                self.xboxConnectBtn.setText(t("xbox_connect"))
        except ImportError:
            self.xboxConnectionLabel.setText("vgamepad N/A")
            self.xboxConnectionLabel.setStyleSheet(f"color: {ThemeColors.ERROR.get()}; font-weight: bold;")

    # === PID Callbacks ===

    def _onPidAxisChanged(self, routeKey: str):
        self.pidStackedWidget.setCurrentIndex(0 if routeKey == 'x' else 1)

    def _pidKpDivisor(self) -> float:
        """200.0 normally (slider 0-100 -> config 0.0-0.5), 100.0 with Unsafe
        Mode on (-> config 0.0-1.0). Both P sliders share one divisor."""
        unsafe = bool(self._config and getattr(self._config, 'pid_unsafe_mode', False))
        return 100.0 if unsafe else 200.0

    def _applyPidKpFormat(self, unsafe: bool):
        """Point both Kp sliders' labels at the divisor matching `unsafe`."""
        divisor = 100.0 if unsafe else 200.0
        self.pidPxCard.setFormatFunc(lambda v, _d=divisor: f"{v/_d:.2f}")
        self.pidPyCard.setFormatFunc(lambda v, _d=divisor: f"{v/_d:.2f}")

    def _onPidUnsafeChanged(self, checked: bool):
        checked = bool(checked)
        if self._config:
            self._config.pid_unsafe_mode = checked
        self._applyPidKpFormat(checked)
        if not checked:
            # Leaving Unsafe Mode: clamp any Kp that's currently above the
            # safe 0.50 cap back down so the slider (now capped at 0.50
            # again) and the stored config value can't silently disagree.
            if self._config:
                self._config.pid_kp_x = min(self._config.pid_kp_x, 0.5)
                self._config.pid_kp_y = min(self._config.pid_kp_y, 0.5)
        kp_divisor = self._pidKpDivisor()
        if self._config:
            self.pidPxCard.setValue(min(100, int(self._config.pid_kp_x * kp_divisor)))
            self.pidPyCard.setValue(min(100, int(self._config.pid_kp_y * kp_divisor)))

    def _onPidChanged(self, attr, value, is_bool=False):
        if self._config:
            if is_bool:
                setattr(self._config, attr, value)
            else:
                # Kp sliders are scaled so full travel (0–100) spans config 0.0–0.5
                # normally, or 0.0-1.0 with Unsafe Mode on — see _pidKpDivisor().
                # Ki/Kd always use /100.
                divisor = self._pidKpDivisor() if attr in ('pid_kp_x', 'pid_kp_y') else 100.0
                setattr(self._config, attr, value / divisor)

    # === Humanization Callbacks ===

    def _loadHumanizationFromConfig(self):
        """Push self._config.humanization into every Humanization widget,
        including the fine-tuning sub-parameter sliders. Shared by
        _loadFromConfig() and the Reset to Defaults button so both stay in
        sync with one code path."""
        if not self._config:
            return
        hcfg = getattr(self._config, 'humanization', None)
        h_on = bool(getattr(hcfg, 'enabled', False))
        self.humanizationEnableCard.setChecked(h_on)
        self.humanizationIntensityCard.setValue(int(getattr(hcfg, 'intensity', 0.5) * 100))
        self.humanizationMicroJitterCard.setChecked(bool(getattr(hcfg, 'micro_jitter_enabled', True)))
        self.humanizationJitterBaseCard.setValue(
            min(200, max(0, int(getattr(hcfg, 'micro_jitter_base', 0.20) * 100))))
        self.humanizationJitterScaleCard.setValue(
            min(200, max(0, int(getattr(hcfg, 'micro_jitter_scale', 0.025) * 1000))))
        self.humanizationMicroJitterIdleCard.setChecked(bool(getattr(hcfg, 'micro_jitter_idle_enabled', False)))
        self.humanizationMotionVariationCard.setChecked(bool(getattr(hcfg, 'motion_variation_enabled', True)))
        self.humanizationMotionVariationRangeCard.setValue(
            min(200, max(0, int(getattr(hcfg, 'motion_variation_range', 0.06) * 1000))))
        self.humanizationSpeedShapingCard.setChecked(bool(getattr(hcfg, 'speed_shaping_enabled', True)))
        self.humanizationSpeedShapingLowCard.setValue(
            min(100, max(0, int(getattr(hcfg, 'speed_shaping_low', 4.0) * 5))))
        self.humanizationSpeedShapingHighCard.setValue(
            min(100, max(0, int(getattr(hcfg, 'speed_shaping_high', 22.0) * 2))))
        self.humanizationSpeedShapingLowFactorCard.setValue(
            min(100, max(0, int(getattr(hcfg, 'speed_shaping_low_factor', 0.88) * 100))))
        self.humanizationMicroStutterCard.setChecked(bool(getattr(hcfg, 'micro_stutter_enabled', False)))
        self.humanizationStutterProbCard.setValue(
            min(200, max(0, int(getattr(hcfg, 'micro_stutter_prob', 0.03) * 1000))))
        self.humanizationStutterMinCard.setValue(
            min(100, max(0, int(getattr(hcfg, 'micro_stutter_min', 0.65) * 100))))
        self.humanizationStutterMaxCard.setValue(
            min(100, max(0, int(getattr(hcfg, 'micro_stutter_max', 0.90) * 100))))
        self.humanizationReactionVariabilityCard.setChecked(bool(getattr(hcfg, 'reaction_variability_enabled', False)))
        self.humanizationReactionSkipProbCard.setValue(
            min(100, max(0, int(getattr(hcfg, 'reaction_skip_prob', 0.015) * 1000))))
        self.humanizationIntensityCard.setEnabled(h_on)
        self.humanizationMicroJitterCard.setEnabled(h_on)
        self.humanizationMotionVariationCard.setEnabled(h_on)
        self.humanizationSpeedShapingCard.setEnabled(h_on)
        self.humanizationMicroStutterCard.setEnabled(h_on)
        self.humanizationReactionVariabilityCard.setEnabled(h_on)
        self._updateHumanizationSubEnabled()

    def _updateHumanizationSubEnabled(self):
        """A sub-parameter slider is shown (and enabled) only while both the
        master Humanization switch AND its own feature's toggle are on —
        hidden rather than just greyed out, so an off feature doesn't leave
        clutter behind."""
        h_on = self.humanizationEnableCard.isChecked()
        jitter_on = h_on and self.humanizationMicroJitterCard.isChecked()
        for card in (self.humanizationJitterBaseCard, self.humanizationJitterScaleCard,
                     self.humanizationMicroJitterIdleCard):
            card.setEnabled(jitter_on)
            card.setVisible(jitter_on)
        variation_on = h_on and self.humanizationMotionVariationCard.isChecked()
        self.humanizationMotionVariationRangeCard.setEnabled(variation_on)
        self.humanizationMotionVariationRangeCard.setVisible(variation_on)
        shaping_on = h_on and self.humanizationSpeedShapingCard.isChecked()
        for card in (self.humanizationSpeedShapingLowCard, self.humanizationSpeedShapingHighCard,
                     self.humanizationSpeedShapingLowFactorCard):
            card.setEnabled(shaping_on)
            card.setVisible(shaping_on)
        stutter_on = h_on and self.humanizationMicroStutterCard.isChecked()
        for card in (self.humanizationStutterProbCard, self.humanizationStutterMinCard,
                     self.humanizationStutterMaxCard):
            card.setEnabled(stutter_on)
            card.setVisible(stutter_on)
        reaction_on = h_on and self.humanizationReactionVariabilityCard.isChecked()
        self.humanizationReactionSkipProbCard.setEnabled(reaction_on)
        self.humanizationReactionSkipProbCard.setVisible(reaction_on)

    def _onHumanizationEnableChanged(self, checked):
        h_on = bool(checked)
        if self._config and getattr(self._config, 'humanization', None) is not None:
            self._config.humanization.enabled = h_on
        self.humanizationIntensityCard.setEnabled(h_on)
        self.humanizationMicroJitterCard.setEnabled(h_on)
        self.humanizationMotionVariationCard.setEnabled(h_on)
        self.humanizationSpeedShapingCard.setEnabled(h_on)
        self.humanizationMicroStutterCard.setEnabled(h_on)
        self.humanizationReactionVariabilityCard.setEnabled(h_on)
        self._updateHumanizationSubEnabled()

    def _onHumanizationIntensityChanged(self, value):
        if self._config and getattr(self._config, 'humanization', None) is not None:
            self._config.humanization.intensity = value / 100.0

    def _onHumanizationMicroJitterChanged(self, checked):
        if self._config and getattr(self._config, 'humanization', None) is not None:
            self._config.humanization.micro_jitter_enabled = bool(checked)
        self._updateHumanizationSubEnabled()

    def _onHumanizationMotionVariationChanged(self, checked):
        if self._config and getattr(self._config, 'humanization', None) is not None:
            self._config.humanization.motion_variation_enabled = bool(checked)
        self._updateHumanizationSubEnabled()

    def _onHumanizationSpeedShapingChanged(self, checked):
        if self._config and getattr(self._config, 'humanization', None) is not None:
            self._config.humanization.speed_shaping_enabled = bool(checked)
        self._updateHumanizationSubEnabled()

    def _onHumanizationMicroStutterChanged(self, checked):
        if self._config and getattr(self._config, 'humanization', None) is not None:
            self._config.humanization.micro_stutter_enabled = bool(checked)
        self._updateHumanizationSubEnabled()

    def _onHumanizationReactionVariabilityChanged(self, checked):
        if self._config and getattr(self._config, 'humanization', None) is not None:
            self._config.humanization.reaction_variability_enabled = bool(checked)
        self._updateHumanizationSubEnabled()

    def _onHumanizationJitterBaseChanged(self, value):
        if self._config and getattr(self._config, 'humanization', None) is not None:
            self._config.humanization.micro_jitter_base = value / 100.0

    def _onHumanizationJitterScaleChanged(self, value):
        if self._config and getattr(self._config, 'humanization', None) is not None:
            self._config.humanization.micro_jitter_scale = value / 1000.0

    def _onHumanizationMicroJitterIdleChanged(self, checked):
        if self._config and getattr(self._config, 'humanization', None) is not None:
            self._config.humanization.micro_jitter_idle_enabled = bool(checked)

    def _onHumanizationMotionVariationRangeChanged(self, value):
        if self._config and getattr(self._config, 'humanization', None) is not None:
            self._config.humanization.motion_variation_range = value / 1000.0

    def _onHumanizationSpeedShapingLowChanged(self, value):
        if self._config and getattr(self._config, 'humanization', None) is not None:
            self._config.humanization.speed_shaping_low = value / 5.0

    def _onHumanizationSpeedShapingHighChanged(self, value):
        if self._config and getattr(self._config, 'humanization', None) is not None:
            self._config.humanization.speed_shaping_high = value / 2.0

    def _onHumanizationSpeedShapingLowFactorChanged(self, value):
        if self._config and getattr(self._config, 'humanization', None) is not None:
            self._config.humanization.speed_shaping_low_factor = value / 100.0

    def _onHumanizationStutterProbChanged(self, value):
        if self._config and getattr(self._config, 'humanization', None) is not None:
            self._config.humanization.micro_stutter_prob = value / 1000.0

    def _onHumanizationStutterMinChanged(self, value):
        if self._config and getattr(self._config, 'humanization', None) is not None:
            self._config.humanization.micro_stutter_min = value / 100.0

    def _onHumanizationStutterMaxChanged(self, value):
        if self._config and getattr(self._config, 'humanization', None) is not None:
            self._config.humanization.micro_stutter_max = value / 100.0

    def _onHumanizationReactionSkipProbChanged(self, value):
        if self._config and getattr(self._config, 'humanization', None) is not None:
            self._config.humanization.reaction_skip_prob = value / 1000.0

    def _onHumanizationResetClicked(self):
        """Reset the entire Humanization block (master toggle, Intensity,
        every feature toggle, and every fine-tuning slider) back to the
        HumanizationConfig dataclass defaults, then refresh the widgets."""
        if not self._config:
            return
        from core.humanization import HumanizationConfig
        self._config.humanization = HumanizationConfig()
        self._loadHumanizationFromConfig()

    # === Target Priority Callbacks ===

    def _onTargetPriorityModeChanged(self, text):
        if self._config:
            self._config.target_priority_mode = str(text).lower()

    def _onTargetPriorityWeightChanged(self, value):
        if self._config:
            self._config.target_priority_confidence_weight = value / 100.0

    # === Target Tracking Callbacks ===

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
        self.kalmanProcessNoiseCard.setEnabled(bool(checked))
        self.kalmanMeasNoiseCard.setEnabled(bool(checked))

    def _onKalmanProcessNoiseChanged(self, value):
        if self._config:
            self._config.kalman_process_noise = value / 100.0

    def _onKalmanMeasNoiseChanged(self, value):
        if self._config:
            self._config.kalman_measurement_noise = value / 100.0

    def _onCamMotionCompChanged(self, checked):
        if self._config:
            self._config.cam_motion_comp_enabled = bool(checked)
        self.camMotionCompSizeCard.setEnabled(bool(checked))

    def _onCamMotionCompSizeChanged(self, key):
        if self._config:
            self._config.cam_motion_comp_size = int(key)

    def _onHeadWidthChanged(self, value):
        if self._config:
            self._config.head_width_ratio = value / 100.0
        self._refreshPreview()

    def _onHeadHeightChanged(self, value):
        if self._config:
            self._config.head_height_ratio = value / 100.0
        self._refreshPreview()

    def _onBodyWidthChanged(self, value):
        if self._config:
            self._config.body_width_ratio = value / 100.0
        self._refreshPreview()

    def _onCustomYChanged(self, value):
        if self._config:
            self._config.aim_custom_y_pct = float(value)
        self._refreshPreview()

    def _refreshPreview(self):
        parts = ["head", "body", "center", "custom"]
        idx = self.aimPartCombo.currentIndex()
        aim_part = parts[idx] if 0 <= idx < len(parts) else "center"
        self.aimPreview.setParams(
            aim_part,
            self.headHeightCard.value() if hasattr(self.headHeightCard, 'value') else 20,
            self.headWidthCard.value()  if hasattr(self.headWidthCard,  'value') else 38,
            self.bodyWidthCard.value()  if hasattr(self.bodyWidthCard,  'value') else 87,
            self.customYCard.value()    if hasattr(self.customYCard,    'value') else 30,
        )

    def _onAdaptiveRatioChanged(self, checked):
        if self._config:
            self._config.aim_adaptive_ratio_enabled = bool(checked)
        self.adaptiveRatioRefHCard.setEnabled(bool(checked))

    def _onAdaptiveRatioRefHChanged(self, value):
        if self._config:
            self._config.aim_adaptive_ratio_ref_h = float(value)

    def _onPostureAwareChanged(self, checked):
        if self._config:
            self._config.aim_posture_aware_enabled = bool(checked)
        self.crouchAspectCard.setEnabled(bool(checked))

    def _onCrouchAspectChanged(self, value):
        if self._config:
            self._config.aim_crouch_aspect_threshold = value / 100.0

    def _onTargetClassSelectionChanged(self, ids):
        if self._config:
            self._config.aim_target_class_ids = list(ids)

    def retranslateUi(self):
        """Refresh translations"""
        super().retranslateUi()

        self.generalGroup.titleLabel.setText(t("general_params"))
        self.aimPartCard.titleLabel.setText(t("aim_part"))
        self.mouseMoveCard.titleLabel.setText(t("mouse_move_method"))

        self.comPortCard.titleLabel.setText(t("arduino_com_port"))
        self.comRefreshBtn.setText(t("refresh"))
        self.arduinoBaudCard.titleLabel.setText(t("arduino_baud_rate", "Baud Rate"))
        self.arduinoBaudCard.contentLabel.setText(t("arduino_baud_rate_desc", "⚠ Must match the baud rate in your Arduino sketch"))
        self.connectionCard.titleLabel.setText(t("connected") + " / " + t("disconnected"))
        self.arduinoConnectCard.titleLabel.setText(t("arduino_connect"))
        self.arduinoConnectCard.contentLabel.setText(t("arduino_connect_desc"))
        self._updateArduinoConnectionStatus()
        self.guideCard.titleLabel.setText(t("arduino_guide"))
        self.guideBtn.setText(t("arduino_guide"))
        self.spoofCard.titleLabel.setText(t("spoof_device"))
        self.spoofBtn.setText(t("spoof_device"))
        self.verifySpoofCard.titleLabel.setText(t("verify_spoof"))
        self.verifySpoofBtn.setText(t("verify_spoof"))
        self.testHeartCard.titleLabel.setText(t("test_move_heart"))
        self.testHeartBtn.setText(t("test_move_heart"))

        self.xboxSensitivityCard.titleLabel.setText(t("xbox_sensitivity"))
        self.xboxDeadzoneCard.titleLabel.setText(t("xbox_deadzone"))
        self.xboxConnectionCard.titleLabel.setText(t("connected") + " / " + t("disconnected"))
        self.xboxConnectCard.titleLabel.setText(t("xbox_connect"))
        self.xboxConnectCard.contentLabel.setText(t("xbox_connect_desc"))
        self._updateXboxConnectionStatus()

        self.pidGroup.titleLabel.setText(t("aim_speed_pid"))
        self.pidUnsafeCard.titleLabel.setText(t("pid_unsafe_mode", "Unsafe Mode"))
        self.pidUnsafeCard.contentLabel.setText(t("pid_unsafe_mode_desc", "Let the P (reaction speed) sliders be dragged past the proven-stable 0.50 cap, up to 1.00. Higher values can cause oscillation/overshoot — tune carefully."))
        self.pidAxisPivot.setItemText('x', t("horizontal_x"))
        self.pidAxisPivot.setItemText('y', t("vertical_y"))
        self.pidPxCard.titleLabel.setText(t("reaction_speed_p"))
        self.pidIxCard.titleLabel.setText(t("error_correction_i"))
        self.pidDxCard.titleLabel.setText(t("stability_suppression_d"))
        self.pidPyCard.titleLabel.setText(t("reaction_speed_p"))
        self.pidIyCard.titleLabel.setText(t("error_correction_i"))
        self.pidDyCard.titleLabel.setText(t("stability_suppression_d"))
        self.maxMovePerFrameCard.titleLabel.setText("Max Move Per Frame")
        self.yReduceGroup.titleLabel.setText("Y-Axis Recoil Suppression")
        self.pidYReduceEnableCard.titleLabel.setText(t("aim_y_reduce_enable"))
        self.pidYReduceDelayCard.titleLabel.setText(t("aim_y_reduce_delay"))
        self.pidYReduceFloorCard.titleLabel.setText("Y Floor")
        self.pidYReduceRampCard.titleLabel.setText("Y Ramp Window")
        self.pidYReduceSettleCard.titleLabel.setText("Y Settle Threshold")
        self.pidYReduceVelCard.titleLabel.setText("Y Velocity Restore")

        self.humanizationGroup.titleLabel.setText(t("humanization", "Humanization"))
        self.humanizationEnableCard.titleLabel.setText(t("humanization_enabled", "Humanization"))
        self.humanizationEnableCard.contentLabel.setText(t("humanization_desc", "Perturb the final mouse output to look less robotic. Operates only on dx/dy — never touches detection or PID state."))
        self.humanizationIntensityCard.titleLabel.setText(t("humanization_intensity", "Intensity"))
        self.humanizationIntensityCard.contentLabel.setText(t("humanization_intensity_desc", "0% = robotic precision, 100% = fully human-like. Scales every effect below."))
        self.humanizationMicroJitterCard.titleLabel.setText(t("humanization_micro_jitter", "Micro-Jitter"))
        self.humanizationMicroJitterCard.contentLabel.setText(t("humanization_micro_jitter_desc", "Small zero-mean noise added to every move, scaled by movement size."))
        self.humanizationJitterBaseCard.titleLabel.setText(t("humanization_jitter_base", "Jitter Base"))
        self.humanizationJitterBaseCard.contentLabel.setText(t("humanization_jitter_base_desc", "Minimum jitter amplitude added every frame, in pixels."))
        self.humanizationJitterScaleCard.titleLabel.setText(t("humanization_jitter_scale", "Jitter Scale"))
        self.humanizationJitterScaleCard.contentLabel.setText(t("humanization_jitter_scale_desc", "Extra jitter added per pixel of movement, as % of movement size."))
        self.humanizationMicroJitterIdleCard.titleLabel.setText(t("humanization_micro_jitter_idle", "Apply While Aiming Idle"))
        self.humanizationMicroJitterIdleCard.contentLabel.setText(t("humanization_micro_jitter_idle_desc", "Also apply Micro-Jitter's tremor while the aim key is held but no target is locked, instead of holding perfectly still."))
        self.humanizationMotionVariationCard.titleLabel.setText(t("humanization_motion_variation", "Motion Variation"))
        self.humanizationMotionVariationCard.contentLabel.setText(t("humanization_motion_variation_desc", "Randomize output scale slightly each frame (mean-preserving, no drift)."))
        self.humanizationMotionVariationRangeCard.titleLabel.setText(t("humanization_motion_variation_range", "Variation Range"))
        self.humanizationMotionVariationRangeCard.contentLabel.setText(t("humanization_motion_variation_range_desc", "Random output-scale range applied each frame (mean-preserving)."))
        self.humanizationSpeedShapingCard.titleLabel.setText(t("humanization_speed_shaping", "Speed Shaping"))
        self.humanizationSpeedShapingCard.contentLabel.setText(t("humanization_speed_shaping_desc", "Compress small corrections and pass through large movements unmodified, like human fine-motor control."))
        self.humanizationSpeedShapingLowCard.titleLabel.setText(t("humanization_speed_shaping_low", "Fine-Control Threshold"))
        self.humanizationSpeedShapingLowCard.contentLabel.setText(t("humanization_speed_shaping_low_desc", "Movements below this size are compressed by the Low-Speed Factor."))
        self.humanizationSpeedShapingHighCard.titleLabel.setText(t("humanization_speed_shaping_high", "Full-Speed Threshold"))
        self.humanizationSpeedShapingHighCard.contentLabel.setText(t("humanization_speed_shaping_high_desc", "Movements above this size pass through unmodified."))
        self.humanizationSpeedShapingLowFactorCard.titleLabel.setText(t("humanization_speed_shaping_low_factor", "Low-Speed Factor"))
        self.humanizationSpeedShapingLowFactorCard.contentLabel.setText(t("humanization_speed_shaping_low_factor_desc", "Magnitude scale applied to movements below the Fine-Control Threshold."))
        self.humanizationMicroStutterCard.titleLabel.setText(t("humanization_micro_stutter", "Micro-Stutter"))
        self.humanizationMicroStutterCard.contentLabel.setText(t("humanization_micro_stutter_desc", "Occasional brief magnitude reduction, modeling muscle hesitation before committing to a move."))
        self.humanizationStutterProbCard.titleLabel.setText(t("humanization_stutter_prob", "Stutter Chance"))
        self.humanizationStutterProbCard.contentLabel.setText(t("humanization_stutter_prob_desc", "Probability per frame of a brief magnitude reduction."))
        self.humanizationStutterMinCard.titleLabel.setText(t("humanization_stutter_min", "Stutter Min"))
        self.humanizationStutterMinCard.contentLabel.setText(t("humanization_stutter_min_desc", "Lower bound of the stutter magnitude factor."))
        self.humanizationStutterMaxCard.titleLabel.setText(t("humanization_stutter_max", "Stutter Max"))
        self.humanizationStutterMaxCard.contentLabel.setText(t("humanization_stutter_max_desc", "Upper bound of the stutter magnitude factor."))
        self.humanizationReactionVariabilityCard.titleLabel.setText(t("humanization_reaction_variability", "Reaction Variability"))
        self.humanizationReactionVariabilityCard.contentLabel.setText(t("humanization_reaction_variability_desc", "Occasionally skip a frame's mouse injection to simulate human micro-hesitation. Adds real per-frame latency — off by default."))
        self.humanizationReactionSkipProbCard.titleLabel.setText(t("humanization_reaction_skip_prob", "Skip Chance"))
        self.humanizationReactionSkipProbCard.contentLabel.setText(t("humanization_reaction_skip_prob_desc", "Probability per frame of skipping the mouse injection entirely."))
        self.humanizationResetCard.titleLabel.setText(t("humanization_reset", "Reset to Defaults"))
        self.humanizationResetCard.contentLabel.setText(t("humanization_reset_desc", "Reset the whole Humanization section — Intensity, every feature toggle, and every fine-tuning slider above — back to default values."))
        self.humanizationResetBtn.setText(t("humanization_reset", "Reset to Defaults"))

        self.targetPriorityGroup.titleLabel.setText(t("target_priority", "Target Priority"))
        self.targetPriorityModeCard.titleLabel.setText(t("target_priority_mode", "Priority Mode"))
        self.targetPriorityModeCard.contentLabel.setText(t("target_priority_mode_desc", "How to select the best target"))
        self.targetPriorityWeightCard.titleLabel.setText(t("target_priority_confidence_weight", "Confidence Weight"))
        self.targetPriorityWeightCard.contentLabel.setText(t("target_priority_weight_desc", "Used in Composite mode only"))

        self.trackingGroup.titleLabel.setText(t("target_tracking", "Target Tracking"))
        self.predictionEnableCard.titleLabel.setText(t("prediction_enabled", "Velocity Prediction"))
        self.predictionEnableCard.contentLabel.setText(t("prediction_desc", "Extrapolate target position forward by the prediction horizon."))
        self.predictionHorizonCard.titleLabel.setText(t("prediction_horizon", "Prediction Horizon"))
        self.predictionMaxVelCard.titleLabel.setText(t("prediction_max_velocity", "Max Velocity Cap"))
        self.predictionMaxVelCard.contentLabel.setText(t("prediction_max_vel_desc", "Velocity spikes above this are treated as detection jumps and reset prediction"))
        self.predictionHistoryCard.titleLabel.setText(t("prediction_history", "History Frames"))
        self.stickyLockCard.titleLabel.setText(t("sticky_lock_enabled", "Sticky Target Lock"))
        self.stickyLockCard.contentLabel.setText(t("sticky_lock_desc", "Lock onto a target and hold aim across short detection gaps."))
        self.lockDecayCard.titleLabel.setText(t("lock_decay_frames", "Lock Decay Frames"))
        self.lockDecayCard.contentLabel.setText(t("lock_decay_desc", "Frames to hold aim after target is lost before releasing the lock"))
        self.lockIouCard.titleLabel.setText(t("lock_iou_threshold", "IoU Match Threshold"))
        self.lockIouCard.contentLabel.setText(t("lock_iou_desc", "Minimum overlap required to match the same target across frames"))
        self.kalmanEnableCard.titleLabel.setText(t("kalman_enabled_label", "Kalman Filter"))
        self.kalmanEnableCard.contentLabel.setText(t("kalman_enabled_desc", "2D Kalman filter for aim-point smoothing."))
        self.kalmanProcessNoiseCard.titleLabel.setText(t("kalman_process_noise_label", "Process Noise"))
        self.kalmanProcessNoiseCard.contentLabel.setText(t("kalman_noise_desc", "Lower = smoother but slower to react"))
        self.kalmanMeasNoiseCard.titleLabel.setText(t("kalman_meas_noise_label", "Measurement Noise"))
        self.kalmanMeasNoiseCard.contentLabel.setText(t("kalman_noise_desc", "Lower = reacts faster but noisier"))
        self.camMotionCompCard.titleLabel.setText(t("cam_motion_comp_enabled", "Camera Motion Compensation"))
        self.camMotionCompCard.contentLabel.setText(t("cam_motion_comp_desc", "Subtract per-frame global scene shift (phase correlation) from aim error to cancel camera shake."))
        self.camMotionCompSizeCard.titleLabel.setText(t("cam_motion_comp_size", "Compensation Resolution"))
        self.camMotionCompSizeCard.contentLabel.setText(t("cam_motion_comp_size_desc", "128 = ~0.2 ms (recommended), 256 = ~0.5 ms (more precise)"))

        parts = ["head", "body", "center", "custom"]
        idx = self.aimPartCombo.currentIndex()
        self._updateTargetAreaVisibility(parts[idx] if 0 <= idx < len(parts) else "head")
        self.customYCard.titleLabel.setText(t("aim_custom_y_pct", "Custom Aim Y Position (%)"))
        self.customYCard.contentLabel.setText(t("aim_custom_y_desc", "0% = top of box, 100% = bottom. ~20% = head, ~60% = body."))
        self.headWidthCard.titleLabel.setText(t("head_width_ratio"))
        self.headHeightCard.titleLabel.setText(t("head_height_ratio"))
        self.headHeightCard.contentLabel.setText(t("body_height_note"))
        self.bodyWidthCard.titleLabel.setText(t("body_width_ratio"))
        self.adaptiveRatioCard.titleLabel.setText(t("aim_adaptive_ratio_enabled", "Distance-Adaptive Ratio"))
        self.adaptiveRatioCard.contentLabel.setText(t("aim_adaptive_ratio_desc", "Scale head ratio inversely with box size — keeps head aim accurate from close to long range."))
        self.adaptiveRatioRefHCard.titleLabel.setText(t("aim_adaptive_ratio_ref_h", "Reference Box Height"))
        self.adaptiveRatioRefHCard.contentLabel.setText(t("aim_adaptive_ratio_ref_h_desc", "Box height (px) where head ratio is nominal. Match to your typical close-range target."))
        self.postureAwareCard.titleLabel.setText(t("aim_posture_aware_enabled", "Posture-Aware Targeting"))
        self.postureAwareCard.contentLabel.setText(t("aim_posture_aware_desc", "Fall back to center-mass when box is wider than tall (crouch / slide / prone)."))
        self.crouchAspectCard.titleLabel.setText(t("aim_crouch_aspect_threshold", "Crouch Aspect Threshold"))
        self.crouchAspectCard.contentLabel.setText(t("aim_crouch_aspect_desc", "box_w / box_h above which player is treated as crouching. Default 1.2×."))
        self.targetClassCard.titleLabel.setText(t("aim_target_class_ids", "Target Classes"))
        self.targetClassCard.contentLabel.setText(t("aim_target_class_ids_desc",
            "Choose which detected classes count as a valid aim target (e.g. never aim at a teammate class). All checked = no restriction."))

        current_aim = self.aimPartCombo.currentIndex()
        self.aimPartCombo.clear()
        self.aimPartCombo.addItems([t("head"), t("body"), t("center", "Smart (Center-mass)"), t("custom", "Custom")])
        self.aimPartCombo.setCurrentIndex(current_aim)
