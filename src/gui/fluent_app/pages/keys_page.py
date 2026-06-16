# keys_page.py
"""Hardware Output page — keybinds, MAKCU connection + keys"""

from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtWidgets import QWidget, QHBoxLayout
from PyQt6.QtGui import QKeySequence
from qfluentwidgets import (
    SettingCardGroup, SettingCard, SwitchSettingCard, FluentIcon,
    PushButton, BodyLabel, ComboBox,
)

from ..components.slider_spin_card import SliderDoubleSpinCard

from ..base_page import BasePage
from ..language_manager import t

# 手柄按鍵讀取
from win_utils.gamepad_input import (
    is_gamepad_vk, poll_pressed_gamepad_button, GP_VK_TRANSLATION_MAP,
    GP_VK_MIN, GP_VK_MAX,
)

MOUSE_VK_BIND_OPTIONS = (
    (0x01, "Mouse Left"),
    (0x02, "Mouse Right"),
    (0x04, "Mouse Middle"),
    (0x05, "Mouse X1"),
    (0x06, "Mouse X2"),
)

# MAKCU mouse-button combo options (label, VK code)
_MAKCU_BTN_OPTIONS = [
    ("Left",   0x01),
    ("Right",  0x02),
    ("Middle", 0x04),
    ("Side 1", 0x05),
    ("Side 2", 0x06),
    ("None",   0x00),
]

# MAKCU aim-trigger combo options (label, config string)
_MAKCU_TRIGGER_OPTIONS = [
    ("Left",  "lmb"),
    ("Right", "rmb"),
    ("Off",   "off"),
]


# 虛擬鍵碼對應翻譯 key 表
VK_CODE_TRANSLATION_MAP = {
    0x00: "key_none",
    0x01: "key_mouse_left",
    0x02: "key_mouse_right",
    0x04: "key_mouse_middle",
    0x05: "key_mouse_x1",
    0x06: "key_mouse_x2",
    0x08: "key_backspace",
    0x09: "key_tab",
    0x0D: "key_enter",
    0x10: "key_shift",
    0x11: "key_ctrl",
    0x12: "key_alt",
    0x14: "key_caps_lock",
    0x1B: "key_esc",
    0x20: "key_space",
    0x25: "key_left",
    0x26: "key_up",
    0x27: "key_right",
    0x28: "key_down",
    0x2D: "key_insert",
    0x2E: "key_delete",
}
# 合併手柄按鍵翻譯
VK_CODE_TRANSLATION_MAP.update(GP_VK_TRANSLATION_MAP)


def vk_to_name(vk_code: int) -> str:
    """將虛擬鍵碼轉換為可讀名稱（支援翻譯，包含手柄）"""
    # 特殊鍵使用翻譯（包含手柄按鍵）
    if vk_code in VK_CODE_TRANSLATION_MAP:
        return t(VK_CODE_TRANSLATION_MAP[vk_code])
    # 手柄按鍵回退顯示
    if is_gamepad_vk(vk_code):
        return f"\U0001f3ae 0x{vk_code:04X}"
    # 字母 A-Z
    if 0x41 <= vk_code <= 0x5A:
        return chr(vk_code)
    # 數字 0-9
    if 0x30 <= vk_code <= 0x39:
        return chr(vk_code)
    # F1-F12
    if 0x70 <= vk_code <= 0x7B:
        return f"F{vk_code - 0x70 + 1}"
    # 未知鍵碼
    return f"0x{vk_code:02X}"


class KeyBindButton(PushButton):
    """按鍵綁定按鈕（支援右鍵清除）"""
    keyBound = pyqtSignal(int)  # 發送虛擬鍵碼

    def __init__(self, parent=None):
        super().__init__(parent)
        self._vkCode = 0
        self._listening = False
        self.setMinimumWidth(120)
        self.clicked.connect(self._startListening)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._showContextMenu)

        # 手柄輪詢計時器
        self._gamepadTimer = QTimer(self)
        self._gamepadTimer.setInterval(50)  # 50ms 輪詢
        self._gamepadTimer.timeout.connect(self._pollGamepad)

    def setVkCode(self, vk_code: int):
        """設定虛擬鍵碼"""
        self._vkCode = vk_code
        self._updateText()

    def _updateText(self):
        """更新按鈕文字"""
        self.setText(vk_to_name(self._vkCode))

    def vkCode(self) -> int:
        return self._vkCode

    def _startListening(self):
        """開始監聽按鍵"""
        self._listening = True
        self.setText(t("key_press_to_bind"))
        self.setFocus()

        # 記錄目前按下的所有鍵，避免一進監聽就偵測到（例如某些滑鼠微動延遲釋放）
        import win32api
        self._initial_keys = set()
        for i in range(1, 255):
            if (win32api.GetAsyncKeyState(i) & 0x8000) != 0:
                self._initial_keys.add(i)

        # 啟動輸入輪詢（包含手柄、滑鼠、全局鍵盤）
        self._gamepadTimer.start()

    def _stopListening(self):
        """停止監聽，釋放滑鼠和鍵盤抓取"""
        self._listening = False
        self._gamepadTimer.stop()

    def _showContextMenu(self, pos):
        """顯示右鍵選單"""
        from PyQt6.QtWidgets import QMenu
        from PyQt6.QtGui import QAction
        menu = QMenu(self)

        # 快速綁定滑鼠按鍵（用於無法直接回報按鍵事件的硬體，例如 MAKCU）
        for vk_code, label in MOUSE_VK_BIND_OPTIONS:
            action = QAction(f"Bind {label}", self)
            action.triggered.connect(lambda checked=False, code=vk_code: self._bindDirectVk(code))
            menu.addAction(action)

        menu.addSeparator()
        clearAction = QAction(t("key_clear"), self)
        clearAction.triggered.connect(self._clearBinding)
        menu.addAction(clearAction)
        menu.exec(self.mapToGlobal(pos))

    def _bindDirectVk(self, vk_code: int):
        """直接綁定指定 VK（例如滑鼠按鍵）"""
        self._vkCode = int(vk_code)
        self._updateText()
        if self._listening:
            self._stopListening()
        self.keyBound.emit(self._vkCode)

    def _clearBinding(self):
        """清除按鍵綁定"""
        self._vkCode = 0
        self._updateText()
        if self._listening:
            self._stopListening()
        else:
            self._gamepadTimer.stop()
        self.keyBound.emit(0)

    def refreshText(self):
        """刷新按鈕文字（用於語言切換）"""
        if not self._listening:
            self._updateText()


    def _pollGamepad(self):
        """輪詢全局按鍵與手柄（由 QTimer 觸發）"""
        if not self._listening:
            self._gamepadTimer.stop()
            return

        import win32api

        # 1. 輪詢系統全局按鍵 (包含滑鼠與鍵盤)
        for i in range(1, 255):
            is_down = (win32api.GetAsyncKeyState(i) & 0x8000) != 0
            if is_down:
                if i not in self._initial_keys:
                    # 偵測到新按下的按鍵
                    self._vkCode = i
                    self.setText(vk_to_name(i))
                    self.keyBound.emit(i)
                    self._stopListening()
                    return
            else:
                # 按鍵釋放後，從黑名單移除
                self._initial_keys.discard(i)

        # 2. 輪詢手柄
        gp_vk = poll_pressed_gamepad_button()
        if gp_vk:
            self._vkCode = gp_vk
            self.setText(vk_to_name(gp_vk))
            self.keyBound.emit(gp_vk)
            self._stopListening()

    def _qtKeyToVk(self, qtKey: int) -> int:
        """將 Qt Key 轉換為 Windows VK code"""
        # 字母
        if Qt.Key.Key_A <= qtKey <= Qt.Key.Key_Z:
            return 0x41 + (qtKey - Qt.Key.Key_A)
        # 數字
        if Qt.Key.Key_0 <= qtKey <= Qt.Key.Key_9:
            return 0x30 + (qtKey - Qt.Key.Key_0)
        # F1-F12
        if Qt.Key.Key_F1 <= qtKey <= Qt.Key.Key_F12:
            return 0x70 + (qtKey - Qt.Key.Key_F1)
        # 特殊鍵
        mapping = {
            Qt.Key.Key_Escape: 0x1B,
            Qt.Key.Key_Tab: 0x09,
            Qt.Key.Key_Backspace: 0x08,
            Qt.Key.Key_Return: 0x0D,
            Qt.Key.Key_Enter: 0x0D,
            Qt.Key.Key_Insert: 0x2D,
            Qt.Key.Key_Delete: 0x2E,
            Qt.Key.Key_Space: 0x20,
            Qt.Key.Key_Left: 0x25,
            Qt.Key.Key_Up: 0x26,
            Qt.Key.Key_Right: 0x27,
            Qt.Key.Key_Down: 0x28,
            Qt.Key.Key_Shift: 0x10,
            Qt.Key.Key_Control: 0x11,
            Qt.Key.Key_Alt: 0x12,
            Qt.Key.Key_CapsLock: 0x14,
        }
        return mapping.get(qtKey, 0)


# Known USB identifiers for MAKCU devices
_MAKCU_VID = 0x1A86
_MAKCU_PID = 0x55D3
_MAKCU_DESC_KEYWORDS = ("USB-Enhanced-SERIAL CH343", "USB Single Serial")


class KeysPage(BasePage):
    """Hardware Output page — keybinds, MAKCU connection + keys"""

    def __init__(self, parent=None):
        super().__init__("tab_hardware_output", parent)
        self._config = None
        self._isMakcuConnected = False
        self._initWidgets()
        self._initLayout()
        self._connectSignals()

        # Aim-status poll timer (50 ms)
        self._aimStatusTimer = QTimer(self)
        self._aimStatusTimer.setInterval(50)
        self._aimStatusTimer.timeout.connect(self._updateMakcuAimStatus)
        self._aimStatusTimer.start()

    def setConfig(self, config):
        """設定 Config 實例並載入值"""
        self._config = config
        self._loadFromConfig()
        self._loadMakcuConnFromConfig()
        self._updateMakcuVisibility()

    def showEvent(self, event):
        """Re-check MAKCU visibility whenever this tab is navigated to."""
        super().showEvent(event)
        self._updateMakcuVisibility()

    # ──────────────────────────────────────────────
    # MAKCU visibility helper
    # ──────────────────────────────────────────────

    def _updateMakcuVisibility(self):
        is_makcu = getattr(self._config, 'mouse_move_method', '') == 'makcu' if self._config else False
        # Aim key cards 1–3 hidden in MAKCU mode; toggle key always visible
        for card in (self.aimKey1Card, self.aimKey2Card, self.aimKey3Card):
            card.setVisible(not is_makcu)
        # Fire keys group hidden in MAKCU mode
        self.fireKeysGroup.setVisible(not is_makcu)
        # MAKCU connection + keys groups visible only in MAKCU mode
        self.makcuConnGroup.setVisible(is_makcu)
        self.makcuKeysGroup.setVisible(is_makcu)

    # ──────────────────────────────────────────────
    # Widget init
    # ──────────────────────────────────────────────

    def _initWidgets(self):
        """初始化所有控制項"""

        # === 瞄準按鍵 ===
        self.aimKeysGroup = SettingCardGroup(t("auto_aim"), self.scrollWidget)

        self.alwaysAimCard = SwitchSettingCard(
            FluentIcon.FINGERPRINT,
            t("always_aim"),
            "",
            parent=self.aimKeysGroup
        )

        # 瞄準鍵 1
        self.aimKey1Btn = KeyBindButton()
        self.aimKey1Card = SettingCard(
            FluentIcon.FINGERPRINT,
            t("aim_key_1"),
            "",
            self.aimKeysGroup
        )
        self.aimKey1Card.hBoxLayout.addWidget(self.aimKey1Btn, 0, Qt.AlignmentFlag.AlignRight)
        self.aimKey1Card.hBoxLayout.addSpacing(16)

        # 瞄準鍵 2
        self.aimKey2Btn = KeyBindButton()
        self.aimKey2Card = SettingCard(
            FluentIcon.FINGERPRINT,
            t("aim_key_2"),
            "",
            self.aimKeysGroup
        )
        self.aimKey2Card.hBoxLayout.addWidget(self.aimKey2Btn, 0, Qt.AlignmentFlag.AlignRight)
        self.aimKey2Card.hBoxLayout.addSpacing(16)

        # 瞄準鍵 3
        self.aimKey3Btn = KeyBindButton()
        self.aimKey3Card = SettingCard(
            FluentIcon.FINGERPRINT,
            t("aim_key_3"),
            "",
            self.aimKeysGroup
        )
        self.aimKey3Card.hBoxLayout.addWidget(self.aimKey3Btn, 0, Qt.AlignmentFlag.AlignRight)
        self.aimKey3Card.hBoxLayout.addSpacing(16)

        # 切換鍵 (always visible)
        self.toggleKeyBtn = KeyBindButton()
        self.toggleKeyCard = SettingCard(
            FluentIcon.POWER_BUTTON,
            t("toggle_key"),
            t("toggle_auto_aim"),
            self.aimKeysGroup
        )
        self.toggleKeyCard.hBoxLayout.addWidget(self.toggleKeyBtn, 0, Qt.AlignmentFlag.AlignRight)
        self.toggleKeyCard.hBoxLayout.addSpacing(16)

        # === 自動射擊按鍵 ===
        self.fireKeysGroup = SettingCardGroup(t("keys_and_auto_fire"), self.scrollWidget)

        # 自動射擊鍵 1
        self.fireKey1Btn = KeyBindButton()
        self.fireKey1Card = SettingCard(
            FluentIcon.RINGER,
            t("auto_fire_key_1"),
            "",
            self.fireKeysGroup
        )
        self.fireKey1Card.hBoxLayout.addWidget(self.fireKey1Btn, 0, Qt.AlignmentFlag.AlignRight)
        self.fireKey1Card.hBoxLayout.addSpacing(16)

        # 自動射擊鍵 2
        self.fireKey2Btn = KeyBindButton()
        self.fireKey2Card = SettingCard(
            FluentIcon.RINGER,
            t("auto_fire_key_2"),
            "",
            self.fireKeysGroup
        )
        self.fireKey2Card.hBoxLayout.addWidget(self.fireKey2Btn, 0, Qt.AlignmentFlag.AlignRight)
        self.fireKey2Card.hBoxLayout.addSpacing(16)

        # === MAKCU Connection (shown only when mouse_move_method == "makcu") ===
        self.makcuConnGroup = SettingCardGroup(t("makcu_connection_group", "MAKCU Connection"), self.scrollWidget)

        # COM port selector + refresh button
        self.makcuComPortCombo = ComboBox()
        self.makcuComPortCombo.setMinimumWidth(120)
        self.makcuComPortCombo.addItem(t("no_com_port", "No COM Port"))
        self._refreshMakcuComPorts()

        self.makcuComRefreshBtn = PushButton(t("refresh", "Refresh"))
        self.makcuComRefreshBtn.setFixedWidth(80)

        self.makcuComPortCard = SettingCard(
            FluentIcon.CONNECT,
            t("makcu_com_port", "COM Port:"),
            "",
            self.makcuConnGroup
        )
        self.makcuComPortCard.hBoxLayout.addWidget(self.makcuComPortCombo, 0, Qt.AlignmentFlag.AlignRight)
        self.makcuComPortCard.hBoxLayout.addWidget(self.makcuComRefreshBtn, 0, Qt.AlignmentFlag.AlignRight)
        self.makcuComPortCard.hBoxLayout.addSpacing(16)

        # Baud rate
        self.makcuBaudCombo = ComboBox()
        self.makcuBaudCombo.addItems(["115200", "460800", "1000000", "2000000", "4000000"])
        self.makcuBaudCombo.setMinimumWidth(120)
        self.makcuBaudCard = SettingCard(
            FluentIcon.SPEED_HIGH,
            t("arduino_baud_rate", "Baud Rate"),
            "",
            self.makcuConnGroup
        )
        self.makcuBaudCard.hBoxLayout.addWidget(self.makcuBaudCombo, 0, Qt.AlignmentFlag.AlignRight)
        self.makcuBaudCard.hBoxLayout.addSpacing(16)

        # Connection status label
        self.makcuConnectionLabel = BodyLabel(t("disconnected", "Disconnected"))
        self.makcuConnectionLabel.setStyleSheet("color: #e74c3c; font-weight: bold;")
        self.makcuConnectionCard = SettingCard(
            FluentIcon.WIFI,
            t("connected", "Connected") + " / " + t("disconnected", "Disconnected"),
            "",
            self.makcuConnGroup
        )
        self.makcuConnectionCard.hBoxLayout.addWidget(self.makcuConnectionLabel, 0, Qt.AlignmentFlag.AlignRight)
        self.makcuConnectionCard.hBoxLayout.addSpacing(16)

        # Connect / Disconnect button
        self.makcuConnectBtn = PushButton(t("makcu_connect", "Connect MAKCU"))
        self.makcuConnectBtn.setFixedWidth(160)
        self.makcuConnectBtnCard = SettingCard(
            FluentIcon.LINK,
            t("makcu_connect", "Connect MAKCU"),
            t("makcu_connect_desc", "Select COM port then click to connect"),
            self.makcuConnGroup
        )
        self.makcuConnectBtnCard.hBoxLayout.addWidget(self.makcuConnectBtn, 0, Qt.AlignmentFlag.AlignRight)
        self.makcuConnectBtnCard.hBoxLayout.addSpacing(16)

        # Aim status label
        self.makcuAimStatusLabel = BodyLabel("—")
        self.makcuAimStatusLabel.setStyleSheet("font-weight: bold;")
        self.makcuAimStatusCard = SettingCard(
            FluentIcon.FINGERPRINT,
            t("aim_status", "Aim Status"),
            "",
            self.makcuConnGroup
        )
        self.makcuAimStatusCard.hBoxLayout.addWidget(self.makcuAimStatusLabel, 0, Qt.AlignmentFlag.AlignRight)
        self.makcuAimStatusCard.hBoxLayout.addSpacing(16)

        # === MAKCU Keys (shown only when mouse_move_method == "makcu") ===
        self.makcuKeysGroup = SettingCardGroup(t("makcu_keys_group", "MAKCU Keys"), self.scrollWidget)

        # Inference key (AimKeys[0])
        self.makcuInferenceCombo = ComboBox()
        self.makcuInferenceCombo.setMinimumWidth(110)
        for label, _ in _MAKCU_BTN_OPTIONS:
            self.makcuInferenceCombo.addItem(label)
        self.makcuInferenceCard = SettingCard(
            FluentIcon.FINGERPRINT,
            t("makcu_key_inference", "Inference"),
            t("makcu_key_inference_desc", "Hold this mouse button to activate inference"),
            self.makcuKeysGroup
        )
        self.makcuInferenceCard.hBoxLayout.addWidget(self.makcuInferenceCombo, 0, Qt.AlignmentFlag.AlignRight)
        self.makcuInferenceCard.hBoxLayout.addSpacing(16)

        # Auto Aim Key (makcu_aim_button) — which mouse button activates aim
        self.makcuTriggerCombo = ComboBox()
        self.makcuTriggerCombo.setMinimumWidth(110)
        for label, _ in _MAKCU_TRIGGER_OPTIONS:
            self.makcuTriggerCombo.addItem(label)
        self.makcuTriggerCard = SettingCard(
            FluentIcon.FINGERPRINT,
            t("makcu_auto_aim_key", "Auto Aim Key"),
            t("makcu_auto_aim_key_desc", "Mouse button that activates Auto Aim"),
            self.makcuKeysGroup
        )
        self.makcuTriggerCard.hBoxLayout.addWidget(self.makcuTriggerCombo, 0, Qt.AlignmentFlag.AlignRight)
        self.makcuTriggerCard.hBoxLayout.addSpacing(16)

        # Auto Aim Mode (makcu_aim_mode) — hold or toggle
        _AIM_MODE_OPTIONS = [
            (t("aim_mode_hold", "Hold"), "hold"),
            (t("aim_mode_toggle", "Toggle"), "toggle"),
        ]
        self._AIM_MODE_OPTIONS = _AIM_MODE_OPTIONS
        self.makcuAimModeCombo = ComboBox()
        self.makcuAimModeCombo.setMinimumWidth(110)
        for label, _ in _AIM_MODE_OPTIONS:
            self.makcuAimModeCombo.addItem(label)
        self.makcuAimModeCard = SettingCard(
            FluentIcon.ROTATE,
            t("makcu_aim_mode", "Aim Mode"),
            t("makcu_aim_mode_desc", "Hold: aim while button held  |  Toggle: click to toggle aim on/off"),
            self.makcuKeysGroup
        )
        self.makcuAimModeCard.hBoxLayout.addWidget(self.makcuAimModeCombo, 0, Qt.AlignmentFlag.AlignRight)
        self.makcuAimModeCard.hBoxLayout.addSpacing(16)

        # Disengage delay — how long aim stays active after releasing the aim button
        self.makcuDisengageDelayCard = SliderDoubleSpinCard(
            FluentIcon.HISTORY,
            t("makcu_disengage_delay", "Disengage Delay"),
            0.0, 20.0,
            decimals=1,
            step=0.1,
            suffix="s",
            description=t("makcu_disengage_delay_desc", "Keep aiming after releasing the aim button (0 = off)"),
            parent=self.makcuKeysGroup
        )

    # ──────────────────────────────────────────────
    # Layout
    # ──────────────────────────────────────────────

    def _initLayout(self):
        """排版所有控制項"""
        # 瞄準按鍵
        self.aimKeysGroup.addSettingCard(self.alwaysAimCard)
        self.aimKeysGroup.addSettingCard(self.aimKey1Card)
        self.aimKeysGroup.addSettingCard(self.aimKey2Card)
        self.aimKeysGroup.addSettingCard(self.aimKey3Card)
        self.aimKeysGroup.addSettingCard(self.toggleKeyCard)
        self.addContent(self.aimKeysGroup)

        # 自動射擊按鍵
        self.fireKeysGroup.addSettingCard(self.fireKey1Card)
        self.fireKeysGroup.addSettingCard(self.fireKey2Card)
        self.addContent(self.fireKeysGroup)

        # MAKCU Connection group (hidden by default until setConfig runs)
        self.makcuConnGroup.addSettingCard(self.makcuComPortCard)
        self.makcuConnGroup.addSettingCard(self.makcuBaudCard)
        self.makcuConnGroup.addSettingCard(self.makcuConnectionCard)
        self.makcuConnGroup.addSettingCard(self.makcuConnectBtnCard)
        self.makcuConnGroup.addSettingCard(self.makcuAimStatusCard)
        self.addContent(self.makcuConnGroup)
        self.makcuConnGroup.setVisible(False)

        # MAKCU Keys (hidden by default until setConfig runs)
        self.makcuKeysGroup.addSettingCard(self.makcuInferenceCard)
        self.makcuKeysGroup.addSettingCard(self.makcuTriggerCard)
        self.makcuKeysGroup.addSettingCard(self.makcuAimModeCard)
        self.makcuKeysGroup.addSettingCard(self.makcuDisengageDelayCard)
        self.addContent(self.makcuKeysGroup)
        self.makcuKeysGroup.setVisible(False)

        self.scrollLayout.addStretch(1)

    # ──────────────────────────────────────────────
    # Signal connections
    # ──────────────────────────────────────────────

    def _connectSignals(self):
        """連接信號"""
        self.alwaysAimCard.checkedChanged.connect(self._onAlwaysAimChanged)
        self.aimKey1Btn.keyBound.connect(lambda vk: self._onAimKeyChanged(0, vk))
        self.aimKey2Btn.keyBound.connect(lambda vk: self._onAimKeyChanged(1, vk))
        self.aimKey3Btn.keyBound.connect(lambda vk: self._onAimKeyChanged(2, vk))
        self.toggleKeyBtn.keyBound.connect(self._onToggleKeyChanged)
        self.fireKey1Btn.keyBound.connect(self._onFireKey1Changed)
        self.fireKey2Btn.keyBound.connect(self._onFireKey2Changed)

        self.makcuComPortCombo.currentTextChanged.connect(self._onMakcuComPortChanged)
        self.makcuComRefreshBtn.clicked.connect(self._refreshMakcuComPorts)
        self.makcuBaudCombo.currentTextChanged.connect(self._onMakcuBaudChanged)
        self.makcuConnectBtn.clicked.connect(self._onMakcuConnectToggle)

        self.makcuInferenceCombo.currentIndexChanged.connect(self._onMakcuInferenceKeyChanged)
        self.makcuTriggerCombo.currentIndexChanged.connect(self._onMakcuTriggerKeyChanged)
        self.makcuAimModeCombo.currentIndexChanged.connect(self._onMakcuAimModeChanged)
        self.makcuDisengageDelayCard.valueChanged.connect(self._onMakcuDisengageDelayChanged)

    # ──────────────────────────────────────────────
    # Config load
    # ──────────────────────────────────────────────

    def _loadFromConfig(self):
        """從 Config 載入值"""
        if not self._config:
            return

        self.alwaysAimCard.setChecked(bool(getattr(self._config, 'always_aim', False)))

        # 瞄準鍵
        if len(self._config.AimKeys) >= 1:
            self.aimKey1Btn.setVkCode(self._config.AimKeys[0])
        if len(self._config.AimKeys) >= 2:
            self.aimKey2Btn.setVkCode(self._config.AimKeys[1])
        if len(self._config.AimKeys) >= 3:
            self.aimKey3Btn.setVkCode(self._config.AimKeys[2])

        # 切換鍵
        self.toggleKeyBtn.setVkCode(self._config.aim_toggle_key)

        # 自動射擊鍵
        self.fireKey1Btn.setVkCode(self._config.auto_fire_key)
        self.fireKey2Btn.setVkCode(self._config.auto_fire_key2)

        # MAKCU Keys
        self._loadMakcuCombos()

    def _loadMakcuCombos(self):
        """Load MAKCU-specific combo boxes from config."""
        if not self._config:
            return

        # Inference key (AimKeys[0])
        aim0 = self._config.AimKeys[0] if len(self._config.AimKeys) >= 1 else 0x01
        for i, (_, vk) in enumerate(_MAKCU_BTN_OPTIONS):
            if vk == aim0:
                self.makcuInferenceCombo.blockSignals(True)
                self.makcuInferenceCombo.setCurrentIndex(i)
                self.makcuInferenceCombo.blockSignals(False)
                break

        # Auto Aim Key (makcu_aim_button)
        trigger = getattr(self._config, 'makcu_aim_button', 'lmb').lower()
        for i, (_, val) in enumerate(_MAKCU_TRIGGER_OPTIONS):
            if val == trigger:
                self.makcuTriggerCombo.blockSignals(True)
                self.makcuTriggerCombo.setCurrentIndex(i)
                self.makcuTriggerCombo.blockSignals(False)
                break

        # Aim Mode (makcu_aim_mode)
        mode = getattr(self._config, 'makcu_aim_mode', 'hold').lower()
        for i, (_, val) in enumerate(self._AIM_MODE_OPTIONS):
            if val == mode:
                self.makcuAimModeCombo.blockSignals(True)
                self.makcuAimModeCombo.setCurrentIndex(i)
                self.makcuAimModeCombo.blockSignals(False)
                break

        # Disengage delay
        delay = float(getattr(self._config, 'makcu_disengage_delay', 0.0) or 0.0)
        self.makcuDisengageDelayCard.setValue(delay)

        self._refreshMakcuVisibility()

    def _refreshMakcuVisibility(self):
        """Show/hide MAKCU cards based on always_aim and keep_detecting."""
        if not self._config:
            return
        always_aim = bool(getattr(self._config, 'always_aim', False))
        keep_detecting = bool(getattr(self._config, 'keep_detecting', True))
        # Inference card: hidden when keep_detecting is on (no need to hold a button)
        self.makcuInferenceCard.setVisible(not keep_detecting)
        # Auto Aim Key + Mode + delay: hidden when always_aim is on (aim is always active)
        self.makcuTriggerCard.setVisible(not always_aim)
        self.makcuAimModeCard.setVisible(not always_aim)
        self.makcuDisengageDelayCard.setVisible(not always_aim)

    # ──────────────────────────────────────────────
    # Callbacks
    # ──────────────────────────────────────────────

    def _onAlwaysAimChanged(self, checked: bool):
        if self._config:
            self._config.always_aim = bool(checked)
            if checked:
                self._config.idle_detect_enabled = False
        self._refreshMakcuVisibility()

    def _onAimKeyChanged(self, index: int, vk: int):
        if self._config:
            while len(self._config.AimKeys) <= index:
                self._config.AimKeys.append(0)
            self._config.AimKeys[index] = vk

    def _onToggleKeyChanged(self, vk: int):
        if self._config:
            self._config.aim_toggle_key = vk

    def _onFireKey1Changed(self, vk: int):
        if self._config:
            self._config.auto_fire_key = vk

    def _onFireKey2Changed(self, vk: int):
        if self._config:
            self._config.auto_fire_key2 = vk

    def _onMakcuInferenceKeyChanged(self, idx: int):
        if self._config and 0 <= idx < len(_MAKCU_BTN_OPTIONS):
            vk = _MAKCU_BTN_OPTIONS[idx][1]
            while len(self._config.AimKeys) < 1:
                self._config.AimKeys.append(0)
            self._config.AimKeys[0] = vk

    def _onMakcuTriggerKeyChanged(self, idx: int):
        if self._config and 0 <= idx < len(_MAKCU_TRIGGER_OPTIONS):
            self._config.makcu_aim_button = _MAKCU_TRIGGER_OPTIONS[idx][1]

    def _onMakcuAimModeChanged(self, idx: int):
        if self._config and 0 <= idx < len(self._AIM_MODE_OPTIONS):
            self._config.makcu_aim_mode = self._AIM_MODE_OPTIONS[idx][1]

    def _onMakcuDisengageDelayChanged(self, value: float):
        if self._config:
            self._config.makcu_disengage_delay = float(value)

    def _loadMakcuConnFromConfig(self):
        """Load MAKCU COM port and baud rate from config, then auto-connect if device found."""
        if not self._config:
            return

        # Auto-detect MAKCU port; fall back to saved port
        auto_port = self._findMakcuPort()
        saved_port = getattr(self._config, 'makcu_com_port', '')
        effective_port = auto_port or saved_port

        if effective_port:
            if auto_port:
                self._config.makcu_com_port = auto_port
            idx = self.makcuComPortCombo.findText(effective_port)
            if idx >= 0:
                self.makcuComPortCombo.blockSignals(True)
                self.makcuComPortCombo.setCurrentIndex(idx)
                self.makcuComPortCombo.blockSignals(False)

        saved_baud = str(getattr(self._config, 'makcu_baud_rate', 4000000))
        baud_idx = self.makcuBaudCombo.findText(saved_baud)
        if baud_idx >= 0:
            self.makcuBaudCombo.blockSignals(True)
            self.makcuBaudCombo.setCurrentIndex(baud_idx)
            self.makcuBaudCombo.blockSignals(False)
        else:
            four_m_idx = self.makcuBaudCombo.findText("4000000")
            if four_m_idx >= 0:
                self.makcuBaudCombo.blockSignals(True)
                self.makcuBaudCombo.setCurrentIndex(four_m_idx)
                self.makcuBaudCombo.blockSignals(False)

        self._updateMakcuConnectionStatus()

        # Auto-connect if a MAKCU device is detected and not yet connected
        if effective_port:
            try:
                from win_utils.makcu_mouse import is_makcu_connected, connect_makcu
                if not is_makcu_connected():
                    ok = connect_makcu(effective_port, 4_000_000)
                    self._isMakcuConnected = ok
                    self._updateMakcuConnectionStatus()
            except Exception:
                pass

    # ──────────────────────────────────────────────
    # MAKCU connection helpers
    # ──────────────────────────────────────────────

    def _findMakcuPort(self) -> str | None:
        """Return the first serial port that matches the MAKCU USB VID/PID or description."""
        try:
            import serial.tools.list_ports
            for p in serial.tools.list_ports.comports():
                if p.vid == _MAKCU_VID and p.pid == _MAKCU_PID:
                    return p.device
                desc = f"{p.description or ''} {p.product or ''}"
                if any(kw in desc for kw in _MAKCU_DESC_KEYWORDS):
                    return p.device
        except Exception:
            pass
        return None

    def _refreshMakcuComPorts(self):
        """Refresh MAKCU COM port list and auto-select a detected MAKCU device."""
        try:
            import serial.tools.list_ports
            ports = list(serial.tools.list_ports.comports())
        except Exception:
            ports = []

        current = self.makcuComPortCombo.currentText()
        self.makcuComPortCombo.blockSignals(True)
        self.makcuComPortCombo.clear()
        self.makcuComPortCombo.addItem(t("no_com_port", "No COM Port"))
        for p in ports:
            self.makcuComPortCombo.addItem(p.device)

        # Prefer previously selected port, otherwise auto-detect MAKCU
        auto_port = self._findMakcuPort()
        preferred = current if current and current != t("no_com_port", "No COM Port") else (auto_port or '')
        idx = self.makcuComPortCombo.findText(preferred)
        if idx >= 0:
            self.makcuComPortCombo.setCurrentIndex(idx)
        self.makcuComPortCombo.blockSignals(False)

        if auto_port and self._config:
            self._config.makcu_com_port = auto_port

    def _onMakcuComPortChanged(self, text: str):
        if self._config:
            self._config.makcu_com_port = text

    def _onMakcuBaudChanged(self, text: str):
        if self._config:
            try:
                self._config.makcu_baud_rate = int(text)
            except ValueError:
                pass

    def _onMakcuConnectToggle(self):
        """Connect or disconnect MAKCU."""
        try:
            from win_utils.makcu_mouse import connect_makcu, disconnect_makcu, is_makcu_connected
        except ImportError:
            return

        if self._isMakcuConnected or is_makcu_connected():
            disconnect_makcu()
            self._isMakcuConnected = False
        else:
            port = self.makcuComPortCombo.currentText()
            if not port or port == t("no_com_port", "No COM Port"):
                return
            try:
                baud = int(self.makcuBaudCombo.currentText())
            except ValueError:
                baud = 4_000_000
            ok = connect_makcu(port, baud)
            self._isMakcuConnected = ok

        self._updateMakcuConnectionStatus()

    def _updateMakcuConnectionStatus(self):
        """Refresh connection label and connect button text."""
        try:
            from win_utils.makcu_mouse import is_makcu_connected
            connected = is_makcu_connected()
        except Exception:
            connected = False

        self._isMakcuConnected = connected
        if connected:
            self.makcuConnectionLabel.setText(t("connected", "Connected"))
            self.makcuConnectionLabel.setStyleSheet("color: #2ecc71; font-weight: bold;")
            self.makcuConnectBtn.setText(t("makcu_disconnect", "Disconnect MAKCU"))
        else:
            self.makcuConnectionLabel.setText(t("disconnected", "Disconnected"))
            self.makcuConnectionLabel.setStyleSheet("color: #e74c3c; font-weight: bold;")
            self.makcuConnectBtn.setText(t("makcu_connect", "Connect MAKCU"))

    def _updateMakcuAimStatus(self):
        """Poll aim status (50 ms timer)."""
        try:
            from win_utils.makcu_mouse import is_makcu_connected
            if not is_makcu_connected():
                self.makcuAimStatusLabel.setText("—")
                self.makcuAimStatusLabel.setStyleSheet("font-weight: bold;")
                return
            aiming = bool(getattr(self._config, 'makcu_aim_active', False))
            if aiming:
                self.makcuAimStatusLabel.setText(t("aiming", "Aiming"))
                self.makcuAimStatusLabel.setStyleSheet("color: #2ecc71; font-weight: bold;")
            else:
                self.makcuAimStatusLabel.setText(t("idle", "Idle"))
                self.makcuAimStatusLabel.setStyleSheet("color: #aaaaaa; font-weight: bold;")
        except Exception:
            self.makcuAimStatusLabel.setText("—")
            self.makcuAimStatusLabel.setStyleSheet("font-weight: bold;")

    # ──────────────────────────────────────────────
    # Retranslate
    # ──────────────────────────────────────────────

    def retranslateUi(self):
        """刷新翻譯"""
        super().retranslateUi()

        # 群組標題
        self.aimKeysGroup.titleLabel.setText(t("auto_aim"))
        self.fireKeysGroup.titleLabel.setText(t("keys_and_auto_fire"))
        self.makcuConnGroup.titleLabel.setText(t("makcu_connection_group", "MAKCU Connection"))
        self.makcuKeysGroup.titleLabel.setText(t("makcu_keys_group", "MAKCU Keys"))

        # MAKCU Connection cards
        self.makcuComPortCard.titleLabel.setText(t("makcu_com_port", "COM Port:"))
        self.makcuBaudCard.titleLabel.setText(t("arduino_baud_rate", "Baud Rate"))
        self.makcuConnectionCard.titleLabel.setText(t("connected", "Connected") + " / " + t("disconnected", "Disconnected"))
        self.makcuConnectBtnCard.titleLabel.setText(t("makcu_connect", "Connect MAKCU"))
        self.makcuConnectBtnCard.contentLabel.setText(t("makcu_connect_desc", "Select COM port then click to connect"))
        self.makcuAimStatusCard.titleLabel.setText(t("aim_status", "Aim Status"))
        self._updateMakcuConnectionStatus()

        # 瞄準按鍵
        self.alwaysAimCard.titleLabel.setText(t("always_aim"))
        self.aimKey1Card.titleLabel.setText(t("aim_key_1"))
        self.aimKey2Card.titleLabel.setText(t("aim_key_2"))
        self.aimKey3Card.titleLabel.setText(t("aim_key_3"))
        self.toggleKeyCard.titleLabel.setText(t("toggle_key"))
        self.toggleKeyCard.contentLabel.setText(t("toggle_auto_aim"))

        # 自動射擊按鍵
        self.fireKey1Card.titleLabel.setText(t("auto_fire_key_1"))
        self.fireKey2Card.titleLabel.setText(t("auto_fire_key_2"))

        # MAKCU Keys
        self.makcuInferenceCard.titleLabel.setText(t("makcu_key_inference", "Inference"))
        self.makcuInferenceCard.contentLabel.setText(t("makcu_key_inference_desc", "Hold this mouse button to activate inference"))
        self.makcuTriggerCard.titleLabel.setText(t("makcu_auto_aim_key", "Auto Aim Key"))
        self.makcuTriggerCard.contentLabel.setText(t("makcu_auto_aim_key_desc", "Mouse button that activates Auto Aim"))
        self.makcuAimModeCard.titleLabel.setText(t("makcu_aim_mode", "Aim Mode"))
        self.makcuAimModeCard.contentLabel.setText(t("makcu_aim_mode_desc", "Hold: aim while button held  |  Toggle: click to toggle aim on/off"))

        # 刷新按鍵綁定按鈕文字
        self.aimKey1Btn.refreshText()
        self.aimKey2Btn.refreshText()
        self.aimKey3Btn.refreshText()
        self.toggleKeyBtn.refreshText()
        self.fireKey1Btn.refreshText()
        self.fireKey2Btn.refreshText()
