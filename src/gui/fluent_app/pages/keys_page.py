# keys_page.py
"""按鍵綁定頁面 - 瞄準鍵、自動射擊鍵設定"""

from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtWidgets import QWidget, QHBoxLayout
from PyQt6.QtGui import QKeySequence
from qfluentwidgets import (
    SettingCardGroup, SettingCard, FluentIcon, 
    PushButton, BodyLabel
)

from ..base_page import BasePage
from ..language_manager import t

# 手柄按鍵讀取
from win_utils.gamepad_input import (
    is_gamepad_vk, poll_pressed_gamepad_button, GP_VK_TRANSLATION_MAP,
    GP_VK_MIN, GP_VK_MAX,
)


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
        clearAction = QAction(t("key_clear"), self)
        clearAction.triggered.connect(self._clearBinding)
        menu.addAction(clearAction)
        menu.exec(self.mapToGlobal(pos))

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


class KeysPage(BasePage):
    """按鍵綁定頁面"""
    
    def __init__(self, parent=None):
        super().__init__("tab_keys", parent)
        self._config = None
        self._initWidgets()
        self._initLayout()
        self._connectSignals()
    
    def setConfig(self, config):
        """設定 Config 實例並載入值"""
        self._config = config
        self._loadFromConfig()
    
    def _initWidgets(self):
        """初始化所有控制項"""
        
        # === 瞄準按鍵 ===
        self.aimKeysGroup = SettingCardGroup(t("auto_aim"), self.scrollWidget)
        
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
        
        # 切換鍵
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
    
    def _initLayout(self):
        """排版所有控制項"""
        # 瞄準按鍵
        self.aimKeysGroup.addSettingCard(self.aimKey1Card)
        self.aimKeysGroup.addSettingCard(self.aimKey2Card)
        self.aimKeysGroup.addSettingCard(self.aimKey3Card)
        self.aimKeysGroup.addSettingCard(self.toggleKeyCard)
        self.addContent(self.aimKeysGroup)
        
        # 自動射擊按鍵
        self.fireKeysGroup.addSettingCard(self.fireKey1Card)
        self.fireKeysGroup.addSettingCard(self.fireKey2Card)
        self.addContent(self.fireKeysGroup)
        
        self.scrollLayout.addStretch(1)
    
    def _connectSignals(self):
        """連接信號"""
        self.aimKey1Btn.keyBound.connect(lambda vk: self._onAimKeyChanged(0, vk))
        self.aimKey2Btn.keyBound.connect(lambda vk: self._onAimKeyChanged(1, vk))
        self.aimKey3Btn.keyBound.connect(lambda vk: self._onAimKeyChanged(2, vk))
        self.toggleKeyBtn.keyBound.connect(self._onToggleKeyChanged)
        self.fireKey1Btn.keyBound.connect(self._onFireKey1Changed)
        self.fireKey2Btn.keyBound.connect(self._onFireKey2Changed)
    
    def _loadFromConfig(self):
        """從 Config 載入值"""
        if not self._config:
            return
        
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
    
    # === 回調函數 ===
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
    
    def retranslateUi(self):
        """刷新翻譯"""
        super().retranslateUi()

        # 群組標題
        self.aimKeysGroup.titleLabel.setText(t("auto_aim"))
        self.fireKeysGroup.titleLabel.setText(t("keys_and_auto_fire"))

        # 瞄準按鍵
        self.aimKey1Card.titleLabel.setText(t("aim_key_1"))
        self.aimKey2Card.titleLabel.setText(t("aim_key_2"))
        self.aimKey3Card.titleLabel.setText(t("aim_key_3"))
        self.toggleKeyCard.titleLabel.setText(t("toggle_key"))
        self.toggleKeyCard.contentLabel.setText(t("toggle_auto_aim"))

        # 自動射擊按鍵
        self.fireKey1Card.titleLabel.setText(t("auto_fire_key_1"))
        self.fireKey2Card.titleLabel.setText(t("auto_fire_key_2"))

        # 刷新按鍵綁定按鈕文字
        self.aimKey1Btn.refreshText()
        self.aimKey2Btn.refreshText()
        self.aimKey3Btn.refreshText()
        self.toggleKeyBtn.refreshText()
        self.fireKey1Btn.refreshText()
        self.fireKey2Btn.refreshText()
