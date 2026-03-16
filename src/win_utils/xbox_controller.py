# xbox_controller.py - Xbox 360 Gamepad Emulation Module
"""
Emulate Xbox 360 controller right stick using vgamepad library
Convert mouse movement (dx, dy) to right stick input for in-game view control

Principle:
- Create virtual Xbox 360 controller (via ViGEmBus driver)
- Map AI-calculated mouse movement to Right Stick offset
- Support sensitivity adjustment, deadzone setting, response curves, etc.
"""

from __future__ import annotations

import os
import sys
import time
import threading
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ViGEmBus installer path (relative to this file)
_VIGEM_INSTALLER = os.path.join(os.path.dirname(__file__), "ViGEmBus_1.22.0_x64_x86_arm64.exe")


def _is_vigem_error(exc: Exception) -> bool:
    """Determine if exception is caused by ViGEmBus driver not installed"""
    msg = str(exc).lower()
    return any(kw in msg for kw in [
        "vigem", "vigembus", "bus not found", "driver", "not found",
        "cannot connect", "failed to connect", "0xE0000001".lower(),
    ])


def _launch_vigem_installer_and_exit() -> None:
    """Launch ViGEmBus installer and exit current program"""
    if os.path.exists(_VIGEM_INSTALLER):
        import subprocess
        print("[Xbox] ViGEmBus driver not detected, launching installer...")
        try:
            subprocess.Popen([_VIGEM_INSTALLER], shell=False)
        except Exception as launch_err:
            print(f"[Xbox] Could not launch installer: {launch_err}")
    else:
        print(f"[Xbox] Could not find ViGEmBus installer: {_VIGEM_INSTALLER}")
        print("[Xbox] Please manually download and install from https://github.com/nefarius/ViGEmBus/releases")

    print("[Xbox] Please restart Axiom after installation. Program will now close...")
    time.sleep(2)
    os._exit(0)

# vgamepad uses lazy import, only loaded when connect() is called
# to avoid triggering ViGEmBus driver check during module import
vg = None


def _import_vgamepad():
    """Attempt to import vgamepad; if ViGEmBus is not installed, launch installer and exit program"""
    global vg
    if vg is not None:
        return True
    try:
        import vgamepad as _vg
        vg = _vg
        return True
    except ImportError:
        print("[Xbox] vgamepad not installed, please run: pip install vgamepad")
        return False
    except Exception as e:
        # ViGEmBus driver not installed (e.g., VIGEM_ERROR_BUS_NOT_FOUND)
        print(f"[Xbox] vgamepad load failed: {e}")
        _launch_vigem_installer_and_exit()
        return False


class XboxController:
    """Xbox 360 Virtual Gamepad Controller
    
    Use vgamepad to create a virtual Xbox 360 controller,
    mapping mouse movement to right stick input.
    
    Attributes:
        sensitivity: Sensitivity multiplier (default 1.0)
        deadzone: Deadzone threshold, input below this value is ignored (default 0.05)
        stick_duration: Stick input duration (seconds) (default 0.03)
        max_stick_value: Maximum stick mapping value (0.0~1.0) (default 1.0)
    """
    
    def __init__(self) -> None:
        self._gamepad = None  # vg.VX360Gamepad instance or None
        self._lock = threading.Lock()
        self._connected = False
        self._init_attempted = False
        
        # Adjustable parameters
        self.sensitivity: float = 1.0
        self.deadzone: float = 0.05
        self.stick_duration: float = 0.03
        self.max_stick_value: float = 1.0
        
        # 統計
        self._move_count: int = 0
        self._error_count: int = 0
        self._last_error: str = ""
    
    def is_available(self) -> bool:
        """檢查 vgamepad 套件是否存在（不實際連線）"""
        try:
            import importlib.util
            return importlib.util.find_spec("vgamepad") is not None
        except Exception:
            return False
    
    def is_connected(self) -> bool:
        """檢查虛擬手把是否已連線"""
        return self._connected and self._gamepad is not None
    
    def connect(self) -> bool:
        """建立虛擬 Xbox 360 控制器
        
        Returns:
            是否成功建立
        """
        if not _import_vgamepad():
            return False

        with self._lock:
            if self._connected and self._gamepad is not None:
                return True
            
            try:
                self._gamepad = vg.VX360Gamepad()
                self._connected = True
                self._init_attempted = True
                self._error_count = 0
                logger.info("[Xbox] 虛擬 Xbox 360 控制器已建立")
                print("[Xbox] 虛擬 Xbox 360 控制器已建立")
                return True
            except Exception as e:
                self._last_error = f"建立虛擬手把失敗: {e}"
                self._connected = False
                self._gamepad = None
                self._init_attempted = True
                logger.error(f"[Xbox] {self._last_error}")
                print(f"[Xbox] {self._last_error}")

                # 若為 ViGEmBus 驅動未安裝，自動啟動安裝程式並結束程序
                if _is_vigem_error(e):
                    _launch_vigem_installer_and_exit()

                print("[Xbox] 請確認已安裝 ViGEmBus 驅動: https://github.com/nefarius/ViGEmBus/releases")
                return False
    
    def disconnect(self) -> None:
        """斷開虛擬手把"""
        with self._lock:
            if self._gamepad is not None:
                try:
                    # 重置所有輸入
                    self._gamepad.reset()
                    self._gamepad.update()
                except Exception:
                    pass
                self._gamepad = None
            self._connected = False
            logger.info("[Xbox] 虛擬手把已斷開")
            print("[Xbox] 虛擬手把已斷開")
    
    def ensure_initialized(self) -> bool:
        """確保手把已初始化，如果未初始化則嘗試連線"""
        if self._connected and self._gamepad is not None:
            return True
        return self.connect()
    
    def move_right_stick(self, dx: float, dy: float) -> bool:
        """移動右搖桿
        
        將滑鼠移動 (dx, dy) 映射到右搖桿偏移量。
        值域為 -1.0 到 1.0，其中：
        - X 軸: 負=左, 正=右
        - Y 軸: 負=上, 正=下 (注意: vgamepad Y 軸負=上)
        
        Args:
            dx: 水平移動量 (像素)
            dy: 垂直移動量 (像素)
            
        Returns:
            是否成功
        """
        if not self.ensure_initialized():
            return False
        
        with self._lock:
            try:
                # 應用靈敏度
                scaled_x = dx * self.sensitivity
                scaled_y = dy * self.sensitivity
                
                # 映射到 -1.0 ~ 1.0 範圍
                # 使用非線性映射：較大的移動量產生較大的搖桿偏移
                # 基準值：50 像素 = 搖桿全推
                BASE_PIXELS = 50.0
                norm_x = max(-1.0, min(1.0, scaled_x / BASE_PIXELS))
                norm_y = max(-1.0, min(1.0, scaled_y / BASE_PIXELS))
                
                # 應用最大值限制
                norm_x *= self.max_stick_value
                norm_y *= self.max_stick_value
                
                # 死區處理
                if abs(norm_x) < self.deadzone:
                    norm_x = 0.0
                if abs(norm_y) < self.deadzone:
                    norm_y = 0.0
                
                if norm_x == 0.0 and norm_y == 0.0:
                    return True
                
                # 設定右搖桿值
                # vgamepad 的 right_joystick_float: x_value_float, y_value_float
                # Y 軸: vgamepad 中 正=上，但遊戲中下移= dy>0
                # 所以反轉 Y 軸
                self._gamepad.right_joystick_float(
                    x_value_float=norm_x,
                    y_value_float=-norm_y  # 反轉 Y
                )
                self._gamepad.update()
                
                # 短暫維持搖桿位置
                if self.stick_duration > 0:
                    time.sleep(self.stick_duration)
                
                # 釋放搖桿（回中）
                self._gamepad.right_joystick_float(
                    x_value_float=0.0,
                    y_value_float=0.0
                )
                self._gamepad.update()
                
                self._move_count += 1
                return True
                
            except Exception as e:
                self._error_count += 1
                self._last_error = str(e)
                if self._error_count <= 3:
                    logger.error(f"[Xbox] 右搖桿移動失敗: {e}")
                
                # 嘗試重新連線
                if self._error_count > 5:
                    self._connected = False
                    self._gamepad = None
                return False
    
    def press_button(self, button) -> bool:
        """按下手把按鈕
        
        Args:
            button: vgamepad 按鈕常數 (例如 vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
            
        Returns:
            是否成功
        """
        if not self.ensure_initialized():
            return False
        
        with self._lock:
            try:
                self._gamepad.press_button(button=button)
                self._gamepad.update()
                return True
            except Exception as e:
                logger.error(f"[Xbox] 按鈕按下失敗: {e}")
                return False
    
    def release_button(self, button) -> bool:
        """釋放手把按鈕"""
        if not self.ensure_initialized():
            return False
        
        with self._lock:
            try:
                self._gamepad.release_button(button=button)
                self._gamepad.update()
                return True
            except Exception as e:
                logger.error(f"[Xbox] 按鈕釋放失敗: {e}")
                return False
    
    def click_button(self, button, duration: float = 0.05) -> bool:
        """點擊手把按鈕 (按下 + 釋放)
        
        Args:
            button: 按鈕常數
            duration: 按住時間（秒）
        """
        if self.press_button(button):
            time.sleep(duration)
            return self.release_button(button)
        return False
    
    def pull_right_trigger(self, value: float = 1.0) -> bool:
        """拉右扳機 (RT)
        
        Args:
            value: 0.0~1.0 (0=未按, 1=全按)
        """
        if not self.ensure_initialized():
            return False
        with self._lock:
            try:
                self._gamepad.right_trigger_float(value_float=value)
                self._gamepad.update()
                return True
            except Exception as e:
                logger.error(f"[Xbox] 右扳機失敗: {e}")
                return False
    
    def pull_left_trigger(self, value: float = 1.0) -> bool:
        """拉左扳機 (LT)"""
        if not self.ensure_initialized():
            return False
        with self._lock:
            try:
                self._gamepad.left_trigger_float(value_float=value)
                self._gamepad.update()
                return True
            except Exception as e:
                logger.error(f"[Xbox] 左扳機失敗: {e}")
                return False
    
    def reset(self) -> bool:
        """重置所有輸入"""
        if not self._connected or self._gamepad is None:
            return True
        with self._lock:
            try:
                self._gamepad.reset()
                self._gamepad.update()
                return True
            except Exception:
                return False
    
    def get_statistics(self) -> dict:
        """取得統計資料"""
        return {
            "connected": self._connected,
            "available": self.is_available(),
            "move_count": self._move_count,
            "error_count": self._error_count,
            "last_error": self._last_error,
            "sensitivity": self.sensitivity,
            "deadzone": self.deadzone,
        }


# ===== 全域單例 =====
xbox_controller = XboxController()


# ===== 公開函數 =====

def send_mouse_move_xbox(dx: float, dy: float) -> None:
    """透過 Xbox 360 虛擬手把右搖桿發送移動
    
    與 send_mouse_move_sendinput / send_mouse_move_mouse_event 相同介面
    """
    xbox_controller.move_right_stick(dx, dy)


def send_mouse_click_xbox(duration: float = 0.05) -> bool:
    """透過 Xbox 360 虛擬手把模擬射擊 (RT 扳機)"""
    if not xbox_controller.ensure_initialized():
        return False
    try:
        xbox_controller.pull_right_trigger(1.0)
        time.sleep(duration)
        xbox_controller.pull_right_trigger(0.0)
        return True
    except Exception:
        return False


def connect_xbox() -> bool:
    """連線虛擬 Xbox 360 手把"""
    return xbox_controller.connect()


def disconnect_xbox() -> None:
    """斷開虛擬 Xbox 360 手把"""
    xbox_controller.disconnect()


def is_xbox_connected() -> bool:
    """檢查虛擬手把是否已連線"""
    return xbox_controller.is_connected()


def is_xbox_available() -> bool:
    """檢查 vgamepad 是否可用"""
    return xbox_controller.is_available()


def set_xbox_sensitivity(value: float) -> None:
    """設定手把靈敏度"""
    xbox_controller.sensitivity = max(0.1, min(5.0, value))


def set_xbox_deadzone(value: float) -> None:
    """設定手把死區"""
    xbox_controller.deadzone = max(0.0, min(0.5, value))


def get_xbox_statistics() -> dict:
    """取得手把統計資料"""
    return xbox_controller.get_statistics()
