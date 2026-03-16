# arduino_mouse.py - Arduino Leonardo Mouse Control Module
"""
Achieve hardware-level mouse movement through the USB HID function of Arduino Leonardo.
Arduino Leonardo can simulate a native USB mouse, making it very stealthy.
"""

import os
import sys
import struct
import threading
import time
from typing import Optional

# 使用本地的依賴模組 (src/python/dependencies)
_src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_python_dir = os.path.join(_src_dir, 'python')
_deps_dir = os.path.join(_python_dir, 'dependencies')

# 確保依賴路徑優先（不刪除已載入的 serial 模組，避免影響 makcu_mouse 等）
if _deps_dir not in sys.path:
    sys.path.insert(0, _deps_dir)

import serial
import serial.tools.list_ports


class ArduinoMouse:
    """Arduino Leonardo Mouse Controller

    Uses the USB HID function of Arduino Leonardo to simulate mouse movement.
    """

    def __init__(self):
        self._serial: Optional[serial.Serial] = None
        self._lock = threading.Lock()
        self._connected = False
        self._com_port: str = ""
        self._baud_rate: int = 115200

    def connect(self, com_port: str, baud_rate: int = 115200) -> bool:
        """Connect to Arduino Leonardo

        Args:
            com_port: COM port (e.g., 'COM7')
            baud_rate: Baud rate, default 115200

        Returns:
            Whether the connection was successful
        """
        with self._lock:
            # Close old connection
            if self._serial and self._serial.is_open:
                try:
                    self._serial.close()
                except Exception:
                    pass
                self._connected = False

            try:
                self._serial = serial.Serial(com_port, baud_rate, timeout=0.1)
                self._com_port = com_port
                self._baud_rate = baud_rate
                self._connected = True
                # Wait for Arduino to restart (Leonardo automatically restarts on connection)
                time.sleep(2)
                print(f"[Arduino] Successfully connected to {com_port}")
                return True
            except serial.SerialException as e:
                print(f"[Arduino] Connection failed: {e}")
                self._connected = False
                return False
            except Exception as e:
                print(f"[Arduino] Error occurred during connection: {e}")
                self._connected = False
                return False

    def disconnect(self):
        """Disconnect"""
        with self._lock:
            if self._serial and self._serial.is_open:
                try:
                    self._serial.close()
                except Exception:
                    pass
            self._connected = False
            print("[Arduino] Disconnected")

    def is_connected(self) -> bool:
        """Check if connected"""
        return self._connected and self._serial is not None and self._serial.is_open

    def move(self, dx: int, dy: int):
        """Move mouse

        Args:
            dx: X 方向移動量 (-128 ~ 127)
            dy: Y 方向移動量 (-128 ~ 127)
        """
        if not self.is_connected():
            return

        # 限制範圍在 -128 到 127 之間 (signed char)
        dx = max(-128, min(127, int(dx)))
        dy = max(-128, min(127, int(dy)))

        try:
            # struct.pack('bbb', cmd, arg1, arg2)
            # cmd=1 表示移動
            data = struct.pack('bbb', 1, dx, dy)
            with self._lock:
                if self._serial and self._serial.is_open:
                    self._serial.write(data)
        except serial.SerialException:
            # 連線可能已斷開
            self._connected = False
        except Exception:
            pass

    @property
    def com_port(self) -> str:
        """當前連線的 COM 埠"""
        return self._com_port

    def click(self, action: int = 1):
        """執行滑鼠點擊
        
        Args:
            action: 1=點擊(按下後放開), 2=按下, 3=放開
        """
        if not self.is_connected():
            return
            
        try:
            # cmd=2 表示點擊, arg1=action, arg2=0
            data = struct.pack('bbb', 2, action, 0)
            with self._lock:
                if self._serial and self._serial.is_open:
                    self._serial.write(data)
        except serial.SerialException:
            # 連線可能已斷開
            self._connected = False
        except Exception:
            pass


# 全域單例
arduino_mouse = ArduinoMouse()


def send_mouse_move_arduino(dx: int, dy: int):
    """Arduino Leonardo 滑鼠移動(直接執行)"""
    arduino_mouse.move(dx, dy)


def send_mouse_click_arduino(action: int = 1):
    """Arduino Leonardo 滑鼠點擊"""
    arduino_mouse.click(action)
    return True


def get_available_com_ports() -> list[str]:
    """獲取可用的 COM 埠列表

    Returns:
        COM 埠名稱列表 (例如 ['COM1', 'COM3', 'COM7'])
    """
    ports = serial.tools.list_ports.comports()
    return [port.device for port in ports]


def connect_arduino(com_port: str, baud_rate: int = 115200) -> bool:
    """連線到 Arduino Leonardo

    Args:
        com_port: COM 埠 (例如 'COM7')
        baud_rate: 波特率, 預設 115200

    Returns:
        是否成功連線
    """
    return arduino_mouse.connect(com_port, baud_rate)


def disconnect_arduino():
    """斷開 Arduino 連線"""
    arduino_mouse.disconnect()


def is_arduino_connected() -> bool:
    """檢查 Arduino 是否已連線"""
    return arduino_mouse.is_connected()
