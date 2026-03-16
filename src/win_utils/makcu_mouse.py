# makcu_mouse.py - MAKCU Mouse Control Module
"""
Achieve hardware-level mouse movement through the MAKCU KM host device.
MAKCU acts as a USB HID proxy, injecting mouse/keyboard inputs at the hardware level.
Uses the Traditional ASCII API (e.g., .move(dx,dy)) over a serial connection.

API Reference: https://www.makcu.com/cn/api
"""

import os
import sys
import threading
import time
import logging
from typing import Optional

# 使用本地的依賴模組 (src/python/dependencies)
_src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_python_dir = os.path.join(_src_dir, 'python')
_deps_dir = os.path.join(_python_dir, 'dependencies')

# 確保依賴路徑優先
if _deps_dir not in sys.path:
    sys.path.insert(0, _deps_dir)

import serial
import serial.tools.list_ports

logger = logging.getLogger(__name__)


class MakcuMouse:
    """MAKCU KM Host Mouse Controller

    Uses the MAKCU device's ASCII serial API to inject hardware-level mouse inputs.
    Unlike Arduino Leonardo, MAKCU does not reset on serial connection, so no
    startup delay is needed. Supports int16 range for move dx/dy (much larger
    than Arduino's -128~127 signed char limit).
    """

    # ASCII command templates (Traditional API)
    # Standard MAKCU KM commands.
    CMD_MOVE = "km.move({dx},{dy})\r\n"
    CMD_CLICK = "km.click({button},{count})\r\n"
    CMD_LEFT_DOWN = "km.left(1)\r\n"
    CMD_LEFT_UP = "km.left(0)\r\n"
    CMD_ECHO_OFF = "km.echo(0)\r\n"
    CMD_VERSION = "km.version()\r\n"

    def __init__(self):
        self._serial: Optional[serial.Serial] = None
        self._lock = threading.Lock()
        self._connected = False
        self._com_port: str = ""
        self._baud_rate: int = 115200

    def connect(self, com_port: str, baud_rate: int = 115200) -> bool:
        """Connect to MAKCU device

        Args:
            com_port: COM port (e.g., 'COM3')
            baud_rate: Baud rate, default 115200 (MAKCU default)

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
                self._serial = serial.Serial(
                    com_port, baud_rate, timeout=0.1,
                    write_timeout=0.1
                )
                self._com_port = com_port
                self._baud_rate = baud_rate

                # MAKCU doesn't reset on connection (unlike Arduino Leonardo),
                # so we can start sending commands immediately.
                # Brief settle time for serial port.
                time.sleep(0.1)

                # Clear old data in the buffer
                self._serial.reset_input_buffer()

                # Send version command to verify it's a real MAKCU device
                self._serial.write(self.CMD_VERSION.encode('ascii'))
                time.sleep(0.1)  # Wait for a response

                # Check if device responded
                if self._serial.in_waiting == 0:
                    logger.error(f"[MAKCU] Handshake failed on {com_port}: No response from device.")
                    print(f"[MAKCU] Handshake failed on {com_port}: No response from device.")
                    self._serial.close()
                    self._connected = False
                    return False

                # Read out the version info
                version_info = self._serial.read(self._serial.in_waiting).decode('ascii', errors='ignore').strip()
                logger.info(f"[MAKCU] Device info: {version_info}")
                print(f"[MAKCU] Device responded: {version_info}")

                # Disable echo to reduce serial traffic
                self._serial.write(self.CMD_ECHO_OFF.encode('ascii'))
                # Flush any pending response
                time.sleep(0.05)
                self._serial.reset_input_buffer()

                self._connected = True
                logger.info(f"[MAKCU] Successfully connected to {com_port}")
                print(f"[MAKCU] Successfully connected to {com_port}")
                return True
            except serial.SerialException as e:
                logger.error(f"[MAKCU] Connection failed: {e}")
                print(f"[MAKCU] Connection failed: {e}")
                self._connected = False
                return False
            except Exception as e:
                logger.error(f"[MAKCU] Error occurred during connection: {e}")
                print(f"[MAKCU] Error occurred during connection: {e}")
                self._connected = False
                return False

    def disconnect(self):
        """Disconnect from MAKCU device"""
        with self._lock:
            if self._serial and self._serial.is_open:
                try:
                    self._serial.close()
                except Exception:
                    pass
            self._connected = False
            logger.info("[MAKCU] Disconnected")
            print("[MAKCU] Disconnected")

    def is_connected(self) -> bool:
        """Check if connected to MAKCU"""
        return self._connected and self._serial is not None and self._serial.is_open

    def move(self, dx: int, dy: int):
        """Move mouse (relative)

        MAKCU supports int16 range for dx/dy (-32768 ~ 32767),
        much larger than Arduino's -128 ~ 127 signed char limit.

        Args:
            dx: X direction movement (-32768 ~ 32767)
            dy: Y direction movement (-32768 ~ 32767)
        """
        if not self.is_connected():
            return

        # Clamp to int16 range
        dx = max(-32768, min(32767, int(dx)))
        dy = max(-32768, min(32767, int(dy)))

        try:
            cmd = self.CMD_MOVE.format(dx=dx, dy=dy)
            with self._lock:
                if self._serial and self._serial.is_open:
                    self._serial.write(cmd.encode('ascii'))
        except serial.SerialException:
            self._connected = False
        except Exception:
            pass

    def click(self, action: int = 1):
        """Perform mouse click

        Args:
            action: 1=click (press and release), 2=press down, 3=release
        """
        if not self.is_connected():
            return

        try:
            if action == 1:
                # Single left click: press then release
                # Use km.left(1) + km.left(0) for reliable hardware-level click
                with self._lock:
                    if self._serial and self._serial.is_open:
                        self._serial.write(self.CMD_LEFT_DOWN.encode('ascii'))
                        time.sleep(0.03)  # Brief hold for hardware to register
                        self._serial.write(self.CMD_LEFT_UP.encode('ascii'))
                return
            elif action == 2:
                # Left button press
                cmd = self.CMD_LEFT_DOWN
            elif action == 3:
                # Left button release
                cmd = self.CMD_LEFT_UP
            else:
                return

            with self._lock:
                if self._serial and self._serial.is_open:
                    self._serial.write(cmd.encode('ascii'))
        except serial.SerialException:
            self._connected = False
        except Exception:
            pass

    @property
    def com_port(self) -> str:
        """Currently connected COM port"""
        return self._com_port


# Global singleton
makcu_mouse = MakcuMouse()


def send_mouse_move_makcu(dx: int, dy: int):
    """MAKCU mouse move (direct execution)"""
    makcu_mouse.move(dx, dy)


def send_mouse_click_makcu(action: int = 1):
    """MAKCU mouse click"""
    makcu_mouse.click(action)
    return True


def connect_makcu(com_port: str, baud_rate: int = 115200) -> bool:
    """Connect to MAKCU device

    Args:
        com_port: COM port (e.g., 'COM3')
        baud_rate: Baud rate, default 115200

    Returns:
        Whether connection was successful
    """
    return makcu_mouse.connect(com_port, baud_rate)


def disconnect_makcu():
    """Disconnect MAKCU"""
    makcu_mouse.disconnect()


def is_makcu_connected() -> bool:
    """Check if MAKCU is connected"""
    return makcu_mouse.is_connected()
