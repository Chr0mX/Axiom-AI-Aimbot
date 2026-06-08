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
    CMD_BAUD_4M = b"km.baud(4000000)\r\n"

    def __init__(self):
        self._serial: Optional[serial.Serial] = None
        self._lock = threading.Lock()
        self._connected = False
        self._com_port: str = ""
        self._baud_rate: int = 115200
        self._lmb_state_cache: int = 0
        self._lmb_cache_time: float = 0.0

    def connect(self, com_port: str, baud_rate: int = 115200) -> bool:
        """Connect to MAKCU device.

        MAKCU supports only 115200 and 4000000 baud.  We always open at 115200
        first (the device's startup rate), perform the handshake, then if the
        caller requested 4 Mbaud we send km.baud(4000000), close and reopen at
        4 Mbaud.
        """
        # Normalise: accept only the two supported rates
        target_baud = 4_000_000 if baud_rate == 4_000_000 else 115200

        with self._lock:
            # Close existing connection
            if self._serial and self._serial.is_open:
                try:
                    self._serial.close()
                except Exception:
                    pass
                self._connected = False

            try:
                # Always open at 115200 — the device's startup/default rate
                if not self._open_serial(com_port):
                    self._connected = False
                    return False

                self._com_port = com_port

                # Brief settle time
                time.sleep(0.1)
                self._serial.reset_input_buffer()

                # Verify device with version command
                self._serial.write(self.CMD_VERSION.encode('ascii'))
                time.sleep(0.1)

                if self._serial.in_waiting == 0:
                    logger.error("[MAKCU] Handshake failed on %s: no response.", com_port)
                    print(f"[MAKCU] Handshake failed on {com_port}: no response from device.")
                    self._serial.close()
                    self._connected = False
                    return False

                version_info = self._serial.read(self._serial.in_waiting).decode('ascii', errors='ignore').strip()
                logger.info("[MAKCU] Device info: %s", version_info)
                print(f"[MAKCU] Device responded: {version_info}")

                # Disable echo to reduce serial traffic
                self._serial.write(self.CMD_ECHO_OFF.encode('ascii'))
                time.sleep(0.05)
                self._serial.reset_input_buffer()

                # Switch to 4 Mbaud if requested
                if target_baud == 4_000_000:
                    # Device switches baud immediately on processing the command.
                    # Flush ensures all bytes leave the host TX buffer, then
                    # close straight away — no extra sleep at the old rate.
                    self._serial.write(self.CMD_BAUD_4M)
                    self._serial.flush()
                    self._serial.close()
                    time.sleep(0.15)       # give OS time to release the port
                    self._serial = serial.Serial(
                        com_port, 4_000_000, timeout=0.3, write_timeout=0.1)
                    time.sleep(0.05)       # port settle
                    self._serial.reset_input_buffer()
                    # Verify 4M link — device must respond at new baud
                    self._serial.write(self.CMD_VERSION.encode('ascii'))
                    time.sleep(0.1)
                    if self._serial.in_waiting == 0:
                        raise serial.SerialException(
                            f"4 Mbaud handshake failed on {com_port}: "
                            "no response after baud switch")
                    self._serial.read(self._serial.in_waiting)  # drain response
                    # Re-disable echo at new baud
                    self._serial.write(self.CMD_ECHO_OFF.encode('ascii'))
                    time.sleep(0.05)
                    self._serial.reset_input_buffer()
                    logger.info("[MAKCU] Switched to 4 Mbaud on %s", com_port)
                    print(f"[MAKCU] Switched to 4 Mbaud on {com_port}")

                self._baud_rate = target_baud
                self._connected = True
                logger.info("[MAKCU] Connected to %s @ %d baud", com_port, target_baud)
                print(f"[MAKCU] Successfully connected to {com_port} @ {target_baud} baud")
                return True

            except serial.SerialException as e:
                logger.error("[MAKCU] Connection failed: %s", e)
                print(f"[MAKCU] Connection failed: {e}")
                self._connected = False
                return False
            except Exception as e:
                logger.error("[MAKCU] Error during connection: %s", e)
                print(f"[MAKCU] Error during connection: {e}")
                self._connected = False
                return False

    def _open_serial(self, com_port: str) -> bool:
        """Open serial port at 115200 (MAKCU startup rate).

        Returns True on success, False on failure.
        """
        try:
            self._serial = serial.Serial(com_port, 115200, timeout=0.1, write_timeout=0.005)
            return True
        except serial.SerialException as e:
            logger.error("[MAKCU] Cannot open %s: %s", com_port, e)
            print(f"[MAKCU] Cannot open {com_port}: {e}")
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

    def query_lmb_state(self) -> int:
        """Send km.left() and parse response.

        Returns:
            0 = released, 1 = pressed (combined physical and software state).
        Cached for 16 ms so the aim loop doesn't flood the serial link.
        """
        import re as _re
        now = time.monotonic()
        if now - self._lmb_cache_time < 0.016:
            return self._lmb_state_cache
        with self._lock:
            if not self._serial or not self._serial.is_open:
                return 0
            try:
                self._serial.write(b"km.left()\r\n")
                resp = self._serial.read_until(b">>>")
            except Exception:
                return 0
        # Echo is disabled (km.echo(0) sent on connect), so response is "0\r\n>>>" or "1\r\n>>>"
        m = _re.search(rb'([01])\r?\n', resp)
        val = int(m.group(1)) if m else 0
        self._lmb_state_cache = val
        self._lmb_cache_time = now
        return val

    @property
    def lmb_held(self) -> bool:
        """True when LMB is pressed (km.left() returns 1=pressed, 0=released)."""
        return self.query_lmb_state() == 1

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
