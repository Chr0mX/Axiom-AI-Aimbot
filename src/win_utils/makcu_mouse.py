# makcu_mouse.py - MAKCU Mouse Control Module
"""
Achieve hardware-level mouse movement through the MAKCU KM host device.
MAKCU acts as a USB HID proxy, injecting mouse/keyboard inputs at the hardware level.
Uses the ASCII API over a serial connection at 4 Mbaud.

API Reference: https://makcu.k4tech.net/native/
"""

import os
import sys
import threading
import time
import logging
from typing import Optional

_src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_python_dir = os.path.join(_src_dir, 'python')
_deps_dir = os.path.join(_python_dir, 'dependencies')

if _deps_dir not in sys.path:
    sys.path.insert(0, _deps_dir)

import queue

import serial
import serial.tools.list_ports

logger = logging.getLogger(__name__)

# Official 4 Mbaud connection constants
_OPERATING_BAUD    = 4_000_000
_BAUD_CHANGE_FRAME = bytes([0xDE, 0xAD, 0x05, 0x00, 0xA5, 0x00, 0x09, 0x3D, 0x00])

# Button stream parsing — prefix emitted before each mask byte
# Do NOT use \n/\r splitting: mask 0x0A (R+S1) and 0x0D (L+M+S1) collide with newlines
_KM_PREFIX = bytes([0x6B, 0x6D, 0x2E])  # "km."
_BTN_BITS = 0x1F  # bits 0-4 = L,R,M,S1,S2 — everything else is noise/wheel/high-byte


class MakcuMouse:
    """MAKCU KM Host Mouse Controller

    Uses the MAKCU device's ASCII serial API to inject hardware-level mouse inputs.
    Connects at 4 Mbaud using the official DE AD baud-change sequence.
    Button state is maintained via the km.buttons(1) event stream — no polling.
    """

    CMD_MOVE      = "km.move({dx},{dy})\r\n"
    CMD_LEFT_DOWN = "km.left(1)\r\n"
    CMD_LEFT_UP   = "km.left(0)\r\n"
    CMD_ECHO_OFF  = "km.echo(0)\r\n"
    CMD_VERSION   = "km.version()\r\n"
    CMD_INFO      = "km.info()\r\n"

    def __init__(self):
        self._serial: Optional[serial.Serial] = None
        self._lock = threading.Lock()
        self._connected = False
        self._com_port: str = ""
        self._version_string: str = ""
        self._device_info: dict = {}

        # Button event stream state
        self._btn_mask: int = 0
        self._stream_stop  = threading.Event()
        self._stream_thread: Optional[threading.Thread] = None

        # Async write thread — inference thread enqueues commands; write thread
        # drains the queue and flushes to serial so inference is never blocked.
        self._cmd_queue: queue.Queue = queue.Queue(maxsize=1)
        self._write_stop = threading.Event()
        self._write_thread: Optional[threading.Thread] = None

        # Reconnect watchdog — re-establishes connection after USB glitches.
        self._reconnect_thread: Optional[threading.Thread] = None

        # Legacy attribute kept so ai_loop.py can set it without errors
        self.lmb_cache_seconds: float = 0.008

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self, com_port: str, baud_rate: int = _OPERATING_BAUD) -> bool:
        """Connect to MAKCU device using the official 4 Mbaud sequence.

        Always targets 4 Mbaud regardless of baud_rate. Tries 4M first
        (warm start / same power cycle), then falls back to sending the
        DE AD baud-change frame at 115200 before reopening at 4M.
        """
        with self._lock:
            self._close_locked()

            # Step 1: device may already be at 4M from a previous session
            if self._try_open_locked(com_port, _OPERATING_BAUD):
                self._connected = True
                self._com_port = com_port
                self._query_info_locked()
                logger.info("[MAKCU] Connected to %s @ %d baud", com_port, _OPERATING_BAUD)
            else:
                # Step 2: send DE AD baud-change at 115200 then reopen at 4M
                try:
                    s = serial.Serial(com_port, 115200, timeout=0.5, write_timeout=0.1)
                    time.sleep(0.05)
                    s.write(_BAUD_CHANGE_FRAME)
                    s.flush()
                    time.sleep(0.1)
                    s.close()
                except serial.SerialException as exc:
                    logger.error("[MAKCU] Baud-change frame failed: %s", exc)
                    return False

                time.sleep(0.05)

                if not self._try_open_locked(com_port, _OPERATING_BAUD):
                    logger.error("[MAKCU] Could not connect to %s after baud change", com_port)
                    return False

                self._connected = True
                self._com_port = com_port
                self._query_info_locked()
                logger.info("[MAKCU] Connected to %s @ %d baud", com_port, _OPERATING_BAUD)

        # Start threads outside the lock
        self._start_stream()
        self._start_write_thread()
        self._start_reconnect_thread()
        return True

    def _try_open_locked(self, com_port: str, baud: int) -> bool:
        """Open port at baud, probe with km.version(). Returns True if km.MAKCU found."""
        try:
            self._serial = serial.Serial(com_port, baud, timeout=0.3, write_timeout=0.1)
            time.sleep(0.05)
            self._serial.reset_input_buffer()
            self._serial.write(self.CMD_VERSION.encode('ascii'))
            self._serial.flush()
            deadline = time.monotonic() + 0.5
            raw = b''
            while time.monotonic() < deadline:
                if self._serial.in_waiting:
                    raw += self._serial.read(self._serial.in_waiting)
                    if b'km.MAKCU' in raw:
                        break
                else:
                    time.sleep(0.01)
            if b'km.MAKCU' not in raw:
                self._close_locked()
                return False
            self._version_string = raw.decode('ascii', errors='ignore').replace('>>>', '').strip()
            self._serial.write(self.CMD_ECHO_OFF.encode('ascii'))
            self._serial.flush()
            time.sleep(0.1)
            self._serial.reset_input_buffer()
            return True
        except serial.SerialException as exc:
            logger.debug("[MAKCU] _try_open_locked %s@%d failed: %s", com_port, baud, exc)
            self._close_locked()
            return False

    def _close_locked(self):
        """Close serial port. Caller must hold _lock or be in single-threaded context."""
        if self._serial:
            try:
                self._serial.close()
            except Exception:
                pass
            self._serial = None
        self._connected = False

    # ------------------------------------------------------------------
    # Button event stream
    # ------------------------------------------------------------------

    def _start_stream(self):
        """Enable km.buttons(1) stream and start the reader thread."""
        self._stream_stop.clear()
        self._btn_mask = 0
        try:
            with self._lock:
                if self._serial and self._serial.is_open:
                    self._serial.write(b'km.buttons(1)\r\n')
                    self._serial.flush()
        except Exception:
            return
        self._stream_thread = threading.Thread(
            target=self._stream_reader, daemon=True, name="makcu-stream")
        self._stream_thread.start()

    def _stop_stream(self):
        """Stop the reader thread and send km.buttons(0)."""
        self._stream_stop.set()
        if self._stream_thread:
            self._stream_thread.join(timeout=1.0)
            self._stream_thread = None
        try:
            with self._lock:
                if self._serial and self._serial.is_open:
                    self._serial.write(b'km.buttons(0)\r\n')
                    self._serial.flush()
        except Exception:
            pass
        self._btn_mask = 0

    # ------------------------------------------------------------------
    # Async write thread
    # ------------------------------------------------------------------

    def _start_write_thread(self):
        """Start the async write thread if not already running."""
        if self._write_thread and self._write_thread.is_alive():
            return
        self._write_stop.clear()
        self._write_thread = threading.Thread(
            target=self._write_worker, daemon=True, name="makcu-write")
        self._write_thread.start()

    def _start_reconnect_thread(self):
        """Start the reconnect watchdog thread if not already running."""
        if self._reconnect_thread and self._reconnect_thread.is_alive():
            return
        self._reconnect_thread = threading.Thread(
            target=self._reconnect_worker, daemon=True, name="makcu-reconnect")
        self._reconnect_thread.start()

    def _write_worker(self):
        """Drain the command queue and write+flush to serial.

        Runs on its own thread so the inference thread (which calls move())
        is never blocked waiting on the serial port.
        """
        while not self._write_stop.is_set():
            try:
                cmd = self._cmd_queue.get(timeout=0.01)
                with self._lock:
                    if self._serial and self._serial.is_open:
                        self._serial.write(cmd.encode('ascii'))
                        self._serial.flush()
            except queue.Empty:
                continue
            except serial.SerialException:
                self._connected = False
            except Exception:
                pass

    def _reconnect_worker(self):
        """Watchdog: re-establish connection automatically after USB glitches."""
        while not self._write_stop.is_set():
            self._write_stop.wait(2.0)
            if self._write_stop.is_set():
                break
            if not self.is_connected() and self._com_port:
                logger.info("[MAKCU] Connection lost — reconnecting on %s", self._com_port)
                try:
                    self.connect(self._com_port)
                except Exception as exc:
                    logger.debug("[MAKCU] Reconnect failed: %s", exc)

    def _stream_reader(self):
        """Daemon thread: parse km. + 2-byte LE button mask frames, update _btn_mask.

        Frame layout (firmware buttons stream): 0x6B 0x6D 0x2E <maskLO> <maskHI>.
        The mask is little-endian; only bits 0-4 are real buttons (L,R,M,S1,S2).
        Masking to _BTN_BITS prevents a desynced wheel/data byte — e.g. a scroll
        delta of +1 (0x01) or -1 (0xFF), both of which carry bit 0 — from being
        misread as a held left button.

        km.echo(0) means the device sends nothing in response to move/click writes,
        so all incoming bytes are button stream events — no lock needed on reads.
        """
        buf = bytearray()
        FRAME_LEN = len(_KM_PREFIX) + 2  # km. + 2-byte mask
        while not self._stream_stop.is_set():
            try:
                ser = self._serial
                if not ser or not ser.is_open:
                    break
                n = ser.in_waiting
                if n:
                    buf.extend(ser.read(n))
                    if len(buf) > 256:
                        buf.clear()
                    while len(buf) >= FRAME_LEN:
                        idx = buf.find(_KM_PREFIX)
                        if idx == -1:
                            # No prefix in buffer; keep only a possible partial tail
                            del buf[:max(0, len(buf) - (len(_KM_PREFIX) - 1))]
                            break
                        if idx + FRAME_LEN > len(buf):
                            # Full frame not yet arrived; drop bytes before prefix and wait
                            del buf[:idx]
                            break
                        lo = buf[idx + 3]
                        hi = buf[idx + 4]
                        self._btn_mask = (lo | (hi << 8)) & _BTN_BITS
                        del buf[:idx + FRAME_LEN]
                else:
                    time.sleep(0.001)
            except Exception:
                break

    # ------------------------------------------------------------------
    # Info query
    # ------------------------------------------------------------------

    def _query_info_locked(self) -> dict:
        """Send km.info() and parse key=value pairs. Caller must hold _lock."""
        if not self._serial:
            return {}
        try:
            self._serial.reset_input_buffer()
            self._serial.write(self.CMD_INFO.encode('ascii'))
            self._serial.flush()
            time.sleep(0.15)
            raw = self._serial.read(self._serial.in_waiting).decode('ascii', errors='ignore')
            info = {}
            for line in raw.splitlines():
                line = line.strip().replace('>>>', '').strip()
                if '=' in line:
                    k, _, v = line.partition('=')
                    info[k.strip().upper()] = v.strip()
            self._device_info = info
            return info
        except Exception:
            return {}

    def query_info(self) -> dict:
        """Return parsed km.info() dict."""
        with self._lock:
            return self._query_info_locked()

    @property
    def device_info(self) -> dict:
        return dict(self._device_info)

    @property
    def version_string(self) -> str:
        return self._version_string

    # ------------------------------------------------------------------
    # Disconnect
    # ------------------------------------------------------------------

    def disconnect(self):
        """Stop all threads then close serial port."""
        self._write_stop.set()
        if self._write_thread:
            self._write_thread.join(timeout=1.0)
            self._write_thread = None
        if self._reconnect_thread:
            self._reconnect_thread.join(timeout=1.0)
            self._reconnect_thread = None
        self._stop_stream()
        with self._lock:
            self._close_locked()
        logger.info("[MAKCU] Disconnected")

    def is_connected(self) -> bool:
        return self._connected and self._serial is not None and self._serial.is_open

    @property
    def com_port(self) -> str:
        return self._com_port

    # ------------------------------------------------------------------
    # Mouse control
    # ------------------------------------------------------------------

    def move(self, dx: int, dy: int):
        """Relative mouse move. Enqueues command for async write thread.

        Latest-only: if a previous move hasn't been sent yet it is replaced,
        so the inference thread never blocks on serial I/O.
        """
        if not self.is_connected():
            return
        dx = max(-32768, min(32767, int(dx)))
        dy = max(-32768, min(32767, int(dy)))
        cmd = self.CMD_MOVE.format(dx=dx, dy=dy)
        # Drop stale command if queue is full, then enqueue the latest.
        try:
            self._cmd_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            self._cmd_queue.put_nowait(cmd)
        except queue.Full:
            pass

    def click(self, action: int = 1):
        """Left mouse click. action: 1=click, 2=press, 3=release."""
        if not self.is_connected():
            return
        try:
            if action == 1:
                with self._lock:
                    if self._serial and self._serial.is_open:
                        self._serial.write(self.CMD_LEFT_DOWN.encode('ascii'))
                time.sleep(0.03)
                with self._lock:
                    if self._serial and self._serial.is_open:
                        self._serial.write(self.CMD_LEFT_UP.encode('ascii'))
                return
            cmd = self.CMD_LEFT_DOWN if action == 2 else self.CMD_LEFT_UP if action == 3 else None
            if cmd:
                with self._lock:
                    if self._serial and self._serial.is_open:
                        self._serial.write(cmd.encode('ascii'))
        except serial.SerialException:
            self._connected = False
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Button state — read from live stream mask, no serial I/O
    # ------------------------------------------------------------------

    @property
    def lmb_held(self) -> bool:
        """True when left mouse button is physically pressed."""
        return bool(self._btn_mask & 0x01)

    @property
    def rmb_held(self) -> bool:
        """True when right mouse button is physically pressed."""
        return bool(self._btn_mask & 0x02)

    def query_lmb_state(self) -> int:
        """Return 1 if LMB pressed, 0 if released. No serial I/O."""
        return 1 if (self._btn_mask & 0x01) else 0

    def query_rmb_state(self) -> int:
        """Return 1 if RMB pressed, 0 if released. No serial I/O."""
        return 1 if (self._btn_mask & 0x02) else 0


# ---------------------------------------------------------------------------
# Module-level singleton and convenience functions
# ---------------------------------------------------------------------------

makcu_mouse = MakcuMouse()


def send_mouse_move_makcu(dx: int, dy: int):
    makcu_mouse.move(dx, dy)


def send_mouse_click_makcu(action: int = 1):
    makcu_mouse.click(action)
    return True


def connect_makcu(com_port: str, baud_rate: int = _OPERATING_BAUD) -> bool:
    return makcu_mouse.connect(com_port, baud_rate)


def disconnect_makcu():
    makcu_mouse.disconnect()


def is_makcu_connected() -> bool:
    return makcu_mouse.is_connected()
