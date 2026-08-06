"""
MAKCU V2 Binary Protocol — high-level serial API.

Drop-in companion to makcu_mouse.py (ASCII API).
Uses the binary frame format:
    RX (PC→Device): [0x50][CMD][LEN_LO][LEN_HI][PAYLOAD...]
    TX (Device→PC): [0x50][CMD][LEN_LO][LEN_HI][STATUS or DATA]

RESERVED — not wired into production.

MAKCU V2 binary-protocol variant, kept for a future firmware revision.
Nothing imports this outside makcu_debug_binary.py. Before wiring it in,
fix its lock-across-sleep violation in connect()/_send_cmd(): holding
_lock across a time.sleep() stalls the inference thread's move()/click()
for the sleep's duration. makcu_mouse.py (the shipping ASCII variant) is
written specifically to avoid that and is the model to follow.
tests/test_gui_invariants.py asserts this violation is still present, so
this note can't quietly go stale.
"""

import os
import re
import struct
import sys
import threading
import time

# Ensure bundled pyserial is on the path
_src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_deps_dir = os.path.join(_src_dir, 'python', 'dependencies')
sys.path.insert(0, _deps_dir)

import serial
import serial.tools.list_ports

# Lazy import decoder so this module can be imported without circular issues
from makcu_binary_decoder import BinaryDecoder, StatusResponse

# ---------------------------------------------------------------------------
# Protocol constants
# ---------------------------------------------------------------------------

_HDR = 0x50

_CMD_MOVE       = 0x0D
_CMD_BTN_LEFT   = 0x08
_CMD_BTN_RIGHT  = 0x11
_CMD_BTN_MID    = 0x0A
_CMD_BTN_SIDE1  = 0x12
_CMD_BTN_SIDE2  = 0x13
_CMD_WHEEL      = 0x18
_CMD_BAUD       = 0xB1
_CMD_ECHO       = 0xB4
_CMD_VERSION    = 0xBF
_CMD_INFO       = 0xB8
_CMD_REBOOT     = 0xBB
_CMD_MOUSE_STREAM   = 0x0C
_CMD_BUTTONS_STREAM = 0x02

_BTN_CMD: dict = {
    "left":   _CMD_BTN_LEFT,
    "right":  _CMD_BTN_RIGHT,
    "middle": _CMD_BTN_MID,
    "side1":  _CMD_BTN_SIDE1,
    "side2":  _CMD_BTN_SIDE2,
}

_DEFAULT_BAUD = 115_200
_RESPONSE_TIMEOUT = 0.3   # seconds


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_frame(cmd: int, payload: bytes = b'') -> bytes:
    length = len(payload)
    return bytes([_HDR, cmd, length & 0xFF, (length >> 8) & 0xFF]) + payload


def _sorted_ports(ports):
    """Sort COM ports descending by trailing port number (highest first)."""
    def _num(p):
        m = re.search(r'(\d+)$', p.device)
        return int(m.group(1)) if m else 0
    return sorted(ports, key=_num, reverse=True)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class MakcuMouseBinary:
    """
    Binary-protocol MAKCU serial controller.

    Usage::

        dev = MakcuMouseBinary()
        dev.connect()          # auto-select highest COM port
        dev.move(10, -5)
        dev.click("left", 1)   # press
        dev.click("left", 0)   # release
        dev.disconnect()
    """

    def __init__(self) -> None:
        self._serial: serial.Serial | None = None
        self._lock = threading.Lock()
        self._connected = False
        self._decoder = BinaryDecoder()

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def connect(self, com_port: str = '', baud: int = _DEFAULT_BAUD) -> bool:
        """Open the serial port and initialise the binary protocol.

        If *com_port* is empty the highest-numbered available port is used.
        Always opens at 115200 first; if *baud* differs, switches after
        disabling echo (mirrors ASCII module behaviour).
        """
        with self._lock:
            self._close_locked()

            if not com_port:
                ports = _sorted_ports(serial.tools.list_ports.comports())
                if not ports:
                    return False
                com_port = ports[0].device

            try:
                self._serial = serial.Serial(
                    com_port, _DEFAULT_BAUD,
                    timeout=0.3, write_timeout=0.1
                )
                time.sleep(0.1)
            except serial.SerialException as exc:
                self._serial = None
                return False

            # Disable echo so responses are clean
            if not self._write_locked(_build_frame(_CMD_ECHO, b'\x00')):
                self._close_locked()
                return False
            self._drain_locked(0.1)  # discard echo-disable response

            # Optional baud switch
            if baud != _DEFAULT_BAUD:
                frame = _build_frame(_CMD_BAUD, struct.pack('<I', baud))
                if not self._write_locked(frame):
                    self._close_locked()
                    return False
                self._serial.flush()
                time.sleep(0.15)
                try:
                    self._serial.close()
                    self._serial = serial.Serial(
                        com_port, baud,
                        timeout=0.3, write_timeout=0.1
                    )
                    time.sleep(0.05)
                except serial.SerialException:
                    self._serial = None
                    return False
                # Re-disable echo at new baud
                self._write_locked(_build_frame(_CMD_ECHO, b'\x00'))
                self._drain_locked(0.1)

            self._connected = True
            return True

    def disconnect(self) -> None:
        with self._lock:
            self._close_locked()

    def is_connected(self) -> bool:
        return self._connected

    # ------------------------------------------------------------------
    # High-level API
    # ------------------------------------------------------------------

    def move(self, dx: int, dy: int) -> bool:
        """Send a relative mouse move."""
        payload = struct.pack('<hh', max(-32768, min(32767, int(dx))),
                                     max(-32768, min(32767, int(dy))))
        return self._send_cmd(_CMD_MOVE, payload)

    def click(self, button: str, state: int) -> bool:
        """Press or release a mouse button.

        *button* is one of: ``"left"``, ``"right"``, ``"middle"``,
        ``"side1"``, ``"side2"``.
        *state*: ``1`` = press, ``0`` = release.
        """
        cmd = _BTN_CMD.get(button.lower())
        if cmd is None:
            raise ValueError(f"Unknown button '{button}'. "
                             f"Valid: {list(_BTN_CMD)}")
        return self._send_cmd(cmd, bytes([state & 0xFF]))

    def scroll(self, delta: int) -> bool:
        """Scroll the mouse wheel by *delta* steps (+up / -down)."""
        return self._send_cmd(_CMD_WHEEL, struct.pack('<b', max(-127, min(127, int(delta)))))

    def send_raw(self, cmd: int, payload: bytes = b'') -> bytes:
        """Build and send an arbitrary binary frame; return raw response bytes."""
        frame = _build_frame(cmd, payload)
        with self._lock:
            if not self._connected or self._serial is None:
                raise RuntimeError("Not connected")
            self._write_locked(frame)
            return self._read_raw_locked(_RESPONSE_TIMEOUT)

    # ------------------------------------------------------------------
    # Internal helpers (all called with _lock held unless noted)
    # ------------------------------------------------------------------

    def _send_cmd(self, cmd: int, payload: bytes) -> bool:
        """Send a frame and return True on OK status, False on ERR/failure."""
        frame = _build_frame(cmd, payload)
        with self._lock:
            if not self._connected or self._serial is None:
                return False
            if not self._write_locked(frame):
                return False
            raw = self._read_raw_locked(_RESPONSE_TIMEOUT)

        frames = self._decoder.feed(raw)
        for f in frames:
            if isinstance(f, StatusResponse) and f.cmd == cmd:
                return f.ok
        return True  # no status frame returned — assume success (fire-and-forget)

    def _write_locked(self, data: bytes) -> bool:
        try:
            self._serial.write(data)
            return True
        except serial.SerialException:
            self._connected = False
            return False

    def _read_raw_locked(self, timeout: float) -> bytes:
        """Poll for incoming bytes for up to *timeout* seconds."""
        deadline = time.monotonic() + timeout
        buf = b''
        while time.monotonic() < deadline:
            waiting = self._serial.in_waiting
            if waiting:
                try:
                    buf += self._serial.read(waiting)
                except serial.SerialException:
                    self._connected = False
                    break
            else:
                time.sleep(0.01)
        return buf

    def _drain_locked(self, timeout: float) -> None:
        """Discard all incoming bytes for *timeout* seconds."""
        self._read_raw_locked(timeout)

    def _close_locked(self) -> None:
        self._connected = False
        if self._serial is not None:
            try:
                self._serial.close()
            except Exception:
                pass
            self._serial = None


# ---------------------------------------------------------------------------
# Module-level singleton + convenience functions
# ---------------------------------------------------------------------------

makcu_binary = MakcuMouseBinary()


def connect_binary(port: str = '', baud: int = _DEFAULT_BAUD) -> bool:
    return makcu_binary.connect(port, baud)


def disconnect_binary() -> None:
    makcu_binary.disconnect()


def is_binary_connected() -> bool:
    return makcu_binary.is_connected()


def send_mouse_move_binary(dx: int, dy: int) -> bool:
    return makcu_binary.move(dx, dy)
