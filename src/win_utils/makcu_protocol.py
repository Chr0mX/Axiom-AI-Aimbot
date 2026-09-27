# makcu_protocol.py - MAK_API framing and helpers for MAKCU V4 / MAKXD devices
"""
Pure protocol helpers for the MAK_API command interface, implemented from the
terrafirma2021/mak-suite protocol documents (protocol/MAK_API.md, KM_API.md,
DEVICE_SETTINGS.md). No serial I/O here except the UDP transport adapter.

Frame (both directions):  DE AD | LEN:u16 LE | CMD:u8 | PAYLOAD[LEN]
Accepted SETs send no reply; a rejected GET/SET replies DE AD 01 00 CMD FF.
"""

from __future__ import annotations

import socket
import struct
import threading
from typing import Dict, Optional, Tuple

SYNC = b"\xde\xad"

# Commands (MAK_API.md)
CMD_DEVICE = 0x02
CMD_FIRMWARE_VERSION = 0x04
CMD_BUTTONS = 0x10
CMD_LEFT = 0x11
CMD_RIGHT = 0x12
CMD_MIDDLE = 0x13
CMD_SIDE1 = 0x14
CMD_SIDE2 = 0x15
CMD_MOVE_MASK = 0x16
CMD_WHEEL_MASK = 0x17
CMD_MOVE = 0x18
CMD_WHEEL = 0x19
CMD_LEFT_MASK = 0x1A  # ..0x1E for right/middle/side1/side2
CMD_KEY_DOWN = 0x20
CMD_KEY_UP = 0x21
CMD_KEY_INIT = 0x22
CMD_KEY_PRESS = 0x23
CMD_KEY_MASK = 0x29
CMD_KEY_REMAP = 0x2A
CMD_KEY_KEYS = 0x2B
CMD_CONNECTION = 0x3E
CMD_INPUT_CHANGE = 0x53

STREAM_KIND_MOUSE = 1
STREAM_KIND_KEYBOARD = 2

# Mouse button index (stream id / mask bit) -> per-button command offset.
BUTTON_NAMES = ("left", "right", "middle", "side1", "side2")

DEVICE_KIND_NAMES = (
    (0x01, "mouse"), (0x02, "keyboard"), (0x04, "HID controller"), (0x08, "DS4"),
    (0x10, "DualSense"), (0x20, "DualSense Edge"), (0x40, "Xbox GIP"), (0x80, "Xbox 360"),
)

# Live device settings (DEVICE_SETTINGS.md "Wire contract")
SETTINGS_RECORD = 0x1D
SETTINGS_OP_INFO, SETTINGS_OP_READ, SETTINGS_OP_BEGIN = 0, 1, 2
SETTINGS_OP_WRITE, SETTINGS_OP_APPLY, SETTINGS_OP_SAVE = 3, 4, 5
SETTINGS_IMAGE_BYTES = 400
SETTINGS_CHUNK = 96
SETTINGS_MOUSE_SPREAD_OFFSET = 396
SETTINGS_SECTION_MOUSE = 0x04
SETTINGS_STATUS = {0: "success", 1: "saving", 2: "busy", 3: "stale", 4: "invalid",
                   5: "unsupported", 6: "storage error"}
TOPOLOGY_NOTICE = 0x12  # async CONNECTION payload prefix — not a settings reply


def frame(cmd: int, payload: bytes = b"") -> bytes:
    return SYNC + struct.pack("<HB", len(payload), cmd) + bytes(payload)


def is_rejection(payload: bytes) -> bool:
    return payload == b"\xff"


def move_frame(dx: int, dy: int) -> bytes:
    dx = max(-32768, min(32767, int(dx)))
    dy = max(-32768, min(32767, int(dy)))
    return frame(CMD_MOVE, struct.pack("<hh", dx, dy))


def button_frame(button: int, state: bool) -> bytes:
    return frame(CMD_LEFT + button, bytes([1 if state else 0]))


def button_mask_frame(button: int, enabled: bool) -> bytes:
    return frame(CMD_LEFT_MASK + button, bytes([1 if enabled else 0]))


def move_mask_frame(left: bool, right: bool, down: bool, up: bool) -> bytes:
    return frame(CMD_MOVE_MASK, bytes(1 if v else 0 for v in (left, right, down, up)))


def wheel_frame(delta: int) -> bytes:
    return frame(CMD_WHEEL, struct.pack("<h", max(-32768, min(32767, int(delta)))))


def wheel_mask_frame(down: bool, up: bool) -> bytes:
    return frame(CMD_WHEEL_MASK, bytes([1 if down else 0, 1 if up else 0]))


def key_press_frame(key: int, hold_ms: Optional[int] = None, random_range: Optional[int] = None) -> bytes:
    payload = bytes([key & 0xFF])
    if hold_ms is not None:
        payload += struct.pack("<I", max(0, int(hold_ms)))
        if random_range is not None:
            payload += struct.pack("<I", max(0, int(random_range)))
    return frame(CMD_KEY_PRESS, payload)


def stream_enable_frame(kind: int, enabled: bool) -> bytes:
    cmd = CMD_BUTTONS if kind == STREAM_KIND_MOUSE else CMD_KEY_KEYS
    return frame(cmd, bytes([1 if enabled else 0]))


def describe_device_kinds(kinds: int) -> str:
    names = [name for bit, name in DEVICE_KIND_NAMES if kinds & bit]
    return "+".join(names) if names else "none"


def parse_km_device(text: str) -> Optional[Dict[str, str]]:
    """Parse a V4 `km.device()` reply `R:<routes>;M:<n>uf;K:<n>uf;C:<n>uf`.

    Returns None for anything else — notably V3 firmware, whose km.device()
    answers `mouse`/`keyboard`/`none`, which is how V4 is told apart safely
    using a command both generations accept.
    """
    for line in text.replace(">>>", "\n").splitlines():
        line = line.strip()
        if not line.startswith("R:") or ";" not in line:
            continue
        out: Dict[str, str] = {}
        for part in line.split(";"):
            key, sep, value = part.partition(":")
            if sep:
                out[key.strip()] = value.strip()
        if "R" in out and "M" in out:
            return out
    return None


def usb_period_to_hz(period: str) -> Optional[float]:
    """`8uf` (USB microframes of 125us) -> 1000.0 Hz. None if unparsable/zero."""
    try:
        frames = int(period.rstrip("uf"))
    except (AttributeError, ValueError):
        return None
    return 8000.0 / frames if frames > 0 else None


# ── Live settings (mouse spread) ─────────────────────────────────────────────

def settings_frame(op: int, body: bytes = b"") -> bytes:
    return frame(CMD_CONNECTION, bytes([SETTINGS_RECORD, op]) + body)


def settings_read_frame(revision: int, offset: int, length: int) -> bytes:
    return settings_frame(SETTINGS_OP_READ, struct.pack("<IHB", revision, offset, length))


def settings_begin_frame(revision: int, sections: int) -> bytes:
    return settings_frame(SETTINGS_OP_BEGIN, struct.pack("<IB", revision, sections))


def settings_write_frame(token: int, offset: int, data: bytes) -> bytes:
    return settings_frame(SETTINGS_OP_WRITE, struct.pack("<IH", token, offset) + bytes(data))


def settings_apply_frame(token: int) -> bytes:
    return settings_frame(SETTINGS_OP_APPLY, struct.pack("<I", token))


def settings_save_frame(revision: int, sections: int) -> bytes:
    return settings_frame(SETTINGS_OP_SAVE, struct.pack("<IB", revision, sections))


def parse_settings_reply(payload: bytes) -> Optional[Tuple[int, int, bytes]]:
    """(op, status, body) for a settings reply payload, else None (topology
    notices and anything that isn't record 0x1D)."""
    if len(payload) < 3 or payload[0] != SETTINGS_RECORD:
        return None
    return payload[1], payload[2], bytes(payload[3:])


def parse_settings_info(body: bytes) -> Optional[Dict[str, int]]:
    if len(body) < 11:
        return None
    version, sections, kinds, save_state, _zero, revision, image_bytes = struct.unpack_from("<BBBBBIH", body)
    return {"version": version, "sections": sections, "kinds": kinds,
            "save_state": save_state, "revision": revision, "image_bytes": image_bytes}


# ── Windows VK code -> USB HID keyboard usage ────────────────────────────────
# The keyboard stream reports HID usages (0..255, modifiers E0..E7); Axiom's
# hotkeys are Windows VK codes. A VK maps to every usage that should count as
# it (the generic Shift/Ctrl/Alt VKs match either side).

def _build_vk_to_hid() -> Dict[int, Tuple[int, ...]]:
    m: Dict[int, Tuple[int, ...]] = {}
    for i in range(26):                     # A..Z
        m[0x41 + i] = (0x04 + i,)
    for i in range(9):                      # 1..9
        m[0x31 + i] = (0x1E + i,)
    m[0x30] = (0x27,)                       # 0
    for i in range(12):                     # F1..F12
        m[0x70 + i] = (0x3A + i,)
    for i in range(12):                     # F13..F24
        m[0x7C + i] = (0x68 + i,)
    for i in range(9):                      # Numpad 1..9
        m[0x61 + i] = (0x59 + i,)
    m.update({
        0x60: (0x62,), 0x6E: (0x63,), 0x6F: (0x54,), 0x6A: (0x55,), 0x6D: (0x56,), 0x6B: (0x57,),
        0x0D: (0x28, 0x58), 0x1B: (0x29,), 0x08: (0x2A,), 0x09: (0x2B,), 0x20: (0x2C,),
        0xBD: (0x2D,), 0xBB: (0x2E,), 0xDB: (0x2F,), 0xDD: (0x30,), 0xDC: (0x31,),
        0xBA: (0x33,), 0xDE: (0x34,), 0xC0: (0x35,), 0xBC: (0x36,), 0xBE: (0x37,), 0xBF: (0x38,),
        0x14: (0x39,), 0x2C: (0x46,), 0x91: (0x47,), 0x13: (0x48,), 0x2D: (0x49,), 0x24: (0x4A,),
        0x21: (0x4B,), 0x2E: (0x4C,), 0x23: (0x4D,), 0x22: (0x4E,), 0x27: (0x4F,), 0x25: (0x50,),
        0x28: (0x51,), 0x26: (0x52,), 0x90: (0x53,), 0x5D: (0x65,),
        0xA2: (0xE0,), 0xA0: (0xE1,), 0xA4: (0xE2,), 0x5B: (0xE3,),
        0xA3: (0xE4,), 0xA1: (0xE5,), 0xA5: (0xE6,), 0x5C: (0xE7,),
        0x11: (0xE0, 0xE4), 0x10: (0xE1, 0xE5), 0x12: (0xE2, 0xE6),
    })
    return m


VK_TO_HID: Dict[int, Tuple[int, ...]] = _build_vk_to_hid()

# Mouse VK -> stream button bit (L, R, M, X1=side1, X2=side2).
MOUSE_VK_TO_BIT: Dict[int, int] = {0x01: 0x01, 0x02: 0x02, 0x04: 0x04, 0x05: 0x08, 0x06: 0x10}


# ── Plaintext UDP transport ──────────────────────────────────────────────────

class UdpSerialAdapter:
    """The subset of pyserial's Serial interface MakcuMouse uses, over
    plaintext UDP (MAK_API.md: "COM and plaintext UDP carry the complete
    frame"). Datagrams are appended to one receive buffer, so the existing
    frame parser consumes them exactly like serial bytes."""

    def __init__(self, host: str, port: int, recv_size: int = 2048):
        self._addr = (host, int(port))
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setblocking(False)
        self._sock.connect(self._addr)
        self._recv_size = recv_size
        self._rx = bytearray()
        self._rx_lock = threading.Lock()
        self.is_open = True

    def _drain(self) -> None:
        while True:
            try:
                data = self._sock.recv(self._recv_size)
            except (BlockingIOError, InterruptedError):
                return
            except OSError:
                return  # ICMP port-unreachable etc. — nothing to read
            if not data:
                return
            with self._rx_lock:
                self._rx.extend(data)

    @property
    def in_waiting(self) -> int:
        self._drain()
        with self._rx_lock:
            return len(self._rx)

    def read(self, n: int = 1) -> bytes:
        self._drain()
        with self._rx_lock:
            data = bytes(self._rx[:n])
            del self._rx[:n]
        return data

    def write(self, data: bytes) -> int:
        self._sock.send(bytes(data))
        return len(data)

    def flush(self) -> None:
        pass

    def reset_input_buffer(self) -> None:
        self._drain()
        with self._rx_lock:
            self._rx.clear()

    def close(self) -> None:
        self.is_open = False
        try:
            self._sock.close()
        except OSError:
            pass
