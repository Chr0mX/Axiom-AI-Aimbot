"""Remote mouse-button state over UDP (sent by vigem-for-a-game).

Datagram (12 bytes, little-endian): "AXMB", version=1, button mask, uint16 seq,
4-byte SHA-256 token tag (zeros when no token). Mask bits match
``MOUSE_VK_TO_BIT``: 0 LMB, 1 RMB, 2 MMB, 3 side1, 4 side2.

A sender that stops sending (crash, unplug) must never leave aim engaged, so
the state reads as "nothing held" after ``STALE_AFTER_S`` without a packet.
"""

from __future__ import annotations

import hashlib
import logging
import socket
import struct
import threading
import time
from typing import Callable, Optional, Tuple

logger = logging.getLogger(__name__)

MAGIC = b"AXMB"
VERSION = 1
PACKET_LEN = 12
STALE_AFTER_S = 0.5

BUTTON_BITS = {"lmb": 0x01, "rmb": 0x02, "mmb": 0x04, "side1": 0x08, "side2": 0x10}

_STRUCT = struct.Struct("<4sBBH4s")


def token_tag(token: str) -> bytes:
    if not token:
        return b"\x00\x00\x00\x00"
    return hashlib.sha256(token.encode("utf-8")).digest()[:4]


def parse_packet(data: bytes, token: str = "") -> Optional[Tuple[int, int]]:
    """Return (seq, mask), or None if the datagram is not valid for this token."""
    if len(data) != PACKET_LEN:
        return None
    magic, version, mask, seq, tag = _STRUCT.unpack(data)
    if magic != MAGIC or version != VERSION:
        return None
    if tag != token_tag(token):
        return None
    return seq, mask & 0x1F


def _seq_is_newer(seq: int, last: int) -> bool:
    return seq != last and ((seq - last) & 0xFFFF) < 0x8000


class RemoteButtonListener:
    def __init__(self, bind_ip: str, bind_port: int, token: str = "",
                 clock: Callable[[], float] = time.monotonic):
        self._token = token
        self._clock = clock
        self._lock = threading.Lock()
        self._mask = 0
        self._last_seq: Optional[int] = None
        self._last_rx = float("-inf")
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._sock.settimeout(0.5)
            self._sock.bind((bind_ip, int(bind_port)))
        except Exception:
            self._sock.close()
            raise
        self.port = self._sock.getsockname()[1]

    def feed(self, data: bytes) -> bool:
        parsed = parse_packet(data, self._token)
        if parsed is None:
            return False
        seq, mask = parsed
        with self._lock:
            now = self._clock()
            stale = (now - self._last_rx) > STALE_AFTER_S
            if self._last_seq is not None and not stale and not _seq_is_newer(seq, self._last_seq):
                return False
            self._last_seq = seq
            self._mask = mask
            self._last_rx = now
        return True

    def mask(self) -> int:
        with self._lock:
            if (self._clock() - self._last_rx) > STALE_AFTER_S:
                return 0
            return self._mask

    def held(self, name: str) -> bool:
        bit = BUTTON_BITS.get(name)
        if bit is None:
            return False
        return bool(self.mask() & bit)

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="RemoteButtons", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        try:
            self._sock.close()
        except OSError:
            pass
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _loop(self) -> None:
        while self._running:
            try:
                data, _ = self._sock.recvfrom(64)
            except socket.timeout:
                continue
            except OSError:
                break
            self.feed(data)


_listener: Optional[RemoteButtonListener] = None
_signature: Optional[tuple] = None
_state_lock = threading.Lock()


def ensure_remote_buttons(config) -> None:
    """Start/stop/rebuild the listener so it matches the current config."""
    global _listener, _signature
    with _state_lock:
        enabled = bool(getattr(config, "remote_btn_enabled", False))
        if not enabled:
            _stop_locked()
            return
        sig = (str(getattr(config, "remote_btn_bind_ip", "0.0.0.0")),
               int(getattr(config, "remote_btn_bind_port", 5602)),
               str(getattr(config, "remote_btn_token", "")))
        if _listener is not None and _signature == sig:
            return
        _stop_locked()
        try:
            _listener = RemoteButtonListener(sig[0], sig[1], sig[2])
            _listener.start()
            _signature = sig
            logger.info("[RemoteButtons] listening on %s:%d", sig[0], sig[1])
        except Exception as exc:
            logger.error("[RemoteButtons] failed to listen on %s:%d: %s", sig[0], sig[1], exc)
            _listener = None
            _signature = None


def _stop_locked() -> None:
    global _listener, _signature
    if _listener is not None:
        try:
            _listener.stop()
        except Exception:
            pass
    _listener = None
    _signature = None


def stop_remote_buttons() -> None:
    with _state_lock:
        _stop_locked()


def remote_button_held(name: str) -> bool:
    lst = _listener
    return lst is not None and lst.held(name)
