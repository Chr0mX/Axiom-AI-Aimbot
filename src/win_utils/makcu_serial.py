from __future__ import annotations

import collections
import logging
import queue
import threading
import time
from typing import Optional

import serial  # type: ignore[import-not-found]

from core.ndi_config_loader import PipelineConfig

log = logging.getLogger(__name__)

# Binary frame that upgrades baud rate to 4,000,000.
# Decoded: header=0xDEAD, len=5 LE, cmd=0xA5, payload=4000000 as u32 LE
_BAUD_UPGRADE_FRAME = bytearray([0xDE, 0xAD, 0x05, 0x00, 0xA5, 0x00, 0x09, 0x3D, 0x00])

# ASCII commands (reused verbatim from src/win_utils/makcu_mouse.py)
_CMD_MOVE = "km.move({dx},{dy})\r\n"
_CMD_ECHO_OFF = "km.echo(0)\r\n"
_CMD_VERSION = "km.version()\r\n"
_CMD_LEFT_DOWN = "km.left(1)\r\n"
_CMD_LEFT_UP = "km.left(0)\r\n"

_QUEUE_MAX = 4
_RECONNECT_BACKOFF = 2.0
_RECONNECT_MAX_ATTEMPTS = 5


class MakcuSerial:
    """MAKCU serial output with two-phase baud upgrade and background writer.

    Connect sequence (performed on every connect/reconnect — upgrade is not
    persistent across MAKCU power cycles):
      Phase 1: open at 115200 → send binary baud upgrade frame → close
      Phase 2: reopen at 4,000,000 → verify km.version response → echo off

    move() enqueues a km.move command; the background writer thread drains the
    queue.  On queue full the oldest item is dropped to avoid stale moves.
    """

    def __init__(self, cfg: PipelineConfig) -> None:
        self._com_port = cfg.com_port
        self._baud_initial = cfg.baud_rate_initial
        self._baud_target = cfg.baud_rate_target

        self._serial: Optional[serial.Serial] = None
        self._lock = threading.Lock()
        self._connected = False

        self._send_queue: queue.Queue[bytes] = queue.Queue(maxsize=_QUEUE_MAX)
        self._stop_event = threading.Event()
        self._writer_thread = threading.Thread(
            target=self._writer_worker, name="MakcuWriter", daemon=True
        )
        self._writer_thread.start()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        """Perform two-phase baud upgrade and verify device handshake."""
        try:
            return self._phase1_upgrade() and self._phase2_open()
        except Exception as exc:
            log.error("[MAKCU] Connect failed: %s", exc)
            self._connected = False
            return False

    def move(self, dx: int, dy: int) -> None:
        """Enqueue a relative mouse move command."""
        if not self._connected:
            return
        dx = max(-32768, min(32767, int(dx)))
        dy = max(-32768, min(32767, int(dy)))
        cmd = _CMD_MOVE.format(dx=dx, dy=dy).encode("ascii")
        self._enqueue(cmd)

    def click_left(self) -> None:
        """Enqueue a left button press + release."""
        if not self._connected:
            return
        self._enqueue(_CMD_LEFT_DOWN.encode("ascii"))
        self._enqueue(_CMD_LEFT_UP.encode("ascii"))

    def is_connected(self) -> bool:
        return self._connected and self._serial is not None and self._serial.is_open

    def reconnect(self) -> bool:
        """Re-run the full two-phase connect sequence. Blocks up to ~10s."""
        log.info("[MAKCU] Attempting reconnect on %s...", self._com_port)
        for attempt in range(1, _RECONNECT_MAX_ATTEMPTS + 1):
            if self.connect():
                log.info("[MAKCU] Reconnected on attempt %d", attempt)
                return True
            log.warning("[MAKCU] Reconnect attempt %d/%d failed", attempt, _RECONNECT_MAX_ATTEMPTS)
            time.sleep(_RECONNECT_BACKOFF)
        log.error("[MAKCU] Reconnect failed after %d attempts", _RECONNECT_MAX_ATTEMPTS)
        return False

    def disconnect(self) -> None:
        self._stop_event.set()
        with self._lock:
            if self._serial and self._serial.is_open:
                try:
                    self._serial.close()
                except Exception:
                    pass
        self._connected = False
        log.info("[MAKCU] Disconnected")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _phase1_upgrade(self) -> bool:
        """Open at 115200, send binary baud upgrade, close."""
        try:
            with serial.Serial(self._com_port, self._baud_initial, timeout=0.5) as s:
                s.reset_input_buffer()
                s.write(_BAUD_UPGRADE_FRAME)
                s.flush()
            time.sleep(0.1)  # allow device to switch baud rate
            log.debug("[MAKCU] Phase 1: baud upgrade frame sent")
            return True
        except serial.SerialException as exc:
            log.error("[MAKCU] Phase 1 failed: %s", exc)
            return False

    def _phase2_open(self) -> bool:
        """Open at 4 Mbaud, verify version response, disable echo."""
        try:
            s = serial.Serial(
                self._com_port,
                self._baud_target,
                timeout=0.5,
                write_timeout=0.1,
            )
            time.sleep(0.05)
            s.reset_input_buffer()

            # Handshake
            s.write(_CMD_VERSION.encode("ascii"))
            s.flush()
            time.sleep(0.1)
            response = s.read(s.in_waiting).decode("ascii", errors="ignore")
            if "km.MAKCU" not in response and "MAKCU" not in response.upper():
                log.warning("[MAKCU] Unexpected version response: %r", response)
                # Continue anyway — some firmware variants respond differently

            # Suppress ACKs to reduce inbound serial traffic
            s.write(_CMD_ECHO_OFF.encode("ascii"))
            s.flush()
            time.sleep(0.05)
            s.reset_input_buffer()

            with self._lock:
                if self._serial and self._serial.is_open:
                    try:
                        self._serial.close()
                    except Exception:
                        pass
                self._serial = s
                self._connected = True

            log.info("[MAKCU] Phase 2: connected at %d baud on %s", self._baud_target, self._com_port)
            return True
        except serial.SerialException as exc:
            log.error("[MAKCU] Phase 2 failed: %s", exc)
            self._connected = False
            return False

    def _enqueue(self, data: bytes) -> None:
        """Put data on the send queue; drop oldest item if full."""
        try:
            self._send_queue.put_nowait(data)
        except queue.Full:
            try:
                self._send_queue.get_nowait()  # discard oldest stale command
            except queue.Empty:
                pass
            try:
                self._send_queue.put_nowait(data)
            except queue.Full:
                pass

    def _writer_worker(self) -> None:
        while not self._stop_event.is_set():
            try:
                data = self._send_queue.get(timeout=0.005)
            except queue.Empty:
                continue
            try:
                with self._lock:
                    if self._serial and self._serial.is_open:
                        self._serial.write(data)
            except serial.SerialException:
                log.warning("[MAKCU] Serial write failed — marking disconnected")
                self._connected = False
            except Exception as exc:
                log.debug("[MAKCU] Writer error: %s", exc)
