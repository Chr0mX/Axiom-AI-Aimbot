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


import serial
import serial.tools.list_ports

from . import makcu_protocol as mp

logger = logging.getLogger(__name__)

# The physical-movement lock (set_physical_move_lock) auto-releases unless the
# caller re-asserts it within this window, so a stalled/crashed aim loop can
# never leave the user's own mouse blocked.
_MOVE_LOCK_REFRESH_S = 0.3
_HEALTH_CHECK_INTERVAL_S = 4.0
_MAK_REQUEST_TIMEOUT_S = 0.3

# Official 4 Mbaud connection constants
_OPERATING_BAUD    = 4_000_000
_BAUD_CHANGE_FRAME = bytes([0xDE, 0xAD, 0x05, 0x00, 0xA5, 0x00, 0x09, 0x3D, 0x00])

# Button stream parsing — see _stream_reader() for the frame format.
_BTN_BITS = 0x1F  # bits 0-4 = L,R,M,S1,S2 — everything else is noise/high-byte

# Unparsed bytes kept between reads. Only ever a partial frame or marker-less
# noise is left over after parsing (a binary frame is at most 69 bytes), so
# this never has to discard a complete, not-yet-applied event.
_STREAM_MAX_LEFTOVER = 256


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
        # Set by _query_info() to make _stream_reader() briefly stand down
        # from calling ser.read() itself, so the two never race for the
        # same bytes on the wire — see _query_info()'s own docstring for
        # why this is a *read pause*, not a stop/restart of the device-side
        # stream (km.buttons(0)/(1)) the way an earlier version of this fix
        # did. _btn_mask is never touched by this — the whole point is that
        # pausing must not lose track of an already-held button.
        self._stream_read_pause = threading.Event()
        # Bytes _query_info() read off the wire while the stream reader was
        # paused. Button events that arrived during the query are in here and
        # must still be parsed — dropping them is how a Mouse1 release got
        # lost and left aim engaged.
        self._stream_carry = bytearray()
        self._stream_carry_lock = threading.Lock()
        # Frame-format detection + startup debug logging — instance-level
        # (not local to _stream_reader()) so both persist across the
        # stream's internal pause/resume cycles (see _query_info()), and
        # only reset on a genuine new connect(). Resetting them on every
        # _stream_reader() call (the original design) meant every ~3s
        # hardware-info refresh — which now briefly stops/restarts the
        # stream to avoid racing with it — silently re-armed both the
        # "first 20 chunks" debug dump and the one-time format-detection
        # log line, spamming the console indefinitely for the life of the
        # connection instead of once at actual connect time.
        self._stream_frame_len: Optional[int] = None
        self._stream_logged_chunks: int = 0
        # A third device/firmware family streams button events as a
        # compact binary report (0xDE 0xAD sync + length-prefixed payload)
        # instead of either ASCII "km." form — see _stream_reader()'s own
        # docstring. Locked in once per connection alongside the two
        # ASCII-format fields above, never re-checked mid-connection.
        self._stream_binary_mode: bool = False

        # Async write thread — inference thread hands off the latest move;
        # write thread drains it and flushes to serial so inference is never
        # blocked. Pending relative movement, SET (not accumulated) by move()
        # and drained by _write_worker: every move() call already carries the
        # complete, freshly-recomputed correction for the current frame (PID
        # + humanization + sub-pixel carry + jitter are all folded in before
        # ai_aiming.py ever calls send_mouse_move()), so a still-pending
        # value here is by definition superseded, not merely "not yet sent
        # on top of". Summing them (an earlier version of this code did)
        # double- and triple-counts the same not-yet-visually-applied error:
        # the aim loop's detect_interval is routinely faster than the real
        # USB-injection + game-frame + capture round trip, so several
        # consecutive PID cycles recompute a full correction against a
        # target that hasn't moved on screen yet, and summing all of them
        # into one write overshoots by however many cycles piled up —
        # exactly the systemic Y-axis overshoot ("aiming snaps past the
        # target") this replaced. Guarded by _pending_lock; _pending_event
        # signals the writer that there is something to send.
        self._pending_dx: int = 0
        self._pending_dy: int = 0
        self._pending_lock = threading.Lock()
        self._pending_event = threading.Event()
        self._write_stop = threading.Event()
        self._write_thread: Optional[threading.Thread] = None

        # Reconnect watchdog — re-establishes connection after USB glitches.
        self._reconnect_thread: Optional[threading.Thread] = None

        # MAK_API (MAKCU V4 / MAKXD) — detected per connection via km.device()
        # (V4 answers "R:..;M:..uf;..", V3 "mouse"/"keyboard"/"none"), or
        # implied by a UDP connection. When set: binary command frames, the
        # binary event stream, replies demultiplexed by _stream_reader().
        self._mak_api: bool = False
        self._transport: str = "serial"
        self._udp_target: Optional[tuple] = None
        self._firmware_version: Optional[int] = None
        self._device_kinds: Optional[int] = None
        self._km_device: dict = {}
        self._reply_lock = threading.Lock()
        self._reply_waiters: dict = {}   # cmd -> list[[Event, payload|None]]
        self._settings_lock = threading.Lock()
        self._last_health_check = 0.0

        # Keyboard input stream (HID usages currently held), V4/MAKXD only.
        self._keys_down: set = set()
        self._keyboard_stream_wanted = False

        # Physical-movement lock state (see set_physical_move_lock).
        self._move_lock_active = False
        self._move_lock_asserted_at = 0.0

        # Firmware mouse spread the caller wants applied (None = leave the
        # device's own setting alone); see request_mouse_spread().
        self._spread_desired: Optional[int] = None
        self._spread_applied: Optional[int] = None
        self._spread_thread: Optional[threading.Thread] = None

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

        The lock is only ever held around the actual state mutations
        (opening the port, writing bytes, flipping _connected/_com_port) —
        never across a time.sleep() — so move()/click() from the inference
        thread are never blocked for the ~2s this handshake can take.
        """
        # Fresh attempt: clear any stale stop signal left over from a prior
        # disconnect() so this attempt isn't immediately treated as
        # mid-disconnect. If disconnect() sets it again while we're running,
        # that's a genuine cancel request for *this* attempt (checked below).
        self._write_stop.clear()

        with self._lock:
            self._close_locked()

        if self._write_stop.is_set():
            return False

        # Step 1: device may already be at 4M from a previous session
        if self._try_open(com_port, _OPERATING_BAUD):
            with self._lock:
                self._connected = True
                self._com_port = com_port
                self._transport = "serial"
            self._identify_device()
            logger.info("[MAKCU] Connected to %s @ %d baud", com_port, _OPERATING_BAUD)
        else:
            if self._write_stop.is_set():
                return False

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

            if self._write_stop.is_set():
                return False
            time.sleep(0.05)

            if self._write_stop.is_set():
                return False
            if not self._try_open(com_port, _OPERATING_BAUD):
                logger.error("[MAKCU] Could not connect to %s after baud change", com_port)
                return False

            with self._lock:
                self._connected = True
                self._com_port = com_port
                self._transport = "serial"
            self._identify_device()
            logger.info("[MAKCU] Connected to %s @ %d baud", com_port, _OPERATING_BAUD)

        return self._finish_connect()

    def connect_udp(self, host: str, port: int = 8080) -> bool:
        """Connect over plaintext UDP (MAKXD Ethernet/Wi-Fi). MAK_API only —
        the device must answer FIRMWARE_VERSION to count as connected.
        Encrypted UDP and the raw-UDP transaction header are not supported."""
        self._write_stop.clear()
        with self._lock:
            self._close_locked()
        try:
            adapter = mp.UdpSerialAdapter(host, int(port))
        except OSError as exc:
            logger.error("[MAKCU] UDP socket to %s:%s failed: %s", host, port, exc)
            return False
        with self._lock:
            self._serial = adapter
        self._mak_api = True
        version = self._mak_request(mp.CMD_FIRMWARE_VERSION, timeout=1.0)
        if version is None or mp.is_rejection(version) or len(version) < 4:
            logger.error("[MAKCU] No MAK_API reply from %s:%s over UDP", host, port)
            with self._lock:
                self._close_locked()
            self._mak_api = False
            return False
        with self._lock:
            self._connected = True
            self._transport = "udp"
            self._udp_target = (host, int(port))
            self._com_port = f"udp://{host}:{port}"
        self._version_string = "MAK_API (UDP)"
        self._load_mak_api_info(version)
        logger.info("[MAKCU] Connected over UDP to %s:%s", host, port)
        return self._finish_connect()

    def _finish_connect(self) -> bool:
        if self._write_stop.is_set():
            # disconnect() requested mid-flight — tear back down instead of
            # leaving a live connection behind after disconnect() already ran.
            with self._lock:
                self._close_locked()
            return False

        # Discard any move left pending from before this connection came up
        # — it describes a target position from a session that's over, and
        # sending it now would fire a stale jump for no reason.
        with self._pending_lock:
            self._pending_dx = 0
            self._pending_dy = 0
            self._pending_event.clear()

        # Fresh connection: re-arm the stream's one-time format detection
        # and startup debug logging. _query_info() also stops/restarts the
        # stream (to avoid racing with it — see its own docstring) but
        # deliberately does NOT touch these, so they stay armed for exactly
        # one real connect(), not once per internal pause/resume cycle.
        self._stream_frame_len = None
        self._stream_logged_chunks = 0
        # A MAK_API device is known to speak the binary event stream, so skip
        # the first-marker guess (its own km.* text replies could win it).
        self._stream_binary_mode = self._mak_api
        self._keys_down.clear()
        self._move_lock_active = False

        # Start threads outside the lock
        self._start_stream()
        self._start_write_thread()
        self._start_reconnect_thread()
        self._spread_applied = None
        if self._spread_desired is not None:
            self._kick_spread_worker()
        return True

    def _identify_device(self) -> None:
        """Tell MAKCU V4/MAKXD (MAK_API) from V3 using `km.device()`, which
        both accept: V4 replies `R:<routes>;M:<n>uf;...`, V3 `mouse` /
        `keyboard` / `none`. Runs before the stream starts. V3 then gets the
        classic km.info() query; V4 has no km.info() and is read via
        FIRMWARE_VERSION/DEVICE instead."""
        self._mak_api = False
        self._firmware_version = None
        self._device_kinds = None
        self._km_device = {}
        text = self._ascii_query("km.device()\r\n")
        parsed = mp.parse_km_device(text)
        if parsed is None:
            self._query_info()
            return
        self._km_device = parsed
        self._mak_api = True
        version = self._mak_request(mp.CMD_FIRMWARE_VERSION)
        self._load_mak_api_info(version)
        logger.info("[MAKCU] MAK_API device detected (routes=%s, firmware=%s)",
                    parsed.get("R"), self._firmware_version)

    def _load_mak_api_info(self, version_payload: Optional[bytes]) -> None:
        if version_payload and not mp.is_rejection(version_payload) and len(version_payload) >= 4:
            self._firmware_version = int.from_bytes(version_payload[:4], "little")
        kinds = self._mak_request(mp.CMD_DEVICE)
        if kinds and not mp.is_rejection(kinds):
            self._device_kinds = kinds[0]
        info = {}
        if self._firmware_version is not None:
            info["FIRMWARE"] = str(self._firmware_version)
        if self._device_kinds is not None:
            info["MODEL"] = mp.describe_device_kinds(self._device_kinds)
        routes = self._km_device.get("R")
        if routes:
            info["ROUTES"] = routes
        hz = mp.usb_period_to_hz(self._km_device.get("M", ""))
        if hz:
            info["MOUSE_POLL_HZ"] = f"{hz:.0f}"
        self._device_info = info

    def _ascii_query(self, command: str, wait_s: float = 0.15) -> str:
        """Pre-stream ASCII query; returns whatever text came back."""
        try:
            with self._lock:
                if not self._serial:
                    return ""
                self._serial.reset_input_buffer()
                self._serial.write(command.encode("ascii"))
                self._serial.flush()
            time.sleep(wait_s)
            with self._lock:
                if not self._serial:
                    return ""
                return self._serial.read(self._serial.in_waiting).decode("ascii", errors="ignore")
        except Exception:
            return ""

    # ------------------------------------------------------------------
    # MAK_API requests (V4 / MAKXD)
    # ------------------------------------------------------------------

    def _mak_request(self, cmd: int, payload: bytes = b"",
                     timeout: float = _MAK_REQUEST_TIMEOUT_S) -> Optional[bytes]:
        """Send a MAK_API GET (or a management transaction that replies) and
        return the reply payload, or None on timeout/not connected.

        With the stream reader running, the reply is demultiplexed by it and
        handed over through a waiter — the reader stays the only thing that
        reads the port, so no button event is ever consumed here. Before the
        stream starts (connect-time), the reply is read directly."""
        stream_running = self._stream_thread is not None and self._stream_thread.is_alive()
        if not stream_running:
            return self._mak_request_direct(cmd, payload, timeout)
        waiter = [threading.Event(), None]
        with self._reply_lock:
            self._reply_waiters.setdefault(cmd, []).append(waiter)
        if not self._write_frame(mp.frame(cmd, payload)):
            with self._reply_lock:
                if waiter in self._reply_waiters.get(cmd, []):
                    self._reply_waiters[cmd].remove(waiter)
            return None
        if waiter[0].wait(timeout):
            return waiter[1]
        with self._reply_lock:
            if waiter in self._reply_waiters.get(cmd, []):
                self._reply_waiters[cmd].remove(waiter)
        return None

    def _mak_request_direct(self, cmd: int, payload: bytes, timeout: float) -> Optional[bytes]:
        try:
            with self._lock:
                ser = self._serial
                if not ser or not ser.is_open:
                    return None
                ser.reset_input_buffer()
                ser.write(mp.frame(cmd, payload))
                ser.flush()
            buf = bytearray()
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                with self._lock:
                    n = ser.in_waiting
                    if n:
                        buf.extend(ser.read(n))
                while True:
                    idx = buf.find(mp.SYNC)
                    if idx == -1 or idx + 5 > len(buf):
                        break
                    length = buf[idx + 2] | (buf[idx + 3] << 8)
                    end = idx + 5 + length
                    if length > 1024:
                        del buf[:idx + 2]
                        continue
                    if end > len(buf):
                        break
                    if buf[idx + 4] == cmd:
                        return bytes(buf[idx + 5:end])
                    del buf[:end]
                if not n:
                    time.sleep(0.002)
        except Exception as exc:
            logger.debug("[MAKCU] MAK_API request 0x%02X failed: %s", cmd, exc)
        return None

    def _dispatch_reply(self, cmd: int, payload: bytes) -> None:
        if cmd == mp.CMD_CONNECTION and payload[:1] == bytes([mp.TOPOLOGY_NOTICE]):
            return  # asynchronous topology notice, not a reply
        with self._reply_lock:
            waiters = self._reply_waiters.get(cmd)
            waiter = waiters.pop(0) if waiters else None
        if waiter is not None:
            waiter[1] = payload
            waiter[0].set()

    def _write_frame(self, data: bytes) -> bool:
        try:
            with self._lock:
                if self._serial and self._serial.is_open:
                    self._serial.write(data)
                    self._serial.flush()
                    return True
        except serial.SerialException:
            self._connected = False
        except Exception as exc:
            logger.debug("[MAKCU] write failed: %s", exc)
        return False

    def _try_open(self, com_port: str, baud: int) -> bool:
        """Open port at baud, probe with km.version(). Returns True if a
        known device identity string is found.

        Two device families are accepted: legacy **MAKCU** (`km.MAKCU`)
        and **MAKXD** (`km.MAKXD`, per the official terrafirma2021/
        mak-suite KM_API.md spec) — both speak the same km.* ASCII
        command surface this class relies on, differing only in identity
        string and (see _stream_reader()) button-stream framing.

        Acquires _lock only around the individual serial operations — never
        across a sleep — so it never blocks move()/click() for the duration
        of this handshake.
        """
        try:
            with self._lock:
                self._serial = serial.Serial(com_port, baud, timeout=0.3, write_timeout=0.1)
                ser = self._serial
            time.sleep(0.05)
            with self._lock:
                ser.reset_input_buffer()
                ser.write(self.CMD_VERSION.encode('ascii'))
                ser.flush()
            deadline = time.monotonic() + 0.5
            raw = b''
            while time.monotonic() < deadline:
                with self._lock:
                    waiting = ser.in_waiting
                    if waiting:
                        raw += ser.read(waiting)
                if b'km.MAKCU' in raw or b'km.MAKXD' in raw:
                    break
                if not waiting:
                    time.sleep(0.01)
            if b'km.MAKCU' not in raw and b'km.MAKXD' not in raw:
                with self._lock:
                    self._close_locked()
                return False
            self._version_string = raw.decode('ascii', errors='ignore').replace('>>>', '').strip()
            with self._lock:
                ser.write(self.CMD_ECHO_OFF.encode('ascii'))
                ser.flush()
            time.sleep(0.1)
            with self._lock:
                ser.reset_input_buffer()
            return True
        except serial.SerialException as exc:
            logger.debug("[MAKCU] _try_open %s@%d failed: %s", com_port, baud, exc)
            with self._lock:
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
        """Enable km.buttons(1) stream and start the reader thread.

        _query_info() (called earlier in connect()) reads only whatever's
        already waiting 0.15s after its km.info() write — a slow/late-
        arriving tail of that reply can still be sitting in the input
        buffer here. _stream_reader() trusts byte-aligned "km.<mask>"
        framing from the very first frame it sees, so any stray leftover
        bytes at this point desync it before it ever reads a real button
        frame — with echo off, nothing else will resync it until an
        unrelated event happens to line the framing back up. Reset the
        buffer right before enabling the stream so it always starts clean.
        """
        self._stream_stop.clear()
        self._stream_read_pause.clear()
        self._btn_mask = 0
        with self._stream_carry_lock:
            self._stream_carry.clear()
        try:
            with self._lock:
                if self._serial and self._serial.is_open:
                    self._serial.reset_input_buffer()
                    if self._mak_api:
                        self._serial.write(mp.stream_enable_frame(mp.STREAM_KIND_MOUSE, True))
                        if self._keyboard_stream_wanted:
                            self._serial.write(mp.stream_enable_frame(mp.STREAM_KIND_KEYBOARD, True))
                    else:
                        self._serial.write(b'km.buttons(1)\r\n')
                    self._serial.flush()
                else:
                    logger.warning("[MAKCU] _start_stream: serial not open, button stream not started")
                    return
        except Exception as exc:
            logger.warning("[MAKCU] _start_stream: failed to enable button stream: %s", exc)
            return
        self._stream_thread = threading.Thread(
            target=self._stream_reader, daemon=True, name="makcu-stream")
        self._stream_thread.start()

    def _set_btn_mask(self, mask: int, frame: bytes) -> None:
        mask &= _BTN_BITS
        old = self._btn_mask
        if mask != old:
            self._btn_mask = mask
            logger.debug("[MAKCU] button mask 0x%02X -> 0x%02X (L=%d R=%d) from %s",
                         old, mask, mask & 1, (mask >> 1) & 1, frame.hex(' '))

    def _reenable_button_stream(self) -> None:
        """Re-subscribe after a device-side overflow (see _stream_reader)."""
        data = mp.stream_enable_frame(mp.STREAM_KIND_MOUSE, True) if self._mak_api else b'km.buttons(1)\r\n'
        if not self._write_frame(data):
            logger.warning("[MAKCU] could not re-enable button stream after overflow")

    def _reenable_keyboard_stream(self) -> None:
        self._write_frame(mp.stream_enable_frame(mp.STREAM_KIND_KEYBOARD, True))

    def _stop_stream(self):
        """Stop the reader thread and send km.buttons(0)."""
        self._stream_stop.set()
        if self._stream_thread:
            self._stream_thread.join(timeout=1.0)
            self._stream_thread = None
        try:
            with self._lock:
                if self._serial and self._serial.is_open:
                    if self._mak_api:
                        self._serial.write(mp.stream_enable_frame(mp.STREAM_KIND_MOUSE, False))
                        if self._keyboard_stream_wanted:
                            self._serial.write(mp.stream_enable_frame(mp.STREAM_KIND_KEYBOARD, False))
                    else:
                        self._serial.write(b'km.buttons(0)\r\n')
                    self._serial.flush()
        except Exception:
            pass
        self._btn_mask = 0
        self._keys_down.clear()

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
        """Drain the pending relative movement and write+flush to serial.

        Runs on its own thread so the inference thread (which calls move())
        is never blocked waiting on the serial port.

        Takes the pending delta and zeroes it in one locked step, so a move()
        landing while this write is in flight goes into a freshly-zeroed slot
        rather than being merged into what's about to be sent. PID calls
        replace that slot; movement-path calls add into it. See move().
        """
        while not self._write_stop.is_set():
            got = self._pending_event.wait(0.01)
            if (self._move_lock_active
                    and time.monotonic() - self._move_lock_asserted_at > _MOVE_LOCK_REFRESH_S):
                self._send_move_lock(False)
            if not got:
                continue
            with self._pending_lock:
                dx, dy = self._pending_dx, self._pending_dy
                self._pending_dx = 0
                self._pending_dy = 0
                self._pending_event.clear()
            if dx == 0 and dy == 0:
                continue
            if self._mak_api:
                data = mp.move_frame(dx, dy)
            else:
                data = self.CMD_MOVE.format(dx=dx, dy=dy).encode('ascii')
            try:
                with self._lock:
                    if self._serial and self._serial.is_open:
                        self._serial.write(data)
                        self._serial.flush()
            except serial.SerialException:
                self._connected = False
            except Exception:
                pass

    def _reconnect_worker(self):
        """Watchdog: re-establish connection automatically after USB glitches,
        and (MAK_API) keep the input subscriptions alive."""
        while not self._write_stop.is_set():
            self._write_stop.wait(2.0)
            if self._write_stop.is_set():
                break
            if not self.is_connected() and self._com_port:
                logger.info("[MAKCU] Connection lost — reconnecting on %s", self._com_port)
                try:
                    if self._transport == "udp" and self._udp_target:
                        self.connect_udp(*self._udp_target)
                    else:
                        self.connect(self._com_port)
                except Exception as exc:
                    logger.debug("[MAKCU] Reconnect failed: %s", exc)
                continue
            now = time.monotonic()
            if self._mak_api and now - self._last_health_check >= _HEALTH_CHECK_INTERVAL_S:
                self._last_health_check = now
                self.check_stream_health()

    def check_stream_health(self) -> dict:
        """Query each subscription (MAK_API BUTTONS / KEY_KEYS GET) and
        re-enable any the device reports as off — e.g. after an overflow
        whose event was lost, or firmware disabling it. Queries don't move
        subscription ownership. Returns {kind: enabled_before_check}."""
        result = {}
        if not self._mak_api or not self.is_connected():
            return result
        checks = [("mouse", mp.CMD_BUTTONS, mp.STREAM_KIND_MOUSE)]
        if self._keyboard_stream_wanted:
            checks.append(("keyboard", mp.CMD_KEY_KEYS, mp.STREAM_KIND_KEYBOARD))
        for name, cmd, kind in checks:
            reply = self._mak_request(cmd)
            if reply is None or mp.is_rejection(reply) or not reply:
                continue
            enabled = bool(reply[0])
            result[name] = enabled
            if not enabled:
                logger.warning("[MAKCU] %s input stream was disabled on the device — re-enabling", name)
                if kind == mp.STREAM_KIND_MOUSE:
                    self._set_btn_mask(0, b"health-check")
                else:
                    self._keys_down.clear()
                self._write_frame(mp.stream_enable_frame(kind, True))
        return result

    def _stream_reader(self):
        """Daemon thread: parse km.buttons(1) event frames, update _btn_mask.

        Three device/firmware families are supported. The first two share
        the same `km.` + 1-byte mask event but differ in how it's framed
        on the wire; the third is a genuinely different, binary encoding
        seen on a newer firmware revision that stopped speaking the ASCII
        buttons(1) stream at all:

        - **Legacy MAKCU** — confirmed against a raw hex capture from real
          hardware (this codebase's prior three attempts at this format
          were each tried and reported as not detecting real clicks — see
          git history on this method for what didn't work and why each
          guess seemed plausible at the time). The actual frame, captured
          verbatim:

              6b 6d 2e 01 0d 0a 3e 3e 3e 20
              "k  m  .  <mask=0x01>  \\r  \\n  >  >  >  ' '"

          i.e. `km.` + mask + the exact same `\\r\\n>>> ` suffix the docs
          describe for one-off ASCII command replies — the buttons stream
          just pushes this unsolicited on every state change, using the
          same reply framing as everything else instead of a distinct
          compact encoding. 10 bytes total.

        - **MAKXD** — per the official terrafirma2021/mak-suite KM_API.md
          spec: "each change in the physical five-button mask emits
          exactly four unframed bytes: `km.` + mask ... Events have no
          CR/LF or prompt." 4 bytes total, nothing to verify.

        Both share the same 5-bit physical mask layout (bit0=L, bit1=R,
        bit2=M, bit3=S1, bit4=S2), so once framing is known, extracting the
        mask is identical — only how many trailing bytes belong to "this
        frame" differs.

        - **Binary** — a newer firmware revision reported to no longer
          emit either ASCII form at all (clicks stopped registering
          entirely — the "km." prefix this parser looks for never
          appears anywhere on the wire). Reverse-engineered from a raw
          capture of two button presses (mask bit 0 and bit 1), each a
          press immediately followed by a release:

              de ad 03 00 53 01 00 01   de ad 03 00 53 01 00 00
              de ad 03 00 53 01 01 01   de ad 03 00 53 01 01 00

          i.e. a 2-byte `0xDE 0xAD` sync marker, a little-endian 2-byte
          length field (here always `3`, counting only the bytes after
          the next field), a 1-byte report-type tag (`0x53`, observed
          only for button events so far), then a 3-byte payload:
          `[0x01 sub-type][button bit-index][state: 1=press, 0=release]`.
          The button-index byte matches this codebase's own existing bit
          numbering (0=L, 1=R, 2=M, 3=S1, 4=S2) exactly — a bit is set/
          cleared in `_btn_mask` directly from it — which is a reasonable
          but *not independently confirmed* inference (see the length-
          driven framing below): the mapping was only ever observed for
          bits 0 and 1 (Left/Right) on real hardware; Middle/Side1/Side2
          are extrapolated from the existing convention, not observed.
          The length field is honored genuinely (not hardcoded to 8
          bytes) so a differently-sized report of some other kind doesn't
          desync this parser — an unrecognized tag byte is simply
          skipped, never misread as a button change. An implausibly large
          length (more than 128 — the largest real frame, a live-settings
          READ reply, is 105) is treated as a corrupted/coincidental sync match and
          skipped past, resyncing on the next `0xDE 0xAD` occurrence,
          mirroring the ASCII legacy format's own resync-on-mismatch
          safety net above.

        Framing is detected **once per connection**, not re-checked per
        frame (so a real button press never waits on this). The very
        first decision is binary vs. ASCII: whichever marker — `km.` or
        `0xDE 0xAD` — is found earliest in the buffer wins (a real device
        only ever speaks one of the two; there's no scenario where both
        legitimately appear). Once binary mode is locked in, parsing
        switches entirely to the length-driven binary path described
        above for the rest of the connection. Otherwise, for the two
        ASCII sub-formats: the first time a "km." prefix is seen, the byte
        immediately following the mask is inspected — `\\r` (0x0D) means
        the legacy 10-byte suffix is coming, anything else (including a
        short bounded wait timing out with nothing further arriving)
        means the unframed 4-byte MAKXD form. A real legacy device can
        never have a non-CR byte there, and a real MAKXD device can never
        legitimately produce a stray CR there either (the byte right
        after its mask is always either the start of the next "km." event
        or nothing yet), so this one byte is unambiguous once it's
        available. Even a wrong first guess self-heals: the resync-on-
        next-"km." logic below discards whatever trailing bytes don't
        belong to the mistaken frame length before the next real frame is
        parsed.

        For legacy frames, the trailing suffix is verified (not just the
        "km." prefix) once enough bytes are buffered — a real mask byte
        can't corrupt into "km.", but requiring the suffix as well catches
        a byte misaligning the frame from either direction and forces a
        resync on the next real "km." instead of misreading a corrupted
        mask. MAKXD's unframed form has no such trailing check to make —
        that's an inherent limitation of the documented protocol itself,
        not something this reader can add.

        "Once per connection" is tracked on `self` (`_stream_frame_len`),
        not as a local variable here, and likewise for the "log the first
        20 raw chunks" debug budget (`_stream_logged_chunks`) — both are
        reset only by connect(), not by this method starting. This method
        itself gets called again on every internal pause/resume cycle
        (see _query_info()'s own docstring for why that happens periodically
        while connected), and a local variable would silently re-arm both
        of these every time, which was a real, reported bug: the console
        got spammed with a repeating "frame format detected"/20-chunk dump
        every ~3s for as long as MAKCU stayed connected, instead of once.

        km.echo(0) means the device sends nothing in response to move/click writes,
        so all incoming bytes are button stream events — no lock needed on reads.
        """
        buf = bytearray()
        _KM_PREFIX = b"km."
        _SUFFIX = b"\r\n>>> "
        _LEGACY_FRAME_LEN = len(_KM_PREFIX) + 1 + len(_SUFFIX)  # km. + mask + \r\n>>>(space) = 10
        _SHORT_FRAME_LEN = len(_KM_PREFIX) + 1  # km. + mask = 4 (MAKXD, unframed)
        _DETECT_GRACE_S = 0.05
        _BIN_SYNC = b"\xde\xad"
        _BIN_HDR_LEN = len(_BIN_SYNC) + 2  # sync + 2-byte little-endian length field = 4
        _BIN_BTN_TAG = 0x53  # observed report-type byte for a button-state event
        # Guards against a coincidental sync match in noise. The largest real
        # frame is a live-settings READ reply: [1D op status] + revision:u32 +
        # offset:u16 + 96 data bytes = 105.
        _BIN_MAX_PLAUSIBLE_LEN = 128
        while not self._stream_stop.is_set():
            try:
                if self._stream_read_pause.is_set():
                    # _query_info() is doing its own exclusive read right
                    # now — stand down without touching _btn_mask or the
                    # buffer, so a still-held button stays reported as held
                    # for the whole pause instead of appearing released.
                    time.sleep(0.005)
                    continue
                ser = self._serial
                if not ser or not ser.is_open:
                    break
                with self._stream_carry_lock:
                    carry = bytes(self._stream_carry)
                    self._stream_carry.clear()
                n = ser.in_waiting
                if n or carry:
                    # Carry first: it was on the wire before anything read now.
                    chunk = carry + (ser.read(n) if n else b"")
                    if self._stream_logged_chunks < 20:
                        logger.info("[MAKCU] stream raw bytes: %s", chunk.hex(' '))
                        self._stream_logged_chunks += 1
                    # Never discard before parsing: a single read can hold a
                    # button release plus a burst of later frames (legacy
                    # firmware re-emits a frame for every wheel HID report), and
                    # clearing the buffer here dropped that release — Mouse1
                    # then read as held until its next physical edge.
                    buf.extend(chunk)

                    # One-time protocol decision for this connection —
                    # binary vs. either ASCII form — made before either
                    # sub-parser below ever runs. Whichever marker appears
                    # earliest in the buffer wins; a real device only ever
                    # speaks one of the two.
                    if self._stream_frame_len is None and not self._stream_binary_mode:
                        idx_km = buf.find(_KM_PREFIX)
                        idx_bin = buf.find(_BIN_SYNC)
                        if idx_bin != -1 and (idx_km == -1 or idx_bin < idx_km):
                            self._stream_binary_mode = True
                            logger.info(
                                "[MAKCU] button stream frame format detected: binary (0xDE 0xAD sync)")

                    if self._stream_binary_mode:
                        while True:
                            idx = buf.find(_BIN_SYNC)
                            if idx == -1:
                                del buf[:max(0, len(buf) - (len(_BIN_SYNC) - 1))]
                                break
                            if idx + _BIN_HDR_LEN > len(buf):
                                # Sync found but the length field hasn't
                                # fully arrived yet.
                                del buf[:idx]
                                break
                            length = buf[idx + 2] | (buf[idx + 3] << 8)
                            if length > _BIN_MAX_PLAUSIBLE_LEN:
                                # Not a real frame — a coincidental 0xDE 0xAD
                                # in unrelated noise. Resync on the next
                                # occurrence, same principle as the ASCII
                                # legacy format's mismatched-suffix resync.
                                resync_idx = buf.find(_BIN_SYNC, idx + len(_BIN_SYNC))
                                if resync_idx == -1:
                                    del buf[:max(0, len(buf) - (len(_BIN_SYNC) - 1))]
                                else:
                                    del buf[:resync_idx]
                                continue
                            frame_len = _BIN_HDR_LEN + 1 + length  # + tag byte + payload
                            if idx + frame_len > len(buf):
                                del buf[:idx]
                                break
                            tag = buf[idx + _BIN_HDR_LEN]
                            payload = bytes(buf[idx + _BIN_HDR_LEN + 1:idx + frame_len])
                            if tag == _BIN_BTN_TAG and len(payload) >= 3 and payload[0] == 0x01:
                                btn_id, state = payload[1], payload[2]
                                frame = bytes(buf[idx:idx + frame_len])
                                if btn_id == 0xFF and state == 0xFF:
                                    # Overflow (mak-suite MAK_API "Input change
                                    # streams"): the device has disabled the mouse
                                    # stream and dropped its queued changes, so no
                                    # release will ever arrive for a held button.
                                    # Spec: discard cached state and re-enable —
                                    # each enable starts from released and the
                                    # next physical report re-sends held buttons.
                                    logger.warning(
                                        "[MAKCU] button stream overflow — resetting button "
                                        "state and re-enabling the stream")
                                    self._set_btn_mask(0, frame)
                                    self._reenable_button_stream()
                                elif 0 <= btn_id <= 4:
                                    bit = 1 << btn_id
                                    self._set_btn_mask(
                                        (self._btn_mask | bit) if state else (self._btn_mask & ~bit),
                                        frame)
                            elif (tag == _BIN_BTN_TAG and len(payload) >= 3
                                    and payload[0] == mp.STREAM_KIND_KEYBOARD):
                                key, state = payload[1], payload[2]
                                if key == 0xFF and state == 0xFF:
                                    logger.warning(
                                        "[MAKCU] keyboard stream overflow — resetting key "
                                        "state and re-enabling the stream")
                                    self._keys_down.clear()
                                    self._reenable_keyboard_stream()
                                elif state:
                                    self._keys_down.add(key)
                                else:
                                    self._keys_down.discard(key)
                            elif tag != _BIN_BTN_TAG and self._mak_api:
                                # A reply to one of our own MAK_API requests.
                                self._dispatch_reply(tag, payload)
                            # Anything else is a report type this parser
                            # doesn't know — skip it rather than misread it.
                            del buf[:idx + frame_len]
                        if len(buf) > _STREAM_MAX_LEFTOVER:
                            del buf[:-_STREAM_MAX_LEFTOVER]
                        continue

                    while True:
                        idx = buf.find(_KM_PREFIX)
                        if idx == -1:
                            # No prefix in buffer; keep only a possible partial tail
                            del buf[:max(0, len(buf) - (len(_KM_PREFIX) - 1))]
                            break

                        if self._stream_frame_len is None:
                            # One-time format detection for this connection —
                            # see docstring. Give the device a short grace
                            # window to deliver the one deciding byte before
                            # falling back to "no more coming" (= short form).
                            needed = idx + _SHORT_FRAME_LEN + 1
                            if len(buf) < needed:
                                deadline = time.monotonic() + _DETECT_GRACE_S
                                while len(buf) < needed and time.monotonic() < deadline:
                                    more = ser.in_waiting
                                    if more:
                                        buf.extend(ser.read(more))
                                    else:
                                        time.sleep(0.001)
                            if len(buf) >= needed and buf[idx + _SHORT_FRAME_LEN] == 0x0D:
                                self._stream_frame_len = _LEGACY_FRAME_LEN
                            else:
                                self._stream_frame_len = _SHORT_FRAME_LEN
                            logger.info(
                                "[MAKCU] button stream frame format detected: %s",
                                "legacy (10-byte)" if self._stream_frame_len == _LEGACY_FRAME_LEN
                                else "makxd (4-byte)")

                        frame_len = self._stream_frame_len
                        if idx + frame_len > len(buf):
                            # Full frame not yet arrived; drop bytes before the
                            # prefix and wait for the rest.
                            del buf[:idx]
                            break
                        mask = buf[idx + 3]
                        if frame_len == _LEGACY_FRAME_LEN:
                            trustworthy = bytes(buf[idx + 4:idx + frame_len]) == _SUFFIX
                        else:
                            # The unframed 4-byte form has no suffix to check,
                            # but its mask is 5 bits: a byte >= 0x20 after "km."
                            # is ASCII (a command reply/echo/ERR on the same
                            # port), never an event — e.g. "km.info()" would
                            # otherwise read as mask 'i' & 0x1F = LMB+Side1.
                            trustworthy = mask <= _BTN_BITS
                        if not trustworthy:
                            # Not a trustworthy frame — resync on the next "km."
                            resync_idx = buf.find(_KM_PREFIX, idx + len(_KM_PREFIX))
                            if resync_idx == -1:
                                del buf[:max(0, len(buf) - (len(_KM_PREFIX) - 1))]
                            else:
                                del buf[:resync_idx]
                            continue
                        self._set_btn_mask(mask, bytes(buf[idx:idx + frame_len]))
                        del buf[:idx + frame_len]
                    if len(buf) > _STREAM_MAX_LEFTOVER:
                        del buf[:-_STREAM_MAX_LEFTOVER]
                else:
                    time.sleep(0.001)
            except Exception:
                # Device likely dropped mid-read. Mark disconnected (matches
                # the write-path pattern in _write_worker/click()) so the
                # reconnect watchdog notices instead of the button state
                # (lmb_held/rmb_held) silently freezing forever.
                self._connected = False
                break

    # ------------------------------------------------------------------
    # Info query
    # ------------------------------------------------------------------

    def _query_info(self) -> dict:
        """Send km.info() and parse key=value pairs.

        Manages its own locking, releasing it across the reply-wait sleep,
        so it never blocks move()/click() for the duration of the query.

        Pauses `_stream_reader()`'s own reading (if a stream is running)
        for the duration of this exchange, so the two never race for the
        same bytes on the wire — this was a real, confirmed bug:
        `_stream_reader()`'s own docstring used to assume every byte
        arriving while the stream is active is a button-stream event, but
        this method (the Keys & HW / Other page's periodic hardware-info
        refresh calls it on a live ~3s timer while connected) writes
        km.info() and reads its reply on the *same* serial port with no
        coordination between the two — confirmed via a real hardware
        capture where `_stream_reader()` itself logged consuming this
        method's own "km.info()\\r\\nERR\\r\\n>>> " traffic. On a legacy-framed
        device the reply's mismatched "\\r\\n>>> " suffix at least gets
        rejected by the stream reader's own resync check, but on a MAKXD
        device (unframed 4-byte events, no suffix to verify at all) the
        4th byte of "km.info()" (`'i'` = 0x69) gets read as a genuine
        button mask (`0x69 & _BTN_BITS = 0x09` — a spurious LMB+Side1
        "press") — i.e. "aim activates for no reason".

        An earlier version of this fix instead stopped and restarted the
        device-side stream itself (`km.buttons(0)`/`km.buttons(1)`) around
        the query, which introduced a second, worse real bug: MAKCU's
        button stream is edge-triggered — "streams only emit on new
        frames" per the protocol docs — so if the aim button was already
        held when the stream got disabled and re-enabled, the device had
        no *new* state change to report and never re-sent "held" after
        re-enabling. `_btn_mask` (reset to 0 by `_stop_stream()`) then
        stayed stuck at "not held" for the rest of that hold, however long
        it lasted, until the next genuine press/release edge — reported
        exactly as "holding the aim key stops aiming after ~2-3 seconds"
        (matching this method's own ~3s call cadence from the Hardware
        panel timer). This version never stops the device-side stream or
        touches `_btn_mask` at all — it only tells `_stream_reader()` to
        skip its own `ser.read()` calls for the ~150ms this query takes,
        which is enough to remove the race without losing in-progress
        button-hold state.

        While streaming, nothing read here is thrown away: button events
        that arrive during the query window (e.g. a Mouse1 release) share
        the wire with the reply, so every byte read is handed back to
        `_stream_reader()` via `_stream_carry`. Purging the input buffer
        and treating the whole window as reply text used to swallow those
        events — this runs every 3s for the whole session (other_page.py's
        hardware panel timer), so a release landing in that ~150ms window
        left Mouse1 reading as held until its next physical edge. The reply
        itself is ASCII, which every stream frame format rejects.
        """
        if self._mak_api:
            # V4/MAKXD has no km.info(); its info was read at connect.
            return dict(self._device_info)
        was_streaming = self._stream_thread is not None and self._stream_thread.is_alive()
        if was_streaming:
            self._stream_read_pause.set()
        try:
            with self._lock:
                if not self._serial:
                    return {}
                if not was_streaming:
                    self._serial.reset_input_buffer()
                self._serial.write(self.CMD_INFO.encode('ascii'))
                self._serial.flush()
            time.sleep(0.15)
            with self._lock:
                if not self._serial:
                    return {}
                raw_bytes = self._serial.read(self._serial.in_waiting)
            if was_streaming and raw_bytes:
                with self._stream_carry_lock:
                    self._stream_carry.extend(raw_bytes)
            raw = raw_bytes.decode('ascii', errors='ignore')
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
        finally:
            self._stream_read_pause.clear()

    def query_info(self) -> dict:
        """Return parsed km.info() dict."""
        return self._query_info()

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
        if self._move_lock_active:
            self._send_move_lock(False)
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

    def move(self, dx: int, dy: int, accumulate: bool = False):
        """Relative mouse move. Handed to the async write thread.

        ``accumulate`` is false for PID. The value is the full correction for
        the current error, so a step still sitting in the pending slot is
        replaced. Summing those re-measured corrections is what made the aim
        snap past the target.

        Movement paths pass ``accumulate=True``. Each call is only the next
        slice of the curve (linear, exponential, bezier, adaptive, perlin).
        Replacing that slice leaves MAKCU with whichever sample happened to
        be pending when the writer woke, so the path never plays. Adding the
        slices preserves the curve until the writer drains it.
        """
        if not self.is_connected():
            return
        with self._pending_lock:
            self._pending_dx, self._pending_dy = mp.merge_relative_move(
                self._pending_dx, self._pending_dy, dx, dy, accumulate)
        self._pending_event.set()

    def click(self, action: int = 1):
        """Left mouse click. action: 1=click, 2=press, 3=release."""
        if not self.is_connected():
            return
        if self._mak_api:
            down, up = mp.button_frame(0, True), mp.button_frame(0, False)
        else:
            down, up = self.CMD_LEFT_DOWN.encode('ascii'), self.CMD_LEFT_UP.encode('ascii')
        try:
            if action == 1:
                with self._lock:
                    if self._serial and self._serial.is_open:
                        self._serial.write(down)
                time.sleep(0.03)
                with self._lock:
                    if self._serial and self._serial.is_open:
                        self._serial.write(up)
                return
            data = down if action == 2 else up if action == 3 else None
            if data:
                with self._lock:
                    if self._serial and self._serial.is_open:
                        self._serial.write(data)
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

    @property
    def side1_held(self) -> bool:
        """True when the first side button (S1, e.g. Mouse4) is physically pressed."""
        return bool(self._btn_mask & 0x08)

    @property
    def side2_held(self) -> bool:
        """True when the second side button (S2, e.g. Mouse5) is physically pressed."""
        return bool(self._btn_mask & 0x10)

    @property
    def mmb_held(self) -> bool:
        """True when the middle mouse button is physically pressed."""
        return bool(self._btn_mask & 0x04)

    # ------------------------------------------------------------------
    # Device info
    # ------------------------------------------------------------------

    @property
    def is_mak_api(self) -> bool:
        """MAKCU V4 / MAKXD (binary MAK_API) rather than V3 ASCII-only."""
        return self._mak_api

    @property
    def transport(self) -> str:
        return self._transport

    @property
    def firmware_version(self) -> Optional[int]:
        return self._firmware_version

    # ------------------------------------------------------------------
    # Hotkeys through the device (keyboard + mouse-button streams)
    # ------------------------------------------------------------------

    def set_keyboard_stream_enabled(self, enabled: bool) -> None:
        """Subscribe to (or drop) the device's keyboard change stream.
        Remembered across reconnects. MAK_API devices only."""
        enabled = bool(enabled)
        if enabled == self._keyboard_stream_wanted:
            return
        self._keyboard_stream_wanted = enabled
        if not enabled:
            self._keys_down.clear()
        if self._mak_api and self.is_connected():
            self._write_frame(mp.stream_enable_frame(mp.STREAM_KIND_KEYBOARD, enabled))

    @property
    def keyboard_stream_active(self) -> bool:
        return self._keyboard_stream_wanted and self._mak_api and self.is_connected()

    def is_vk_down(self, vk: int) -> Optional[bool]:
        """Whether Windows virtual-key `vk` is held, as seen by the device.

        Mouse-button VKs come from the button stream; keyboard VKs from the
        keyboard stream (MAK_API with set_keyboard_stream_enabled(True)).
        None means the device can't answer for this VK — the caller should
        fall back to the local OS state."""
        if not self.is_connected():
            return None
        bit = mp.MOUSE_VK_TO_BIT.get(vk)
        if bit is not None:
            return bool(self._btn_mask & bit)
        usages = mp.VK_TO_HID.get(vk)
        if usages is None or not self.keyboard_stream_active:
            return None
        return any(u in self._keys_down for u in usages)

    # ------------------------------------------------------------------
    # Physical-movement lock (masks physical input only; injection bypasses it)
    # ------------------------------------------------------------------

    def set_physical_move_lock(self, active: bool) -> None:
        """Block (or restore) the user's own mouse movement.

        Must be re-asserted at least every _MOVE_LOCK_REFRESH_S while active;
        the write thread releases it otherwise, so a stalled caller can't
        leave the mouse dead. Idempotent — only state changes hit the wire."""
        if active:
            self._move_lock_asserted_at = time.monotonic()
        if bool(active) != self._move_lock_active and self.is_connected():
            self._send_move_lock(bool(active))

    def _send_move_lock(self, active: bool) -> None:
        if self._mak_api:
            ok = self._write_frame(mp.move_mask_frame(active, active, active, active))
        else:
            n = 1 if active else 0
            ok = self._write_frame(f"km.lock_mx({n})\r\nkm.lock_my({n})\r\n".encode("ascii"))
        if ok or not active:
            self._move_lock_active = active and ok
        logger.debug("[MAKCU] physical movement lock %s", "on" if self._move_lock_active else "off")

    @property
    def physical_move_locked(self) -> bool:
        return self._move_lock_active

    # ------------------------------------------------------------------
    # Firmware mouse spread (live device settings, MAK_API only)
    # ------------------------------------------------------------------

    def _settings_call(self, op_frame: bytes, op: int, timeout: float = 0.5):
        """Send one settings transaction; returns (status, body) or None."""
        stream_running = self._stream_thread is not None and self._stream_thread.is_alive()
        if stream_running:
            waiter = [threading.Event(), None]
            with self._reply_lock:
                self._reply_waiters.setdefault(mp.CMD_CONNECTION, []).append(waiter)
            if not self._write_frame(op_frame) or not waiter[0].wait(timeout):
                with self._reply_lock:
                    if waiter in self._reply_waiters.get(mp.CMD_CONNECTION, []):
                        self._reply_waiters[mp.CMD_CONNECTION].remove(waiter)
                return None
            payload = waiter[1]
        else:
            payload = self._mak_request_direct(mp.CMD_CONNECTION, op_frame[5:], timeout)
        parsed = mp.parse_settings_reply(payload or b"")
        if parsed is None or parsed[0] != op:
            return None
        return parsed[1], parsed[2]

    def _settings_info(self) -> Optional[dict]:
        reply = self._settings_call(mp.settings_frame(mp.SETTINGS_OP_INFO), mp.SETTINGS_OP_INFO)
        if not reply or reply[0] != 0:
            return None
        return mp.parse_settings_info(reply[1])

    def _settings_read_image(self, revision: int) -> Optional[bytearray]:
        image = bytearray()
        while len(image) < mp.SETTINGS_IMAGE_BYTES:
            length = min(mp.SETTINGS_CHUNK, mp.SETTINGS_IMAGE_BYTES - len(image))
            reply = self._settings_call(
                mp.settings_read_frame(revision, len(image), length), mp.SETTINGS_OP_READ)
            if not reply or reply[0] != 0 or len(reply[1]) < 6 + length:
                return None
            image.extend(reply[1][6:6 + length])
        return image

    def get_mouse_spread(self) -> Optional[int]:
        """Current firmware mouse spread (0-100 %), or None if unsupported."""
        if not (self._mak_api and self.is_connected()):
            return None
        with self._settings_lock:
            info = self._settings_info()
            if not info or not info["sections"] & mp.SETTINGS_SECTION_MOUSE:
                return None
            image = self._settings_read_image(info["revision"])
            return image[mp.SETTINGS_MOUSE_SPREAD_OFFSET] if image else None

    def set_mouse_spread(self, percent: int, persist: bool = False) -> tuple:
        """Apply firmware mouse spread live (DEVICE_SETTINGS.md: INFO, READ the
        400-byte image, BEGIN, WRITE it back with byte 396 changed, APPLY;
        SAVE only when `persist`). Returns (ok, reason)."""
        if not (self._mak_api and self.is_connected()):
            return False, "unsupported"
        percent = max(0, min(100, int(percent)))
        with self._settings_lock:
            info = self._settings_info()
            if not info:
                return False, "no_reply"
            if not info["sections"] & mp.SETTINGS_SECTION_MOUSE:
                return False, "unsupported"
            revision = info["revision"]
            image = self._settings_read_image(revision)
            if image is None:
                return False, "read_failed"
            if image[mp.SETTINGS_MOUSE_SPREAD_OFFSET] == percent and not persist:
                return True, "unchanged"
            image[mp.SETTINGS_MOUSE_SPREAD_OFFSET] = percent
            reply = self._settings_call(
                mp.settings_begin_frame(revision, mp.SETTINGS_SECTION_MOUSE), mp.SETTINGS_OP_BEGIN)
            if not reply or reply[0] != 0 or len(reply[1]) < 4:
                return False, mp.SETTINGS_STATUS.get(reply[0], "begin_failed") if reply else "no_reply"
            token = int.from_bytes(reply[1][:4], "little")
            for offset in range(0, mp.SETTINGS_IMAGE_BYTES, mp.SETTINGS_CHUNK):
                chunk = bytes(image[offset:offset + mp.SETTINGS_CHUNK])
                reply = self._settings_call(
                    mp.settings_write_frame(token, offset, chunk), mp.SETTINGS_OP_WRITE)
                if not reply or reply[0] != 0:
                    return False, "write_failed"
            reply = self._settings_call(mp.settings_apply_frame(token), mp.SETTINGS_OP_APPLY)
            if not reply or reply[0] != 0:
                return False, mp.SETTINGS_STATUS.get(reply[0], "apply_failed") if reply else "no_reply"
            if persist:
                info = self._settings_info()
                if not info:
                    return False, "no_reply"
                reply = self._settings_call(
                    mp.settings_save_frame(info["revision"], mp.SETTINGS_SECTION_MOUSE),
                    mp.SETTINGS_OP_SAVE)
                if not reply or reply[0] not in (0, 1):
                    return False, "save_failed"
        logger.info("[MAKCU] firmware mouse spread set to %d%%%s", percent, " (saved)" if persist else "")
        return True, "applied"

    def request_mouse_spread(self, percent: Optional[int]) -> None:
        """Apply `percent` in the background (None = stop managing it).
        Cheap to call repeatedly; only a changed value reaches the device,
        and it is re-applied after every reconnect."""
        self._spread_desired = None if percent is None else max(0, min(100, int(percent)))
        if self._spread_desired is not None and self._spread_desired != self._spread_applied:
            self._kick_spread_worker()

    def _kick_spread_worker(self) -> None:
        if self._spread_thread is not None and self._spread_thread.is_alive():
            return
        self._spread_thread = threading.Thread(
            target=self._spread_worker, daemon=True, name="makcu-spread")
        self._spread_thread.start()

    def _spread_worker(self) -> None:
        # Newest value wins: one transaction in flight, then re-check.
        while True:
            want = self._spread_desired
            if want is None or want == self._spread_applied:
                return
            if not (self._mak_api and self.is_connected()):
                return
            ok, reason = self.set_mouse_spread(want)
            if not ok:
                logger.warning("[MAKCU] firmware mouse spread not applied: %s", reason)
                return
            self._spread_applied = want

    # ------------------------------------------------------------------
    # Injection / masks / remaps (MAK_API; wheel also works on V3)
    # ------------------------------------------------------------------

    def wheel(self, delta: int) -> bool:
        if not self.is_connected():
            return False
        if self._mak_api:
            return self._write_frame(mp.wheel_frame(delta))
        return self._write_frame(f"km.wheel({int(delta)})\r\n".encode("ascii"))

    def key_down(self, hid_usage: int) -> bool:
        return self._mak_api and self.is_connected() and self._write_frame(
            mp.frame(mp.CMD_KEY_DOWN, bytes([hid_usage & 0xFF])))

    def key_up(self, hid_usage: int) -> bool:
        return self._mak_api and self.is_connected() and self._write_frame(
            mp.frame(mp.CMD_KEY_UP, bytes([hid_usage & 0xFF])))

    def key_press(self, hid_usage: int, hold_ms: Optional[int] = None,
                  random_range: Optional[int] = None) -> bool:
        return self._mak_api and self.is_connected() and self._write_frame(
            mp.key_press_frame(hid_usage, hold_ms, random_range))

    def keys_release_all(self) -> bool:
        """KEY_INIT: releases injected keys and clears keyboard masks/remaps."""
        return self._mak_api and self.is_connected() and self._write_frame(mp.frame(mp.CMD_KEY_INIT))

    def set_key_mask(self, hid_usage: int, mode: int) -> bool:
        return self._mak_api and self.is_connected() and self._write_frame(
            mp.frame(mp.CMD_KEY_MASK, bytes([hid_usage & 0xFF, mode & 0xFF])))

    def set_key_remap(self, source: int, target: int) -> bool:
        return self._mak_api and self.is_connected() and self._write_frame(
            mp.frame(mp.CMD_KEY_REMAP, bytes([source & 0xFF, target & 0xFF])))

    def set_button_mask(self, button: int, enabled: bool) -> bool:
        """button: 0=left 1=right 2=middle 3=side1 4=side2."""
        return (0 <= button <= 4 and self._mak_api and self.is_connected()
                and self._write_frame(mp.button_mask_frame(button, enabled)))

    def set_wheel_mask(self, down: bool, up: bool) -> bool:
        return self._mak_api and self.is_connected() and self._write_frame(
            mp.wheel_mask_frame(down, up))


# ---------------------------------------------------------------------------
# Module-level singleton and convenience functions
# ---------------------------------------------------------------------------

makcu_mouse = MakcuMouse()


def send_mouse_move_makcu(dx: int, dy: int, accumulate: bool = False):
    if accumulate:
        makcu_mouse.move(dx, dy, accumulate=True)
    else:
        makcu_mouse.move(dx, dy)


def send_mouse_click_makcu(action: int = 1):
    makcu_mouse.click(action)
    return True


def connect_makcu(com_port: str, baud_rate: int = _OPERATING_BAUD) -> bool:
    return makcu_mouse.connect(com_port, baud_rate)


def connect_makcu_udp(host: str, port: int = 8080) -> bool:
    return makcu_mouse.connect_udp(host, port)


def disconnect_makcu():
    makcu_mouse.disconnect()


def is_makcu_connected() -> bool:
    return makcu_mouse.is_connected()
