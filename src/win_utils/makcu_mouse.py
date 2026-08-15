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

logger = logging.getLogger(__name__)

# Official 4 Mbaud connection constants
_OPERATING_BAUD    = 4_000_000
_BAUD_CHANGE_FRAME = bytes([0xDE, 0xAD, 0x05, 0x00, 0xA5, 0x00, 0x09, 0x3D, 0x00])

# Button stream parsing — see _stream_reader() for the frame format.
_BTN_BITS = 0x1F  # bits 0-4 = L,R,M,S1,S2 — everything else is noise/high-byte


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
            self._query_info()
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
            self._query_info()
            logger.info("[MAKCU] Connected to %s @ %d baud", com_port, _OPERATING_BAUD)

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

        # Start threads outside the lock
        self._start_stream()
        self._start_write_thread()
        self._start_reconnect_thread()
        return True

    def _try_open(self, com_port: str, baud: int) -> bool:
        """Open port at baud, probe with km.version(). Returns True if km.MAKCU found.

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
                if b'km.MAKCU' in raw:
                    break
                if not waiting:
                    time.sleep(0.01)
            if b'km.MAKCU' not in raw:
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
        self._btn_mask = 0
        try:
            with self._lock:
                if self._serial and self._serial.is_open:
                    self._serial.reset_input_buffer()
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
        """Drain the pending relative movement and write+flush to serial.

        Runs on its own thread so the inference thread (which calls move())
        is never blocked waiting on the serial port.

        Takes the latest pending delta and zeroes it in one locked step, so
        a move() landing while this write is in flight goes into a freshly-
        zeroed slot rather than being lost or merged into what's about to be
        sent. If two or more move() calls land before this loop gets back
        around to draining, only the newest survives — see move()'s
        docstring for why replacing (not summing) the stale one is correct
        here, not a regression.
        """
        while not self._write_stop.is_set():
            if not self._pending_event.wait(0.01):
                continue
            with self._pending_lock:
                dx, dy = self._pending_dx, self._pending_dy
                self._pending_dx = 0
                self._pending_dy = 0
                self._pending_event.clear()
            if dx == 0 and dy == 0:
                continue
            cmd = self.CMD_MOVE.format(dx=dx, dy=dy)
            try:
                with self._lock:
                    if self._serial and self._serial.is_open:
                        self._serial.write(cmd.encode('ascii'))
                        self._serial.flush()
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
        """Daemon thread: parse km.<mask>\\r\\n>>> button frames, update _btn_mask.

        Confirmed against a raw hex capture from real hardware (this
        codebase's prior three attempts at this format were each tried and
        reported as not detecting real clicks — see git history on this
        method for what didn't work and why each guess seemed plausible at
        the time). The actual frame, captured verbatim:

            6b 6d 2e 01 0d 0a 3e 3e 3e 20
            "k  m  .  <mask=0x01>  \\r  \\n  >  >  >  ' '"

        i.e. `km.` + a single mask byte + the exact same `\\r\\n>>> ` suffix
        the docs describe for one-off ASCII command replies — the buttons
        stream just pushes this unsolicited on every state change, using
        the same reply framing as everything else instead of a distinct
        compact encoding. 10 bytes total, mask is 1 byte (not 2).

        Verify the trailing suffix too (not just the "km." prefix) when
        enough bytes are buffered to check — a real mask byte can't corrupt
        into "km.", but requiring the suffix as well catches a byte
        misaligning the frame from either direction and forces a resync on
        the next real "km." instead of misreading a corrupted mask.

        km.echo(0) means the device sends nothing in response to move/click writes,
        so all incoming bytes are button stream events — no lock needed on reads.
        """
        buf = bytearray()
        _KM_PREFIX = b"km."
        _SUFFIX = b"\r\n>>> "
        FRAME_LEN = len(_KM_PREFIX) + 1 + len(_SUFFIX)  # km. + mask + \r\n>>>(space) = 10
        logged_chunks = 0
        while not self._stream_stop.is_set():
            try:
                ser = self._serial
                if not ser or not ser.is_open:
                    break
                n = ser.in_waiting
                if n:
                    chunk = ser.read(n)
                    if logged_chunks < 20:
                        logger.info("[MAKCU] stream raw bytes: %s", chunk.hex(' '))
                        logged_chunks += 1
                    buf.extend(chunk)
                    if len(buf) > 256:
                        buf.clear()
                    while True:
                        idx = buf.find(_KM_PREFIX)
                        if idx == -1:
                            # No prefix in buffer; keep only a possible partial tail
                            del buf[:max(0, len(buf) - (len(_KM_PREFIX) - 1))]
                            break
                        if idx + FRAME_LEN > len(buf):
                            # Full frame not yet arrived; drop bytes before the
                            # prefix and wait for the rest.
                            del buf[:idx]
                            break
                        mask = buf[idx + 3]
                        suffix = bytes(buf[idx + 4:idx + FRAME_LEN])
                        if suffix != _SUFFIX:
                            # Not a trustworthy frame — resync on the next "km."
                            resync_idx = buf.find(_KM_PREFIX, idx + len(_KM_PREFIX))
                            if resync_idx == -1:
                                del buf[:max(0, len(buf) - (len(_KM_PREFIX) - 1))]
                            else:
                                del buf[:resync_idx]
                            continue
                        self._btn_mask = mask & _BTN_BITS
                        del buf[:idx + FRAME_LEN]
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
        """
        try:
            with self._lock:
                if not self._serial:
                    return {}
                self._serial.reset_input_buffer()
                self._serial.write(self.CMD_INFO.encode('ascii'))
                self._serial.flush()
            time.sleep(0.15)
            with self._lock:
                if not self._serial:
                    return {}
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
        """Relative mouse move. Handed to the async write thread.

        Latest-only: if a previous move hasn't been written yet, this one
        *replaces* it rather than adding to it.

        dx/dy arrive here as the single, complete correction ai_aiming.py
        computed for the CURRENT frame — PID output, humanization, sub-pixel
        carry and jitter are all already folded in before send_mouse_move()
        is ever called. A pending value that hasn't reached the wire yet
        doesn't describe "movement still owed on top of the next frame's" —
        it describes an error the next frame's fresh PID output already
        re-measures and re-corrects for, because the aim loop is a closed
        feedback loop, not a source of independent deltas. Summing them
        (an earlier version of this code did, to avoid losing displacement)
        double-counts that same not-yet-visually-applied error every time
        the real USB-injection + game-frame + capture round trip is slower
        than detect_interval — which is routinely true — so N queued-up
        PID cycles land as one N-times-too-large jump. That's the systemic
        Y-axis overshoot ("aim snaps past the target") this fixes: replacing
        the stale pending value with the fresh one is what keeps each
        physical move sized to exactly one frame's correction.
        """
        if not self.is_connected():
            return
        with self._pending_lock:
            self._pending_dx = max(-32768, min(32767, int(dx)))
            self._pending_dy = max(-32768, min(32767, int(dy)))
        self._pending_event.set()

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
