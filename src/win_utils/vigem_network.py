"""ViGEm Network backend — drives a remote ViGEm virtual Xbox 360 controller
over TCP using the MAKCU-style km.controller() text protocol.

The same connection also carries the remote PC's mouse buttons back to Axiom:
when ``config.remote_btn_enabled`` is on, the client sends ``km.mouse_stream(1)``
after auth and the server (GamepadEnhancer) pushes ``km.mouse(mask)`` lines
(bit0 L, 1 R, 2 M, 3 X1, 4 X2) on change plus a 1 s keep-alive. One address,
one port, one token for both directions. The mask reads 0 when the link drops,
after ``_MOUSE_STALE_S`` without a push, and until fresh data arrives after a
reconnect.

Pure stdlib (socket/threading/time). No win32api, no ViGEmBus driver needed on
the Axiom machine; the driver only needs to be installed on the remote PC
running GamepadEnhancer.

Protocol reference: gamepadenhancer/docs/NETWORK_API.md
"""

from __future__ import annotations

import logging
import socket
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)

_ID_RIGHT_TRIGGER = 11
_ID_RIGHT_X = 14
_ID_RIGHT_Y = 15

_MAX_STICK = 32767
_MIN_STICK = -32768
_MAX_TRIGGER = 1023

_KEEPALIVE_S = 5.0
_WRITE_POLL_S = 0.002

_RECONNECT_BASE_S = 0.5
_RECONNECT_CAP_S = 4.0
_CONNECT_TIMEOUT_S = 4.0
_SEND_TIMEOUT_S = 0.15

BASE_PIXELS = 50.0

_MOUSE_STALE_S = 3.0
_MOUSE_MASK_ALL = 0x1F
_READ_CHUNK = 512
_MAX_PENDING_LINE = 1024

# Remote mouse button name -> bit in the km.mouse mask (same names as config.remote_aim_button).
BUTTON_BITS = {"lmb": 0x01, "rmb": 0x02, "mmb": 0x04, "side1": 0x08, "side2": 0x10}


def parse_push_line(line: str) -> Optional[int]:
    """Mask from a pushed ``km.mouse(N)`` line, or None for anything else.

    Replies end in a prompt (``\\r\\n>>> ``) with no newline after it, so the next
    pushed line arrives glued to it (``>>> km.mouse(5)``); the prompt is dropped.
    """
    line = line.strip()
    while line.startswith(">>>"):
        line = line[3:].lstrip()
    if not (line.startswith("km.mouse(") and line.endswith(")")):
        return None
    arg = line[len("km.mouse("):-1].strip()
    if not arg.isdigit():
        return None
    value = int(arg)
    if value > _MOUSE_MASK_ALL:
        return None
    return value


def _clamp(v: int, lo: int, hi: int) -> int:
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


def map_pixels_to_stick(dx: float, dy: float, sensitivity: float = 1.0,
                        input_dz: float = 0.0, game_dz: float = 0.0) -> tuple:
    """Pixel delta -> (rx, ry) stick values.

    input_dz: |v| below this fraction of full stick is dropped (jitter filter).
    game_dz:  the game's own inner dead zone; non-zero values are remapped to
              start just past it so small corrections still register.
    """
    def axis(d: float) -> int:
        v = max(-1.0, min(1.0, d * sensitivity / BASE_PIXELS))
        if abs(v) < input_dz or v == 0.0:
            return 0
        if game_dz > 0.0:
            mag = min(1.0, game_dz + abs(v) * (1.0 - game_dz))
            v = mag if v > 0 else -mag
        return _clamp(int(v * _MAX_STICK), _MIN_STICK, _MAX_STICK)

    return axis(dx), axis(dy)


class VigemNetworkClient:

    def __init__(self) -> None:
        self._sock: Optional[socket.socket] = None
        self._lock = threading.Lock()
        self._connected = False

        self._pending_rx: Optional[int] = None
        self._pending_ry: Optional[int] = None
        self._pending_rt: Optional[int] = None
        self._pending_lock = threading.Lock()
        self._last_move_t = 0.0

        self._last_sent_rx: Optional[int] = None
        self._last_sent_ry: Optional[int] = None
        self._last_sent_rt: Optional[int] = None

        self._stop_event = threading.Event()
        self._write_thread: Optional[threading.Thread] = None
        self._reconnect_thread: Optional[threading.Thread] = None
        self._reader_thread: Optional[threading.Thread] = None
        self._resend_all = False

        # Remote mouse buttons pushed back by the server on this same socket.
        self._mouse_lock = threading.Lock()
        self._mouse_mask = 0
        self._mouse_t = float("-inf")
        self._mouse_subscribed = False

        self._host: str = ""
        self._port: int = 5555
        self._token: str = ""

    def connect(self, host: str, port: int, token: str) -> bool:
        self._host = host
        self._port = port
        self._token = token
        self._stop_event.clear()
        try:
            sock = self._dial(host, port, token, _CONNECT_TIMEOUT_S)
        except Exception as e:
            logger.error("[VigemNet] connect failed: %s", e)
            return False
        with self._lock:
            self._sock = sock
            self._connected = True
        self._resend_all = True
        self._mouse_subscribed = False
        self._clear_mouse()
        self._last_sent_rx = None
        self._last_sent_ry = None
        self._last_sent_rt = None
        self._write_thread = threading.Thread(
            target=self._write_loop, daemon=True, name="vigem-net-write")
        self._write_thread.start()
        self._start_reader(sock)
        logger.info("[VigemNet] connected to %s:%d", host, port)
        return True

    def disconnect(self) -> None:
        self._stop_event.set()
        with self._lock:
            if self._sock is not None:
                try:
                    self._sock.sendall(b"km.controller_clear()\n")
                except Exception:
                    pass
                try:
                    self._sock.close()
                except Exception:
                    pass
                self._sock = None
            self._connected = False
        if self._write_thread and self._write_thread.is_alive():
            self._write_thread.join(timeout=1.0)
        if self._reconnect_thread and self._reconnect_thread.is_alive():
            self._reconnect_thread.join(timeout=2.0)
        if self._reader_thread and self._reader_thread.is_alive():
            self._reader_thread.join(timeout=1.0)
        self._write_thread = None
        self._reconnect_thread = None
        self._reader_thread = None
        self._clear_mouse()
        _publish_stick(0, 0)
        logger.info("[VigemNet] disconnected")

    def is_connected(self) -> bool:
        return self._connected

    def enqueue_move(self, rx: int, ry: int) -> None:
        with self._pending_lock:
            self._pending_rx = rx
            self._pending_ry = ry
            self._last_move_t = time.monotonic()

    def enqueue_click(self, value: int) -> None:
        with self._pending_lock:
            self._pending_rt = value

    @staticmethod
    def _dial(host: str, port: int, token: str, timeout: float) -> socket.socket:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        try:
            sock.connect((host, port))
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.sendall(f"auth {token}\n".encode("utf-8"))
            reply = sock.recv(64)
            if not reply or not reply.decode("utf-8", errors="replace").startswith("OK"):
                raise ConnectionError("auth rejected or no reply")
            sock.settimeout(_SEND_TIMEOUT_S)
            return sock
        except Exception:
            sock.close()
            raise

    def _write_loop(self) -> None:
        last_send_t = time.monotonic()
        while not self._stop_event.is_set():
            self._stop_event.wait(_WRITE_POLL_S)
            if self._stop_event.is_set():
                break
            if not self._connected:
                continue

            with self._pending_lock:
                rx = self._pending_rx
                ry = self._pending_ry
                rt = self._pending_rt
                self._pending_rx = None
                self._pending_ry = None
                self._pending_rt = None

            # A stick is held state, not a one-shot delta: with no fresh move
            # for a moment (aim released, target lost, move too small to be
            # sent at all) recenter it instead of leaving it deflected.
            if (rx is None and ry is None
                    and (self._last_sent_rx or self._last_sent_ry)
                    and time.monotonic() - self._last_move_t > _release_seconds()):
                rx = ry = 0

            resend = self._resend_all
            if resend:
                self._resend_all = False

            buf = []
            # Ask the server to push its mouse buttons (or stop) to match the setting.
            # Re-armed after every (re)connect because _mouse_subscribed resets there.
            want_mouse = _mouse_wanted()
            if want_mouse != self._mouse_subscribed:
                buf.append(f"km.mouse_stream({1 if want_mouse else 0})\n")
                self._mouse_subscribed = want_mouse
                if not want_mouse:
                    self._clear_mouse()
            if rx is not None and (resend or rx != self._last_sent_rx):
                buf.append(f"km.controller({_ID_RIGHT_X},{rx})\n")
                self._last_sent_rx = rx
            if ry is not None and (resend or ry != self._last_sent_ry):
                buf.append(f"km.controller({_ID_RIGHT_Y},{ry})\n")
                self._last_sent_ry = ry
            if rt is not None and (resend or rt != self._last_sent_rt):
                buf.append(f"km.controller({_ID_RIGHT_TRIGGER},{rt})\n")
                self._last_sent_rt = rt

            _publish_stick(self._last_sent_rx, self._last_sent_ry)

            now = time.monotonic()
            if not buf:
                if now - last_send_t < _KEEPALIVE_S:
                    continue
                buf.append("km.version()\n")

            data = "".join(buf).encode("utf-8")
            with self._lock:
                sock = self._sock
            if sock is None:
                continue
            try:
                sock.sendall(data)
                last_send_t = time.monotonic()
            except Exception:
                self._mark_broken()

    def _mark_broken(self) -> None:
        with self._lock:
            if not self._connected:
                return
            self._connected = False
            if self._sock is not None:
                try:
                    self._sock.close()
                except Exception:
                    pass
                self._sock = None
        self._clear_mouse()          # a dropped link must never leave a remote button "held"
        logger.warning("[VigemNet] connection lost, will reconnect")
        if not self._stop_event.is_set():
            if self._reconnect_thread is None or not self._reconnect_thread.is_alive():
                self._reconnect_thread = threading.Thread(
                    target=self._reconnect_loop, daemon=True,
                    name="vigem-net-reconnect")
                self._reconnect_thread.start()

    # ---- remote mouse buttons (pushed by the server on this socket) -----------

    def _clear_mouse(self) -> None:
        with self._mouse_lock:
            self._mouse_mask = 0
            self._mouse_t = float("-inf")

    def _set_mouse(self, mask: int) -> None:
        with self._mouse_lock:
            self._mouse_mask = mask
            self._mouse_t = time.monotonic()

    def mouse_mask(self) -> int:
        """Remote buttons currently held (0 when not connected, not subscribed or stale)."""
        if not self._connected:
            return 0
        with self._mouse_lock:
            if time.monotonic() - self._mouse_t > _MOUSE_STALE_S:
                return 0
            return self._mouse_mask

    def _start_reader(self, sock: socket.socket) -> None:
        t = threading.Thread(target=self._read_loop, args=(sock,), daemon=True,
                             name="vigem-net-read")
        self._reader_thread = t
        t.start()

    def _read_loop(self, sock: socket.socket) -> None:
        """Read everything the server sends: replies are ignored, km.mouse(N) pushes update the mask."""
        buf = b""
        while not self._stop_event.is_set():
            with self._lock:
                if self._sock is not sock:        # replaced by a reconnect, or closed
                    return
            try:
                chunk = sock.recv(_READ_CHUNK)
            except socket.timeout:
                continue                          # the socket's short send timeout also applies to recv
            except OSError:
                chunk = b""
            if not chunk:
                with self._lock:
                    current = self._sock is sock
                if current and not self._stop_event.is_set():
                    self._mark_broken()           # server closed on us: reconnect like a failed send
                return
            buf += chunk
            while b"\n" in buf:
                raw, buf = buf.split(b"\n", 1)
                mask = parse_push_line(raw.decode("utf-8", errors="replace"))
                if mask is not None:
                    self._set_mouse(mask)
            if len(buf) > _MAX_PENDING_LINE:      # prompt text with no newline: keep only the tail
                buf = buf[-64:]

    def _reconnect_loop(self) -> None:
        delay = _RECONNECT_BASE_S
        while not self._stop_event.is_set():
            self._stop_event.wait(delay)
            if self._stop_event.is_set():
                return
            try:
                sock = self._dial(self._host, self._port, self._token, 2.0)
                with self._lock:
                    if self._stop_event.is_set():
                        sock.close()
                        return
                    self._sock = sock
                    self._connected = True
                self._resend_all = True
                self._mouse_subscribed = False     # subscribe again on the new socket
                self._clear_mouse()
                self._start_reader(sock)
                logger.info("[VigemNet] reconnected to %s:%d",
                            self._host, self._port)
                return
            except Exception:
                delay = min(delay * 2, _RECONNECT_CAP_S)


# ===== Module-level singleton + public API =====

vigem_net_client: Optional[VigemNetworkClient] = None
_config_ref = None


def connect_vigem_network(host: str, port: int, token: str, config=None) -> bool:
    global vigem_net_client, _config_ref
    _config_ref = config
    if vigem_net_client is not None:
        vigem_net_client.disconnect()
    vigem_net_client = VigemNetworkClient()
    return vigem_net_client.connect(host, port, token)


def disconnect_vigem_network() -> None:
    global vigem_net_client
    if vigem_net_client is not None:
        vigem_net_client.disconnect()
        vigem_net_client = None


def is_vigem_network_connected() -> bool:
    return vigem_net_client is not None and vigem_net_client.is_connected()


def _cfg_float(name: str, default: float) -> float:
    try:
        return float(getattr(_config_ref, name, default))
    except (TypeError, ValueError):
        return default


def _mouse_wanted() -> bool:
    """Whether to receive the remote PC's mouse buttons (config.remote_btn_enabled, read live)."""
    try:
        return bool(getattr(_config_ref, 'remote_btn_enabled', False))
    except Exception:
        return False


def vigem_net_mouse_mask() -> int:
    """Remote mouse buttons held right now (bit0 L, 1 R, 2 M, 3 X1, 4 X2); 0 if not connected/subscribed/stale."""
    c = vigem_net_client
    return c.mouse_mask() if c is not None else 0


def vigem_net_button_held(name: str) -> bool:
    """True while the named remote button ('lmb'/'rmb'/'mmb'/'side1'/'side2') is held on the GamepadEnhancer PC."""
    bit = BUTTON_BITS.get(name)
    if bit is None:
        return False
    return bool(vigem_net_mouse_mask() & bit)


def _release_seconds() -> float:
    return max(10.0, min(500.0, _cfg_float('vigem_net_release_ms', 50.0))) / 1000.0


def _publish_stick(rx, ry) -> None:
    """Expose the stick value actually sent (-1..1) for the overlay gauge."""
    if _config_ref is None:
        return
    try:
        _config_ref.vigem_stick_x = max(-1.0, min(1.0, (rx or 0) / _MAX_STICK))
        _config_ref.vigem_stick_y = max(-1.0, min(1.0, (ry or 0) / _MAX_STICK))
    except Exception:
        pass


def send_mouse_move_vigem_network(dx: float, dy: float) -> None:
    if vigem_net_client is None or not vigem_net_client.is_connected():
        return
    rx, ry = map_pixels_to_stick(
        dx, dy,
        _cfg_float('vigem_net_sensitivity', 1.0),
        max(0.0, min(0.5, _cfg_float('vigem_net_input_deadzone', 0.0))),
        max(0.0, min(0.5, _cfg_float('vigem_net_game_deadzone', 0.0))),
    )
    vigem_net_client.enqueue_move(rx, ry)


def send_mouse_click_vigem_network(duration: float = 0.05) -> bool:
    if vigem_net_client is None or not vigem_net_client.is_connected():
        return False
    vigem_net_client.enqueue_click(_MAX_TRIGGER)
    time.sleep(duration)
    vigem_net_client.enqueue_click(0)
    return True


def send_vigem_network_button(down: bool) -> bool:
    if vigem_net_client is None or not vigem_net_client.is_connected():
        return False
    vigem_net_client.enqueue_click(_MAX_TRIGGER if down else 0)
    return True
