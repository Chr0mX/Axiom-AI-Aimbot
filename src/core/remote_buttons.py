"""Remote mouse-button state, received over GamepadEnhancer's Network API (TCP).

The sender (GamepadEnhancer, "Send to another PC" with "Also send this PC's mouse
buttons") connects here, authenticates with ``auth <token>`` and then sends
``km.mouse(mask)``. Mask bits match ``BUTTON_BITS``: 0 LMB, 1 RMB, 2 MMB, 3 side1,
4 side2. Lines are text, one command per line (max 256 characters); queries answer
``result\\r\\n>>> ``, ``ERR`` means a bad command. The pad commands that also flow
on that link (``km.controller(...)``, ``km.controller_clear()``) are accepted and
ignored, so a sender that is also relaying a virtual pad does not break anything.

A sender that goes away (crash, unplug, closed link) must never leave aim engaged:
the state reads as "nothing held" when the connection ends, on
``km.controller_clear()``, and after ``STALE_AFTER_S`` without a ``km.mouse``.
One client at a time; a new authenticated connection replaces the old one.
"""

from __future__ import annotations

import hmac
import logging
import socket
import threading
import time
from typing import Callable, Optional, Tuple

logger = logging.getLogger(__name__)

PROMPT = "\r\n>>> "
MAX_LINE = 256
STALE_AFTER_S = 3.0
AUTH_TIMEOUT_S = 5.0
IDLE_TIMEOUT_S = 30.0
MASK_ALL = 0x1F
VERSION_TEXT = "Axiom remote buttons 1.0"

BUTTON_BITS = {"lmb": 0x01, "rmb": 0x02, "mmb": 0x04, "side1": 0x08, "side2": 0x10}


def _kill(sock: Optional[socket.socket]) -> None:
    """Close a socket that another thread may be blocked on: shutdown first, or the peer never sees the close."""
    if sock is None:
        return
    try:
        sock.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass
    try:
        sock.close()
    except OSError:
        pass


class _Session:
    """Per-connection protocol state."""

    def __init__(self) -> None:
        self.authed = False
        self.echo = False


class RemoteButtonListener:
    def __init__(self, bind_ip: str, bind_port: int, token: str = "",
                 clock: Callable[[], float] = time.monotonic):
        self._token = token or ""
        self._clock = clock
        self._lock = threading.Lock()
        self._mask = 0
        self._last_rx = float("-inf")
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._current: Optional[socket.socket] = None
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._sock.bind((bind_ip, int(bind_port)))
            self._sock.listen(4)
            self._sock.settimeout(0.5)
        except Exception:
            self._sock.close()
            raise
        self.port = self._sock.getsockname()[1]

    # ---- state -------------------------------------------------------------

    def set_mask(self, mask: int) -> None:
        with self._lock:
            self._mask = mask & MASK_ALL
            self._last_rx = self._clock()

    def clear(self) -> None:
        with self._lock:
            self._mask = 0

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

    # ---- protocol (pure: no sockets) --------------------------------------

    def _token_ok(self, supplied: str) -> bool:
        # An empty configured token never matches: this port must not be open to anyone.
        if not self._token:
            return False
        return hmac.compare_digest(supplied.encode("utf-8"), self._token.encode("utf-8"))

    def handle_line(self, line: str, session: _Session) -> Tuple[Optional[str], bool]:
        """Process one received line. Returns (reply or None, close_connection)."""
        line = line.strip()
        if not line:
            return None, False
        if len(line) > MAX_LINE:
            return "ERR", True

        if not session.authed:
            if line.lower().startswith("auth ") and self._token_ok(line[5:].strip()):
                session.authed = True
                return "OK" + PROMPT, False
            return "ERR auth", True

        open_idx = line.find("(")
        if open_idx < 0 or not line.endswith(")"):
            return "ERR", False
        name = line[:open_idx].strip().lower()
        args = [a.strip().strip("'\"") for a in line[open_idx + 1:-1].split(",")] if line[open_idx + 1:-1].strip() else []

        reply: Optional[str]
        if name == "km.mouse":
            if not args:
                reply = str(self.mask()) + PROMPT
            elif len(args) == 1 and args[0].lstrip("+-").isdigit() and 0 <= int(args[0]) <= MASK_ALL:
                self.set_mask(int(args[0]))
                reply = None
            else:
                return "ERR", False
        elif name == "km.controller_clear":
            self.clear()
            reply = None
        elif name == "km.controller":
            reply = "0" + PROMPT if not args else None        # pad commands from a combined sender: accepted, ignored
        elif name == "km.version":
            reply = VERSION_TEXT + PROMPT
        elif name == "km.api_version":
            reply = "1" + PROMPT
        elif name == "km.echo":
            if not args:
                reply = ("1" if session.echo else "0") + PROMPT
            elif len(args) == 1 and args[0] in ("0", "1"):
                session.echo = args[0] == "1"
                reply = None
            else:
                return "ERR", False
        else:
            return "ERR", False

        if session.echo:
            return line + "\r\n" + (reply if reply is not None else PROMPT.lstrip("\r\n")), False
        return reply, False

    # ---- sockets -----------------------------------------------------------

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._accept_loop, name="RemoteButtons", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        _kill(self._swap_current(None))
        try:
            self._sock.close()
        except OSError:
            pass
        self.clear()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _swap_current(self, new: Optional[socket.socket]) -> Optional[socket.socket]:
        with self._lock:
            old, self._current = self._current, new
        return old

    def _accept_loop(self) -> None:
        while self._running:
            try:
                conn, _ = self._sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            threading.Thread(target=self._serve, args=(conn,), name="RemoteButtonsClient", daemon=True).start()

    def _serve(self, conn: socket.socket) -> None:
        session = _Session()
        buf = b""
        became_current = False
        try:
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            while self._running:
                conn.settimeout(IDLE_TIMEOUT_S if session.authed else AUTH_TIMEOUT_S)
                data = conn.recv(512)
                if not data:
                    break
                buf += data
                while b"\n" in buf:
                    raw, buf = buf.split(b"\n", 1)
                    was_authed = session.authed
                    reply, close = self.handle_line(raw.decode("utf-8", "replace"), session)
                    if session.authed and not was_authed:
                        old = self._swap_current(conn)            # newest client wins
                        became_current = True
                        if old is not None and old is not conn:
                            _kill(old)
                        logger.info("[RemoteButtons] sender connected")
                    if reply is not None:
                        text = reply if reply.endswith("> ") else reply + "\r\n"
                        conn.sendall(text.encode("utf-8"))
                    if close:
                        return
                if len(buf) > MAX_LINE * 2:                          # no newline in a huge chunk: not our protocol
                    return
        except (OSError, socket.timeout):
            pass
        finally:
            try:
                conn.close()
            except OSError:
                pass
            if became_current:
                with self._lock:
                    still_current = self._current is conn
                    if still_current:
                        self._current = None
                if still_current:
                    self.clear()                                    # a dropped sender must never leave aim engaged
                    logger.info("[RemoteButtons] sender disconnected")


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
               int(getattr(config, "remote_btn_bind_port", 5555)),
               str(getattr(config, "remote_btn_token", "")))
        if _listener is not None and _signature == sig:
            return
        _stop_locked()
        if not sig[2]:
            logger.warning("[RemoteButtons] a token is required; set one (and the same on the sender) to listen")
            return
        try:
            _listener = RemoteButtonListener(sig[0], sig[1], sig[2])
            _listener.start()
            _signature = sig
            logger.info("[RemoteButtons] listening on %s:%d (TCP)", sig[0], sig[1])
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
