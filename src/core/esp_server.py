"""Web ESP overlay server.

Exposes Axiom's existing detection state (already screen-space) to a browser-based
Canvas renderer, modeled on apexsky's "backend streams state, frontend draws it"
design. The PyQt overlay stays as the on-device renderer / the GUI stays as config;
this server lets any device on the LAN open a browser and view the ESP.

Architecture (all daemon threads, read-only against Config):
  - HTTP server (stdlib) serves the static client from src/web_overlay/.
  - WebSocket server (stdlib RFC 6455, no external dep) accepts browser clients.
  - A fixed-tick broadcaster (~web_esp_fps Hz, latest-state wins) serializes a
    snapshot of Config and pushes it to all connected clients.

Access is LAN-wide (binds 0.0.0.0); no authentication required.
"""
from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import socket
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from typing import List, Optional, Set
from urllib.parse import urlparse

from .ai_loop_utils import get_capture_dimensions, get_effective_detect_range_size

logger = logging.getLogger(__name__)

_WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

_WEB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web_overlay")

# ---------------------------------------------------------------------------
# Module state
# ---------------------------------------------------------------------------

_config = None
_stop = threading.Event()

_http_server: Optional[ThreadingHTTPServer] = None
_http_thread: Optional[threading.Thread] = None

_ws_listener: Optional[socket.socket] = None
_ws_accept_thread: Optional[threading.Thread] = None
_broadcast_thread: Optional[threading.Thread] = None

_clients: Set[socket.socket] = set()
_clients_lock = threading.Lock()

# Per-client send deadline for the shared broadcast thread. Generous relative
# to a broadcast tick (typically 16 ms at 60 Hz) so an ordinary scheduling
# hiccup never drops a healthy client, but short enough that one wedged
# client can't hold the whole feed hostage. A client that can't absorb a
# state snapshot within this window is not keeping up with the stream
# anyway.
_BROADCAST_SEND_TIMEOUT = 2.0

_actual_ws_port: int = 0

_capture_fps: float = 0.0
_inference_fps: float = 0.0
_prev_capture_count: int = 0
_prev_inference_count: int = 0
_fps_last_t: float = 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _lan_ip() -> str:
    """Best-effort LAN IP via the UDP-socket trick (no traffic is actually sent)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


def connect_url() -> str:
    """URL a browser should open to view the overlay."""
    if not _config:
        return ""
    port = int(getattr(_config, "web_esp_http_port", 8080))
    ws_port = _actual_ws_port or int(getattr(_config, "web_esp_ws_port", 8765))
    return f"http://{_lan_ip()}:{port}/?ws={ws_port}"


def is_running() -> bool:
    return _http_thread is not None and _http_thread.is_alive()


# ---------------------------------------------------------------------------
# State snapshot — built entirely from existing Config fields
# ---------------------------------------------------------------------------

def _build_snapshot() -> dict:
    c = _config

    # Same dimension logic ai_loop.py uses to size the detection region —
    # for 'uvc'/'ndi'/'udp' backends this is the actual live capture
    # resolution, not necessarily the full desktop.
    #
    # 'udp' used to be special-cased here: on the theory that a UDP stream
    # is a small crop out of the user's real desktop, boxes/center were
    # shifted by an assumed "crop centered on the desktop" offset so they'd
    # land at real desktop coordinates — under the assumption the user is
    # looking at their actual full desktop directly. That assumption doesn't
    # hold: overlay.py's in-game PyQt overlay explicitly skips drawing for
    # 'uvc'/'ndi'/'udp' (see its paintEvent), so the Web ESP is the *only*
    # overlay renderer for a UDP setup — there's no "look at your real
    # desktop and see boxes there" path to align against. What the user
    # actually looks at is whatever their OBS scene shows, and the common
    # case scales the small crop up to fill the view rather than showing it
    # at native size centered on a black canvas — the old offset model
    # placed boxes for a framing nobody was looking at, landing them nowhere
    # near the actual (scaled-up) target. Treating 'udp' exactly like
    # 'uvc'/'ndi' — screen == the live capture frame's own resolution, no
    # offset — aligns correctly regardless of how the user scales that frame
    # in OBS, as long as the Web ESP's own Browser Source is scaled/
    # positioned the same way in the scene (both undergo the same OBS
    # transform, so the client's own contain-fit scaling lines up with it).
    cap_w, cap_h = get_capture_dimensions(c)
    screen_w, screen_h = cap_w, cap_h

    def _boxes(raw) -> List[List[int]]:
        out = []
        for b in (raw or []):
            try:
                out.append([int(b[0]), int(b[1]), int(b[2]), int(b[3])])
            except (TypeError, ValueError, IndexError):
                continue
        return out

    locked = getattr(c, "display_locked_box", None)
    try:
        locked = [int(locked[0]), int(locked[1]), int(locked[2]), int(locked[3])] if locked else None
    except (TypeError, ValueError, IndexError):
        locked = None

    return {
        "t": int(time.monotonic() * 1000),
        "screen": {"w": screen_w, "h": screen_h},
        "center": {
            "x": int(getattr(c, "crosshairX", 0)),
            "y": int(getattr(c, "crosshairY", 0)),
        },
        "settings": {
            "fov_size": int(getattr(c, "fov_size", 200)),
            # Effective size (clamped to the active capture method's own live
            # dimensions), not the raw config field — for 'uvc'/'ndi'/'udp',
            # the raw field is only validated against the full desktop
            # height and can hold a value far bigger than the actual live
            # capture frame (e.g. a small UDP crop), which would otherwise
            # draw a detect-range box that dwarfs the frame it's supposed to
            # outline. See get_effective_detect_range_size()'s docstring.
            "detect_range_size": get_effective_detect_range_size(c, (cap_w, cap_h)),
            "fov_circle_filter_enabled": bool(getattr(c, "fov_circle_filter_enabled", False)),
            "show_fov": bool(getattr(c, "show_fov", True)),
            "show_boxes": bool(getattr(c, "show_boxes", True)),
            "box_full_rect": bool(getattr(c, "box_full_rect", False)),
            "show_detect_range": bool(getattr(c, "show_detect_range", True)),
            "show_confidence": bool(getattr(c, "show_confidence", True)),
            "show_tracer_line": bool(getattr(c, "show_tracer_line", True)),
            "show_crosshair": bool(getattr(c, "show_crosshair", False)),
            "crosshair_style": str(getattr(c, "crosshair_style", "dot")),
            "crosshair_size": int(getattr(c, "crosshair_size", 4)),
            "crosshair_color": [
                int(getattr(c, "crosshair_color_r", 255)),
                int(getattr(c, "crosshair_color_g", 255)),
                int(getattr(c, "crosshair_color_b", 255)),
            ],
            "box_color_theme": str(getattr(c, "box_color_theme", "default")),
            "chroma_box_speed": float(getattr(c, "chroma_box_speed", 1.0)),
            "aim_part": str(getattr(c, "aim_part", "head")),
            "head_height_ratio": float(getattr(c, "head_height_ratio", 0.26)),
            "aim_custom_y_pct": float(getattr(c, "aim_custom_y_pct", 30.0)),
            "min_confidence": float(getattr(c, "min_confidence", 0.5)),
        },
        # Unreduced by single_target_mode — matches what the in-game overlay draws,
        # so Web ESP always shows every detection regardless of aiming mode.
        "boxes": _boxes(getattr(c, "latest_all_boxes", [])),
        "confidences": [float(x) for x in (getattr(c, "latest_all_confidences", []) or [])],
        "locked_box": locked,
        "locked_decaying": bool(getattr(c, "display_locked_box_is_decaying", False)),
        "active": bool(getattr(c, "AimToggle", False)),
        "aim_firing": bool(getattr(c, "makcu_aim_active", False)),
        "model": os.path.basename(getattr(c, "model_path", "") or ""),
        "screenshot_method": str(getattr(c, "screenshot_method", "dxcam")),
        "source_fps": float(getattr(c, "source_nominal_fps", 0.0)),
        "udp_recv_fps": round(float(getattr(c, "udp_recv_fps", 0.0)), 1),
        "capture_fps": round(_capture_fps, 1),
        "inference_fps": round(_inference_fps, 1),
    }


# ---------------------------------------------------------------------------
# HTTP server (static client)
# ---------------------------------------------------------------------------

class _Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=_WEB_DIR, **kwargs)

    def log_message(self, *args):  # silence default stderr logging
        pass

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/ping":
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            return
        if path in ("/", "/index.html"):
            if "ws=" not in (parsed.query or ""):
                port = _actual_ws_port or int(getattr(_config, "web_esp_ws_port", 8765))
                self.send_response(302)
                self.send_header("Location", f"/?ws={port}")
                self.end_headers()
                return
            self.path = "/index.html"
        return super().do_GET()


def _serve_http(port: int):
    global _http_server
    try:
        _http_server = ThreadingHTTPServer(("0.0.0.0", port), _Handler)
        _http_server.serve_forever(poll_interval=0.5)
    except Exception as exc:
        logger.error("[WebESP] HTTP server stopped: %s", exc)


# ---------------------------------------------------------------------------
# WebSocket server (stdlib RFC 6455)
# ---------------------------------------------------------------------------

def _ws_encode_text(payload: str) -> bytes:
    data = payload.encode("utf-8")
    n = len(data)
    header = bytearray([0x81])  # FIN + text opcode
    if n < 126:
        header.append(n)
    elif n < 65536:
        header.append(126)
        header += n.to_bytes(2, "big")
    else:
        header.append(127)
        header += n.to_bytes(8, "big")
    return bytes(header) + data


def _ws_handshake(conn: socket.socket) -> bool:
    """Read the client upgrade request and send 101. Returns success."""
    conn.settimeout(5.0)
    request = b""
    while b"\r\n\r\n" not in request:
        chunk = conn.recv(1024)
        if not chunk:
            return False
        request += chunk
        if len(request) > 8192:
            return False
    text = request.decode("latin-1", errors="ignore")
    lines = text.split("\r\n")
    key = ""
    for line in lines[1:]:
        if line.lower().startswith("sec-websocket-key:"):
            key = line.split(":", 1)[1].strip()
            break
    if not key:
        return False
    accept = base64.b64encode(hashlib.sha1((key + _WS_GUID).encode()).digest()).decode()
    resp = (
        "HTTP/1.1 101 Switching Protocols\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        f"Sec-WebSocket-Accept: {accept}\r\n\r\n"
    )
    try:
        conn.sendall(resp.encode())
    except Exception:
        return False
    # Deliberately NOT settimeout(None). The broadcast loop sends to every
    # client from a single thread, so a blocking sendall to one stalled
    # client blocks the feed for all of them — and "stalled" is an ordinary
    # state, not a failure: a backgrounded browser tab stops draining its
    # TCP window within seconds. With a timeout the slow client trips
    # _BROADCAST_SEND_TIMEOUT, gets dropped, and everyone else keeps their
    # frame rate.
    conn.settimeout(_BROADCAST_SEND_TIMEOUT)
    return True


def _accept_loop(port: int):
    global _ws_listener, _actual_ws_port
    for attempt in range(10):
        try:
            _ws_listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            _ws_listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            _ws_listener.bind(("0.0.0.0", port + attempt))
            _ws_listener.listen(8)
            _ws_listener.settimeout(0.5)
            _actual_ws_port = port + attempt
            if attempt:
                logger.warning("[WebESP] WS port %d in use, using %d instead", port, _actual_ws_port)
            break
        except Exception as exc:
            try:
                _ws_listener.close()
            except Exception:
                pass
            _ws_listener = None
            if attempt == 9:
                logger.error("[WebESP] WS listener failed after 10 attempts: %s", exc)
                return
    while not _stop.is_set():
        listener = _ws_listener
        if listener is None:
            break
        try:
            conn, _addr = listener.accept()
        except socket.timeout:
            continue
        except OSError:
            break
        except (AttributeError, TypeError):
            # _ws_listener was concurrently set to None by stop() mid-iteration.
            break
        try:
            # Every broadcast frame is a small, latency-sensitive write —
            # Nagle's algorithm (on by default) can hold each one back up to
            # ~40ms waiting to coalesce with more data or an ACK. That's
            # invisible to the client's own draw-loop FPS counter but reads
            # as overlay lag, so disable it for this connection.
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except Exception:
            pass
        try:
            if _ws_handshake(conn):
                with _clients_lock:
                    _clients.add(conn)
                logger.info("[WebESP] client connected (%d total)", len(_clients))
            else:
                conn.close()
        except Exception:
            try:
                conn.close()
            except Exception:
                pass


def _broadcast_loop():
    while not _stop.is_set():
        # Re-read every tick rather than once at thread start: web_esp_fps is
        # a live GUI setting like every other, and caching it here meant
        # changing it silently did nothing until the server was restarted.
        fps = max(1, int(getattr(_config, "web_esp_fps", 60)))
        interval = 1.0 / fps
        start = time.monotonic()
        with _clients_lock:
            targets = list(_clients)
        if targets:
            try:
                frame = _ws_encode_text(json.dumps(_build_snapshot(), separators=(",", ":")))
            except Exception:
                frame = None
            if frame is not None:
                dead = []
                for conn in targets:
                    try:
                        conn.sendall(frame)
                    except socket.timeout:
                        # Client isn't draining its TCP window (backgrounded
                        # tab, wedged connection). Drop it rather than let it
                        # throttle the broadcast for everyone else — the
                        # client can simply reconnect.
                        logger.info("[WebESP] dropping client that stalled for >%.0fs", _BROADCAST_SEND_TIMEOUT)
                        dead.append(conn)
                    except Exception:
                        dead.append(conn)
                if dead:
                    with _clients_lock:
                        for conn in dead:
                            _clients.discard(conn)
                            try:
                                conn.close()
                            except Exception:
                                pass
        elapsed = time.monotonic() - start
        _stop.wait(max(0.0, interval - elapsed))

        # FPS computation (~1 Hz, mirrors StatusPanel logic)
        global _capture_fps, _inference_fps, _prev_capture_count, _prev_inference_count, _fps_last_t
        now_t = time.monotonic()
        if _fps_last_t == 0.0:
            _fps_last_t = now_t
            _prev_capture_count = int(getattr(_config, "screenshot_frame_count", 0))
            _prev_inference_count = int(getattr(_config, "detection_frame_count", 0))
        elif now_t - _fps_last_t >= 1.0:
            cap_c = int(getattr(_config, "screenshot_frame_count", 0))
            inf_c = int(getattr(_config, "detection_frame_count", 0))
            dt = now_t - _fps_last_t
            _capture_fps = max(0.0, (cap_c - _prev_capture_count) / dt)
            _inference_fps = max(0.0, (inf_c - _prev_inference_count) / dt)
            _prev_capture_count = cap_c
            _prev_inference_count = inf_c
            _fps_last_t = now_t


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def start(config) -> bool:
    """Start the Web ESP server (idempotent). Returns True if running afterwards."""
    global _config, _actual_ws_port, _http_thread, _ws_accept_thread, _broadcast_thread
    if is_running():
        return True
    if _http_thread is not None or _ws_accept_thread is not None or _broadcast_thread is not None:
        # is_running() only reflects the HTTP thread — if a previous start()
        # bound the HTTP port successfully but it later died (or the HTTP bind
        # itself failed) while the WS-accept/broadcast threads and the bound WS
        # listener socket kept running, they'd otherwise leak: a retried
        # start() would spawn a brand new thread set on top of the orphans.
        stop()
    _config = config
    _stop.clear()

    http_port = int(getattr(config, "web_esp_http_port", 8080))
    ws_port = int(getattr(config, "web_esp_ws_port", 8765))

    _http_thread = threading.Thread(target=_serve_http, args=(http_port,), name="webesp-http", daemon=True)
    _http_thread.start()
    _ws_accept_thread = threading.Thread(target=_accept_loop, args=(ws_port,), name="webesp-ws", daemon=True)
    _ws_accept_thread.start()
    _broadcast_thread = threading.Thread(target=_broadcast_loop, name="webesp-cast", daemon=True)
    _broadcast_thread.start()

    logger.info("[WebESP] started — open %s", connect_url())
    return True


def stop():
    """Stop the server and close all client connections."""
    global _http_server, _ws_listener, _actual_ws_port
    global _http_thread, _ws_accept_thread, _broadcast_thread
    global _capture_fps, _inference_fps, _prev_capture_count, _prev_inference_count, _fps_last_t
    _actual_ws_port = 0
    _capture_fps = _inference_fps = 0.0
    _prev_capture_count = _prev_inference_count = 0
    _fps_last_t = 0.0
    _stop.set()
    if _http_server is not None:
        try:
            _http_server.shutdown()
        except Exception:
            pass
        _http_server = None
    if _ws_listener is not None:
        try:
            _ws_listener.close()
        except Exception:
            pass
        _ws_listener = None
    with _clients_lock:
        for conn in _clients:
            try:
                conn.close()
            except Exception:
                pass
        _clients.clear()
    for thread in (_http_thread, _ws_accept_thread, _broadcast_thread):
        if thread is not None:
            thread.join(timeout=2.0)
    _http_thread = _ws_accept_thread = _broadcast_thread = None
    logger.info("[WebESP] stopped")
