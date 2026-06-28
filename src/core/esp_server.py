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

Access is LAN-wide (binds 0.0.0.0) and guarded by a per-session token appended to
the URL; the WS handshake rejects a missing/wrong token.
"""
from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import secrets
import socket
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from typing import List, Optional, Set
from urllib.parse import urlparse, parse_qs

logger = logging.getLogger(__name__)

_WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

_WEB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web_overlay")

# ---------------------------------------------------------------------------
# Module state
# ---------------------------------------------------------------------------

_config = None
_token: str = ""
_stop = threading.Event()

_http_server: Optional[ThreadingHTTPServer] = None
_http_thread: Optional[threading.Thread] = None

_ws_listener: Optional[socket.socket] = None
_ws_accept_thread: Optional[threading.Thread] = None
_broadcast_thread: Optional[threading.Thread] = None

_clients: Set[socket.socket] = set()
_clients_lock = threading.Lock()


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
    """Full URL (token + ws port) a browser should open to view the overlay."""
    if not _config:
        return ""
    port = int(getattr(_config, "web_esp_http_port", 8080))
    ws_port = int(getattr(_config, "web_esp_ws_port", 8765))
    return f"http://{_lan_ip()}:{port}/?token={_token}&ws={ws_port}"


def is_running() -> bool:
    return _http_thread is not None and _http_thread.is_alive()


# ---------------------------------------------------------------------------
# State snapshot — built entirely from existing Config fields
# ---------------------------------------------------------------------------

def _build_snapshot() -> dict:
    c = _config

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
        "screen": {"w": int(getattr(c, "width", 1920)), "h": int(getattr(c, "height", 1080))},
        "center": {"x": int(getattr(c, "crosshairX", 0)), "y": int(getattr(c, "crosshairY", 0))},
        "settings": {
            "fov_size": int(getattr(c, "fov_size", 200)),
            "detect_range_size": int(getattr(c, "detect_range_size", 320)),
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
            "min_confidence": float(getattr(c, "min_confidence", 0.5)),
        },
        "boxes": _boxes(getattr(c, "latest_boxes", [])),
        "confidences": [float(x) for x in (getattr(c, "latest_confidences", []) or [])],
        "locked_box": locked,
        "locked_decaying": bool(getattr(c, "display_locked_box_is_decaying", False)),
        "active": bool(getattr(c, "AimToggle", False)),
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
        # Token is enforced on the page entry; static assets (js/css) carry no data.
        if path in ("/", "/index.html"):
            token = parse_qs(parsed.query).get("token", [""])[0]
            if token != _token:
                self.send_error(403, "Forbidden")
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
    """Read the client upgrade request, validate token, send 101. Returns success."""
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
    request_line = lines[0] if lines else ""
    # Token check from the request target query string
    try:
        target = request_line.split(" ")[1]
        token = parse_qs(urlparse(target).query).get("token", [""])[0]
    except Exception:
        token = ""
    if token != _token:
        try:
            conn.sendall(b"HTTP/1.1 403 Forbidden\r\n\r\n")
        except Exception:
            pass
        return False
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
    conn.settimeout(None)
    return True


def _accept_loop(port: int):
    global _ws_listener
    try:
        _ws_listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        _ws_listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        _ws_listener.bind(("0.0.0.0", port))
        _ws_listener.listen(8)
        _ws_listener.settimeout(0.5)
    except Exception as exc:
        logger.error("[WebESP] WS listener failed: %s", exc)
        return
    while not _stop.is_set():
        try:
            conn, _addr = _ws_listener.accept()
        except socket.timeout:
            continue
        except OSError:
            break
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
    fps = max(1, int(getattr(_config, "web_esp_fps", 60)))
    interval = 1.0 / fps
    while not _stop.is_set():
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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def start(config) -> bool:
    """Start the Web ESP server (idempotent). Returns True if running afterwards."""
    global _config, _token, _http_thread, _ws_accept_thread, _broadcast_thread
    if is_running():
        return True
    _config = config
    _token = secrets.token_urlsafe(16)
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
    global _http_server, _ws_listener
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
    logger.info("[WebESP] stopped")
