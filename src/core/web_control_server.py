"""Web Control server — a control-plane LAN API for Axiom's main functions.

Sibling to esp_server.py (Web ESP), not a replacement: esp_server.py is a
read-only telemetry broadcaster with no authentication, which is fine for
passive viewing but not for anything that can mutate live state. This
module is the opposite shape on purpose:
  - REST (FastAPI + uvicorn) for commands, so a malformed/partial request
    body is rejected with a 422 by Pydantic validation before it ever
    reaches Config, instead of needing hand-written validation per route.
  - Every route requires a shared token (config.web_control_token) via the
    `X-Axiom-Token` header — see core/app_controller.py's docstring for why
    a control-plane server can't reuse esp_server's "no auth" posture.
  - Route handlers never mutate Config directly; they call into
    core/app_controller.py's plain functions — the same ones a Qt slot
    calls — so the web client and the Qt GUI are two callers of one shared
    application-logic layer, not two independent implementations.

Runs uvicorn's ASGI server inside this process on its own background
thread (mirrors esp_server.py's "blocking call parked on a daemon thread"
shape, just fronting an asyncio loop instead of a blocking socket loop).
Access is LAN-wide (binds 0.0.0.0, matching esp_server.py's own precedent)
— the token is the actual safeguard, not the bind address.

fastapi/uvicorn are vendored straight into src/python/dependencies/ —
the same tracked-in-git directory main.py already puts on sys.path at
startup (unconditionally, alongside qfluentwidgets/vgamepad/pywin32/
directshow_capture.dll) — rather than installed on demand, so no
separate installer script or first-run step is needed. Every import of
them here is still deferred to inside start() rather than at module top
level, so this module stays importable (and start() can fail gracefully
with a clear log message) on a checkout that predates them being
vendored in, the same way main.py already wraps esp_server.start() in a
try/except so one subsystem's absence never blocks the rest of the app.
"""
from __future__ import annotations

import logging
import os
import secrets
import socket
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)

_WEB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web_control_client")

# ---------------------------------------------------------------------------
# Module state
# ---------------------------------------------------------------------------

_config = None
_server = None            # uvicorn.Server instance, once start() succeeds
_thread: Optional[threading.Thread] = None
_actual_port: int = 0

# Passed in from main.py's start() call so /api/control/ai_start can call
# app_controller.start_ai_threads() with the *same* queue objects
# PyQtOverlay/auto_fire_loop already read — never independently created,
# or a web-started AI thread's output would silently never reach them.
_overlay_boxes_queue = None
_overlay_confidences_queue = None
_auto_fire_boxes_queue = None

# Per-request FPS diffing (capture_fps/inference_fps) — same technique
# esp_server.py's broadcast loop uses, just recomputed lazily against
# whenever /api/status is actually polled instead of on a fixed tick, since
# there is no continuous broadcast loop here (REST is pull, not push).
_fps_lock = threading.Lock()
_fps_state = {"t": 0.0, "cap": 0, "inf": 0, "cap_fps": 0.0, "inf_fps": 0.0}


def _lan_ip() -> str:
    """Best-effort LAN IP via the UDP-socket trick (no traffic is actually sent).

    Deliberately duplicated from esp_server.py's identical helper rather
    than imported — these are two independent network servers, each
    self-contained, matching esp_server.py's own style of not reaching into
    sibling network modules for a five-line helper.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


def connect_url() -> str:
    """URL a web client should point at."""
    if not _config:
        return ""
    port = _actual_port or int(getattr(_config, "web_control_port", 8090))
    return f"http://{_lan_ip()}:{port}/"


def is_running() -> bool:
    return _thread is not None and _thread.is_alive()


# ---------------------------------------------------------------------------
# Status payload — a smaller, control-oriented field set, deliberately not
# a reuse of esp_server._build_snapshot(): this is "what a remote operator
# needs to see to decide what to click," not full detection telemetry.
# ---------------------------------------------------------------------------

def _build_status(config) -> dict:
    now = time.monotonic()
    cap_c = int(getattr(config, "screenshot_frame_count", 0))
    inf_c = int(getattr(config, "detection_frame_count", 0))
    with _fps_lock:
        cap_fps = _fps_state["cap_fps"]
        inf_fps = _fps_state["inf_fps"]
        if _fps_state["t"] == 0.0:
            _fps_state.update(t=now, cap=cap_c, inf=inf_c)
        elif now - _fps_state["t"] >= 1.0:
            dt = now - _fps_state["t"]
            cap_fps = max(0.0, (cap_c - _fps_state["cap"]) / dt)
            inf_fps = max(0.0, (inf_c - _fps_state["inf"]) / dt)
            _fps_state.update(t=now, cap=cap_c, inf=inf_c, cap_fps=cap_fps, inf_fps=inf_fps)

    try:
        from win_utils import is_makcu_connected
        makcu_connected = bool(is_makcu_connected())
    except Exception:
        makcu_connected = False

    return {
        "active": bool(getattr(config, "AimToggle", False)),
        "always_aim": bool(getattr(config, "always_aim", False)),
        "aim_firing": bool(getattr(config, "makcu_aim_active", False)),
        "running": bool(getattr(config, "Running", False)),
        "model": os.path.basename(getattr(config, "model_path", "") or ""),
        "inference_backend": str(getattr(config, "current_provider", "") or getattr(config, "inference_backend", "auto")),
        "mouse_move_method": str(getattr(config, "mouse_move_method", "")),
        "makcu_connected": makcu_connected,
        "makcu_com_port": str(getattr(config, "makcu_com_port", "") or ""),
        "capture_fps": round(cap_fps, 1),
        "inference_fps": round(inf_fps, 1),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def start(
    config,
    overlay_boxes_queue=None,
    overlay_confidences_queue=None,
    auto_fire_boxes_queue=None,
) -> bool:
    """Start the Web Control server (idempotent). Returns True if running afterwards.

    overlay_boxes_queue/overlay_confidences_queue/auto_fire_boxes_queue are
    optional only so this signature doesn't force every caller to have them
    on hand — main.py's real call site always passes the actual queue
    objects it already created (the same ones PyQtOverlay/auto_fire_loop
    read), so /api/control/ai_start can call
    app_controller.start_ai_threads() with them. Without real queues,
    ai_start responds with {"ok": false, "reason": "queues_not_configured"}
    rather than silently creating disconnected ones.
    """
    global _config, _server, _thread, _actual_port
    global _overlay_boxes_queue, _overlay_confidences_queue, _auto_fire_boxes_queue

    if is_running():
        return True
    if _thread is not None:
        # Mirrors esp_server.start()'s own orphan self-healing: a previous
        # start() whose thread died without going through stop() shouldn't
        # leak a stale uvicorn.Server reference under a fresh one.
        stop()

    try:
        import uvicorn
        from fastapi import Depends, FastAPI, Header, HTTPException
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.staticfiles import StaticFiles
        from pydantic import BaseModel
    except ImportError as exc:
        logger.error(
            "[WebControl] fastapi/uvicorn not installed — Web Control server "
            "not started. Vendor them into src\\python\\dependencies\\ (e.g. "
            "src\\python\\python.exe -m pip install --target "
            "src\\python\\dependencies fastapi uvicorn) and restart Axiom to "
            "enable this feature. (%s)", exc,
        )
        return False

    _config = config
    _overlay_boxes_queue = overlay_boxes_queue
    _overlay_confidences_queue = overlay_confidences_queue
    _auto_fire_boxes_queue = auto_fire_boxes_queue

    if not getattr(config, "web_control_token", ""):
        config.web_control_token = secrets.token_urlsafe(24)
        logger.warning(
            "[WebControl] generated a new access token (shown once — copy it "
            "into your web client now): %s", config.web_control_token,
        )
    token = config.web_control_token

    def _check_token(x_axiom_token: str = Header(default="")) -> None:
        if not token or x_axiom_token != token:
            raise HTTPException(status_code=401, detail="invalid or missing X-Axiom-Token")

    class AlwaysAimBody(BaseModel):
        enabled: bool

    class ModelChangeBody(BaseModel):
        model_path: str
        inference_backend: Optional[str] = None

    class _ThreadServer(uvicorn.Server):
        def install_signal_handlers(self) -> None:
            # Overridden to a no-op: uvicorn's default installs SIGINT/
            # SIGTERM handlers, which only works on the main thread — this
            # server always runs on its own background thread (below), where
            # Python raises ValueError attempting that. main.py already owns
            # process-level shutdown (aboutToQuit -> stop()), so nothing is
            # lost by skipping uvicorn's own signal handling here.
            pass

    app = FastAPI(title="Axiom Web Control")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/ping")
    def ping():
        # Unauthenticated on purpose — mirrors esp_server.py's own /ping —
        # just reachability, no state, nothing to protect.
        return {"ok": True}

    @app.get("/api/status", dependencies=[Depends(_check_token)])
    def get_status():
        return _build_status(config)

    @app.get("/api/models", dependencies=[Depends(_check_token)])
    def get_models():
        from .app_controller import list_models
        return {"models": list_models()}

    @app.post("/api/control/always_aim", dependencies=[Depends(_check_token)])
    def post_always_aim(body: AlwaysAimBody):
        from .app_controller import set_always_aim
        set_always_aim(config, body.enabled)
        return {"ok": True, "always_aim": bool(getattr(config, "always_aim", False))}

    @app.post("/api/control/makcu_connect", dependencies=[Depends(_check_token)])
    def post_makcu_connect():
        from .app_controller import connect_makcu

        com_port = str(getattr(config, "makcu_com_port", "") or "")
        if not com_port:
            return {"ok": False, "makcu_connected": False, "reason": "no_port_configured"}

        ok = connect_makcu(config)
        if not ok:
            return {"ok": False, "makcu_connected": False, "reason": "connect_failed"}
        return {"ok": True, "makcu_connected": True, "com_port": com_port}

    @app.post("/api/control/makcu_disconnect", dependencies=[Depends(_check_token)])
    def post_makcu_disconnect():
        from .app_controller import disconnect_makcu
        disconnect_makcu(config)
        return {"ok": True, "makcu_connected": False}

    @app.post("/api/control/ai_start", dependencies=[Depends(_check_token)])
    def post_ai_start():
        from .app_controller import resolve_model_path, start_ai_threads

        if _overlay_boxes_queue is None or _overlay_confidences_queue is None or _auto_fire_boxes_queue is None:
            return {"ok": False, "running": False, "reason": "queues_not_configured"}

        # Always the configured model, same as main.py's own launch-time
        # auto-start — a caller-supplied model_path belongs to
        # POST /api/control/model instead, not this route.
        model_path = str(getattr(config, "model_path", "") or "")
        resolved_path, reason = resolve_model_path(model_path)
        if resolved_path is None:
            return {"ok": False, "running": False, "reason": reason}

        ok = start_ai_threads(
            config, _overlay_boxes_queue, _overlay_confidences_queue, _auto_fire_boxes_queue, model_path,
        )
        if not ok:
            return {"ok": False, "running": False, "reason": "start_failed"}
        return {"ok": True, "running": True, "model": os.path.basename(resolved_path)}

    @app.post("/api/control/ai_stop", dependencies=[Depends(_check_token)])
    def post_ai_stop():
        from .app_controller import stop_ai_threads
        stop_ai_threads(config)
        return {"ok": True, "running": bool(getattr(config, "Running", False))}

    @app.post("/api/control/model", dependencies=[Depends(_check_token)])
    def post_model_change(body: ModelChangeBody):
        from .app_controller import request_model_change
        return request_model_change(config, body.model_path, body.inference_backend)

    if os.path.isdir(_WEB_DIR):
        app.mount("/", StaticFiles(directory=_WEB_DIR, html=True), name="client")
    else:
        logger.warning("[WebControl] client directory not found: %s", _WEB_DIR)

    port = int(getattr(config, "web_control_port", 8090))
    uv_config = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="warning")
    _server = _ThreadServer(uv_config)
    _actual_port = port

    _thread = threading.Thread(target=_server.run, name="webcontrol-http", daemon=True)
    _thread.start()

    # Give uvicorn a brief window to actually bind before declaring success —
    # if the port is taken, its own startup raises inside the thread and the
    # thread exits almost immediately.
    for _ in range(50):
        if not _thread.is_alive() or getattr(_server, "started", False):
            break
        time.sleep(0.05)

    if not _thread.is_alive():
        logger.error("[WebControl] server thread exited immediately — port %d may be in use", port)
        _server = None
        _thread = None
        _actual_port = 0
        return False

    logger.info("[WebControl] started — %s", connect_url())
    return True


def stop() -> None:
    """Stop the server."""
    global _server, _thread, _actual_port
    global _overlay_boxes_queue, _overlay_confidences_queue, _auto_fire_boxes_queue

    if _server is not None:
        try:
            _server.should_exit = True
        except Exception:
            pass
    if _thread is not None:
        _thread.join(timeout=3.0)

    _server = None
    _thread = None
    _actual_port = 0
    _overlay_boxes_queue = None
    _overlay_confidences_queue = None
    _auto_fire_boxes_queue = None
    with _fps_lock:
        _fps_state.update(t=0.0, cap=0, inf=0, cap_fps=0.0, inf_fps=0.0)
    logger.info("[WebControl] stopped")
