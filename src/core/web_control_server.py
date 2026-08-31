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

Deliberately does NOT use `from __future__ import annotations`: every
POST route's body model (AlwaysAimBody, ModelChangeBody, etc.) is a
Pydantic BaseModel class defined *locally* inside start() (since
`from pydantic import BaseModel` is itself deferred to inside start(),
per the module-stays-importable-without-fastapi goal above). With
postponed evaluation on, a route handler's `body: ModelChangeBody`
annotation becomes the *string* "ModelChangeBody", and FastAPI resolves
that string via the handler function's `__globals__` (this module's
top-level namespace) — which does not include ModelChangeBody, since
it's a local variable of start(), not a module global. Pydantic's
resolver swallows that failure and leaves the annotation as an
unresolved ForwardRef instead of raising, so FastAPI never recognizes it
as a BaseModel and silently reclassifies the parameter as a required
*query* parameter named "body" — every such route then 422s with
`{"loc": ["query", "body"], "msg": "Field required"}` no matter what the
client POSTs. Without the future import, `body: ModelChangeBody` is
evaluated eagerly at function-definition time through the normal closure
over start()'s locals, so it's already the real class object and needs
no runtime resolution at all — confirmed directly (this bug reproduces
with a bare `typing.get_type_hints()` call on a nested function with a
locally-scoped annotation, no fastapi/pydantic needed to see it).
"""

import logging
import os
import secrets
import socket
import threading
import time
from typing import Literal, Optional

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
        # The Model panel's Backend <select> must track config.inference_backend
        # (the user's selected/persisted backend name) exactly the way
        # model_page.py's own _loadFromConfig() does, NOT the field above —
        # that one prefers the live ONNX EP string (e.g.
        # "TensorrtExecutionProvider") for the plain-text status readout,
        # which never matches any of the select's four option values and
        # was leaving the dropdown stuck on its default "Auto".
        "selected_backend": str(getattr(config, "inference_backend", "auto")),
        "mouse_move_method": str(getattr(config, "mouse_move_method", "")),
        "makcu_connected": makcu_connected,
        "capture_fps": round(cap_fps, 1),
        "inference_fps": round(inf_fps, 1),
        # Stream/source FPS — only meaningful for the three capture backends
        # that read from an external device/stream rather than the desktop
        # itself (uvc/ndi/udp); the client shows a "Stream FPS" stat only
        # for those three, same condition status_panel.py's own
        # source_fps_row uses. Two separate fields rather than one merged
        # value, mirroring esp_server.py's own _build_snapshot() convention:
        # source_nominal_fps is the device's own reported rate (uvc/ndi),
        # while udp_recv_fps is the actual assembled-frames/sec rate from
        # the sender — source_nominal_fps for udp is only the local decode
        # throughput, not the real stream rate (see status_panel.py's own
        # comment on this exact distinction).
        "screenshot_method": str(getattr(config, "screenshot_method", "mss")),
        "source_fps": round(float(getattr(config, "source_nominal_fps", 0.0)), 1),
        "udp_recv_fps": round(float(getattr(config, "udp_recv_fps", 0.0)), 1),
        "udp_dropped_fps": round(float(getattr(config, "udp_dropped_fps", 0.0)), 1),
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

    class ModelNotesBody(BaseModel):
        model: str
        text: str

    class WebEspEnabledBody(BaseModel):
        enabled: bool

    class ConfigNameBody(BaseModel):
        name: str

    class ConfigRenameBody(BaseModel):
        old_name: str
        new_name: str

    class PresetSlotBody(BaseModel):
        index: int
        name: str

    class ConfigImportBody(BaseModel):
        content: str

    class ConvertBody(BaseModel):
        model_path: str
        fp16: bool = True
        workspace_mb: int = 2048

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
        from .app_controller import list_models, get_model_cache_status
        cache_status = get_model_cache_status(config)
        return {"models": list_models(), **cache_status}

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

    @app.post("/api/control/model_restart", dependencies=[Depends(_check_token)])
    def post_model_restart_route(body: ModelChangeBody):
        # The confirmed-restart counterpart to /api/control/model above —
        # only call this after that route has already refused the exact
        # same body with {"reason": "needs_restart"} AND the client's own
        # human has explicitly confirmed the restart (a window.confirm()
        # dialog) — this route does not ask again, it just does it. See
        # confirm_model_change_with_restart()'s own docstring for why this
        # is safe to expose as a route at all (still refuses outright for
        # a genuinely bad model_path/backend).
        from .app_controller import confirm_model_change_with_restart
        return confirm_model_change_with_restart(config, body.model_path, body.inference_backend)

    # -----------------------------------------------------------------
    # Tab settings — generic get/apply covering the Model/Capture/
    # Inference panels' plain Config fields. See web_control_settings.py's
    # module docstring for why this is a separate module from
    # app_controller.py (nothing here is called by a Qt slot).
    # -----------------------------------------------------------------

    @app.get("/api/settings/{tab}", dependencies=[Depends(_check_token)])
    def get_settings(tab: Literal["model", "capture", "inference", "aim", "keys", "visuals", "trigger", "convert"]):
        from .web_control_settings import get_tab_settings
        return get_tab_settings(config, tab)

    @app.post("/api/settings/{tab}", dependencies=[Depends(_check_token)])
    def post_settings(tab: Literal["model", "capture", "inference", "aim", "keys", "visuals", "trigger", "convert"], body: dict):
        from .web_control_settings import apply_tab_settings
        return apply_tab_settings(config, tab, body)

    @app.get("/api/model_info", dependencies=[Depends(_check_token)])
    def get_model_info_route(model: str = ""):
        from .web_control_settings import get_model_info
        return get_model_info(config, model)

    @app.get("/api/model_notes", dependencies=[Depends(_check_token)])
    def get_model_notes_route(model: str = ""):
        from .web_control_settings import get_model_notes
        return {"text": get_model_notes(model)}

    @app.post("/api/model_notes", dependencies=[Depends(_check_token)])
    def post_model_notes_route(body: ModelNotesBody):
        from .web_control_settings import save_model_notes
        ok = save_model_notes(body.model, body.text)
        return {"ok": ok}

    @app.post("/api/control/open_model_folder", dependencies=[Depends(_check_token)])
    def post_open_model_folder():
        from .web_control_settings import open_model_folder
        return {"ok": open_model_folder()}

    @app.get("/api/game_profiles", dependencies=[Depends(_check_token)])
    def get_game_profiles_route():
        from .web_control_settings import get_game_profiles
        return get_game_profiles()

    @app.get("/api/hud_models", dependencies=[Depends(_check_token)])
    def get_hud_models_route():
        from .web_control_settings import get_hud_models
        return {"models": get_hud_models()}

    @app.get("/api/uvc_probe", dependencies=[Depends(_check_token)])
    def get_uvc_probe_route(device: int = 0, method: str = "msmf", width: int = 1920, height: int = 1080):
        from .web_control_settings import probe_uvc
        return probe_uvc(device, method, width, height)

    @app.get("/api/ndi_sources", dependencies=[Depends(_check_token)])
    def get_ndi_sources_route():
        from .web_control_settings import get_ndi_sources
        return get_ndi_sources()

    @app.get("/api/vk_options", dependencies=[Depends(_check_token)])
    def get_vk_options_route():
        from .web_control_settings import list_vk_options
        return {"options": list_vk_options()}

    @app.get("/api/serial_ports", dependencies=[Depends(_check_token)])
    def get_serial_ports_route():
        from .web_control_settings import get_serial_ports
        return get_serial_ports()

    @app.post("/api/control/humanization_reset", dependencies=[Depends(_check_token)])
    def post_humanization_reset_route():
        from .web_control_settings import reset_humanization
        return reset_humanization(config)

    # -----------------------------------------------------------------
    # Visuals — Web ESP overlay start/stop/restart/open. A real service
    # lifecycle action (not a plain Config write), same tier as
    # always_aim/ai_start/makcu_connect — see app_controller.py.
    # -----------------------------------------------------------------

    @app.post("/api/control/web_esp_enabled", dependencies=[Depends(_check_token)])
    def post_web_esp_enabled_route(body: WebEspEnabledBody):
        from .app_controller import set_web_esp_enabled
        running = set_web_esp_enabled(config, body.enabled)
        return {"ok": True, "running": running}

    @app.post("/api/control/web_esp_restart", dependencies=[Depends(_check_token)])
    def post_web_esp_restart_route():
        from .app_controller import restart_web_esp_if_running
        running = restart_web_esp_if_running(config)
        return {"ok": True, "running": running}

    @app.post("/api/control/web_esp_open", dependencies=[Depends(_check_token)])
    def post_web_esp_open_route():
        from .web_control_settings import open_web_esp_in_browser
        return {"ok": open_web_esp_in_browser()}

    # -----------------------------------------------------------------
    # Configs — preset management via core.config_manager.ConfigManager.
    # -----------------------------------------------------------------

    @app.get("/api/configs", dependencies=[Depends(_check_token)])
    def get_configs_route():
        from .web_control_settings import list_config_presets
        return {"presets": list_config_presets()}

    @app.post("/api/configs/save", dependencies=[Depends(_check_token)])
    def post_configs_save_route(body: ConfigNameBody):
        from .web_control_settings import save_config_preset
        return save_config_preset(config, body.name)

    @app.get("/api/configs/preview", dependencies=[Depends(_check_token)])
    def get_configs_preview_route(name: str = ""):
        from .web_control_settings import preview_config_preset
        return preview_config_preset(config, name)

    @app.post("/api/configs/load", dependencies=[Depends(_check_token)])
    def post_configs_load_route(body: ConfigNameBody):
        from .web_control_settings import load_config_preset
        return load_config_preset(config, body.name)

    @app.post("/api/configs/delete", dependencies=[Depends(_check_token)])
    def post_configs_delete_route(body: ConfigNameBody):
        from .web_control_settings import delete_config_preset
        return delete_config_preset(body.name)

    @app.post("/api/configs/rename", dependencies=[Depends(_check_token)])
    def post_configs_rename_route(body: ConfigRenameBody):
        from .web_control_settings import rename_config_preset
        return rename_config_preset(body.old_name, body.new_name)

    @app.get("/api/configs/export", dependencies=[Depends(_check_token)])
    def get_configs_export_route(name: str = ""):
        from .web_control_settings import export_config_preset_content
        return export_config_preset_content(name)

    @app.post("/api/configs/import", dependencies=[Depends(_check_token)])
    def post_configs_import_route(body: ConfigImportBody):
        from .web_control_settings import import_config_preset_content
        return import_config_preset_content(body.content)

    @app.post("/api/control/open_configs_folder", dependencies=[Depends(_check_token)])
    def post_open_configs_folder_route():
        from .web_control_settings import open_configs_folder
        return {"ok": open_configs_folder()}

    # Full config snapshots — the counterpart to the aim-only preset routes
    # just above, backed by a separate ConfigManager(aim_only=False)
    # instance (see web_control_settings.py's own module docstring on the
    # two-manager split). Reuses the exact same body models
    # (ConfigNameBody/ConfigRenameBody/ConfigImportBody) since the request
    # shapes are identical — only which manager the handler delegates to
    # differs, mirroring configs_page.py's own two-_ManagerBox split in the
    # Qt app.
    @app.get("/api/full_configs", dependencies=[Depends(_check_token)])
    def get_full_configs_route():
        from .web_control_settings import list_full_configs
        return {"presets": list_full_configs()}

    @app.post("/api/full_configs/save", dependencies=[Depends(_check_token)])
    def post_full_configs_save_route(body: ConfigNameBody):
        from .web_control_settings import save_full_config
        return save_full_config(config, body.name)

    @app.get("/api/full_configs/preview", dependencies=[Depends(_check_token)])
    def get_full_configs_preview_route(name: str = ""):
        from .web_control_settings import preview_full_config
        return preview_full_config(config, name)

    @app.post("/api/full_configs/load", dependencies=[Depends(_check_token)])
    def post_full_configs_load_route(body: ConfigNameBody):
        from .web_control_settings import load_full_config
        return load_full_config(config, body.name)

    @app.post("/api/full_configs/delete", dependencies=[Depends(_check_token)])
    def post_full_configs_delete_route(body: ConfigNameBody):
        from .web_control_settings import delete_full_config
        return delete_full_config(body.name)

    @app.post("/api/full_configs/rename", dependencies=[Depends(_check_token)])
    def post_full_configs_rename_route(body: ConfigRenameBody):
        from .web_control_settings import rename_full_config
        return rename_full_config(body.old_name, body.new_name)

    @app.get("/api/full_configs/export", dependencies=[Depends(_check_token)])
    def get_full_configs_export_route(name: str = ""):
        from .web_control_settings import export_full_config_content
        return export_full_config_content(name)

    @app.post("/api/full_configs/import", dependencies=[Depends(_check_token)])
    def post_full_configs_import_route(body: ConfigImportBody):
        from .web_control_settings import import_full_config_content
        return import_full_config_content(body.content)

    @app.post("/api/control/open_full_configs_folder", dependencies=[Depends(_check_token)])
    def post_open_full_configs_folder_route():
        from .web_control_settings import open_full_configs_folder
        return {"ok": open_full_configs_folder()}

    # Quick Presets — 5 one-click shortcuts in the web client's sidebar,
    # each independently assignable to a saved aim preset. Assignment is
    # persisted here; the actual load reuses POST /api/configs/load above
    # directly (a bare load with no confirmation step, same route the
    # Configs tab's own Load button calls after its own client-side
    # diff-preview confirm — there's no separate "load a slot" route).
    @app.get("/api/preset_slots", dependencies=[Depends(_check_token)])
    def get_preset_slots_route():
        from .web_control_settings import get_preset_slots
        return {"slots": get_preset_slots(config)}

    @app.post("/api/preset_slots", dependencies=[Depends(_check_token)])
    def post_preset_slots_route(body: PresetSlotBody):
        from .web_control_settings import set_preset_slot
        return set_preset_slot(config, body.index, body.name)

    # -----------------------------------------------------------------
    # TensorRT conversion — the Convert tab's one real action. A build is
    # a genuine 1-5 minute subprocess (see app_controller.start_conversion()'s
    # docstring), so this is start-then-poll, not a single blocking route:
    # POST kicks off the background thread and returns immediately, GET
    # streams the accumulated log via a since-cursor so a client polling
    # every ~1s never re-fetches lines it already has.
    # -----------------------------------------------------------------

    @app.post("/api/control/convert", dependencies=[Depends(_check_token)])
    def post_convert_route(body: ConvertBody):
        from .app_controller import start_conversion
        return start_conversion(config, body.model_path, body.fp16, body.workspace_mb)

    @app.get("/api/convert/status", dependencies=[Depends(_check_token)])
    def get_convert_status_route(since: int = 0):
        from .app_controller import get_conversion_status
        return get_conversion_status(since)

    # ------------------------------------------------------------------
    # Live capture preview (uvc/ndi/udp only — see index.html/app.js's
    # visibility condition) — streams screen_capture.py's own preview
    # frame (the exact same one the Qt CapturePreviewPanel already reads
    # via get_preview_frame()/_preview_lock) to a remote browser as a
    # throttled MJPEG stream. Built for the "1PC remote-config" case: a
    # 2nd PC running Axiom fed by UVC/NDI/UDP, configured from a phone or
    # laptop on the LAN without wanting to remote-desktop into it just to
    # see the capture framing. Adds zero cost to the actual capture/
    # inference pipeline — it only ever reads a frame that already gets
    # written every tick regardless of whether anyone's watching; the
    # only new cost is the JPEG encode below, and even that only runs for
    # as long as a client is actually connected (see
    # _preview_jpeg_frames()'s own docstring for why no separate
    # viewer-count flag is needed for that).
    # ------------------------------------------------------------------
    _PREVIEW_STREAM_FPS = 8.0
    _PREVIEW_JPEG_QUALITY = 60
    _PREVIEW_MAX_EDGE_PX = 640

    def _check_preview_token(preview_token: str = "") -> None:
        # <img src="..."> can't set a custom header, so this one route
        # accepts the token as a query param instead of X-Axiom-Token — a
        # deliberate, scoped exception to every other route's
        # header-only auth (_check_token above), matching the standard
        # pattern for embedding authenticated media in <img>/<video> tags
        # (e.g. pre-signed URLs). Same comparison, just a different
        # source for the value.
        if not token or preview_token != token:
            raise HTTPException(status_code=401, detail="invalid or missing preview_token")

    def _preview_jpeg_frames():
        """Generator body for the MJPEG stream. Starlette's
        StreamingResponse pulls one chunk at a time (running each next()
        call, including this function's own time.sleep(), on a worker
        thread via iterate_in_threadpool() — never blocking the asyncio
        event loop) and stops + closes this generator (raising
        GeneratorExit at the current yield) the instant the client
        disconnects. That lifecycle IS the "only encode while someone's
        watching" cost control on its own — no separate viewer-count
        flag needed; the loop below simply never runs at all with no
        client connected.

        Each concurrent viewer gets its own independent generator/encode
        loop rather than one shared broadcast — an acceptable trade for
        this feature's actual use case (one remote operator configuring
        a single 2nd PC), not worth a proper pub/sub broadcaster the way
        esp_server.py's WebSocket loop is for its own many-viewer case.
        """
        import cv2
        from core import screen_capture

        interval = 1.0 / _PREVIEW_STREAM_FPS
        boundary = b"--frame\r\n"
        try:
            while True:
                t0 = time.monotonic()
                frame = screen_capture.get_preview_frame()
                if frame is not None:
                    try:
                        if frame.ndim == 3 and frame.shape[2] == 4:
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                        h, w = frame.shape[:2]
                        longest = max(h, w)
                        if longest > _PREVIEW_MAX_EDGE_PX:
                            scale = _PREVIEW_MAX_EDGE_PX / float(longest)
                            frame = cv2.resize(
                                frame, (max(1, int(w * scale)), max(1, int(h * scale))))
                        ok, buf = cv2.imencode(
                            ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, _PREVIEW_JPEG_QUALITY])
                        if ok:
                            jpg = buf.tobytes()
                            yield (
                                boundary
                                + b"Content-Type: image/jpeg\r\n"
                                + b"Content-Length: " + str(len(jpg)).encode("ascii") + b"\r\n\r\n"
                                + jpg
                                + b"\r\n"
                            )
                    except Exception:
                        # Never let one bad frame (e.g. a mid-reinit capture
                        # backend swap) kill the whole stream — just skip it
                        # and try again next tick.
                        pass
                remaining = interval - (time.monotonic() - t0)
                if remaining > 0:
                    time.sleep(remaining)
        except GeneratorExit:
            # Client disconnected (tab closed, navigated away, method
            # switched client-side) — nothing to clean up, just stop.
            return

    @app.get("/api/preview_stream", dependencies=[Depends(_check_preview_token)])
    def get_preview_stream_route():
        from fastapi.responses import StreamingResponse
        return StreamingResponse(
            _preview_jpeg_frames(),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

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
