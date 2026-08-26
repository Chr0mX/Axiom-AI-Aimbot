"""Web Control settings — generic Config field read/write for the tabs that
mirror a Qt GUI settings page (Model/Capture/Inference so far), plus a
handful of tab-specific reads/actions that aren't plain Config fields.

Deliberately a separate module from app_controller.py: everything in
app_controller.py is a boundary "a Qt slot and a web route both call into"
(see its own module docstring) — nothing here is ever called by a Qt slot,
it exists purely so the Web Control client can offer the same settings the
Qt GUI pages do. Splitting it out keeps that "shared boundary" framing
honest in app_controller.py rather than diluting it with web-only code.

Field names/ranges/descriptions in _SCHEMA below were copied by hand from
reading model_page.py/capture_page.py/inference_page.py — not imported at
runtime. Those are Qt GUI files (PyQt6/qfluentwidgets); importing one here
would drag that whole dependency chain into the web control server, which
has been kept Qt-free since milestone 1 specifically so it stays testable
without Windows deps. Every field name was cross-checked against
Config._FIELD_MAP/__init__ in core/config.py before being added here — a
typo'd setattr() wouldn't raise, it would just silently create a fresh
unpersisted attribute on Config, so getting the name right here matters.

Two generic routes (see web_control_server.py) cover every field in
_SCHEMA instead of one dedicated route per field:
  - get_tab_settings(config, tab)            -> GET  /api/settings/{tab}
  - apply_tab_settings(config, tab, updates)  -> POST /api/settings/{tab}

Every "int"/"float" field's min/max (and its optional display "scale") are
expressed in the same units the GUI's own slider already uses — e.g.
screenshot_interval is stored on Config in seconds, but the GUI slider (and
this schema, via scale=1000) shows/edits it in milliseconds. This keeps the
web client's fields numerically identical to what the Qt page shows,
without needing every value to also be renamed.

Coupled writes the GUI does as a single action (e.g. picking an NDI source
also sets ndi_force_reconnect=True; picking a game profile also sets
hud_roi_coords from game.json) are just two keys in one POST body from the
client's side — it already has both values in hand from a separate fetch
(get_game_profiles()/get_ndi_sources() below) — so no server-side "decide"
logic is needed here the way set_always_aim()/request_model_change() need
it in app_controller.py.
"""
from __future__ import annotations

import glob
import json
import os
import socket

# .../src/core/web_control_settings.py -> .../src/core -> .../src -> repo root.
# Same derivation as app_controller.py's project_root.
_this_dir = os.path.dirname(os.path.abspath(__file__))
_src_dir = os.path.dirname(_this_dir)
project_root = os.path.dirname(_src_dir)


# ---------------------------------------------------------------------------
# Generic tab-settings schema
# ---------------------------------------------------------------------------

_SCHEMA: dict[str, dict[str, dict]] = {
    "model": {
        "hud_game": {"type": "str"},
        "hud_roi_coords": {"type": "str"},
        "hud_model_path": {"type": "str"},
    },
    "capture": {
        "screenshot_method": {"type": "choice", "choices": ["mss", "dxcam", "uvc", "ndi", "udp"]},
        "screenshot_interval": {"type": "float", "scale": 1000.0, "min": 1, "max": 100},
        "uvc_capture_method": {"type": "choice", "choices": ["msmf", "dshow", "any"]},
        "uvc_dshow_backend": {"type": "choice", "choices": ["v1", "v2"]},
        "uvc_ffmpeg_enabled": {"type": "bool"},
        "uvc_device_index": {"type": "int"},
        "uvc_width": {"type": "int"},
        "uvc_height": {"type": "int"},
        "uvc_fps": {"type": "int"},
        "uvc_video_format": {"type": "choice", "choices": ["mjpeg", "yuy2", "nv12", "yuv420p"]},
        "uvc_crop_mode": {"type": "choice", "choices": ["dynamic", "fixed"]},
        "uvc_ffmpeg_path": {"type": "str"},
        "ndi_source_name": {"type": "str"},
        "ndi_force_reconnect": {"type": "bool"},
        "ndi_bandwidth": {"type": "choice", "choices": ["highest", "lowest"]},
        "udp_bind_ip": {"type": "str"},
        "udp_bind_port": {"type": "int", "min": 1, "max": 65535},
        "udp_force_restart": {"type": "bool"},
    },
    "inference": {
        "fov_size": {"type": "int", "min": 50, "max": 500},
        "fov_height": {"type": "int", "min": 50, "max": 500},
        "fov_follow_mouse": {"type": "bool"},
        "fov_circle_filter_enabled": {"type": "bool"},
        "fov_reduce_on_target_enabled": {"type": "bool"},
        "fov_min_size_pct": {"type": "float", "min": 1, "max": 100},
        "fov_min_size_duration": {"type": "float", "min": 0.0, "max": 10.0},
        "detect_range_size": {"type": "int", "min": 100, "max": 1080},
        "detect_interval": {"type": "float", "scale": 1000.0, "min": 1, "max": 100},
        "min_confidence": {"type": "float", "scale": 100.0, "min": 1, "max": 100},
        "detect_semantic_filter_enabled": {"type": "bool"},
        "keep_detecting": {"type": "bool"},
        "idle_detect_enabled": {"type": "bool"},
        "idle_detect_interval": {"type": "float", "scale": 1000.0, "min": 5, "max": 500},
        "single_target_mode": {"type": "bool"},
        "cuda_io_binding_enabled": {"type": "bool"},
        "frame_skip_enabled": {"type": "bool"},
        # The GUI's own slider ticks in raw ints 5-100 internally but labels
        # itself "v/10" (a SliderLabelCard quirk) — expose the human value
        # (0.5-10.0) directly here instead of replicating that internal
        # raw-int convention, since a plain number input has no separate
        # label to carry the /10 translation the GUI's format_func provides.
        "frame_skip_threshold": {"type": "float", "min": 0.5, "max": 10.0},
    },
}

TABS = tuple(_SCHEMA.keys())


def get_tab_settings(config, tab: str) -> dict:
    """Read every field in _SCHEMA[tab] off config, in display units.

    Unknown tab -> {}. capture also gets two extra, non-schema keys
    (system_ip, bind_ip_options) computed from get_local_ips() below, since
    the GUI's Bind IP combo/System IP label are derived facts, not stored
    Config fields.
    """
    schema = _SCHEMA.get(tab)
    if schema is None:
        return {}

    result: dict = {}
    for field, spec in schema.items():
        raw = getattr(config, field, None)
        ftype = spec["type"]
        if raw is None:
            result[field] = None
        elif ftype == "bool":
            result[field] = bool(raw)
        elif ftype in ("int", "float"):
            value = float(raw) * spec.get("scale", 1.0)
            result[field] = int(round(value)) if ftype == "int" else value
        else:  # "str" / "choice"
            result[field] = raw

    if tab == "capture":
        ips = get_local_ips()
        result["system_ip"] = ", ".join(ips) if ips else "—"
        result["bind_ip_options"] = ["0.0.0.0"] + ips
        # Read-only "actual negotiated" readouts — published by the live
        # capture backend itself (UVCCapture.__init__/NDICapture), not
        # user-settable, so they live here as extra keys rather than in
        # _SCHEMA (which apply_tab_settings() would then accept writes to).
        result["uvc_actual_width"] = int(getattr(config, "uvc_actual_width", 0) or 0)
        result["uvc_actual_height"] = int(getattr(config, "uvc_actual_height", 0) or 0)
        result["uvc_actual_fps"] = float(getattr(config, "uvc_actual_fps", 0.0) or 0.0)
        result["ndi_width"] = int(getattr(config, "ndi_width", 0) or 0)
        result["ndi_height"] = int(getattr(config, "ndi_height", 0) or 0)
        result["ndi_source_nominal_fps"] = float(getattr(config, "source_nominal_fps", 0.0) or 0.0)

    return result


def apply_tab_settings(config, tab: str, updates: dict) -> dict:
    """Validate + write a partial {field: display_value} dict.

    Two-pass: every field is validated and coerced before anything is
    written to config, so a bad field anywhere in the body leaves config
    completely untouched instead of partially applied.

    Returns {"ok": False, "reason": ..., "field": ...} on the first
    problem found (unknown_tab / invalid_body / unknown_field /
    invalid_choice / invalid_value), or {"ok": True, "applied": {...}}
    (post-write values, in display units) on success.
    """
    schema = _SCHEMA.get(tab)
    if schema is None:
        return {"ok": False, "reason": "unknown_tab"}
    if not isinstance(updates, dict):
        return {"ok": False, "reason": "invalid_body"}

    to_apply: dict = {}
    for field, value in updates.items():
        spec = schema.get(field)
        if spec is None:
            return {"ok": False, "reason": "unknown_field", "field": field}
        ftype = spec["type"]

        if ftype == "bool":
            to_apply[field] = bool(value)
        elif ftype == "choice":
            text = str(value).strip().lower()
            if text not in spec["choices"]:
                return {"ok": False, "reason": "invalid_choice", "field": field}
            to_apply[field] = text
        elif ftype == "str":
            to_apply[field] = str(value)
        else:  # "int" / "float" — bounds are expressed in display units
            try:
                num = float(value)
            except (TypeError, ValueError):
                return {"ok": False, "reason": "invalid_value", "field": field}
            lo, hi = spec.get("min"), spec.get("max")
            if lo is not None:
                num = max(num, lo)
            if hi is not None:
                num = min(num, hi)
            config_value = num / spec.get("scale", 1.0)
            to_apply[field] = int(round(config_value)) if ftype == "int" else config_value

    for field, coerced in to_apply.items():
        setattr(config, field, coerced)

    return {"ok": True, "applied": get_tab_settings(config, tab)}


# ---------------------------------------------------------------------------
# Model Info — mirrors model_page.py's _ModelInspectWorker
# ---------------------------------------------------------------------------

def get_model_info(config, model_path: str) -> dict:
    """Live-inspect model_path, joining the same "  •  "-separated parts
    string _ModelInspectWorker.run() builds in the GUI.

    Prefers a cached trt_cache/<stem>*.engine over the raw .onnx when
    TensorRT is the active provider, same as _updateModelInfo(). Needs
    onnxruntime (via model_detect.inspect_model) — already a hard
    dependency of the whole app on the host machine, just unusable in a
    sandbox without it.
    """
    from .app_controller import resolve_model_path

    resolved, reason = resolve_model_path(model_path)
    if resolved is None:
        return {"ok": False, "reason": reason}

    inspect_path = resolved
    provider = getattr(config, "current_provider", "")
    if provider == "TensorrtExecutionProvider":
        trt_cache = os.path.join(project_root, "trt_cache")
        if os.path.isdir(trt_cache):
            stem = os.path.splitext(os.path.basename(resolved))[0]
            engine_files = glob.glob(os.path.join(trt_cache, f"{stem}*.engine"))
            if engine_files:
                inspect_path = sorted(engine_files)[-1]

    try:
        from model_detect import inspect_model
    except ImportError as exc:
        return {"ok": False, "reason": "model_detect_unavailable", "detail": str(exc)}

    try:
        info = inspect_model(inspect_path)
    except BaseException as exc:
        # Mirrors the GUI worker's own broad except — inspection failures
        # (corrupt file, unsupported opset, missing TRT bindings) are
        # displayed as the info text itself, not treated as a request error.
        return {"ok": True, "text": str(exc)[:120]}

    parts = []
    if info.get("format"):
        parts.append(info["format"])
    parts.append(f"Input: {info['input_size']}")
    if info.get("num_classes"):
        parts.append(f"Classes: {info['num_classes']}")
    if info.get("precision"):
        parts.append(f"Precision: {info['precision']}")
    if info.get("file_size"):
        parts.append(info["file_size"])
    return {"ok": True, "text": "  •  ".join(parts)}


# ---------------------------------------------------------------------------
# Model Notes — mirrors model_page.py's _load_notes()/_save_notes()/
# _default_template() (same model_info.json, same default text)
# ---------------------------------------------------------------------------

def _notes_path() -> str:
    return os.path.join(project_root, "model_info.json")


def _default_template(model_name: str) -> str:
    return (
        f"### Recommend settings for {model_name}\n"
        "**Game Settings**\n"
        "Enter settings here\n\n"
        "**AI Settings**\n"
        "Enter settings here"
    )


def _load_notes() -> dict:
    p = _notes_path()
    if os.path.exists(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def get_model_notes(model_name: str) -> str:
    """Saved markdown notes for model_name, or the same default template
    the GUI's _ModelNotesCard shows when nothing has been saved yet."""
    if not model_name:
        return ""
    text = _load_notes().get(model_name)
    return text if text else _default_template(model_name)


def save_model_notes(model_name: str, text: str) -> bool:
    if not model_name:
        return False
    notes = _load_notes()
    notes[model_name] = text
    with open(_notes_path(), "w", encoding="utf-8") as f:
        json.dump(notes, f, ensure_ascii=False, indent=2)
    return True


# ---------------------------------------------------------------------------
# Open Model Folder — mirrors model_page.py's _openModelFolder()
# ---------------------------------------------------------------------------

def open_model_folder() -> bool:
    """Open Model/ in the file explorer of the machine running Axiom —
    NOT on the remote web client, same as the GUI button. Returns False if
    Model/ doesn't exist or os.startfile isn't available (non-Windows)."""
    model_dir = os.path.join(project_root, "Model")
    if not os.path.isdir(model_dir):
        return False
    if not hasattr(os, "startfile"):
        return False
    os.startfile(model_dir)
    return True


# ---------------------------------------------------------------------------
# Model HUD Settings — game.json + Model_Hud/ listing
# ---------------------------------------------------------------------------

def get_game_profiles() -> dict:
    """{"games": {name: roi_coords}} from game.json, mirroring
    model_page.py's _loadGameJson()."""
    path = os.path.join(project_root, "game.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {"games": {k: v for k, v in data.items() if isinstance(k, str)}}
    except Exception:
        return {"games": {}}


def get_hud_models() -> list[str]:
    """Sorted .onnx basenames under Model_Hud/ — same pattern as
    app_controller.list_models(), different directory."""
    model_dir = os.path.join(project_root, "Model_Hud")
    if not os.path.exists(model_dir):
        return []
    return sorted(os.path.basename(m) for m in glob.glob(os.path.join(model_dir, "*.onnx")))


# ---------------------------------------------------------------------------
# UVC / NDI device probing — mirrors capture_page.py's _UvcProbeWorker and
# _refreshNdiSources(), combined into single-shot responses
# ---------------------------------------------------------------------------

def probe_uvc(device_index: int, method: str, width: int, height: int) -> dict:
    """One-shot device/resolution/FPS enumeration. Mirrors _UvcProbeWorker,
    which makes these same three calls off the Qt GUI thread so device I/O
    doesn't freeze the UI — here they just run in FastAPI's own threadpool
    (a plain `def` route, not `async def`), same effect."""
    try:
        from core.screen_capture import (
            list_supported_uvc_resolutions,
            list_supported_uvc_fps,
            list_uvc_device_names,
        )
    except Exception as exc:
        return {"ok": False, "reason": "screen_capture_unavailable", "detail": str(exc)}

    try:
        resolutions = list_supported_uvc_resolutions(device_index, method)
    except Exception:
        resolutions = []
    try:
        fps_list = list_supported_uvc_fps(device_index, width, height, method)
    except Exception:
        fps_list = []
    try:
        device_names = list_uvc_device_names()
    except Exception:
        device_names = []

    return {
        "ok": True,
        "resolutions": [[w, h] for w, h in resolutions],
        "fps_list": list(fps_list),
        "device_names": list(device_names),
    }


def get_ndi_sources() -> dict:
    """Mirrors _refreshNdiSources()'s discovery half (not the reconnect
    side-effect — the client sets ndi_source_name/ndi_force_reconnect
    itself via apply_tab_settings once it knows which source to pick)."""
    try:
        from core.screen_capture import list_available_ndi_source_details
    except Exception as exc:
        return {"ok": False, "reason": "screen_capture_unavailable", "detail": str(exc)}
    try:
        sources = list_available_ndi_source_details()
    except Exception:
        sources = []
    return {"ok": True, "sources": sources}


def get_local_ips() -> list[str]:
    """Non-loopback IPv4 addresses for this machine. Reimplements
    capture_page.py's module-level _get_local_ips() (pure socket logic, no
    Qt import) rather than importing that GUI file."""
    ips: list[str] = []
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None):
            if info[0] == socket.AF_INET:
                ip = info[4][0]
                if ip and not ip.startswith("127.") and ip not in ips:
                    ips.append(ip)
    except Exception:
        pass
    if not ips:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                s.connect(("8.8.8.8", 80))
                ip = s.getsockname()[0]
                if ip and not ip.startswith("127."):
                    ips.append(ip)
            finally:
                s.close()
        except Exception:
            pass
    return ips
