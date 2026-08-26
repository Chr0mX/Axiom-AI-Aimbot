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
    "aim": {
        "aim_part": {"type": "choice", "choices": ["head", "body", "center", "custom"]},
        # mouse_click_method isn't set generically here when mouse_move_method
        # becomes "makcu" — that coupling (aim_page.py's _onMouseMoveChanged)
        # is left to the client, same precedent as ndi_source_name/
        # ndi_force_reconnect: it already has both values in hand, so no
        # server-side "decide" logic is needed for a plain two-key POST.
        "mouse_move_method": {"type": "choice", "choices": ["ddxoft", "mouse_event", "sendinput", "arduino", "makcu", "xbox"]},
        "mouse_click_method": {"type": "str"},
        "pid_unsafe_mode": {"type": "bool"},
        # Effective safe range is 0.0-0.5 unless pid_unsafe_mode is on (up to
        # 1.0) — the GUI enforces this by capping slider travel and clamping
        # pid_kp_x/y down to 0.5 when Unsafe Mode is turned off. This schema
        # always allows the full 0.0-1.0 range; the clamp-on-disable is done
        # client-side (same reasoning as the mouse_click_method note above).
        "pid_kp_x": {"type": "float", "min": 0.0, "max": 1.0},
        "pid_ki_x": {"type": "float", "min": 0.0, "max": 1.0},
        "pid_kd_x": {"type": "float", "min": 0.0, "max": 1.0},
        "pid_kp_y": {"type": "float", "min": 0.0, "max": 1.0},
        "pid_ki_y": {"type": "float", "min": 0.0, "max": 1.0},
        "pid_kd_y": {"type": "float", "min": 0.0, "max": 1.0},
        "max_move_per_frame_px": {"type": "float", "min": 0, "max": 500},
        "aim_y_reduce_enabled": {"type": "bool"},
        "aim_y_reduce_delay": {"type": "float", "min": 0.0, "max": 5.0},
        "aim_y_reduce_floor": {"type": "float", "min": 0.0, "max": 1.0},
        "aim_y_reduce_ramp": {"type": "float", "min": 0.0, "max": 2.0},
        "aim_y_reduce_settle_px": {"type": "float", "min": 0, "max": 50},
        "aim_y_vel_restore_px_s": {"type": "float", "min": 0, "max": 500},
        # Humanization fields live on config.humanization (its own
        # HumanizationConfig dataclass), reached via the dotted-path
        # mechanism in _get_nested()/_set_nested() above.
        "humanization.enabled": {"type": "bool"},
        "humanization.intensity": {"type": "float", "min": 0.0, "max": 1.0},
        "humanization.micro_jitter_enabled": {"type": "bool"},
        "humanization.micro_jitter_base": {"type": "float", "min": 0.0, "max": 2.0},
        "humanization.micro_jitter_scale": {"type": "float", "min": 0.0, "max": 0.2},
        "humanization.micro_jitter_idle_enabled": {"type": "bool"},
        "humanization.motion_variation_enabled": {"type": "bool"},
        "humanization.motion_variation_range": {"type": "float", "min": 0.0, "max": 0.2},
        "humanization.speed_shaping_enabled": {"type": "bool"},
        "humanization.speed_shaping_low": {"type": "float", "min": 0.0, "max": 20.0},
        "humanization.speed_shaping_high": {"type": "float", "min": 0.0, "max": 50.0},
        "humanization.speed_shaping_low_factor": {"type": "float", "min": 0.0, "max": 1.0},
        "humanization.micro_stutter_enabled": {"type": "bool"},
        "humanization.micro_stutter_prob": {"type": "float", "min": 0.0, "max": 0.2},
        "humanization.micro_stutter_min": {"type": "float", "min": 0.0, "max": 1.0},
        "humanization.micro_stutter_max": {"type": "float", "min": 0.0, "max": 1.0},
        "humanization.reaction_variability_enabled": {"type": "bool"},
        "humanization.reaction_skip_prob": {"type": "float", "min": 0.0, "max": 0.1},
        "target_priority_mode": {"type": "choice", "choices": ["distance", "confidence", "composite"]},
        "target_priority_confidence_weight": {"type": "float", "min": 0.0, "max": 1.0},
        "prediction_enabled": {"type": "bool"},
        "prediction_horizon_ms": {"type": "float", "min": 5, "max": 50},
        "prediction_max_velocity": {"type": "float", "min": 300, "max": 3000},
        "prediction_history_len": {"type": "int", "min": 2, "max": 6},
        "sticky_lock_enabled": {"type": "bool"},
        "lock_decay_frames": {"type": "int", "min": 3, "max": 60},
        "lock_iou_threshold": {"type": "float", "min": 0.10, "max": 0.70},
        "kalman_enabled": {"type": "bool"},
        "kalman_process_noise": {"type": "float", "min": 0.01, "max": 1.0},
        "kalman_measurement_noise": {"type": "float", "min": 0.01, "max": 1.0},
        "cam_motion_comp_enabled": {"type": "bool"},
        "cam_motion_comp_size": {"type": "choice", "choices": [128, 256]},
        "aim_custom_y_pct": {"type": "float", "min": 0, "max": 100},
        "head_width_ratio": {"type": "float", "min": 0.10, "max": 1.0},
        "head_height_ratio": {"type": "float", "min": 0.10, "max": 1.0},
        "body_width_ratio": {"type": "float", "min": 0.10, "max": 1.0},
        "aim_adaptive_ratio_enabled": {"type": "bool"},
        "aim_adaptive_ratio_ref_h": {"type": "float", "min": 20, "max": 200},
        "aim_posture_aware_enabled": {"type": "bool"},
        "aim_crouch_aspect_threshold": {"type": "float", "min": 0.80, "max": 2.00},
    },
    "keys": {
        # AimKeys[0]/[1]/[2] double as "Aim Key 1/2/3" (non-MAKCU mode) and
        # "Inference" (MAKCU mode, index 0 only) — same underlying field the
        # GUI itself reuses across both presentations. VK codes are plain
        # ints here (no choice validation against the whole VK table): the
        # web client only offers a <select> populated from
        # GET /api/vk_options, so a normal client can't send an invalid
        # one, and any other int just falls back to a hex label.
        "AimKeys.0": {"type": "int"},
        "AimKeys.1": {"type": "int"},
        "AimKeys.2": {"type": "int"},
        "aim_toggle_key": {"type": "int"},
        "auto_fire_key": {"type": "int"},
        "auto_fire_key2": {"type": "int"},
        "makcu_com_port": {"type": "str"},
        "makcu_baud_rate": {"type": "choice", "choices": [115200, 460800, 1000000, 2000000, 4000000]},
        "makcu_aim_button": {"type": "choice", "choices": ["lmb", "rmb", "off"]},
        "makcu_aim_mode": {"type": "choice", "choices": ["hold", "toggle"]},
        "makcu_disengage_delay": {"type": "float", "min": 0.0, "max": 20.0},
    },
    "visuals": {
        "show_fov": {"type": "bool"},
        "show_boxes": {"type": "bool"},
        "box_full_rect": {"type": "bool"},
        "show_confidence": {"type": "bool"},
        "show_detect_range": {"type": "bool"},
        "show_tracer_line": {"type": "bool"},
        "box_color_theme": {"type": "choice", "choices": ["default", "cyan", "red", "yellow", "white", "purple"]},
        "chroma_box_speed": {"type": "int", "min": 0, "max": 20},
        "show_status_panel": {"type": "bool"},
        "status_panel_show_auto_aim": {"type": "bool"},
        "status_panel_show_model": {"type": "bool"},
        "status_panel_show_mouse_move": {"type": "bool"},
        "status_panel_show_mouse_click": {"type": "bool"},
        "status_panel_show_screenshot_method": {"type": "bool"},
        "status_panel_show_screenshot_fps": {"type": "bool"},
        "status_panel_show_detection_fps": {"type": "bool"},
        "show_crosshair": {"type": "bool"},
        "crosshair_style": {"type": "choice", "choices": ["dot", "cross"]},
        "crosshair_size": {"type": "int", "min": 1, "max": 20},
        "crosshair_color_r": {"type": "int", "min": 0, "max": 255},
        "crosshair_color_g": {"type": "int", "min": 0, "max": 255},
        "crosshair_color_b": {"type": "int", "min": 0, "max": 255},
        "enable_acrylic": {"type": "bool"},
        "acrylic_window_alpha": {"type": "int", "min": 60, "max": 255},
        # web_esp_enabled is deliberately NOT a plain schema field here — a
        # bare Config write wouldn't actually start/stop the real esp_server
        # the way visuals_page.py's own toggle does (see set_web_esp_enabled()
        # in app_controller.py), so it's a dedicated action route instead,
        # same reasoning as always_aim.
        "web_esp_http_port": {"type": "int", "min": 1024, "max": 65535},
        "web_esp_ws_port": {"type": "int", "min": 1024, "max": 65535},
    },
    "trigger": {
        "auto_fire_target_part": {"type": "choice", "choices": ["head", "body", "both"]},
        # always_auto_fire's one-directional coupling (turning it ON also
        # turns idle_detect_enabled off, mirroring trigger_page.py's
        # _onAlwaysAutoFireChanged) is left to the client — same "it
        # already has both values in hand" precedent as ndi_source_name/
        # ndi_force_reconnect, not a server-side decide function.
        "always_auto_fire": {"type": "bool"},
        # A second, independently-validated entry for the same
        # mouse_click_method Config field the "aim" tab's schema already
        # exposes (as a looser, unrestricted "str" — see that entry's own
        # comment) — trigger_page.py's own combo only offers these 6
        # choices, so this tab's POST validates against that narrower list
        # instead. Two schema entries mapping to one flat attribute is
        # harmless: get_tab_settings()/apply_tab_settings() have no
        # cross-tab awareness of each other, exactly as if two separate Qt
        # pages both had a control bound to the same Config field.
        "mouse_click_method": {
            "type": "choice",
            "choices": ["mouse_event", "sendinput", "ddxoft", "arduino", "makcu", "xbox"],
        },
        "auto_fire_delay": {"type": "float", "min": 0.0, "max": 2.0},
        "auto_fire_interval": {"type": "float", "min": 0.01, "max": 1.0},
    },
    "convert": {
        # Only the FP16 default is a plain persisted Config field (pre-
        # populates the toggle, mirroring ConvertPage.setConfig()) — the
        # model to build and the workspace budget are one-shot parameters
        # to the /api/control/convert action below, not persisted settings
        # of their own, matching convert_page.py's own workspaceCombo
        # (never written back to Config either).
        "trt_fp16_enabled": {"type": "bool"},
    },
}

TABS = tuple(_SCHEMA.keys())


def _path_part(obj, part: str):
    """One segment of a dotted field path — a plain attribute name, or an
    all-digit segment addressing a list index (e.g. "AimKeys.0" reaches
    config.AimKeys[0]). Returns None if obj is None or the segment doesn't
    resolve (missing attribute / out-of-range index)."""
    if obj is None:
        return None
    if part.isdigit():
        try:
            return obj[int(part)]
        except (IndexError, TypeError, KeyError):
            return None
    return getattr(obj, part, None)


def _get_nested(config, dotted_field: str):
    """Resolve a "."-separated field path against config — e.g.
    "humanization.intensity" reads config.humanization.intensity,
    "AimKeys.0" reads config.AimKeys[0]. A plain field with no "." behaves
    exactly like a bare getattr(config, field, None) (the loop below runs
    zero iterations for a single-segment path)."""
    parts = dotted_field.split(".")
    obj = config
    for p in parts[:-1]:
        obj = _path_part(obj, p)
        if obj is None:
            return None
    return _path_part(obj, parts[-1])


def _set_nested(config, dotted_field: str, value) -> bool:
    """Write to the same path _get_nested() reads. Returns False (without
    raising) if an intermediate segment doesn't resolve — e.g. writing
    "AimKeys.5" when AimKeys only has 3 elements."""
    parts = dotted_field.split(".")
    obj = config
    for p in parts[:-1]:
        obj = _path_part(obj, p)
        if obj is None:
            return False
    last = parts[-1]
    if last.isdigit():
        try:
            obj[int(last)] = value
            return True
        except (IndexError, TypeError):
            return False
    setattr(obj, last, value)
    return True


def get_tab_settings(config, tab: str) -> dict:
    """Read every field in _SCHEMA[tab] off config, in display units.

    Field names may be dotted/indexed paths (see _get_nested()) for values
    that don't live directly on Config — e.g. "humanization.intensity"
    (Config.humanization is its own HumanizationConfig dataclass) or
    "AimKeys.0" (Config.AimKeys is a list[int]).

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
        raw = _get_nested(config, field)
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

    if tab == "keys":
        # Read-only extras — these three drive the Keys & HW panel's own
        # visibility rules (MAKCU-mode groups vs. Aim/Fire Keys groups;
        # Inference card hidden when keep_detecting; Auto Aim Key/Aim Mode/
        # Disengage Delay hidden when always_aim) but each is *owned* by a
        # different tab/route: mouse_move_method by the "aim" tab's own
        # schema entry, keep_detecting by the "inference" tab's, always_aim
        # by the dedicated POST /api/control/always_aim route (its
        # idle_detect_enabled coupling means it must never be a plain
        # generic-schema field — see app_controller.set_always_aim()).
        result["mouse_move_method"] = getattr(config, "mouse_move_method", "")
        result["keep_detecting"] = bool(getattr(config, "keep_detecting", False))
        result["always_aim"] = bool(getattr(config, "always_aim", False))

    if tab == "visuals":
        # Read-only extras — web_esp_enabled's canonical write is the
        # dedicated action route (see _SCHEMA["visuals"]'s comment above),
        # but the panel still needs to know the live enabled/running state
        # and connect URL to render its Web ESP group correctly.
        result["web_esp_enabled"] = bool(getattr(config, "web_esp_enabled", False))
        try:
            from core import esp_server
            result["web_esp_running"] = esp_server.is_running()
            result["web_esp_url"] = esp_server.connect_url() if result["web_esp_running"] else ""
        except Exception:
            result["web_esp_running"] = False
            result["web_esp_url"] = ""

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
            # Choices are usually strings (e.g. "mss"), matched case-
            # insensitively — but a few fields (cam_motion_comp_size,
            # makcu_baud_rate) have int-valued choices instead, matched by
            # string comparison then stored back as the matched choice's
            # own type, so an int-valued choice list round-trips as int
            # rather than being coerced into a string.
            text = str(value).strip()
            matched = None
            for choice in spec["choices"]:
                if isinstance(choice, str):
                    if text.lower() == choice.lower():
                        matched = choice
                        break
                elif text == str(choice):
                    matched = choice
                    break
            if matched is None:
                return {"ok": False, "reason": "invalid_choice", "field": field}
            to_apply[field] = matched
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
        _set_nested(config, field, coerced)

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


# ---------------------------------------------------------------------------
# Hotkey rebinding — VK-code options for the Keys & HW panel's select-based
# rebind fields. The Qt GUI's KeyBindButton live-captures a keypress (a
# global win32api.GetAsyncKeyState() poll, so it works even while the app is
# unfocused) — that mechanism can't be safely replicated from a remote
# browser: there's no equivalent of global OS-level key polling from a web
# page, and browser key/mouse-button codes (KeyboardEvent.code,
# MouseEvent.button) don't match this app's own VK-code scheme at all,
# so a naively-translated live-captured browser event could silently bind
# the wrong input. A known-good dropdown sacrifices "press to bind" UX for
# correctness instead.
# ---------------------------------------------------------------------------

def list_vk_options() -> list[dict]:
    """Named VK-code options, sourced from win_utils.vk_codes.VK_CODE_MAP —
    a plain pure-Python table (no Qt/win32api import of its own), reused
    directly rather than copied, unlike keys_page.py's separate/parallel
    VK_CODE_TRANSLATION_MAP (a GUI file). Importing win_utils.vk_codes still
    executes win_utils/__init__.py first (Python always runs a package's
    __init__ on any submodule import), which does pull in win32api — fine on
    the real Windows host this runs on, just unusable in a non-Windows
    sandbox, same as every other win_utils-touching function in this
    codebase.

    A synthetic 0 = "None / Unbound" entry is prepended, matching this
    app's own "0 = unbound" convention (not itself a key in VK_CODE_MAP).
    """
    try:
        from win_utils.vk_codes import VK_CODE_MAP
    except Exception:
        VK_CODE_MAP = {}
    options = [{"code": 0, "label": "None / Unbound"}]
    for code, label in VK_CODE_MAP.items():
        options.append({"code": code, "label": label})
    return options


def get_serial_ports() -> dict:
    """Enumerate COM ports, mirroring keys_page.py's/aim_page.py's own live
    serial-port refresh (serial.tools.list_ports.comports()) — used here
    for the MAKCU COM Port select (Arduino's own COM Port isn't exposed;
    the whole Arduino group is out of scope, see CLAUDE.md)."""
    try:
        import serial.tools.list_ports
    except Exception as exc:
        return {"ok": False, "reason": "serial_unavailable", "detail": str(exc)}
    try:
        ports = [p.device for p in serial.tools.list_ports.comports()]
    except Exception:
        ports = []
    return {"ok": True, "ports": ports}


def reset_humanization(config) -> dict:
    """Replace config.humanization with a fresh HumanizationConfig(),
    mirroring aim_page.py's _onHumanizationResetClicked() exactly — reusing
    the real dataclass's own defaults rather than duplicating them here, so
    this can never drift from core/humanization.py's actual defaults."""
    try:
        from core.humanization import HumanizationConfig
    except Exception as exc:
        return {"ok": False, "reason": "humanization_unavailable", "detail": str(exc)}
    config.humanization = HumanizationConfig()
    return {"ok": True}


# ---------------------------------------------------------------------------
# Web ESP overlay — start/stop lives in app_controller.py (mirrors AI start/
# stop and MAKCU connect/disconnect: a real service lifecycle action, not a
# plain Config write). This module only holds the "open in browser" action,
# same tier as open_model_folder() — reads state, no Config mutation.
# ---------------------------------------------------------------------------

def open_web_esp_in_browser() -> bool:
    """Open the running Web ESP overlay's URL in a browser ON THE HOST
    MACHINE running Axiom — not on the remote client — mirroring
    visuals_page.py's _onWebEspOpen(). Returns False if the server isn't
    running."""
    try:
        from core import esp_server
        if not esp_server.is_running():
            return False
        url = esp_server.connect_url()
    except Exception:
        return False
    if not url:
        return False
    import webbrowser
    webbrowser.open(url)
    return True


# ---------------------------------------------------------------------------
# Config presets — mirrors configs_page.py's preset management, via
# core.config_manager.ConfigManager (a pure-Python core module, no Qt/
# onnxruntime — safe to import directly rather than defer). A fresh
# instance is created per call rather than threading the Qt app's real one
# through web_control_server.start(): ConfigManager holds no state beyond
# a directory path, so a throwaway instance pointed at the same "config"
# directory behaves identically — every operation is plain file I/O.
# ---------------------------------------------------------------------------

def _config_manager():
    from core.config_manager import ConfigManager
    return ConfigManager()


def list_config_presets() -> list[str]:
    return _config_manager().get_config_list()


def save_config_preset(config, name: str) -> dict:
    """Mirrors configs_page.py's Create/Save buttons (both call
    ConfigManager.save_config() — Create for a new name, Save to overwrite
    the selected one; the GUI's distinction is just which confirmation
    dialog it shows first, not a different underlying call)."""
    ok = _config_manager().save_config(config, name)
    return {"ok": ok}


def preview_config_preset(config, name: str) -> dict:
    """Dry-run of load_config_preset() — what would change without
    applying it, mirroring the GUI's pre-load diff confirmation."""
    changes = _config_manager().preview_config_changes(config, name)
    if changes is None:
        return {"ok": False, "reason": "read_failed"}
    return {"ok": True, "changes": changes}


def load_config_preset(config, name: str) -> dict:
    ok = _config_manager().load_config(config, name)
    return {"ok": ok}


def delete_config_preset(name: str) -> dict:
    ok = _config_manager().delete_config(name)
    return {"ok": ok}


def rename_config_preset(old_name: str, new_name: str) -> dict:
    ok = _config_manager().rename_config(old_name, new_name)
    return {"ok": ok}


def open_configs_folder() -> bool:
    """Opens the configs directory in the host machine's file explorer —
    same "opens on the host, not the remote browser" caveat as
    open_model_folder()."""
    mgr = _config_manager()
    if not os.path.isdir(mgr.configs_dir):
        return False
    if not hasattr(os, "startfile"):
        return False
    os.startfile(mgr.configs_dir)
    return True


def export_config_preset_content(name: str) -> dict:
    """Returns a preset's raw JSON file content for the browser to
    download — the natural web equivalent of the GUI's Export (which opens
    a save-path dialog on the host, meaningless remotely, since the server
    already has the file — the browser downloading it to the client's own
    machine is actually the more useful direction for a remote operator)."""
    mgr = _config_manager()
    if name not in mgr.get_config_list():
        return {"ok": False, "reason": "not_found"}
    path = os.path.join(mgr.configs_dir, f"{name}.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
    except OSError:
        return {"ok": False, "reason": "read_failed"}
    return {"ok": True, "name": name, "content": content}


def import_config_preset_content(content: str) -> dict:
    """Web equivalent of ConfigManager.import_config(path) — that method
    takes a HOST-side file path, which a browser's file picker can never
    hand to a network request (browsers only ever expose the *content* of
    a locally-picked file, never its path on the picking machine). This
    reimplements the same name-extraction/uniqueness/write logic directly
    against parsed JSON content instead of a path.
    """
    try:
        data = json.loads(content)
    except (json.JSONDecodeError, TypeError, ValueError):
        return {"ok": False, "reason": "invalid_json"}
    if not isinstance(data, dict):
        return {"ok": False, "reason": "invalid_json"}

    from core.config_manager import _sanitize_config_name

    name = _sanitize_config_name(data.get("name", "imported_config")) or "imported_config"

    mgr = _config_manager()
    # Ensure uniqueness — never overwrite an existing preset on import,
    # same as ConfigManager.import_config()'s own _1/_2/... suffix loop.
    original_name = name
    counter = 1
    existing = set(mgr.get_config_list())
    while name in existing:
        name = f"{original_name}_{counter}"
        counter += 1

    data["name"] = name
    path = os.path.join(mgr.configs_dir, f"{name}.json")
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except OSError:
        return {"ok": False, "reason": "write_failed"}
    return {"ok": True, "name": name}
