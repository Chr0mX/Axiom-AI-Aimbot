# config.py
"""Configuration management module - Type-safe configuration using type hints"""

from __future__ import annotations

import ctypes
import dataclasses
import json
import logging
import os
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

from .humanization import HumanizationConfig


def _get_screen_size() -> tuple[int, int]:
    """Get screen resolution"""
    user32 = ctypes.windll.user32
    user32.SetProcessDPIAware()
    return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)


_MISSING = object()

# One-time app state — persisted to state.json, NOT config.json.
STATE_FIELDS = ('disclaimer_agreed', 'first_run_complete', 'ndi_installer_ran_once')

_TRUE_STRINGS = frozenset({'1', 'true', 'yes', 'on'})
_FALSE_STRINGS = frozenset({'0', 'false', 'no', 'off'})


def _coerce_bool(val: Any) -> bool:
    """Coerce a persisted value to bool, handling string-typed booleans.

    ``bool(val)`` alone is wrong here: ``bool("false")`` is ``True`` (any
    non-empty string is truthy), so a hand-edited, legacy, or imported
    config.json storing a *string* for a bool field (``"false"``, ``"0"``)
    silently flips the value to its opposite on load. Recognized bool-like
    strings are parsed explicitly first; anything else falls back to normal
    Python truthiness, which is only reached for values that were never a
    stringified bool in the first place (e.g. already a real bool/int).
    """
    if isinstance(val, str):
        low = val.strip().lower()
        if low in _TRUE_STRINGS:
            return True
        if low in _FALSE_STRINGS:
            return False
    return bool(val)

# Single source of truth for the grouped (v2) schema: maps each flat Config
# attribute to its dotted path in the nested JSON. Drives to_dict() and from_dict().
# Fields intentionally absent (never persisted): runtime-derived (current_provider),
# auto-detected (model_input_size), constants (latency_stats_alpha), state (see
# STATE_FIELDS), and the specially-handled crosshair color triplet + humanization
# dataclass.
_FIELD_MAP = {
    # --- model ---
    'model_path':                 'model.path',
    'inference_backend':          'model.backend',
    'dml_cpu_fallback':           'model.dml_cpu_fallback',
    'trt_fp16_enabled':           'model.trt_fp16_enabled',

    # --- capture ---
    'screenshot_method':          'capture.screenshot_method',
    'uvc_device_index':           'capture.uvc.device_index',
    'uvc_width':                  'capture.uvc.width',
    'uvc_height':                 'capture.uvc.height',
    'uvc_fps':                    'capture.uvc.fps',
    'uvc_capture_method':         'capture.uvc.capture_method',
    'uvc_dshow_backend':          'capture.uvc.dshow_backend',
    'uvc_ffmpeg_enabled':         'capture.uvc.ffmpeg_enabled',
    'uvc_video_format':           'capture.uvc.video_format',
    'uvc_ffmpeg_path':            'capture.uvc.ffmpeg_path',
    'uvc_crop_mode':              'capture.uvc.crop_mode',
    'ndi_source_name':            'capture.ndi.source_name',
    'ndi_bandwidth':              'capture.ndi.bandwidth',
    'udp_bind_ip':                'capture.udp.bind_ip',
    'udp_bind_port':              'capture.udp.bind_port',
    'udp_recv_buffer_size':       'capture.udp.recv_buffer_size',
    'udp_frame_timeout':          'capture.udp.frame_timeout',
    'uvc_show_window':            'capture.preview.enabled',
    'uvc_always_on_top':          'capture.preview.always_on_top',
    'preview_crop_to_detection':  'capture.preview.crop_to_detection',
    'preview_fps_cap':            'capture.preview.fps_cap',

    # --- aim ---
    'fov_size':                   'aim.fov_size',
    'fov_height':                 'aim.fov_height',
    'detect_range_size':          'aim.detect_range_size',
    'min_confidence':             'aim.min_confidence',
    'aim_part':                   'aim.aim_part',
    'AimKeys':                    'aim.aim_keys',
    'aim_toggle_key':             'aim.aim_toggle_key',
    'AimToggle':                  'aim.aim_toggle',
    'always_aim':                 'aim.always_aim',
    'keep_detecting':             'aim.keep_detecting',
    'single_target_mode':         'aim.single_target_mode',
    'fov_follow_mouse':           'aim.fov_follow_mouse',
    'fov_circle_filter_enabled':  'aim.fov_circle_filter_enabled',
    'fov_reduce_on_target_enabled': 'aim.fov_reduce.enabled',
    'fov_min_size_pct':           'aim.fov_reduce.min_size_pct',
    'fov_min_size_duration':      'aim.fov_reduce.duration',
    'max_move_per_frame_px':      'aim.max_move_per_frame_px',
    'detect_semantic_filter_enabled': 'aim.detect_semantic_filter_enabled',
    'detect_min_bbox_area_px':    'aim.semantic_filter.min_bbox_area_px',
    'detect_min_bbox_short_side_px': 'aim.semantic_filter.min_bbox_short_side_px',
    'detect_min_bbox_max_side_frac': 'aim.semantic_filter.min_bbox_max_side_frac',
    'pid_kp_x':                   'aim.pid.x.kp',
    'pid_ki_x':                   'aim.pid.x.ki',
    'pid_kd_x':                   'aim.pid.x.kd',
    'pid_kp_y':                   'aim.pid.y.kp',
    'pid_ki_y':                   'aim.pid.y.ki',
    'pid_kd_y':                   'aim.pid.y.kd',
    'pid_unsafe_mode':            'aim.pid.unsafe_mode',
    'aim_y_reduce_enabled':       'aim.y_reduce.enabled',
    'aim_y_reduce_delay':         'aim.y_reduce.delay',
    'aim_y_reduce_floor':         'aim.y_reduce.floor',
    'aim_y_reduce_ramp':          'aim.y_reduce.ramp',
    'aim_y_reduce_settle_px':     'aim.y_reduce.settle_px',
    'aim_y_vel_restore_px_s':     'aim.y_reduce.vel_restore_px_s',
    'kalman_enabled':             'aim.kalman.enabled',
    'kalman_process_noise':       'aim.kalman.process_noise',
    'kalman_measurement_noise':   'aim.kalman.measurement_noise',
    'cam_motion_comp_enabled':    'aim.cam_motion_comp.enabled',
    'cam_motion_comp_size':       'aim.cam_motion_comp.size',
    'aim_deadzone_enabled':       'aim.deadzone.enabled',
    'aim_deadzone_min_px':        'aim.deadzone.min_px',
    'aim_deadzone_close_px':      'aim.deadzone.close_px',
    'head_width_ratio':           'aim.target_area.head_width_ratio',
    'head_height_ratio':          'aim.target_area.head_height_ratio',
    'body_width_ratio':           'aim.target_area.body_width_ratio',
    'aim_adaptive_ratio_enabled': 'aim.target_area.adaptive_ratio.enabled',
    'aim_adaptive_ratio_ref_h':   'aim.target_area.adaptive_ratio.ref_h',
    'aim_posture_aware_enabled':  'aim.target_area.posture_aware.enabled',
    'aim_crouch_aspect_threshold':'aim.target_area.posture_aware.crouch_aspect',
    'aim_custom_y_pct':           'aim.target_area.custom_y_pct',

    # --- autofire ---
    'auto_fire_key':              'autofire.key',
    'auto_fire_key2':             'autofire.key2',
    'always_auto_fire':           'autofire.always',
    'auto_fire_delay':            'autofire.delay',
    'auto_fire_interval':         'autofire.interval',
    'auto_fire_target_part':      'autofire.target_part',

    # --- tracking ---
    'prediction_enabled':         'tracking.prediction.enabled',
    'prediction_horizon_ms':      'tracking.prediction.horizon_ms',
    'prediction_max_velocity':    'tracking.prediction.max_velocity',
    'prediction_history_len':     'tracking.prediction.history_len',
    'sticky_lock_enabled':        'tracking.sticky_lock.enabled',
    'lock_decay_frames':          'tracking.sticky_lock.decay_frames',
    'lock_iou_threshold':         'tracking.sticky_lock.iou_threshold',
    'sticky_adaptive_iou':        'tracking.sticky_lock.adaptive_iou',
    'target_priority_mode':       'tracking.target_priority.mode',
    'target_priority_confidence_weight': 'tracking.target_priority.confidence_weight',

    # --- performance ---
    'thread_priority':            'performance.thread_priority',
    'max_queue_size':             'performance.max_queue_size',
    'cuda_io_binding_enabled':    'performance.cuda_io_binding_enabled',
    'detect_interval':            'performance.timing.detect_interval',
    'screenshot_interval':        'performance.timing.capture_interval',
    'idle_detect_interval':       'performance.timing.idle_interval',
    'idle_detect_enabled':        'performance.timing.idle_enabled',
    'frame_skip_enabled':         'performance.frame_skip.enabled',
    'frame_skip_threshold':       'performance.frame_skip.threshold',
    'enable_latency_stats':       'performance.latency_stats.enabled',
    'latency_stats_interval':     'performance.latency_stats.interval',

    # --- display ---
    'show_fov':                   'display.show_fov',
    'show_boxes':                 'display.show_boxes',
    'box_full_rect':              'display.box_full_rect',
    'show_detect_range':          'display.show_detect_range',
    'show_confidence':            'display.show_confidence',
    'show_tracer_line':           'display.show_tracer_line',
    'box_color_theme':            'display.box_color_theme',
    'chroma_box_speed':           'display.chroma_box_speed',
    'show_status_panel':          'display.status_panel.show',
    'status_panel_show_auto_aim': 'display.status_panel.show_auto_aim',
    'status_panel_show_model':    'display.status_panel.show_model',
    'status_panel_show_mouse_move': 'display.status_panel.show_mouse_move',
    'status_panel_show_mouse_click': 'display.status_panel.show_mouse_click',
    'status_panel_show_screenshot_method': 'display.status_panel.show_screenshot_method',
    'status_panel_show_screenshot_fps': 'display.status_panel.show_screenshot_fps',
    'status_panel_show_detection_fps': 'display.status_panel.show_detection_fps',
    'show_crosshair':             'display.crosshair.show',
    'crosshair_style':            'display.crosshair.style',
    'crosshair_size':             'display.crosshair.size',
    'web_esp_enabled':            'web_esp.enabled',
    'web_esp_http_port':          'web_esp.http_port',
    'web_esp_ws_port':            'web_esp.ws_port',
    'web_esp_fps':                'web_esp.fps',
    'web_control_enabled':        'web_control.enabled',
    'web_control_port':           'web_control.port',
    'web_control_token':          'web_control.token',

    # --- hardware ---
    'mouse_move_method':          'hardware.mouse_move_method',
    'mouse_click_method':         'hardware.mouse_click_method',
    'arduino_com_port':           'hardware.devices.arduino.port',
    'arduino_baud_rate':          'hardware.devices.arduino.baud_rate',
    'makcu_com_port':             'hardware.devices.makcu.port',
    'makcu_baud_rate':            'hardware.devices.makcu.baud_rate',
    'makcu_aim_button':           'hardware.makcu.aim_button',
    'makcu_aim_mode':             'hardware.makcu.aim_mode',
    'makcu_disengage_delay':      'hardware.makcu.disengage_delay',
    'xbox_sensitivity':           'hardware.xbox.sensitivity',
    'xbox_deadzone':              'hardware.xbox.deadzone',

    # --- ui ---
    'dark_mode':                  'ui.dark_mode',
    'enable_acrylic':             'ui.enable_acrylic',
    'acrylic_window_alpha':       'ui.acrylic_window_alpha',
    'acrylic_element_alpha':      'ui.acrylic_element_alpha',
    'show_console':               'ui.show_console',

    # --- ocr / 2nd inference ---
    'second_inference_mode':      'ocr.mode',
    'second_inference_fps':       'ocr.fps',
    'hud_model_path':             'ocr.hud_model_path',
    'hud_confidence':             'ocr.hud_confidence',
    'hud_game':                   'ocr.hud_game',
    'hud_roi_coords':             'ocr.hud_roi_coords',
}


def _set_path(d: Dict[str, Any], path: str, value: Any) -> None:
    """Set a nested dict value from a dotted path, creating intermediate dicts."""
    keys = path.split('.')
    node = d
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value


def _get_path(d: Dict[str, Any], path: str) -> Any:
    """Read a nested dict value from a dotted path; returns _MISSING if absent."""
    node = d
    for k in path.split('.'):
        if not isinstance(node, dict) or k not in node:
            return _MISSING
        node = node[k]
    return node


class Config:
    """Main configuration class - All configuration items for Axiom
    
    Contains all configurable parameters, including:
    - Screen and detection area settings
    - Model and inference parameters
    - PID controller parameters
    - Aim and autofire settings
    - Display and performance options
    - Audio hint system
    
    All attributes have type hints to ensure type safety.
    Configuration can be converted between objects and JSON files via to_dict/from_dict methods.
    """
    
    def __init__(self) -> None:
        self.config_version: int = 2

        # Automatically get screen resolution
        self.width, self.height = _get_screen_size()

        self.screenshot_method: str = "dxcam"  # 螢幕截圖方式
        self.uvc_device_index: int = 0
        self.uvc_width: int = self.width
        self.uvc_height: int = self.height
        self.uvc_fps: int = 60
        self.uvc_capture_method: str = "msmf"
        # DirectShow implementation to use when uvc_capture_method == 'dshow':
        #   'v1' = cv2.VideoCapture(CAP_DSHOW) — OpenCV's generic DirectShow
        #     wrapper. No control over the driver's internal buffer/allocator
        #     depth, which is why capture through this path can lag behind
        #     a purpose-built app like OBS on the same device/settings.
        #   'v2' = the native DirectShow-Capture-DLL — owns the filter graph
        #     and allocator directly (see the DirectShow Capture DLL roadmap),
        #     closing that gap. Supports MJPEG and NV12 (uvc_video_format,
        #     restricted to those two in the GUI while v2 is active — see
        #     _resolve_native_dll_pixel_format in screen_capture.py).
        #     Irrelevant/ignored for 'msmf' or 'any'.
        self.uvc_dshow_backend: str = "v1"
        # FFmpeg subprocess capture, only meaningful for capture_method ==
        # 'dshow' + dshow_backend == 'v1' — ffmpeg has no MSMF demuxer on
        # Windows (its capture input is DirectShow only), and v2 already
        # owns the DirectShow graph directly, so routing v2 through an
        # ffmpeg subprocess would defeat the point of using it. Silently
        # ignored outside that one combination.
        self.uvc_ffmpeg_enabled: bool = False
        # Requested pixel/codec format: 'mjpeg' (compressed, highest FPS
        # headroom over USB), 'yuy2' or 'nv12' (raw, uncompressed).
        self.uvc_video_format: str = "mjpeg"
        # uvc_crop_mode applies to every uvc_capture_method:
        #   'dynamic' = the full negotiated frame is captured, Axiom crops
        #     per-frame to the live Detection Range (supports live Detection
        #     Range changes without restarting capture).
        #   'fixed' = the crop rectangle is computed once, centered, at
        #     capture-start (only makes sense centered — matches
        #     FOV-follow-mouse already being forced off for UVC) and reused
        #     every frame; a live Detection Range change needs a capture
        #     restart to take effect. With uvc_ffmpeg_enabled on, the crop
        #     itself also happens inside the ffmpeg subprocess before the
        #     frame is piped back — far less data crosses the subprocess
        #     pipe. Otherwise (in-process cv2/native-DLL capture, no pipe), this only freezes
        #     which region grab() slices — no measurable throughput/CPU
        #     difference from 'dynamic', since the crop already happens
        #     in-process either way.
        # uvc_ffmpeg_path: explicit path to ffmpeg.exe (ffmpeg method only).
        #   Empty = auto-detect (bundled location, then system PATH).
        self.uvc_ffmpeg_path: str = ""
        self.uvc_crop_mode: str = "dynamic"
        self.uvc_show_window: bool = True
        self.uvc_always_on_top: bool = True
        self.preview_crop_to_detection: bool = False
        self.preview_fps_cap: int = 60
        self.ndi_source_name: str = ""
        self.ndi_bandwidth: str = "highest"
        self.ndi_force_reconnect: bool = False
        self.ndi_width: int = self.width
        self.ndi_height: int = self.height
        self.udp_bind_ip: str = "0.0.0.0"
        self.udp_bind_port: int = 5600
        self.udp_recv_buffer_size: int = 65536
        self.udp_frame_timeout: float = 1.0
        self.udp_force_restart: bool = False
        # Actual live stream resolution, updated continuously by
        # UdpCapture._reader_worker from the real decoded frame size — not a
        # user-configured value, since the sender can crop/resize at any
        # time. 0 = not yet probed (no frame received); get_capture_dimensions()
        # falls back to config.width/height in that case.
        self.udp_width: int = 0
        self.udp_height: int = 0
        # Actual negotiated resolution/FPS of the live UVC device, published
        # by UVCCapture.__init__ from its own already-open handle. Lets the
        # GUI show hardware info (capture_page.py's "Query Device") without
        # opening a second competing cv2.VideoCapture to the same device
        # index — most UVC/webcam drivers don't handle two simultaneous open
        # handles gracefully, and a second handle opened while the first is
        # actively streaming (feeding the AI loop) can stall/corrupt frames
        # on the live handle. 0 = UVC not yet initialized as the live backend.
        self.uvc_actual_width: int = 0
        self.uvc_actual_height: int = 0
        self.uvc_actual_fps: float = 0.0
        self.crosshairX: int = self.width // 2
        self.crosshairY: int = self.height // 2

        # Program execution state
        self.Running: bool = True
        self.AimToggle: bool = True
        
        # ONNX model related settings
        self.model_input_size: int = 640
        self.model_path: str = os.path.join('Model', 'ApexLegendsOrbeet_15k.onnx')
        self.current_provider: str = "DmlExecutionProvider"
        self.inference_backend: str = "auto"
        self.thread_priority: str = "high"   # "normal", "above_normal", "high", "time_critical"
        self.ndi_installer_ran_once: bool = False
        # Hybrid computing: Automatically fallback to CPU when operators are not supported by DirectML
        # ONNX Runtime providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
        self.dml_cpu_fallback: bool = True

        # Aiming and display settings
        self.AimKeys: List[int] = [0x01, 0x06, 0x02]  # Left Click + X2 Key + Right Click
        self.fov_size: int = 200    # FOV width (px). Kept as the pre-existing name/meaning.
        self.fov_height: int = 200  # FOV height (px). Equal to fov_size by default -> a square,
                                     # exactly matching pre-rectangle-support behavior. Independent
                                     # from fov_size lets the FOV become a rectangle (or, with
                                     # fov_circle_filter_enabled, an ellipse) instead of forcing a
                                     # square/circle — see ai_loop_utils.filter_boxes_by_fov().

        # Reduce FOV size while actively tracking a locked target, so a
        # random target entering the edge of the (normally larger) FOV can't
        # steal the lock mid-engagement. Shrinks both fov_size and fov_height
        # to fov_min_size_pct% of their configured values (same percentage on
        # each axis, so a rectangular/elliptical FOV keeps its aspect ratio
        # while shrunk) for up to fov_min_size_duration seconds after a
        # target is acquired, then reverts to the full FOV — bounded rather
        # than permanent, so it widens back out to reacquire once the
        # engagement window passes (e.g. current target moved on or died).
        # See filter_boxes_by_fov()'s caller in ai_loop.py.
        self.fov_reduce_on_target_enabled: bool = False
        self.fov_min_size_pct: float = 50.0     # % of fov_size/fov_height to shrink to while active
        self.fov_min_size_duration: float = 1.0  # seconds the shrink holds after acquisition (0 = indefinite while locked)

        # Runtime state — not serialized (see _FIELD_MAP: neither key appears
        # there). The FOV width/height actually in effect THIS frame
        # (fov_size/fov_height, or the shrunk values while the reduce-on-
        # target window is active) — written every frame by ai_loop.py so
        # the in-game overlay, the UVC/NDI/UDP preview's baked-in overlay,
        # and the Web ESP overlay can all draw/reason about the FOV that's
        # actually being used right now, not just the static configured
        # size. Default to fov_size/fov_height themselves so anything
        # reading these before the AI loop's first frame still gets a sane
        # value.
        self.fov_effective_size: float = float(self.fov_size)
        self.fov_effective_height: float = float(self.fov_height)

        # AI detection range (square edge length): Separated from fov_size, but must not be smaller than fov_size, and must not be larger than screen height
        self.detect_range_size: int = 320 # AI 偵測範圍（正方形邊長），獨立於 fov_size，但不得小於 fov_size，且不得大於螢幕高度，預設為螢幕高度（與舊版行為相同）
        self.show_confidence: bool = True # 是否在框上顯示置信度
        self.min_confidence: float = 0.80  # 最小置信度，範圍 0~1
        self.aim_part: str = "head"
        
        # Single target mode
        self.single_target_mode: bool = True  # 啟用單一目標模式（只瞄準置信度最高的目標）
        
        # Disclaimer agreement status
        self.disclaimer_agreed: bool = False 

        # 首次啟動設置精靈
        self.first_run_complete: bool = False 
        
        # 頭部和身體區域占比設定
        self.head_width_ratio: float = 0.38    # 頭部寬度占檢測框寬度的比例
        self.head_height_ratio: float = 0.26   # 頭部高度占檢測框高度的比例
        self.body_width_ratio: float = 0.87    # 身體寬度占檢測框寬度的比例

        # Distance-adaptive head ratio — scales head_height_ratio inversely with box height
        # so the aim point stays on the head at all engagement ranges.
        self.aim_adaptive_ratio_enabled: bool = False
        self.aim_adaptive_ratio_ref_h: float = 80.0  # box height (px) where ratio is nominal

        # Posture-aware targeting — detects crouch/slide/prone via box aspect ratio and
        # falls back to center-mass so the aim doesn't overshoot into empty space.
        self.aim_posture_aware_enabled: bool = False
        self.aim_crouch_aspect_threshold: float = 1.2  # box_w/box_h above which = crouching
        self.aim_custom_y_pct: float = 30.0  # Custom aim Y as % of box height (0=top, 100=bottom)
        
        # PID 控制器參數 (分離 X 和 Y 軸)
        self.pid_kp_x: float = 0.26      # 水平 P: 比例 - 主要影響反應速度
        self.pid_ki_x: float = 0.0       # 水平 I: 積分 - 修正靜態誤差
        self.pid_kd_x: float = 0.12      # 水平 D: 微分 - 抑制抖動與過衝
        self.pid_kp_y: float = 0.26      # 垂直 P: 比例
        self.pid_ki_y: float = 0.0       # 垂直 I: 積分
        self.pid_kd_y: float = 0.08      # 垂直 D: 微分
        # GUI-only convenience flag: nothing downstream clamps Kp, the P
        # sliders just cap their travel at 0.50 (the proven-stable band) by
        # default. Enabling this lets the same slider travel span 0.0-1.0
        # instead. Persisted so the unlocked range survives a restart.
        self.pid_unsafe_mode: bool = False

        # Y軸壓槍速度逐漸歸零
        self.aim_y_reduce_enabled: bool = False   # 是否啟用 Y 軸歸零功能
        self.aim_y_reduce_delay: float = 0.6      # 按下瞄準鍵後多久開始歸零 (秒)
        self.aim_y_reduce_floor: float = 0.0      # Minimum Y multiplier after ramp (0.0=full cut, 1.0=no suppression)
        self.aim_y_reduce_ramp: float = 0.0       # Seconds to ramp from 1.0 → floor (0=instant, backwards compat)
        self.aim_y_reduce_settle_px: float = 0.0  # Skip suppression if |errorY| > this px (0=disabled)
        self.aim_y_vel_restore_px_s: float = 0.0  # Restore full Y if target vy > this px/s (0=disabled)

        # Target priority scoring
        self.target_priority_mode: str = "composite"      # "distance" | "confidence" | "composite"
        self.target_priority_confidence_weight: float = 0.75  # Weight for confidence in composite mode

        # Confidence box color theme
        self.box_color_theme: str = "default"  # "default" | "cyan" | "red" | "yellow" | "white" | "purple"
        self.chroma_box_speed: float = 1.0  # rainbow cycle speed for in-FOV boxes

        # Tracer line from screen center to detected targets
        self.show_tracer_line: bool = True

        # Crosshair overlay
        self.show_crosshair: bool = False
        self.crosshair_style: str = "dot"         # "dot" | "cross"
        self.crosshair_color_r: int = 255
        self.crosshair_color_g: int = 255
        self.crosshair_color_b: int = 255
        self.crosshair_size: int = 4

        # 滑鼠控制方式
        self.mouse_move_method: str = "makcu"  # 滑鼠移動方式
        self.mouse_click_method: str = "mouse_event" # 滑鼠點擊方式
        self.arduino_com_port: str = ""              # Arduino Leonardo COM 埠
        self.makcu_com_port: str = ""                  # MAKCU KM Host COM 埠
        self.makcu_baud_rate: int = 4_000_000        # MAKCU 串列傳輸速率（使用官方 DE AD 幀序列切換至 4 Mbaud）
        self.arduino_baud_rate: int = 115200         # Arduino 串列傳輸速率

        # Xbox 360 虛擬手把設定
        self.xbox_sensitivity: float = 1.0          # 手把靈敏度 (0.1~5.0)
        self.xbox_deadzone: float = 0.05            # 手把死區 (0.0~0.5)

        # 檢測設定
        # 偵測節流：
        # - detect_interval: 進入瞄準/需要即時反應時的間隔
        # - screenshot_interval: 螢幕截圖間隔（獨立於偵測間隔）
        # - idle_detect_interval: 未瞄準但 keep_detecting=True 時的間隔（降低占用）
        self.detect_interval: float = 0.01       # 秒，預設 10ms
        self.screenshot_interval: float = 0.01   # 秒，預設 10ms
        self.idle_detect_interval: float = 0.05  # 秒，預設 50ms
        self.idle_detect_enabled: bool = False     # 是否啟用未瞄準時降低偵測頻率
        self.aim_toggle_key: int = 45       # Insert 鍵
        self.auto_fire_key2: int = 0x04     # 滑鼠中鍵
        
        # 自動開槍
        self.auto_fire_key: int = 0x06           # 滑鼠X2鍵
        self.always_auto_fire: bool = False      # 不按自動開槍鍵也持續自動開槍
        self.auto_fire_delay: float = 0.0        # 無延遲
        self.auto_fire_interval: float = 0.01    # 射擊間隔
        self.auto_fire_target_part: str = "both" # 可選: "head", "body", "both"

        # 保持檢測功能
        self.keep_detecting: bool = True   # 啟用保持檢測
        self.always_aim: bool = False      # 不按瞄準鍵也執行自動瞄準
        self.makcu_aim_button: str = "lmb"   # "lmb", "rmb", or "off"
        self.makcu_aim_mode: str = "hold"    # "hold" = aim while held; "toggle" = click to toggle
        self.makcu_aim_active: bool = False  # runtime state — not serialized
        self.makcu_disengage_delay: float = 0.0  # seconds to keep aiming after releasing aim button (0 = off)
        self.fov_follow_mouse: bool = False # FOV 跟隨鼠標

        # 顯示開關
        self.show_fov: bool = True
        self.show_boxes: bool = True
        self.box_full_rect: bool = False
        self.show_detect_range: bool = True
        self.show_status_panel: bool = True
        self.status_panel_show_auto_aim: bool = True
        self.status_panel_show_model: bool = True
        self.status_panel_show_mouse_move: bool = False
        self.status_panel_show_mouse_click: bool = False
        self.status_panel_show_screenshot_method: bool = True
        self.status_panel_show_screenshot_fps: bool = True
        self.status_panel_show_detection_fps: bool = True
        self.show_console: bool = True  # 終端視窗

        # 主題設定
        self.dark_mode: bool = False  # 深色主題

        # Acrylic 毛玻璃效果專用設置
        self.enable_acrylic: bool = True
        self.acrylic_window_alpha: int = 187  # 0-255, 視窗底層不透明度 (約 73%)
        self.acrylic_element_alpha: int = 25   # 0-255, UI 元素不透明度 (約 10%)
        
        # 優化：性能相關設置
        self.max_queue_size: int = 1        # 減少隊列大小，降低延遲

        # TensorRT FP16 加速（需要 NVIDIA GPU 及 TensorRT 安裝）
        self.trt_fp16_enabled: bool = False

        # CUDA IO Binding 零拷貝推理（僅 CUDA provider 有效）
        self.cuda_io_binding_enabled: bool = False

        # Kalman filter aim-point smoother — independent toggle from
        # prediction_enabled (VelocityPredictor); both can run together
        # (velocity prediction extrapolates ahead, Kalman then smooths it)
        self.kalman_enabled: bool = False
        self.kalman_process_noise: float = 0.01   # lower = smoother / lags more
        self.kalman_measurement_noise: float = 0.1  # lower = reacts faster / noisier

        # Frame skip gate
        self.frame_skip_enabled: bool = False
        self.frame_skip_threshold: float = 2.0     # avg pixel diff below this → skip

        # Camera motion compensation — subtract per-frame global scene shift before PID
        self.cam_motion_comp_enabled: bool = False
        self.cam_motion_comp_size: int = 128   # downsample resolution for phase correlation (128 or 256)

        # 速度預測瞄準（基於歷史位置估算目標未來位置）
        self.prediction_enabled: bool = False
        self.prediction_horizon_ms: float = 10.0    # 預測時間窗口 (ms)
        self.prediction_max_velocity: float = 1200.0  # 最大有效速度 (px/s)
        self.prediction_history_len: int = 3         # 歷史點數量

        # 目標鎖定（Sticky Lock）
        self.sticky_lock_enabled: bool = False
        self.lock_decay_frames: int = 15
        self.lock_iou_threshold: float = 0.3
        self.sticky_adaptive_iou: bool = True

        # FOV filter mode
        self.fov_circle_filter_enabled: bool = False  # circular FOV test instead of square

        # Web ESP overlay — stream detection state to a browser Canvas renderer over LAN
        self.web_esp_enabled: bool = False
        self.web_esp_http_port: int = 8080   # static page server
        self.web_esp_ws_port: int = 8765     # state broadcast websocket
        self.web_esp_fps: int = 60           # broadcast tick rate (latest-state wins)

        # Web Control — a *control-plane* LAN server (unlike Web ESP above,
        # which is read-only telemetry): lets a browser call the same
        # main-function actions the Qt GUI does (see core/app_controller.py).
        # Because it can mutate state rather than just observe it, it is
        # gated by a shared token (web_control_token) checked on every
        # request/WS handshake — see core/web_control_server.py.
        self.web_control_enabled: bool = False
        self.web_control_port: int = 8090
        # Empty until the feature is first enabled — web_control_server.start()
        # (and the GUI's enable toggle) generate one via secrets.token_urlsafe()
        # the first time it's needed, then persist it here so it survives restarts.
        self.web_control_token: str = ""

        # Aim shaping (ported from Someone_idea)
        self.aim_deadzone_enabled: bool = False
        self.aim_deadzone_min_px: float = 0.4
        self.aim_deadzone_close_px: float = 0.2
        self.max_move_per_frame_px: float = 85.0

        # Semantic false-positive filter (ported from Someone_idea)
        self.detect_semantic_filter_enabled: bool = False
        # Minimum-geometry layer of the semantic filter (detection_semantics.py).
        # 0 = disabled (matches pre-existing behavior); not yet exposed in the
        # GUI, but now a real persisted field instead of an unreachable default.
        self.detect_min_bbox_area_px: float = 0.0
        self.detect_min_bbox_short_side_px: float = 0.0
        self.detect_min_bbox_max_side_frac: float = 0.0

        # 供 _draw_overlay 使用的鎖定框顯示狀態（由 process_aiming 更新）
        self.display_locked_box: list | None = None
        self.display_locked_box_is_decaying: bool = False

        # 延遲/性能統計（預設關閉，避免輸出干擾）
        self.enable_latency_stats: bool = False
        self.latency_stats_interval: float = 1.0  # 秒

        # 供統計使用的時間戳（由不同線程更新）
        self.last_detection_time: float = 0.0
        self.last_overlay_update_time: float = 0.0

        # FPS 計數器（運行期狀態，不寫入配置檔）
        self.screenshot_frame_count: int = 0
        self.detection_frame_count: int = 0
        self.latest_boxes: List[List[float]] = []
        self.latest_confidences: List[float] = []
        # Unreduced by single_target_mode — same set the in-game overlay draws.
        # Web ESP reads this instead of latest_boxes so it shows every detection
        # regardless of single-target aiming mode.
        self.latest_all_boxes: List[List[float]] = []
        self.latest_all_confidences: List[float] = []

        # Runtime-only flags — never persisted to config.json
        # Set to True to pause inference without stopping threads or closing UI
        self.inference_paused: bool = False
        # Nominal FPS of the active capture source (UVC/NDI reports this;
        # screen capture uses monitor refresh rate or measured rate)
        self.source_nominal_fps: float = 0.0
        self.udp_recv_fps: float = 0.0        # raw assembled-frame rate from UDP sender
        self.udp_dropped_fps: float = 0.0     # incomplete frames evicted/sec (packet loss)

        # Secondary inference (V1 = PaddleOCR, V2 = ONNX HUD detector)
        self.second_inference_mode: str = "off"          # "off" | "v1_ocr" | "v2_onnx"
        self.second_inference_fps: int = 2               # scan rate (1-10 FPS)
        self.hud_model_path: str = ""                    # relative path to V2 .onnx inside Model_Hud/
        self.hud_confidence: float = 0.10                # V2 minimum detection confidence
        self.hud_game: str = "Apex Legends"              # selected game profile key from game.json
        self.hud_roi_coords: str = "1490,953,1870,1041"  # HUD ROI as "x1,y1,x2,y2" (from game.json)

        # Humanization post-processing layer (operates only on final dx/dy output)
        self.humanization: HumanizationConfig = HumanizationConfig()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize persisted config into the grouped (v2) JSON schema.

        State flags (STATE_FIELDS) are written to state.json instead and are
        intentionally excluded here.
        """
        out: Dict[str, Any] = {'config_version': self.config_version}
        for attr, path in _FIELD_MAP.items():
            _set_path(out, path, getattr(self, attr))
        # Crosshair RGB triplet → single [r, g, b] array.
        _set_path(out, 'display.crosshair.color',
                  [self.crosshair_color_r, self.crosshair_color_g, self.crosshair_color_b])
        out['humanization'] = dataclasses.asdict(self.humanization)
        return out

    def from_dict(self, data: Dict[str, Any]) -> None:
        """Load config from either the grouped (v2) or legacy flat (v1) JSON.

        Each field is read from its nested path first, then falls back to the
        flat key — so old flat config.json files load transparently (the
        dual-read is the migration).
        """
        for attr, path in _FIELD_MAP.items():
            val = _get_path(data, path)
            if val is _MISSING:
                val = data.get(attr, _MISSING)  # legacy flat fallback
            if val is not _MISSING and hasattr(self, attr):
                expected = type(getattr(self, attr))
                if expected in (int, float, bool, str) and not isinstance(val, expected):
                    try:
                        val = _coerce_bool(val) if expected is bool else expected(val)
                    except (ValueError, TypeError):
                        logger.warning("Config field '%s': could not coerce %r to %s, using default",
                                       attr, val, expected.__name__)
                        continue
                setattr(self, attr, val)

        # Crosshair color: nested [r, g, b], else legacy flat r/g/b.
        color = _get_path(data, 'display.crosshair.color')
        if isinstance(color, (list, tuple)) and len(color) >= 3:
            self.crosshair_color_r = int(color[0])
            self.crosshair_color_g = int(color[1])
            self.crosshair_color_b = int(color[2])
        else:
            for k in ('crosshair_color_r', 'crosshair_color_g', 'crosshair_color_b'):
                if k in data:
                    setattr(self, k, data[k])

        # Humanization dataclass — update in place, ignore unknown keys.
        hud = data.get('humanization')
        if isinstance(hud, dict):
            for hk, hv in hud.items():
                if hasattr(self.humanization, hk):
                    setattr(self.humanization, hk, hv)

        # Legacy flat state fields (back-compat; canonical source is state.json).
        for f in STATE_FIELDS:
            if f in data and hasattr(self, f):
                setattr(self, f, data[f])


def _state_path_for(config_path: str) -> str:
    """Return the state.json path that lives alongside the given config file."""
    return os.path.join(os.path.dirname(config_path), 'state.json')


def save_state(config_instance: Config, filepath: str = 'state.json') -> bool:
    """Persist one-time app state (STATE_FIELDS) to state.json."""
    try:
        data = {f: getattr(config_instance, f) for f in STATE_FIELDS}
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except (OSError, TypeError, ValueError) as e:
        logger.error("State save failed: %s", e)
        return False


def load_state(config_instance: Config, filepath: str = 'state.json') -> bool:
    """Load one-time app state from state.json (overrides any inline values)."""
    if not os.path.exists(filepath):
        return False
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for f_name in STATE_FIELDS:
            if f_name in data and hasattr(config_instance, f_name):
                setattr(config_instance, f_name, data[f_name])
        return True
    except (OSError, json.JSONDecodeError) as e:
        logger.error("State load failed: %s", e)
        return False


def save_config(config_instance: Config, filepath: str = 'config.json') -> bool:
    """
    將配置儲存到 JSON 檔案

    Writes the grouped (v2) schema via to_dict() — no stale keys survive. One-time
    app state goes to state.json, and language preference to language.json, so
    config.json holds only user settings.
    """
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config_instance.to_dict(), f, ensure_ascii=False, indent=2)
        save_state(config_instance, _state_path_for(filepath))
        logger.info("Config saved")
        return True
    except OSError as e:
        logger.error("Config save failed (IO): %s", e)
        return False
    except (TypeError, ValueError) as e:
        logger.error("Config save failed (serialization): %s", e)
        return False


def _migrate_config(data: dict) -> dict:
    """Apply forward migrations keyed by config_version.

    When a field is renamed or removed, add a block here and bump
    config_version in Config.__init__. The migrated dict is passed to
    from_dict(), so field names must match the current schema on exit.
    """
    # Example (not yet needed):
    # if data.get('config_version', 0) < 2:
    #     data['new_field'] = data.pop('old_field', default_value)
    #     data['config_version'] = 2
    return data


def load_config(config_instance: Config, filepath: str = 'config.json') -> bool:
    """
    從 JSON 檔案載入配置
    
    Args:
        config_instance: Config 實例
        filepath: 載入路徑
        
    Returns:
        是否成功載入
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        data = _migrate_config(data)
        config_instance.from_dict(data)

        # One-time app state lives in state.json (overrides any legacy inline values).
        load_state(config_instance, _state_path_for(filepath))

        # 向後兼容：確保檢測間隔在合理範圍內 (1-100ms)
        _validate_detect_interval(config_instance)

        # 向後兼容：確保截圖間隔在合理範圍內 (1-100ms)
        _validate_screenshot_interval(config_instance)

        # 向後兼容：確保閒置檢測間隔在合理範圍內 (5-500ms)
        _validate_idle_detect_interval(config_instance)

        # 向後兼容：修正截圖方式
        _validate_screenshot_method(config_instance)
        
        # 向後兼容：修正滑鼠移動方式
        _validate_mouse_method(config_instance)

        # 向後兼容：修正推理後端選擇
        _validate_inference_backend(config_instance)
        _validate_thread_priority(config_instance)

        # 向後兼容：確保偵測範圍在合理範圍內
        _validate_detect_range_size(config_instance)

        # 向後兼容：確保 UDP 接收緩衝區大於單一資料包最大值
        _validate_udp_recv_buffer_size(config_instance)
        
        logger.info("Config loaded")
        return True
        
    except FileNotFoundError:
        logger.info("Config file not found, using defaults")
        load_state(config_instance, _state_path_for(filepath))  # state.json may persist independently
        return False
    except json.JSONDecodeError as e:
        logger.error("Config load failed (JSON error): %s", e)
        return False
    except OSError as e:
        logger.error("Config load failed (IO): %s", e)
        return False


def _validate_detect_interval(config: Config) -> None:
    """驗證並修正檢測間隔"""
    detect_interval_ms = config.detect_interval * 1000
    if detect_interval_ms < 1:
        config.detect_interval = 0.001  # 1ms
        logger.warning("[Config] detect_interval too small, clamped to 1ms")
    elif detect_interval_ms > 100:
        config.detect_interval = 0.1  # 100ms
        logger.warning("[Config] detect_interval too large, clamped to 100ms")


def _validate_idle_detect_interval(config: Config) -> None:
    """驗證並修正閒置檢測間隔"""
    idle_ms = getattr(config, 'idle_detect_interval', 0.05) * 1000
    if idle_ms < 5:
        config.idle_detect_interval = 0.005
        logger.warning("[Config] idle_detect_interval too small, clamped to 5ms")
    elif idle_ms > 500:
        config.idle_detect_interval = 0.5
        logger.warning("[Config] idle_detect_interval too large, clamped to 500ms")


def _validate_screenshot_interval(config: Config) -> None:
    """驗證並修正截圖間隔"""
    screenshot_interval_ms = getattr(config, 'screenshot_interval', getattr(config, 'detect_interval', 0.008)) * 1000
    if screenshot_interval_ms < 1:
        config.screenshot_interval = 0.001  # 1ms
        logger.warning("[Config] screenshot_interval too small, clamped to 1ms")
    elif screenshot_interval_ms > 100:
        config.screenshot_interval = 0.1  # 100ms
        logger.warning("[Config] screenshot_interval too large, clamped to 100ms")


def _validate_udp_recv_buffer_size(config: Config) -> None:
    """Keep udp_recv_buffer_size at or above the largest datagram the OBS
    sender can emit.

    This is the size passed to ``socket.recvfrom()``. UDP is message-oriented:
    if the buffer is smaller than the arriving datagram, the OS delivers only
    that many bytes and **silently discards the rest** — no error, no short-read
    signal. The truncated payload still satisfies the receiver's completeness
    check (which counts chunks, not bytes), so the corruption survives all the
    way to ``cv2.imdecode`` and surfaces as unexplained decode failures rather
    than anything pointing back at this setting.

    The sender's ceiling is ``UDP_HEADER_SIZE + UDP_MAX_PAYLOAD`` = 14 + 60000
    (see udp_stream_filter.cpp); round up to 65536, which is also the maximum a
    single UDP datagram can be, so this can never be the limiting factor.
    """
    minimum = 65536
    try:
        raw = int(getattr(config, 'udp_recv_buffer_size', minimum))
    except (TypeError, ValueError):
        raw = minimum
    if raw < minimum:
        config.udp_recv_buffer_size = minimum
        logger.warning(
            "[Config] udp_recv_buffer_size %d is below the %d-byte maximum datagram "
            "the sender can emit — raised to %d. Values below this silently "
            "truncate frames instead of erroring.",
            raw, minimum, minimum,
        )
    else:
        config.udp_recv_buffer_size = raw


def _validate_mouse_method(config: Config) -> None:
    """驗證並修正滑鼠移動方式"""
    # 驗證滑鼠移動方式是否為有效值
    valid_move_methods = ('mouse_event', 'sendinput', 'ddxoft', 'arduino', 'makcu', 'xbox')
    if config.mouse_move_method not in valid_move_methods:
        config.mouse_move_method = 'mouse_event'
    
    # 驗證滑鼠點擊方式是否為有效值
    valid_click_methods = ('mouse_event', 'sendinput', 'ddxoft', 'arduino', 'makcu', 'xbox')
    if config.mouse_click_method not in valid_click_methods:
        config.mouse_click_method = 'mouse_event'


def _validate_screenshot_method(config: Config) -> None:
    """驗證並修正螢幕截圖方式"""
    valid_screenshot_methods = ('mss', 'dxcam', 'uvc', 'ndi', 'udp')
    if getattr(config, 'screenshot_method', 'mss') not in valid_screenshot_methods:
        config.screenshot_method = 'mss'
    # Legacy migration: 'ffmpeg' used to be its own top-level capture_method
    # value. It was always DirectShow under the hood (ffmpeg's Windows
    # capture input has no MSMF demuxer) — fold it into the new
    # dshow + v1 + ffmpeg_enabled shape instead of a bare 4th method.
    if getattr(config, 'uvc_capture_method', 'msmf') == 'ffmpeg':
        config.uvc_capture_method = 'dshow'
        config.uvc_dshow_backend = 'v1'
        config.uvc_ffmpeg_enabled = True
    if getattr(config, 'uvc_capture_method', 'msmf') not in ('dshow', 'msmf', 'any'):
        config.uvc_capture_method = 'msmf'
    if getattr(config, 'uvc_dshow_backend', 'v1') not in ('v1', 'v2'):
        config.uvc_dshow_backend = 'v1'
    config.uvc_ffmpeg_enabled = bool(getattr(config, 'uvc_ffmpeg_enabled', False))
    if getattr(config, 'uvc_video_format', 'mjpeg') not in ('mjpeg', 'yuy2', 'nv12', 'yuv420p'):
        config.uvc_video_format = 'mjpeg'
    if getattr(config, 'uvc_crop_mode', 'dynamic') not in ('dynamic', 'fixed'):
        config.uvc_crop_mode = 'dynamic'
    config.ndi_source_name = str(getattr(config, 'ndi_source_name', '') or '').strip()


def _validate_detect_range_size(config: Config) -> None:
    """驗證並修正 AI 偵測範圍（正方形邊長）

    規則：
    - 最小不得小於 max(fov_size, fov_height) — the square detection region
      must be able to contain the full FOV rectangle, not just its width
    - 最大不得大於螢幕高度
    """
    try:
        raw = int(getattr(config, 'detect_range_size', config.height))
    except (TypeError, ValueError):
        raw = int(config.height)

    fov_w = int(getattr(config, 'fov_size', 0) or 0)
    fov_h = int(getattr(config, 'fov_height', 0) or 0)
    min_size = max(fov_w, fov_h)
    max_size = int(getattr(config, 'height', raw) or raw)
    if max_size <= 0:
        max_size = raw if raw > 0 else 1
    # fov_size is itself user-configured and isn't validated against the
    # screen height anywhere else, so a config with fov_size > height would
    # otherwise make min_size > max_size below — max(min_size, min(max_size,
    # raw)) then evaluates to min_size, violating the documented "must not
    # be larger than screen height" invariant. Clamp the lower bound to the
    # upper bound first so that can't happen.
    min_size = min(min_size, max_size)

    clamped = max(min_size, min(max_size, raw))
    config.detect_range_size = clamped


def _validate_inference_backend(config: Config) -> None:
    """驗證並修正推理後端選擇"""
    valid_backends = ("auto", "tensorrt", "cuda", "directml", "cpu")
    if getattr(config, "inference_backend", "auto") not in valid_backends:
        config.inference_backend = "auto"


def _validate_thread_priority(config: Config) -> None:
    valid = ("normal", "above_normal", "high", "time_critical")
    if getattr(config, "thread_priority", "high") not in valid:
        config.thread_priority = "high"
