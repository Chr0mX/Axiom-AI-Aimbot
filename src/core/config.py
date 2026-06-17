# config.py
"""Configuration management module - Type-safe configuration using type hints"""

from __future__ import annotations

import ctypes
import dataclasses
import json
import os
from typing import List, Dict, Any

from .humanization import HumanizationConfig


def _get_screen_size() -> tuple[int, int]:
    """Get screen resolution"""
    user32 = ctypes.windll.user32
    user32.SetProcessDPIAware()
    return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)


_MISSING = object()

# One-time app state — persisted to state.json, NOT config.json.
STATE_FIELDS = ('disclaimer_agreed', 'first_run_complete', 'ndi_installer_ran_once')

# Single source of truth for the grouped (v2) schema: maps each flat Config
# attribute to its dotted path in the nested JSON. Drives to_dict() and from_dict().
# Fields intentionally absent (never persisted): runtime-derived (current_provider),
# auto-detected (model_input_size), derived (uvc_resolution), constants
# (uvc_window_name, latency_stats_alpha), state (see STATE_FIELDS), and the
# specially-handled crosshair color triplet + humanization dataclass.
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
    'ndi_source_name':            'capture.ndi.source_name',
    'ndi_bandwidth':              'capture.ndi.bandwidth',
    'uvc_show_window':            'capture.preview.enabled',
    'uvc_preview_scale_mode':     'capture.preview.scale_mode',
    'uvc_always_on_top':          'capture.preview.always_on_top',
    'preview_crop_to_detection':  'capture.preview.crop_to_detection',
    'preview_fps_cap':            'capture.preview.fps_cap',

    # --- aim ---
    'fov_size':                   'aim.fov_size',
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
    'max_move_per_frame_px':      'aim.max_move_per_frame_px',
    'detect_semantic_filter_enabled': 'aim.detect_semantic_filter_enabled',
    'pid_kp_x':                   'aim.pid.x.kp',
    'pid_ki_x':                   'aim.pid.x.ki',
    'pid_kd_x':                   'aim.pid.x.kd',
    'pid_kp_y':                   'aim.pid.y.kp',
    'pid_ki_y':                   'aim.pid.y.ki',
    'pid_kd_y':                   'aim.pid.y.kd',
    'aim_y_reduce_enabled':       'aim.y_reduce.enabled',
    'aim_y_reduce_delay':         'aim.y_reduce.delay',
    'aim_y_reduce_floor':         'aim.y_reduce.floor',
    'aim_y_reduce_ramp':          'aim.y_reduce.ramp',
    'aim_y_reduce_settle_px':     'aim.y_reduce.settle_px',
    'aim_y_vel_restore_px_s':     'aim.y_reduce.vel_restore_px_s',
    'kalman_enabled':             'aim.kalman.enabled',
    'kalman_process_noise':       'aim.kalman.process_noise',
    'kalman_measurement_noise':   'aim.kalman.measurement_noise',
    'ema_enabled':                'aim.ema.enabled',
    'ema_alpha':                  'aim.ema.alpha',
    'jitter_enabled':             'aim.jitter.enabled',
    'jitter_strength':            'aim.jitter.strength',
    'smart_jitter_enabled':       'aim.smart_jitter.enabled',
    'smart_jitter_strength':      'aim.smart_jitter.strength',
    'smart_jitter_box_threshold_pct': 'aim.smart_jitter.box_threshold_pct',
    'smart_jitter_lmb_gate':      'aim.smart_jitter.lmb_gate',
    'jitter_pattern_file':        'aim.smart_jitter.pattern_file',
    'aim_deadzone_enabled':       'aim.deadzone.enabled',
    'aim_deadzone_min_px':        'aim.deadzone.min_px',
    'aim_deadzone_close_px':      'aim.deadzone.close_px',
    'aim_lateral_brake_enabled':  'aim.lateral_brake.enabled',
    'aim_lateral_brake_strength': 'aim.lateral_brake.strength',
    'aim_lateral_brake_dom_trigger': 'aim.lateral_brake.dom_trigger',
    'aim_lateral_brake_dom_max':  'aim.lateral_brake.dom_max',
    'aim_lateral_brake_min_scale': 'aim.lateral_brake.min_scale',
    'head_width_ratio':           'aim.target_area.head_width_ratio',
    'head_height_ratio':          'aim.target_area.head_height_ratio',
    'body_width_ratio':           'aim.target_area.body_width_ratio',

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
    'box_ema_enabled':            'tracking.box_ema.enabled',
    'box_ema_alpha_x':            'tracking.box_ema.alpha_x',
    'box_ema_alpha_y':            'tracking.box_ema.alpha_y',
    'target_priority_mode':       'tracking.target_priority.mode',
    'target_priority_confidence_weight': 'tracking.target_priority.confidence_weight',

    # --- performance ---
    'thread_priority':            'performance.thread_priority',
    'performance_mode':           'performance.performance_mode',
    'max_queue_size':             'performance.max_queue_size',
    'cuda_io_binding_enabled':    'performance.cuda_io_binding_enabled',
    'skip_letterbox':             'performance.skip_letterbox',
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
    'xbox_auto_connect':          'hardware.xbox.auto_connect',

    # --- ui ---
    'dark_mode':                  'ui.dark_mode',
    'enable_acrylic':             'ui.enable_acrylic',
    'acrylic_window_alpha':       'ui.acrylic_window_alpha',
    'acrylic_element_alpha':      'ui.acrylic_element_alpha',
    'show_console':               'ui.show_console',
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
        
        # Full screen detection
        self.capture_width: int = self.width
        self.capture_height: int = self.height
        self.screenshot_method: str = "dxcam"  # 螢幕截圖方式
        self.uvc_device_index: int = 0
        self.uvc_width: int = self.width
        self.uvc_height: int = self.height
        self.uvc_fps: int = 60
        self.uvc_capture_method: str = "msmf"
        self.uvc_show_window: bool = True
        self.uvc_preview_scale_mode: str = "scale_to_fit"
        self.uvc_always_on_top: bool = True
        self.preview_crop_to_detection: bool = False
        self.preview_fps_cap: int = 0
        self.ndi_source_name: str = ""
        self.ndi_bandwidth: str = "highest"
        self.ndi_force_reconnect: bool = False
        self.ndi_width: int = self.width
        self.ndi_height: int = self.height
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
        self.fov_size: int = 200

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
        
        # PID 控制器參數 (分離 X 和 Y 軸)
        self.pid_kp_x: float = 0.26      # 水平 P: 比例 - 主要影響反應速度
        self.pid_ki_x: float = 0.0       # 水平 I: 積分 - 修正靜態誤差
        self.pid_kd_x: float = 0.0       # 水平 D: 微分 - 抑制抖動與過衝
        self.pid_kp_y: float = 0.26      # 垂直 P: 比例
        self.pid_ki_y: float = 0.0       # 垂直 I: 積分
        self.pid_kd_y: float = 0.0       # 垂直 D: 微分

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
        self.xbox_auto_connect: bool = True          # 選擇 xbox 時自動連線

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
        self.performance_mode: bool = True  # 預設啟用性能模式
        self.max_queue_size: int = 1        # 減少隊列大小，降低延遲

        # TensorRT FP16 加速（需要 NVIDIA GPU 及 TensorRT 安裝）
        self.trt_fp16_enabled: bool = False

        # CUDA IO Binding 零拷貝推理（僅 CUDA provider 有效）
        self.cuda_io_binding_enabled: bool = False

        self.skip_letterbox: bool = False         # 直接縮放取代 letterbox（略快，正方形擷取無失真）

        # Kalman filter aim-point smoother (mutually exclusive with EMA in UI)
        self.kalman_enabled: bool = False
        self.kalman_process_noise: float = 0.01   # lower = smoother / lags more
        self.kalman_measurement_noise: float = 0.1  # lower = reacts faster / noisier

        # Basic jitter
        self.jitter_enabled: bool = False
        self.jitter_strength: float = 1.5          # pixel offset radius

        # Frame skip gate
        self.frame_skip_enabled: bool = False
        self.frame_skip_threshold: float = 2.0     # avg pixel diff below this → skip

        # Smart jitter — fires when bounding box is small (target is far away)
        self.smart_jitter_enabled: bool = False
        self.smart_jitter_strength: float = 6.0                # max pixel offset radius applied each frame
        self.smart_jitter_box_threshold_pct: float = 15.0   # box_h / detect_range_size < threshold% → jitter
        self.smart_jitter_lmb_gate: bool = True             # only jitter while aim key is held
        self.jitter_pattern_file: str = ""                  # path to recorded .json; empty = procedural

        # EMA 瞄準點平滑（在 PID 前平滑目標座標）
        self.ema_enabled: bool = False
        self.ema_alpha: float = 0.7  # 1.0=原始，0.3=強平滑

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
        self.box_ema_enabled: bool = False
        self.box_ema_alpha_x: float = 0.8
        self.box_ema_alpha_y: float = 0.5

        # FOV filter mode
        self.fov_circle_filter_enabled: bool = False  # circular FOV test instead of square

        # Aim shaping (ported from Someone_idea)
        self.aim_deadzone_enabled: bool = False
        self.aim_deadzone_min_px: float = 0.4
        self.aim_deadzone_close_px: float = 0.2
        self.aim_lateral_brake_enabled: bool = False
        self.aim_lateral_brake_strength: float = 0.75
        self.aim_lateral_brake_dom_trigger: float = 1.12
        self.aim_lateral_brake_dom_max: float = 3.0
        self.aim_lateral_brake_min_scale: float = 0.26
        self.max_move_per_frame_px: float = 85.0

        # Semantic false-positive filter (ported from Someone_idea)
        self.detect_semantic_filter_enabled: bool = False

        # 供 _draw_overlay 使用的鎖定框顯示狀態（由 process_aiming 更新）
        self.display_locked_box: list | None = None
        self.display_locked_box_is_decaying: bool = False

        # 延遲/性能統計（預設關閉，避免輸出干擾）
        self.enable_latency_stats: bool = False
        self.latency_stats_interval: float = 1.0  # 秒

        # 供統計使用的時間戳（由不同線程更新）
        self.last_screenshot_time: float = 0.0
        self.last_detection_time: float = 0.0
        self.last_overlay_update_time: float = 0.0

        # FPS 計數器（運行期狀態，不寫入配置檔）
        self.screenshot_frame_count: int = 0
        self.detection_frame_count: int = 0
        self.latest_boxes: List[List[float]] = []
        self.latest_confidences: List[float] = []

        # Runtime-only flags — never persisted to config.json
        # Set to True to pause inference without stopping threads or closing UI
        self.inference_paused: bool = False
        # Nominal FPS of the active capture source (UVC/NDI reports this;
        # screen capture uses monitor refresh rate or measured rate)
        self.source_nominal_fps: float = 0.0

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

        # Legacy flat state fields (back-compat; canonical source is state.json).
        for f in STATE_FIELDS:
            if f in data and hasattr(self, f):
                setattr(self, f, data[f])

        # Humanization dataclass — update in place, ignore unknown keys.
        hud = data.get('humanization')
        if isinstance(hud, dict):
            for hk, hv in hud.items():
                if hasattr(self.humanization, hk):
                    setattr(self.humanization, hk, hv)


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
        print(f"狀態儲存失敗: {e}")
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
        print(f"狀態載入失敗: {e}")
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
        print("設定已儲存")
        return True
    except OSError as e:
        print(f"設定儲存失敗 (IO錯誤): {e}")
        return False
    except (TypeError, ValueError) as e:
        print(f"設定儲存失敗 (序列化錯誤): {e}")
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
        
        print("設定檔已載入")
        return True
        
    except FileNotFoundError:
        print("未找到設定檔，使用預設值")
        load_state(config_instance, _state_path_for(filepath))  # state.json may persist independently
        return False
    except json.JSONDecodeError as e:
        print(f"設定載入失敗 (JSON 格式錯誤): {e}")
        return False
    except OSError as e:
        print(f"設定載入失敗 (IO錯誤): {e}")
        return False


def _validate_detect_interval(config: Config) -> None:
    """驗證並修正檢測間隔"""
    detect_interval_ms = config.detect_interval * 1000
    if detect_interval_ms < 1:
        config.detect_interval = 0.001  # 1ms
        print("[配置修正] 檢測間隔過小，已調整為 1ms")
    elif detect_interval_ms > 100:
        config.detect_interval = 0.1  # 100ms
        print("[配置修正] 檢測間隔過大，已調整為 100ms")


def _validate_idle_detect_interval(config: Config) -> None:
    """驗證並修正閒置檢測間隔"""
    idle_ms = getattr(config, 'idle_detect_interval', 0.05) * 1000
    if idle_ms < 5:
        config.idle_detect_interval = 0.005
        print("[配置修正] 閒置檢測間隔過小，已調整為 5ms")
    elif idle_ms > 500:
        config.idle_detect_interval = 0.5
        print("[配置修正] 閒置檢測間隔過大，已調整為 500ms")


def _validate_screenshot_interval(config: Config) -> None:
    """驗證並修正截圖間隔"""
    screenshot_interval_ms = getattr(config, 'screenshot_interval', getattr(config, 'detect_interval', 0.008)) * 1000
    if screenshot_interval_ms < 1:
        config.screenshot_interval = 0.001  # 1ms
        print("[配置修正] 截圖間隔過小，已調整為 1ms")
    elif screenshot_interval_ms > 100:
        config.screenshot_interval = 0.1  # 100ms
        print("[配置修正] 截圖間隔過大，已調整為 100ms")


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
    valid_screenshot_methods = ('mss', 'dxcam', 'uvc', 'ndi')
    if getattr(config, 'screenshot_method', 'mss') not in valid_screenshot_methods:
        config.screenshot_method = 'mss'
    if getattr(config, 'uvc_capture_method', 'dshow') not in ('auto', 'dshow', 'msmf', 'any'):
        config.uvc_capture_method = 'dshow'
    if getattr(config, 'uvc_preview_scale_mode', 'scale_to_fit') not in (
        'scale_to_fit', 'scale_to_canvas', 'fit_to_screen'
    ):
        config.uvc_preview_scale_mode = 'scale_to_fit'
    config.ndi_source_name = str(getattr(config, 'ndi_source_name', '') or '').strip()


def _validate_detect_range_size(config: Config) -> None:
    """驗證並修正 AI 偵測範圍（正方形邊長）

    規則：
    - 最小不得小於 fov_size
    - 最大不得大於螢幕高度
    """
    try:
        raw = int(getattr(config, 'detect_range_size', config.height))
    except (TypeError, ValueError):
        raw = int(config.height)

    min_size = int(getattr(config, 'fov_size', 0) or 0)
    max_size = int(getattr(config, 'height', raw) or raw)
    if max_size <= 0:
        max_size = raw if raw > 0 else 1

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
