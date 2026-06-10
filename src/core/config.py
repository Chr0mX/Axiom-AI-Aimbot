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
        # Automatically get screen resolution
        self.width, self.height = _get_screen_size()
        
        self.center_x: int = self.width // 2
        self.center_y: int = self.height // 2
        
        # Full screen detection
        self.capture_width: int = self.width
        self.capture_height: int = self.height
        self.capture_left: int = 0
        self.capture_top: int = 0
        self.screenshot_method: str = "dxcam"  # 螢幕截圖方式
        self.uvc_device_index: int = 0
        self.uvc_width: int = self.width
        self.uvc_height: int = self.height
        self.uvc_fps: int = 60
        self.uvc_capture_method: str = "msmf"
        self.uvc_resolution: str = f"{self.uvc_width}x{self.uvc_height}"
        self.uvc_show_window: bool = True
        self.uvc_window_name: str = "Axiom UVC Preview"
        self.uvc_preview_scale_mode: str = "scale_to_fit"
        self.preview_crop_to_detection: bool = False
        self.ndi_source_name: str = ""
        self.ndi_bandwidth: str = "highest"
        self.ndi_pre_resize: bool = False   # resize NDI frames to model_input_size in capture thread
        self.ndi_width: int = self.width
        self.ndi_height: int = self.height
        self.crosshairX: int = self.width // 2
        self.crosshairY: int = self.height // 2
        self.region: Dict[str, int] = {
            "top": 0, "left": 0, 
            "width": self.width, "height": self.height
        }

        # Program execution state
        self.Running: bool = True
        self.AimToggle: bool = True
        
        # ONNX model related settings
        self.model_input_size: int = 640
        self.model_path: str = os.path.join('Model', 'ApexLegendsOrbeet_15k.onnx')
        self.current_provider: str = "DmlExecutionProvider"
        self.inference_backend: str = "auto"
        self.ndi_installer_ran_once: bool = False
        # Hybrid computing: Automatically fallback to CPU when operators are not supported by DirectML
        # ONNX Runtime providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
        self.dml_cpu_fallback: bool = True

        # Aiming and display settings
        self.AimKeys: List[int] = [0x01, 0x06, 0x02]  # Left Click + X2 Key + Right Click
        self.fov_size: int = 222

        # AI detection range (square edge length): Separated from fov_size, but must not be smaller than fov_size, and must not be larger than screen height
        # Defaults to screen height (same as legacy behavior)
        self.detect_range_size: int = self.height # AI 偵測範圍（正方形邊長），獨立於 fov_size，但不得小於 fov_size，且不得大於螢幕高度，預設為螢幕高度（與舊版行為相同）
        self.show_confidence: bool = True # 是否在框上顯示置信度
        self.min_confidence: float = 0.80  # 最小置信度，範圍 0~1
        self.aim_part: str = "head"
        
        # Single target mode
        self.single_target_mode: bool = True  # 啟用單一目標模式（只瞄準置信度最高的目標）
        
        # Smart tracking prediction settings (replaces Kalman)
        self.tracker_enabled: bool = False          # SmartTracker removed; kept for config compatibility
        self.tracker_prediction_time: float = 0.025   # Prediction time (seconds)
        self.tracker_smoothing_factor: float = 0.66   # Velocity smoothing factor (0~1)
        self.tracker_stop_threshold: float = 10.0    # Low speed zeroing threshold (pixels/sec)
        self.tracker_show_prediction: bool = True    # Show prediction visualization

        # Tracker prediction data (updated by ai_loop, read by overlay)
        self.tracker_predicted_x: float = 0.0        # Predicted X coordinate
        self.tracker_predicted_y: float = 0.0        # Predicted Y coordinate
        self.tracker_current_x: float = 0.0          # Current observed X coordinate
        self.tracker_current_y: float = 0.0          # Current observed Y coordinate
        self.tracker_has_prediction: bool = False    # Whether a valid prediction exists

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

        # Target priority scoring
        self.target_priority_mode: str = "distance"       # "distance" | "confidence" | "composite"
        self.target_priority_confidence_weight: float = 0.5  # Weight for confidence in composite mode

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
        self.auto_match_fps: bool = False         # 截圖間隔自動跟隨推理間隔
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
        self.makcu_aim_button: str = "lmb"  # "lmb", "rmb", or "off"
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

        # Smart jitter — fires when bounding box is small (target is far away)
        self.smart_jitter_enabled: bool = False
        self.smart_jitter_strength: float = 6.0                # max pixel offset radius applied each frame
        self.smart_jitter_box_threshold_pct: float = 15.0   # box_h / detect_range_size < threshold% → jitter
        self.smart_jitter_lmb_gate: bool = True             # only jitter while aim key is held

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
        self.lock_decay_frames: int = 15       # 鎖定目標消失後維持的幀數
        self.lock_iou_threshold: float = 0.3   # 視為同一目標的最低 IoU（adaptive 模式下作為基礎值）
        self.sticky_adaptive_iou: bool = True  # adaptive IoU scaling by box area (Someone_idea)

        # FOV filter mode
        self.fov_circle_filter_enabled: bool = True  # circular FOV test instead of square

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
        self.detect_semantic_filter_enabled: bool = True

        # 供 _draw_overlay 使用的鎖定框顯示狀態（由 process_aiming 更新）
        self.display_locked_box: list | None = None
        self.display_locked_box_is_decaying: bool = False

        # 延遲/性能統計（預設關閉，避免輸出干擾）
        self.enable_latency_stats: bool = False
        self.latency_stats_interval: float = 1.0  # 秒
        self.latency_stats_alpha: float = 0.2     # EMA 平滑係數 (0~1)

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
        """將可儲存的配置轉為字典"""
        return {
            'fov_size': self.fov_size,
            'detect_range_size': self.detect_range_size,
            'model_path': self.model_path,
            'model_input_size': self.model_input_size,
            'current_provider': self.current_provider,
            'inference_backend': self.inference_backend,
            'ndi_installer_ran_once': self.ndi_installer_ran_once,
            'dml_cpu_fallback': self.dml_cpu_fallback,
            'pid_kp_x': self.pid_kp_x,
            'pid_ki_x': self.pid_ki_x,
            'pid_kd_x': self.pid_kd_x,
            'pid_kp_y': self.pid_kp_y,
            'pid_ki_y': self.pid_ki_y,
            'pid_kd_y': self.pid_kd_y,
            'aim_y_reduce_enabled': self.aim_y_reduce_enabled,
            'aim_y_reduce_delay': self.aim_y_reduce_delay,
            'aim_part': self.aim_part,
            'AimKeys': self.AimKeys,
            'auto_fire_key': self.auto_fire_key,
            'always_auto_fire': self.always_auto_fire,
            'auto_fire_delay': self.auto_fire_delay,
            'auto_fire_interval': self.auto_fire_interval,
            'auto_fire_target_part': self.auto_fire_target_part,
            'min_confidence': self.min_confidence,
            'show_confidence': self.show_confidence,
            'detect_interval': self.detect_interval,
            'screenshot_interval': self.screenshot_interval,
            'idle_detect_interval': self.idle_detect_interval,
            'idle_detect_enabled': self.idle_detect_enabled,
            'screenshot_method': self.screenshot_method,
            'uvc_device_index': self.uvc_device_index,
            'uvc_width': self.uvc_width,
            'uvc_height': self.uvc_height,
            'uvc_fps': self.uvc_fps,
            'uvc_capture_method': self.uvc_capture_method,
            'uvc_resolution': self.uvc_resolution,
            'uvc_show_window': self.uvc_show_window,
            'uvc_window_name': self.uvc_window_name,
            'uvc_preview_scale_mode': self.uvc_preview_scale_mode,
            'preview_crop_to_detection': self.preview_crop_to_detection,
            'ndi_source_name': self.ndi_source_name,
            'ndi_bandwidth': self.ndi_bandwidth,
            'ndi_pre_resize': self.ndi_pre_resize,
            'keep_detecting': self.keep_detecting,
            'always_aim': self.always_aim,
            'makcu_aim_button': self.makcu_aim_button,
            'fov_follow_mouse': self.fov_follow_mouse,
            'aim_toggle_key': self.aim_toggle_key,
            'auto_fire_key2': self.auto_fire_key2,
            'AimToggle': self.AimToggle,
            'show_fov': self.show_fov,
            'show_boxes': self.show_boxes,
            'show_detect_range': self.show_detect_range,
            'show_status_panel': self.show_status_panel,
            'status_panel_show_auto_aim': self.status_panel_show_auto_aim,
            'status_panel_show_model': self.status_panel_show_model,
            'status_panel_show_mouse_move': self.status_panel_show_mouse_move,
            'status_panel_show_mouse_click': self.status_panel_show_mouse_click,
            'status_panel_show_screenshot_method': self.status_panel_show_screenshot_method,
            'status_panel_show_screenshot_fps': self.status_panel_show_screenshot_fps,
            'status_panel_show_detection_fps': self.status_panel_show_detection_fps,
            'single_target_mode': self.single_target_mode,
            'head_width_ratio': self.head_width_ratio,
            'head_height_ratio': self.head_height_ratio,
            'body_width_ratio': self.body_width_ratio,
            'performance_mode': self.performance_mode,
            'max_queue_size': self.max_queue_size,
            'enable_latency_stats': self.enable_latency_stats,
            'latency_stats_interval': self.latency_stats_interval,
            'latency_stats_alpha': self.latency_stats_alpha,

            'mouse_move_method': self.mouse_move_method,
            'mouse_click_method': self.mouse_click_method,
            'arduino_com_port': self.arduino_com_port,
            'makcu_com_port': self.makcu_com_port,
            'makcu_baud_rate': self.makcu_baud_rate,
            'arduino_baud_rate': self.arduino_baud_rate,
            'xbox_sensitivity': self.xbox_sensitivity,
            'xbox_deadzone': self.xbox_deadzone,
            'xbox_auto_connect': self.xbox_auto_connect,
            'show_console': self.show_console,

            'trt_fp16_enabled': self.trt_fp16_enabled,
            'cuda_io_binding_enabled': self.cuda_io_binding_enabled,
            'skip_letterbox': self.skip_letterbox,
            'auto_match_fps': self.auto_match_fps,

            'kalman_enabled': self.kalman_enabled,
            'kalman_process_noise': self.kalman_process_noise,
            'kalman_measurement_noise': self.kalman_measurement_noise,

            'smart_jitter_enabled': self.smart_jitter_enabled,
            'smart_jitter_strength': self.smart_jitter_strength,
            'smart_jitter_box_threshold_pct': self.smart_jitter_box_threshold_pct,
            'smart_jitter_lmb_gate': self.smart_jitter_lmb_gate,

            'ema_enabled': self.ema_enabled,
            'ema_alpha': self.ema_alpha,
            'prediction_enabled': self.prediction_enabled,
            'prediction_horizon_ms': self.prediction_horizon_ms,
            'prediction_max_velocity': self.prediction_max_velocity,
            'prediction_history_len': self.prediction_history_len,

            'sticky_lock_enabled': self.sticky_lock_enabled,
            'lock_decay_frames': self.lock_decay_frames,
            'lock_iou_threshold': self.lock_iou_threshold,
            'sticky_adaptive_iou': self.sticky_adaptive_iou,

            'fov_circle_filter_enabled': self.fov_circle_filter_enabled,

            'aim_deadzone_enabled': self.aim_deadzone_enabled,
            'aim_deadzone_min_px': self.aim_deadzone_min_px,
            'aim_deadzone_close_px': self.aim_deadzone_close_px,
            'aim_lateral_brake_enabled': self.aim_lateral_brake_enabled,
            'aim_lateral_brake_strength': self.aim_lateral_brake_strength,
            'aim_lateral_brake_dom_trigger': self.aim_lateral_brake_dom_trigger,
            'aim_lateral_brake_dom_max': self.aim_lateral_brake_dom_max,
            'aim_lateral_brake_min_scale': self.aim_lateral_brake_min_scale,
            'max_move_per_frame_px': self.max_move_per_frame_px,

            'detect_semantic_filter_enabled': self.detect_semantic_filter_enabled,

            'target_priority_mode': self.target_priority_mode,
            'target_priority_confidence_weight': self.target_priority_confidence_weight,

            'box_color_theme': self.box_color_theme,
            'chroma_box_speed': self.chroma_box_speed,
            'show_tracer_line': self.show_tracer_line,

            'show_crosshair': self.show_crosshair,
            'crosshair_style': self.crosshair_style,
            'crosshair_color_r': self.crosshair_color_r,
            'crosshair_color_g': self.crosshair_color_g,
            'crosshair_color_b': self.crosshair_color_b,
            'crosshair_size': self.crosshair_size,
            'disclaimer_agreed': self.disclaimer_agreed,
            'first_run_complete': self.first_run_complete,

            'tracker_enabled': self.tracker_enabled,
            'tracker_prediction_time': self.tracker_prediction_time,
            'tracker_smoothing_factor': self.tracker_smoothing_factor,
            'tracker_stop_threshold': self.tracker_stop_threshold,
            'tracker_show_prediction': self.tracker_show_prediction,

            'dark_mode': self.dark_mode,

            'enable_acrylic': self.enable_acrylic,
            'acrylic_window_alpha': self.acrylic_window_alpha,
            'acrylic_element_alpha': self.acrylic_element_alpha,

            'humanization': dataclasses.asdict(self.humanization),
        }
    
    def from_dict(self, data: Dict[str, Any]) -> None:
        """從字典載入配置"""
        for key, value in data.items():
            if key == 'humanization' and isinstance(value, dict):
                # Update the dataclass fields in-place rather than replacing the object,
                # so unknown/future keys in the JSON are ignored gracefully.
                for hk, hv in value.items():
                    if hasattr(self.humanization, hk):
                        setattr(self.humanization, hk, hv)
            elif hasattr(self, key):
                setattr(self, key, value)


def save_config(config_instance: Config, filepath: str = 'config.json') -> bool:
    """
    將配置儲存到 JSON 檔案
    
    Args:
        config_instance: Config 實例
        filepath: 儲存路徑
        
    Returns:
        是否成功儲存
    """
    try:
        # 先讀取現有的 config.json，保留不在 Config 類中的欄位（如 language）
        existing_data = {}
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
            except (json.JSONDecodeError, OSError):
                existing_data = {}
        
        # 將新的配置資料合併到現有資料上（新值覆蓋舊值，但保留額外欄位）
        data = config_instance.to_dict()
        existing_data.update(data)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=2)
        print("設定已儲存")
        return True
    except OSError as e:
        print(f"設定儲存失敗 (IO錯誤): {e}")
        return False
    except (TypeError, ValueError) as e:
        print(f"設定儲存失敗 (序列化錯誤): {e}")
        return False


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
        
        config_instance.from_dict(data)
        
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

        # 向後兼容：確保偵測範圍在合理範圍內
        _validate_detect_range_size(config_instance)
        
        print("設定檔已載入")
        return True
        
    except FileNotFoundError:
        print("未找到設定檔，使用預設值")
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
