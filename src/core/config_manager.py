# config_manager.py
"""參數配置管理模組 - 純業務邏輯，無 GUI 依賴"""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime
from typing import List, Optional, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .config import Config


class ConfigManager:
    """參數配置管理器
    
    處理參數配置檔案的保存、載入、刪除、重命名、匯入匯出等操作。
    配置檔案以 JSON 格式儲存在指定目錄中。
    
    Attributes:
        configs_dir: 參數配置儲存目錄路徑
    """
    
    def __init__(self, configs_dir: str = "config") -> None:
        self.configs_dir = configs_dir
        self.ensure_configs_directory()
        
    def ensure_configs_directory(self) -> None:
        """確保參數配置目錄存在"""
        if not os.path.exists(self.configs_dir):
            os.makedirs(self.configs_dir)
            
    def get_config_list(self) -> List[str]:
        """獲取所有參數配置列表"""
        if not os.path.exists(self.configs_dir):
            return []
        
        configs = []
        for file in os.listdir(self.configs_dir):
            if file.endswith('.json'):
                config_name = file[:-5]  # 移除.json後綴
                configs.append(config_name)
        return sorted(configs)
    
    def save_config(self, config_instance: Config, config_name: str) -> bool:
        """保存當前配置為參數配置"""
        config_path = os.path.join(self.configs_dir, f"{config_name}.json")
        
        # 創建參數配置數據
        config_data = {
            'name': config_name,
            'created_time': datetime.now().isoformat(),
            'description': f"參數配置 - {config_name}",
            'config': self._get_config_data(config_instance)
        }
        
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, ensure_ascii=False, indent=2)
            return True
        except OSError as e:
            print(f"保存參數配置失敗: {e}")
            return False
    
    def _get_config_data(self, config_instance: Config) -> Dict[str, Any]:
        """從配置實例獲取配置數據"""
        return {
            # 基本檢測參數
            'fov_size': config_instance.fov_size,
            'detect_range_size': getattr(config_instance, 'detect_range_size', getattr(config_instance, 'height', 0)),
            'min_confidence': config_instance.min_confidence,
            'detect_interval': config_instance.detect_interval,
            'screenshot_interval': getattr(config_instance, 'screenshot_interval', getattr(config_instance, 'detect_interval', 0.008)),
            'idle_detect_interval': getattr(config_instance, 'idle_detect_interval', 0.05),
            'model_path': config_instance.model_path,
            'model_input_size': config_instance.model_input_size,
            'current_provider': config_instance.current_provider,
            
            # PID控制器參數
            'pid_kp_x': config_instance.pid_kp_x,
            'pid_ki_x': config_instance.pid_ki_x,
            'pid_kd_x': config_instance.pid_kd_x,
            'pid_kp_y': config_instance.pid_kp_y,
            'pid_ki_y': config_instance.pid_ki_y,
            'pid_kd_y': config_instance.pid_kd_y,
            
            # 瞄準設定
            'aim_part': config_instance.aim_part,
            'single_target_mode': config_instance.single_target_mode,
            'head_width_ratio': config_instance.head_width_ratio,
            'head_height_ratio': config_instance.head_height_ratio,
            'body_width_ratio': config_instance.body_width_ratio,

            # 瞄準曲線平滑
            'bezier_curve_enabled': getattr(config_instance, 'bezier_curve_enabled', False),
            'bezier_curve_strength': getattr(config_instance, 'bezier_curve_strength', 0.35),
            'bezier_curve_steps': getattr(config_instance, 'bezier_curve_steps', 4),
            
            # 按鍵設定
            'AimKeys': config_instance.AimKeys,
            'aim_toggle_key': config_instance.aim_toggle_key,
            'auto_fire_key': config_instance.auto_fire_key,
            'auto_fire_key2': config_instance.auto_fire_key2,
            'always_auto_fire': getattr(config_instance, 'always_auto_fire', False),
            
            # 自動開火設定
            'auto_fire_delay': config_instance.auto_fire_delay,
            'auto_fire_interval': config_instance.auto_fire_interval,
            'auto_fire_target_part': config_instance.auto_fire_target_part,
            
            # 顯示設定
            'show_confidence': config_instance.show_confidence,
            'show_fov': config_instance.show_fov,
            'show_boxes': config_instance.show_boxes,
            'show_detect_range': getattr(config_instance, 'show_detect_range', False),
            'show_status_panel': config_instance.show_status_panel,
            'status_panel_show_auto_aim': getattr(config_instance, 'status_panel_show_auto_aim', True),
            'status_panel_show_model': getattr(config_instance, 'status_panel_show_model', True),
            'status_panel_show_mouse_move': getattr(config_instance, 'status_panel_show_mouse_move', True),
            'status_panel_show_mouse_click': getattr(config_instance, 'status_panel_show_mouse_click', True),
            'status_panel_show_screenshot_method': getattr(config_instance, 'status_panel_show_screenshot_method', True),
            'status_panel_show_screenshot_fps': getattr(config_instance, 'status_panel_show_screenshot_fps', True),
            'status_panel_show_detection_fps': getattr(config_instance, 'status_panel_show_detection_fps', True),
            'show_console': config_instance.show_console,
            
            # 功能開關
            'AimToggle': config_instance.AimToggle,
            'keep_detecting': config_instance.keep_detecting,
            'always_aim': getattr(config_instance, 'always_aim', False),
            'fov_follow_mouse': config_instance.fov_follow_mouse,
            
            # 性能設定
            'performance_mode': config_instance.performance_mode,
            'max_queue_size': config_instance.max_queue_size,

            # 模型回退
            'dml_cpu_fallback': getattr(config_instance, 'dml_cpu_fallback', True),

            # 滑鼠與手把控制
            'mouse_move_method': getattr(config_instance, 'mouse_move_method', 'mouse_event'),
            'mouse_click_method': getattr(config_instance, 'mouse_click_method', 'mouse_event'),
            'arduino_com_port': getattr(config_instance, 'arduino_com_port', ''),
            'makcu_com_port': getattr(config_instance, 'makcu_com_port', ''),
            'xbox_sensitivity': getattr(config_instance, 'xbox_sensitivity', 1.0),
            'xbox_deadzone': getattr(config_instance, 'xbox_deadzone', 0.05),
            'xbox_auto_connect': getattr(config_instance, 'xbox_auto_connect', True),
            'screenshot_method': getattr(config_instance, 'screenshot_method', 'mss'),
            'uvc_device_index': getattr(config_instance, 'uvc_device_index', 0),
            'uvc_width': getattr(config_instance, 'uvc_width', getattr(config_instance, 'width', 1920)),
            'uvc_height': getattr(config_instance, 'uvc_height', getattr(config_instance, 'height', 1080)),
            'uvc_resolution': getattr(config_instance, 'uvc_resolution', ''),
            'uvc_fps': getattr(config_instance, 'uvc_fps', 60),
            'uvc_capture_method': getattr(config_instance, 'uvc_capture_method', 'dshow'),
            'uvc_show_window': getattr(config_instance, 'uvc_show_window', True),
            'uvc_preview_scale_mode': getattr(config_instance, 'uvc_preview_scale_mode', 'scale_to_fit'),

            # Y軸壓槍速度歸零
            'aim_y_reduce_enabled': getattr(config_instance, 'aim_y_reduce_enabled', False),
            'aim_y_reduce_delay': getattr(config_instance, 'aim_y_reduce_delay', 0.6),

            # 智慧追蹤預判
            'tracker_enabled': getattr(config_instance, 'tracker_enabled', False),
            'tracker_prediction_time': getattr(config_instance, 'tracker_prediction_time', 0.025),
            'tracker_smoothing_factor': getattr(config_instance, 'tracker_smoothing_factor', 0.66),
            'tracker_stop_threshold': getattr(config_instance, 'tracker_stop_threshold', 10.0),
            'tracker_show_prediction': getattr(config_instance, 'tracker_show_prediction', True),

            # 延遲統計
            'enable_latency_stats': getattr(config_instance, 'enable_latency_stats', False),
            'latency_stats_interval': getattr(config_instance, 'latency_stats_interval', 1.0),
            'latency_stats_alpha': getattr(config_instance, 'latency_stats_alpha', 0.2),
        }
    
    def load_config(self, config_instance: Config, config_name: str) -> bool:
        """載入參數配置"""
        config_path = os.path.join(self.configs_dir, f"{config_name}.json")
        
        if not os.path.exists(config_path):
            return False
            
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            # 載入配置到實例
            config_data = config_data.get('config', {})
            for key, value in config_data.items():
                if hasattr(config_instance, key):
                    setattr(config_instance, key, value)
            
            return True
        except (OSError, json.JSONDecodeError) as e:
            print(f"載入參數配置失敗: {e}")
            return False
    
    def delete_config(self, config_name: str) -> bool:
        """刪除參數配置"""
        config_path = os.path.join(self.configs_dir, f"{config_name}.json")
        
        if os.path.exists(config_path):
            try:
                os.remove(config_path)
                return True
            except OSError as e:
                print(f"刪除參數配置失敗: {e}")
                return False
        return False
    
    def rename_config(self, old_name: str, new_name: str) -> bool:
        """重命名參數配置"""
        old_path = os.path.join(self.configs_dir, f"{old_name}.json")
        new_path = os.path.join(self.configs_dir, f"{new_name}.json")
        
        if os.path.exists(old_path) and not os.path.exists(new_path):
            try:
                # 讀取舊文件並更新名稱
                with open(old_path, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                config_data['name'] = new_name
                
                # 寫入新文件
                with open(new_path, 'w', encoding='utf-8') as f:
                    json.dump(config_data, f, ensure_ascii=False, indent=2)
                
                # 刪除舊文件
                os.remove(old_path)
                return True
            except (OSError, json.JSONDecodeError) as e:
                print(f"重命名參數配置失敗: {e}")
                return False
        return False
    
    def export_config(self, config_name: str, export_path: str) -> bool:
        """匯出參數配置"""
        config_path = os.path.join(self.configs_dir, f"{config_name}.json")
        
        if os.path.exists(config_path):
            try:
                shutil.copy2(config_path, export_path)
                return True
            except OSError as e:
                print(f"匯出參數配置失敗: {e}")
                return False
        return False
    
    def import_config(self, import_path: str) -> Optional[str]:
        """
        匯入參數配置
        
        Returns:
            成功時返回參數名稱，失敗時返回 None
        """
        if not os.path.exists(import_path):
            return None
            
        try:
            # 讀取匯入的配置
            with open(import_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            # 獲取配置名稱
            config_name = config_data.get('name', 'imported_config')
            
            # 確保名稱唯一
            original_name = config_name
            counter = 1
            while os.path.exists(os.path.join(self.configs_dir, f"{config_name}.json")):
                config_name = f"{original_name}_{counter}"
                counter += 1
            
            # 更新名稱並保存
            config_data['name'] = config_name
            config_path = os.path.join(self.configs_dir, f"{config_name}.json")
            
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, ensure_ascii=False, indent=2)
            
            return config_name
        except (OSError, json.JSONDecodeError) as e:
            print(f"匯入參數配置失敗: {e}")
            return None 
