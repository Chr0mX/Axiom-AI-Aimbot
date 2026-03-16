# tests/test_config.py
"""
Config 模組單元測試

測試範圍：
1. Config 類的初始化和預設值
2. to_dict / from_dict 序列化與反序列化
3. save_config / load_config 檔案讀寫
4. _validate_detect_interval 驗證
5. _validate_idle_detect_interval 驗證
6. _validate_mouse_method 驗證
7. _validate_detect_range_size 驗證
8. _validate_screenshot_method 驗證
9. _validate_screenshot_interval 驗證
"""

import json
import os
import tempfile
from unittest.mock import patch, MagicMock

import pytest


def _make_config():
    """建立一個 mock Config 以避免呼叫 _get_screen_size"""
    with patch("core.config._get_screen_size", return_value=(1920, 1080)):
        from core.config import Config
        return Config()


# ============================================================
# 1. Config 初始化與預設值測試
# ============================================================

class TestConfigInit:
    """測試 Config 初始化與預設值"""

    def test_screen_dimensions(self):
        c = _make_config()
        assert c.width == 1920
        assert c.height == 1080

    def test_center_calculations(self):
        c = _make_config()
        assert c.center_x == 960
        assert c.center_y == 540

    def test_capture_defaults_match_screen(self):
        c = _make_config()
        assert c.capture_width == 1920
        assert c.capture_height == 1080
        assert c.capture_left == 0
        assert c.capture_top == 0
        assert c.screenshot_method == "mss"

    def test_crosshair_defaults(self):
        c = _make_config()
        assert c.crosshairX == 960
        assert c.crosshairY == 540

    def test_region_dict(self):
        c = _make_config()
        assert c.region == {"top": 0, "left": 0, "width": 1920, "height": 1080}

    def test_running_defaults(self):
        c = _make_config()
        assert c.Running is True
        assert c.AimToggle is True

    def test_model_defaults(self):
        c = _make_config()
        assert c.model_input_size == 640
        assert c.model_path == os.path.join('Model', 'Roblox_8n.onnx')
        assert c.current_provider == "DmlExecutionProvider"
        assert c.dml_cpu_fallback is True

    def test_aim_keys_default(self):
        c = _make_config()
        assert c.AimKeys == [0x01, 0x06, 0x02]

    def test_fov_defaults(self):
        c = _make_config()
        assert c.fov_size == 222
        assert c.detect_range_size == 1080  # 等於 height
        assert c.detect_interval == 0.008
        assert c.screenshot_interval == 0.008

    def test_pid_defaults(self):
        c = _make_config()
        assert c.pid_kp_x == 0.26
        assert c.pid_ki_x == 0.0
        assert c.pid_kd_x == 0.0
        assert c.pid_kp_y == 0.26
        assert c.pid_ki_y == 0.0
        assert c.pid_kd_y == 0.0

    def test_mouse_method_defaults(self):
        c = _make_config()
        assert c.mouse_move_method == "mouse_event"
        assert c.mouse_click_method == "mouse_event"
        assert c.arduino_com_port == ""

    def test_xbox_defaults(self):
        c = _make_config()
        assert c.xbox_sensitivity == 1.0
        assert c.xbox_deadzone == 0.05
        assert c.xbox_auto_connect is True

    def test_auto_fire_defaults(self):
        c = _make_config()
        assert c.auto_fire_key == 0x06
        assert c.always_auto_fire is False
        assert c.auto_fire_delay == 0.0
        assert c.auto_fire_interval == 0.08
        assert c.auto_fire_target_part == "both"

    def test_bezier_defaults(self):
        c = _make_config()
        assert c.bezier_curve_enabled is False
        assert c.bezier_curve_strength == 0.35
        assert c.bezier_curve_steps == 4

    def test_tracker_defaults(self):
        c = _make_config()
        assert c.tracker_enabled is False
        assert c.tracker_prediction_time == 0.025
        assert c.tracker_smoothing_factor == 0.66

    def test_display_switch_defaults(self):
        c = _make_config()
        assert c.show_fov is True
        assert c.show_boxes is True
        assert c.show_detect_range is False
        assert c.show_status_panel is True
        assert c.status_panel_show_auto_aim is True
        assert c.status_panel_show_model is True
        assert c.status_panel_show_mouse_move is True
        assert c.status_panel_show_mouse_click is True
        assert c.status_panel_show_screenshot_method is True
        assert c.status_panel_show_screenshot_fps is True
        assert c.status_panel_show_detection_fps is True
        assert c.show_console is True

    def test_theme_defaults(self):
        c = _make_config()
        assert c.dark_mode is False
        assert c.enable_acrylic is True

    def test_disclaimer_defaults(self):
        c = _make_config()
        assert c.disclaimer_agreed is False
        assert c.first_run_complete is False

    def test_different_screen_size(self):
        """測試不同螢幕解析度"""
        with patch("core.config._get_screen_size", return_value=(2560, 1440)):
            from core.config import Config
            c = Config()
            assert c.width == 2560
            assert c.height == 1440
            assert c.center_x == 1280
            assert c.center_y == 720
            assert c.detect_range_size == 1440


# ============================================================
# 2. to_dict / from_dict 測試
# ============================================================

class TestConfigSerialization:
    """測試 Config 的序列化與反序列化"""

    def test_to_dict_has_all_expected_keys(self):
        c = _make_config()
        d = c.to_dict()
        expected_keys = [
            'fov_size', 'detect_range_size', 'model_path', 'model_input_size',
            'current_provider', 'dml_cpu_fallback', 'pid_kp_x', 'pid_ki_x',
            'pid_kd_x', 'pid_kp_y', 'pid_ki_y', 'pid_kd_y', 'aim_part',
            'AimKeys', 'auto_fire_key', 'always_auto_fire', 'auto_fire_delay',
            'auto_fire_interval', 'auto_fire_target_part', 'min_confidence',
            'show_confidence', 'screenshot_method', 'mouse_move_method', 'mouse_click_method',
            'screenshot_interval',
            'arduino_com_port', 'xbox_sensitivity', 'xbox_deadzone',
            'xbox_auto_connect', 'dark_mode', 'bezier_curve_enabled',
            'tracker_enabled', 'enable_acrylic',
        ]
        for key in expected_keys:
            assert key in d, f"Missing key: {key}"

    def test_to_dict_values_match_instance(self):
        c = _make_config()
        c.fov_size = 333
        c.pid_kp_x = 0.5
        d = c.to_dict()
        assert d['fov_size'] == 333
        assert d['pid_kp_x'] == 0.5

    def test_from_dict_updates_attributes(self):
        c = _make_config()
        c.from_dict({
            'fov_size': 444,
            'pid_kp_x': 0.8,
            'screenshot_method': 'mss',
            'screenshot_interval': 0.012,
            'mouse_move_method': 'arduino',
            'dark_mode': True,
        })
        assert c.fov_size == 444
        assert c.pid_kp_x == 0.8
        assert c.screenshot_method == 'mss'
        assert c.screenshot_interval == 0.012
        assert c.mouse_move_method == 'arduino'
        assert c.dark_mode is True

    def test_from_dict_ignores_unknown_keys(self):
        c = _make_config()
        c.from_dict({'nonexistent_key': 'value'})
        assert not hasattr(c, 'nonexistent_key') or getattr(c, 'nonexistent_key', None) != 'value'

    def test_roundtrip_serialization(self):
        """to_dict -> from_dict 來回應保持值不變"""
        c1 = _make_config()
        c1.fov_size = 555
        c1.pid_kp_x = 0.99
        c1.mouse_click_method = "xbox"
        c1.bezier_curve_enabled = True
        d = c1.to_dict()

        c2 = _make_config()
        c2.from_dict(d)
        assert c2.fov_size == 555
        assert c2.pid_kp_x == 0.99
        assert c2.mouse_click_method == "xbox"
        assert c2.bezier_curve_enabled is True


# ============================================================
# 3. save_config / load_config 檔案讀寫測試
# ============================================================

class TestConfigFileIO:
    """測試 save_config 和 load_config"""

    def test_save_config_creates_file(self):
        from core.config import save_config
        c = _make_config()
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filepath = f.name
        try:
            result = save_config(c, filepath)
            assert result is True
            assert os.path.exists(filepath)
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            assert 'fov_size' in data
        finally:
            os.unlink(filepath)

    def test_load_config_reads_file(self):
        from core.config import save_config, load_config
        c = _make_config()
        c.fov_size = 999
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filepath = f.name
        try:
            save_config(c, filepath)
            c2 = _make_config()
            result = load_config(c2, filepath)
            assert result is True
            assert c2.fov_size == 999
        finally:
            os.unlink(filepath)

    def test_load_config_file_not_found(self):
        from core.config import load_config
        c = _make_config()
        result = load_config(c, '/nonexistent/path.json')
        assert result is False

    def test_load_config_invalid_json(self):
        from core.config import load_config
        c = _make_config()
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
            f.write("NOT VALID JSON {{{")
            filepath = f.name
        try:
            result = load_config(c, filepath)
            assert result is False
        finally:
            os.unlink(filepath)

    def test_save_config_preserves_extra_fields(self):
        """save_config 應保留 config.json 中不屬於 Config 的欄位（如 language）"""
        from core.config import save_config
        c = _make_config()
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
            json.dump({"language": "zh_tw", "extra_field": 42}, f)
            filepath = f.name
        try:
            save_config(c, filepath)
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            assert data.get('language') == 'zh_tw'
            assert data.get('extra_field') == 42
        finally:
            os.unlink(filepath)


# ============================================================
# 4. _validate_detect_interval 測試
# ============================================================

class TestValidateDetectInterval:
    """測試檢測間隔驗證"""

    def test_normal_value_unchanged(self):
        from core.config import _validate_detect_interval
        c = _make_config()
        c.detect_interval = 0.02  # 20ms，正常範圍
        _validate_detect_interval(c)
        assert c.detect_interval == 0.02

    def test_too_small_corrected_to_1ms(self):
        from core.config import _validate_detect_interval
        c = _make_config()
        c.detect_interval = 0.0001  # 0.1ms，太小
        _validate_detect_interval(c)
        assert c.detect_interval == 0.001

    def test_too_large_corrected_to_100ms(self):
        from core.config import _validate_detect_interval
        c = _make_config()
        c.detect_interval = 0.5  # 500ms，太大
        _validate_detect_interval(c)
        assert c.detect_interval == 0.1

    def test_boundary_1ms_unchanged(self):
        from core.config import _validate_detect_interval
        c = _make_config()
        c.detect_interval = 0.001  # 剛好 1ms
        _validate_detect_interval(c)
        assert c.detect_interval == 0.001

    def test_boundary_100ms_unchanged(self):
        from core.config import _validate_detect_interval
        c = _make_config()
        c.detect_interval = 0.1  # 剛好 100ms
        _validate_detect_interval(c)
        assert c.detect_interval == 0.1


# ============================================================
# 5. _validate_idle_detect_interval 測試
# ============================================================

class TestValidateIdleDetectInterval:
    """測試閒置檢測間隔驗證"""

    def test_normal_value_unchanged(self):
        from core.config import _validate_idle_detect_interval
        c = _make_config()
        c.idle_detect_interval = 0.05  # 50ms
        _validate_idle_detect_interval(c)
        assert c.idle_detect_interval == 0.05

    def test_too_small_corrected(self):
        from core.config import _validate_idle_detect_interval
        c = _make_config()
        c.idle_detect_interval = 0.001  # 1ms < 5ms
        _validate_idle_detect_interval(c)
        assert c.idle_detect_interval == 0.005

    def test_too_large_corrected(self):
        from core.config import _validate_idle_detect_interval
        c = _make_config()
        c.idle_detect_interval = 1.0  # 1000ms > 500ms
        _validate_idle_detect_interval(c)
        assert c.idle_detect_interval == 0.5


# ============================================================
# 6. _validate_mouse_method 測試
# ============================================================

class TestValidateMouseMethod:
    """測試滑鼠方式驗證"""

    def test_hardware_move_corrected(self):
        from core.config import _validate_mouse_method
        c = _make_config()
        c.mouse_move_method = "hardware"
        _validate_mouse_method(c)
        assert c.mouse_move_method == "mouse_event"

    def test_valid_move_methods_preserved(self):
        from core.config import _validate_mouse_method
        for method in ["mouse_event", "sendinput", "ddxoft", "arduino", "xbox"]:
            c = _make_config()
            c.mouse_move_method = method
            _validate_mouse_method(c)
            # mouse_move_method 只修正 'hardware'，其他不管
            if method == "hardware":
                assert c.mouse_move_method == "mouse_event"
            else:
                assert c.mouse_move_method == method

    def test_invalid_click_method_corrected(self):
        from core.config import _validate_mouse_method
        c = _make_config()
        c.mouse_click_method = "invalid_xyz"
        _validate_mouse_method(c)
        assert c.mouse_click_method == "mouse_event"

    @pytest.mark.parametrize("method", ["mouse_event", "sendinput", "ddxoft", "arduino", "xbox"])
    def test_valid_click_methods_preserved(self, method):
        from core.config import _validate_mouse_method
        c = _make_config()
        c.mouse_click_method = method
        _validate_mouse_method(c)
        assert c.mouse_click_method == method


# ============================================================
# 7. _validate_detect_range_size 測試
# ============================================================

class TestValidateDetectRangeSize:
    """測試 AI 偵測範圍驗證"""

    def test_normal_value_unchanged(self):
        from core.config import _validate_detect_range_size
        c = _make_config()
        c.fov_size = 222
        c.detect_range_size = 640
        _validate_detect_range_size(c)
        assert c.detect_range_size == 640

    def test_too_small_clamped_to_fov(self):
        from core.config import _validate_detect_range_size
        c = _make_config()
        c.fov_size = 222
        c.detect_range_size = 100  # 小於 fov_size
        _validate_detect_range_size(c)
        assert c.detect_range_size == 222

    def test_too_large_clamped_to_height(self):
        from core.config import _validate_detect_range_size
        c = _make_config()
        c.detect_range_size = 9999  # 大於 height (1080)
        _validate_detect_range_size(c)
        assert c.detect_range_size == 1080

    def test_equal_to_fov_preserved(self):
        from core.config import _validate_detect_range_size
        c = _make_config()
        c.fov_size = 222
        c.detect_range_size = 222
        _validate_detect_range_size(c)
        assert c.detect_range_size == 222

    def test_equal_to_height_preserved(self):
        from core.config import _validate_detect_range_size
        c = _make_config()
        c.detect_range_size = 1080
        _validate_detect_range_size(c)
        assert c.detect_range_size == 1080


# ============================================================
# 8. _validate_screenshot_method 測試
# ============================================================

class TestValidateScreenshotMethod:
    """測試截圖方式驗證"""

    def test_dxcam_is_preserved(self):
        from core.config import _validate_screenshot_method
        c = _make_config()
        c.screenshot_method = 'dxcam'
        _validate_screenshot_method(c)
        assert c.screenshot_method == 'dxcam'

    def test_invalid_method_falls_back_to_mss(self):
        from core.config import _validate_screenshot_method
        c = _make_config()
        c.screenshot_method = 'unknown_backend'
        _validate_screenshot_method(c)
        assert c.screenshot_method == 'mss'


# ============================================================
# 9. _validate_screenshot_interval 測試
# ============================================================

class TestValidateScreenshotInterval:
    """測試截圖間隔驗證"""

    def test_normal_value_unchanged(self):
        from core.config import _validate_screenshot_interval
        c = _make_config()
        c.screenshot_interval = 0.008
        _validate_screenshot_interval(c)
        assert c.screenshot_interval == 0.008

    def test_too_small_corrected_to_1ms(self):
        from core.config import _validate_screenshot_interval
        c = _make_config()
        c.screenshot_interval = 0.0001
        _validate_screenshot_interval(c)
        assert c.screenshot_interval == 0.001

    def test_too_large_corrected_to_100ms(self):
        from core.config import _validate_screenshot_interval
        c = _make_config()
        c.screenshot_interval = 0.5
        _validate_screenshot_interval(c)
        assert c.screenshot_interval == 0.1
