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

    def test_capture_defaults_match_screen(self):
        c = _make_config()
        assert c.screenshot_method == "mss"

    def test_crosshair_defaults(self):
        c = _make_config()
        assert c.crosshairX == 960
        assert c.crosshairY == 540

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

    def test_auto_fire_defaults(self):
        c = _make_config()
        assert c.auto_fire_key == 0x06
        assert c.always_auto_fire is False
        assert c.auto_fire_delay == 0.0
        assert c.auto_fire_interval == 0.08
        assert c.auto_fire_target_part == "both"

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
            assert c.detect_range_size == 1440


# ============================================================
# 2. to_dict / from_dict 測試
# ============================================================

class TestConfigSerialization:
    """測試 Config 的序列化與反序列化"""

    def test_to_dict_has_all_expected_keys(self):
        c = _make_config()
        d = c.to_dict()
        # v2 grouped schema: top-level sections present, config_version stays top-level.
        for section in ('model', 'capture', 'aim', 'autofire', 'tracking',
                        'performance', 'display', 'hardware', 'ui', 'humanization'):
            assert section in d, f"Missing section: {section}"
        assert d['config_version'] == 2
        # Spot-check nested paths.
        assert 'fov_size' in d['aim']
        assert d['aim']['pid']['x']['kp'] == c.pid_kp_x
        assert d['model']['backend'] == c.inference_backend
        assert d['hardware']['devices']['makcu']['port'] == c.makcu_com_port
        assert d['display']['crosshair']['color'] == [
            c.crosshair_color_r, c.crosshair_color_g, c.crosshair_color_b]
        # Dropped/derived/state keys must NOT be persisted.
        assert 'model_input_size' not in d['model']
        assert 'current_provider' not in d['model']
        for state_key in ('disclaimer_agreed', 'first_run_complete', 'ndi_installer_ran_once'):
            assert state_key not in json.dumps(d), f"state key {state_key} leaked into config"

    def test_to_dict_values_match_instance(self):
        c = _make_config()
        c.fov_size = 333
        c.pid_kp_x = 0.5
        d = c.to_dict()
        assert d['aim']['fov_size'] == 333
        assert d['aim']['pid']['x']['kp'] == 0.5

    def test_fov_height_round_trips_independently_of_fov_size(self):
        """fov_height must persist as its own aim.fov_height path, not get
        conflated with fov_size — that's what lets the FOV become a
        rectangle instead of always tracking fov_size as a square."""
        c = _make_config()
        c.fov_size = 300
        c.fov_height = 150
        d = c.to_dict()
        assert d['aim']['fov_size'] == 300
        assert d['aim']['fov_height'] == 150

        c2 = _make_config()
        c2.from_dict(d)
        assert c2.fov_size == 300
        assert c2.fov_height == 150

    def test_fov_reduce_on_target_fields_round_trip(self):
        c = _make_config()
        c.fov_reduce_on_target_enabled = True
        c.fov_min_size_pct = 35.0
        c.fov_min_size_duration = 2.5
        d = c.to_dict()
        assert d['aim']['fov_reduce']['enabled'] is True
        assert d['aim']['fov_reduce']['min_size_pct'] == 35.0
        assert d['aim']['fov_reduce']['duration'] == 2.5

        c2 = _make_config()
        c2.from_dict(d)
        assert c2.fov_reduce_on_target_enabled is True
        assert c2.fov_min_size_pct == 35.0
        assert c2.fov_min_size_duration == 2.5

    def test_fov_reduce_on_target_defaults(self):
        c = _make_config()
        assert c.fov_reduce_on_target_enabled is False
        assert c.fov_min_size_pct == 50.0
        assert c.fov_min_size_duration == 1.0

    def test_hud_udp_fields_defaults(self):
        c = _make_config()
        assert c.hud_udp_enabled is False
        assert c.hud_udp_bind_ip == "0.0.0.0"
        assert c.hud_udp_bind_port == 5601

    def test_hud_udp_fields_round_trip(self):
        """A dedicated second UDP stream for the HUD strip (for a 2PC/OBS
        setup where the main udp stream is itself already a center crop
        that excludes the HUD region) — persisted under the ocr.* section,
        independent of hud_roi_coords."""
        c = _make_config()
        c.hud_udp_enabled = True
        c.hud_udp_bind_ip = "192.168.1.50"
        c.hud_udp_bind_port = 5602
        d = c.to_dict()
        assert d['ocr']['hud_udp_enabled'] is True
        assert d['ocr']['hud_udp_bind_ip'] == "192.168.1.50"
        assert d['ocr']['hud_udp_bind_port'] == 5602

        c2 = _make_config()
        c2.from_dict(d)
        assert c2.hud_udp_enabled is True
        assert c2.hud_udp_bind_ip == "192.168.1.50"
        assert c2.hud_udp_bind_port == 5602

    def test_kalman_adaptive_noise_enabled_default_and_round_trip(self):
        """Off by default (no behavior change for existing configs/presets
        until explicitly enabled) — persisted under aim.kalman.*, the same
        prefix as the other kalman_* fields."""
        c = _make_config()
        assert c.kalman_adaptive_noise_enabled is False

        c.kalman_adaptive_noise_enabled = True
        d = c.to_dict()
        assert d['aim']['kalman']['adaptive_noise_enabled'] is True

        c2 = _make_config()
        c2.from_dict(d)
        assert c2.kalman_adaptive_noise_enabled is True

    def test_fov_effective_size_is_runtime_only_not_persisted(self):
        """fov_effective_size/_height default to fov_size/fov_height but
        must never appear in the persisted schema — they're written every
        frame by ai_loop.py, not something a user configures or a preset
        should save/restore."""
        c = _make_config()
        assert c.fov_effective_size == c.fov_size
        assert c.fov_effective_height == c.fov_height
        c.fov_effective_size = 42.0  # simulate a mid-engagement shrink
        d = c.to_dict()
        assert 'fov_effective_size' not in d.get('aim', {})
        assert 'fov_effective_height' not in d.get('aim', {})

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

    def test_from_dict_string_bool_false_coerced_correctly(self):
        """Regression: bool("false") is True (any non-empty string is
        truthy), so a hand-edited/legacy config.json storing the *string*
        "false" for a bool field used to silently flip it to True on load."""
        c = _make_config()
        c.from_dict({'dark_mode': 'false', 'show_fov': '0', 'show_boxes': 'no'})
        assert c.dark_mode is False
        assert c.show_fov is False
        assert c.show_boxes is False

    def test_from_dict_string_bool_true_coerced_correctly(self):
        c = _make_config()
        c.from_dict({'dark_mode': 'true', 'show_fov': '1', 'show_boxes': 'yes'})
        assert c.dark_mode is True
        assert c.show_fov is True
        assert c.show_boxes is True

    def test_roundtrip_serialization(self):
        """to_dict -> from_dict 來回應保持值不變"""
        c1 = _make_config()
        c1.fov_size = 555
        c1.pid_kp_x = 0.99
        c1.mouse_click_method = "xbox"
        c1.aim_y_reduce_enabled = True
        d = c1.to_dict()

        c2 = _make_config()
        c2.from_dict(d)
        assert c2.fov_size == 555
        assert c2.pid_kp_x == 0.99
        assert c2.mouse_click_method == "xbox"
        assert c2.aim_y_reduce_enabled is True


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
            assert 'fov_size' in data['aim']
        finally:
            os.unlink(filepath)
            _sp = os.path.join(os.path.dirname(filepath), 'state.json')
            if os.path.exists(_sp):
                os.unlink(_sp)

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

    def test_save_config_drops_extra_fields(self):
        """save_config 應只寫入 to_dict() 的欄位，不保留舊版殘留的未知欄位。
        語言偏好現在儲存在 language.json（由 LanguageManager 管理），不再混入 config.json。
        """
        from core.config import save_config
        c = _make_config()
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
            json.dump({"language": "zh_tw", "ghost_key": 42}, f)
            filepath = f.name
        try:
            save_config(c, filepath)
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # Orphaned keys must not survive a save — that's the whole point of the cleanup.
            assert 'language' not in data
            assert 'ghost_key' not in data
            # config_version must be present (v2 grouped schema)
            assert data.get('config_version') == 2
        finally:
            os.unlink(filepath)
            _sp = os.path.join(os.path.dirname(filepath), 'state.json')
            if os.path.exists(_sp):
                os.unlink(_sp)

    def test_migration_v1_flat_to_v2_nested(self):
        """A legacy flat config.json loads correctly and re-saves as nested v2."""
        from core.config import load_config, save_config
        c = _make_config()
        flat = {
            'config_version': 1,
            'fov_size': 277,
            'pid_kp_x': 0.42,
            'makcu_com_port': 'COM9',
            'crosshair_color_r': 10, 'crosshair_color_g': 20, 'crosshair_color_b': 30,
            'inference_backend': 'tensorrt',
            'disclaimer_agreed': True,
        }
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
            json.dump(flat, f)
            filepath = f.name
        statepath = os.path.join(os.path.dirname(filepath), 'state.json')
        try:
            load_config(c, filepath)
            assert c.fov_size == 277
            assert c.pid_kp_x == 0.42
            assert c.makcu_com_port == 'COM9'
            assert (c.crosshair_color_r, c.crosshair_color_g, c.crosshair_color_b) == (10, 20, 30)
            assert c.inference_backend == 'tensorrt'
            # state field read via legacy back-compat
            assert c.disclaimer_agreed is True
            # re-save → on disk is now nested v2 without state keys
            save_config(c, filepath)
            with open(filepath, 'r', encoding='utf-8') as f:
                nested = json.load(f)
            assert nested['config_version'] == 2
            assert nested['aim']['fov_size'] == 277
            assert nested['model']['backend'] == 'tensorrt'
            assert 'disclaimer_agreed' not in json.dumps(nested)
        finally:
            os.unlink(filepath)
            if os.path.exists(statepath):
                os.unlink(statepath)

    def test_state_json_roundtrip(self):
        """save_state/load_state persist the state flags independently of config."""
        from core.config import save_state, load_state
        c = _make_config()
        c.disclaimer_agreed = True
        c.first_run_complete = True
        c.ndi_installer_ran_once = True
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
            statepath = f.name
        try:
            save_state(c, statepath)
            c2 = _make_config()
            assert c2.disclaimer_agreed is False  # default
            load_state(c2, statepath)
            assert c2.disclaimer_agreed is True
            assert c2.first_run_complete is True
            assert c2.ndi_installer_ran_once is True
        finally:
            os.unlink(statepath)


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

    def test_fov_larger_than_height_still_clamped_to_height(self):
        """Regression: fov_size > height used to make min_size > max_size,
        so max(min_size, min(max_size, raw)) evaluated to min_size (=
        fov_size) — violating the "must not be larger than screen height"
        invariant. The lower bound must be clamped to the upper bound
        first."""
        from core.config import _validate_detect_range_size
        c = _make_config()
        c.fov_size = 2000  # larger than height (1080)
        c.detect_range_size = 1080
        _validate_detect_range_size(c)
        assert c.detect_range_size <= 1080

    def test_fov_larger_than_height_with_small_raw_clamped_to_height(self):
        from core.config import _validate_detect_range_size
        c = _make_config()
        c.fov_size = 2000  # larger than height (1080)
        c.detect_range_size = 50  # smaller than both fov_size and height
        _validate_detect_range_size(c)
        assert c.detect_range_size == 1080

    def test_too_small_clamped_to_taller_fov_height(self):
        """The lower bound is max(fov_size, fov_height) — a rectangular FOV
        taller than it is wide must raise detect_range_size to its height,
        not just fov_size, or the square detection region couldn't contain
        the whole FOV rectangle."""
        from core.config import _validate_detect_range_size
        c = _make_config()
        c.fov_size = 100
        c.fov_height = 500
        c.detect_range_size = 200  # < fov_height, would satisfy fov_size alone
        _validate_detect_range_size(c)
        assert c.detect_range_size == 500

    def test_wider_fov_size_still_wins_over_shorter_height(self):
        from core.config import _validate_detect_range_size
        c = _make_config()
        c.fov_size = 500
        c.fov_height = 100
        c.detect_range_size = 200
        _validate_detect_range_size(c)
        assert c.detect_range_size == 500


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


# ============================================================
# 10. _validate_udp_recv_buffer_size 測試
# ============================================================

class TestValidateUdpRecvBufferSize:
    """A recv buffer smaller than the sender's largest datagram makes
    recvfrom() silently truncate frames — the truncated payload still passes
    the receiver's chunk-count completeness check, so corruption reaches
    cv2.imdecode with nothing pointing back at this setting."""

    def test_default_is_unchanged(self):
        from core.config import _validate_udp_recv_buffer_size
        c = _make_config()
        c.udp_recv_buffer_size = 65536
        _validate_udp_recv_buffer_size(c)
        assert c.udp_recv_buffer_size == 65536

    def test_larger_value_is_kept(self):
        from core.config import _validate_udp_recv_buffer_size
        c = _make_config()
        c.udp_recv_buffer_size = 262144
        _validate_udp_recv_buffer_size(c)
        assert c.udp_recv_buffer_size == 262144

    def test_below_max_datagram_is_raised(self):
        from core.config import _validate_udp_recv_buffer_size
        c = _make_config()
        # 32 KiB: smaller than the sender's 14 + 60000 byte ceiling, so
        # multi-chunk frames would arrive truncated.
        c.udp_recv_buffer_size = 32768
        _validate_udp_recv_buffer_size(c)
        assert c.udp_recv_buffer_size == 65536

    def test_covers_senders_max_datagram(self):
        """Must be at least UDP_HEADER_SIZE + UDP_MAX_PAYLOAD from
        udp_stream_filter.cpp (14 + 60000)."""
        from core.config import _validate_udp_recv_buffer_size
        c = _make_config()
        c.udp_recv_buffer_size = 1
        _validate_udp_recv_buffer_size(c)
        assert c.udp_recv_buffer_size >= 14 + 60000

    def test_garbage_value_falls_back_to_minimum(self):
        from core.config import _validate_udp_recv_buffer_size
        c = _make_config()
        c.udp_recv_buffer_size = "not-a-number"
        _validate_udp_recv_buffer_size(c)
        assert c.udp_recv_buffer_size == 65536
