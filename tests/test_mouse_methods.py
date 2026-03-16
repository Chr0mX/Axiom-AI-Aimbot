# tests/test_mouse_methods.py
"""
滑鼠移動方式與點擊方式的完整測試套件

測試範圍：
1. mouse_move.py  - send_mouse_move_sendinput, send_mouse_move_mouse_event
2. ddxoft_mouse.py - DDXoftMouse 類, send_mouse_move_ddxoft
3. arduino_mouse.py - ArduinoMouse 類, send_mouse_move_arduino, send_mouse_click_arduino
4. xbox_controller.py - XboxController 類, send_mouse_move_xbox, send_mouse_click_xbox
5. mouse_click.py - 所有點擊方式 + send_mouse_click 調度
6. __init__.py - send_mouse_move 調度
7. config.py - _validate_mouse_method 驗證邏輯
8. ai_loop.py - cached_mouse_move_method 邏輯
9. auto_fire.py - mouse_click_method 使用邏輯
"""

import sys
import os
import struct
import time
from unittest import mock
from unittest.mock import patch, MagicMock, PropertyMock

import pytest

# 確保 src 目錄在路徑中
src_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)


# ============================================================
# 1. mouse_move.py 測試
# ============================================================

class TestMouseMoveSendInput:
    """測試 SendInput 滑鼠移動"""

    @patch("ctypes.windll.user32.SendInput")
    def test_sendinput_basic(self, mock_send):
        from win_utils.mouse_move import send_mouse_move_sendinput
        send_mouse_move_sendinput(10, 20)
        mock_send.assert_called_once()

    @patch("ctypes.windll.user32.SendInput")
    def test_sendinput_zero(self, mock_send):
        from win_utils.mouse_move import send_mouse_move_sendinput
        send_mouse_move_sendinput(0, 0)
        mock_send.assert_called_once()

    @patch("ctypes.windll.user32.SendInput")
    def test_sendinput_negative(self, mock_send):
        from win_utils.mouse_move import send_mouse_move_sendinput
        send_mouse_move_sendinput(-5, -10)
        mock_send.assert_called_once()


class TestMouseMoveMouseEvent:
    """測試 mouse_event 滑鼠移動"""

    @patch("win32api.mouse_event")
    def test_mouse_event_basic(self, mock_me):
        from win_utils.mouse_move import send_mouse_move_mouse_event
        send_mouse_move_mouse_event(10, 20)
        mock_me.assert_called_once()

    @patch("win32api.mouse_event", side_effect=Exception("fail"))
    def test_mouse_event_exception_suppressed(self, mock_me):
        """mouse_event 異常應被靜默捕獲"""
        from win_utils.mouse_move import send_mouse_move_mouse_event
        send_mouse_move_mouse_event(10, 20)  # 不應拋出異常


# ============================================================
# 2. DDXoft 滑鼠測試
# ============================================================

class TestDDXoftMouse:
    """測試 DDXoftMouse 類"""

    def _make_ddxoft(self):
        from win_utils.ddxoft_mouse import DDXoftMouse
        return DDXoftMouse()

    def test_initial_state(self):
        d = self._make_ddxoft()
        assert d.available is False
        assert d.success_count == 0
        assert d.failure_count == 0
        assert d.last_status is None

    def test_ensure_initialized_returns_false_when_dll_missing(self):
        d = self._make_ddxoft()
        d.subsequent_init_failed = True
        assert d.ensure_initialized() is False

    def test_ensure_initialized_returns_true_when_available(self):
        d = self._make_ddxoft()
        d.available = True
        assert d.ensure_initialized() is True

    def test_move_relative_fails_without_init(self):
        d = self._make_ddxoft()
        d.subsequent_init_failed = True
        result = d.move_relative(10, 20)
        assert result is False
        assert d.failure_count == 1
        assert d.last_status == "DLL_NOT_AVAILABLE"

    def test_move_relative_clamps_values(self):
        d = self._make_ddxoft()
        d.available = True
        d.dll = MagicMock()
        d.dll.DD_movR.return_value = 1
        d.move_relative(99999, -99999)
        d.dll.DD_movR.assert_called_once_with(32767, -32767)

    def test_move_relative_success(self):
        d = self._make_ddxoft()
        d.available = True
        d.dll = MagicMock()
        d.dll.DD_movR.return_value = 1
        assert d.move_relative(5, 10) is True
        assert d.success_count == 1
        assert d.last_status == "SUCCESS"

    def test_move_relative_failure_code(self):
        d = self._make_ddxoft()
        d.available = True
        d.dll = MagicMock()
        d.dll.DD_movR.return_value = 0
        assert d.move_relative(5, 10) is False
        assert d.failure_count == 1

    def test_click_left_success(self):
        d = self._make_ddxoft()
        d.available = True
        d.dll = MagicMock()
        d.dll.DD_btn.return_value = 1
        assert d.click_left() is True
        assert d.success_count == 1
        assert d.last_status == "CLICK_SUCCESS"

    def test_click_left_fails_without_init(self):
        d = self._make_ddxoft()
        d.subsequent_init_failed = True
        assert d.click_left() is False
        assert d.failure_count == 1

    def test_get_statistics(self):
        d = self._make_ddxoft()
        d.success_count = 5
        d.failure_count = 2
        d.last_status = "SUCCESS"
        stats = d.get_statistics()
        assert stats['total_count'] == 7
        assert abs(stats['success_rate'] - (5/7*100)) < 0.1

    def test_reset_statistics(self):
        d = self._make_ddxoft()
        d.success_count = 10
        d.failure_count = 3
        d.last_status = "X"
        d.reset_statistics()
        assert d.success_count == 0
        assert d.failure_count == 0
        assert d.last_status is None


class TestSendMouseMoveDdxoft:
    """測試 send_mouse_move_ddxoft 函數"""

    @patch("win_utils.ddxoft_mouse.ddxoft_mouse")
    @patch("win_utils.ddxoft_mouse.send_mouse_move_mouse_event")
    def test_fallback_when_not_initialized(self, mock_fallback, mock_ddx):
        mock_ddx.ensure_initialized.return_value = False
        from win_utils.ddxoft_mouse import send_mouse_move_ddxoft
        send_mouse_move_ddxoft(10, 20)
        mock_fallback.assert_called_once_with(10, 20)

    @patch("win_utils.ddxoft_mouse.ddxoft_mouse")
    def test_success_no_fallback(self, mock_ddx):
        mock_ddx.ensure_initialized.return_value = True
        mock_ddx.move_relative.return_value = True
        from win_utils.ddxoft_mouse import send_mouse_move_ddxoft
        send_mouse_move_ddxoft(10, 20)
        mock_ddx.move_relative.assert_called_once_with(10, 20)


# ============================================================
# 3. Arduino 滑鼠測試
# ============================================================

class TestArduinoMouse:
    """測試 ArduinoMouse 類"""

    def _make_arduino(self):
        from win_utils.arduino_mouse import ArduinoMouse
        return ArduinoMouse()

    def test_initial_state(self):
        a = self._make_arduino()
        assert a.is_connected() is False
        assert a.com_port == ""

    def test_move_no_op_when_disconnected(self):
        a = self._make_arduino()
        a.move(10, 20)  # 不應拋異常

    def test_click_no_op_when_disconnected(self):
        a = self._make_arduino()
        a.click(1)  # 不應拋異常

    def test_move_clamps_values(self):
        a = self._make_arduino()
        mock_serial = MagicMock()
        mock_serial.is_open = True
        a._serial = mock_serial
        a._connected = True
        a.move(200, -200)
        expected = struct.pack('bbb', 1, 127, -128)
        mock_serial.write.assert_called_once_with(expected)

    def test_move_sends_correct_data(self):
        a = self._make_arduino()
        mock_serial = MagicMock()
        mock_serial.is_open = True
        a._serial = mock_serial
        a._connected = True
        a.move(5, -3)
        expected = struct.pack('bbb', 1, 5, -3)
        mock_serial.write.assert_called_once_with(expected)

    def test_click_sends_correct_data(self):
        a = self._make_arduino()
        mock_serial = MagicMock()
        mock_serial.is_open = True
        a._serial = mock_serial
        a._connected = True
        a.click(1)
        expected = struct.pack('bbb', 2, 1, 0)
        mock_serial.write.assert_called_once_with(expected)

    def test_disconnect(self):
        a = self._make_arduino()
        mock_serial = MagicMock()
        mock_serial.is_open = True
        a._serial = mock_serial
        a._connected = True
        a.disconnect()
        assert a.is_connected() is False
        mock_serial.close.assert_called_once()


# ============================================================
# 4. Xbox 控制器測試
# ============================================================

class TestXboxController:
    """測試 XboxController 類"""

    def _make_xbox(self):
        from win_utils.xbox_controller import XboxController
        return XboxController()

    def test_initial_state(self):
        x = self._make_xbox()
        assert x.is_connected() is False
        assert x.sensitivity == 1.0
        assert x.deadzone == 0.05

    def test_move_right_stick_fails_without_connect(self):
        x = self._make_xbox()
        x._init_attempted = True
        # 模擬 connect 失敗
        with patch.object(x, 'connect', return_value=False):
            assert x.move_right_stick(10, 20) is False

    def test_move_right_stick_success(self):
        x = self._make_xbox()
        mock_gp = MagicMock()
        x._gamepad = mock_gp
        x._connected = True
        x.stick_duration = 0  # 避免 sleep
        result = x.move_right_stick(25, 25)
        assert result is True
        assert mock_gp.right_joystick_float.call_count == 2  # 設定 + 歸零
        assert mock_gp.update.call_count == 2

    def test_deadzone_filters_small_input(self):
        x = self._make_xbox()
        mock_gp = MagicMock()
        x._gamepad = mock_gp
        x._connected = True
        x.deadzone = 0.5
        x.stick_duration = 0
        result = x.move_right_stick(1, 1)  # 1/50=0.02 < 0.5 deadzone
        assert result is True
        # 不應呼叫 right_joystick_float（因死區過濾）
        mock_gp.right_joystick_float.assert_not_called()

    def test_disconnect(self):
        x = self._make_xbox()
        mock_gp = MagicMock()
        x._gamepad = mock_gp
        x._connected = True
        x.disconnect()
        assert x.is_connected() is False
        mock_gp.reset.assert_called_once()

    def test_pull_right_trigger(self):
        x = self._make_xbox()
        mock_gp = MagicMock()
        x._gamepad = mock_gp
        x._connected = True
        assert x.pull_right_trigger(1.0) is True
        mock_gp.right_trigger_float.assert_called_once_with(value_float=1.0)

    def test_get_statistics(self):
        x = self._make_xbox()
        x._move_count = 10
        x._error_count = 2
        stats = x.get_statistics()
        assert stats["move_count"] == 10
        assert stats["error_count"] == 2


# ============================================================
# 5. mouse_click.py 測試
# ============================================================

class TestMouseClick:
    """測試所有滑鼠點擊函數"""

    @patch("win32api.mouse_event")
    def test_click_mouse_event(self, mock_me):
        from win_utils.mouse_click import send_mouse_click_mouse_event
        send_mouse_click_mouse_event()
        assert mock_me.call_count == 2  # down + up

    @patch("win32api.mouse_event")
    def test_click_sendinput(self, mock_me):
        from win_utils.mouse_click import send_mouse_click_sendinput
        send_mouse_click_sendinput()
        assert mock_me.call_count == 2

    @patch("win_utils.mouse_click.send_mouse_click_sendinput")
    def test_click_hardware_fallback(self, mock_si):
        """hardware 模式應回退到 sendinput"""
        from win_utils.mouse_click import send_mouse_click_hardware
        import win_utils.mouse_click as mc
        mc._hardware_not_impl_warned = False
        send_mouse_click_hardware()
        mock_si.assert_called_once()


class TestSendMouseClickDispatch:
    """測試 send_mouse_click 調度函數"""

    @patch("win_utils.mouse_click.send_mouse_click_sendinput")
    def test_dispatch_sendinput(self, mock_fn):
        from win_utils.mouse_click import send_mouse_click
        send_mouse_click("sendinput")
        mock_fn.assert_called_once()

    @patch("win_utils.mouse_click.send_mouse_click_mouse_event")
    def test_dispatch_mouse_event(self, mock_fn):
        from win_utils.mouse_click import send_mouse_click
        send_mouse_click("mouse_event")
        mock_fn.assert_called_once()

    @patch("win_utils.mouse_click.send_mouse_click_hardware")
    def test_dispatch_hardware(self, mock_fn):
        from win_utils.mouse_click import send_mouse_click
        send_mouse_click("hardware")
        mock_fn.assert_called_once()

    @patch("win_utils.mouse_click.send_mouse_click_ddxoft")
    def test_dispatch_ddxoft(self, mock_fn):
        mock_fn.return_value = True
        from win_utils.mouse_click import send_mouse_click
        send_mouse_click("ddxoft")
        mock_fn.assert_called_once()

    @patch("win_utils.mouse_click.send_mouse_click_ddxoft")
    def test_dispatch_unknown_defaults_to_ddxoft(self, mock_fn):
        mock_fn.return_value = True
        from win_utils.mouse_click import send_mouse_click
        send_mouse_click("unknown_method")
        mock_fn.assert_called_once()

    @patch("win_utils.mouse_click.send_mouse_click_mouse_event")
    def test_dispatch_exception_fallback(self, mock_me):
        """任何異常應回退到 mouse_event"""
        from win_utils.mouse_click import send_mouse_click
        with patch("win_utils.mouse_click.send_mouse_click_sendinput", side_effect=Exception("boom")):
            result = send_mouse_click("sendinput")
            assert result is True
            mock_me.assert_called()


# ============================================================
# 6. send_mouse_move 調度測試
# ============================================================

class TestSendMouseMoveDispatch:
    """測試 __init__.py 中的 send_mouse_move 調度"""

    @patch("win_utils.send_mouse_move_sendinput")
    def test_dispatch_sendinput(self, mock_fn):
        from win_utils import send_mouse_move
        send_mouse_move(10, 20, method="sendinput")
        mock_fn.assert_called_once_with(10, 20)

    @patch("win_utils.send_mouse_move_mouse_event")
    def test_dispatch_mouse_event(self, mock_fn):
        from win_utils import send_mouse_move
        send_mouse_move(10, 20, method="mouse_event")
        mock_fn.assert_called_once_with(10, 20)

    @patch("win_utils.send_mouse_move_ddxoft")
    def test_dispatch_ddxoft(self, mock_fn):
        from win_utils import send_mouse_move
        send_mouse_move(10, 20, method="ddxoft")
        mock_fn.assert_called_once_with(10, 20)

    @patch("win_utils.send_mouse_move_arduino")
    def test_dispatch_arduino(self, mock_fn):
        from win_utils import send_mouse_move
        send_mouse_move(10, 20, method="arduino")
        mock_fn.assert_called_once_with(10, 20)

    @patch("win_utils.send_mouse_move_xbox")
    def test_dispatch_xbox(self, mock_fn):
        from win_utils import send_mouse_move
        send_mouse_move(10, 20, method="xbox")
        mock_fn.assert_called_once_with(10, 20)

    @patch("win_utils.send_mouse_move_mouse_event")
    def test_dispatch_unknown_defaults_mouse_event(self, mock_fn):
        from win_utils import send_mouse_move
        send_mouse_move(10, 20, method="nonexistent")
        mock_fn.assert_called_once_with(10, 20)

    @patch("win_utils.send_mouse_move_mouse_event")
    def test_skip_tiny_movement(self, mock_fn):
        """dx, dy 都 < 1 時應跳過"""
        from win_utils import send_mouse_move
        send_mouse_move(0.5, 0.3, method="mouse_event")
        mock_fn.assert_not_called()

    @patch("win_utils.send_mouse_move_mouse_event")
    def test_zero_movement_skipped(self, mock_fn):
        from win_utils import send_mouse_move
        send_mouse_move(0, 0, method="mouse_event")
        mock_fn.assert_not_called()


# ============================================================
# 7. Config 驗證測試
# ============================================================

class TestConfigValidation:
    """測試 config.py 中滑鼠方式的驗證邏輯"""

    def _make_config(self):
        """建立一個 mock Config 以避免呼叫 _get_screen_size"""
        with patch("core.config._get_screen_size", return_value=(1920, 1080)):
            from core.config import Config
            return Config()

    def test_default_mouse_move_method(self):
        c = self._make_config()
        assert c.mouse_move_method == "mouse_event"

    def test_default_mouse_click_method(self):
        c = self._make_config()
        assert c.mouse_click_method == "mouse_event"

    def test_validate_hardware_move_method_corrected(self):
        from core.config import _validate_mouse_method
        c = self._make_config()
        c.mouse_move_method = "hardware"
        _validate_mouse_method(c)
        assert c.mouse_move_method == "mouse_event"

    def test_validate_invalid_click_method_corrected(self):
        from core.config import _validate_mouse_method
        c = self._make_config()
        c.mouse_click_method = "invalid_method"
        _validate_mouse_method(c)
        assert c.mouse_click_method == "mouse_event"

    @pytest.mark.parametrize("method", ["mouse_event", "sendinput", "ddxoft", "arduino", "xbox"])
    def test_valid_click_methods_preserved(self, method):
        from core.config import _validate_mouse_method
        c = self._make_config()
        c.mouse_click_method = method
        _validate_mouse_method(c)
        assert c.mouse_click_method == method

    def test_to_dict_includes_mouse_methods(self):
        c = self._make_config()
        c.mouse_move_method = "ddxoft"
        c.mouse_click_method = "arduino"
        d = c.to_dict()
        assert d["mouse_move_method"] == "ddxoft"
        assert d["mouse_click_method"] == "arduino"

    def test_from_dict_loads_mouse_methods(self):
        c = self._make_config()
        c.from_dict({"mouse_move_method": "arduino", "mouse_click_method": "xbox"})
        assert c.mouse_move_method == "arduino"
        assert c.mouse_click_method == "xbox"


# ============================================================
# 8. Xbox 公開函數測試
# ============================================================

class TestXboxPublicFunctions:
    """測試 xbox_controller.py 的公開函數"""

    def test_set_sensitivity_clamp(self):
        from win_utils.xbox_controller import set_xbox_sensitivity, xbox_controller
        set_xbox_sensitivity(10.0)
        assert xbox_controller.sensitivity == 5.0
        set_xbox_sensitivity(0.01)
        assert xbox_controller.sensitivity == 0.1

    def test_set_deadzone_clamp(self):
        from win_utils.xbox_controller import set_xbox_deadzone, xbox_controller
        set_xbox_deadzone(1.0)
        assert xbox_controller.deadzone == 0.5
        set_xbox_deadzone(-0.5)
        assert xbox_controller.deadzone == 0.0


# ============================================================
# 9. DDXoft 公開函數測試
# ============================================================

class TestDdxoftPublicFunctions:
    """測試 ddxoft_mouse.py 的公開接口函數"""

    @patch("win_utils.ddxoft_mouse.ddxoft_mouse")
    def test_ensure_ddxoft_ready(self, mock_ddx):
        mock_ddx.ensure_initialized.return_value = True
        from win_utils.ddxoft_mouse import ensure_ddxoft_ready
        assert ensure_ddxoft_ready() is True

    @patch("win_utils.ddxoft_mouse.ddxoft_mouse")
    def test_get_ddxoft_statistics(self, mock_ddx):
        mock_ddx.get_statistics.return_value = {"success_count": 10}
        from win_utils.ddxoft_mouse import get_ddxoft_statistics
        assert get_ddxoft_statistics()["success_count"] == 10

    def test_reset_ddxoft_statistics(self):
        from win_utils.ddxoft_mouse import reset_ddxoft_statistics
        reset_ddxoft_statistics()  # 不應拋異常


# ============================================================
# 10. 整合場景測試
# ============================================================

class TestIntegrationScenarios:
    """整合測試：模擬實際使用場景"""

    @patch("win32api.mouse_event")
    def test_click_ddxoft_fallback_chain(self, mock_me):
        """ddxoft 點擊失敗時應回退到 mouse_event"""
        from win_utils.mouse_click import send_mouse_click_ddxoft
        with patch("win_utils.mouse_click.ddxoft_mouse") as mock_ddx:
            mock_ddx.ensure_initialized.return_value = True
            mock_ddx.click_left.return_value = False
            result = send_mouse_click_ddxoft()
            assert result is True
            mock_me.assert_called()  # 回退到 mouse_event

    @patch("win32api.mouse_event")
    def test_click_ddxoft_init_fail_fallback(self, mock_me):
        """ddxoft 初始化失敗時應回退到 mouse_event"""
        from win_utils.mouse_click import send_mouse_click_ddxoft
        with patch("win_utils.mouse_click.ddxoft_mouse") as mock_ddx:
            mock_ddx.ensure_initialized.return_value = False
            result = send_mouse_click_ddxoft()
            assert result is True
            mock_me.assert_called()

    def test_config_save_load_roundtrip(self):
        """配置的 mouse 方法在存取後應保持不變"""
        with patch("core.config._get_screen_size", return_value=(1920, 1080)):
            from core.config import Config
            c = Config()
            c.mouse_move_method = "arduino"
            c.mouse_click_method = "xbox"
            d = c.to_dict()
            c2 = Config()
            c2.from_dict(d)
            assert c2.mouse_move_method == "arduino"
            assert c2.mouse_click_method == "xbox"
