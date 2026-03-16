# tests/test_makcu_mouse.py
"""
MAKCU KM Host 滑鼠控制模組測試套件

測試範圍：
1. MakcuMouse 類 - connect, disconnect, is_connected, move, click
2. 模組級函式 - send_mouse_move_makcu, send_mouse_click_makcu, connect_makcu, etc.
3. ASCII 命令格式驗證
4. send_mouse_move / send_mouse_click 調度包含 makcu
5. config _validate_mouse_method 包含 makcu
"""

import sys
import os
from unittest import mock
from unittest.mock import patch, MagicMock, PropertyMock, call

import pytest

# 確保 src 目錄在路徑中
src_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)


# ============================================================
# 1. MakcuMouse 類測試
# ============================================================

class TestMakcuMouseConnect:
    """測試 MAKCU 連線/斷線"""

    @patch("win_utils.makcu_mouse.serial.Serial")
    def test_connect_success(self, mock_serial_cls):
        from win_utils.makcu_mouse import MakcuMouse
        mock_ser = MagicMock()
        mock_ser.is_open = True
        mock_serial_cls.return_value = mock_ser

        m = MakcuMouse()
        result = m.connect("COM3")

        assert result is True
        assert m.is_connected() is True
        assert m.com_port == "COM3"
        mock_serial_cls.assert_called_once_with("COM3", 115200, timeout=0.1, write_timeout=0.1)
        # Should send version then echo-off command on connect
        mock_ser.write.assert_any_call(b"km.version()\r\n")
        mock_ser.write.assert_any_call(b"km.echo(0)\r\n")

    @patch("win_utils.makcu_mouse.serial.Serial", side_effect=Exception("port busy"))
    def test_connect_failure(self, mock_serial_cls):
        from win_utils.makcu_mouse import MakcuMouse
        m = MakcuMouse()
        result = m.connect("COM99")

        assert result is False
        assert m.is_connected() is False

    @patch("win_utils.makcu_mouse.serial.Serial")
    def test_disconnect(self, mock_serial_cls):
        from win_utils.makcu_mouse import MakcuMouse
        mock_ser = MagicMock()
        mock_ser.is_open = True
        mock_serial_cls.return_value = mock_ser

        m = MakcuMouse()
        m.connect("COM3")
        m.disconnect()

        assert m.is_connected() is False
        mock_ser.close.assert_called()

    @patch("win_utils.makcu_mouse.serial.Serial")
    def test_reconnect_closes_old(self, mock_serial_cls):
        """再次連線應先關閉舊連線"""
        from win_utils.makcu_mouse import MakcuMouse
        mock_ser1 = MagicMock()
        mock_ser1.is_open = True
        mock_ser2 = MagicMock()
        mock_ser2.is_open = True
        mock_serial_cls.side_effect = [mock_ser1, mock_ser2]

        m = MakcuMouse()
        m.connect("COM3")
        m.connect("COM4")

        mock_ser1.close.assert_called()
        assert m.com_port == "COM4"


class TestMakcuMouseMove:
    """測試 MAKCU 滑鼠移動指令格式"""

    @patch("win_utils.makcu_mouse.serial.Serial")
    def test_move_basic(self, mock_serial_cls):
        from win_utils.makcu_mouse import MakcuMouse
        mock_ser = MagicMock()
        mock_ser.is_open = True
        mock_serial_cls.return_value = mock_ser

        m = MakcuMouse()
        m.connect("COM3")
        mock_ser.write.reset_mock()

        m.move(10, -3)

        mock_ser.write.assert_called_once_with(b"km.move(10,-3)\r\n")

    @patch("win_utils.makcu_mouse.serial.Serial")
    def test_move_large_values(self, mock_serial_cls):
        """MAKCU 支援 int16 範圍（遠大於 Arduino 的 -128~127）"""
        from win_utils.makcu_mouse import MakcuMouse
        mock_ser = MagicMock()
        mock_ser.is_open = True
        mock_serial_cls.return_value = mock_ser

        m = MakcuMouse()
        m.connect("COM3")
        mock_ser.write.reset_mock()

        m.move(500, -300)
        mock_ser.write.assert_called_once_with(b"km.move(500,-300)\r\n")

    @patch("win_utils.makcu_mouse.serial.Serial")
    def test_move_clamps_to_int16(self, mock_serial_cls):
        """超過 int16 範圍的值應被 clamp"""
        from win_utils.makcu_mouse import MakcuMouse
        mock_ser = MagicMock()
        mock_ser.is_open = True
        mock_serial_cls.return_value = mock_ser

        m = MakcuMouse()
        m.connect("COM3")
        mock_ser.write.reset_mock()

        m.move(50000, -50000)
        mock_ser.write.assert_called_once_with(b"km.move(32767,-32768)\r\n")

    def test_move_not_connected(self):
        """未連線時移動不應報錯"""
        from win_utils.makcu_mouse import MakcuMouse
        m = MakcuMouse()
        m.move(10, 20)  # 不應拋出異常

    @patch("win_utils.makcu_mouse.serial.Serial")
    def test_move_zero(self, mock_serial_cls):
        from win_utils.makcu_mouse import MakcuMouse
        mock_ser = MagicMock()
        mock_ser.is_open = True
        mock_serial_cls.return_value = mock_ser

        m = MakcuMouse()
        m.connect("COM3")
        mock_ser.write.reset_mock()

        m.move(0, 0)
        mock_ser.write.assert_called_once_with(b"km.move(0,0)\r\n")


class TestMakcuMouseClick:
    """測試 MAKCU 滑鼠點擊指令格式"""

    @patch("win_utils.makcu_mouse.serial.Serial")
    def test_click_action1(self, mock_serial_cls):
        """action=1: 點擊（按下後放開）"""
        from win_utils.makcu_mouse import MakcuMouse
        mock_ser = MagicMock()
        mock_ser.is_open = True
        mock_serial_cls.return_value = mock_ser

        m = MakcuMouse()
        m.connect("COM3")
        mock_ser.write.reset_mock()

        m.click(1)
        assert mock_ser.write.call_args_list == [
            call(b"km.left(1)\r\n"),
            call(b"km.left(0)\r\n"),
        ]

    @patch("win_utils.makcu_mouse.serial.Serial")
    def test_click_action2_press(self, mock_serial_cls):
        """action=2: 按下"""
        from win_utils.makcu_mouse import MakcuMouse
        mock_ser = MagicMock()
        mock_ser.is_open = True
        mock_serial_cls.return_value = mock_ser

        m = MakcuMouse()
        m.connect("COM3")
        mock_ser.write.reset_mock()

        m.click(2)
        mock_ser.write.assert_called_once_with(b"km.left(1)\r\n")

    @patch("win_utils.makcu_mouse.serial.Serial")
    def test_click_action3_release(self, mock_serial_cls):
        """action=3: 放開"""
        from win_utils.makcu_mouse import MakcuMouse
        mock_ser = MagicMock()
        mock_ser.is_open = True
        mock_serial_cls.return_value = mock_ser

        m = MakcuMouse()
        m.connect("COM3")
        mock_ser.write.reset_mock()

        m.click(3)
        mock_ser.write.assert_called_once_with(b"km.left(0)\r\n")

    def test_click_not_connected(self):
        """未連線時點擊不應報錯"""
        from win_utils.makcu_mouse import MakcuMouse
        m = MakcuMouse()
        m.click(1)  # 不應拋出異常

    @patch("win_utils.makcu_mouse.serial.Serial")
    def test_click_invalid_action(self, mock_serial_cls):
        """無效 action 不應發送任何命令"""
        from win_utils.makcu_mouse import MakcuMouse
        mock_ser = MagicMock()
        mock_ser.is_open = True
        mock_serial_cls.return_value = mock_ser

        m = MakcuMouse()
        m.connect("COM3")
        mock_ser.write.reset_mock()

        m.click(99)
        mock_ser.write.assert_not_called()


# ============================================================
# 2. 模組級便利函式測試
# ============================================================

class TestModuleFunctions:
    """測試模組級便利函式"""

    @patch("win_utils.makcu_mouse.makcu_mouse")
    def test_send_mouse_move_makcu(self, mock_singleton):
        from win_utils.makcu_mouse import send_mouse_move_makcu
        send_mouse_move_makcu(10, -5)
        mock_singleton.move.assert_called_once_with(10, -5)

    @patch("win_utils.makcu_mouse.makcu_mouse")
    def test_send_mouse_click_makcu(self, mock_singleton):
        from win_utils.makcu_mouse import send_mouse_click_makcu
        result = send_mouse_click_makcu(1)
        mock_singleton.click.assert_called_once_with(1)
        assert result is True

    @patch("win_utils.makcu_mouse.makcu_mouse")
    def test_connect_makcu(self, mock_singleton):
        mock_singleton.connect.return_value = True
        from win_utils.makcu_mouse import connect_makcu
        result = connect_makcu("COM3", 115200)
        mock_singleton.connect.assert_called_once_with("COM3", 115200)
        assert result is True

    @patch("win_utils.makcu_mouse.makcu_mouse")
    def test_disconnect_makcu(self, mock_singleton):
        from win_utils.makcu_mouse import disconnect_makcu
        disconnect_makcu()
        mock_singleton.disconnect.assert_called_once()

    @patch("win_utils.makcu_mouse.makcu_mouse")
    def test_is_makcu_connected(self, mock_singleton):
        mock_singleton.is_connected.return_value = True
        from win_utils.makcu_mouse import is_makcu_connected
        assert is_makcu_connected() is True


# ============================================================
# 3. 調度層測試 (send_mouse_move / send_mouse_click)
# ============================================================

class TestDispatchMakcu:
    """測試 makcu 在 send_mouse_move 和 send_mouse_click 調度中的整合"""

    @patch("win_utils.makcu_mouse.makcu_mouse")
    def test_send_mouse_move_dispatch_makcu(self, mock_singleton):
        from win_utils import send_mouse_move
        send_mouse_move(10, 20, method="makcu")
        mock_singleton.move.assert_called_once_with(10, 20)

    @patch("win_utils.makcu_mouse.send_mouse_click_makcu", return_value=True)
    def test_send_mouse_click_dispatch_makcu(self, mock_click):
        from win_utils.mouse_click import send_mouse_click
        result = send_mouse_click(method="makcu")
        mock_click.assert_called_once()
        assert result is True


# ============================================================
# 4. Config 驗證測試
# ============================================================

class TestConfigValidation:
    """測試 config 中 makcu 作為有效的滑鼠方式"""

    def test_makcu_is_valid_click_method(self):
        from core.config import Config, _validate_mouse_method
        config = Config()
        config.mouse_click_method = "makcu"
        _validate_mouse_method(config)
        assert config.mouse_click_method == "makcu"

    def test_invalid_method_falls_back(self):
        from core.config import Config, _validate_mouse_method
        config = Config()
        config.mouse_click_method = "nonexistent"
        _validate_mouse_method(config)
        assert config.mouse_click_method == "mouse_event"

    def test_makcu_com_port_in_config(self):
        from core.config import Config
        config = Config()
        assert hasattr(config, "makcu_com_port")
        assert config.makcu_com_port == ""

    def test_makcu_com_port_in_dict(self):
        from core.config import Config
        config = Config()
        config.makcu_com_port = "COM5"
        d = config.to_dict()
        assert "makcu_com_port" in d
        assert d["makcu_com_port"] == "COM5"

    def test_makcu_com_port_from_dict(self):
        from core.config import Config
        config = Config()
        config.from_dict({"makcu_com_port": "COM7"})
        assert config.makcu_com_port == "COM7"


# ============================================================
# 5. Serial 錯誤處理測試
# ============================================================

class TestMakcuSerialErrors:
    """測試串列通訊錯誤處理"""

    @patch("win_utils.makcu_mouse.serial.Serial")
    def test_move_serial_exception_disconnects(self, mock_serial_cls):
        """串列異常應將連線狀態設為 False"""
        import serial as real_serial
        from win_utils.makcu_mouse import MakcuMouse

        mock_ser = MagicMock()
        mock_ser.is_open = True
        mock_ser.write.side_effect = real_serial.SerialException("port gone")
        mock_serial_cls.return_value = mock_ser

        m = MakcuMouse()
        m.connect("COM3")
        m.move(10, 20)

        assert m._connected is False

    @patch("win_utils.makcu_mouse.serial.Serial")
    def test_click_serial_exception_disconnects(self, mock_serial_cls):
        """點擊時串列異常應將連線狀態設為 False"""
        import serial as real_serial
        from win_utils.makcu_mouse import MakcuMouse

        mock_ser = MagicMock()
        mock_ser.is_open = True
        mock_ser.write.side_effect = [None, real_serial.SerialException("port gone")]
        mock_serial_cls.return_value = mock_ser

        m = MakcuMouse()
        m.connect("COM3")
        m.click(1)

        assert m._connected is False
