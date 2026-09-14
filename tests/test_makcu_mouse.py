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
# 1b. 按鍵事件流解析測試 (_stream_reader)
# ============================================================

class _FakeStreamSerial:
    """Minimal serial stand-in that feeds a fixed byte payload once, then
    signals the reader thread to stop so _stream_reader() returns."""

    def __init__(self, payload, stop_event):
        self._data = bytearray(payload)
        self.is_open = True
        self._stop = stop_event

    @property
    def in_waiting(self):
        return len(self._data)

    def read(self, n):
        chunk = bytes(self._data[:n])
        del self._data[:n]
        if not self._data:
            self._stop.set()  # drained — let the reader loop exit
        return chunk


def _run_stream(payload):
    """Drive _stream_reader() over `payload` and return the final _btn_mask."""
    from win_utils.makcu_mouse import MakcuMouse
    m = MakcuMouse()
    m._serial = _FakeStreamSerial(payload, m._stream_stop)
    m._stream_reader()
    return m


_KM_PREFIX = b"km."
_SUFFIX = b"\r\n>>> "


def _frame(mask):
    """A single button-stream frame in the format confirmed against real
    MAKCU hardware (raw hex capture, 2026-08-04): "km." + a single mask
    byte + the same "\\r\\n>>> " suffix the docs describe for one-off ASCII
    command replies. E.g. a Left-button press is literally
    `6b 6d 2e 01 0d 0a 3e 3e 3e 20` on the wire.

    This is the fourth framing model this reader has used — three earlier
    guesses (a "km."-prefixed 5-byte frame with a 2-byte mask, a bare
    2-byte mask with no prefix, and the MAKCU V2 [0x50]-wrapped binary
    protocol) were each tried and reported as not detecting real clicks.
    See _stream_reader()'s docstring for the confirmed capture.
    """
    return _KM_PREFIX + bytes([mask]) + _SUFFIX


class TestMakcuStreamReader:
    """測試按鍵事件流 (buttons stream) 的 km.<mask>\\r\\n>>> 封包解析與按鍵位元遮罩"""

    def test_real_lmb_press_parses(self):
        """km. + 01 + \\r\\n>>>  → 左鍵按下"""
        m = _run_stream(_frame(0x01))
        assert m._btn_mask == 0x01
        assert m.lmb_held is True
        assert m.rmb_held is False

    def test_release_parses(self):
        """press then release → 全部放開"""
        m = _run_stream(_frame(0x01) + _frame(0x00))
        assert m._btn_mask == 0x00
        assert m.lmb_held is False

    def test_rmb_press_parses(self):
        """mask=0x02 → 右鍵按下，不誤報左鍵"""
        m = _run_stream(_frame(0x02))
        assert m._btn_mask == 0x02
        assert m.lmb_held is False
        assert m.rmb_held is True

    def test_side1_press_parses(self):
        """mask=0x08 → 側鍵一按下，不誤報其他按鍵"""
        m = _run_stream(_frame(0x08))
        assert m._btn_mask == 0x08
        assert m.side1_held is True
        assert m.side2_held is False
        assert m.lmb_held is False
        assert m.rmb_held is False

    def test_side2_press_parses(self):
        """mask=0x10 → 側鍵二按下，不誤報其他按鍵"""
        m = _run_stream(_frame(0x10))
        assert m._btn_mask == 0x10
        assert m.side2_held is True
        assert m.side1_held is False

    def test_high_bits_masked_off(self):
        """遮罩位元組的高位元 (bits 5-7) 應被 _BTN_BITS 遮罩"""
        # 0xE1 = 0b111_00001 -> only bits 0-4 (0x01, Left) should survive
        m = _run_stream(_frame(0xE1))
        assert m._btn_mask == 0x01
        assert m.lmb_held is True

    def test_last_frame_wins_in_batch(self):
        """同一次讀取含多幀時，最後一幀為最終狀態"""
        m = _run_stream(_frame(0x01) + _frame(0x03) + _frame(0x00))
        assert m._btn_mask == 0x00

    def test_partial_frame_not_consumed(self):
        """不足一個完整封包的資料不應更新狀態"""
        # "km." + mask byte, but the "\r\n>>> " suffix hasn't arrived yet
        m = _run_stream(_KM_PREFIX + bytes([0x01]))
        assert m._btn_mask == 0x00  # untouched — suffix incomplete

    def test_isolated_single_frame_applies(self):
        """孤立的單一封包應直接套用。"""
        m = _run_stream(_frame(0x01))
        assert m._btn_mask == 0x01
        assert m.lmb_held is True

    def test_consecutive_frames_all_apply_in_order(self):
        """連續多幀應依序套用，以最後一幀為最終狀態。"""
        m = _run_stream(_frame(0x01) + _frame(0x02) + _frame(0x00))
        assert m._btn_mask == 0x00

    def test_mismatched_suffix_discarded_and_resynced(self):
        """後綴不符（失步/損毀）的候選幀應被捨棄，並在下一個真正的 km. 前綴重新同步。"""
        corrupt = _KM_PREFIX + bytes([0x01]) + b"\r\nXXXXX"  # wrong suffix
        m = _run_stream(corrupt + _frame(0x02))
        assert m._btn_mask == 0x02
        assert m.rmb_held is True

    def test_real_captured_sequence_from_hardware(self):
        """回歸測試：對真實硬體擷取的原始位元組序列逐幀套用，最終狀態應為左鍵按下。

        Verbatim raw capture (2026-08-04) of a user testing L, R, side-top,
        side-bottom, middle, middle, L — confirms the parser tracks the
        entire real sequence correctly end to end, not just synthetic frames.
        """
        raw_hex = (
            "6b6d2e010d0a3e3e3e20"
            "6b6d2e000d0a3e3e3e20"
            "6b6d2e020d0a3e3e3e20"
            "6b6d2e000d0a3e3e3e20"
            "6b6d2e100d0a3e3e3e20"
            "6b6d2e000d0a3e3e3e20"
            "6b6d2e080d0a3e3e3e20"
            "6b6d2e000d0a3e3e3e20"
            "6b6d2e040d0a3e3e3e20"
            "6b6d2e000d0a3e3e3e20"
            "6b6d2e040d0a3e3e3e20"
            "6b6d2e000d0a3e3e3e20"
            "6b6d2e010d0a3e3e3e20"
        )
        m = _run_stream(bytes.fromhex(raw_hex))
        assert m._btn_mask == 0x01
        assert m.lmb_held is True

    def test_resyncs_past_garbage_before_prefix(self):
        """"km." 之前的雜訊位元組應被丟棄，並在下一個有效封包重新同步。"""
        garbage = bytes([0x11, 0x22, 0x33])
        m = _run_stream(garbage + _frame(0x01))
        assert m._btn_mask == 0x01
        assert m.lmb_held is True


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
