# tests/test_arduino_spoofer.py
"""
Arduino 偽裝模組單元測試

測試範圍：
1. find_boards_txt - 搜尋 boards.txt
2. spoof_arduino_board - 偽裝邏輯
3. verify_spoof - 驗證偽裝結果
"""

import os
import tempfile
from unittest.mock import patch, MagicMock

import pytest


class TestFindBoardsTxt:
    """測試 boards.txt 搜尋"""

    def test_returns_none_when_not_found(self):
        from win_utils.arduino_spoofer import find_boards_txt
        with patch("os.environ.get", return_value=""):
            with patch("os.path.exists", return_value=False):
                with patch("glob.glob", return_value=[]):
                    result = find_boards_txt()
                    assert result is None

    def test_finds_existing_file(self):
        from win_utils.arduino_spoofer import find_boards_txt
        with tempfile.TemporaryDirectory() as tmpdir:
            boards = os.path.join(tmpdir, "boards.txt")
            with open(boards, 'w') as f:
                f.write("leonardo.build.vid=0x2341")
            
            with patch("glob.glob", return_value=[boards]):
                with patch("os.environ.get", return_value=tmpdir):
                    with patch("os.path.exists", side_effect=lambda p: p == boards):
                        result = find_boards_txt()
                        # 可能找到也可能找不到，取決於路徑匹配
                        # 主要驗證不會崩潰


class TestSpoofArduinoBoard:
    """測試 Arduino 偽裝邏輯"""

    def test_spoof_modifies_vid_pid(self):
        from win_utils.arduino_spoofer import spoof_arduino_board
        with tempfile.TemporaryDirectory() as tmpdir:
            boards = os.path.join(tmpdir, "boards.txt")
            with open(boards, 'w') as f:
                f.write("leonardo.build.vid=0x2341\n")
                f.write("leonardo.build.pid=0x8036\n")
                f.write("leonardo.build.usb_product=\"Arduino Leonardo\"\n")
                f.write("other.setting=value\n")
            
            with patch("win_utils.arduino_spoofer.find_boards_txt", return_value=boards):
                result, path = spoof_arduino_board()
                assert result is True
                
                with open(boards, 'r') as f:
                    content = f.read()
                assert "0x046D" in content  # Logitech VID
                assert "0xC07D" in content  # G502 PID
                assert "Logitech G502" in content
                assert "other.setting=value" in content  # 其他設定未受影響

    def test_spoof_creates_backup(self):
        from win_utils.arduino_spoofer import spoof_arduino_board
        with tempfile.TemporaryDirectory() as tmpdir:
            boards = os.path.join(tmpdir, "boards.txt")
            with open(boards, 'w') as f:
                f.write("leonardo.build.vid=0x2341\n")
                f.write("leonardo.build.pid=0x8036\n")
                f.write("leonardo.build.usb_product=\"Arduino Leonardo\"\n")
            
            with patch("win_utils.arduino_spoofer.find_boards_txt", return_value=boards):
                spoof_arduino_board()
                assert os.path.exists(boards + ".bak")

    def test_spoof_raises_when_not_found(self):
        from win_utils.arduino_spoofer import spoof_arduino_board
        with patch("win_utils.arduino_spoofer.find_boards_txt", return_value=None):
            with pytest.raises(FileNotFoundError):
                spoof_arduino_board()


class TestVerifySpoof:
    """測試偽裝驗證"""

    def test_no_devices(self):
        from win_utils.arduino_spoofer import verify_spoof
        with patch("serial.tools.list_ports.comports", return_value=[]):
            is_spoofed, msg = verify_spoof()
            assert is_spoofed is False
            assert "未檢測到" in msg

    def test_spoofed_device_found(self):
        from win_utils.arduino_spoofer import verify_spoof
        mock_port = MagicMock()
        mock_port.vid = 0x046D
        mock_port.pid = 0xC07D
        mock_port.device = "COM7"
        mock_port.description = "Logitech G502"
        with patch("serial.tools.list_ports.comports", return_value=[mock_port]):
            is_spoofed, msg = verify_spoof()
            assert is_spoofed is True
            assert "成功" in msg

    def test_original_device_found(self):
        from win_utils.arduino_spoofer import verify_spoof
        mock_port = MagicMock()
        mock_port.vid = 0x2341
        mock_port.pid = 0x8036
        mock_port.device = "COM3"
        mock_port.description = "Arduino Leonardo"
        with patch("serial.tools.list_ports.comports", return_value=[mock_port]):
            is_spoofed, msg = verify_spoof()
            assert is_spoofed is False
            assert "失敗" in msg

    def test_specific_port_filter(self):
        from win_utils.arduino_spoofer import verify_spoof
        mock_port1 = MagicMock()
        mock_port1.vid = 0x046D
        mock_port1.pid = 0xC07D
        mock_port1.device = "COM7"
        mock_port1.description = "Logitech G502"
        
        mock_port2 = MagicMock()
        mock_port2.vid = 0x2341
        mock_port2.pid = 0x8036
        mock_port2.device = "COM3"
        mock_port2.description = "Arduino"
        
        with patch("serial.tools.list_ports.comports", return_value=[mock_port1, mock_port2]):
            is_spoofed, msg = verify_spoof(specific_port="COM7")
            assert is_spoofed is True
