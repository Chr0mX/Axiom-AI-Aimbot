# tests/test_admin_console.py
"""
管理員權限與終端控制模組單元測試

測試範圍：
1. admin.py - is_admin, check_and_request_admin
2. console.py - get_console_window, show_console, hide_console, is_console_visible
"""

import sys
from unittest.mock import patch, MagicMock

import pytest


# ============================================================
# 1. admin.py 測試
# ============================================================

class TestIsAdmin:
    """測試管理員權限檢查"""

    @patch("ctypes.windll.shell32.IsUserAnAdmin", return_value=1)
    def test_is_admin_true(self, mock_admin):
        from win_utils.admin import is_admin
        assert is_admin() is True

    @patch("ctypes.windll.shell32.IsUserAnAdmin", return_value=0)
    def test_is_admin_false(self, mock_admin):
        from win_utils.admin import is_admin
        assert is_admin() is False

    @patch("ctypes.windll.shell32.IsUserAnAdmin", side_effect=OSError("fail"))
    def test_is_admin_exception_returns_false(self, mock_admin):
        from win_utils.admin import is_admin
        assert is_admin() is False


class TestCheckAndRequestAdmin:
    """測試管理員權限檢查與請求"""

    @patch("win_utils.admin.is_admin", return_value=True)
    def test_already_admin(self, mock_admin):
        from win_utils.admin import check_and_request_admin
        assert check_and_request_admin() is True

    @patch("win_utils.admin.is_admin", return_value=False)
    @patch("win_utils.admin.request_admin_privileges", return_value=False)
    def test_not_admin_request_fails(self, mock_req, mock_admin):
        from win_utils.admin import check_and_request_admin
        # 移除可能的 --no-admin 參數
        original_argv = sys.argv[:]
        sys.argv = [a for a in sys.argv if a != '--no-admin']
        try:
            result = check_and_request_admin()
            assert result is False
        finally:
            sys.argv = original_argv

    def test_no_admin_flag_skips(self):
        from win_utils.admin import check_and_request_admin
        original_argv = sys.argv[:]
        sys.argv = sys.argv + ['--no-admin']
        try:
            result = check_and_request_admin()
            assert result is False
        finally:
            sys.argv = original_argv


# ============================================================
# 2. console.py 測試
# ============================================================

class TestConsoleWindow:
    """測試終端視窗控制"""

    @patch("ctypes.windll.kernel32.GetConsoleWindow", return_value=12345)
    def test_get_console_window(self, mock_get):
        from win_utils.console import get_console_window
        hwnd = get_console_window()
        assert hwnd == 12345

    @patch("ctypes.windll.kernel32.GetConsoleWindow", side_effect=Exception("fail"))
    def test_get_console_window_error(self, mock_get):
        from win_utils.console import get_console_window
        hwnd = get_console_window()
        assert hwnd is None

    @patch("win_utils.console.get_console_window", return_value=12345)
    @patch("ctypes.windll.user32.ShowWindow")
    def test_show_console(self, mock_show, mock_get):
        from win_utils.console import show_console
        result = show_console()
        assert result is True
        mock_show.assert_called_once_with(12345, 5)  # SW_SHOW = 5

    @patch("win_utils.console.get_console_window", return_value=None)
    def test_show_console_no_window(self, mock_get):
        from win_utils.console import show_console
        result = show_console()
        assert result is False

    @patch("win_utils.console.get_console_window", return_value=12345)
    @patch("ctypes.windll.user32.ShowWindow")
    def test_hide_console(self, mock_show, mock_get):
        from win_utils.console import hide_console
        result = hide_console()
        assert result is True
        mock_show.assert_called_once_with(12345, 0)  # SW_HIDE = 0

    @patch("win_utils.console.get_console_window", return_value=None)
    def test_hide_console_no_window(self, mock_get):
        from win_utils.console import hide_console
        result = hide_console()
        assert result is False

    @patch("win_utils.console.get_console_window", return_value=12345)
    @patch("ctypes.windll.user32.IsWindowVisible", return_value=True)
    def test_is_visible(self, mock_visible, mock_get):
        from win_utils.console import is_console_visible
        assert is_console_visible() is True

    @patch("win_utils.console.get_console_window", return_value=None)
    def test_is_visible_no_window(self, mock_get):
        from win_utils.console import is_console_visible
        assert is_console_visible() is False
