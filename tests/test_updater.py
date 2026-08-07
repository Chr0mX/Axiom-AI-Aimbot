# tests/test_updater.py
"""
更新檢查模組單元測試

測試範圍：
1. parse_version - 版本號解析
2. open_update_url - 瀏覽器開啟
"""

from unittest.mock import patch

import pytest


# ============================================================
# 1. parse_version 測試
# ============================================================

class TestParseVersion:
    """測試版本號解析"""

    def test_simple_version(self):
        from core.updater import parse_version
        assert parse_version("1.0.0") == (1, 0, 0)

    def test_with_v_prefix(self):
        from core.updater import parse_version
        assert parse_version("v1.2.3") == (1, 2, 3)

    def test_with_V_prefix(self):
        from core.updater import parse_version
        assert parse_version("V2.0.1") == (2, 0, 1)

    def test_two_parts_padded(self):
        from core.updater import parse_version
        assert parse_version("1.0") == (1, 0, 0)

    def test_single_part_padded(self):
        from core.updater import parse_version
        assert parse_version("6") == (6, 0, 0)

    def test_four_parts(self):
        from core.updater import parse_version
        result = parse_version("1.2.3.4")
        assert result[:3] == (1, 2, 3)

    def test_whitespace_stripped(self):
        from core.updater import parse_version
        assert parse_version("  v1.0.0  ") == (1, 0, 0)

    def test_invalid_part_defaults_to_zero(self):
        from core.updater import parse_version
        result = parse_version("1.abc.3")
        assert result == (1, 0, 3)

    def test_version_comparison(self):
        from core.updater import parse_version
        assert parse_version("v2.0.0") > parse_version("v1.9.9")
        assert parse_version("v1.0.1") > parse_version("v1.0.0")
        assert parse_version("v1.0.0") == parse_version("1.0.0")
        assert parse_version("6.1") == parse_version("v6.1.0")


# ============================================================
# 2. open_update_url 測試
# ============================================================

class TestOpenUpdateUrl:
    """測試開啟更新連結"""

    @patch("webbrowser.open")
    def test_opens_url(self, mock_open):
        from core.updater import open_update_url
        open_update_url("https://example.com/release")
        mock_open.assert_called_once_with("https://example.com/release")
