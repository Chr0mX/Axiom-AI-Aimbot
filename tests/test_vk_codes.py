# tests/test_vk_codes.py
"""
虛擬按鍵碼模組單元測試

測試範圍：
1. VK_CODE_MAP - 按鍵碼到名稱映射
2. VK_TRANSLATIONS - 多語言翻譯
3. get_vk_name - 按鍵名稱取得
"""

import pytest


class TestVKCodeMap:
    """測試按鍵碼映射"""

    def test_mouse_buttons(self):
        from win_utils.vk_codes import VK_CODE_MAP
        assert VK_CODE_MAP[0x01] == "Mouse Left"
        assert VK_CODE_MAP[0x02] == "Mouse Right"
        assert VK_CODE_MAP[0x04] == "Mouse Middle"
        assert VK_CODE_MAP[0x05] == "Mouse X1"
        assert VK_CODE_MAP[0x06] == "Mouse X2"

    def test_common_keys(self):
        from win_utils.vk_codes import VK_CODE_MAP
        assert VK_CODE_MAP[0x20] == "Space"
        assert VK_CODE_MAP[0x0D] == "Enter"
        assert VK_CODE_MAP[0x1B] == "Esc"
        assert VK_CODE_MAP[0x09] == "Tab"

    def test_letter_keys(self):
        from win_utils.vk_codes import VK_CODE_MAP
        assert VK_CODE_MAP[0x41] == "A"
        assert VK_CODE_MAP[0x5A] == "Z"

    def test_function_keys(self):
        from win_utils.vk_codes import VK_CODE_MAP
        assert VK_CODE_MAP[0x70] == "F1"
        assert VK_CODE_MAP[0x7B] == "F12"

    def test_gamepad_buttons(self):
        from win_utils.vk_codes import VK_CODE_MAP
        assert 0x0301 in VK_CODE_MAP  # A
        assert 0x0310 in VK_CODE_MAP  # D-Right

    def test_modifier_keys(self):
        from win_utils.vk_codes import VK_CODE_MAP
        assert VK_CODE_MAP[0xA0] == "Shift(L)"
        assert VK_CODE_MAP[0xA1] == "Shift(R)"
        assert VK_CODE_MAP[0xA2] == "Ctrl(L)"
        assert VK_CODE_MAP[0xA3] == "Ctrl(R)"


class TestVKTranslations:
    """測試多語言翻譯"""

    def test_zh_tw_translations(self):
        from win_utils.vk_codes import VK_TRANSLATIONS
        assert "zh_tw" in VK_TRANSLATIONS
        zh = VK_TRANSLATIONS["zh_tw"]
        assert zh["Mouse Left"] == "滑鼠左鍵"
        assert zh["Mouse Right"] == "滑鼠右鍵"
        assert zh["Space"] == "Space"

    def test_en_empty(self):
        from win_utils.vk_codes import VK_TRANSLATIONS
        assert "en" in VK_TRANSLATIONS
        assert VK_TRANSLATIONS["en"] == {}  # 英文直接顯示原名


class TestGetVkName:
    """測試按鍵名稱取得"""

    def test_known_key(self):
        from win_utils.vk_codes import get_vk_name
        # get_vk_name 行為取決於 language_manager 狀態
        # 但基本上已知 key 不應回傳 hex
        name = get_vk_name(0x20)
        assert "Space" in name or "空" in name  # 可能是中文翻譯

    def test_unknown_key_returns_hex(self):
        from win_utils.vk_codes import get_vk_name
        name = get_vk_name(0xFF)  # 通常不在 map 中
        assert "0x" in name or "FF" in name
