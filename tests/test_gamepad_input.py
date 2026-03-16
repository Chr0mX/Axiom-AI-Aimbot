# tests/test_gamepad_input.py
"""
手柄輸入模組單元測試

測試範圍：
1. is_gamepad_vk - 虛擬鍵碼範圍判斷
2. GP_VK 常數完整性
3. GP_VK_TRANSLATION_MAP 完整性
4. _BUTTON_FLAG_TO_GP_VK / _GP_VK_TO_BUTTON_FLAG 映射
"""

import pytest


class TestIsGamepadVk:
    """測試手柄虛擬鍵碼判斷"""

    def test_valid_range(self):
        from win_utils.gamepad_input import is_gamepad_vk, GP_VK_MIN, GP_VK_MAX
        for vk in range(GP_VK_MIN, GP_VK_MAX + 1):
            assert is_gamepad_vk(vk) is True

    def test_below_range(self):
        from win_utils.gamepad_input import is_gamepad_vk, GP_VK_MIN
        assert is_gamepad_vk(GP_VK_MIN - 1) is False
        assert is_gamepad_vk(0) is False
        assert is_gamepad_vk(0x01) is False  # Mouse Left

    def test_above_range(self):
        from win_utils.gamepad_input import is_gamepad_vk, GP_VK_MAX
        assert is_gamepad_vk(GP_VK_MAX + 1) is False
        assert is_gamepad_vk(0xFFFF) is False


class TestGamepadConstants:
    """測試手柄常數完整性"""

    def test_all_vk_constants_in_range(self):
        from win_utils.gamepad_input import (
            GP_VK_A, GP_VK_B, GP_VK_X, GP_VK_Y,
            GP_VK_LB, GP_VK_RB, GP_VK_LT, GP_VK_RT,
            GP_VK_BACK, GP_VK_START,
            GP_VK_LSTICK, GP_VK_RSTICK,
            GP_VK_DPAD_UP, GP_VK_DPAD_DOWN,
            GP_VK_DPAD_LEFT, GP_VK_DPAD_RIGHT,
            GP_VK_MIN, GP_VK_MAX, is_gamepad_vk,
        )
        all_vks = [
            GP_VK_A, GP_VK_B, GP_VK_X, GP_VK_Y,
            GP_VK_LB, GP_VK_RB, GP_VK_LT, GP_VK_RT,
            GP_VK_BACK, GP_VK_START,
            GP_VK_LSTICK, GP_VK_RSTICK,
            GP_VK_DPAD_UP, GP_VK_DPAD_DOWN,
            GP_VK_DPAD_LEFT, GP_VK_DPAD_RIGHT,
        ]
        assert len(all_vks) == 16
        for vk in all_vks:
            assert is_gamepad_vk(vk) is True

    def test_vk_values_unique(self):
        from win_utils.gamepad_input import (
            GP_VK_A, GP_VK_B, GP_VK_X, GP_VK_Y,
            GP_VK_LB, GP_VK_RB, GP_VK_LT, GP_VK_RT,
            GP_VK_BACK, GP_VK_START,
            GP_VK_LSTICK, GP_VK_RSTICK,
            GP_VK_DPAD_UP, GP_VK_DPAD_DOWN,
            GP_VK_DPAD_LEFT, GP_VK_DPAD_RIGHT,
        )
        all_vks = [
            GP_VK_A, GP_VK_B, GP_VK_X, GP_VK_Y,
            GP_VK_LB, GP_VK_RB, GP_VK_LT, GP_VK_RT,
            GP_VK_BACK, GP_VK_START,
            GP_VK_LSTICK, GP_VK_RSTICK,
            GP_VK_DPAD_UP, GP_VK_DPAD_DOWN,
            GP_VK_DPAD_LEFT, GP_VK_DPAD_RIGHT,
        ]
        assert len(set(all_vks)) == 16


class TestTranslationMap:
    """測試翻譯映射"""

    def test_all_buttons_have_translation_key(self):
        from win_utils.gamepad_input import (
            GP_VK_TRANSLATION_MAP,
            GP_VK_A, GP_VK_B, GP_VK_X, GP_VK_Y,
            GP_VK_LB, GP_VK_RB, GP_VK_LT, GP_VK_RT,
            GP_VK_BACK, GP_VK_START,
            GP_VK_LSTICK, GP_VK_RSTICK,
            GP_VK_DPAD_UP, GP_VK_DPAD_DOWN,
            GP_VK_DPAD_LEFT, GP_VK_DPAD_RIGHT,
        )
        for vk in [GP_VK_A, GP_VK_B, GP_VK_X, GP_VK_Y,
                    GP_VK_LB, GP_VK_RB, GP_VK_LT, GP_VK_RT,
                    GP_VK_BACK, GP_VK_START,
                    GP_VK_LSTICK, GP_VK_RSTICK,
                    GP_VK_DPAD_UP, GP_VK_DPAD_DOWN,
                    GP_VK_DPAD_LEFT, GP_VK_DPAD_RIGHT]:
            assert vk in GP_VK_TRANSLATION_MAP

    def test_translation_keys_prefix(self):
        from win_utils.gamepad_input import GP_VK_TRANSLATION_MAP
        for key, value in GP_VK_TRANSLATION_MAP.items():
            assert value.startswith("key_gp_")


class TestButtonFlagMapping:
    """測試 XInput 按鈕旗標映射"""

    def test_bidirectional_mapping(self):
        from win_utils.gamepad_input import _BUTTON_FLAG_TO_GP_VK, _GP_VK_TO_BUTTON_FLAG
        # 正向映射的每個值都應在反向映射中
        for flag, gp_vk in _BUTTON_FLAG_TO_GP_VK.items():
            assert gp_vk in _GP_VK_TO_BUTTON_FLAG
            assert _GP_VK_TO_BUTTON_FLAG[gp_vk] == flag

    def test_has_14_digital_buttons(self):
        """數位按鈕映射不含扳機（扳機用類比處理）"""
        from win_utils.gamepad_input import _BUTTON_FLAG_TO_GP_VK
        assert len(_BUTTON_FLAG_TO_GP_VK) == 14  # A,B,X,Y,LB,RB,Back,Start,LStick,RStick,DU,DD,DL,DR
