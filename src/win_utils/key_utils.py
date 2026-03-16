# key_utils.py - Key Detection Module
"""Key state detection (supports keyboard, mouse, gamepad)"""

import win32api

from .gamepad_input import is_gamepad_vk, is_gamepad_button_pressed


def is_key_pressed(key_code):
    """Check if the specified key is pressed (supports keyboard/mouse/gamepad)"""
    if is_gamepad_vk(key_code):
        return is_gamepad_button_pressed(key_code)
    return (win32api.GetAsyncKeyState(key_code) & 0x8000) != 0

