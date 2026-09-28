# key_utils.py - Key Detection Module
"""Key state detection (supports keyboard, mouse, gamepad)"""

import win32api

from .gamepad_input import is_gamepad_vk, is_gamepad_button_pressed

# When on, hotkeys also count as held if the MAKCU reports them (button
# stream for mouse VKs, keyboard stream for keyboard VKs) — on a 2-PC setup
# this PC's own GetAsyncKeyState never sees the gaming PC's input.
_device_hotkeys = False


def set_device_hotkeys(enabled):
    """Route hotkeys through the MAKCU too. Also (un)subscribes the MAKCU
    keyboard stream. Idempotent — safe to call every loop tick."""
    global _device_hotkeys
    enabled = bool(enabled)
    from .makcu_mouse import makcu_mouse
    makcu_mouse.set_keyboard_stream_enabled(enabled)
    _device_hotkeys = enabled


def is_key_pressed(key_code):
    """Check if the specified key is pressed (supports keyboard/mouse/gamepad)"""
    if is_gamepad_vk(key_code):
        return is_gamepad_button_pressed(key_code)
    if (win32api.GetAsyncKeyState(key_code) & 0x8000) != 0:
        return True
    if _device_hotkeys:
        from .makcu_mouse import makcu_mouse
        return bool(makcu_mouse.is_vk_down(key_code))
    return False
