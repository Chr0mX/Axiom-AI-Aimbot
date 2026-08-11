# aim_input.py - Auto Un-ADS synthetic aim-key release/press
"""Synthetically press/release the user's *configured* aim key/button
(config.AimKeys / config.makcu_aim_button — never a separate keybind) so a
consumer (auto-unads) can make the game exit/re-enter ADS on the user's
behalf.

KNOWN LIMITATION (generic backends only — sendinput/mouse_event): once this
injects an up-event, win32api.GetAsyncKeyState() reflects it exactly like a
real hardware release (that's what makes injected input work at all), so
is_key_pressed() can no longer distinguish "still physically held" from "we
released it programmatically". If the user genuinely releases the real
button/key while a release window is active, the later synthetic re-press
can leave GetAsyncKeyState reading "held" until one more real press+release
cycle happens. MAKCU is unaffected: rmb_held/lmb_held (makcu_mouse.py) read
a raw-physical button telemetry stream (km.buttons(1), mode 1 = "raw
(physical)" per docs/MAKCU_Native_API.md) that our own right()/left()
command writes never touch.
"""

import logging

import win32api
import win32con

from .gamepad_input import is_gamepad_vk

logger = logging.getLogger(__name__)

# Mouse VK -> (down_flag, up_flag, mouseData). mouseData is only meaningful
# for the X-buttons: MOUSEEVENTF_XDOWN/XUP need the literal WinUser.h
# XBUTTON1=1 / XBUTTON2=2 value in mouseData — NOT VK_XBUTTON1/VK_XBUTTON2
# (0x05/0x06), a completely different namespace that this repo's vendored
# win32con.py doesn't even define constants for.
_MOUSE_VK_FLAGS = {
    0x01: (win32con.MOUSEEVENTF_LEFTDOWN,   win32con.MOUSEEVENTF_LEFTUP,   0),  # VK_LBUTTON
    0x02: (win32con.MOUSEEVENTF_RIGHTDOWN,  win32con.MOUSEEVENTF_RIGHTUP,  0),  # VK_RBUTTON
    0x04: (win32con.MOUSEEVENTF_MIDDLEDOWN, win32con.MOUSEEVENTF_MIDDLEUP, 0),  # VK_MBUTTON
    0x05: (win32con.MOUSEEVENTF_XDOWN,      win32con.MOUSEEVENTF_XUP,      1),  # VK_XBUTTON1
    0x06: (win32con.MOUSEEVENTF_XDOWN,      win32con.MOUSEEVENTF_XUP,      2),  # VK_XBUTTON2
}

# Mouse VK -> MAKCU button name (makcu_mouse.MakcuMouse.press_button()).
_MAKCU_BUTTON_NAMES = {
    0x01: 'left',
    0x02: 'right',
    0x04: 'middle',
    0x05: 'side1',
    0x06: 'side2',
}

# mouse_move_method values that can actually send a synthetic mouse-button
# event today ('makcu' is handled separately above this check). ddxoft/
# arduino/xbox have no right/middle/side primitive in this repo
# (ddxoft_mouse.py / arduino_mouse.py are left-click-only; xbox_controller.py's
# "click" maps to the RT trigger, an unrelated mapping) — log-once and no-op
# rather than silently doing nothing forever.
_GENERIC_MOUSE_METHODS = ('sendinput', 'mouse_event')

_warned: set = set()


def _warn_once(key, msg, *args) -> None:
    if key in _warned:
        return
    _warned.add(key)
    logger.warning(msg, *args)


def set_aim_key_state(vk_code: int, down: bool, mouse_move_method: str) -> None:
    """Synthetically press (down=True) or release (down=False) an AimKeys VK.

    Dispatches by VK class (mouse button / keyboard / gamepad), mirroring
    is_key_pressed()'s own VK-class dispatch (key_utils.py).
    """
    if is_gamepad_vk(vk_code):
        _warn_once(
            ('gamepad', vk_code),
            "[AutoUnADS] VK 0x%X is a gamepad button; cannot spoof a physical "
            "controller trigger release. Skipping.",
            vk_code,
        )
        return

    if vk_code in _MOUSE_VK_FLAGS:
        if mouse_move_method == 'makcu':
            from .makcu_mouse import makcu_mouse, is_makcu_connected
            if not is_makcu_connected():
                return
            button = _MAKCU_BUTTON_NAMES[vk_code]
            makcu_mouse.press_button(button, 2 if down else 3)
            return

        if mouse_move_method in _GENERIC_MOUSE_METHODS:
            down_flag, up_flag, data = _MOUSE_VK_FLAGS[vk_code]
            try:
                win32api.mouse_event(down_flag if down else up_flag, 0, 0, data, 0)
            except Exception:
                pass
            return

        _warn_once(
            ('nomethod', mouse_move_method),
            "[AutoUnADS] mouse_move_method=%s has no button release/press "
            "primitive implemented; skipping.",
            mouse_move_method,
        )
        return

    # Plain keyboard VK — win32api.keybd_event is sufficient here, no new
    # ctypes SendInput struct needed (mirrors mouse_click.py's preference
    # for the simple wrapper API over raw SendInput structs for movement).
    try:
        win32api.keybd_event(vk_code, 0, 0 if down else win32con.KEYEVENTF_KEYUP, 0)
    except Exception:
        pass
