# aim_input.py - Auto Un-ADS synthetic key click
"""Synthetically click the user's *dedicated* Auto Un-ADS key/button
(config.auto_unads_key generically, or config.auto_unads_makcu_button when
MAKCU is the active mouse_move_method) so a consumer (Auto Un-ADS) can make
the game exit/re-enter ADS on the user's behalf.

This is a deliberately separate control from config.AimKeys / makcu_aim_button
(the normal aim-activation trigger) — most commonly a distinct in-game
"toggle ADS" bind. A single momentary click (down, brief pause, up), not a
sustained press/release state, so it works correctly whether the user's
in-game ADS bind is:
  - a toggle key (each click flips ADS on/off — this is the mechanism a
    toggle bind needs: a fresh press edge), or
  - a hold key (a click's final state is "up", i.e. released, same as if
    the user briefly tapped and let go — the game exits ADS and stays out
    until the *next* click brings it back).

MAKCU's click is a single device-side command (MakcuMouse.press_button with
action=1) — see makcu_mouse.py. Generic backends synthesize the same
down-then-up shape here via win32api.
"""

import logging
import time

import win32api
import win32con

from .gamepad_input import is_gamepad_vk

logger = logging.getLogger(__name__)

# Gap between the synthetic down and up events, matching MAKCU's own click()
# timing (makcu_mouse.py's action=1 sleeps 0.03s between down/up).
_CLICK_HOLD_S = 0.03

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

# mouse_move_method values that can send a synthetic mouse-button event via
# win32api today. ddxoft/arduino/xbox have no right/middle/side primitive in
# this repo (ddxoft_mouse.py / arduino_mouse.py are left-click-only;
# xbox_controller.py's "click" maps to the RT trigger, an unrelated mapping)
# — log-once and no-op rather than silently doing nothing forever. MAKCU is
# handled separately by the caller (it goes through makcu_mouse.press_button,
# not this module) since it needs its own auto_unads_makcu_button field.
_GENERIC_MOUSE_METHODS = ('sendinput', 'mouse_event')

_warned: set = set()


def _warn_once(key, msg, *args) -> None:
    if key in _warned:
        return
    _warned.add(key)
    logger.warning(msg, *args)


def click_aim_key(vk_code: int, mouse_move_method: str) -> None:
    """Send a single click (down, ~30ms, up) of vk_code.

    Dispatches by VK class (mouse button / keyboard / gamepad), mirroring
    is_key_pressed()'s own VK-class dispatch (key_utils.py). vk_code == 0
    ("unbound") is a silent no-op — callers should already guard on this,
    but staying a no-op here too keeps this function safe to call blind.
    """
    if vk_code == 0:
        return

    if is_gamepad_vk(vk_code):
        _warn_once(
            ('gamepad', vk_code),
            "[AutoUnADS] VK 0x%X is a gamepad button; cannot spoof a physical "
            "controller trigger. Skipping.",
            vk_code,
        )
        return

    if vk_code in _MOUSE_VK_FLAGS:
        if mouse_move_method not in _GENERIC_MOUSE_METHODS:
            _warn_once(
                ('nomethod', mouse_move_method),
                "[AutoUnADS] mouse_move_method=%s has no click primitive for "
                "mouse-button auto_unads_key; skipping.",
                mouse_move_method,
            )
            return
        down_flag, up_flag, data = _MOUSE_VK_FLAGS[vk_code]
        try:
            win32api.mouse_event(down_flag, 0, 0, data, 0)
            time.sleep(_CLICK_HOLD_S)
            win32api.mouse_event(up_flag, 0, 0, data, 0)
        except Exception:
            pass
        return

    # Plain keyboard VK — win32api.keybd_event is sufficient here, no new
    # ctypes SendInput struct needed (mirrors mouse_click.py's preference
    # for the simple wrapper API over raw SendInput structs for movement).
    # Keyboard injection is independent of mouse_move_method.
    try:
        win32api.keybd_event(vk_code, 0, 0, 0)
        time.sleep(_CLICK_HOLD_S)
        win32api.keybd_event(vk_code, 0, win32con.KEYEVENTF_KEYUP, 0)
    except Exception:
        pass
