# tests/test_win_utils_aim_input.py
"""Tests for src/win_utils/aim_input.py's click_aim_key() — the synthetic
click Auto Un-ADS fires on its dedicated key/button.

win_utils/__init__.py transitively imports win32api (mouse_move.py) at
module level, which fails at *collection* on a non-Windows box (see
CLAUDE.md). Every test here stubs win32api/win32con/pywintypes/win32gui/
win32file/win32com(.client) in sys.modules before importing anything from
win_utils, matching the sys.modules-stubbing pattern already used in
test_ai_aiming.py for the same reason.
"""

import sys
from unittest.mock import MagicMock, call

import pytest


@pytest.fixture
def win32_stub(monkeypatch):
    """Stub the win32 modules win_utils' import chain touches, and return
    the (win32api, win32con) mocks so tests can assert exact call args."""
    win32api = MagicMock()
    win32con = MagicMock()
    for name, mod in (
        ('win32api', win32api),
        ('win32con', win32con),
        ('win32gui', MagicMock()),
        ('win32com', MagicMock()),
        ('win32com.client', MagicMock()),
        ('pywintypes', MagicMock()),
        ('win32file', MagicMock()),
    ):
        monkeypatch.setitem(sys.modules, name, mod)
    # aim_input.py (and its dependency gamepad_input.py) may already be
    # cached from an earlier test's import under different mocks — drop
    # them so this test's win32api/win32con mocks are the ones actually
    # bound into _MOUSE_VK_FLAGS (built at module-import time).
    for mod_name in list(sys.modules):
        if mod_name == 'win_utils' or mod_name.startswith('win_utils.'):
            monkeypatch.delitem(sys.modules, mod_name, raising=False)
    return win32api, win32con


def test_mouse_button_click_sends_down_then_up(win32_stub):
    win32api, win32con = win32_stub
    from win_utils.aim_input import click_aim_key

    click_aim_key(0x02, 'mouse_event')  # VK_RBUTTON

    assert win32api.mouse_event.call_args_list == [
        call(win32con.MOUSEEVENTF_RIGHTDOWN, 0, 0, 0, 0),
        call(win32con.MOUSEEVENTF_RIGHTUP, 0, 0, 0, 0),
    ]


def test_left_button_click(win32_stub):
    win32api, win32con = win32_stub
    from win_utils.aim_input import click_aim_key

    click_aim_key(0x01, 'sendinput')  # VK_LBUTTON

    assert win32api.mouse_event.call_args_list == [
        call(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0),
        call(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0),
    ]


def test_x_button_click_uses_literal_mousedata_not_vk_constant(win32_stub):
    """XBUTTON1/XBUTTON2 mouseData values are 1/2 per WinUser.h — NOT the
    same as VK_XBUTTON1/VK_XBUTTON2 (0x05/0x06). This locks in that the
    literal 1/2 is what actually gets sent, not the VK code itself."""
    win32api, win32con = win32_stub
    from win_utils.aim_input import click_aim_key

    click_aim_key(0x05, 'mouse_event')  # VK_XBUTTON1
    click_aim_key(0x06, 'mouse_event')  # VK_XBUTTON2

    assert win32api.mouse_event.call_args_list == [
        call(win32con.MOUSEEVENTF_XDOWN, 0, 0, 1, 0),
        call(win32con.MOUSEEVENTF_XUP, 0, 0, 1, 0),
        call(win32con.MOUSEEVENTF_XDOWN, 0, 0, 2, 0),
        call(win32con.MOUSEEVENTF_XUP, 0, 0, 2, 0),
    ]


def test_mouse_button_skipped_for_unsupported_mouse_move_method(win32_stub):
    """ddxoft/arduino/xbox have no click primitive here — must no-op, not
    raise, and must not touch win32api at all."""
    win32api, win32con = win32_stub
    from win_utils.aim_input import click_aim_key

    click_aim_key(0x02, 'ddxoft')

    win32api.mouse_event.assert_not_called()


def test_makcu_mouse_move_method_not_handled_by_this_module(win32_stub):
    """MAKCU routing goes through makcu_mouse.press_button() directly
    (called from ai_loop_utils.apply_unads_transition), not through this
    generic module — click_aim_key() must not touch win32api for it."""
    win32api, win32con = win32_stub
    from win_utils.aim_input import click_aim_key

    click_aim_key(0x02, 'makcu')

    win32api.mouse_event.assert_not_called()


def test_keyboard_key_click_sends_down_then_up(win32_stub):
    """Keyboard VKs work via keybd_event regardless of mouse_move_method —
    keyboard injection is independent of which mouse backend moves the
    cursor."""
    win32api, win32con = win32_stub
    from win_utils.aim_input import click_aim_key

    click_aim_key(0x56, 'ddxoft')  # 'V' key, arbitrary mouse_move_method

    assert win32api.keybd_event.call_args_list == [
        call(0x56, 0, 0, 0),
        call(0x56, 0, win32con.KEYEVENTF_KEYUP, 0),
    ]
    win32api.mouse_event.assert_not_called()


def test_unbound_key_is_a_silent_noop(win32_stub):
    win32api, win32con = win32_stub
    from win_utils.aim_input import click_aim_key

    click_aim_key(0, 'mouse_event')

    win32api.mouse_event.assert_not_called()
    win32api.keybd_event.assert_not_called()


def test_gamepad_vk_is_a_silent_noop_no_win32_call(win32_stub, monkeypatch):
    win32api, win32con = win32_stub
    import win_utils.aim_input as aim_input_mod

    monkeypatch.setattr(aim_input_mod, 'is_gamepad_vk', lambda vk: True)
    aim_input_mod.click_aim_key(0x0301, 'sendinput')

    win32api.mouse_event.assert_not_called()
    win32api.keybd_event.assert_not_called()
