"""Device-hotkey OR logic. win32api is stubbed; the MAKCU side is the real driver."""

import sys
from unittest.mock import MagicMock

sys.modules.setdefault("win32api", MagicMock())
sys.modules.setdefault("win32con", MagicMock())

import win32api  # noqa: E402

from win_utils.key_utils import is_key_pressed, set_device_hotkeys  # noqa: E402
from win_utils.makcu_mouse import MakcuMouse  # noqa: E402


def _install(monkeypatch):
    mouse = MakcuMouse()
    mouse._serial = MagicMock()
    mouse._serial.is_open = True
    mouse._connected = True
    mouse._mak_api = True
    mouse._btn_mask = 0x01
    mouse._keyboard_stream_wanted = True
    mouse._keys_down.add(0x04)  # HID 'A'
    # win_utils/__init__.py binds the name makcu_mouse to the singleton, which
    # shadows the submodule on the package. The functions import the submodule.
    module = sys.modules["win_utils.makcu_mouse"]
    monkeypatch.setattr(module, "makcu_mouse", mouse)
    return mouse


def test_windows_down_short_circuits(monkeypatch):
    _install(monkeypatch)
    win32api.GetAsyncKeyState.return_value = 0x8000
    set_device_hotkeys(False)
    assert is_key_pressed(0x41) is True


def test_device_mouse_and_keyboard_count_when_enabled(monkeypatch):
    _install(monkeypatch)
    win32api.GetAsyncKeyState.return_value = 0
    set_device_hotkeys(True)
    assert is_key_pressed(0x01) is True  # left button from the stream
    assert is_key_pressed(0x41) is True  # keyboard HID
    assert is_key_pressed(0x42) is False
    set_device_hotkeys(False)
    assert is_key_pressed(0x01) is False


def test_unknown_vk_does_not_count_as_pressed(monkeypatch):
    _install(monkeypatch)
    win32api.GetAsyncKeyState.return_value = 0
    set_device_hotkeys(True)
    # A VK the device cannot answer for must not read as held.
    assert is_key_pressed(0xFF) is False
