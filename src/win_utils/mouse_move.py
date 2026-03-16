# mouse_move.py - Mouse Movement Base Module
"""Basic structure and functions for mouse movement"""

import ctypes
import win32api
import win32con


# ===== Mouse Input Structure =====

class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.c_long),
        ("dy", ctypes.c_long),
        ("mouseData", ctypes.c_ulong),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong))
    ]


class INPUT(ctypes.Structure):
    class _INPUT_UNION(ctypes.Union):
        _fields_ = [("mi", MOUSEINPUT)]
    _anonymous_ = ("u",)
    _fields_ = [("type", ctypes.c_ulong), ("u", _INPUT_UNION)]


INPUT_MOUSE = 0
MOUSEEVENTF_MOVE = 0x0001


# ===== Mouse Movement Methods =====

def send_mouse_move_sendinput(dx, dy):
    """SendInput API (original method, easily detected)"""
    extra = ctypes.c_ulong(0)
    ii_ = INPUT._INPUT_UNION()
    ii_.mi = MOUSEINPUT(dx, dy, 0, MOUSEEVENTF_MOVE, 0, ctypes.pointer(extra))
    command = INPUT(INPUT_MOUSE, ii_)
    ctypes.windll.user32.SendInput(1, ctypes.byref(command), ctypes.sizeof(command))


def send_mouse_move_mouse_event(dx, dy):
    """mouse_event move (executes directly)"""
    try:
        win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, dx, dy, 0, 0)
    except Exception:
        pass

