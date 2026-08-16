# mouse_click.py - Mouse Click Module
"""Mouse click related functions"""

import logging
import win32api
import win32con

from .ddxoft_mouse import ddxoft_mouse


logger = logging.getLogger(__name__)


# ===== Mouse Click Functions =====

def send_mouse_click_sendinput():
    """Left click via win32api.mouse_event.

    Despite the name, this is currently byte-identical to
    send_mouse_click_mouse_event() below — unlike mouse_move.py's
    send_mouse_move_sendinput(), which genuinely uses the SendInput struct/
    API, no distinct SendInput-based click has been implemented here. Kept
    as a separate name because "sendinput"/"mouse_event" are both valid
    mouse_click_method config values with their own dispatch entries below.
    """
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)


def send_mouse_click_mouse_event():
    """Left click via win32api.mouse_event."""
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)


def send_mouse_click_ddxoft():
    """ddxoft left click"""
    try:
        if not ddxoft_mouse.ensure_initialized():
            send_mouse_click_mouse_event()
            return True

        if ddxoft_mouse.click_left():
            return True
        else:
            # If ddxoft fails, silently fall back to mouse_event
            send_mouse_click_mouse_event()
            return True
    except Exception:
        send_mouse_click_mouse_event()
        return True


def send_mouse_click(method="ddxoft"):
    """
    Unified mouse click function, supports multiple methods
    method options:
    - "sendinput" / "mouse_event": win32api.mouse_event left click
      (currently identical implementations — see
      send_mouse_click_sendinput()'s docstring)
    - "ddxoft": ddxoft (most stealthy, requires ddxoft.dll)
    - "xbox": Xbox 360 Virtual Gamepad (RT trigger)
    """
    try:
        if method == "sendinput":
            send_mouse_click_sendinput()
        elif method == "mouse_event":
            send_mouse_click_mouse_event()
        elif method == "ddxoft":
            return send_mouse_click_ddxoft()
        elif method == "xbox":
            from .xbox_controller import send_mouse_click_xbox
            return send_mouse_click_xbox()
        elif method == "arduino":
            from .arduino_mouse import send_mouse_click_arduino
            return send_mouse_click_arduino()
        elif method == "makcu":
            from .makcu_mouse import send_mouse_click_makcu
            return send_mouse_click_makcu()
        else:
            return send_mouse_click_ddxoft()  # Default method
        return True
    except Exception:
        # Silently fall back to mouse_event
        try:
            send_mouse_click_mouse_event()
            return True
        except Exception:
            return False

