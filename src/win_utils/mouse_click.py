# mouse_click.py - Mouse Click Module
"""Mouse click related functions"""

import time
import logging
import win32api
import win32con

from .ddxoft_mouse import ddxoft_mouse


_hardware_not_impl_warned = False
logger = logging.getLogger(__name__)


# ===== Mouse Click Functions =====

def send_mouse_click_sendinput():
    """SendInput left click"""
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)


def send_mouse_click_hardware():
    """Hardware-level left click
    
    TODO: Implement real hardware-level mouse click
    Currently using SendInput method as a fallback, 
    integrate ddxoft or other driver-level solutions in the future.
    """
    global _hardware_not_impl_warned
    if not _hardware_not_impl_warned:
        logger.warning("hardware mode not implemented, falling back to sendinput")
        _hardware_not_impl_warned = True
    # Temporarily using the same implementation as sendinput
    send_mouse_click_sendinput()


def send_mouse_click_mouse_event():
    """mouse_event left click"""
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
    - "sendinput": SendInput (original method, easily detected)
    - "hardware": Hardware-level (more stealthy)
    - "mouse_event": mouse_event (very stealthy)
    - "ddxoft": ddxoft (most stealthy, requires ddxoft.dll)
    - "xbox": Xbox 360 Virtual Gamepad (RT trigger)
    """
    try:
        if method == "sendinput":
            send_mouse_click_sendinput()
        elif method == "hardware":
            send_mouse_click_hardware()
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


def test_mouse_click_methods():
    """Test all mouse click methods"""
    print("[測試] 開始測試所有滑鼠點擊方式...")
    
    methods = ["mouse_event", "sendinput", "hardware", "ddxoft"]
    
    for method in methods:
        print(f"[測試] 測試 {method} 點擊方式...")
        try:
            success = send_mouse_click(method)
            if success:
                print(f"[測試] {method} 點擊成功")
            else:
                print(f"[測試] ✗ {method} 點擊失敗")
        except Exception as e:
            print(f"[測試] ✗ {method} 點擊異常: {e}")
        
        time.sleep(0.5)  # 延遲0.5秒避免連點
    
    print("[測試] 滑鼠點擊測試完成")

