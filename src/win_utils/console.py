# console.py - Terminal Window Control Module
"""Windows Terminal Window Control"""

import ctypes


def get_console_window():
    """Get the handle of the current console window"""
    try:
        kernel32 = ctypes.windll.kernel32
        return kernel32.GetConsoleWindow()
    except Exception as e:
        print(f"[Terminal Control] Failed to get console window: {e}")
        return None


def show_console():
    """Show the terminal window"""
    try:
        hwnd = get_console_window()
        if hwnd:
            user32 = ctypes.windll.user32
            SW_SHOW = 5
            user32.ShowWindow(hwnd, SW_SHOW)
            print("[Terminal Control] Terminal window shown")
            return True
        else:
            print("[Terminal Control] Could not get terminal window handle")
            return False
    except Exception as e:
        print(f"[Terminal Control] Failed to show terminal window: {e}")
        return False


def hide_console():
    """Hide the terminal window"""
    try:
        hwnd = get_console_window()
        if hwnd:
            user32 = ctypes.windll.user32
            SW_HIDE = 0
            user32.ShowWindow(hwnd, SW_HIDE)
            return True
        else:
            return False
    except Exception as e:
        print(f"[Terminal Control] Failed to hide terminal window: {e}")
        return False


def is_console_visible():
    """Check if the terminal window is visible"""
    try:
        hwnd = get_console_window()
        if hwnd:
            user32 = ctypes.windll.user32
            return user32.IsWindowVisible(hwnd)
        return False
    except Exception:
        return False

