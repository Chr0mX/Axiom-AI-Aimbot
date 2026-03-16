# admin.py - Administrator Privileges Module
"""Windows Administrator Privilege Management"""

from __future__ import annotations

import ctypes
import os
import sys


def is_admin() -> bool:
    """Check if the current process is running with administrator privileges"""
    try:
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except (AttributeError, OSError):
        return False


def request_admin_privileges():
    """Request administrator privileges and restart the program"""
    if is_admin():
        return True
    
    try:
        print("[Privilege Management] Restarting program with administrator privileges...")
        
        # Get the full path of the current script
        script_path = os.path.abspath(sys.argv[0])
        
        # 使用 ShellExecute 以管理員權限啟動
        result = ctypes.windll.shell32.ShellExecuteW(
            None, 
            "runas", 
            sys.executable, 
            f'"{script_path}"', 
            None, 
            1  # SW_SHOW
        )
        
        # If successfully started, exit the current program
        if result > 32:  # ShellExecute success return value > 32
            print("[Privilege Management] Administrator privileged program started, terminating current process")
            sys.exit(0)
        else:
            print(f"[Privilege Management] Failed to start administrator privileged program, error code: {result}")
            print("[Privilege Management] Continuing with normal privileges (some features may be limited)")
            return False
            
    except Exception as e:
        print(f"[Privilege Management] Error occurred while requesting administrator privileges: {e}")
        print("[Privilege Management] Continuing with normal privileges (some features may be limited)")
        return False


def check_and_request_admin():
    """Check and request administrator privileges if needed"""
    # Check if there is a command line argument to skip administrator privileges
    if "--no-admin" in sys.argv:
        return False
    
    if is_admin():
        print("[Privilege Management] Program is already running with administrator privileges")
        return True
    else:
        print("[Privilege Management] Program is not running with administrator privileges")
        return request_admin_privileges()

