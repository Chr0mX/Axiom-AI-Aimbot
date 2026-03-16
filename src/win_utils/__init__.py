# win_utils/__init__.py - Windows Toolkit
"""
Windows Toolkit - Providing mouse control, key detection, administrator privileges, terminal control, etc.

Module Structure:
- vk_codes: Virtual key codes and translation
- mouse_move: Basic mouse movement functions

- ddxoft_mouse: DDXoft mouse control
- mouse_click: Mouse click functions
- key_utils: Key detection
- admin: Administrator privilege management
- console: Terminal window control
"""

# Virtual key codes
from .vk_codes import (
    VK_CODE_MAP,
    VK_TRANSLATIONS,
    get_vk_name,
)

# Mouse move - Basic
from .mouse_move import (
    MOUSEINPUT,
    INPUT,
    INPUT_MOUSE,
    MOUSEEVENTF_MOVE,
    send_mouse_move_sendinput,
    send_mouse_move_mouse_event,
)



# Mouse move - ddxoft
from .ddxoft_mouse import (
    DDXoftMouse,
    ddxoft_mouse,
    send_mouse_move_ddxoft,
    ensure_ddxoft_ready,
    test_ddxoft_functions,
    get_ddxoft_statistics,
    print_ddxoft_statistics,
    reset_ddxoft_statistics,
)

# Mouse move - Arduino Leonardo
from .arduino_mouse import (
    ArduinoMouse,
    arduino_mouse,
    send_mouse_move_arduino,
    get_available_com_ports,
    connect_arduino,
    disconnect_arduino,
    is_arduino_connected,
)

# Mouse move - MAKCU KM Host
from .makcu_mouse import (
    MakcuMouse,
    makcu_mouse,
    send_mouse_move_makcu,
    send_mouse_click_makcu,
    connect_makcu,
    disconnect_makcu,
    is_makcu_connected,
)

# Mouse move - Xbox 360 Virtual Gamepad
from .xbox_controller import (
    XboxController,
    xbox_controller,
    send_mouse_move_xbox,
    send_mouse_click_xbox,
    connect_xbox,
    disconnect_xbox,
    is_xbox_connected,
    is_xbox_available,
    set_xbox_sensitivity,
    set_xbox_deadzone,
    get_xbox_statistics,
)

# Mouse click
from .mouse_click import (
    send_mouse_click_sendinput,
    send_mouse_click_hardware,
    send_mouse_click_mouse_event,
    send_mouse_click_ddxoft,
    send_mouse_click,
    test_mouse_click_methods,
)
from .arduino_mouse import send_mouse_click_arduino

# Key detection
from .key_utils import is_key_pressed

# Gamepad button reading
from .gamepad_input import (
    is_gamepad_vk,
    is_gamepad_button_pressed,
    poll_pressed_gamepad_button,
    GP_VK_TRANSLATION_MAP,
    GP_VK_MIN,
    GP_VK_MAX,
)

# Administrator privileges
from .admin import (
    is_admin,
    request_admin_privileges,
    check_and_request_admin,
)

# 終端控制
from .console import (
    get_console_window,
    show_console,
    hide_console,
    is_console_visible,
)


# ===== 主要滑鼠移動函數 =====

def send_mouse_move(dx, dy, method="mouse_event"):
    """
    主要滑鼠移動函數
    method 選項:
    - "sendinput": SendInput (原始方式，容易被檢測)
    - "mouse_event": mouse_event (預設，穩定且安全)
    - "ddxoft": ddxoft (最隱蔽，需要 ddxoft.dll，但可能導致藍屏)
    - "arduino": Arduino Leonardo (USB HID，非常隱蔽)
    - "makcu": MAKCU KM Host (硬體級 USB HID 注入，非常隱蔽)
    - "xbox": Xbox 360 虛擬手把 (透過 ViGEmBus，適用手把遊戲)
    """
    if abs(dx) < 1 and abs(dy) < 1:
        return  # 移動量太小，跳過
    
    if method == "sendinput":
        send_mouse_move_sendinput(dx, dy)
    elif method == "mouse_event":
        send_mouse_move_mouse_event(dx, dy)
    elif method == "ddxoft":
        send_mouse_move_ddxoft(dx, dy)
    elif method == "arduino":
        send_mouse_move_arduino(dx, dy)
    elif method == "makcu":
        send_mouse_move_makcu(dx, dy)
    elif method == "xbox":
        send_mouse_move_xbox(dx, dy)
    else:
        # 默認使用 mouse_event 方式（安全穩定）
        send_mouse_move_mouse_event(dx, dy)


# 公開的 API 列表
__all__ = [
    # 虛擬按鍵碼
    'VK_CODE_MAP',
    'VK_TRANSLATIONS',
    'get_vk_name',
    
    # 滑鼠移動
    'MOUSEINPUT',
    'INPUT',
    'INPUT_MOUSE',
    'MOUSEEVENTF_MOVE',
    'send_mouse_move',
    'send_mouse_move_sendinput',
    'send_mouse_move_mouse_event',
    'send_mouse_move_ddxoft',
    'send_mouse_move_arduino',
    'send_mouse_move_makcu',
    'send_mouse_move_xbox',
    
    # 控制器類
    'DDXoftMouse',
    'XboxController',
    'xbox_controller',
    'ddxoft_mouse',
    
    # ddxoft 公共接口
    'ensure_ddxoft_ready',
    'test_ddxoft_functions',
    'get_ddxoft_statistics',
    'print_ddxoft_statistics',
    'reset_ddxoft_statistics',
    
    # Arduino 控制
    'ArduinoMouse',
    'arduino_mouse',
    'get_available_com_ports',
    'connect_arduino',
    'disconnect_arduino',
    'is_arduino_connected',
    
    # MAKCU KM Host 控制
    'MakcuMouse',
    'makcu_mouse',
    'connect_makcu',
    'disconnect_makcu',
    'is_makcu_connected',
    'send_mouse_click_makcu',
    
    # Xbox 360 虛擬手把
    'connect_xbox',
    'disconnect_xbox',
    'is_xbox_connected',
    'is_xbox_available',
    'set_xbox_sensitivity',
    'set_xbox_deadzone',
    'get_xbox_statistics',
    'send_mouse_click_xbox',
    
    # 滑鼠點擊
    'send_mouse_click',
    'send_mouse_click_sendinput',
    'send_mouse_click_hardware',
    'send_mouse_click_mouse_event',
    'send_mouse_click_ddxoft',
    'send_mouse_click_arduino',
    'send_mouse_click_makcu',
    'test_mouse_click_methods',
    
    # 按鍵檢測
    'is_key_pressed',
    
    # 手柄按鍵
    'is_gamepad_vk',
    'is_gamepad_button_pressed',
    'poll_pressed_gamepad_button',
    'GP_VK_TRANSLATION_MAP',
    'GP_VK_MIN',
    'GP_VK_MAX',
    
    # 管理員權限
    'is_admin',
    'request_admin_privileges',
    'check_and_request_admin',
    
    # 終端控制
    'get_console_window',
    'show_console',
    'hide_console',
    'is_console_visible',
]

