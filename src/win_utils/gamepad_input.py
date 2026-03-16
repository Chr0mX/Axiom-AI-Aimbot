# gamepad_input.py - Gamepad Input Reading Module
"""
Use XInput API to read physical gamepad button states
For key bindings and runtime key detection
"""

import ctypes
import ctypes.wintypes
from typing import Optional, Dict

# ===== XInput 結構定義 =====

class XINPUT_GAMEPAD(ctypes.Structure):
    _fields_ = [
        ("wButtons", ctypes.wintypes.WORD),
        ("bLeftTrigger", ctypes.c_ubyte),
        ("bRightTrigger", ctypes.c_ubyte),
        ("sThumbLX", ctypes.c_short),
        ("sThumbLY", ctypes.c_short),
        ("sThumbRX", ctypes.c_short),
        ("sThumbRY", ctypes.c_short),
    ]

class XINPUT_STATE(ctypes.Structure):
    _fields_ = [
        ("dwPacketNumber", ctypes.wintypes.DWORD),
        ("Gamepad", XINPUT_GAMEPAD),
    ]

# ===== XInput 按鈕常數 =====
XINPUT_GAMEPAD_DPAD_UP        = 0x0001
XINPUT_GAMEPAD_DPAD_DOWN      = 0x0002
XINPUT_GAMEPAD_DPAD_LEFT      = 0x0004
XINPUT_GAMEPAD_DPAD_RIGHT     = 0x0008
XINPUT_GAMEPAD_START          = 0x0010
XINPUT_GAMEPAD_BACK           = 0x0020
XINPUT_GAMEPAD_LEFT_THUMB     = 0x0040
XINPUT_GAMEPAD_RIGHT_THUMB    = 0x0080
XINPUT_GAMEPAD_LEFT_SHOULDER  = 0x0100
XINPUT_GAMEPAD_RIGHT_SHOULDER = 0x0200
XINPUT_GAMEPAD_A              = 0x1000
XINPUT_GAMEPAD_B              = 0x2000
XINPUT_GAMEPAD_X              = 0x4000
XINPUT_GAMEPAD_Y              = 0x8000

# Trigger threshold (0-255, values above this are considered pressed)
TRIGGER_THRESHOLD = 100

# ===== Custom Virtual Key Codes (0x300+ range, avoid conflict with Windows VK codes) =====
# Buttons
GP_VK_A              = 0x0301
GP_VK_B              = 0x0302
GP_VK_X              = 0x0303
GP_VK_Y              = 0x0304
GP_VK_LB             = 0x0305  # Left Shoulder / LB
GP_VK_RB             = 0x0306  # Right Shoulder / RB
GP_VK_LT             = 0x0307  # Left Trigger
GP_VK_RT             = 0x0308  # Right Trigger
GP_VK_BACK           = 0x0309
GP_VK_START          = 0x030A
GP_VK_LSTICK         = 0x030B  # Left Stick Click
GP_VK_RSTICK         = 0x030C  # Right Stick Click
GP_VK_DPAD_UP        = 0x030D
GP_VK_DPAD_DOWN      = 0x030E
GP_VK_DPAD_LEFT      = 0x030F
GP_VK_DPAD_RIGHT     = 0x0310

# Range check
GP_VK_MIN = 0x0301
GP_VK_MAX = 0x0310

def is_gamepad_vk(vk_code: int) -> bool:
    """Check if the virtual key code is a gamepad button"""
    return GP_VK_MIN <= vk_code <= GP_VK_MAX

# Mapping of XInput button flags to custom VK codes
_BUTTON_FLAG_TO_GP_VK = {
    XINPUT_GAMEPAD_A:              GP_VK_A,
    XINPUT_GAMEPAD_B:              GP_VK_B,
    XINPUT_GAMEPAD_X:              GP_VK_X,
    XINPUT_GAMEPAD_Y:              GP_VK_Y,
    XINPUT_GAMEPAD_LEFT_SHOULDER:  GP_VK_LB,
    XINPUT_GAMEPAD_RIGHT_SHOULDER: GP_VK_RB,
    XINPUT_GAMEPAD_BACK:           GP_VK_BACK,
    XINPUT_GAMEPAD_START:          GP_VK_START,
    XINPUT_GAMEPAD_LEFT_THUMB:     GP_VK_LSTICK,
    XINPUT_GAMEPAD_RIGHT_THUMB:    GP_VK_RSTICK,
    XINPUT_GAMEPAD_DPAD_UP:        GP_VK_DPAD_UP,
    XINPUT_GAMEPAD_DPAD_DOWN:      GP_VK_DPAD_DOWN,
    XINPUT_GAMEPAD_DPAD_LEFT:      GP_VK_DPAD_LEFT,
    XINPUT_GAMEPAD_DPAD_RIGHT:     GP_VK_DPAD_RIGHT,
}

# Reverse mapping of custom VK codes to XInput button flags
_GP_VK_TO_BUTTON_FLAG: Dict[int, int] = {v: k for k, v in _BUTTON_FLAG_TO_GP_VK.items()}

# Mapping of custom VK codes to translation keys
GP_VK_TRANSLATION_MAP = {
    GP_VK_A:          "key_gp_a",
    GP_VK_B:          "key_gp_b",
    GP_VK_X:          "key_gp_x",
    GP_VK_Y:          "key_gp_y",
    GP_VK_LB:         "key_gp_lb",
    GP_VK_RB:         "key_gp_rb",
    GP_VK_LT:         "key_gp_lt",
    GP_VK_RT:         "key_gp_rt",
    GP_VK_BACK:       "key_gp_back",
    GP_VK_START:      "key_gp_start",
    GP_VK_LSTICK:     "key_gp_lstick",
    GP_VK_RSTICK:     "key_gp_rstick",
    GP_VK_DPAD_UP:    "key_gp_dpad_up",
    GP_VK_DPAD_DOWN:  "key_gp_dpad_down",
    GP_VK_DPAD_LEFT:  "key_gp_dpad_left",
    GP_VK_DPAD_RIGHT: "key_gp_dpad_right",
}

# ===== XInput DLL 載入 =====

_xinput = None
_xinput_loaded = False

def _load_xinput():
    """嘗試載入 XInput DLL"""
    global _xinput, _xinput_loaded
    if _xinput_loaded:
        return _xinput is not None
    
    _xinput_loaded = True
    # 依序嘗試不同版本的 XInput
    for dll_name in ["xinput1_4", "xinput1_3", "xinput9_1_0"]:
        try:
            _xinput = ctypes.windll.LoadLibrary(dll_name + ".dll")
            return True
        except OSError:
            continue
    
    print("[Gamepad] 找不到 XInput DLL，手柄按鍵綁定不可用")
    _xinput = None
    return False


def get_gamepad_state(user_index: int = 0) -> Optional[XINPUT_STATE]:
    """取得指定玩家的手柄狀態
    
    Args:
        user_index: 玩家索引 (0-3)
        
    Returns:
        XINPUT_STATE 或 None（未連接/錯誤）
    """
    if not _load_xinput():
        return None
    
    state = XINPUT_STATE()
    result = _xinput.XInputGetState(user_index, ctypes.byref(state))
    if result == 0:  # ERROR_SUCCESS
        return state
    return None


def poll_pressed_gamepad_button(user_index: int = 0) -> int:
    """輪詢手柄，回傳目前按下的第一個按鈕的自訂 VK 碼
    
    用於按鍵綁定時偵測手柄按鈕。
    
    Returns:
        自訂 VK 碼 (0x0301-0x0310)，若無按下則回傳 0
    """
    state = get_gamepad_state(user_index)
    if state is None:
        return 0
    
    gp = state.Gamepad
    
    # 檢查數位按鈕
    for flag, gp_vk in _BUTTON_FLAG_TO_GP_VK.items():
        if gp.wButtons & flag:
            return gp_vk
    
    # 檢查扳機（類比 → 數位）
    if gp.bLeftTrigger > TRIGGER_THRESHOLD:
        return GP_VK_LT
    if gp.bRightTrigger > TRIGGER_THRESHOLD:
        return GP_VK_RT
    
    return 0


def is_gamepad_button_pressed(gp_vk: int, user_index: int = 0) -> bool:
    """檢查指定的手柄按鈕是否被按下
    
    Args:
        gp_vk: 自訂手柄 VK 碼 (0x0301-0x0310)
        user_index: 玩家索引 (0-3)
        
    Returns:
        是否被按下
    """
    if not is_gamepad_vk(gp_vk):
        return False
    
    state = get_gamepad_state(user_index)
    if state is None:
        return False
    
    gp = state.Gamepad
    
    # 扳機特殊處理
    if gp_vk == GP_VK_LT:
        return gp.bLeftTrigger > TRIGGER_THRESHOLD
    if gp_vk == GP_VK_RT:
        return gp.bRightTrigger > TRIGGER_THRESHOLD
    
    # 數位按鈕
    flag = _GP_VK_TO_BUTTON_FLAG.get(gp_vk, 0)
    if flag:
        return bool(gp.wButtons & flag)
    
    return False
