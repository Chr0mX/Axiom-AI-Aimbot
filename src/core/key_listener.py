# key_listener.py
"""快捷鍵監聽模組 - 處理全域快捷鍵事件"""

import time
import win32api

from win_utils import get_vk_name
from win_utils.gamepad_input import is_gamepad_vk, is_gamepad_button_pressed


def aim_toggle_key_listener(config, update_gui_callback=None):
    """持續監聽自動瞄準開關快捷鍵
    
    在獨立線程中運行，監測指定按鍵的按下事件，
    按下時切換 config.AimToggle 的狀態。
    
    Args:
        config: 配置實例，需包含以下屬性：
            - Running: bool，控制監聽循環是否繼續
            - aim_toggle_key: int，切換用的虛擬按鍵碼
            - AimToggle: bool，自動瞄準的開關狀態
        update_gui_callback: 可選的回調函數，狀態變更時調用
    
    Note:
        此函數應在 daemon 線程中運行，會在 config.Running 為 False 時結束
    """
    last_state = False
    key_code = getattr(config, 'aim_toggle_key', 0x78)  # 備用預設值 F9 鍵（實際預設由 config.py 定義為 Insert 鍵）
    
    # 獲取按鍵名稱
    key_name = get_vk_name(key_code)
    
    sleep_interval = 0.03  # 30ms 檢查間隔
    
    
    # 使用無限循環，因為此線程需要在應用程式整個生命週期內運行
    # config.Running 會在重啟 AI 線程時被設為 False，不應影響快捷鍵監聽
    while True:
        try:
            # 重新獲取快捷鍵設置
            current_key_code = getattr(config, 'aim_toggle_key', 0x78)
            if current_key_code != key_code:
                key_code = current_key_code
                key_name = get_vk_name(key_code)
            
            # 檢測按鍵狀態（支援鍵盤/滑鼠/手柄）
            if is_gamepad_vk(key_code):
                state = is_gamepad_button_pressed(key_code)
            else:
                state = bool(win32api.GetAsyncKeyState(key_code) & 0x8000)
            
            # 檢測按鍵按下事件
            if state and not last_state:
                old_state = config.AimToggle
                config.AimToggle = not config.AimToggle
                print(f"[快捷鍵] 自動瞄準: {old_state} → {config.AimToggle}")
                
                if update_gui_callback:
                    update_gui_callback(config.AimToggle)
            
            last_state = state
            
        except Exception as e:
            print(f"[快捷鍵監聽] 錯誤: {e}")
            import traceback
            traceback.print_exc()
        
        time.sleep(sleep_interval)

