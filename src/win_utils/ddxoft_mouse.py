# ddxoft_mouse.py - DDXoft Mouse Control Module
"""ddxoft (Most Stealthy) - Object-Oriented Interface"""

import ctypes
import logging
import threading
import time

from .mouse_move import send_mouse_move_mouse_event

logger = logging.getLogger(__name__)


class DDXoftMouse:
    """DDXoft Mouse Controller
    
    Achieve driver-level mouse movement and clicking through ddxoft.dll.
    This method is more stealthy than the Windows API and less likely to be detected by anti-cheat systems.
    
    Requirements:
    - ddxoft.dll placed in the program directory
    - Program running with administrator privileges
    
    Attributes:
        available: Whether the DLL was successfully initialized
        success_count: Number of successful operations
        failure_count: Number of failed operations
        last_status: Status of the last operation
    """
    
    _INIT_RETRY_COOLDOWN_S = 10.0  # don't hammer a failing DLL/driver every frame

    def __init__(self):
        self.dll = None
        self.available = False
        self.subsequent_init_failed = False  # 記錄是否初始化失敗過，防止重複嘗試
        self._last_init_attempt = 0.0        # for the retry cooldown below
        self.success_count = 0      # 成功次數
        self.failure_count = 0      # 失敗次數
        self.last_status = None     # 最後一次操作狀態
        # Unlike every other mouse backend in this codebase (MAKCU, Arduino,
        # Xbox), this class had no lock at all — move_relative() (called
        # from the inference thread) and click_left() (called from the
        # separate auto-fire thread) both touch the same DLL handle and
        # stats counters with no synchronization. Held only around the
        # actual DLL calls/state mutations, never across a sleep — same
        # discipline the other backends already follow.
        self._lock = threading.Lock()

    def ensure_initialized(self):
        """Lazy-load the ddxoft DLL when needed."""
        with self._lock:
            return self._ensure_initialized_locked()

    def _ensure_initialized_locked(self):
        """Caller must hold self._lock."""
        if self.available:
            return True
        # If it failed before, don't retry every single frame — but do retry
        # after a cooldown, since the failure may be transient (driver not
        # loaded yet, DLL locked by another process, etc.) rather than
        # permanent.
        if self.subsequent_init_failed:
            if time.time() - self._last_init_attempt < self._INIT_RETRY_COOLDOWN_S:
                return False
            self.subsequent_init_failed = False

        return self._init_dll()


    def _init_dll(self):
        """Initialize ddxoft DLL. Caller must hold self._lock."""
        if self.available:
            return True

        self._last_init_attempt = time.time()

        try:
            # Try loading ddxoft DLL (common locations)
            dll_paths = [
                "ddxoft.dll",  # Current directory
                "src/ddxoft.dll",  # src directory
                "lib/ddxoft.dll",  # lib directory
            ]
            
            for dll_path in dll_paths:
                try:
                    self.dll = ctypes.CDLL(dll_path)
                    break
                except OSError:
                    continue
            
            if self.dll is None:
                self.subsequent_init_failed = True
                return False
                
            # Set function prototypes
            self.dll.DD_btn.argtypes = [ctypes.c_int]
            self.dll.DD_btn.restype = ctypes.c_int
            self.dll.DD_str.argtypes = [ctypes.c_char_p]
            self.dll.DD_str.restype = ctypes.c_int
            self.dll.DD_movR.argtypes = [ctypes.c_int, ctypes.c_int]
            self.dll.DD_movR.restype = ctypes.c_int
            
            # Execute initialization sequence
            # Step 1: Call DD_btn(0) for initialization
            # Note: If driver is missing, this step may pop up a "Scarica ddxxxx.sys" message box
            btn_result = self.dll.DD_btn(0)
            
            # Step 2: Call DD_str to set the free version identifier
            str_result = self.dll.DD_str(b"dd2")
            
            # Check initialization results
            if btn_result == 1 and str_result == 1:
                self.available = True
                return True
            else:
                self.subsequent_init_failed = True
                logger.error("[ddxoft] Initialization failed: DD_btn or DD_str returned error code. "
                             "Tip: Windows Memory Integrity may be preventing the driver from loading.")
                return False
            
        except Exception as e:
            self.subsequent_init_failed = True
            logger.error("[ddxoft] Initialization exception: %s", e)
            return False
    
    def move_relative(self, dx, dy):
        """相對移動滑鼠"""
        with self._lock:
            if not self._ensure_initialized_locked():
                self.failure_count += 1
                self.last_status = "DLL_NOT_AVAILABLE"
                return False

            try:
                # 確保參數為整數且在合理範圍內
                dx = max(-32767, min(32767, int(dx)))
                dy = max(-32767, min(32767, int(dy)))

                # 使用 DD_movR 進行相對移動
                result = self.dll.DD_movR(dx, dy)

                if result == 1:
                    self.success_count += 1
                    self.last_status = "SUCCESS"
                    return True
                else:
                    self.failure_count += 1
                    self.last_status = f"FAILED_CODE_{result}"
                    return False

            except Exception as e:
                self.failure_count += 1
                self.last_status = f"EXCEPTION_{type(e).__name__}"
                return False

    def click_left(self):
        """左鍵點擊"""
        with self._lock:
            if not self._ensure_initialized_locked():
                self.failure_count += 1
                self.last_status = "DLL_NOT_AVAILABLE"
                return False
            try:
                # 使用 DD_btn 進行滑鼠點擊: 1 = 左鍵按下
                down_result = self.dll.DD_btn(1)
            except Exception as e:
                self.failure_count += 1
                self.last_status = f"CLICK_EXCEPTION_{type(e).__name__}"
                return False

        # Released before the delay — never hold the lock across
        # time.sleep(), same discipline every other backend follows.
        time.sleep(0.001)  # 1ms延遲，確保按下和釋放被正確識別

        with self._lock:
            try:
                up_result = self.dll.DD_btn(2)  # 2 = 左鍵釋放
            except Exception as e:
                self.failure_count += 1
                self.last_status = f"CLICK_EXCEPTION_{type(e).__name__}"
                return False

            if down_result == 1 and up_result == 1:
                self.success_count += 1
                self.last_status = "CLICK_SUCCESS"
                return True
            else:
                self.failure_count += 1
                self.last_status = f"CLICK_FAILED_DOWN_{down_result}_UP_{up_result}"
                return False
    
    def is_available(self):
        """檢查 ddxoft 是否可用"""
        return self.available
    
    def get_statistics(self):
        """獲取使用統計"""
        total = self.success_count + self.failure_count
        success_rate = (self.success_count / total * 100) if total > 0 else 0
        return {
            'success_count': self.success_count,
            'failure_count': self.failure_count,
            'total_count': total,
            'success_rate': success_rate,
            'last_status': self.last_status
        }
    
    def reset_statistics(self):
        """重置統計數據"""
        self.success_count = 0
        self.failure_count = 0
        self.last_status = None
    
    def print_statistics(self):
        stats = self.get_statistics()
        logger.info(
            "[ddxoft] Stats — success: %d  failure: %d  total: %d  rate: %.1f%%  last: %s",
            stats['success_count'], stats['failure_count'], stats['total_count'],
            stats['success_rate'], stats['last_status'],
        )
    
    def test_functionality(self):
        """測試 ddxoft 功能並診斷問題"""
        if not self.ensure_initialized():
            return False
        
        # 測試小幅度移動
        test_moves = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        success_count = 0
        
        for dx, dy in test_moves:
            if self.move_relative(dx, dy):
                success_count += 1
            time.sleep(0.1)  # 短暫延遲
        
        return success_count > 0


# 創建全局 ddxoft_mouse 實例
ddxoft_mouse = DDXoftMouse()


def send_mouse_move_ddxoft(dx, dy):
    """ddxoft 移動（最隱蔽）"""
    if not ddxoft_mouse.ensure_initialized():
        send_mouse_move_mouse_event(dx, dy)
        return

    # 嘗試使用 ddxoft
    if ddxoft_mouse.move_relative(dx, dy):
        return  # 成功，直接返回
    
    # ddxoft 失敗時靜默回退到 mouse_event
    send_mouse_move_mouse_event(dx, dy)


# ===== 公共接口函數 =====

def ensure_ddxoft_ready():
    """確保 ddxoft DLL 已初始化。"""
    return ddxoft_mouse.ensure_initialized()


def test_ddxoft_functions():
    """測試 ddxoft 功能的公共接口"""
    return ddxoft_mouse.test_functionality()


def get_ddxoft_statistics():
    """獲取 ddxoft 統計信息的公共接口"""
    return ddxoft_mouse.get_statistics()


def reset_ddxoft_statistics():
    """重置 ddxoft 統計信息的公共接口"""
    global _ddxoft_move_count
    _ddxoft_move_count = 0
    return ddxoft_mouse.reset_statistics()
