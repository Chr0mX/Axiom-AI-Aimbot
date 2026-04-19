# main.py
"""Program entry point and startup logic"""

from __future__ import annotations

import sys
import os

# Qt must see relevant environment variables before any PyQt module is imported, 
# otherwise scaling strategy will not take effect
if sys.platform == "win32":
    os.environ.setdefault('QT_ENABLE_HIGHDPI_SCALING', '0')
    os.environ.setdefault('QT_SCALE_FACTOR_ROUNDING_POLICY', 'PassThrough')

# Set DPI awareness before importing any Qt-related modules
if sys.platform == "win32":
    import ctypes
    try:
        # Priority: Consistent with Qt default: Per-Monitor V2
        _PM_V2 = ctypes.c_void_p(-4)  # DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2
        if ctypes.windll.user32.SetProcessDpiAwarenessContext(_PM_V2):
            pass
        else:
            raise OSError("SetProcessDpiAwarenessContext returned FALSE")
    except (AttributeError, OSError):
        try:
            # Fallback: System DPI aware (avoid permission errors if Qt tries to elevate later)
            ctypes.windll.shcore.SetProcessDpiAwareness(1)  # PROCESS_SYSTEM_DPI_AWARE
        except (AttributeError, OSError):
            try:
                ctypes.windll.user32.SetProcessDPIAware()
            except (AttributeError, OSError):
                pass  # DPI awareness setup failed, using system default

# Add src directory to Python path to import modules in the same directory
src_dir = os.path.dirname(os.path.abspath(__file__))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Add dependencies directory to Python path (located in src/python/dependencies)
python_dir = os.path.join(src_dir, "python")
dependencies_dir = os.path.join(python_dir, "dependencies")
if dependencies_dir not in sys.path:
    sys.path.insert(0, dependencies_dir)

# Add extra paths for pywin32
win32_dir = os.path.join(dependencies_dir, "win32")
win32_lib_dir = os.path.join(win32_dir, "lib")
if win32_dir not in sys.path:
    sys.path.insert(0, win32_dir)
if win32_lib_dir not in sys.path:
    sys.path.insert(0, win32_lib_dir)

# Ensure DLLs in dependencies directory can be found (e.g., pythoncom311.dll)
if sys.platform == "win32":
    os.environ["PATH"] = f"{dependencies_dir};{os.environ.get('PATH', '')}"
    try:
        os.add_dll_directory(dependencies_dir)
    except (AttributeError, OSError):
        pass
    
    # Try to manually preload pywin32 DLLs to resolve ImportError
    import ctypes
    import glob
    try:
        # Find and load pywintypesXXX.dll
        pywintypes_dlls = glob.glob(os.path.join(dependencies_dir, "pywintypes*.dll"))
        if pywintypes_dlls:
            ctypes.WinDLL(pywintypes_dlls[0])
            
        # Find and load pythoncomXXX.dll
        pythoncom_dlls = glob.glob(os.path.join(dependencies_dir, "pythoncom*.dll"))
        if pythoncom_dlls:
            ctypes.WinDLL(pythoncom_dlls[0])
    except Exception as e:
        print(f"Warning: Failed to preload pywin32 DLLs: {e}")

# 初始化統一的 logging 設定
from core.logging_config import setup_logging


from version import __version__
logger = setup_logging("INFO")
logger.info(f"Axiom v{__version__} Starting...")

# 獲取項目根目錄（src 的父目錄）
project_root = os.path.dirname(src_dir)

import threading
import queue
from typing import Optional

# 初始化 pywin32 - 必須先導入 pywintypes
import pywintypes
import onnxruntime as ort

# When bundled with PyInstaller, ensure native dependencies are discoverable.
_DLL_DIR_HANDLES = []
if sys.platform == "win32":
    def _maybe_add_dll_dir(path: str):
        if not path:
            return
        try:
            handle = os.add_dll_directory(path)
            _DLL_DIR_HANDLES.append(handle)
        except AttributeError:
            os.environ["PATH"] = f"{path};{os.environ.get('PATH', '')}"
        except (FileNotFoundError, NotADirectoryError):
            pass

    if getattr(sys, "frozen", False):
        base_dir = getattr(sys, '_MEIPASS', '')
        search_roots = [
            base_dir,
            os.path.join(base_dir, 'onnxruntime'),
            os.path.join(base_dir, 'onnxruntime', 'capi'),
        ]
        for candidate in search_roots:
            if candidate and os.path.isdir(candidate):
                _maybe_add_dll_dir(candidate)

        exe_dir = os.path.dirname(sys.executable)
        fallback_dirs = [
            os.path.join(exe_dir, 'onnxruntime'),
            os.path.join(exe_dir, 'onnxruntime', 'capi'),
        ]
        for candidate in fallback_dirs:
            if os.path.isdir(candidate):
                _maybe_add_dll_dir(candidate)

# 從我們自己建立的模組中導入
from core.config import Config, load_config, save_config
from win_utils import check_and_request_admin, test_ddxoft_functions, ensure_ddxoft_ready
from core.session_utils import build_provider_list, optimize_onnx_session
from core.ai_loop import ai_logic_loop
from core.auto_fire import auto_fire_loop
from core.key_listener import aim_toggle_key_listener
from gui.overlay import PyQtOverlay

from gui.status_panel import StatusPanel
from gui.disclaimer_dialog import DisclaimerDialog


# 全域變數宣告
ai_thread: Optional[threading.Thread] = None
auto_fire_thread: Optional[threading.Thread] = None

def start_ai_threads(
    config: Config,
    overlay_boxes_queue: queue.Queue,
    overlay_confidences_queue: queue.Queue,
    auto_fire_boxes_queue: queue.Queue,
    model_path: str
) -> bool:
    """由 GUI 呼叫，載入模型並啟動/重啟 AI 執行緒
    
    Args:
        config: 配置實例
        boxes_queue: 檢測框隊列
        confidences_queue: 置信度隊列
        model_path: 模型路徑
        
    Returns:
        是否成功啟動
    """
    global ai_thread, auto_fire_thread
    
    # 停止現有線程
    if ai_thread is not None and ai_thread.is_alive():
        config.Running = False
        ai_thread.join(timeout=3.0)
        if auto_fire_thread is not None:
            auto_fire_thread.join(timeout=3.0)
        # 確認舊執行緒已結束
        if ai_thread.is_alive():
            logger.warning("AI 執行緒在 3 秒內未停止，強制繼續")

    config.Running = True
    
    # 僅支持 ONNX 模型
    if not model_path.endswith('.onnx'):
        logger.error("僅支援 .onnx 模型格式: %s", model_path)
        return False
    
    # 將相對路徑轉換為絕對路徑（相對於項目根目錄）
    if not os.path.isabs(model_path):
        model_path = os.path.join(project_root, model_path)
    
    # 檢查文件是否存在
    if not os.path.exists(model_path):
        logger.error("模型文件不存在: %s", model_path)
        return False
    
    model = None
    try:
        providers = build_provider_list(config)
        logger.info("嘗試載入 ONNX providers: %s", providers)

        # 獲取優化的會話選項
        session_options = optimize_onnx_session(config)
        if session_options:
            model = ort.InferenceSession(model_path, providers=providers, sess_options=session_options)
        else:
            model = ort.InferenceSession(model_path, providers=providers)

        # 獲取實際使用的提供者
        actual_providers = model.get_providers()
        if actual_providers:
            config.current_provider = actual_providers[0]
            logger.info("模型載入使用提供者: %s", actual_providers[0])
            logger.info("最終啟用 ONNX provider: %s", config.current_provider)

            requested_backend = getattr(config, "inference_backend", "auto")
            if requested_backend == "cuda" and config.current_provider != "CUDAExecutionProvider":
                logger.warning(
                    "已選擇 CUDA 後端，但實際 provider 為 %s。請檢查 onnxruntime-gpu、NVIDIA Driver、CUDA/cuDNN 相容性。",
                    config.current_provider,
                )
        else:
            logger.warning("無法獲取提供者資訊")
            config.current_provider = providers[0] if providers else 'CPUExecutionProvider'
            logger.info("最終啟用 ONNX provider: %s", config.current_provider)
    except Exception as e:
        logger.error("載入 ONNX 模型失敗: %s", e)
        logger.error("請確認已安裝對應 ONNX Runtime 後端（CUDA/DirectML/CPU）")
        return False

    ai_thread = threading.Thread(
        target=ai_logic_loop,
        args=(config, model, 'onnx', overlay_boxes_queue, overlay_confidences_queue, auto_fire_boxes_queue),
        daemon=True
    )
    auto_fire_thread = threading.Thread(
        target=auto_fire_loop,
        args=(config, auto_fire_boxes_queue),
        daemon=True
    )
    
    ai_thread.start()
    auto_fire_thread.start()
    return True


def main():
    """主程式入口"""
    # 檢查管理員權限
    check_and_request_admin()

    config = Config()
    load_config(config)

    try:
        available_providers = ort.get_available_providers()
    except Exception as e:
        available_providers = ["CPUExecutionProvider"]
        logger.warning("取得可用 ONNX providers 失敗，預設為 CPUExecutionProvider：%s", e)

    selected_backend = getattr(config, "inference_backend", "auto")
    logger.info("ONNX 可用 providers: %s", available_providers)
    logger.info("設定選擇推理後端: %s", selected_backend)
    logger.info("最終啟用 ONNX provider: 尚未載入模型")
    
    # 調試：顯示載入的滑鼠移動方式
    logger.info("配置載入：滑鼠移動方式 %s", config.mouse_move_method)
    
    # 僅在使用者配置選擇 ddxoft 時才初始化/測試，避免啟動即載入高風險元件
    if config.mouse_move_method == 'ddxoft':
        try:
            if ensure_ddxoft_ready():
                test_ddxoft_functions()
            else:
                logger.warning("ddxoft 初始化失敗，已改用 mouse_event 以降低崩潰風險")
                config.mouse_move_method = 'mouse_event'
                config.mouse_click_method = 'mouse_event'
        except Exception as e:
            logger.warning("ddxoft 初始化/測試時發生例外，已改用 mouse_event：%s", e)
            config.mouse_move_method = 'mouse_event'
            config.mouse_click_method = 'mouse_event'
    
    # 優化：使用配置中的隊列大小設置
    overlay_boxes_queue: queue.Queue = queue.Queue(maxsize=config.max_queue_size)
    overlay_confidences_queue: queue.Queue = queue.Queue(maxsize=config.max_queue_size)
    auto_fire_boxes_queue: queue.Queue = queue.Queue(maxsize=config.max_queue_size)

    # 創建啟動函數的閉包
    def start_threads_callback(model_path: str) -> bool:
        return start_ai_threads(
            config,
            overlay_boxes_queue,
            overlay_confidences_queue,
            auto_fire_boxes_queue,
            model_path,
        )

    # 啟動快捷鍵監聽
    toggle_thread = threading.Thread(
        target=aim_toggle_key_listener, 
        args=(config,), 
        daemon=True
    )
    toggle_thread.start()

    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtCore import Qt

    # 必須在 QApplication 建立前設定屬性
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_UseDesktopOpenGL)

    # 在主線程中創建 QApplication
    app = QApplication(sys.argv)
    
    # 檢查免責聲明同意狀態
    if not config.disclaimer_agreed:
        disclaimer = DisclaimerDialog()
        if disclaimer.exec() == 1:  # 1 = Accepted
            config.disclaimer_agreed = True
            save_config(config)
        else:
            sys.exit(0)

    # ── 首次啟動設置精靈 ──────────────────────────────
    if not config.first_run_complete:
        from gui.fluent_app.setup_wizard import SetupWizard
        wizard = SetupWizard(config)
        wizard.exec()
        # 無論完成或跳過，都套用主題並標記完成
        config.dark_mode = wizard._isDark          # ← 同步暗色主題到 config
        wizard.applyChosenTheme()
        config.first_run_complete = True
        save_config(config)
    
    # 建立並顯示主要的繪圖覆蓋層 (人物框, FOV)
    main_overlay = PyQtOverlay(overlay_boxes_queue, overlay_confidences_queue, config)
    main_overlay.show()

    # 建立並顯示新的狀態面板（根據配置決定是否顯示）
    status_panel = StatusPanel(config)
    if config.show_status_panel:
        status_panel.show()
    else:
        status_panel.hide()
    
    # 根據配置控制終端視窗的顯示
    from win_utils import show_console, hide_console
    if config.show_console:
        show_console()
    else:
        hide_console()
    
    # 在主線程中創建設置 GUI（不使用線程）
    from gui.fluent_app.window import AxiomWindow
    from core.config_manager import ConfigManager
    
    settings_window = AxiomWindow()
    
    # 注入配置實例給 GUI
    settings_window.setConfig(config)
    settings_window.setConfigManager(ConfigManager())
    
    if settings_window:
        settings_window.show()
    
    # 啟動 AI 偵測線程（使用配置中的模型路徑）
    if config.model_path:
        if not start_threads_callback(config.model_path):
            logger.warning("AI 偵測線程啟動失敗，請檢查模型路徑")
    
    # 啟動 PyQt 應用程式事件循環，這會管理所有 PyQt 視窗
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
