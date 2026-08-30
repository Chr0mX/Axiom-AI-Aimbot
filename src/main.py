# main.py
"""Program entry point and startup logic"""

from __future__ import annotations

import sys
import os

# Cap OpenMP / MKL / OpenBLAS thread pools to 1 before any DLL is loaded.
# PaddleOCR's cpu_threads=1 only limits Paddle's predictor; the underlying
# libopenblas / libomp the paddle wheel ships ignores that and uses all cores,
# starving the main inference loop and the Qt UI thread. These env vars are
# read at DLL load time so they must be set here, before any import.
# TensorRT / CUDA inference is unaffected — it runs on GPU.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

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

# ── AppData GPU packages — inject BEFORE importing onnxruntime ───────────────
# Read the configured backend before any onnxruntime import so we can decide
# whether to inject the AppData GPU packages.
# DirectML users must NOT get the GPU build — it shadows DmlExecutionProvider.
import json as _json
_cfg_path = os.path.join(project_root, "config.json")
try:
    with open(_cfg_path, "r", encoding="utf-8") as _f:
        _cfg_data = _json.load(_f)
    # v2 schema nests this under model.backend; fall back to legacy flat key.
    _early_backend = _cfg_data.get("model", {}).get("backend") \
        or _cfg_data.get("inference_backend", "auto")
except Exception:
    _early_backend = "auto"
os.environ.setdefault("AXIOM_BACKEND", _early_backend)

if _early_backend != "directml":
    _axiom_lad = os.environ.get("LOCALAPPDATA", "")
    if _axiom_lad:
        _axiom_pkg_dir = os.path.join(_axiom_lad, "AxiomAI", "site-packages")
        if os.path.isdir(_axiom_pkg_dir) and _axiom_pkg_dir not in sys.path:
            sys.path.insert(0, _axiom_pkg_dir)
            print(f"[ORT] Injected AppData packages path: {_axiom_pkg_dir}")
# ─────────────────────────────────────────────────────────────────────────────

# 初始化 pywin32 - 必須先導入 pywintypes
import pywintypes
import onnxruntime as ort

# ── CUDA / cuDNN DLL pre-registration ────────────────────────────────────────
# MUST happen immediately after `import onnxruntime` and before ANY ort call
# (including ort.get_available_providers).  ORT probes for CUDA DLLs the first
# time it is touched; if the nvidia site-package bin dirs are not on the DLL
# search path by then, it silently falls back to CPU.
#
# Expected layout installed by pip (nvidia-cublas-cu12, nvidia-cudnn-cu12, …):
#   <site-packages>/nvidia/cuda_runtime/bin/  – cudart64_12.dll
#   <site-packages>/nvidia/cublas/bin/        – cublas64_12.dll, cublasLt64_12.dll
#   <site-packages>/nvidia/cudnn/bin/         – cudnn*.dll
# ---------------------------------------------------------------------------
def _register_nvidia_dll_dirs() -> None:
    """Add every nvidia sub-package bin/ dir to the Windows DLL search path."""
    if sys.platform != "win32":
        return
    try:
        import site
        all_site_dirs: list[str] = list(site.getsitepackages())
        try:
            all_site_dirs.append(site.getusersitepackages())
        except (AttributeError, NotImplementedError):
            pass

        # Also scan AppData packages dir (installed by install_tensorrt_local.py)
        _lad = os.environ.get("LOCALAPPDATA", "")
        if _lad:
            _axiom = os.path.join(_lad, "AxiomAI", "site-packages")
            if os.path.isdir(_axiom) and _axiom not in all_site_dirs:
                all_site_dirs.append(_axiom)

        nvidia_sub_packages = [
            "cuda_runtime",
            "cublas",       # ships cublasLt64_12.dll – the one ORT needs
            "cufft",
            "curand",
            "cusolver",
            "cusparse",
            "cudnn",
        ]

        registered: list[str] = []
        for sp in all_site_dirs:
            # CUDA runtime DLLs — installed by nvidia-* wheels under nvidia/<sub>/bin/
            for sub in nvidia_sub_packages:
                bin_dir = os.path.join(sp, "nvidia", sub, "bin")
                if not os.path.isdir(bin_dir):
                    continue
                # Both PATH and add_dll_directory are needed on Win 10+
                os.environ["PATH"] = f"{bin_dir};{os.environ.get('PATH', '')}"
                try:
                    os.add_dll_directory(bin_dir)
                except (AttributeError, OSError):
                    pass
                registered.append(bin_dir)

            # TensorRT DLLs — tensorrt-cu12-libs wheel puts nvinfer_10.dll and
            # nvonnxparser_10.dll directly in site-packages/tensorrt_libs/ (not
            # under nvidia/).  ORT's TensorRT EP probes for these at load time.
            trt_libs = os.path.join(sp, "tensorrt_libs")
            if os.path.isdir(trt_libs):
                os.environ["PATH"] = f"{trt_libs};{os.environ.get('PATH', '')}"
                try:
                    os.add_dll_directory(trt_libs)
                except (AttributeError, OSError):
                    pass
                registered.append(trt_libs)

        if registered:
            logger.info("[CUDA] Registered %d nvidia/TRT DLL dirs from site-packages", len(registered))
        else:
            logger.warning("[CUDA] No nvidia site-package bin dirs found — "
                           "install nvidia-cublas-cu12, nvidia-cudnn-cu12, tensorrt-cu12, etc.")
    except Exception as exc:
        logger.error("[CUDA] DLL pre-registration failed: %s", exc)

_register_nvidia_dll_dirs()
# ─────────────────────────────────────────────────────────────────────────────

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
from core.key_listener import aim_toggle_key_listener
from gui.overlay import PyQtOverlay

from gui.status_panel import StatusPanel
from gui.disclaimer_dialog import DisclaimerDialog


# AI thread lifecycle (start_ai_threads/stop_ai_threads/pause_ai_inference/
# resume_ai_inference, plus the ai_thread/auto_fire_thread handles and the
# lock guarding them) now lives in core/app_controller.py — moved verbatim,
# no behavior change — so a web control route can call the exact same
# functions this file's own startup/shutdown code below does, instead of
# these being module-private to main.py with no other caller reachable.
from core.app_controller import (
    start_ai_threads,
    stop_ai_threads,
    pause_ai_inference,
    resume_ai_inference,
)


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

    # Web ESP overlay — stream detection state to a browser Canvas renderer (LAN)
    if getattr(config, 'web_esp_enabled', False):
        try:
            from core import esp_server
            esp_server.start(config)
        except Exception as exc:
            import logging as _logging
            _logging.getLogger(__name__).error("[WebESP] failed to start: %s", exc)

    # Web Control — control-plane LAN API (see core/web_control_server.py).
    # Unlike Web ESP above, this can mutate state, so it's gated by its own
    # flag and fails the same soft way if fastapi/uvicorn aren't vendored in
    # yet — one subsystem missing must never block the rest of the app.
    if getattr(config, 'web_control_enabled', False):
        try:
            from core import web_control_server
            web_control_server.start(
                config,
                overlay_boxes_queue=overlay_boxes_queue,
                overlay_confidences_queue=overlay_confidences_queue,
                auto_fire_boxes_queue=auto_fire_boxes_queue,
            )
        except Exception as exc:
            import logging as _logging
            _logging.getLogger(__name__).error("[WebControl] failed to start: %s", exc)

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
    settings_window.setConfigManager(ConfigManager())  # aim-only Preset manager (presets/)
    settings_window.setFullConfigManager(ConfigManager(configs_dir="configs", aim_only=False))
    
    if settings_window:
        settings_window.show()
    
    # 啟動 AI 偵測線程（使用配置中的模型路徑）
    if config.model_path:
        if not start_threads_callback(config.model_path):
            logger.warning("AI 偵測線程啟動失敗，請檢查模型路徑")

    def _shutdown() -> None:
        """Release everything the OS won't clean up for us, before exit.

        Without this, quitting relied entirely on daemon threads being
        killed mid-step and the OS reclaiming the process. That left the
        capture worker's `finally` block — which calls `_cleanup_capture()`
        and restores the 1 ms multimedia timer resolution — unreachable, so
        an open UVC device was never released cleanly: on the native-DLL
        path `capture_stop`/`capture_close` never ran, meaning
        `IMediaControl::Stop()` was never issued and the DirectShow graph
        was torn down by process death rather than by us. Some UVC bridges
        need a re-plug to recover from that.

        Every step is independently guarded: shutdown must not be the thing
        that throws, and one subsystem failing to stop must not prevent the
        rest from trying.
        """
        logger.info("Shutting down…")
        try:
            stop_ai_threads(config)
        except Exception:
            logger.exception("Error stopping AI threads during shutdown")
        try:
            from core import esp_server
            if esp_server.is_running():
                esp_server.stop()
        except Exception:
            logger.exception("Error stopping Web ESP server during shutdown")
        try:
            from core import web_control_server
            if web_control_server.is_running():
                web_control_server.stop()
        except Exception:
            logger.exception("Error stopping Web Control server during shutdown")
        try:
            save_config(config)
        except Exception:
            logger.exception("Error saving config during shutdown")
        logger.info("Shutdown complete")

    # aboutToQuit fires while the event loop is still alive, so Qt objects
    # are still usable here — unlike code after app.exec() returns, which
    # would run too late to stop anything cleanly.
    app.aboutToQuit.connect(_shutdown)

    # 啟動 PyQt 應用程式事件循環，這會管理所有 PyQt 視窗
    sys.exit(app.exec())


if __name__ == "__main__":
    # Required for multiprocessing 'spawn' (OCR child process) if this app is
    # ever frozen with PyInstaller; a harmless no-op when run from source.
    import multiprocessing
    multiprocessing.freeze_support()
    main()
