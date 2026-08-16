# session_utils.py
"""ONNX Runtime session optimization - inference performance configuration."""

import logging
import os
import sys
import threading
import time

# ── AppData packages path ─────────────────────────────────────────────────────
# install_tensorrt_local.py writes packages to %LOCALAPPDATA%\AxiomAI\site-packages.
# Add that directory to sys.path before any package imports so Python finds them.

def _inject_axiom_packages() -> None:
    if os.environ.get("AXIOM_BACKEND") == "directml":
        return
    _localappdata = os.environ.get("LOCALAPPDATA", "")
    if not _localappdata:
        return
    pkg_dir = os.path.join(_localappdata, "AxiomAI", "site-packages")
    if os.path.isdir(pkg_dir) and pkg_dir not in sys.path:
        sys.path.insert(0, pkg_dir)


_inject_axiom_packages()

# ── DLL pre-registration ──────────────────────────────────────────────────────
# TensorRT pip wheels install DLLs under their package dir but do NOT add
# themselves to PATH.  Without this, onnxruntime_providers_tensorrt.dll fails
# with "nvinfer_10.dll missing" even when TRT is installed.  Register every
# known DLL directory before the first import of onnxruntime.

def _register_trt_dll_dirs() -> None:
    """Add TensorRT and CUDA pip-wheel DLL dirs to PATH/add_dll_directory (Windows only)."""
    if os.environ.get("AXIOM_BACKEND") == "directml":
        return
    if sys.platform != "win32":
        return
    try:
        # Collect all candidate package roots: standard site-packages + AxiomAI AppData dir
        import site
        site_dirs: list = list(site.getsitepackages())
        try:
            site_dirs.append(site.getusersitepackages())
        except (AttributeError, NotImplementedError):
            pass
        _localappdata = os.environ.get("LOCALAPPDATA", "")
        if _localappdata:
            axiom_pkg = os.path.join(_localappdata, "AxiomAI", "site-packages")
            if os.path.isdir(axiom_pkg):
                site_dirs.append(axiom_pkg)

        _CUDA_SUBS = (
            "cuda_runtime", "cublas", "cudnn",
            "cufft", "curand", "cusolver", "cusparse",
        )

        def _add(path: str) -> None:
            os.environ["PATH"] = f"{path};{os.environ.get('PATH', '')}"
            try:
                os.add_dll_directory(path)
            except (AttributeError, OSError):
                pass

        for sp in site_dirs:
            trt_libs = os.path.join(sp, "tensorrt_libs")
            if os.path.isdir(trt_libs):
                _add(trt_libs)
            for sub in _CUDA_SUBS:
                bin_dir = os.path.join(sp, "nvidia", sub, "bin")
                if os.path.isdir(bin_dir):
                    _add(bin_dir)
    except Exception:
        pass  # never crash the app over a PATH tweak


_register_trt_dll_dirs()

# ─────────────────────────────────────────────────────────────────────────────

import onnxruntime as ort


class InferenceController:
    """Thread-safe pause/stop controller for the AI inference loop.

    Use pause() / resume() to temporarily halt inference without killing threads
    or destroying the ONNX session — useful for in-app operations that need the
    GPU free (e.g. driver updates, lightweight config reloads).

    Use request_stop() to signal a full shutdown; the loop exits cooperatively on
    its next iteration.

    Event semantics
    ---------------
    _pause_event  SET   → loop should sleep (paused)
                  CLEAR → loop should run  (normal)
    _stop_event   SET   → loop should exit
                  CLEAR → loop should keep running
    """

    def __init__(self) -> None:
        self._pause_event: threading.Event = threading.Event()
        self._stop_event: threading.Event = threading.Event()

    # ── Public API ────────────────────────────────────────────────────────────

    def pause(self) -> None:
        """Signal the inference loop to pause on its next iteration."""
        self._pause_event.set()

    def resume(self) -> None:
        """Clear the pause signal so the inference loop resumes."""
        self._pause_event.clear()

    def request_stop(self) -> None:
        """Signal the inference loop to exit cleanly."""
        self._stop_event.set()
        self._pause_event.clear()  # unblock wait_while_paused so thread can exit

    def clear_stop(self) -> None:
        """Reset the stop flag (e.g. before restarting a loop)."""
        self._stop_event.clear()

    # ── State queries ─────────────────────────────────────────────────────────

    @property
    def should_pause(self) -> bool:
        return self._pause_event.is_set()

    @property
    def should_stop(self) -> bool:
        return self._stop_event.is_set()

    # ── Blocking helper for use inside the inference loop ─────────────────────

    def wait_while_paused(self, check_interval: float = 0.05) -> bool:
        """Block the calling thread while paused.

        Returns True if the loop should continue, False if a stop was requested
        while waiting (caller should exit the loop in that case).
        """
        while self._pause_event.is_set():
            if self._stop_event.is_set():
                return False
            time.sleep(check_interval)
        return not self._stop_event.is_set()


# Module-level singleton — imported by ai_loop and main to share state.
inference_controller = InferenceController()

# Project root: src/core/session_utils.py → up two levels → project root
_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_MODULE_DIR))
_TRT_CACHE_DIR = os.path.join(_PROJECT_ROOT, "trt_cache")


def _ensure_trt_cache_dir() -> str:
    """Create and return the TRT engine / timing cache directory."""
    os.makedirs(_TRT_CACHE_DIR, exist_ok=True)
    return _TRT_CACHE_DIR


def build_provider_list(config) -> list:
    """Build ORT provider priority list based on user backend preference.

    Priority order when backend == 'auto':
        TensorRT > DirectML > CUDA > CPU

    Priority order when backend == 'cuda':
        TensorRT > CUDA > CPU  (TRT is tried first; falls back gracefully)
    """
    logger = logging.getLogger(__name__)
    try:
        available = set(ort.get_available_providers())
    except Exception:
        available = {"CPUExecutionProvider"}

    backend = getattr(config, "inference_backend", "auto")

    if backend == "auto":
        if "TensorrtExecutionProvider" in available:
            preferred = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
            logger.info("Auto-selected backend: TensorRT")
        elif "DmlExecutionProvider" in available:
            preferred = ["DmlExecutionProvider", "CPUExecutionProvider"]
            logger.info("Auto-selected backend: DirectML")
        elif "CUDAExecutionProvider" in available:
            preferred = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            logger.info("Auto-selected backend: CUDA")
        else:
            preferred = ["CPUExecutionProvider"]
            logger.info("Auto-selected backend: CPU")
    else:
        provider_map = {
            "tensorrt": ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
            "cuda":     ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
            "directml": ["DmlExecutionProvider", "CPUExecutionProvider"],
            "cpu":      ["CPUExecutionProvider"],
        }
        preferred = provider_map.get(backend, ["CUDAExecutionProvider", "CPUExecutionProvider"])
        active = [p for p in preferred if p in available]
        logger.info("Backend '%s' → using %s", backend, active[0] if active else "CPUExecutionProvider")

    # Only keep providers actually reported by this ORT build
    filtered = [p for p in preferred if p in available]

    trt_cache = _ensure_trt_cache_dir()

    result: list = []
    for provider in filtered:
        if provider == "TensorrtExecutionProvider":
            fp16_enabled = bool(getattr(config, 'trt_fp16_enabled', True))
            _model_stem = os.path.splitext(
                os.path.basename(getattr(config, 'model_path', '') or '')
            )[0]
            trt_opts: dict = {
                # ── Engine cache ─────────────────────────────────────────
                # Persist the compiled engine so the 1-5 min build cost is
                # paid only on the first run.  Subsequent launches are instant.
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": trt_cache,

                # ── Timing cache ─────────────────────────────────────────
                # Reuse layer-timing data across engine rebuilds (e.g. after
                # a model update).  Drastically reduces re-build time.
                "trt_timing_cache_enable": True,
                "trt_timing_cache_path": trt_cache,

                # ── Precision ────────────────────────────────────────────
                # FP16 is native on RTX (Turing+) and roughly 2x faster than
                # FP32 with negligible accuracy loss for YOLO detection.
                # Controlled by config.trt_fp16_enabled.
                "trt_fp16_enable": fp16_enabled,

                # ── Builder memory budget ────────────────────────────────
                # 2 GiB is enough for YOLOv8-n/s.  Increase to 4 GiB for
                # larger models (YOLOv8-m/l/x) if the build OOMs.
                "trt_max_workspace_size": 2 * 1024 * 1024 * 1024,

                # ── Optimization level ───────────────────────────────────
                # 3 = good balance of build time vs runtime speed (range 0-5).
                # Use 5 only when you can afford a multi-hour build.
                "trt_builder_optimization_level": 3,

                # ── Auxiliary streams ────────────────────────────────────
                # -1 = TRT manages its own CUDA streams automatically.
                "trt_auxiliary_streams": -1,
            }
            # Prefix engine cache filenames with the model stem so files are
            # human-readable (e.g. Roblox_8n_<hash>_sm75_fp16.engine).
            # Must match the prefix used during convert_to_engine.py pre-build.
            if _model_stem:
                trt_opts["trt_engine_cache_prefix"] = _model_stem
            result.append(("TensorrtExecutionProvider", trt_opts))
        elif provider == "CUDAExecutionProvider":
            result.append((
                "CUDAExecutionProvider",
                {
                    "cudnn_conv_algo_search": "HEURISTIC",
                    "do_copy_in_default_stream": True,
                    "arena_extend_strategy": "kSameAsRequested",
                },
            ))
        else:
            result.append(provider)

    return result or ["CPUExecutionProvider"]


def optimize_onnx_session(config):
    """Create ORT SessionOptions with graph and memory optimizations."""
    logger = logging.getLogger(__name__)
    try:
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        session_options.enable_mem_pattern = True
        session_options.enable_cpu_mem_arena = True

        try:
            session_options.intra_op_num_threads = 1
            session_options.inter_op_num_threads = 1
        except Exception as e:
            logger.warning("Thread count config failed: %s", e)

        try:
            session_options.add_session_config_entry("session.intra_op.allow_spinning", "0")
            session_options.add_session_config_entry("session.inter_op.allow_spinning", "0")
        except Exception as e:
            logger.warning("allow_spinning config failed: %s", e)

        return session_options

    except Exception as e:
        logger.error("Session options creation failed: %s", e)
        return None
