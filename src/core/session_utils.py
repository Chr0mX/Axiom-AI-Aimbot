# session_utils.py
"""ONNX Runtime session optimization - inference performance configuration."""

import logging
import os

# When AXIOM_INSTALLING=1 the install scripts are running and onnxruntime may
# not be in a usable state (CUDA DLLs absent, mid-install, etc.).  Skip all
# ORT imports and return safe CPU-only defaults so the process does not exit.
_INSTALLING = os.environ.get("AXIOM_INSTALLING", "0") == "1"

if not _INSTALLING:
    try:
        import onnxruntime as ort
        _ORT_AVAILABLE = True
    except Exception as _ort_err:
        logging.getLogger(__name__).warning(
            "onnxruntime import failed, falling back to CPU: %s", _ort_err
        )
        _ORT_AVAILABLE = False
else:
    _ORT_AVAILABLE = False

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

    Returns ["CPUExecutionProvider"] when ORT is unavailable or when
    AXIOM_INSTALLING=1 (installation mode) to prevent process exit.

    Priority order when backend == 'auto':
        TensorRT > DirectML > CUDA > CPU

    Priority order when backend == 'cuda':
        TensorRT > CUDA > CPU  (TRT is tried first; falls back gracefully)
    """
    logger = logging.getLogger(__name__)

    if not _ORT_AVAILABLE:
        logger.warning("ORT not available (installing=%s) — using CPUExecutionProvider", _INSTALLING)
        return ["CPUExecutionProvider"]

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
            "cuda":    ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
            "directml":["DmlExecutionProvider", "CPUExecutionProvider"],
            "cpu":     ["CPUExecutionProvider"],
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
            result.append((
                "TensorrtExecutionProvider",
                {
                    # Persist the compiled engine so the 1-5 min build cost is
                    # paid only on the first run.  Subsequent launches are instant.
                    "trt_engine_cache_enable": True,
                    "trt_engine_cache_path": trt_cache,

                    # Reuse layer-timing data across engine rebuilds (e.g. after
                    # a model update).  Drastically reduces re-build time.
                    "trt_timing_cache_enable": True,
                    "trt_timing_cache_path": trt_cache,

                    # FP16 is native on RTX (Turing+) and roughly 2x faster than
                    # FP32 with negligible accuracy loss for YOLO detection.
                    "trt_fp16_enable": True,

                    # 2 GiB is enough for YOLOv8-n/s.  Increase to 4 GiB for
                    # larger models (YOLOv8-m/l/x) if the build OOMs.
                    "trt_max_workspace_size": 2 * 1024 * 1024 * 1024,

                    # 3 = good balance of build time vs runtime speed (range 0-5).
                    "trt_builder_optimization_level": 3,

                    # -1 = TRT manages its own CUDA streams automatically.
                    "trt_auxiliary_streams": -1,
                },
            ))
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
    """Create ORT SessionOptions with graph and memory optimizations.

    Returns None when ORT is unavailable (e.g. during installation).
    """
    logger = logging.getLogger(__name__)

    if not _ORT_AVAILABLE:
        return None

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
