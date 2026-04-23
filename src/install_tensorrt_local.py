"""Install TensorRT Python bindings and onnxruntime-gpu into the embedded Python.

Usage (from project root):
    src\\python\\python.exe src\\install_tensorrt_local.py

What it does:
  1. Verifies the embedded Python at src/python/python.exe
  2. Installs onnxruntime-gpu (CUDA 12 wheels) if not already present
  3. Installs tensorrt-cu12 (bindings + DLLs) from pypi.nvidia.com
  4. Verifies TensorrtExecutionProvider appears in ort.get_available_providers()

Compatibility:
  - CUDA 12.x toolkit (driver >= 525.x)
  - cuDNN 9.x (bundled in nvidia-cudnn-cu12 wheel)
  - TensorRT 10.x (bundled in tensorrt-cu12 wheel)
  - onnxruntime-gpu >= 1.19 (required for TRT 10 EP)
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
LOCAL_PYTHON_DIR = BASE_DIR / "python"
PYTHON_EXE = LOCAL_PYTHON_DIR / "python.exe"
SITE_PACKAGES = LOCAL_PYTHON_DIR / "Lib" / "site-packages"

# ── Package lists ────────────────────────────────────────────────────────────
# TensorRT 10.x for CUDA 12 — hosted on pypi.nvidia.com
# tensorrt-cu12 is a meta-package that pulls in:
#   tensorrt-cu12-bindings  (Python API)
#   tensorrt-cu12-libs      (nvinfer_10.dll, nvonnxparser_10.dll, …)
TENSORRT_PACKAGES = ["tensorrt-cu12"]

# onnxruntime-gpu and its CUDA runtime wheels.
# These come from the standard PyPI index.
ONNXRUNTIME_GPU_PACKAGES = [
    "onnxruntime-gpu",
    "nvidia-cublas-cu12",
    "nvidia-cuda-runtime-cu12",
    "nvidia-cudnn-cu12",
    "nvidia-cufft-cu12",
    "nvidia-curand-cu12",
    "nvidia-cusolver-cu12",
    "nvidia-cusparse-cu12",
]

COMMON_DEPS = [
    "numpy",
    "flatbuffers",
    "packaging",
    "protobuf",
    "sympy",
    "coloredlogs",
]

# ── Helpers ──────────────────────────────────────────────────────────────────

def log(msg: str) -> None:
    print(f"[INFO] {msg}")


def warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def error(msg: str) -> None:
    print(f"[ERROR] {msg}", file=sys.stderr)


def pause_exit() -> None:
    try:
        input("Press Enter to exit...")
    except EOFError:
        pass


def fail(msg: str, code: int = 1) -> None:
    error(msg)
    pause_exit()
    sys.exit(code)


def run(cmd: list) -> None:
    display = " ".join(f'"{c}"' if " " in str(c) else str(c) for c in cmd)
    log(f"Running: {display}")
    subprocess.run(cmd, check=True)


def _query_embedded(snippet: str) -> str:
    """Run a Python snippet in the embedded interpreter and return stdout."""
    result = subprocess.run(
        [str(PYTHON_EXE), "-c", snippet],
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


# ── Detection ────────────────────────────────────────────────────────────────

def is_cuda_available() -> bool:
    out = _query_embedded(
        "import onnxruntime as ort; print('CUDAExecutionProvider' in ort.get_available_providers())"
    )
    return out == "True"


def is_tensorrt_available() -> bool:
    out = _query_embedded(
        "import onnxruntime as ort; print('TensorrtExecutionProvider' in ort.get_available_providers())"
    )
    return out == "True"


def is_tensorrt_importable() -> bool:
    out = _query_embedded("import tensorrt; print(tensorrt.__version__)")
    return bool(out)


# ── Installation ─────────────────────────────────────────────────────────────

def _pip(packages: list, upgrade: bool = True) -> None:
    cmd = [
        str(PYTHON_EXE), "-m", "pip", "install",
        "--target", str(SITE_PACKAGES),
        "--extra-index-url", "https://pypi.nvidia.com",
    ]
    if upgrade:
        cmd.append("--upgrade")
    cmd.extend(packages)
    run(cmd)


def install_onnxruntime_gpu() -> None:
    log("Installing onnxruntime-gpu and CUDA runtime wheels...")
    _pip(COMMON_DEPS)
    _pip(ONNXRUNTIME_GPU_PACKAGES)


def install_tensorrt() -> None:
    log("Installing TensorRT Python bindings (tensorrt-cu12)...")
    # Use --no-upgrade to avoid re-downloading large DLL wheels if already present
    _pip(TENSORRT_PACKAGES, upgrade=False)


# ── Verification ─────────────────────────────────────────────────────────────

def verify_installation() -> None:
    log("Verifying installation...")
    checks = [
        ("CUDAExecutionProvider",     is_cuda_available),
        ("TensorrtExecutionProvider", is_tensorrt_available),
        ("tensorrt Python package",   is_tensorrt_importable),
    ]
    all_ok = True
    for name, fn in checks:
        ok = fn()
        log(f"  {'[OK]' if ok else '[MISSING]'} {name}")
        if not ok:
            all_ok = False

    if not all_ok:
        warn("")
        warn("One or more components are missing. Common causes:")
        warn("  1. CUDA 12.x toolkit is not installed on this system")
        warn("  2. GPU driver is outdated — need >= 525.x for CUDA 12")
        warn("  3. Network error downloading from pypi.nvidia.com")
        warn("  4. GPU does not support TensorRT (requires Volta / Turing / Ampere / Ada+)")
    else:
        log("All TensorRT components installed successfully.")


def print_next_steps() -> None:
    log("")
    log("=== Next Steps ===")
    log("1. Set inference_backend = 'cuda' in the app (enables TRT > CUDA fallback)")
    log("2. On first inference the TRT engine is built — allow 1-5 minutes")
    log("3. Subsequent runs load the cached engine from trt_cache/ instantly")
    log("4. To pre-build the engine without starting the full app, run:")
    log("   src\\python\\python.exe src\\core\\convert_to_engine.py --model Model\\Roblox_8n.onnx")
    log("")


# ── Entry point ───────────────────────────────────────────────────────────────

def ensure_paths() -> None:
    if not PYTHON_EXE.exists():
        fail(
            f"Embedded Python not found: {PYTHON_EXE}\n"
            "       Run the main installer first to set up src/python."
        )
    SITE_PACKAGES.mkdir(parents=True, exist_ok=True)


def main() -> None:
    ensure_paths()

    log(f"Embedded Python : {PYTHON_EXE}")
    log(f"Site-packages   : {SITE_PACKAGES}")
    log("")

    if is_tensorrt_available():
        log("TensorrtExecutionProvider already available — nothing to do.")
        print_next_steps()
        return

    if not is_cuda_available():
        log("onnxruntime-gpu not yet installed. Installing CUDA packages first...")
        install_onnxruntime_gpu()
    else:
        log("CUDAExecutionProvider already available — skipping onnxruntime-gpu.")

    if not is_tensorrt_importable():
        install_tensorrt()
    else:
        log("tensorrt package already installed — skipping.")

    log("")
    verify_installation()
    print_next_steps()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        error("Interrupted by user.")
        pause_exit()
        sys.exit(1)
    except Exception as exc:
        error(f"Unexpected error: {exc}")
        import traceback
        traceback.print_exc()
        pause_exit()
        sys.exit(1)
