"""Install TensorRT Python bindings and onnxruntime-gpu.

Packages are written to:
    %LOCALAPPDATA%\\Packages\\AxiomAI

This location survives app reinstalls and is isolated from other Python
environments on the machine.  Axiom picks the packages up at startup via
sys.path injection in session_utils.py.

Usage (from project root, any Python ≥ 3.10):
    python src\\install_tensorrt_local.py
    -- or --
    src\\python\\python.exe src\\install_tensorrt_local.py

Compatibility:
  CUDA 12.x toolkit  (driver >= 525.x)
  cuDNN 9.x          (bundled in nvidia-cudnn-cu12)
  TensorRT 10.x      (tensorrt-cu12 < 11)
  onnxruntime-gpu    >= 1.19
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# ── Install target ────────────────────────────────────────────────────────────
_LOCALAPPDATA = os.environ.get("LOCALAPPDATA", "")
if not _LOCALAPPDATA:
    print("[ERROR] LOCALAPPDATA environment variable is not set.", file=sys.stderr)
    sys.exit(1)

PACKAGES_DIR = Path(_LOCALAPPDATA) / "Packages" / "AxiomAI"

# ── Package lists ─────────────────────────────────────────────────────────────
# TensorRT 10.x for CUDA 12.  Pinned to <11 because onnxruntime-gpu links
# nvinfer_10.dll; TensorRT 11 ships nvinfer_11.dll instead, breaking the EP.
TENSORRT_PACKAGES = ["tensorrt-cu12<11"]

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


# ── Output helpers ────────────────────────────────────────────────────────────

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


# ── Subprocess helpers ────────────────────────────────────────────────────────

def run(cmd: list) -> None:
    display = " ".join(f'"{c}"' if " " in str(c) else str(c) for c in cmd)
    log(f"Running: {display}")
    subprocess.run(cmd, check=True)


def _run_check(snippet: str) -> str:
    """Run a Python snippet with PACKAGES_DIR injected into sys.path."""
    inject = (
        f"import sys; sys.path.insert(0, {str(PACKAGES_DIR)!r}); "
        + snippet
    )
    result = subprocess.run(
        [sys.executable, "-c", inject],
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


# ── Detection ─────────────────────────────────────────────────────────────────

def is_cuda_available() -> bool:
    out = _run_check(
        "import onnxruntime as ort; "
        "print('CUDAExecutionProvider' in ort.get_available_providers())"
    )
    return out == "True"


def is_tensorrt_available() -> bool:
    out = _run_check(
        "import onnxruntime as ort; "
        "print('TensorrtExecutionProvider' in ort.get_available_providers())"
    )
    return out == "True"


def is_tensorrt_importable() -> bool:
    out = _run_check("import tensorrt; print(tensorrt.__version__)")
    return bool(out)


# ── Installation ──────────────────────────────────────────────────────────────

def _pip(packages: list, upgrade: bool = True) -> None:
    cmd = [
        sys.executable, "-m", "pip", "install",
        "--target", str(PACKAGES_DIR),
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
    log("Installing TensorRT Python bindings (tensorrt-cu12 < 11)...")
    _pip(TENSORRT_PACKAGES, upgrade=False)


# ── Verification ──────────────────────────────────────────────────────────────

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
        warn("  1. CUDA 12.x toolkit is not installed (driver >= 525.x required)")
        warn("  2. GPU does not support TensorRT (requires Volta / Turing / Ampere / Ada+)")
        warn("  3. Network error downloading from pypi.nvidia.com")
        warn("  4. Axiom has not been restarted since installation")
    else:
        log("All TensorRT components installed successfully.")


def print_next_steps() -> None:
    log("")
    log("=== Next Steps ===")
    log(f"1. Packages installed to: {PACKAGES_DIR}")
    log("2. Restart Axiom — session_utils.py will add the above path to sys.path")
    log("3. Set inference_backend = 'cuda' in the app (enables TRT > CUDA fallback)")
    log("4. On first inference the TRT engine is built — allow 1-5 minutes")
    log("5. Subsequent runs load the cached engine from trt_cache/ instantly")
    log("")


# ── Entry point ───────────────────────────────────────────────────────────────

def ensure_packages_dir() -> None:
    try:
        PACKAGES_DIR.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        fail(f"Cannot create packages directory {PACKAGES_DIR}: {e}")
    log(f"Install target  : {PACKAGES_DIR}")


def main() -> None:
    ensure_packages_dir()
    log(f"Python          : {sys.executable}")
    log("")

    if is_tensorrt_available():
        log("TensorrtExecutionProvider already available — nothing to do.")
        print_next_steps()
        return

    if not is_cuda_available():
        log("onnxruntime-gpu not detected. Installing CUDA packages first...")
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
