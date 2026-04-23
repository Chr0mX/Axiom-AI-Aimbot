import os
import re
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
LOCAL_PYTHON_DIR = BASE_DIR / "python"
PYTHON_EXE = LOCAL_PYTHON_DIR / "python.exe"
SITE_PACKAGES = LOCAL_PYTHON_DIR / "Lib" / "site-packages"

COMMON_DEPS = [
    "coloredlogs",
    "flatbuffers",
    "numpy",
    "packaging",
    "protobuf",
    "sympy",
]


def log(msg: str) -> None:
    print(f"[INFO] {msg}")


def warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def pause_exit() -> None:
    try:
        input("Press Enter to exit...")
    except EOFError:
        pass


def fail(msg: str, code: int = 1) -> None:
    print(f"[ERROR] {msg}", file=sys.stderr)
    pause_exit()
    sys.exit(code)


def run(cmd, check=True):
    log("Running: " + " ".join(f'"{c}"' if " " in str(c) else str(c) for c in cmd))
    return subprocess.run(cmd, check=check)


def run_capture(cmd):
    log("Checking: " + " ".join(f'"{c}"' if " " in str(c) else str(c) for c in cmd))
    return subprocess.run(cmd, capture_output=True, text=True)


def ensure_paths() -> None:
    if not PYTHON_EXE.exists():
        fail(f"Local Python not found at: {PYTHON_EXE}")
    SITE_PACKAGES.mkdir(parents=True, exist_ok=True)



def is_cuda_installed() -> bool:
    """Return True if onnxruntime-gpu with CUDAExecutionProvider is already available."""
    result = subprocess.run(
        [
            str(PYTHON_EXE), "-c",
            "import onnxruntime as ort; print('CUDAExecutionProvider' in ort.get_available_providers())",
        ],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0 and result.stdout.strip() == "True"


def pip_install(args):
    cmd = [str(PYTHON_EXE), "-m", "pip", "install", "--upgrade", "--target", str(SITE_PACKAGES)] + args
    run(cmd)


def pip_install_fresh(args):
    """Install without --upgrade to avoid overwriting locked DLLs mid-session."""
    cmd = [str(PYTHON_EXE), "-m", "pip", "install", "--target", str(SITE_PACKAGES)] + args
    run(cmd)


def install_python_packages() -> None:
    pip_install(COMMON_DEPS)
    # Use pip_install_fresh (no --upgrade) so pip does not try to overwrite
    # already-loaded DLLs if somehow called mid-session.  Fresh installs will
    # still download the GPU wheel; only re-installs skip the overwrite.
    pip_install_fresh(["onnxruntime-gpu[cuda,cudnn]"])


def prompt_tensorrt() -> bool:
    """Ask the user whether to also install TensorRT.

    Returns True if the user answers yes.
    """
    try:
        answer = input(
            "\nDo you also want to install TensorRT for maximum GPU performance? (yes/no): "
        ).strip().lower()
        return answer in ("yes", "y")
    except EOFError:
        return False


def install_tensorrt() -> None:
    """Delegate to install_tensorrt_local.py for TRT packages."""
    trt_script = os.path.join(os.path.dirname(__file__), "install_tensorrt_local.py")
    if not os.path.exists(trt_script):
        warn("install_tensorrt_local.py not found — skipping TensorRT install.")
        return
    log("Running TensorRT installer...")
    import subprocess
    result = subprocess.run([str(PYTHON_EXE), trt_script], check=False)
    if result.returncode != 0:
        warn("TensorRT installer returned a non-zero exit code. Check output above.")
    else:
        log("TensorRT installation completed.")


def main() -> None:
    ensure_paths()

    if is_cuda_installed():
        log("CUDAExecutionProvider already available — skipping CUDA installation.")
    else:
        install_python_packages()
        log("CUDA packages installed.")

    # Always offer TensorRT after a successful (or pre-existing) CUDA install.
    if prompt_tensorrt():
        install_tensorrt()
    else:
        log("Skipping TensorRT installation.")

    log("All requested installs completed.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("[ERROR] Interrupted by user.", file=sys.stderr)
        pause_exit()
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}", file=sys.stderr)
        pause_exit()
        sys.exit(1)