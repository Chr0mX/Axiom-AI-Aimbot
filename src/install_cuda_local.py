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



def pip_install(args):
    cmd = [str(PYTHON_EXE), "-m", "pip", "install", "--upgrade", "--target", str(SITE_PACKAGES)] + args
    run(cmd)


def install_python_packages() -> None:
    pip_install(COMMON_DEPS)

    pip_install(["onnxruntime-gpu[cuda,cudnn]"])


def main() -> None:
    ensure_paths()
    install_python_packages()
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