"""
Bootstrap dependencies for bundled runtime.

Behavior:
1) Check if core runtime imports are available.
2) If missing, install default runtime requirements (DirectML path).
3) Pre-download dependency wheels for all backend types (CPU/DirectML/CUDA)
   into a local wheelhouse to reduce future setup time.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
WHEELHOUSE = ROOT / "src" / "python" / "wheelhouse"


def _run(cmd: list[str]) -> int:
    print(f"[Axiom][deps] $ {' '.join(cmd)}")
    return subprocess.call(cmd, cwd=str(ROOT))


def _ensure_pip() -> int:
    if _run([sys.executable, "-m", "pip", "--version"]) == 0:
        return 0
    print("[Axiom][deps] pip missing, running ensurepip...")
    code = _run([sys.executable, "-m", "ensurepip", "--upgrade"])
    if code != 0:
        return code
    return _run([sys.executable, "-m", "pip", "--version"])


def _is_module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _missing_core_modules() -> list[str]:
    modules = ["numpy", "cv2", "onnxruntime", "mss", "PyQt6", "win32api"]
    return [m for m in modules if not _is_module_available(m)]


def _install_default_runtime() -> int:
    req = ROOT / "requirements-directml.txt"
    return _run([sys.executable, "-m", "pip", "install", "-r", str(req)])


def _download_all_backend_wheels() -> int:
    WHEELHOUSE.mkdir(parents=True, exist_ok=True)
    req_files = [
        ROOT / "requirements-cpu.txt",
        ROOT / "requirements-directml.txt",
        ROOT / "requirements-cuda.txt",
    ]
    for req in req_files:
        code = _run([sys.executable, "-m", "pip", "download", "-r", str(req), "-d", str(WHEELHOUSE)])
        if code != 0:
            return code
    return 0


def main() -> int:
    if _ensure_pip() != 0:
        print("[Axiom][deps] Failed to initialize pip.")
        return 1

    missing = _missing_core_modules()
    if missing:
        print(f"[Axiom][deps] Missing modules: {', '.join(missing)}")
        print("[Axiom][deps] Installing default runtime dependencies (DirectML)...")
        if _install_default_runtime() != 0:
            return 1
    else:
        print("[Axiom][deps] Core dependencies already available.")

    print("[Axiom][deps] Downloading dependency wheels for CPU/DirectML/CUDA...")
    if _download_all_backend_wheels() != 0:
        print("[Axiom][deps] Wheel download failed.")
        return 1

    print("[Axiom][deps] Dependency bootstrap completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
