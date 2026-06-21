"""Install PaddleOCR and PaddlePaddle 3.x (GPU) into the shared AxiomAI packages dir.

Packages are written to:
    %LOCALAPPDATA%\\AxiomAI\\site-packages

This location survives app reinstalls and is isolated from other Python
environments on the machine.  Axiom picks the packages up at startup via
sys.path injection in session_utils.py.

Usage (from project root, any Python >= 3.10):
    python src\\install_paddleocr_local.py
    -- or --
    src\\python\\python.exe src\\install_paddleocr_local.py

Default: CUDA 13.0 + paddlepaddle-gpu 3.3.0
To change CUDA version, edit PADDLE_CUDA_TAG and PADDLE_PACKAGE below.
CPU fallback: set PADDLE_PACKAGE = 'paddlepaddle' and PADDLE_INDEX = ''.
"""

from __future__ import annotations

__version__ = "1.0"

import os
import subprocess
import sys
from pathlib import Path

# ── Install target ────────────────────────────────────────────────────────────
_LOCALAPPDATA = os.environ.get("LOCALAPPDATA", "")
if not _LOCALAPPDATA:
    print("[ERROR] LOCALAPPDATA environment variable is not set.", file=sys.stderr)
    sys.exit(1)

PACKAGES_DIR = Path(_LOCALAPPDATA) / "AxiomAI" / "site-packages"

# paddlepaddle 3.3.0 CPU (PyPI) + latest paddleocr.
# CPU build has no CUDA dependency and is numpy-2.x compatible.
# OneDNN is disabled at runtime via paddle.set_flags (env vars are ignored
# by the PIR executor in 3.x).
# Do NOT use paddlepaddle==2.6.2: it requires numpy<2 and will downgrade
# numpy in the shared site-packages dir, breaking cv2 which needs numpy 2.x.
PADDLE_INDEX    = ""
PADDLE_PACKAGE  = "paddlepaddle==3.3.0"
PADDLEOCR_PACKAGES = [
    "paddleocr",
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


def _build_inject(snippet: str) -> str:
    return f"import sys; sys.path.insert(0, {str(PACKAGES_DIR)!r}); " + snippet


def _run_check(snippet: str) -> str:
    result = subprocess.run(
        [sys.executable, "-c", _build_inject(snippet)],
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


# ── Detection ─────────────────────────────────────────────────────────────────

def is_paddleocr_importable() -> bool:
    out = _run_check("import paddleocr; print(paddleocr.__version__)")
    return bool(out)


def is_paddle_importable() -> bool:
    out = _run_check("import paddle; print(paddle.__version__)")
    return bool(out)


def is_paddle_gpu() -> bool:
    out = _run_check("import paddle; print(paddle.is_compiled_with_cuda())")
    return out == "True"


# ── Installation ──────────────────────────────────────────────────────────────

def _pip(packages: list, upgrade: bool = True, extra_index: str = "") -> None:
    # PyPI is the primary index for fast dependency resolution.
    # Paddle's CN CDN is added as an extra index only when needed for the CUDA wheel.
    cmd = [
        sys.executable, "-m", "pip", "install",
        "--target", str(PACKAGES_DIR),
        "-i", "https://pypi.org/simple",
    ]
    if upgrade:
        cmd.append("--upgrade")
    if extra_index:
        cmd += ["--extra-index-url", extra_index]
    cmd.extend(packages)
    run(cmd)


def install_paddleocr() -> None:
    log(f"Installing PaddlePaddle CPU ({PADDLE_PACKAGE})...")
    log(f"  Index : https://pypi.org/simple")
    _pip([PADDLE_PACKAGE])
    log(f"Installing: {', '.join(PADDLEOCR_PACKAGES)}")
    _pip(PADDLEOCR_PACKAGES)


# ── Verification ──────────────────────────────────────────────────────────────

def verify_installation() -> None:
    log("Verifying installation...")
    checks = [
        ("paddle importable",    is_paddle_importable,    True),
        ("paddleocr importable", is_paddleocr_importable, True),
    ]
    all_ok = True
    for name, fn, required in checks:
        ok = fn()
        if ok:
            log(f"  [OK]      {name}")
        else:
            log(f"  [MISSING] {name}")
            all_ok = False

    if not all_ok:
        warn("")
        warn("One or more required components are missing. Common causes:")
        warn("  1. Network error downloading packages")
        warn("  2. Axiom has not been restarted since installation")
    else:
        log("PaddleOCR installation complete.")


def print_next_steps() -> None:
    log("")
    log("=== Next Steps ===")
    log(f"1. Packages installed to: {PACKAGES_DIR}")
    log("2. Restart Axiom")
    log("3. Inference page → General Parameters → enable 'Second Inference (OCR)'")
    log("4. Capture page → 'Inferred Text' section will show live OCR results")
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

    paddle_ver = _run_check("import paddle; print(paddle.__version__)")
    ocr_ver    = _run_check("import paddleocr; print(paddleocr.__version__)")
    if paddle_ver and ocr_ver:
        is_gpu_build = _run_check("import paddle; print(paddle.is_compiled_with_cuda())") == "True"
        is_wrong_ver = not paddle_ver.startswith("3.")
        need_reinstall = is_wrong_ver or is_gpu_build
        if not need_reinstall:
            log("PaddleOCR is already installed at the correct versions.")
            log(f"  paddle version    : {paddle_ver}")
            log(f"  paddleocr version : {ocr_ver}")
            print_next_steps()
            return
        if is_gpu_build:
            warn(f"Installed paddle {paddle_ver} is the GPU build — switching to CPU.")
        else:
            warn(f"Installed paddle {paddle_ver} is not 3.x — reinstalling.")
        warn("Reinstalling...")

    install_paddleocr()
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
