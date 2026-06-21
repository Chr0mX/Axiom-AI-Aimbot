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

# PaddlePaddle 3.x + PaddleOCR 2.9+ are co-designed — no pinning needed.
# The earlier set_optimization_level error was a 2.x/3.x mismatch; using both
# at current versions avoids it entirely.
#
# Change PADDLE_CUDA_TAG to match your CUDA version:
#   cu130 → CUDA 13.0   cu120 → CUDA 12.0   cu118 → CUDA 11.8
# For CPU-only set PADDLE_PACKAGE to 'paddlepaddle' and PADDLE_INDEX to "".
PADDLE_CUDA_TAG = "cu120"
PADDLE_INDEX    = f"https://www.paddlepaddle.org.cn/packages/stable/{PADDLE_CUDA_TAG}/"
PADDLE_PACKAGE  = "paddlepaddle-gpu==3.3.0"
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
    cmd = [
        sys.executable, "-m", "pip", "install",
        "--target", str(PACKAGES_DIR),
    ]
    if upgrade:
        cmd.append("--upgrade")
    if extra_index:
        cmd += ["--extra-index-url", extra_index]
    cmd.extend(packages)
    run(cmd)


def install_paddleocr() -> None:
    log(f"Installing PaddlePaddle GPU ({PADDLE_PACKAGE}) from Paddle index ({PADDLE_INDEX})...")
    _pip([PADDLE_PACKAGE], extra_index=PADDLE_INDEX)
    log(f"Installing: {', '.join(PADDLEOCR_PACKAGES)}")
    _pip(PADDLEOCR_PACKAGES)


# ── Verification ──────────────────────────────────────────────────────────────

def verify_installation() -> None:
    log("Verifying installation...")
    checks = [
        ("paddle importable",    is_paddle_importable, True),
        ("paddle GPU support",   is_paddle_gpu,        False),
        ("paddleocr importable", is_paddleocr_importable, True),
    ]
    all_ok = True
    for name, fn, required in checks:
        ok = fn()
        if ok:
            log(f"  [OK]      {name}")
        elif required:
            log(f"  [MISSING] {name}")
            all_ok = False
        else:
            log(f"  [WARN]    {name} — GPU not available, OCR will run on CPU")

    if not all_ok:
        warn("")
        warn("One or more required components are missing. Common causes:")
        warn("  1. CUDA 12.x toolkit not installed (driver >= 525.x required for GPU)")
        warn("  2. Network error downloading packages")
        warn("  3. Axiom has not been restarted since installation")
        warn("")
        warn("  For CPU-only install, set PADDLE_PACKAGE='paddlepaddle==2.6.2' and PADDLE_INDEX='' in this script")
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
        need_reinstall = not paddle_ver.startswith("3.")
        if not need_reinstall:
            log("PaddleOCR is already installed at the correct versions.")
            log(f"  paddle version    : {paddle_ver}")
            log(f"  paddleocr version : {ocr_ver}")
            log(f"  GPU support       : {_run_check('import paddle; print(paddle.is_compiled_with_cuda())')}")
            print_next_steps()
            return
        warn(f"Installed versions (paddle {paddle_ver}, paddleocr {ocr_ver}) are incompatible.")
        warn("Reinstalling pinned compatible versions...")

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
