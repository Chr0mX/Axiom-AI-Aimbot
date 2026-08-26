"""Install FastAPI/uvicorn into the shared AxiomAI packages dir.

core/web_control_server.py (the Web Control API) needs fastapi + uvicorn,
neither of which ship in src/python/'s bundled dependency tree. Rather than
vendor multi-megabyte wheels into the git repo, this mirrors
install_tensorrt_local.py / install_paddleocr_local.py: packages are
installed once, on the user's own machine (which has real internet
access), to a location the app already knows to pick up.

Packages are written to:
    %LOCALAPPDATA%\\AxiomAI\\site-packages

This location survives app reinstalls, is isolated from other Python
environments on the machine, and is shared with the TensorRT/PaddleOCR
installers above. Axiom picks the packages up at startup via the sys.path
injection in session_utils.py's _inject_axiom_packages() — no other wiring
is needed once this script has run.

Usage (from project root, any Python >= 3.10):
    python src\\install_web_control_local.py
    -- or --
    src\\python\\python.exe src\\install_web_control_local.py
    -- or --
    "Install Web Control.bat"
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

# Plain uvicorn (not uvicorn[standard]) — no uvloop/httptools needed for a
# personal LAN control API; keeps the install smaller. fastapi pulls in
# starlette/pydantic/anyio automatically as ordinary pip dependencies.
PACKAGES = ["fastapi", "uvicorn"]


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

def is_fastapi_importable() -> bool:
    out = _run_check("import fastapi; print(fastapi.__version__)")
    return bool(out)


def is_uvicorn_importable() -> bool:
    out = _run_check("import uvicorn; print(uvicorn.__version__)")
    return bool(out)


# ── Installation ──────────────────────────────────────────────────────────────

def install_web_control_deps() -> None:
    log(f"Installing: {', '.join(PACKAGES)}")
    cmd = [
        sys.executable, "-m", "pip", "install",
        "--target", str(PACKAGES_DIR),
        "-i", "https://pypi.org/simple",
        "--upgrade",
    ]
    cmd.extend(PACKAGES)
    run(cmd)


# ── Verification ──────────────────────────────────────────────────────────────

def verify_installation() -> None:
    log("Verifying installation...")
    checks = [
        ("fastapi importable", is_fastapi_importable),
        ("uvicorn importable", is_uvicorn_importable),
    ]
    all_ok = True
    for name, fn in checks:
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
        log("Web Control dependency installation complete.")


def print_next_steps() -> None:
    log("")
    log("=== Next Steps ===")
    log(f"1. Packages installed to: {PACKAGES_DIR}")
    log("2. Restart Axiom")
    log("3. Other page -> Remote Control -> Enable Web Control")
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

    if is_fastapi_importable() and is_uvicorn_importable():
        log("fastapi/uvicorn are already installed.")
        print_next_steps()
        return

    install_web_control_deps()
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
