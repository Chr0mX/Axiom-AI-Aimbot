from __future__ import annotations

import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

NDI_RUNTIME_URL = "https://downloads.ndi.tv/SDK/NDI_SDK/NDI%206%20Runtime.exe"
BASE_DIR = Path(__file__).resolve().parent
LOCAL_PYTHON_DIR = BASE_DIR / "python"
LOCAL_PYTHON_EXE = LOCAL_PYTHON_DIR / "python.exe"
INSTALL_TARGET = LOCAL_PYTHON_DIR / "Lib" / "site-packages"
DOWNLOAD_DIR = BASE_DIR / "win_utils"
NDI_INSTALLER_PATH = DOWNLOAD_DIR / "NDI 6 Runtime.exe"


def log(message: str) -> None:
    print(f"[INFO] {message}")


def warn(message: str) -> None:
    print(f"[WARN] {message}")


def fail(message: str, code: int = 1) -> None:
    print(f"[ERROR] {message}")
    sys.exit(code)


def run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    log("Running: " + " ".join(f'"{c}"' if " " in c else c for c in cmd))
    return subprocess.run(cmd, check=check, text=True)


def ensure_paths() -> None:
    if not LOCAL_PYTHON_EXE.exists():
        fail(f"Local Python not found: {LOCAL_PYTHON_EXE}")
    INSTALL_TARGET.mkdir(parents=True, exist_ok=True)
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)


def package_installed(package_name: str) -> bool:
    package_dir = INSTALL_TARGET / package_name
    if package_dir.exists():
        return True

    dist_infos = list(INSTALL_TARGET.glob(f"{package_name.replace('-', '_')}*.dist-info"))
    return bool(dist_infos)


def is_cyndilib_installed() -> bool:
    if package_installed("cyndilib"):
        return True

    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(INSTALL_TARGET) + (os.pathsep + existing_pythonpath if existing_pythonpath else "")

    check_cmd = [
        str(LOCAL_PYTHON_EXE),
        "-c",
        "import cyndilib; print(getattr(cyndilib, '__file__', 'ok'))",
    ]
    result = subprocess.run(check_cmd, text=True, capture_output=True, env=env)
    return result.returncode == 0


def find_ndi_runtime_dll() -> Path | None:
    candidates = [
        Path(r"C:\Program Files\NDI\NDI 6 Runtime\Processing.NDI.Lib.x64.dll"),
        Path(r"C:\Program Files\NDI\NDI 6 Runtime\Bin\x64\Processing.NDI.Lib.x64.dll"),
        Path(r"C:\Program Files\NDI\Runtime\v6\Processing.NDI.Lib.x64.dll"),
        Path(r"C:\Windows\System32\Processing.NDI.Lib.x64.dll"),
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def is_ndi_runtime_installed() -> bool:
    return find_ndi_runtime_dll() is not None


def download_file(url: str, destination: Path) -> None:
    log(f"Downloading: {url}")
    with urllib.request.urlopen(url) as response, open(destination, "wb") as f:
        shutil.copyfileobj(response, f)
    log(f"Saved to: {destination}")


def install_cyndilib() -> None:
    if is_cyndilib_installed():
        log("cyndilib is already installed.")
        return

    log("cyndilib not found. Installing...")
    run([
        str(LOCAL_PYTHON_EXE),
        "-m",
        "pip",
        "install",
        "--target",
        str(INSTALL_TARGET),
        "cyndilib",
    ])

    if not is_cyndilib_installed():
        fail("cyndilib installation finished, but the package still could not be detected.")

    log("cyndilib installed successfully.")


def install_ndi_runtime() -> None:
    if is_ndi_runtime_installed():
        log("NDI Runtime is already installed.")
        return

    if not NDI_INSTALLER_PATH.exists():
        download_file(NDI_RUNTIME_URL, NDI_INSTALLER_PATH)

    log("NDI Runtime not found. Installing silently...")
    result = subprocess.run([str(NDI_INSTALLER_PATH), "/verysilent"], text=True)
    if result.returncode != 0:
        fail(f"NDI Runtime installer exited with code {result.returncode}")

    if not is_ndi_runtime_installed():
        warn("Installer completed, but the NDI runtime DLL was not found in the expected locations.")
    else:
        log("NDI Runtime installed successfully.")


def main() -> None:
    ensure_paths()
    install_cyndilib()
    install_ndi_runtime()
    log("Done.")


if __name__ == "__main__":
    main()
