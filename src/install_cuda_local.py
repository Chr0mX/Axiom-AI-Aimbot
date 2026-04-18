import os
import re
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

PYTHON_DIR = Path("./python")
PYTHON_EXE = PYTHON_DIR / "python.exe"
SITE_PACKAGES = PYTHON_DIR / "Lib" / "site-packages"
DOWNLOAD_DIR = Path("./win_utils")

CUDA13_URL = "https://developer.download.nvidia.com/compute/cuda/13.2.1/local_installers/cuda_13.2.1_windows.exe"
CUDA12_URL = "https://developer.download.nvidia.com/compute/cuda/12.0.0/local_installers/cuda_12.0.0_527.41_windows.exe"

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
    print(f"[ERROR] {msg}")
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
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)


def detect_cuda_from_nvidia_smi() -> str:
    smi = shutil.which("nvidia-smi")
    if not smi:
        fail("nvidia-smi not found in PATH.")

    result = run_capture([smi])
    output = (result.stdout or "") + "\n" + (result.stderr or "")
    if result.returncode != 0:
        fail(f"nvidia-smi failed with exit code {result.returncode}.\n{output}")

    match = re.search(r"CUDA Version\s*:\s*(\d+)(?:\.(\d+))?", output)
    if not match:
        match = re.search(r"CUDA Version:\s*(\d+)(?:\.(\d+))?", output)
    if not match:
        fail("Could not parse CUDA Version from nvidia-smi output.")

    major = match.group(1)
    log(f"nvidia-smi reports CUDA Version: {match.group(0).split(':',1)[1].strip()}")

    if major not in {"12", "13"}:
        fail(f"Unsupported CUDA major version from nvidia-smi: {major}. Expected 12 or 13.")

    return major


CUDA_PATH_PATTERNS = {
    "12": [
        Path(r"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.0"),
        Path(r"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1"),
        Path(r"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.2"),
        Path(r"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.3"),
        Path(r"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4"),
        Path(r"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.5"),
        Path(r"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6"),
        Path(r"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.7"),
        Path(r"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8"),
        Path(r"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9"),
    ],
    "13": [
        Path(r"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0"),
        Path(r"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1"),
        Path(r"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2"),
    ],
}


def toolkit_installed(cuda_major: str) -> bool:
    for path in CUDA_PATH_PATTERNS[cuda_major]:
        nvcc = path / "bin" / "nvcc.exe"
        cudart = path / "bin" / "cudart64*.dll"
        if path.exists() and nvcc.exists():
            log(f"Detected CUDA Toolkit {cuda_major} at: {path}")
            return True

    env_path = os.environ.get("CUDA_PATH")
    if env_path:
        p = Path(env_path)
        if p.exists() and p.name.lower().startswith(f"v{cuda_major}."):
            nvcc = p / "bin" / "nvcc.exe"
            if nvcc.exists():
                log(f"Detected CUDA Toolkit {cuda_major} from CUDA_PATH: {p}")
                return True

    return False


def download_file(url: str, dest: Path) -> None:
    log(f"Downloading: {url}")
    log(f"Saving to: {dest}")

    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0",
                "Accept": "*/*",
            },
        )
        with urllib.request.urlopen(req) as response, open(dest, "wb") as f:
            shutil.copyfileobj(response, f)
        return
    except Exception as e:
        warn(f"Python download failed: {e}. Trying PowerShell fallback...")

    result = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            f"Invoke-WebRequest -Uri '{url}' -OutFile '{dest}'"
        ],
        text=True,
    )
    if result.returncode != 0:
        fail(f"Failed to download file from {url}")


def install_cuda_toolkit(cuda_major: str) -> None:
    if cuda_major == "13":
        url = CUDA13_URL
        filename = "cuda_13.2.1_windows.exe"
    else:
        url = CUDA12_URL
        filename = "cuda_12.0.0_windows.exe"

    installer = DOWNLOAD_DIR / filename
    if not installer.exists():
        download_file(url, installer)
    else:
        log(f"Installer already exists: {installer}")

    # NVIDIA documents -s for silent install on Windows.
    result = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            f'Start-Process -FilePath "{installer}" -ArgumentList \'-s\' -Verb RunAs -Wait'
        ],
        text=True,
    )
    if result.returncode != 0:
        fail(f"CUDA installer exited with code {result.returncode}")


def pip_install(args):
    cmd = [str(PYTHON_EXE), "-m", "pip", "install", "--target", str(SITE_PACKAGES)] + args
    run(cmd)


def install_python_packages(cuda_major: str) -> None:
    pip_install(COMMON_DEPS)

    if cuda_major == "13":
        pip_install([
            "--pre",
            "--index-url",
            "https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ort-cuda-13-nightly/pypi/simple/",
            "onnxruntime-gpu",
            "--no-deps",
        ])
        pip_install(["nvidia-cudnn-cu13"])
    else:
        pip_install([
            "--pre",
            "--index-url",
            "https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple/",
            "onnxruntime-gpu",
            "--no-deps",
        ])
        pip_install(["nvidia-cudnn-cu12"])


def main() -> None:
    ensure_paths()

    cuda_major = detect_cuda_from_nvidia_smi()
    log(f"Selected CUDA major version: {cuda_major}")

    if toolkit_installed(cuda_major):
        log(f"CUDA Toolkit {cuda_major} is already installed.")
    else:
        warn(f"CUDA Toolkit {cuda_major} not detected. Installing silently...")
        install_cuda_toolkit(cuda_major)
        if toolkit_installed(cuda_major):
            log(f"CUDA Toolkit {cuda_major} installation completed.")
        else:
            warn(
                f"CUDA Toolkit {cuda_major} installer finished, but the toolkit path was not detected yet. "
                "A reboot or fresh shell may be needed before it becomes visible."
            )

    install_python_packages(cuda_major)
    log("All requested installs completed.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("[ERROR] Interrupted by user.")
        pause_exit()
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        pause_exit()
        sys.exit(1)
