# tests/conftest.py
"""
Pytest 全域配置 - 確保 src 目錄在 Python 路徑中
"""

import sys
import os

# 確保 src 目錄在路徑中，使所有測試都能正確 import
src_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)


import pytest


@pytest.fixture(autouse=True)
def _isolated_cwd(tmp_path, monkeypatch):
    """Run every test from its own empty temp directory.

    ``core.config.save_config()``/``load_config()`` default to the relative
    path ``"config.json"`` (and ``state.json``/``presets/``/``configs/`` are
    CWD-relative too), so a test that reaches the real defaults used to read
    and *write* the developer's actual files in the directory pytest was
    started from. Concretely: ``tests/test_app_controller.py`` truncated a
    real ``config.json`` to 0 bytes on every run. ``src/`` is put on
    ``sys.path`` by absolute path above, so nothing here depends on the CWD.
    """
    monkeypatch.chdir(tmp_path)


# ---------------------------------------------------------------------------
# Windows-only dependency handling
# ---------------------------------------------------------------------------
# This is a Windows app: several modules need pywin32, PyQt6, the Windows
# ctypes API or the onnxruntime native extension. On a non-Windows machine
# (CI on Linux, a sandbox) a test that reaches one of those fails with an
# environment error that says nothing about the code. Report exactly those
# narrow errors as skips instead, so a Linux run's failures are real ones.
# Nothing changes on Windows, and any other exception type or message still
# fails normally.
_WINDOWS_ONLY_MODULES = (
    "win32", "pywintypes", "pythoncom", "winerror", "PyQt6", "qfluentwidgets",
    "onnxruntime", "vgamepad", "dxcam", "cv2",
)


def _is_windows_only_env_error(excinfo):
    exc = excinfo.value
    if isinstance(exc, ModuleNotFoundError):
        name = getattr(exc, "name", "") or ""
        return name.split(".")[0].startswith(_WINDOWS_ONLY_MODULES)
    if isinstance(exc, AttributeError):
        return "windll" in str(exc) and "ctypes" in str(exc)
    return False


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    if sys.platform == "win32":
        return
    report = outcome.get_result()
    if call.when == "call" and report.failed and call.excinfo is not None:
        if _is_windows_only_env_error(call.excinfo):
            report.outcome = "skipped"
            report.longrepr = (
                str(item.fspath), item.location[1] or 0,
                "Skipped: requires Windows-only dependency "
                f"({call.excinfo.value})",
            )
