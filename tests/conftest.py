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
    # User files (config.json, state.json, presets/, ...) default to
    # <Axiom>/user/ via core/paths.py; point that at the temp dir too.
    monkeypatch.setenv("AXIOM_USER_DIR", str(tmp_path))


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
    # Not Windows-only, but optional runtime packages a bare dev checkout may lack
    # (requirements-dev.txt installs them for CI).
    "mss", "serial",
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
    # "setup" too: a fixture that imports a Windows-only module (or one that needs
    # OpenCV) fails there, before the test body runs.
    if call.when in ("setup", "call") and report.failed and call.excinfo is not None:
        if _is_windows_only_env_error(call.excinfo):
            report.outcome = "skipped"
            report.longrepr = (
                str(item.fspath), item.location[1] or 0,
                "Skipped: requires Windows-only dependency "
                f"({call.excinfo.value})",
            )


# ---------------------------------------------------------------------------
# Stubbing platform modules without polluting the rest of the session
# ---------------------------------------------------------------------------
# A few test modules need ``win32api``/``win32con`` (and, where OpenCV isn't
# installed, ``cv2``) to be importable. Doing that with a module-level
# ``sys.modules.setdefault(name, MagicMock())`` fakes the module for the whole
# remaining run: later files that rely on the real thing (or on conftest's
# "Windows-only dependency => skip" rule) then run against a MagicMock and fail
# for reasons unrelated to the code under test. Use the ``win32_stubs`` fixture
# instead -- it is scoped to one test module and undoes everything on exit.

# First-party modules that import win32api (directly or via ``win_utils``) at
# import time. They are dropped from ``sys.modules`` before and after a stubbed
# module so each is imported against the stubs it is meant to see, and no module
# bound to a stub is left behind for a later test file.
_WIN32_DEPENDENT_MODULES = (
    "win_utils",
    "core.ai_aiming",
    "core.ai_loop",
    "core.ai_loop_utils",
    "core.auto_fire",
    "core.key_listener",
    "core.app_controller",
)


def drop_modules_imported_since(before):
    """Forget every first-party module imported after the ``set(sys.modules)`` snapshot
    ``before``. A module first imported while a stub was installed keeps its reference to
    that stub for good (``core.inference`` bound to a MagicMock ``cv2`` fails every later
    test that uses it), so a fixture that stubs a dependency must drop what it pulled in."""
    for name in set(sys.modules) - set(before):
        if name.split(".")[0] in ("core", "win_utils", "gui"):
            del sys.modules[name]
            parent, _, child = name.rpartition(".")
            if parent in sys.modules and hasattr(sys.modules[parent], child):
                delattr(sys.modules[parent], child)


def purge_win32_dependent_modules():
    """Forget the first-party modules that bind win32api at import time."""
    for name in list(sys.modules):
        for prefix in _WIN32_DEPENDENT_MODULES:
            if name == prefix or name.startswith(prefix + "."):
                del sys.modules[name]
                parent, _, child = name.rpartition(".")
                if parent in sys.modules and hasattr(sys.modules[parent], child):
                    delattr(sys.modules[parent], child)
                break


@pytest.fixture(scope="module")
def win32_stubs():
    """MagicMock ``win32api``/``win32con`` (and ``cv2`` only if it isn't installed)
    for the duration of one test module. Yields the stub mapping."""
    from unittest.mock import MagicMock

    stubs = {"win32api": MagicMock(), "win32con": MagicMock()}
    try:
        import cv2  # noqa: F401  (real OpenCV present: leave it alone)
    except ImportError:
        stubs["cv2"] = MagicMock()

    purge_win32_dependent_modules()
    before = set(sys.modules)
    with pytest.MonkeyPatch.context() as mp:
        for name, stub in stubs.items():
            mp.setitem(sys.modules, name, stub)
        try:
            yield stubs
        finally:
            # Everything first imported against the stubs (core.inference bound to the
            # stubbed cv2, win_utils bound to the stubbed win32api, ...) goes too.
            drop_modules_imported_since(before)
            purge_win32_dependent_modules()
