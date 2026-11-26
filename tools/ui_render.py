"""Render / stress the real Qt window without a display (Linux, macOS or CI).

    python tools/ui_render.py shots OUTDIR [WxH,WxH]   # every page, dark + light, PNGs in OUTDIR
    python tools/ui_render.py stress                   # rapid nav, theme/accent/dock churn, leak + paint checks

Needs a Python 3.12 environment with ``PyQt6==6.10.0 numpy pyserial`` (3.13 segfaults in sip on the
language manager singleton). Qt's offscreen platform never calls EGL, but libQt6Gui still links it: on a
machine without libEGL put a stub ``libEGL.so.1`` (empty functions for the ``egl*`` symbols Qt imports)
on ``LD_LIBRARY_PATH``. Windows-only modules (win32*, onnxruntime, cv2, ...) are replaced by stubs and
the ``LanguageManager`` singleton ``__new__`` is bypassed. DWM acrylic, native snap layouts and Segoe
fonts are Windows-only and cannot be checked this way.
"""
from __future__ import annotations

import ctypes
import importlib.abc
import importlib.machinery
import os
import statistics
import sys
import tempfile
import time
import types
from unittest.mock import MagicMock, patch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("AXIOM_USER_DIR", tempfile.mkdtemp(prefix="axiom_ui_"))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "src", "python", "dependencies")]
if not hasattr(ctypes, "WinDLL"):
    ctypes.WinDLL = MagicMock(name="WinDLL")
    ctypes.windll = MagicMock(name="windll")

_STUBBED = ("win32api", "win32con", "win32gui", "win32process", "win32file", "pywintypes", "winreg",
            "onnxruntime", "cv2", "mss", "dxcam", "vgamepad", "cyndilib")


class _Stub(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        value = MagicMock()
        setattr(self, name, value)
        return value


class _Finder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in _STUBBED:
            return importlib.machinery.ModuleSpec(name, self, is_package=True)
        return None

    def create_module(self, spec):
        module = _Stub(spec.name)
        module.__path__ = []
        return module

    def exec_module(self, module):
        pass


sys.meta_path.append(_Finder())

PAGES = ["modelInterface", "captureInterface", "inferenceInterface", "aimInterface", "triggerInterface",
         "keysInterface", "displayInterface", "configInterface", "convertInterface", "otherInterface"]


def make_window():
    from PyQt6.QtCore import QObject
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication(sys.argv)
    from gui.fluent_app import language_manager as lm
    lm.LanguageManager._initialized = False
    lm.LanguageManager.__new__ = lambda cls: QObject.__new__(cls)
    with patch("core.config._get_screen_size", return_value=(1920, 1080)):
        from core.config import Config
        cfg = Config()
    cfg.dark_mode = True
    cfg.preset_slots = ["Fast", "", "Precise", "", ""]
    cfg.active_preset_name = "Precise"
    from gui.fluent_app.window import AxiomWindow
    win = AxiomWindow()
    win.setConfig(cfg)
    win.show()
    return app, cfg, win


def pump(app, ms=450):
    end = time.time() + ms / 1000
    while time.time() < end:
        app.processEvents()
        time.sleep(0.005)


def cmd_shots(out: str, sizes_arg: str = "1280x820,900x600") -> None:
    from PyQt6.QtGui import QColor
    os.makedirs(out, exist_ok=True)
    sizes = [tuple(int(v) for v in s.split("x")) for s in sizes_arg.split(",")]
    app, _cfg, win = make_window()
    pump(app, 300)
    for dark in (True, False):
        if not dark:
            win.toggleTheme()
            pump(app, 500)
        win.setCustomBackgroundColor(QColor("#F5F5F5"), QColor("#1A1A1A"))     # no acrylic off Windows
        for w, h in sizes:
            win.resize(w, h)
            pump(app, 300)
            for name in PAGES:
                win.switchTo(getattr(win, name))
                pump(app, 450)
                win.grab().save(os.path.join(out, f"{name.replace('Interface', '')}_{'dark' if dark else 'light'}_{w}x{h}.png"))
    print("wrote", len(os.listdir(out)), "images to", out)


def cmd_stress() -> None:
    from PyQt6.QtCore import QPoint, Qt
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QWidget
    app, cfg, win = make_window()
    pump(app, 400)
    nav = win.floatingNav
    routes = [e.route for e in nav._entries]
    widgets_before = len(win.findChildren(QWidget))
    t0 = time.perf_counter()
    for i in range(200):
        rect = nav._layout()[0][(i * 7) % len(routes)]
        QTest.mouseClick(nav, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
                         QPoint(int(rect.center().x()), int(rect.center().y())))
        app.processEvents()
    print(f"200 rapid nav clicks: {time.perf_counter() - t0:.2f}s")
    pump(app, 500)
    assert win.stackedWidget.currentWidget().objectName() == nav.currentRoute()
    for _ in range(20):
        win.toggleTheme()
        app.processEvents()
    for hx in ["#e05fb0", "#4f8cff", "#2fb8a6", "#a662aa", "#7c6cf0"] * 4:
        win.setAccentColor(hx)
        app.processEvents()
    for i in range(20):
        win.setNavDock("left" if i % 2 == 0 else "top")
        app.processEvents()
    win.setNavDock("top")
    for width in list(range(900, 1700, 50)) + list(range(1700, 900, -50)):
        win.resize(width, 700)
        app.processEvents()
    win.setAnimationsEnabled(False)
    for i in range(30):
        nav.setCurrent(routes[i % len(routes)], emit=True)
        app.processEvents()
    assert not any(t.running for t in nav._expand) and not nav._cap.running
    win.setAnimationsEnabled(True)
    pump(app, 800)
    win.switchTo(win.aimInterface)
    pump(app, 500)
    slider = win.aimInterface.pidPxCard.slider
    times = []
    for v in list(range(0, 101, 2)) * 5:
        t = time.perf_counter()
        slider.setValue(v)
        app.processEvents()
        times.append((time.perf_counter() - t) * 1000)
    print(f"slider update ms: median {statistics.median(times):.2f} max {max(times):.2f}")
    leaked = [type(w).__name__ for w in win.findChildren(QWidget) if w.graphicsEffect() is not None]
    assert not leaked, f"graphics effects left attached: {leaked}"
    assert len(win.findChildren(QWidget)) == widgets_before, "widgets were recreated"
    print("stress OK; widgets", widgets_before, "accent", cfg.accent_color, "dock", cfg.nav_dock)


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "shots":
        cmd_shots(*sys.argv[2:4])
    elif len(sys.argv) >= 2 and sys.argv[1] == "stress":
        cmd_stress()
    else:
        print(__doc__)
        sys.exit(2)
    sys.stdout.flush()
    os._exit(0)          # the app's background threads (update check, terminal) would keep the process alive
