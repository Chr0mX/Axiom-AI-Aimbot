# tests/test_ddxoft_lock.py
"""Regression tests for DDXoftMouse's thread-safety fix.

DDXoftMouse had no lock at all, unlike every other mouse backend in this
codebase (MAKCU, Arduino, Xbox) — move_relative() (called from the inference
thread) and click_left() (called from the separate auto-fire thread) both
touched the same DLL handle and stats counters with no synchronization.

win_utils/__init__.py (and ddxoft_mouse.py's own `from .mouse_move import
send_mouse_move_mouse_event`) transitively needs win32api, which fails at
*collection* on non-Windows (see CLAUDE.md). Every test here stubs
sys.modules['win32api'] (and friends) before importing, matching the
pattern established in test_ai_aiming.py for the same class of issue.
"""

import sys
import threading
import time
import types

import pytest


@pytest.fixture
def ddxoft_mouse_cls(monkeypatch):
    for name in ("win32api", "win32con", "win32gui", "win32process", "win32event", "pywintypes"):
        stub = types.ModuleType(name)
        monkeypatch.setitem(sys.modules, name, stub)
    sys.modules["win32api"].GetCursorPos = lambda: (0, 0)
    sys.modules["win32api"].SetCursorPos = lambda pos: None
    sys.modules["win32api"].mouse_event = lambda *a, **k: None
    sys.modules["win32con"].MOUSEEVENTF_MOVE = 0x0001

    from win_utils.ddxoft_mouse import DDXoftMouse
    return DDXoftMouse


class _FakeDll:
    def DD_movR(self, dx, dy):
        return 1

    def DD_btn(self, code):
        return 1


def test_concurrent_move_relative_does_not_race_the_counter(ddxoft_mouse_cls):
    """500 concurrent move_relative() calls must all be counted — a missing
    lock would let concurrent `self.success_count += 1` read-modify-writes
    clobber each other and undercount."""
    m = ddxoft_mouse_cls()
    m.available = True
    m.dll = _FakeDll()

    n = 500
    threads = [threading.Thread(target=lambda: m.move_relative(1, 1)) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert m.success_count == n
    assert m.failure_count == 0


def test_click_left_does_not_hold_lock_across_its_sleep(ddxoft_mouse_cls):
    """click_left() sleeps 1ms between the down/up DD_btn calls — the lock
    must be released for that sleep, or a concurrent move_relative() call
    (inference thread) would stall for the full sleep duration on every
    single click, matching the no-lock-across-sleep discipline every other
    backend in this codebase already follows (see MAKCU)."""
    m = ddxoft_mouse_cls()
    m.available = True
    m.dll = _FakeDll()

    t_click = threading.Thread(target=m.click_left)
    t_click.start()
    time.sleep(0.0003)  # let click_left acquire+release its first lock, entering the sleep window

    t0 = time.perf_counter()
    ok = m.move_relative(2, 2)
    elapsed = time.perf_counter() - t0
    t_click.join()

    assert ok is True
    assert elapsed < 0.0008, f"move_relative() blocked for {elapsed * 1000:.3f}ms — lock held across the sleep"


def test_click_left_reports_success(ddxoft_mouse_cls):
    m = ddxoft_mouse_cls()
    m.available = True
    m.dll = _FakeDll()

    assert m.click_left() is True
    assert m.success_count == 1
    assert m.last_status == "CLICK_SUCCESS"
