"""Console Quick Edit pause must not freeze the process.

Loaded straight from win_utils/console.py so collection does not import
win_utils/__init__.py (that module imports win32api at import time).
"""

import importlib.util
import os
import sys
import threading
import time

import pytest


def _load_console():
    src = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "src", "win_utils", "console.py")
    spec = importlib.util.spec_from_file_location("console_safety_under_test", src)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


console = _load_console()


class TestQuickEditMode:
    def test_clears_quick_edit_and_mouse_input_and_sets_extended_flags(self):
        processed = 0x0001
        line = 0x0002
        echo = 0x0004
        mouse = console.ENABLE_MOUSE_INPUT
        insert = 0x0020
        quick = console.ENABLE_QUICK_EDIT_MODE
        extended = console.ENABLE_EXTENDED_FLAGS
        mode = processed | line | echo | mouse | insert | quick | extended
        updated = console.console_mode_without_selection_pause(mode)
        assert updated & quick == 0
        assert updated & mouse == 0
        assert updated & extended
        assert updated & insert
        assert updated & processed
        assert updated & line
        assert updated & echo

    def test_disable_writes_the_cleared_mode(self):
        seen = {}

        def get_mode():
            return 0x00F7  # default conhost input mode, Quick Edit on

        def set_mode(mode):
            seen["mode"] = mode
            return True

        assert console.disable_console_quick_edit(get_mode, set_mode) is True
        assert seen["mode"] & console.ENABLE_QUICK_EDIT_MODE == 0
        assert seen["mode"] & console.ENABLE_MOUSE_INPUT == 0
        assert seen["mode"] & console.ENABLE_EXTENDED_FLAGS

    def test_disable_skips_set_when_already_clear(self):
        def get_mode():
            return console.ENABLE_EXTENDED_FLAGS

        def set_mode(mode):
            raise AssertionError("SetConsoleMode should not be called")

        assert console.disable_console_quick_edit(get_mode, set_mode) is True

    def test_disable_false_without_a_console(self):
        assert console.disable_console_quick_edit(lambda: None, lambda mode: True) is False

    def test_disable_false_when_set_fails(self):
        assert console.disable_console_quick_edit(lambda: 0x40, lambda mode: False) is False

    def test_disable_is_noop_off_windows(self):
        if sys.platform == "win32":
            pytest.skip("live console mode is only touched on Windows")
        assert console.disable_console_quick_edit() is False


class _Stall:
    def __init__(self):
        self.release = threading.Event()
        self.entered = threading.Event()
        self.received = []

    def write(self, s):
        self.entered.set()
        assert self.release.wait(2)
        self.received.append(s)
        return len(s)

    def flush(self):
        return None

    def isatty(self):
        return True


class TestAsyncConsoleStream:
    def test_write_returns_while_console_write_is_blocked(self):
        stall = _Stall()
        stream = console._AsyncTextStream(stall, max_chunks=8)
        try:
            started = time.perf_counter()
            assert stream.write("hello\n") == 6
            assert time.perf_counter() - started < 0.3
            assert stall.entered.wait(1)
        finally:
            stall.release.set()
            stream.close()

    def test_paused_console_does_not_hold_the_logging_lock(self):
        """StreamHandler holds its lock across write()+flush().

        A paused console used to keep that lock, so the next log line on the
        UI thread blocked until the selection ended. The async stream must
        leave the caller's critical section before the real write.
        """
        stall = _Stall()
        stream = console._AsyncTextStream(stall, max_chunks=8)
        log_lock = threading.Lock()

        def emit(text):
            with log_lock:
                stream.write(text)
                stream.flush()

        worker = threading.Thread(target=emit, args=("from-worker\n",))
        try:
            worker.start()
            assert stall.entered.wait(1)
            started = time.perf_counter()
            emit("from-ui\n")
            assert time.perf_counter() - started < 0.3
        finally:
            stall.release.set()
            worker.join(2)
            stream.close()
        assert "from-ui" in "".join(stall.received)

    def test_full_queue_drops_oldest_and_reports_it(self):
        stall = _Stall()
        stream = console._AsyncTextStream(stall, max_chunks=1)
        try:
            stream.write("first")
            assert stall.entered.wait(1)
            stream.write("second")
            stream.write("third")
        finally:
            stall.release.set()
            stream.close()
        blob = "".join(stall.received)
        assert "first" in blob
        assert "second" not in blob
        assert "third" in blob
        assert "dropped" in blob

    def test_chunks_are_delivered_in_order(self):
        received = []

        class Sink:
            def write(self, s):
                received.append(s)
                return len(s)

            def flush(self):
                return None

        stream = console._AsyncTextStream(Sink(), max_chunks=8)
        stream.write("abc")
        stream.write("def")
        stream.close()
        assert "".join(received) == "abcdef"

    def test_install_wraps_console_streams_once(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setattr(sys, "stdout", _Stall())
        monkeypatch.setattr(sys, "stderr", _Stall())
        assert console.install_nonblocking_stdio() is True
        stdout = sys.stdout
        stderr = sys.stderr
        try:
            assert stdout._axiom_async_stdio
            assert stderr._axiom_async_stdio
            assert console.install_nonblocking_stdio() is True
            assert sys.stdout is stdout
            assert sys.stderr is stderr
        finally:
            stdout._underlying.release.set()
            stderr._underlying.release.set()
            stdout.close()
            stderr.close()

    def test_install_leaves_redirected_streams_alone(self, monkeypatch):
        class Pipe:
            def isatty(self):
                return False

        monkeypatch.setattr(sys, "platform", "win32")
        out, err = Pipe(), Pipe()
        monkeypatch.setattr(sys, "stdout", out)
        monkeypatch.setattr(sys, "stderr", err)
        assert console.install_nonblocking_stdio() is False
        assert sys.stdout is out
        assert sys.stderr is err

    def test_install_is_noop_off_windows(self):
        if sys.platform == "win32":
            pytest.skip("install is live on Windows")
        stdout = sys.stdout
        assert console.install_nonblocking_stdio() is False
        assert sys.stdout is stdout
