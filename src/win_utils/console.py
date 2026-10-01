# console.py - Terminal Window Control Module
"""Windows console window control, and a guard against the console freezing the app.

Windows conhost (the console opened by ``python.exe`` / ``啟動Launcher.bat``)
ships with Quick Edit mode on. Clicking the console, or dragging a text
selection, puts it in selection mode. While that mode is active, conhost
stops draining the process's stdout/stderr and the next ``WriteConsole``
blocks the calling thread.

That stall takes the whole UI down with it:

- ``print()`` on the Qt thread blocks inside the write, so the event loop
  never returns.
- ``logging`` holds the ``StreamHandler`` lock across ``stream.write()``.
  One blocked log line on any worker thread makes the next log call on the
  UI thread wait on that same lock.

Two independent guards live here. ``disable_console_quick_edit()`` turns off
the click-to-select mode so an accidental click never enters the pause.
``install_nonblocking_stdio()`` moves the actual console write onto a daemon
thread, so a selection started from the console's system menu (which still
pauses ``WriteConsole``) only stalls that writer — callers enqueue and
return.
"""

import ctypes
import queue
import sys
import threading

ENABLE_MOUSE_INPUT = 0x0010
ENABLE_QUICK_EDIT_MODE = 0x0040
ENABLE_EXTENDED_FLAGS = 0x0080

# (DWORD)-10, the STD_INPUT_HANDLE argument to GetStdHandle.
_STD_INPUT_HANDLE = 0xFFFFFFF5

_STOP = object()


def console_mode_without_selection_pause(mode: int) -> int:
    """Input mode with Quick Edit and mouse-select cleared.

    ``ENABLE_EXTENDED_FLAGS`` has to be set in the same ``SetConsoleMode``
    call or Windows ignores the Quick Edit bit. Mouse input is cleared too:
    with Quick Edit off, mouse events would otherwise queue into the console
    input buffer, which this app never reads, and a full buffer stalls the
    console again.
    """
    mode |= ENABLE_EXTENDED_FLAGS
    mode &= ~ENABLE_QUICK_EDIT_MODE
    mode &= ~ENABLE_MOUSE_INPUT
    return mode


def _win32_console_mode_io():
    """Return ``(get_mode, set_mode)`` for the attached console's input mode."""
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetStdHandle.argtypes = (ctypes.c_uint32,)
    kernel32.GetStdHandle.restype = ctypes.c_void_p
    kernel32.GetConsoleMode.argtypes = (
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32))
    kernel32.GetConsoleMode.restype = ctypes.c_int
    kernel32.SetConsoleMode.argtypes = (ctypes.c_void_p, ctypes.c_uint32)
    kernel32.SetConsoleMode.restype = ctypes.c_int

    def _handle():
        handle = kernel32.GetStdHandle(_STD_INPUT_HANDLE)
        if not handle:
            return None
        invalid = ctypes.c_void_p(-1).value
        if invalid is not None and handle == invalid:
            return None
        return handle

    def get_mode():
        handle = _handle()
        if handle is None:
            return None
        mode = ctypes.c_uint32()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return None
        return int(mode.value)

    def set_mode(mode: int) -> bool:
        handle = _handle()
        if handle is None:
            return False
        return bool(kernel32.SetConsoleMode(handle, int(mode)))

    return get_mode, set_mode


def disable_console_quick_edit(get_mode=None, set_mode=None) -> bool:
    """Turn off Quick Edit so clicking the console does not pause the process.

    Returns True when the mode is already clear or was updated. Returns False
    when there is no console (pythonw, redirected stdin) or the call failed.
    """
    if get_mode is None or set_mode is None:
        if sys.platform != "win32":
            return False
        try:
            get_mode, set_mode = _win32_console_mode_io()
        except Exception:
            return False
    try:
        current = get_mode()
    except Exception:
        return False
    if current is None:
        return False
    updated = console_mode_without_selection_pause(int(current))
    if updated == int(current):
        return True
    try:
        return bool(set_mode(updated))
    except Exception:
        return False


class _AsyncTextStream:
    """Text stream whose ``write`` returns even if the real console is paused.

    Chunks sit in a bounded queue. The writer thread is the only one that
    touches the underlying console, so a blocked ``WriteConsole`` cannot hold
    the logging lock or the Qt thread. When the queue is full the oldest
    chunk is dropped so a long pause cannot grow memory without bound.
    """

    _axiom_async_stdio = True

    def __init__(self, underlying, max_chunks: int = 2048):
        self._underlying = underlying
        self._q: queue.Queue = queue.Queue(maxsize=max(1, int(max_chunks)))
        self._dropped = 0
        self._drop_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="console-writer", daemon=True)
        self._thread.start()

    def write(self, s) -> int:
        if not isinstance(s, str):
            s = str(s)
        if not s:
            return 0
        self._enqueue(s)
        return len(s)

    def writelines(self, lines) -> None:
        for line in lines:
            self.write(line)

    def flush(self) -> None:
        # Waiting here would re-introduce the stall this class exists to avoid.
        return None

    def isatty(self) -> bool:
        try:
            return bool(self._underlying.isatty())
        except Exception:
            return False

    def close(self, timeout: float = 1.0) -> None:
        """Stop the writer. Used by tests; production leaves the daemon running.

        ``_STOP`` is only a wake-up. If the queue is already full it is left
        untouched so the last real line is not discarded to make room.
        """
        self._stop.set()
        try:
            self._q.put_nowait(_STOP)
        except queue.Full:
            pass
        self._thread.join(timeout)

    def _enqueue(self, item) -> None:
        try:
            self._q.put_nowait(item)
            return
        except queue.Full:
            pass
        try:
            self._q.get_nowait()
        except queue.Empty:
            pass
        with self._drop_lock:
            self._dropped += 1
        try:
            self._q.put_nowait(item)
        except queue.Full:
            pass

    def _run(self) -> None:
        while True:
            try:
                item = self._q.get(timeout=0.25)
            except queue.Empty:
                if self._stop.is_set():
                    return
                continue
            if item is _STOP:
                return
            parts = [item]
            while True:
                try:
                    nxt = self._q.get_nowait()
                except queue.Empty:
                    break
                if nxt is _STOP:
                    break
                parts.append(nxt)
            self._write_parts(parts)
            if self._stop.is_set() and self._q.empty():
                return

    def _write_parts(self, parts) -> None:
        with self._drop_lock:
            dropped = self._dropped
            self._dropped = 0
        text = "".join(parts)
        if dropped:
            text = f"[console] dropped {dropped} lines while the console was paused\n{text}"
        try:
            self._underlying.write(text)
            flush = getattr(self._underlying, "flush", None)
            if callable(flush):
                flush()
        except Exception:
            pass

    def __getattr__(self, name):
        return getattr(self._underlying, name)


def _stream_is_console(stream) -> bool:
    isatty = getattr(stream, "isatty", None)
    if not callable(isatty):
        return False
    try:
        return bool(isatty())
    except Exception:
        return False


def install_nonblocking_stdio() -> bool:
    """Point ``sys.stdout`` / ``sys.stderr`` at async wrappers when they are consoles.

    Idempotent. A redirected pipe (no console) is left alone so log files stay
    synchronous and complete. Returns True when at least one stream is wrapped.
    """
    if sys.platform != "win32":
        return False
    wrapped = False
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        if getattr(stream, "_axiom_async_stdio", False):
            wrapped = True
            continue
        if not _stream_is_console(stream):
            continue
        setattr(sys, name, _AsyncTextStream(stream))
        wrapped = True
    return wrapped


def prepare_console() -> None:
    """Apply both console-freeze guards. Safe to call more than once."""
    try:
        disable_console_quick_edit()
    except Exception:
        pass
    try:
        install_nonblocking_stdio()
    except Exception:
        pass


def get_console_window():
    """Get the handle of the current console window"""
    try:
        kernel32 = ctypes.windll.kernel32
        return kernel32.GetConsoleWindow()
    except Exception as e:
        print(f"[Terminal Control] Failed to get console window: {e}")
        return None


def show_console():
    """Show the terminal window"""
    # Re-apply in case the console was recreated or the early startup call
    # ran before this process was attached to one.
    disable_console_quick_edit()
    try:
        hwnd = get_console_window()
        if hwnd:
            user32 = ctypes.windll.user32
            SW_SHOW = 5
            user32.ShowWindow(hwnd, SW_SHOW)
            print("[Terminal Control] Terminal window shown")
            return True
        else:
            print("[Terminal Control] Could not get terminal window handle")
            return False
    except Exception as e:
        print(f"[Terminal Control] Failed to show terminal window: {e}")
        return False


def hide_console():
    """Hide the terminal window"""
    try:
        hwnd = get_console_window()
        if hwnd:
            user32 = ctypes.windll.user32
            SW_HIDE = 0
            user32.ShowWindow(hwnd, SW_HIDE)
            return True
        else:
            return False
    except Exception as e:
        print(f"[Terminal Control] Failed to hide terminal window: {e}")
        return False


def is_console_visible():
    """Check if the terminal window is visible"""
    try:
        hwnd = get_console_window()
        if hwnd:
            user32 = ctypes.windll.user32
            return user32.IsWindowVisible(hwnd)
        return False
    except Exception:
        return False
