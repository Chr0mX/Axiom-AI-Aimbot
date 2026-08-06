# dshow_capture_native.py
"""ctypes binding for the native DirectShow-Capture-DLL (uvc_dshow_backend == 'v2').

This wraps ``directshow_capture.dll`` (vendored at
``src/python/dependencies/directshow_capture.dll``) — a purpose-built native
capture library that owns the DirectShow filter graph and allocator directly,
closing the buffering-control gap ``cv2.VideoCapture`` can't (see the
"DirectShow Capture DLL" roadmap artifact / CLAUDE.md's Capture Pipeline Audit
for the full "why v2 exists" writeup).

IMPORTANT — ABI provenance: no header file shipped with the DLL. The function
signatures below were reconstructed from the PE export table (11 functions,
ordinal-based, confirmed via ``objdump -p``) plus embedded error-message
strings (confirms the result-code/last-error shape) and import-table evidence
(``QueryPerformanceCounter``/``QueryPerformanceFrequency`` from kernel32 —
confirms QPC-based timestamps; wide-char CRT imports — confirms device names
are UTF-16). The exact field order/types of ``capture_params_t`` and the
exact struct layout returned by ``capture_get_latest_frame`` are inferred to
match the C ABI already specified in the DLL roadmap doc, not verified
against source. This could not be executed or tested in this environment
(Linux sandbox, no Windows/Wine) — first real use on Windows should confirm
the struct layout matches (a crash or garbage frame data on first
``capture_open()``/``capture_get_latest_frame()`` call means it doesn't, and
this file needs updating against the DLL's actual header once available).

Exports (ordinal order from the PE export table):
    capture_close, capture_default_params, capture_get_device_count,
    capture_get_device_name, capture_get_last_error, capture_get_latest_frame,
    capture_get_qpc_frequency, capture_open, capture_result_to_string,
    capture_start, capture_stop
"""

from __future__ import annotations

import ctypes
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

_DLL_FILENAME = "directshow_capture.dll"
_dll: Optional[ctypes.WinDLL] = None  # lazily loaded — never at import time (see win32api pattern in CLAUDE.md)


class CaptureParams(ctypes.Structure):
    """Inferred layout — device/format/resolution/fps request, mirrored from
    capture_default_params()'s output before being handed to capture_open().
    Only NV12 has been observed in this build (single pixel-format string in
    the binary), so pixel_format is effectively fixed at 0 for now.
    """
    _fields_ = [
        ("device_index", ctypes.c_int),
        ("width", ctypes.c_int),
        ("height", ctypes.c_int),
        ("fps", ctypes.c_int),
        ("pixel_format", ctypes.c_int),  # 0 = NV12 (only value this build implements)
    ]


class CaptureError(RuntimeError):
    """Raised on any non-zero capture_result_t from the native DLL."""

    def __init__(self, code: int, message: str):
        self.code = code
        super().__init__(f"[directshow_capture.dll] result={code}: {message}")


def _load_dll() -> ctypes.WinDLL:
    global _dll
    if _dll is not None:
        return _dll
    if os.name != 'nt':
        raise RuntimeError(
            "directshow_capture.dll is a Windows-only native DLL — cannot be "
            "loaded on this platform."
        )

    candidates = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python", "dependencies", _DLL_FILENAME),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), _DLL_FILENAME),
    ]
    path = next((os.path.normpath(p) for p in candidates if os.path.isfile(p)), None)
    if path is None:
        raise RuntimeError(
            f"{_DLL_FILENAME} not found — expected at src/python/dependencies/{_DLL_FILENAME}."
        )

    try:
        dll = ctypes.WinDLL(path)
    except OSError as exc:
        # The single most likely cause for a freshly-shipped native DLL
        # failing to load on an arbitrary machine: this build links against
        # MSVCP140.dll/VCRUNTIME140.dll/VCRUNTIME140_1.dll (confirmed via
        # its PE import table) — those ship with the Visual C++ 2015-2022
        # x64 Redistributable, not with Windows itself. ctypes.WinDLL()'s
        # own error ("WinError 126 / module not found") doesn't say which
        # dependency is missing, so spell it out here instead of leaving
        # the user to guess from a cryptic OSError.
        raise RuntimeError(
            f"directshow_capture.dll failed to load ({exc}). This almost "
            "always means a required runtime DLL is missing — install the "
            "\"Microsoft Visual C++ 2015-2022 Redistributable (x64)\" from "
            "https://aka.ms/vs/17/release/vc_redist.x64.exe and retry. If "
            "that doesn't fix it, run `where directshow_capture.dll` and "
            "confirm nothing else on PATH is shadowing the vendored copy."
        ) from exc

    dll.capture_default_params.argtypes = [ctypes.POINTER(CaptureParams)]
    dll.capture_default_params.restype = ctypes.c_int

    dll.capture_open.argtypes = [ctypes.POINTER(CaptureParams), ctypes.POINTER(ctypes.c_void_p)]
    dll.capture_open.restype = ctypes.c_int

    dll.capture_start.argtypes = [ctypes.c_void_p]
    dll.capture_start.restype = ctypes.c_int

    dll.capture_stop.argtypes = [ctypes.c_void_p]
    dll.capture_stop.restype = ctypes.c_int

    dll.capture_close.argtypes = [ctypes.c_void_p]
    dll.capture_close.restype = ctypes.c_int

    dll.capture_get_latest_frame.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.POINTER(ctypes.c_ubyte)),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int64),
    ]
    dll.capture_get_latest_frame.restype = ctypes.c_int

    dll.capture_get_device_count.argtypes = []
    dll.capture_get_device_count.restype = ctypes.c_int

    dll.capture_get_device_name.argtypes = [ctypes.c_int, ctypes.c_wchar_p, ctypes.c_int]
    dll.capture_get_device_name.restype = ctypes.c_int

    dll.capture_get_last_error.argtypes = [ctypes.c_void_p]
    dll.capture_get_last_error.restype = ctypes.c_char_p

    dll.capture_get_qpc_frequency.argtypes = []
    dll.capture_get_qpc_frequency.restype = ctypes.c_int64

    dll.capture_result_to_string.argtypes = [ctypes.c_int]
    dll.capture_result_to_string.restype = ctypes.c_char_p

    _dll = dll
    logger.info("[dshow_capture_native] loaded %s", path)
    return dll


def _last_error_detail(dll: ctypes.WinDLL, handle: ctypes.c_void_p | None) -> str:
    """Best-effort extra detail via capture_get_last_error(), on top of the
    generic capture_result_to_string() code→string mapping. Never raises —
    a wrong assumption about this call's exact behavior (e.g. whether it
    accepts a NULL handle) shouldn't hide the original error.
    """
    try:
        raw = dll.capture_get_last_error(handle if handle is not None else ctypes.c_void_p(None))
        if raw:
            return raw.decode('utf-8', errors='replace')
    except Exception:
        pass
    return ""


def _check(dll: ctypes.WinDLL, code: int, handle: ctypes.c_void_p | None = None) -> None:
    if code == 0:
        return
    try:
        msg = dll.capture_result_to_string(code).decode('utf-8', errors='replace')
    except Exception:
        msg = "(capture_result_to_string call itself failed)"
    detail = _last_error_detail(dll, handle)
    if detail and detail != msg:
        msg = f"{msg} — {detail}"
    raise CaptureError(code, msg)


def list_devices() -> list[str]:
    """Enumerate UVC device friendly names via the native DLL."""
    dll = _load_dll()
    count = dll.capture_get_device_count()
    names: list[str] = []
    buf = ctypes.create_unicode_buffer(256)
    for i in range(count):
        rc = dll.capture_get_device_name(i, buf, len(buf))
        names.append(buf.value if rc == 0 else f"Device {i}")
    return names


def qpc_frequency() -> int:
    """Ticks-per-second for the timestamps capture_get_latest_frame() returns."""
    return int(_load_dll().capture_get_qpc_frequency())


class NativeDshowCapture:
    """Thin RAII-ish wrapper: open() → start() → get_latest_frame() → stop() → close().

    Mirrors the lifecycle every other backend in screen_capture.py follows
    (see UVCCapture) so it slots into the same reader-thread pattern.
    """

    def __init__(self, device_index: int, width: int, height: int, fps: int):
        self._dll = _load_dll()
        logger.info("[dshow_capture_native] DLL loaded, requesting device=%d %dx%d@%dfps (NV12)",
                    device_index, width, height, fps)
        self._handle = ctypes.c_void_p(None)

        params = CaptureParams()
        _check(self._dll, self._dll.capture_default_params(ctypes.byref(params)))
        params.device_index = device_index
        params.width = width
        params.height = height
        params.fps = fps
        params.pixel_format = 0  # NV12 — the only format this build implements
        logger.info("[dshow_capture_native] capture_default_params ok, calling capture_open...")

        _check(self._dll, self._dll.capture_open(ctypes.byref(params), ctypes.byref(self._handle)))
        logger.info("[dshow_capture_native] capture_open ok, handle=%s", self._handle)
        self._started = False

    def start(self) -> None:
        _check(self._dll, self._dll.capture_start(self._handle), self._handle)
        self._started = True
        logger.info("[dshow_capture_native] capture_start ok")

    def get_latest_frame(self):
        """Returns (nv12_ndarray_view, width, height, timestamp_qpc) or None
        if no frame has been captured yet (capture_result_t for that case is
        treated as "not ready" rather than an error — see the DLL's
        "no frame has been captured yet" error string).
        """
        import numpy as np  # local import — keeps this module importable without numpy at load time

        data_ptr = ctypes.POINTER(ctypes.c_ubyte)()
        w = ctypes.c_int(0)
        h = ctypes.c_int(0)
        stride = ctypes.c_int(0)
        ts = ctypes.c_int64(0)

        rc = self._dll.capture_get_latest_frame(
            self._handle, ctypes.byref(data_ptr), ctypes.byref(w), ctypes.byref(h),
            ctypes.byref(stride), ctypes.byref(ts),
        )
        if rc != 0:
            # "no frame yet" is expected on the first poll or two after
            # start() — don't raise, just report not-ready like every other
            # backend's grab() does with a None return.
            return None
        if not data_ptr or w.value <= 0 or h.value <= 0:
            return None

        # NV12: Y plane (h rows) + interleaved UV plane (h/2 rows), each row
        # `stride` bytes. Wrap without copying — caller crops/converts before
        # the DLL's next capture_get_latest_frame() call can overwrite it.
        total_rows = h.value + h.value // 2
        buf = (ctypes.c_ubyte * (stride.value * total_rows)).from_address(ctypes.addressof(data_ptr.contents))
        arr = np.frombuffer(buf, dtype=np.uint8).reshape(total_rows, stride.value)
        return arr, w.value, h.value, ts.value

    def stop(self) -> None:
        if self._started:
            _check(self._dll, self._dll.capture_stop(self._handle), self._handle)
            self._started = False

    def close(self) -> None:
        if self._handle:
            self._dll.capture_close(self._handle)
            self._handle = ctypes.c_void_p(None)
