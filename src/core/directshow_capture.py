"""
Low-level ctypes C ABI bindings for the vendored ``directshow_capture.dll``
(from `chr0mx/DirectShow-Capture-DLL <https://github.com/chr0mx/DirectShow-Capture-DLL>`_).

This module deliberately mirrors that repo's own ``python/directshow_capture.py``
reference binding (same struct layout, same dsc_result/dsc_pixel_format
values — they must match the C header exactly), but stops at "give me the
latest raw frame bytes". It does **not** decode MJPEG or convert NV12 to
BGR/BGRA, and does not apply a region crop — those are the caller's job,
same split as ``udp_receiver.py`` (raw JPEG bytes in, ``UdpCapture`` in
``screen_capture.py`` decodes/crops) and OpenCV's own ``cv2.imdecode``
contract. Keeping the decode/crop logic in ``screen_capture.py`` alongside
every other backend means DirectShowCapture's region-crop-before-convert
optimization can reuse the same ``_crop_nv12`` helper UVCCapture already
uses, instead of duplicating it here.

Windows-only at runtime (raises cleanly the first time a DLL call is
actually attempted on another platform) — importing this module on Linux
(e.g. under pytest) is safe; nothing here touches ctypes.WinDLL at import
time.
"""
from __future__ import annotations

import ctypes
import os
import platform
from typing import Optional

# --- dsc_result (must match directshow_capture.h exactly) ---
DSC_OK = 0
DSC_ERR_INVALID_ARG = -1
DSC_ERR_INVALID_HANDLE = -2
DSC_ERR_DEVICE_NOT_FOUND = -3
DSC_ERR_FORMAT_NOT_SUPPORTED = -4
DSC_ERR_RESOLUTION_NOT_SUPPORTED = -5
DSC_ERR_GRAPH_BUILD_FAILED = -6
DSC_ERR_START_FAILED = -7
DSC_ERR_NOT_RUNNING = -8
DSC_ERR_NO_FRAME_YET = -9
DSC_ERR_ALREADY_RUNNING = -10
DSC_ERR_UNKNOWN = -99

# --- dsc_pixel_format ---
PIXEL_FORMAT_NV12 = 0
PIXEL_FORMAT_MJPEG = 1


class DscOpenParams(ctypes.Structure):
    # Field order/types must match struct dsc_open_params in
    # directshow_capture.h exactly - this is a raw memory layout match, not
    # a name-based one.
    _fields_ = [
        ("device_substr", ctypes.c_char_p),
        ("device_index", ctypes.c_int32),
        ("pixel_format", ctypes.c_int32),
        ("width", ctypes.c_int32),
        ("height", ctypes.c_int32),
        ("fps", ctypes.c_int32),
        ("buffer_count", ctypes.c_int32),
    ]


class DirectShowCaptureError(RuntimeError):
    """Raised for any non-OK dsc_result, with the DLL's own error string."""


def _default_dll_path() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "native", "directshow_capture.dll")


class _Lib:
    """Lazily-loaded, signature-bound DLL handle. One per process is
    enough - DscSession instances underneath are independent."""

    _instance: Optional["_Lib"] = None

    def __init__(self, dll_path: str):
        if platform.system() != "Windows":
            raise RuntimeError("directshow_capture.dll is Windows-only")
        if not os.path.isfile(dll_path):
            raise RuntimeError(
                f"directshow_capture.dll not found at '{dll_path}' — see "
                "src/core/native/README.md for how it's vendored."
            )

        # WinDLL (not CDLL) - matches the __stdcall calling convention the
        # header declares via DSC_CALL. Only matters on 32-bit targets, but
        # costs nothing to get right now.
        self.dll = ctypes.WinDLL(dll_path)

        self.dll.capture_default_params.argtypes = [ctypes.POINTER(DscOpenParams)]
        self.dll.capture_default_params.restype = None

        self.dll.capture_get_device_count.argtypes = []
        self.dll.capture_get_device_count.restype = ctypes.c_int32

        self.dll.capture_get_device_name.argtypes = [
            ctypes.c_int32, ctypes.c_char_p, ctypes.c_int32,
        ]
        self.dll.capture_get_device_name.restype = ctypes.c_int32

        self.dll.capture_open.argtypes = [
            ctypes.POINTER(DscOpenParams), ctypes.POINTER(ctypes.c_int32),
        ]
        self.dll.capture_open.restype = ctypes.c_void_p

        self.dll.capture_start.argtypes = [ctypes.c_void_p]
        self.dll.capture_start.restype = ctypes.c_int32

        self.dll.capture_stop.argtypes = [ctypes.c_void_p]
        self.dll.capture_stop.restype = ctypes.c_int32

        self.dll.capture_close.argtypes = [ctypes.c_void_p]
        self.dll.capture_close.restype = None

        self.dll.capture_get_latest_frame.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.POINTER(ctypes.c_uint8)),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int64),
        ]
        self.dll.capture_get_latest_frame.restype = ctypes.c_int32

        self.dll.capture_get_qpc_frequency.argtypes = []
        self.dll.capture_get_qpc_frequency.restype = ctypes.c_int64

        self.dll.capture_result_to_string.argtypes = [ctypes.c_int32]
        self.dll.capture_result_to_string.restype = ctypes.c_char_p

        self.dll.capture_get_last_error.argtypes = [ctypes.c_void_p]
        self.dll.capture_get_last_error.restype = ctypes.c_char_p

    @classmethod
    def get(cls, dll_path: Optional[str] = None) -> "_Lib":
        if cls._instance is None:
            cls._instance = _Lib(dll_path or _default_dll_path())
        return cls._instance


def _check(lib: "_Lib", result: int, handle=None) -> None:
    if result == DSC_OK:
        return
    reason = lib.dll.capture_result_to_string(result).decode("utf-8", "replace")
    detail = ""
    if handle is not None:
        detail = lib.dll.capture_get_last_error(handle).decode("utf-8", "replace")
    msg = f"{reason} (dsc_result={result})"
    if detail:
        msg += f": {detail}"
    raise DirectShowCaptureError(msg)


def list_devices(dll_path: Optional[str] = None) -> list[str]:
    """Returns the friendly names of all connected UVC devices, in the
    same order capture_get_device_name/device_index expect."""
    lib = _Lib.get(dll_path)
    count = lib.dll.capture_get_device_count()
    if count < 0:
        _check(lib, count)

    names = []
    for i in range(count):
        needed = lib.dll.capture_get_device_name(i, None, 0)
        if needed < 0:
            _check(lib, needed)
        buf = ctypes.create_string_buffer(needed + 1)
        lib.dll.capture_get_device_name(i, buf, len(buf))
        names.append(buf.value.decode("utf-8", "replace"))
    return names


class DscSession:
    """Thin wrapper around one open/start/grab/stop/close handle lifecycle.

    Hands back raw frame bytes exactly as the DLL delivered them — a
    compressed MJPEG blob, or a planar NV12 buffer (Y plane followed by an
    interleaved UV plane, ``height * 3 // 2`` rows total at ``width``
    columns) — with no decode/convert/crop applied. See the module
    docstring for why that split exists.
    """

    def __init__(self, *, device_substr: Optional[str] = None, device_index: int = -1,
                 width: int = 0, height: int = 0, fps: int = 0,
                 pixel_format: int = PIXEL_FORMAT_NV12, buffer_count: int = 4,
                 dll_path: Optional[str] = None) -> None:
        self._lib = _Lib.get(dll_path)
        self.pixel_format = pixel_format
        self.is_compressed = (pixel_format == PIXEL_FORMAT_MJPEG)
        self._handle = None
        self._closed = True

        params = DscOpenParams()
        self._lib.dll.capture_default_params(ctypes.byref(params))
        if device_substr:
            # ctypes doesn't keep bytes alive on its own - hold a reference
            # for as long as `params` (and thus the pointer) might be used.
            self._device_substr_bytes = device_substr.encode("utf-8")
            params.device_substr = self._device_substr_bytes
        params.device_index = device_index
        params.pixel_format = pixel_format
        params.width = width
        params.height = height
        params.fps = fps
        params.buffer_count = buffer_count

        result = ctypes.c_int32()
        handle = self._lib.dll.capture_open(ctypes.byref(params), ctypes.byref(result))
        if not handle:
            _check(self._lib, result.value)

        self._handle = handle
        _check(self._lib, self._lib.dll.capture_start(self._handle), self._handle)
        self._closed = False

    def get_latest_frame(self) -> "tuple[bytes, int, int, int, int] | None":
        """Returns ``(raw_bytes, width, height, stride, timestamp_qpc)``, or
        None if no frame has arrived yet. ``stride`` is 0 for MJPEG (varying
        blob size — use ``len(raw_bytes)`` instead)."""
        if self._closed:
            return None

        data_ptr = ctypes.POINTER(ctypes.c_uint8)()
        data_len = ctypes.c_int32()
        width = ctypes.c_int32()
        height = ctypes.c_int32()
        stride = ctypes.c_int32()
        ts_qpc = ctypes.c_int64()

        result = self._lib.dll.capture_get_latest_frame(
            self._handle, ctypes.byref(data_ptr), ctypes.byref(data_len),
            ctypes.byref(width), ctypes.byref(height), ctypes.byref(stride),
            ctypes.byref(ts_qpc),
        )
        if result == DSC_ERR_NO_FRAME_YET:
            return None
        _check(self._lib, result, self._handle)

        # Copy out of the DLL's buffer immediately - it's only valid until
        # the next capture_get_latest_frame call (see directshow_capture.h).
        raw = bytes(ctypes.cast(data_ptr, ctypes.POINTER(ctypes.c_uint8 * data_len.value))[0])
        return raw, width.value, height.value, stride.value, ts_qpc.value

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._lib.dll.capture_stop(self._handle)
        self._lib.dll.capture_close(self._handle)
        self._handle = None

    def __enter__(self) -> "DscSession":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
