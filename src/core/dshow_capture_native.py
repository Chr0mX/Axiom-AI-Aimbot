# dshow_capture_native.py
"""ctypes binding for the native DirectShow-Capture-DLL (uvc_dshow_backend == 'v2').

This wraps ``directshow_capture.dll`` (vendored at
``src/python/dependencies/directshow_capture.dll``) — a purpose-built native
capture library that owns the DirectShow filter graph and allocator directly,
closing the buffering-control gap ``cv2.VideoCapture`` can't (see the
"DirectShow Capture DLL" roadmap artifact / CLAUDE.md's Capture Pipeline Audit
for the full "why v2 exists" writeup).

ABI correction: the version of this file that first shipped this backend
reverse-engineered the C ABI from the DLL's PE export table alone (no header
was available to that session) via ``objdump -p``. That guess turned out to
diverge from the real, public ABI — the upstream repo
(``chr0mx/DirectShow-Capture-DLL``, ``src/include/directshow_capture.h``) —
in several load-bearing ways, all fixed here:

- ``capture_open``'s return value and out-param were backwards: the real ABI
  returns the ``dsc_handle`` directly and writes the ``dsc_result`` through
  an out-param, not the other way around.
- The params struct (``dsc_open_params``) was missing 2 of 7 fields
  (``device_substr``, ``buffer_count``) and had the remaining ones in the
  wrong order — a straight memory-layout mismatch.
- ``capture_get_latest_frame`` was called with one fewer argument than the
  real (``__stdcall``) function takes (missing ``out_data_len``), which
  misaligns the stack on every call.
- ``capture_get_device_name`` assumed UTF-16 (``wchar_t*``); the real ABI is
  UTF-8 (``char*``).
- ``capture_default_params``/``capture_close`` are ``void`` in the real ABI,
  not ``dsc_result`` — the old code fed their (undefined) return value
  through ``_check()``, e.g. treating garbage as a possible error code.

Any one of the first three would misbehave or crash on first real use
against the actual DLL. This version matches the real header field-for-field
(open it alongside this file to verify) and adds real MJPEG support — the
previous version assumed NV12 was the only format this build implements,
which was itself an artifact of not having the header (the real DLL supports
both ``DSC_PIXEL_FORMAT_NV12`` and ``DSC_PIXEL_FORMAT_MJPEG``).

Native crop (V2 ABI addition): ``dsc_open_params`` gained four trailing
fields (``crop_x``/``crop_y``/``crop_width``/``crop_height``) for a real,
DLL-side spatial crop — see ``directshow_capture.h``'s doc comment on why
that's a distinct thing from just requesting a smaller resolution (a device
advertising a smaller mode carries no guarantee it's a crop of the same
framing rather than the whole scene rescaled; ``screen_capture.py``'s
``uvc_crop_mode == 'fixed'`` path used to do exactly that resolution-request
trick before this field existed — see its own history for the corrected
architecture). ``CaptureParams`` below is a strict append — same 7 fields as
before, unchanged offsets, plus these 4 new ones at the end — so a DLL build
that predates this addition still opens fine (it never reads past
``buffer_count``, so trailing crop fields are silently ignored, not
rejected). That "silently ignored" behavior is exactly why callers here
can't just trust that requesting a crop worked because ``capture_open()``
returned OK — see ``UVCCapture._init_native_dll``'s first-frame dimension
check for how that gets verified for real.
"""

from __future__ import annotations

import ctypes
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

_DLL_FILENAME = "directshow_capture.dll"
_dll: Optional[ctypes.WinDLL] = None  # lazily loaded — never at import time (see win32api pattern in CLAUDE.md)

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
DSC_ERR_CROP_NOT_SUPPORTED = -11  # crop requested for MJPEG (NV12-only), or an invalid/misaligned rect
DSC_ERR_UNKNOWN = -99

# --- dsc_pixel_format ---
PIXEL_FORMAT_NV12 = 0
PIXEL_FORMAT_MJPEG = 1


class CaptureParams(ctypes.Structure):
    """Raw memory-layout match with ``dsc_open_params`` in
    directshow_capture.h — field order/types are load-bearing, not just
    internally consistent (see module docstring). The four crop_* fields are
    appended at the end (V2) — a superset of the original 7-field layout,
    not a reordering."""
    _fields_ = [
        ("device_substr", ctypes.c_char_p),  # UTF-8, case-insensitive substring; nullable
        ("device_index", ctypes.c_int32),    # -1 = use device_substr / first available
        ("pixel_format", ctypes.c_int32),    # PIXEL_FORMAT_NV12 / PIXEL_FORMAT_MJPEG
        ("width", ctypes.c_int32),
        ("height", ctypes.c_int32),
        ("fps", ctypes.c_int32),
        ("buffer_count", ctypes.c_int32),
        ("crop_x", ctypes.c_int32),
        ("crop_y", ctypes.c_int32),
        ("crop_width", ctypes.c_int32),   # <= 0 = no crop, full negotiated frame
        ("crop_height", ctypes.c_int32),
    ]


class CaptureError(RuntimeError):
    """Raised on any non-OK dsc_result from the native DLL."""

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
    dll.capture_default_params.restype = None

    dll.capture_get_device_count.argtypes = []
    dll.capture_get_device_count.restype = ctypes.c_int32

    dll.capture_get_device_name.argtypes = [ctypes.c_int32, ctypes.c_char_p, ctypes.c_int32]
    dll.capture_get_device_name.restype = ctypes.c_int32

    # capture_open returns the dsc_handle directly (c_void_p) and writes the
    # dsc_result through the second (out) param — NOT the other way around.
    dll.capture_open.argtypes = [ctypes.POINTER(CaptureParams), ctypes.POINTER(ctypes.c_int32)]
    dll.capture_open.restype = ctypes.c_void_p

    dll.capture_start.argtypes = [ctypes.c_void_p]
    dll.capture_start.restype = ctypes.c_int32

    dll.capture_stop.argtypes = [ctypes.c_void_p]
    dll.capture_stop.restype = ctypes.c_int32

    dll.capture_close.argtypes = [ctypes.c_void_p]
    dll.capture_close.restype = None

    dll.capture_get_latest_frame.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.POINTER(ctypes.c_uint8)),
        ctypes.POINTER(ctypes.c_int32),  # out_data_len — the field the old bindings dropped entirely
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int64),
    ]
    dll.capture_get_latest_frame.restype = ctypes.c_int32

    dll.capture_set_crop.argtypes = [
        ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
    ]
    dll.capture_set_crop.restype = ctypes.c_int32

    dll.capture_get_qpc_frequency.argtypes = []
    dll.capture_get_qpc_frequency.restype = ctypes.c_int64

    dll.capture_result_to_string.argtypes = [ctypes.c_int32]
    dll.capture_result_to_string.restype = ctypes.c_char_p

    dll.capture_get_last_error.argtypes = [ctypes.c_void_p]
    dll.capture_get_last_error.restype = ctypes.c_char_p

    _dll = dll
    logger.info("[dshow_capture_native] loaded %s", path)
    return dll


def _last_error_detail(dll: ctypes.WinDLL, handle: ctypes.c_void_p | int | None) -> str:
    """Best-effort extra detail via capture_get_last_error(), on top of the
    generic capture_result_to_string() code→string mapping. Never raises —
    a wrong assumption about this call's exact behavior (e.g. whether it
    accepts a NULL handle) shouldn't hide the original error.
    """
    try:
        raw = dll.capture_get_last_error(handle if handle else ctypes.c_void_p(None))
        if raw:
            return raw.decode('utf-8', errors='replace')
    except Exception:
        pass
    return ""


def _check(dll: ctypes.WinDLL, code: int, handle: ctypes.c_void_p | int | None = None) -> None:
    if code == DSC_OK:
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
    """Enumerate UVC device friendly names via the native DLL (UTF-8)."""
    dll = _load_dll()
    count = dll.capture_get_device_count()
    if count < 0:
        _check(dll, count)

    names: list[str] = []
    for i in range(count):
        needed = dll.capture_get_device_name(i, None, 0)
        if needed < 0:
            names.append(f"Device {i}")
            continue
        buf = ctypes.create_string_buffer(needed + 1)
        dll.capture_get_device_name(i, buf, len(buf))
        names.append(buf.value.decode('utf-8', errors='replace'))
    return names


def qpc_frequency() -> int:
    """Ticks-per-second for the timestamps capture_get_latest_frame() returns."""
    return int(_load_dll().capture_get_qpc_frequency())


class NativeDshowCapture:
    """Thin RAII-ish wrapper: open() → start() → get_latest_frame() → stop() → close().

    Mirrors the lifecycle every other backend in screen_capture.py follows
    (see UVCCapture) so it slots into the same reader-thread pattern.
    """

    def __init__(self, device_index: int, width: int, height: int, fps: int,
                 pixel_format: int = PIXEL_FORMAT_NV12, buffer_count: int = 4,
                 device_substr: Optional[str] = None,
                 crop: Optional[tuple[int, int, int, int]] = None):
        """crop: optional (x, y, width, height), in the negotiated width/
        height's own coordinate space — requests a real, DLL-side spatial
        crop of every captured frame (NV12 only; raises CaptureError with
        DSC_ERR_CROP_NOT_SUPPORTED if combined with pixel_format=
        PIXEL_FORMAT_MJPEG). All four values must be even and fit inside
        width x height, or capture_open() raises DSC_ERR_INVALID_ARG.
        Device negotiation itself (width/height/fps/pixel_format) is
        unaffected either way — this is a separate, later, in-DLL step, not
        a request for a different resolution mode."""
        self._dll = _load_dll()
        format_name = 'MJPEG' if pixel_format == PIXEL_FORMAT_MJPEG else 'NV12'
        self.is_compressed = (pixel_format == PIXEL_FORMAT_MJPEG)
        logger.info(
            "[dshow_capture_native] DLL loaded, requesting device=%s %dx%d@%dfps (%s)%s",
            device_substr if device_substr else device_index, width, height, fps, format_name,
            f", crop={crop}" if crop else "",
        )

        params = CaptureParams()
        self._dll.capture_default_params(ctypes.byref(params))  # void — nothing to _check()
        # ctypes doesn't keep bytes alive on its own — hold a reference for
        # as long as `params` (and thus capture_open's read of it) might be used.
        self._device_substr_bytes: Optional[bytes] = None
        if device_substr:
            self._device_substr_bytes = device_substr.encode('utf-8')
            params.device_substr = self._device_substr_bytes
        params.device_index = device_index
        params.pixel_format = pixel_format
        params.width = width
        params.height = height
        params.fps = fps
        params.buffer_count = buffer_count
        if crop is not None:
            crop_x, crop_y, crop_w, crop_h = crop
            params.crop_x = crop_x
            params.crop_y = crop_y
            params.crop_width = crop_w
            params.crop_height = crop_h
        logger.info("[dshow_capture_native] capture_default_params ok, calling capture_open...")

        result = ctypes.c_int32()
        handle = self._dll.capture_open(ctypes.byref(params), ctypes.byref(result))
        if not handle:
            _check(self._dll, result.value)
        self._handle = ctypes.c_void_p(handle)
        logger.info("[dshow_capture_native] capture_open ok, handle=%s", self._handle)
        self._started = False

    def start(self) -> None:
        _check(self._dll, self._dll.capture_start(self._handle), self._handle)
        self._started = True
        logger.info("[dshow_capture_native] capture_start ok")

    def get_latest_frame(self):
        """Returns ``(raw_bytes, width, height, timestamp_qpc)`` — a
        compressed MJPEG blob when ``is_compressed``, otherwise a raw NV12
        buffer (``height * 3 // 2`` rows at ``width`` columns, no row
        padding) — or None if no frame has been captured yet.

        Copies out of the DLL's own buffer immediately, matching the DLL's
        documented contract: the pointer ``capture_get_latest_frame()``
        hands back is only valid until the next call on this handle.
        """
        data_ptr = ctypes.POINTER(ctypes.c_uint8)()
        data_len = ctypes.c_int32(0)
        w = ctypes.c_int32(0)
        h = ctypes.c_int32(0)
        stride = ctypes.c_int32(0)
        ts = ctypes.c_int64(0)

        rc = self._dll.capture_get_latest_frame(
            self._handle, ctypes.byref(data_ptr), ctypes.byref(data_len),
            ctypes.byref(w), ctypes.byref(h), ctypes.byref(stride), ctypes.byref(ts),
        )
        if rc == DSC_ERR_NO_FRAME_YET:
            # Expected on the first poll or two after start() — don't raise,
            # just report not-ready like every other backend's grab() does
            # with a None return.
            return None
        _check(self._dll, rc, self._handle)
        if not data_ptr or w.value <= 0 or h.value <= 0:
            return None

        raw = bytes(ctypes.cast(data_ptr, ctypes.POINTER(ctypes.c_uint8 * data_len.value))[0])
        return raw, w.value, h.value, ts.value

    def set_crop(self, x: int, y: int, width: int, height: int) -> None:
        """Live-adjusts the native crop rectangle without a graph rebuild —
        see capture_set_crop()'s doc comment in directshow_capture.h.
        Coordinates are in the negotiated capture resolution's own space,
        same rules as the constructor's `crop`. Not currently called from
        UVCCapture (a Detection Range change already triggers a full backend
        reinit via _uvc_signature), but kept available for a future
        live-adjust path without paying that reinit cost."""
        _check(self._dll, self._dll.capture_set_crop(self._handle, x, y, width, height), self._handle)

    def disable_crop(self) -> None:
        """Stops cropping — get_latest_frame() goes back to full frames."""
        _check(self._dll, self._dll.capture_set_crop(self._handle, 0, 0, 0, 0), self._handle)

    def stop(self) -> None:
        if self._started:
            _check(self._dll, self._dll.capture_stop(self._handle), self._handle)
            self._started = False

    def close(self) -> None:
        if self._handle:
            self._dll.capture_close(self._handle)  # void — nothing to _check()
            self._handle = ctypes.c_void_p(None)
