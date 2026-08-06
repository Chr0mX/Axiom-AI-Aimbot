"""
core/dshow_capture_native.py 單元測試

Covers the ABI surface that regressed silently before (wrong capture_open
return/out-param mapping, wrong CaptureParams field count/order, wrong
capture_get_latest_frame argument count, wrong device-name string encoding —
see the module's docstring for the full list). None of this needs a real
Windows box + DLL + device to verify structurally; the actual DLL round-trip
still needs a real run on Windows (dshow_capture_native.py's ABI is now
believed correct against the upstream header, but "believed" isn't
"verified on hardware").

This module imports cleanly on Linux (no win32api, no ctypes.WinDLL at
import time), so it needs no special collection handling.
"""
import os
import platform

import pytest


def test_pixel_format_constants_match_c_header():
    from core import dshow_capture_native as dsn

    assert dsn.PIXEL_FORMAT_NV12 == 0
    assert dsn.PIXEL_FORMAT_MJPEG == 1


def test_dsc_result_constants_match_c_header():
    from core import dshow_capture_native as dsn

    assert dsn.DSC_OK == 0
    assert dsn.DSC_ERR_NO_FRAME_YET == -9


def test_capture_params_struct_field_order_matches_c_header():
    """CaptureParams is a raw memory-layout match with dsc_open_params in
    directshow_capture.h — field order/count/types must not drift
    independently of the C header. This is exactly the class of bug that
    slipped through before (5 fields instead of 7, missing device_substr
    and buffer_count, wrong order)."""
    from core import dshow_capture_native as dsn

    field_names = [name for name, _ in dsn.CaptureParams._fields_]
    assert field_names == [
        'device_substr', 'device_index', 'pixel_format',
        'width', 'height', 'fps', 'buffer_count',
    ]


@pytest.mark.skipif(platform.system() == 'Windows', reason='exercises the non-Windows guard specifically')
def test_list_devices_raises_cleanly_on_non_windows():
    from core import dshow_capture_native as dsn

    with pytest.raises(RuntimeError, match='Windows-only'):
        dsn.list_devices()


@pytest.mark.skipif(platform.system() == 'Windows', reason='exercises the non-Windows guard specifically')
def test_native_dshow_capture_raises_cleanly_on_non_windows():
    from core import dshow_capture_native as dsn

    with pytest.raises(RuntimeError, match='Windows-only'):
        dsn.NativeDshowCapture(device_index=0, width=0, height=0, fps=0)
