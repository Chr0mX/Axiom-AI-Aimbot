"""
core/directshow_capture.py 單元測試

Only covers what's safe/meaningful without a real Windows box + DLL + UVC
device attached (constants, struct layout, path resolution, the
non-Windows guard) — the actual DLL round-trip is exercised by
scripts/test_directshow_capture.py on real hardware instead. This module
imports cleanly on Linux (no win32api, no ctypes.WinDLL at import time —
see the module docstring), so it needs no special collection handling
unlike ai_loop.py/ai_loop_utils.py/ai_aiming.py.
"""
import os
import platform

import pytest


def test_pixel_format_constants_match_c_header():
    from core import directshow_capture as dsc

    # Raw ctypes struct field values sent across the C ABI - must match
    # dsc_pixel_format in directshow_capture.h exactly, not just be
    # internally consistent.
    assert dsc.PIXEL_FORMAT_NV12 == 0
    assert dsc.PIXEL_FORMAT_MJPEG == 1


def test_dsc_result_constants_match_c_header():
    from core import directshow_capture as dsc

    assert dsc.DSC_OK == 0
    assert dsc.DSC_ERR_NO_FRAME_YET == -9


def test_default_dll_path_points_at_vendored_native_dir():
    from core import directshow_capture as dsc

    path = dsc._default_dll_path()
    assert path.endswith(os.path.join('native', 'directshow_capture.dll'))
    assert os.path.isfile(path), (
        'src/core/native/directshow_capture.dll must be vendored — '
        'see src/core/native/README.md'
    )


def test_open_params_struct_field_order_matches_c_header():
    """DscOpenParams is a raw memory-layout match with dsc_open_params in
    directshow_capture.h - field order/types must not drift independently
    of the C header."""
    from core import directshow_capture as dsc

    field_names = [name for name, _ in dsc.DscOpenParams._fields_]
    assert field_names == [
        'device_substr', 'device_index', 'pixel_format',
        'width', 'height', 'fps', 'buffer_count',
    ]


@pytest.mark.skipif(platform.system() == 'Windows', reason='exercises the non-Windows guard specifically')
def test_list_devices_raises_cleanly_on_non_windows():
    from core import directshow_capture as dsc

    with pytest.raises(RuntimeError, match='Windows-only'):
        dsc.list_devices()


@pytest.mark.skipif(platform.system() == 'Windows', reason='exercises the non-Windows guard specifically')
def test_dsc_session_raises_cleanly_on_non_windows():
    from core import directshow_capture as dsc

    with pytest.raises(RuntimeError, match='Windows-only'):
        dsc.DscSession()
