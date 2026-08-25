from types import SimpleNamespace

import numpy as np


def test_initialize_screen_capture_uses_dxcam_when_available(monkeypatch):
    from core import screen_capture as sc

    fake_dxcam_backend = object()
    monkeypatch.setattr(sc, '_initialize_dxcam_capture', lambda: fake_dxcam_backend)

    config = SimpleNamespace(screenshot_method='dxcam')
    backend = sc.initialize_screen_capture(config)

    assert backend is fake_dxcam_backend
    assert config.screenshot_method == 'dxcam'


def test_initialize_screen_capture_fallbacks_to_mss_when_dxcam_unavailable(monkeypatch):
    from core import screen_capture as sc

    fake_mss_backend = object()
    monkeypatch.setattr(sc, '_initialize_dxcam_capture', lambda: None)
    monkeypatch.setattr(sc.mss, 'mss', lambda: fake_mss_backend)

    config = SimpleNamespace(screenshot_method='dxcam')
    backend = sc.initialize_screen_capture(config)

    assert backend is fake_mss_backend
    assert config.screenshot_method == 'mss'


def test_capture_frame_success_returns_bgra_ndarray():
    from core import screen_capture as sc

    class FakeShot:
        width = 2
        height = 1
        bgra = bytes([1, 2, 3, 4, 5, 6, 7, 8])

    class FakeCapture:
        def grab(self, region):
            return FakeShot()

    frame = sc.capture_frame(FakeCapture(), {'left': 0, 'top': 0, 'width': 2, 'height': 1})

    assert frame is not None
    assert frame.shape == (1, 2, 4)
    assert frame.dtype == np.uint8


def test_capture_frame_returns_none_on_screenshot_error(monkeypatch):
    from core import screen_capture as sc

    class FakeScreenShotError(Exception):
        pass

    monkeypatch.setattr(sc.mss.exception, 'ScreenShotError', FakeScreenShotError)

    class FakeCapture:
        def grab(self, region):
            raise FakeScreenShotError('capture failed')

    frame = sc.capture_frame(FakeCapture(), {'left': 0, 'top': 0, 'width': 10, 'height': 10})
    assert frame is None


def test_capture_frame_supports_dxcam_region_tuple_and_ndarray():
    from core import screen_capture as sc

    expected_region = (10, 20, 40, 60)
    frame_data = np.zeros((40, 30, 4), dtype=np.uint8)

    class FakeDxcamCapture:
        def grab(self, region=None):
            if isinstance(region, dict):
                raise TypeError('dxcam expects tuple region')
            assert region == expected_region
            return frame_data

    frame = sc.capture_frame(FakeDxcamCapture(), {'left': 10, 'top': 20, 'width': 30, 'height': 40})
    assert frame is frame_data


def test_capture_frame_detects_dxcam_explicitly_via_module_name():
    """capture_frame() must detect a dxcam backend up front (by module name,
    same check _detect_active_capture_method uses) and go straight to the
    tuple-region call — not rely on catching whatever TypeError dxcam's own
    region validation happens to raise for a dict. Give the fake capture a
    grab() that raises on ANY dict argument (even as a kwarg) to prove the
    dict form is never attempted at all, not just that the fallback works.
    """
    from core import screen_capture as sc

    expected_region = (5, 5, 15, 15)
    frame_data = np.zeros((10, 10, 4), dtype=np.uint8)

    class FakeDxcamCapture:
        def grab(self, region=None):
            if isinstance(region, dict):
                raise AssertionError('capture_frame must not try a dict region against dxcam')
            assert region == expected_region
            return frame_data

    FakeDxcamCapture.__module__ = 'dxcam.core'  # matches the real dxcam package's module prefix
    frame = sc.capture_frame(FakeDxcamCapture(), {'left': 5, 'top': 5, 'width': 10, 'height': 10})
    assert frame is frame_data


def test_initialize_screen_capture_prints_fallback_prompt_once(monkeypatch, capsys):
    from core import screen_capture as sc

    sc._WARNED_MESSAGES.clear()
    fake_mss_backend = object()
    monkeypatch.setattr(sc, '_initialize_dxcam_capture', lambda: None)
    monkeypatch.setattr(sc.mss, 'mss', lambda: fake_mss_backend)

    config1 = SimpleNamespace(screenshot_method='dxcam')
    config2 = SimpleNamespace(screenshot_method='dxcam')

    sc.initialize_screen_capture(config1)
    sc.initialize_screen_capture(config2)

    output = capsys.readouterr().out
    assert output.count('dxcam 不可用，已自動切換為 mss') == 1


def test_capture_frame_prints_error_prompt_once(monkeypatch, capsys):
    from core import screen_capture as sc

    sc._WARNED_MESSAGES.clear()

    class FakeScreenShotError(Exception):
        pass

    monkeypatch.setattr(sc.mss.exception, 'ScreenShotError', FakeScreenShotError)

    class FakeCapture:
        def grab(self, region):
            raise FakeScreenShotError('capture failed')

    sc.capture_frame(FakeCapture(), {'left': 0, 'top': 0, 'width': 10, 'height': 10})
    sc.capture_frame(FakeCapture(), {'left': 0, 'top': 0, 'width': 10, 'height': 10})

    output = capsys.readouterr().out
    assert output.count('[Capture] Screenshot failed: capture failed') == 1


def test_find_ndi_source_by_name_accepts_stream_name_case_insensitive():
    from core import screen_capture as sc

    source = SimpleNamespace(name='OBS-PC', stream_name='MainFeed')

    class FakeFinder:
        def get_source(self, target):
            return None

        def __iter__(self):
            yield source

    assert sc._find_ndi_source_by_name(FakeFinder(), 'mainfeed') is source


def test_detect_active_capture_method_identifies_fallback_to_mss():
    from core import screen_capture as sc

    class FakeMSSBackend:
        __module__ = 'mss.windows'

    backend = FakeMSSBackend()
    assert sc._detect_active_capture_method(backend, 'ndi') == 'mss'


def test_reinitialize_if_method_changed_uses_detected_active_method(monkeypatch):
    from core import screen_capture as sc

    class FakeMSSBackend:
        __module__ = 'mss.windows'

    class FakeNDIBackend:
        pass

    calls = {'count': 0}

    def fake_initialize(config):
        calls['count'] += 1
        return FakeNDIBackend()

    monkeypatch.setattr(sc, 'initialize_screen_capture', fake_initialize)
    config = SimpleNamespace(screenshot_method='ndi')

    # Simulate previous NDI init failed and app is currently running on mss.
    backend, active = sc.reinitialize_if_method_changed(config, FakeMSSBackend(), 'ndi')

    assert calls['count'] == 1
    assert isinstance(backend, FakeNDIBackend)
    assert active == 'ndi'


def test_capture_backend_is_stale_true_after_timeout():
    from core import screen_capture as sc
    import time

    backend = SimpleNamespace(_last_frame_perf_time=time.perf_counter() - (sc._CAPTURE_STALE_TIMEOUT_SECONDS + 1.0))
    assert sc._capture_backend_is_stale(backend) is True


def test_capture_backend_is_stale_false_when_recent():
    from core import screen_capture as sc
    import time

    backend = SimpleNamespace(_last_frame_perf_time=time.perf_counter())
    assert sc._capture_backend_is_stale(backend) is False


def test_capture_backend_is_stale_false_when_untracked():
    from core import screen_capture as sc

    # mss/dxcam and anything else without the attribute is never "stale" —
    # only uvc/udp reader threads publish _last_frame_perf_time.
    assert sc._capture_backend_is_stale(SimpleNamespace()) is False


def _uvc_test_config(**overrides):
    base = dict(
        screenshot_method='uvc', uvc_device_index=0, uvc_width=1920, uvc_height=1080,
        uvc_fps=60, uvc_show_window=False, uvc_capture_method='dshow',
        uvc_preview_scale_mode='scale_to_fit', uvc_video_format='mjpeg',
        uvc_ffmpeg_path='', uvc_crop_mode='dynamic', detect_range_size=320,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_reinitialize_if_method_changed_recovers_stalled_uvc_backend(monkeypatch):
    """A UVC backend that opened fine but has gone silent at runtime (device
    unplugged, driver hiccup) must be torn down and recreated automatically —
    this is the fix for 'switching to uvc/udp after it fails falls back and
    never tries to reconnect, requiring a relaunch'."""
    from core import screen_capture as sc
    import time

    config = _uvc_test_config()
    stale_backend = object.__new__(sc.UVCCapture)
    stale_backend.config_signature = sc._uvc_signature(config)
    stale_backend._last_frame_perf_time = time.perf_counter() - (sc._CAPTURE_STALE_TIMEOUT_SECONDS + 1.0)

    fresh_backend = object()
    calls = {'count': 0}

    def fake_initialize(cfg):
        calls['count'] += 1
        return fresh_backend

    monkeypatch.setattr(sc, 'initialize_screen_capture', fake_initialize)
    monkeypatch.setattr(sc, '_cleanup_capture', lambda backend: None)

    backend, active = sc.reinitialize_if_method_changed(config, stale_backend, 'uvc')

    assert calls['count'] == 1
    assert backend is fresh_backend
    assert active == 'uvc'


def test_reinitialize_if_method_changed_leaves_healthy_uvc_backend_alone(monkeypatch):
    from core import screen_capture as sc
    import time

    config = _uvc_test_config()
    healthy_backend = object.__new__(sc.UVCCapture)
    healthy_backend.config_signature = sc._uvc_signature(config)
    healthy_backend._last_frame_perf_time = time.perf_counter()

    calls = {'count': 0}
    monkeypatch.setattr(sc, 'initialize_screen_capture', lambda cfg: calls.__setitem__('count', calls['count'] + 1) or object())

    backend, active = sc.reinitialize_if_method_changed(config, healthy_backend, 'uvc')

    assert calls['count'] == 0
    assert backend is healthy_backend
    assert active == 'uvc'


def test_reinitialize_if_method_changed_throttles_repeated_stale_recovery(monkeypatch):
    """Even if the recreated backend goes stale again immediately (device
    still gone), reinit attempts must respect the retry interval instead of
    hammering the driver every 0.5s capture-worker tick."""
    from core import screen_capture as sc
    import time

    config = _uvc_test_config()
    stale_backend = object.__new__(sc.UVCCapture)
    stale_backend.config_signature = sc._uvc_signature(config)
    stale_backend._last_frame_perf_time = time.perf_counter() - (sc._CAPTURE_STALE_TIMEOUT_SECONDS + 1.0)

    calls = {'count': 0}

    def fake_initialize(cfg):
        calls['count'] += 1
        return object()

    monkeypatch.setattr(sc, 'initialize_screen_capture', fake_initialize)
    monkeypatch.setattr(sc, '_cleanup_capture', lambda backend: None)

    sc.reinitialize_if_method_changed(config, stale_backend, 'uvc')
    # Immediately call again with the same (still-stale) backend reference —
    # must be throttled, not re-attempted.
    sc.reinitialize_if_method_changed(config, stale_backend, 'uvc')

    assert calls['count'] == 1


def test_reinitialize_if_method_changed_recovers_stalled_udp_backend(monkeypatch):
    from core import screen_capture as sc
    import time

    config = SimpleNamespace(
        screenshot_method='udp', udp_bind_ip='0.0.0.0', udp_bind_port=5600,
        udp_recv_buffer_size=65536, udp_frame_timeout=1.0, udp_force_restart=False,
    )
    stale_backend = object.__new__(sc.UdpCapture)
    stale_backend.config_signature = sc._udp_signature(config)
    stale_backend._last_frame_perf_time = time.perf_counter() - (sc._CAPTURE_STALE_TIMEOUT_SECONDS + 1.0)

    fresh_backend = object()
    calls = {'count': 0}

    def fake_initialize(cfg):
        calls['count'] += 1
        return fresh_backend

    monkeypatch.setattr(sc, 'initialize_screen_capture', fake_initialize)
    monkeypatch.setattr(sc, '_cleanup_capture', lambda backend: None)

    backend, active = sc.reinitialize_if_method_changed(config, stale_backend, 'udp')

    assert calls['count'] == 1
    assert backend is fresh_backend
    assert active == 'udp'


def test_uvc_grab_fixed_crop_mode_ignores_live_region():
    """Fixed crop mode (uvc_crop_mode='fixed', dshow/msmf path) must use the
    rect frozen at capture-start, not whatever region grab() is called
    with each frame — mirrors ffmpeg mode's frozen -vf crop, generalized to
    the in-process cv2 capture path."""
    from core import screen_capture as sc
    import threading

    capture = object.__new__(sc.UVCCapture)
    capture._latest_frame_lock = threading.Lock()
    capture._latest_frame_ref = [np.zeros((100, 100, 3), dtype=np.uint8)]
    capture._region_ref = [None]
    capture.is_raw_nv12 = False
    capture.preview_width = 100
    capture.preview_height = 100
    capture._fixed_region = {'left': 20, 'top': 20, 'width': 40, 'height': 40}

    live_region = {'left': 0, 'top': 0, 'width': 10, 'height': 10}
    result = capture.grab(region=live_region)

    assert result.shape == (40, 40, 4)
    # The preview overlay must be told the rect actually used (the frozen
    # one), not the live one it was called with.
    assert capture._region_ref[0] == capture._fixed_region


def test_uvc_grab_dynamic_mode_uses_live_region():
    from core import screen_capture as sc
    import threading

    capture = object.__new__(sc.UVCCapture)
    capture._latest_frame_lock = threading.Lock()
    capture._latest_frame_ref = [np.zeros((100, 100, 3), dtype=np.uint8)]
    capture._region_ref = [None]
    capture.is_raw_nv12 = False
    capture.preview_width = 100
    capture.preview_height = 100
    capture._fixed_region = None  # dynamic mode

    live_region = {'left': 0, 'top': 0, 'width': 10, 'height': 10}
    result = capture.grab(region=live_region)

    assert result.shape == (10, 10, 4)
    assert capture._region_ref[0] == live_region


def test_uvc_grab_native_nv12_uses_frame_shape_not_stale_preview_dims():
    """Regression: _reader_worker_native_dll publishes the frame under
    _latest_frame_lock, then writes preview_width/height *outside* the
    lock right after — so a resolution/crop transition landing between
    those two reads can leave preview_width/height describing a
    *different* frame than the one grab() just fetched (BUG-07). NV12
    plane math must derive from the fetched buffer's own shape instead,
    which can't be stale relative to itself."""
    from core import screen_capture as sc
    import threading

    w, h = 64, 64
    # Real NV12 buffer: h*3//2 rows (luma + interleaved UV) at w columns.
    frame = np.full((h * 3 // 2, w), 128, dtype=np.uint8)

    capture = object.__new__(sc.UVCCapture)
    capture._latest_frame_lock = threading.Lock()
    capture._latest_frame_ref = [frame]
    capture._region_ref = [None]
    capture.is_raw_nv12 = True
    capture._fixed_region = None
    # Deliberately stale/wrong, smaller than the real buffer — simulates the
    # race where these were updated for a since-superseded frame.
    capture.preview_width = 32
    capture.preview_height = 32

    region = {'left': 0, 'top': 0, 'width': w, 'height': h}
    result = capture.grab(region=region)

    # A correct crop against the buffer's real 64x64 dimensions, not the
    # stale 32x32 preview_width/height.
    assert result is not None
    assert result.shape == (h, w, 4)


def test_resolve_native_dll_pixel_format_maps_nv12():
    from core import screen_capture as sc
    from core import dshow_capture_native as dsn

    config = SimpleNamespace(uvc_video_format='nv12')
    fmt, name = sc._resolve_native_dll_pixel_format(config)
    assert fmt == dsn.PIXEL_FORMAT_NV12
    assert name == 'NV12'


def test_resolve_native_dll_pixel_format_maps_mjpeg():
    from core import screen_capture as sc
    from core import dshow_capture_native as dsn

    config = SimpleNamespace(uvc_video_format='mjpeg')
    fmt, name = sc._resolve_native_dll_pixel_format(config)
    assert fmt == dsn.PIXEL_FORMAT_MJPEG
    assert name == 'MJPEG'


def test_resolve_native_dll_pixel_format_falls_back_for_unsupported_format():
    """yuy2/yuv420p are valid for the cv2 (v1) path but the native DLL only
    implements NV12/MJPEG — must fall back rather than silently misconfigure
    the DLL open call with a format it doesn't have."""
    from core import screen_capture as sc
    from core import dshow_capture_native as dsn

    config = SimpleNamespace(uvc_video_format='yuy2')
    fmt, name = sc._resolve_native_dll_pixel_format(config)
    assert fmt == dsn.PIXEL_FORMAT_MJPEG
    assert name == 'MJPEG'


def test_compute_fixed_uvc_crop_region_centers_the_crop():
    from core import screen_capture as sc

    config = SimpleNamespace(uvc_crop_mode='fixed', detect_range_size=640)
    region = sc._compute_fixed_uvc_crop_region(config, capture_width=1920, capture_height=1080)

    assert region == {'left': 640, 'top': 220, 'width': 640, 'height': 640}


def test_compute_fixed_uvc_crop_region_none_when_dynamic():
    from core import screen_capture as sc

    config = SimpleNamespace(uvc_crop_mode='dynamic', detect_range_size=640)
    assert sc._compute_fixed_uvc_crop_region(config, 1920, 1080) is None


def test_compute_fixed_uvc_crop_region_none_when_detect_range_size_missing():
    from core import screen_capture as sc

    config = SimpleNamespace(uvc_crop_mode='fixed', detect_range_size=0)
    assert sc._compute_fixed_uvc_crop_region(config, 1920, 1080) is None


def test_compute_fixed_uvc_crop_region_clamped_to_capture_size():
    """detect_range_size larger than the actual capture resolution must be
    clamped, not produce a crop rect that doesn't fit inside the frame."""
    from core import screen_capture as sc

    config = SimpleNamespace(uvc_crop_mode='fixed', detect_range_size=4000)
    region = sc._compute_fixed_uvc_crop_region(config, capture_width=1920, capture_height=1080)

    assert region == {'left': 420, 'top': 0, 'width': 1080, 'height': 1080}


def _wait_until(predicate, timeout=2.0, interval=0.01):
    import time
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


def test_native_dll_fixed_crop_requests_native_crop_not_a_resolution_swap(monkeypatch):
    """Fixed crop mode on the native DLL (v2, NV12) must open at the FULL
    requested resolution — device negotiation is unaffected — and pass the
    centered crop rect via the DLL's native crop_x/y/width/height fields
    instead. This is deliberately not "open at 640x640 instead of
    1920x1080": that resolution-swap trick (a prior version of this method)
    isn't a real crop guarantee — a device accepting a smaller mode carries
    no promise it's a spatial crop of the same framing."""
    from core import screen_capture as sc

    opened_with = []

    class FakeNative:
        def __init__(self, device_index, w, h, fps, pixel_format=0, buffer_count=4, crop=None):
            opened_with.append((w, h, crop))

        def start(self):
            pass

        def get_latest_frame(self):
            return None  # keeps the reader thread quiet for this test's brief lifetime

    monkeypatch.setattr('core.dshow_capture_native.NativeDshowCapture', FakeNative)

    config = SimpleNamespace(
        uvc_device_index=0, uvc_width=1920, uvc_height=1080, uvc_fps=60,
        uvc_show_window=False, uvc_video_format='nv12', uvc_crop_mode='fixed',
        detect_range_size=640, source_nominal_fps=0.0, uvc_actual_width=0,
        uvc_actual_height=0, uvc_actual_fps=0.0,
    )

    capture = object.__new__(sc.UVCCapture)
    capture.config = config
    capture.show_window = False
    capture._init_native_dll(0, 1920, 1080, 60)
    try:
        assert opened_with == [(1920, 1080, (640, 220, 640, 640))]
        assert capture.preview_width == 1920 and capture.preview_height == 1080
        # Optimistic default pending the reader thread's first-frame
        # confirmation — no frame has arrived yet in this test (FakeNative
        # returns None), so _fixed_region stays at its "assume it worked"
        # starting value.
        assert capture._fixed_region is None
    finally:
        capture._reader_stop.set()


def test_native_dll_fixed_crop_self_heals_when_dll_silently_ignores_crop(monkeypatch):
    """A DLL build that predates the native crop ABI addition doesn't
    reject the trailing crop_x/y/width/height fields — it just never reads
    them, so capture_open() succeeds and frames keep arriving at the full
    negotiated resolution. capture_open() returning OK is therefore not
    proof the crop actually happened; only the first real frame's actual
    dimensions are. This must self-heal to a software crop rather than
    silently feeding the full frame to detection as if it were the crop."""
    from core import screen_capture as sc

    class FakeNative:
        def __init__(self, device_index, w, h, fps, pixel_format=0, buffer_count=4, crop=None):
            pass

        def start(self):
            pass

        def get_latest_frame(self):
            # Full negotiated resolution, NOT the requested 640x640 crop —
            # simulates a DLL that silently ignored the crop request.
            h, w = 1080, 1920
            raw = bytes(h * 3 // 2 * w)
            return raw, w, h, 1

    monkeypatch.setattr('core.dshow_capture_native.NativeDshowCapture', FakeNative)

    config = SimpleNamespace(
        uvc_device_index=0, uvc_width=1920, uvc_height=1080, uvc_fps=60,
        uvc_show_window=False, uvc_video_format='nv12', uvc_crop_mode='fixed',
        detect_range_size=640, source_nominal_fps=0.0, uvc_actual_width=0,
        uvc_actual_height=0, uvc_actual_fps=0.0,
    )

    capture = object.__new__(sc.UVCCapture)
    capture.config = config
    capture.show_window = False
    capture._init_native_dll(0, 1920, 1080, 60)
    try:
        assert _wait_until(lambda: capture._native_crop_confirmed)
        assert capture._fixed_region == {'left': 640, 'top': 220, 'width': 640, 'height': 640}
        assert capture.preview_width == 1920 and capture.preview_height == 1080
    finally:
        capture._reader_stop.set()


def test_native_dll_fixed_crop_falls_back_to_software_crop_when_open_rejects_crop(monkeypatch):
    """If capture_open() itself rejects the crop request (e.g.
    DSC_ERR_INVALID_ARG/DSC_ERR_CROP_NOT_SUPPORTED), _init_native_dll must
    retry without a crop request (still at the full resolution — never a
    different size) and fall back to the pre-existing software-crop
    behavior rather than letting the whole backend fail (→ MSS)."""
    from core import screen_capture as sc

    opened_with = []

    class FakeNative:
        def __init__(self, device_index, w, h, fps, pixel_format=0, buffer_count=4, crop=None):
            opened_with.append((w, h, crop))
            if crop is not None:
                raise RuntimeError('native crop is not supported for this session')

        def start(self):
            pass

        def get_latest_frame(self):
            return None

    monkeypatch.setattr('core.dshow_capture_native.NativeDshowCapture', FakeNative)

    config = SimpleNamespace(
        uvc_device_index=0, uvc_width=1920, uvc_height=1080, uvc_fps=60,
        uvc_show_window=False, uvc_video_format='nv12', uvc_crop_mode='fixed',
        detect_range_size=640, source_nominal_fps=0.0, uvc_actual_width=0,
        uvc_actual_height=0, uvc_actual_fps=0.0,
    )

    capture = object.__new__(sc.UVCCapture)
    capture.config = config
    capture.show_window = False
    capture._init_native_dll(0, 1920, 1080, 60)
    try:
        # Crop attempt first (still at the full resolution), then the
        # no-crop retry — never a smaller-resolution open at any point.
        assert opened_with == [(1920, 1080, (640, 220, 640, 640)), (1920, 1080, None)]
        assert capture.preview_width == 1920 and capture.preview_height == 1080
        assert capture._fixed_region == {'left': 640, 'top': 220, 'width': 640, 'height': 640}
    finally:
        capture._reader_stop.set()


def test_native_dll_fixed_crop_skips_native_crop_for_mjpeg(monkeypatch):
    """Native crop is NV12-only — for MJPEG, _init_native_dll must not even
    attempt it (the DLL's own contract rejects that combination with
    DSC_ERR_CROP_NOT_SUPPORTED), going straight to the pre-existing
    software crop-after-decode path instead."""
    from core import screen_capture as sc

    opened_with = []

    class FakeNative:
        def __init__(self, device_index, w, h, fps, pixel_format=0, buffer_count=4, crop=None):
            opened_with.append((w, h, crop))

        def start(self):
            pass

        def get_latest_frame(self):
            return None

    monkeypatch.setattr('core.dshow_capture_native.NativeDshowCapture', FakeNative)

    config = SimpleNamespace(
        uvc_device_index=0, uvc_width=1920, uvc_height=1080, uvc_fps=60,
        uvc_show_window=False, uvc_video_format='mjpeg', uvc_crop_mode='fixed',
        detect_range_size=640, source_nominal_fps=0.0, uvc_actual_width=0,
        uvc_actual_height=0, uvc_actual_fps=0.0,
    )

    capture = object.__new__(sc.UVCCapture)
    capture.config = config
    capture.show_window = False
    capture._init_native_dll(0, 1920, 1080, 60)
    try:
        assert opened_with == [(1920, 1080, None)]
        assert capture._fixed_region == {'left': 640, 'top': 220, 'width': 640, 'height': 640}
    finally:
        capture._reader_stop.set()


def test_wait_for_receiver_connection_succeeds_after_connect(monkeypatch):
    from core import screen_capture as sc

    class FakeVideoFrame:
        xres = 0

    class FakeFrameSync:
        def __init__(self, vf):
            self.vf = vf
            self.calls = 0

        def capture_video(self):
            self.calls += 1
            if self.calls >= 2:
                self.vf.xres = 1920

    class FakeReceiver:
        def __init__(self):
            self.calls = 0

        def is_connected(self):
            self.calls += 1
            return self.calls >= 2

    monkeypatch.setattr(sc.time, 'sleep', lambda _: None)
    vf = FakeVideoFrame()
    frame_sync = FakeFrameSync(vf)
    receiver = FakeReceiver()

    ok = sc._wait_for_receiver_connection(
        receiver,
        frame_sync,
        vf,
        None,
        None,
        attempts=5,
        interval_seconds=0.0,
    )

    assert ok is True


# ── _draw_detection_overlay: rectangular / elliptical FOV ──────────────────
#
# The Qt in-game overlay (overlay.py) can't render over a UVC/NDI/UDP source
# the way it does over the real desktop, so those three backends bake FOV/
# box annotations straight into the preview frame via this shared function
# instead. It used to only read fov_size (a square, or a plain circle) —
# fov_height support landed everywhere else (the real filter in
# ai_loop_utils.py, the Qt overlay, the Web ESP client) but was missed here,
# which is exactly why a rectangular FOV only appeared to work for
# dxcam/mss: those are the only two backends that go through overlay.py
# instead of this function.

def _make_overlay_config(**overrides):
    base = dict(
        AimToggle=True,
        crosshairX=500,
        crosshairY=500,
        show_detect_range=False,
        show_fov=True,
        fov_size=200,
        fov_height=200,
        fov_circle_filter_enabled=False,
        show_boxes=False,
        latest_boxes=[],
        latest_confidences=[],
        show_confidence=False,
        box_color_theme='default',
        chroma_box_speed=0.0,
        box_full_rect=False,
        show_tracer_line=False,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_draw_detection_overlay_fov_corners_use_independent_width_and_height(monkeypatch):
    from core import screen_capture as sc

    lines = []
    monkeypatch.setattr(sc.cv2, 'line', lambda *a, **k: lines.append(a))

    config = _make_overlay_config(fov_size=200, fov_height=400)  # width 200, height 400
    frame = np.zeros((1000, 1000, 4), dtype=np.uint8)
    sc._draw_detection_overlay(frame, None, config, has_alpha=True)

    xs = [pt[0] for call in lines for pt in (call[1], call[2])]
    ys = [pt[1] for call in lines for pt in (call[1], call[2])]
    assert min(xs) == 400 and max(xs) == 600, "width=200 -> crosshair (500) +/- 100"
    assert min(ys) == 300 and max(ys) == 700, "height=400 -> crosshair (500) +/- 200"


def test_draw_detection_overlay_fov_corners_shrink_with_fov_effective_size(monkeypatch):
    """"Reduce FOV on Active Target" (ai_loop.py) writes fov_effective_size/
    _height every frame; this baked-in preview overlay must draw THOSE, not
    the static fov_size/fov_height, while a target is locked and the shrink
    window is active."""
    from core import screen_capture as sc

    lines = []
    monkeypatch.setattr(sc.cv2, 'line', lambda *a, **k: lines.append(a))

    config = _make_overlay_config(
        fov_size=200, fov_height=400,          # the user's real configured FOV
        fov_effective_size=100, fov_effective_height=200,  # mid-engagement shrink to 50%
    )
    frame = np.zeros((1000, 1000, 4), dtype=np.uint8)
    sc._draw_detection_overlay(frame, None, config, has_alpha=True)

    xs = [pt[0] for call in lines for pt in (call[1], call[2])]
    ys = [pt[1] for call in lines for pt in (call[1], call[2])]
    assert min(xs) == 450 and max(xs) == 550, "shrunk width=100 -> crosshair (500) +/- 50"
    assert min(ys) == 400 and max(ys) == 600, "shrunk height=200 -> crosshair (500) +/- 100"


def test_draw_detection_overlay_fov_corners_fall_back_when_no_effective_size(monkeypatch):
    """A config predating fov_effective_size/_height (or before ai_loop.py's
    first frame writes them) must draw the plain configured FOV, unchanged
    from before this feature existed."""
    from core import screen_capture as sc

    lines = []
    monkeypatch.setattr(sc.cv2, 'line', lambda *a, **k: lines.append(a))

    config = _make_overlay_config(fov_size=200, fov_height=400)  # no fov_effective_* at all
    frame = np.zeros((1000, 1000, 4), dtype=np.uint8)
    sc._draw_detection_overlay(frame, None, config, has_alpha=True)

    xs = [pt[0] for call in lines for pt in (call[1], call[2])]
    ys = [pt[1] for call in lines for pt in (call[1], call[2])]
    assert min(xs) == 400 and max(xs) == 600
    assert min(ys) == 300 and max(ys) == 700


def test_draw_detection_overlay_ellipse_axes_match_width_and_height(monkeypatch):
    """fov_circle_filter_enabled must draw a true ellipse (independent
    semi-axes), not a circle keyed off fov_size alone."""
    from core import screen_capture as sc

    ellipse_calls = []
    monkeypatch.setattr(sc.cv2, 'ellipse', lambda *a, **k: ellipse_calls.append(a))

    config = _make_overlay_config(fov_size=200, fov_height=400, fov_circle_filter_enabled=True)
    frame = np.zeros((1000, 1000, 4), dtype=np.uint8)
    sc._draw_detection_overlay(frame, None, config, has_alpha=True)

    assert len(ellipse_calls) == 1
    axes = ellipse_calls[0][2]
    assert axes == (100, 200), "semi-axes must be half of (fov_size, fov_height)"


def test_draw_detection_overlay_box_highlight_reaches_the_taller_dimension(monkeypatch):
    """A box beyond a square/circle's reach vertically, but within the
    rectangle's greater height, must still be recognized as in-FOV (drawn
    with the chroma color) once fov_height supplies that extra reach."""
    from core import screen_capture as sc

    rects = []
    monkeypatch.setattr(sc.cv2, 'rectangle', lambda *a, **k: rects.append(a))

    # Box centered ~100px below crosshair on Y, well within +/-2px on X.
    box = [498, 598, 502, 602]
    config = _make_overlay_config(
        crosshairX=500, crosshairY=500,
        fov_size=50, fov_height=250,  # half_x=25 (misses), half_y=125 (reaches)
        fov_circle_filter_enabled=True,
        show_boxes=True, latest_boxes=[box], latest_confidences=[0.9],
        chroma_box_speed=1.0, box_full_rect=True,
    )
    frame = np.zeros((1000, 1000, 4), dtype=np.uint8)

    sc._draw_detection_overlay(frame, None, config, has_alpha=True)
    assert len(rects) == 1
    drawn_color = rects[0][3]
    # The theme default color (BGR 0,255,0 + alpha) would be drawn if the
    # box were judged out-of-FOV; anything else means the chroma color won,
    # i.e. the box was correctly recognized as in-FOV.
    assert drawn_color != (0, 255, 0, 220)


def test_draw_detection_overlay_tracer_line_respects_ellipse_height(monkeypatch):
    from core import screen_capture as sc

    lines = []
    monkeypatch.setattr(sc.cv2, 'line', lambda *a, **k: lines.append(a))

    box = [495, 595, 505, 605]  # midpoint (500, 600) -> 100px below crosshair
    config = _make_overlay_config(
        crosshairX=500, crosshairY=500,
        fov_size=50, fov_height=250,  # half_x=25, half_y=125
        fov_circle_filter_enabled=True,
        show_fov=False, show_tracer_line=True, latest_boxes=[box],
    )
    frame = np.zeros((1000, 1000, 4), dtype=np.uint8)
    sc._draw_detection_overlay(frame, None, config, has_alpha=True)

    assert len(lines) == 1, "the tracer line must reach a box within the ellipse's height, not just its width"
