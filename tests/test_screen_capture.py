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


def test_native_dll_fixed_crop_region_is_set_on_init(monkeypatch):
    """Regression test: uvc_dshow_backend == 'v2' must apply the same
    centered fixed-crop rect the cv2 (v1) path does — it used to stay None
    for this backend (early-return skipped the computation entirely),
    silently behaving like 'dynamic' and letting grab() crop whatever the
    live region's top-left corner happened to be instead of a centered
    square."""
    from core import screen_capture as sc

    class FakeNative:
        def __init__(self, *a, **kw):
            pass

        def start(self):
            pass

        def get_latest_frame(self):
            return None  # keeps the reader thread quiet for this test's brief lifetime

    monkeypatch.setattr(
        'core.dshow_capture_native.NativeDshowCapture', lambda *a, **kw: FakeNative()
    )

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
