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
    assert output.count('[截圖] 抓圖失敗: capture failed') == 1


def test_find_ndi_source_by_name_accepts_stream_name_case_insensitive():
    from core import screen_capture as sc

    source = SimpleNamespace(name='OBS-PC', stream_name='MainFeed')

    class FakeFinder:
        def get_source(self, target):
            return None

        def __iter__(self):
            yield source

    assert sc._find_ndi_source_by_name(FakeFinder(), 'mainfeed') is source


def test_extract_ndi_source_video_meta_supports_rate_fraction():
    from core import screen_capture as sc

    source = SimpleNamespace(width=1920, height=1080, frame_rate_N=60000, frame_rate_D=1001)
    width, height, fps = sc._extract_ndi_source_video_meta(source)

    assert (width, height) == (1920, 1080)
    assert fps is not None
    assert abs(fps - 59.94) < 0.02
