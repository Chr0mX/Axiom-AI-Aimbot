"""Demand gating in screen_capture: the preview thread only draws while
someone reads preview frames, and the uvc/udp readers only decode while a
frame has a consumer (a recent grab() or a preview reader).

cv2/mss are stubbed so this runs without OpenCV; nothing here needs a real
decode.
"""
import importlib
import sys
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest


@pytest.fixture
def sc(monkeypatch):
    had_module = 'core.screen_capture' in sys.modules
    if not had_module:
        monkeypatch.setitem(sys.modules, 'cv2', MagicMock())
        monkeypatch.setitem(sys.modules, 'mss', MagicMock())
    module = importlib.import_module('core.screen_capture')
    monkeypatch.setattr(module, '_preview_last_read', [float('-inf')])
    yield module
    if not had_module:
        # Unload the stubbed-cv2 copy so other tests import (or skip) for real.
        sys.modules.pop('core.screen_capture', None)
        if hasattr(sys.modules.get('core'), 'screen_capture'):
            delattr(sys.modules['core'], 'screen_capture')


def test_preview_wanted_follows_reads(sc):
    assert not sc.preview_wanted()
    sc.get_preview_frame()
    assert sc.preview_wanted()
    sc._preview_last_read[0] = time.perf_counter() - sc._PREVIEW_DEMAND_HOLD_S - 0.1
    assert not sc.preview_wanted()


def test_decode_wanted_by_recent_grab_or_preview(sc):
    now = time.perf_counter()
    assert sc._decode_wanted(now)
    assert not sc._decode_wanted(now - sc._DECODE_DEMAND_HOLD_S - 0.1)
    sc.get_preview_frame()
    assert sc._decode_wanted(float('-inf'))


class _OneFrameReceiver:
    """Delivers one JPEG, then stops the reader loop."""

    def __init__(self, capture):
        self._capture = capture
        self._sent = False
        self.recv_fps = 0.0
        self.dropped_fps = 0.0

    def get_latest_frame_with_id(self, block=True, timeout=0.1):
        if not self._sent:
            self._sent = True
            return b'not-a-real-jpeg', 1
        self._capture._stop.set()
        return None, None


def _udp_capture(sc, last_grab_t):
    capture = object.__new__(sc.UdpCapture)
    capture.config = SimpleNamespace(source_nominal_fps=0.0)
    capture._latest_frame_lock = threading.Lock()
    capture._latest_frame_ref = [np.zeros((4, 4, 3), dtype=np.uint8)]  # a stale frame
    capture._stop = threading.Event()
    capture._last_frame_perf_time = 0.0
    capture._last_grab_t = last_grab_t
    capture.preview_width = 4
    capture.preview_height = 4
    capture._receiver = _OneFrameReceiver(capture)
    return capture


def test_udp_reader_skips_decode_without_consumer(sc, monkeypatch):
    imdecode = MagicMock(return_value=np.zeros((8, 8, 3), dtype=np.uint8))
    monkeypatch.setattr(sc.cv2, 'imdecode', imdecode)
    capture = _udp_capture(sc, last_grab_t=float('-inf'))

    capture._reader_worker()

    imdecode.assert_not_called()
    # The stale frame is dropped so grab() can't serve it after a pause...
    assert capture._latest_frame_ref[0] is None
    # ...while the staleness clock still sees the stream as alive.
    assert capture._last_frame_perf_time > 0.0


def test_udp_reader_decodes_while_grabbed(sc, monkeypatch):
    decoded = np.zeros((8, 8, 3), dtype=np.uint8)
    imdecode = MagicMock(return_value=decoded)
    monkeypatch.setattr(sc.cv2, 'imdecode', imdecode)
    capture = _udp_capture(sc, last_grab_t=time.perf_counter())
    capture.config.udp_width = 0
    capture.config.udp_height = 0

    capture._reader_worker()

    imdecode.assert_called_once()
    assert capture._latest_frame_ref[0] is decoded
    assert (capture.config.udp_width, capture.config.udp_height) == (8, 8)


def test_preview_thread_draws_only_while_wanted(sc):
    draws = []
    stop = threading.Event()
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    thread = sc._UVCPreviewThread(
        frame_lock=threading.Lock(),
        frame_ref=[frame],
        stop_event=stop,
        draw_overlay_fn=lambda f, region: draws.append(1) or f,
        region_ref=[None],
        target_fps=200,
    )
    thread.start()
    try:
        time.sleep(0.15)
        assert draws == []  # nobody reading preview frames

        deadline = time.perf_counter() + 2.0
        while not draws and time.perf_counter() < deadline:
            sc.get_preview_frame()
            time.sleep(0.01)
        assert draws
    finally:
        stop.set()
        thread.join(timeout=1.0)
