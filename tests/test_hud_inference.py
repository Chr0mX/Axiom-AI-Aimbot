"""Tests for core/hud_inference.py's parent-side helpers (no child process)."""

import numpy as np

from core.hud_inference import _crop_roi


def _frame(h=1080, w=1920):
    return np.arange(h * w, dtype=np.uint32).reshape(h, w)


def test_roi_inside_frame():
    out = _crop_roi(_frame(), {"left": 1490, "top": 953, "width": 380, "height": 88}, [True])
    assert out.shape == (88, 380)


def test_roi_clamped_to_frame_edge():
    out = _crop_roi(_frame(), {"left": 1800, "top": 1000, "width": 380, "height": 88}, [True])
    assert out.shape == (80, 120)


def test_roi_outside_smaller_frame_is_none():
    """1080p HUD coords against a 640x640 cropped stream don't overlap it at
    all — must be skipped, not sent to the child as a zero-size array."""
    assert _crop_roi(_frame(640, 640), {"left": 1490, "top": 953, "width": 380, "height": 88}, [True]) is None


def test_negative_origin_does_not_wrap():
    """A negative bound is a from-the-end index in numpy; the crop must be
    the frame's top-left corner, not a slice counted from the far edge."""
    frame = _frame(100, 100)
    out = _crop_roi(frame, {"left": -10, "top": -10, "width": 30, "height": 30}, [True])
    assert out.shape == (20, 20)
    assert out[0, 0] == frame[0, 0]
