"""Tests for core/auto_fire.py's handling of stale detection results.

auto_fire imports win_utils (win32api) at module scope, so the import is
deferred into a fixture — a missing win32api fails just these tests instead
of aborting collection for the whole suite.
"""

import queue
from types import SimpleNamespace

import pytest


@pytest.fixture
def auto_fire():
    from core import auto_fire
    return auto_fire


class _FakeClock:
    """Stands in for the `time` module inside auto_fire: sleep() advances the
    clock, and stops the loop once `stop_at` is reached."""

    def __init__(self, config, stop_at):
        self.now = 1000.0
        self._config = config
        self._stop_at = stop_at

    def time(self):
        return self.now

    def sleep(self, dt):
        self.now += dt
        if self.now >= self._stop_at:
            self._config.Running = False


def _config():
    return SimpleNamespace(
        Running=True,
        auto_fire_key=0x05,
        auto_fire_key2=None,
        always_auto_fire=True,
        auto_fire_delay=0.0,
        auto_fire_interval=0.1,
        auto_fire_target_part="both",
        head_height_ratio=0.2,
        head_width_ratio=0.5,
        body_width_ratio=0.8,
        crosshairX=50,
        crosshairY=50,
        mouse_click_method="mouse_event",
    )


def test_does_not_keep_firing_on_boxes_that_stopped_arriving(auto_fire, monkeypatch):
    """One detection arrives, then inference goes quiet (source died, or
    detection stopped). Before the fix auto-fire kept clicking on that single
    cached box list for as long as the trigger was held."""
    config = _config()
    clock = _FakeClock(config, stop_at=1003.0)
    clicks = []
    monkeypatch.setattr(auto_fire, "time", clock)
    monkeypatch.setattr(auto_fire, "is_key_pressed", lambda _k: False)
    monkeypatch.setattr(auto_fire, "send_mouse_click", lambda _m: clicks.append(clock.now))

    q = queue.Queue(maxsize=1)
    q.put([[0, 0, 100, 100]])  # covers the crosshair at (50, 50)
    auto_fire.auto_fire_loop(config, q)

    assert clicks, "should fire while the detection is fresh"
    assert max(clicks) - 1000.0 <= 1.0 + 0.1, "kept firing on a box list that stopped updating"


def test_keeps_firing_while_detections_keep_arriving(auto_fire, monkeypatch):
    config = _config()
    clock = _FakeClock(config, stop_at=1003.0)
    clicks = []
    q = queue.Queue(maxsize=1)

    def _sleep(dt):
        clock.sleep(dt)
        if q.empty():
            q.put([[0, 0, 100, 100]])

    monkeypatch.setattr(auto_fire, "time", SimpleNamespace(time=clock.time, sleep=_sleep))
    monkeypatch.setattr(auto_fire, "is_key_pressed", lambda _k: False)
    monkeypatch.setattr(auto_fire, "send_mouse_click", lambda _m: clicks.append(clock.now))

    q.put([[0, 0, 100, 100]])
    auto_fire.auto_fire_loop(config, q)

    assert max(clicks) - 1000.0 > 2.5
