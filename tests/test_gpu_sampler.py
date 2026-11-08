"""GpuSampler (Web Control GPU stats): lazy start, per-sensor gating, idle stop, terminal states."""
import time
from types import SimpleNamespace

import pytest

from core import gpu_monitor as gm


class FakeMonitor:
    instances = []

    def __init__(self, status=gm.STATUS_OK, name="NVIDIA GeForce RTX 2070 SUPER"):
        self._status_after_start = status
        self.status = gm.STATUS_NOT_STARTED
        self.gpu_name = name
        self.unsupported = {"hotspot", "voltage"}
        self.read_keys = []
        self.stopped = False
        FakeMonitor.instances.append(self)

    @property
    def supported(self):
        return set(gm.SENSOR_KEYS) - self.unsupported

    def start(self):
        self.status = self._status_after_start
        return self.status

    def read(self, keys=gm.SENSOR_KEYS):
        self.read_keys.append(list(keys))
        values = {"temp": 65.0, "util": 40.0, "vram": (6 * 1024 ** 3, 8 * 1024 ** 3), "fan": (1500, 45.0),
                  "core_clock": 1900.0, "mem_clock": 7000.0, "power": 180.0}
        return {k: values.get(k) for k in keys}

    def stop(self):
        self.stopped = True


def cfg(**kw):
    base = dict(status_panel_show_gpu=True, status_panel_gpu_interval_ms=250)
    for k in gm.SENSOR_KEYS:
        base[f"status_panel_gpu_{k}"] = True
    base.update(kw)
    return SimpleNamespace(**base)


def wait_for(pred, timeout=3.0):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.01)
    return False


@pytest.fixture(autouse=True)
def _reset():
    FakeMonitor.instances = []
    yield


def make(**kw):
    kw.setdefault("min_interval_s", 0.01)
    return gm.GpuSampler(monitor_factory=kw.pop("factory", FakeMonitor), **kw)


def test_first_snapshot_is_checking_then_real_readings():
    s = make()
    c = cfg()
    first = s.snapshot(c)
    assert first["status"] == gm.STATUS_CHECKING and first["readings"] == {}
    assert wait_for(lambda: s.snapshot(c)["status"] == gm.STATUS_OK)
    snap = s.snapshot(c)
    assert snap["short_name"] == "RTX 2070 SUPER" and snap["name"].startswith("NVIDIA")
    # preformatted exactly like the Qt rows
    assert snap["text"]["temp"] == "65 °C" and snap["text"]["vram"] == "6.0 / 8.0 GB"
    assert snap["text"]["fan"] == "1500 RPM · 45 %" and snap["text"]["power"] == "180 W"
    s.stop()


def test_unsupported_sensors_are_never_reported_as_supported():
    s = make()
    c = cfg()
    s.snapshot(c)
    assert wait_for(lambda: s.snapshot(c)["status"] == gm.STATUS_OK)
    sup = s.snapshot(c)["supported"]
    assert "hotspot" not in sup and "voltage" not in sup and "temp" in sup
    s.stop()


def test_only_switched_on_sensors_are_read():
    s = make()
    c = cfg(status_panel_gpu_fan=False, status_panel_gpu_power=False)
    s.snapshot(c)
    assert wait_for(lambda: s.snapshot(c)["status"] == gm.STATUS_OK)
    keys = FakeMonitor.instances[0].read_keys[-1]
    assert "fan" not in keys and "power" not in keys and "temp" in keys
    assert "fan" not in s.snapshot(c)["text"]
    s.stop()


def test_terminal_status_is_published_and_never_restarted():
    s = make(factory=lambda: FakeMonitor(status=gm.STATUS_NO_NVML))
    c = cfg()
    s.snapshot(c)
    assert wait_for(lambda: s.snapshot(c)["status"] == gm.STATUS_NO_NVML)
    calls = len(FakeMonitor.instances)
    for _ in range(5):
        s.snapshot(c)
    time.sleep(0.05)
    assert len(FakeMonitor.instances) == calls == 1
    assert FakeMonitor.instances[0].stopped
    assert c.gpu_sensor_support["status"] == gm.STATUS_NO_NVML


def test_stops_itself_when_nobody_polls_and_restarts_on_demand():
    now = [100.0]
    s = make(clock=lambda: now[0], idle_stop_s=10.0)
    c = cfg()
    s.snapshot(c)
    assert wait_for(lambda: s.snapshot(c)["status"] == gm.STATUS_OK)
    first = FakeMonitor.instances[0]
    now[0] += 60.0                                       # a minute with no /api/status poll
    assert wait_for(lambda: first.stopped)               # NVML released
    assert wait_for(lambda: not s._thread.is_alive())
    s.snapshot(c)                                        # demand again
    assert wait_for(lambda: len(FakeMonitor.instances) == 2)
    assert wait_for(lambda: s.snapshot(c)["status"] == gm.STATUS_OK)
    s.stop()


def test_publishes_sensor_support_for_the_visuals_hint():
    s = make()
    c = cfg()
    s.snapshot(c)
    assert wait_for(lambda: getattr(c, "gpu_sensor_support", {}).get("status") == gm.STATUS_OK)
    assert c.gpu_sensor_support["name"].startswith("NVIDIA") and "temp" in c.gpu_sensor_support["supported"]
    s.stop()


def test_stop_joins_the_thread():
    s = make()
    c = cfg()
    s.snapshot(c)
    assert wait_for(lambda: s.snapshot(c)["status"] == gm.STATUS_OK)
    s.stop()
    assert not s._thread.is_alive() and FakeMonitor.instances[0].stopped


class TestStatusRoute:
    def test_gpu_block_only_when_the_section_is_on(self, monkeypatch):
        from core import web_control_server as wcs
        fake = SimpleNamespace(snapshot=lambda config: {"status": "ok", "marker": True})
        monkeypatch.setattr(gm, "get_sampler", lambda: fake)
        off = wcs._build_status(SimpleNamespace(status_panel_show_gpu=False))
        assert "gpu" not in off
        on = wcs._build_status(SimpleNamespace(status_panel_show_gpu=True))
        assert on["gpu"] == {"status": "ok", "marker": True}

    def test_sampler_failure_never_breaks_status(self, monkeypatch):
        from core import web_control_server as wcs

        def boom():
            raise RuntimeError("nvml exploded")
        monkeypatch.setattr(gm, "get_sampler", boom)
        st = wcs._build_status(SimpleNamespace(status_panel_show_gpu=True))
        assert "gpu" not in st and "running" in st
