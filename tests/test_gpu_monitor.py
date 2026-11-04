"""core/gpu_monitor.py against a fake pynvml (no GPU or driver needed)."""
import sys
import types

import pytest

from core import gpu_monitor as gm


class NVMLError(Exception):
    pass


def _fake_nvml(**overrides):
    calls = {"shutdown": 0, "power": 0}
    mod = types.ModuleType("pynvml")
    mod.NVMLError = NVMLError
    mod.NVML_TEMPERATURE_GPU = 0
    mod.NVML_CLOCK_GRAPHICS = 0
    mod.NVML_CLOCK_MEM = 2
    mod.nvmlInit = lambda: None

    def shutdown():
        calls["shutdown"] += 1
    mod.nvmlShutdown = shutdown
    mod.nvmlDeviceGetCount = lambda: 1
    mod.nvmlDeviceGetHandleByIndex = lambda i: "h0"
    mod.nvmlDeviceGetName = lambda h: b"NVIDIA GeForce RTX 3060 Ti"
    mod.nvmlDeviceGetTemperature = lambda h, s: 62
    mod.nvmlDeviceGetUtilizationRates = lambda h: types.SimpleNamespace(gpu=45)
    mod.nvmlDeviceGetMemoryInfo = lambda h: types.SimpleNamespace(
        used=int(1.2 * 1024 ** 3), total=8 * 1024 ** 3)
    mod.nvmlDeviceGetFanSpeed = lambda h: 55
    mod.nvmlDeviceGetFanSpeedRPM = lambda h: types.SimpleNamespace(speed=1650)
    mod.nvmlDeviceGetClockInfo = lambda h, c: 1905 if c == 0 else 7001

    def power(h):
        calls["power"] += 1
        return 135000
    mod.nvmlDeviceGetPowerUsage = power
    for name, value in overrides.items():
        setattr(mod, name, value)
    return mod, calls


@pytest.fixture
def install(monkeypatch):
    def _install(mod):
        monkeypatch.setitem(sys.modules, "pynvml", mod)
        return mod
    return _install


def test_no_nvml_when_package_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "pynvml", None)  # import raises ImportError
    mon = gm.GpuMonitor()
    assert mon.start() == gm.STATUS_NO_NVML
    assert all(v is None for v in mon.read().values())


def test_no_gpu_when_init_fails(install):
    def boom():
        raise NVMLError("driver not loaded")
    mod, _ = _fake_nvml(nvmlInit=boom)
    install(mod)
    assert gm.GpuMonitor().start() == gm.STATUS_NO_GPU


def test_no_gpu_when_no_device(install):
    mod, calls = _fake_nvml(nvmlDeviceGetCount=lambda: 0)
    install(mod)
    assert gm.GpuMonitor().start() == gm.STATUS_NO_GPU
    assert calls["shutdown"] == 1


def test_reads_and_formats_every_sensor(install):
    mod, _ = _fake_nvml()
    install(mod)
    mon = gm.GpuMonitor()
    assert mon.start() == gm.STATUS_OK
    assert mon.gpu_name == "NVIDIA GeForce RTX 3060 Ti"
    r = mon.read()
    text = {k: gm.format_reading(k, v) for k, v in r.items() if v is not None}
    assert text == {
        "temp": "62 °C", "util": "45 %", "vram": "1.2 / 8.0 GB",
        "fan": "1650 RPM · 55 %", "core_clock": "1905 MHz",
        "mem_clock": "7001 MHz", "power": "135 W",
    }
    # NVML has no hotspot/voltage reading on GeForce.
    assert r["hotspot"] is None and r["voltage"] is None
    assert {"hotspot", "voltage"} <= mon.unsupported


def test_unsupported_sensor_is_never_polled_again(install):
    calls = {"n": 0}

    def no_power(h):
        calls["n"] += 1
        raise NVMLError("Not Supported")
    mod, _ = _fake_nvml(nvmlDeviceGetPowerUsage=no_power)
    install(mod)
    mon = gm.GpuMonitor()
    mon.start()
    mon.read()
    mon.read()
    assert calls["n"] == 1
    assert "power" in mon.unsupported and "power" not in mon.supported


def test_transient_error_keeps_sensor_supported(install):
    state = {"fail": False}

    def flaky(h, s):
        if state["fail"]:
            raise NVMLError("GPU is lost")
        return 60
    mod, _ = _fake_nvml(nvmlDeviceGetTemperature=flaky)
    install(mod)
    mon = gm.GpuMonitor()
    mon.start()
    assert mon.read(["temp"])["temp"] == 60
    state["fail"] = True
    assert mon.read(["temp"])["temp"] is None
    assert "temp" in mon.supported


def test_only_requested_sensors_are_read(install):
    mod, calls = _fake_nvml()
    install(mod)
    mon = gm.GpuMonitor()
    mon.start()
    out = mon.read(["temp"])
    assert list(out) == ["temp"] and calls["power"] == 0


@pytest.mark.parametrize("rpm, pct, expected", [
    (1650.0, 55.0, "1650 RPM · 55 %"),
    (None, 55.0, "55 %"),
    (1650.0, None, "1650 RPM"),
])
def test_fan_formats_whatever_is_available(rpm, pct, expected):
    assert gm.format_reading("fan", (rpm, pct)) == expected


def test_fan_without_rpm_api(install):
    mod, _ = _fake_nvml()
    del mod.nvmlDeviceGetFanSpeedRPM
    install(mod)
    mon = gm.GpuMonitor()
    mon.start()
    assert gm.format_reading("fan", mon.read(["fan"])["fan"]) == "55 %"


def test_stop_shuts_nvml_down(install):
    mod, calls = _fake_nvml()
    install(mod)
    mon = gm.GpuMonitor()
    mon.start()
    mon.stop()
    assert calls["shutdown"] == 1
    assert mon.status == gm.STATUS_NOT_STARTED


def test_none_formats_as_na():
    assert gm.format_reading("temp", None) == "N/A"


# -- config & Web Control ---------------------------------------------------

def test_config_gpu_fields_map_and_clamp():
    from core import config as cfg
    assert cfg._FIELD_MAP["status_panel_show_gpu"] == "display.status_panel.gpu.enabled"
    for key in gm.SENSOR_KEYS:
        assert cfg._FIELD_MAP[f"status_panel_gpu_{key}"] == f"display.status_panel.gpu.{key}"
    c = types.SimpleNamespace(status_panel_gpu_interval_ms=50)
    cfg._validate_gpu_monitor(c)
    assert c.status_panel_gpu_interval_ms == 250
    c.status_panel_gpu_interval_ms = 99999
    cfg._validate_gpu_monitor(c)
    assert c.status_panel_gpu_interval_ms == 5000
    c.status_panel_gpu_interval_ms = "junk"
    cfg._validate_gpu_monitor(c)
    assert c.status_panel_gpu_interval_ms == 1000


def test_web_control_visuals_gpu_fields():
    from core import web_control_settings as wcs
    c = types.SimpleNamespace(status_panel_show_gpu=False, status_panel_gpu_temp=True,
                              status_panel_gpu_interval_ms=1000)
    got = wcs.get_tab_settings(c, "visuals")
    assert got["status_panel_show_gpu"] is False and got["status_panel_gpu_temp"] is True
    res = wcs.apply_tab_settings(c, "visuals", {"status_panel_show_gpu": True,
                                                "status_panel_gpu_interval_ms": 100})
    assert res["ok"] and c.status_panel_show_gpu is True
    assert c.status_panel_gpu_interval_ms == 250


# -- installed check ---------------------------------------------------------

def test_nvml_installed_reflects_find_spec(monkeypatch):
    import importlib.util
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    assert gm.nvml_installed() is False
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    assert gm.nvml_installed() is True


def test_web_control_gpu_status_line(monkeypatch):
    from core import web_control_settings as wcs
    monkeypatch.setattr(gm, "nvml_installed", lambda: False)
    c = types.SimpleNamespace(status_panel_show_gpu=True)
    assert wcs.gpu_monitor_status(c).startswith("NVML not installed")
    monkeypatch.setattr(gm, "nvml_installed", lambda: True)
    c.gpu_sensor_support = {"status": "ok", "name": "RTX 3060 Ti"}
    assert wcs.gpu_monitor_status(c) == "Reading RTX 3060 Ti."
    c.gpu_sensor_support = {"status": "no_gpu"}
    assert "no NVIDIA GPU" in wcs.gpu_monitor_status(c)
    assert "gpu_monitor_status" in wcs.get_tab_settings(c, "visuals")
