"""NVIDIA GPU sensor reader for the Status Panel's GPU section.

Pure Python, no Qt: status_panel.py's GpuMonitorThread owns one instance and
calls read() on its own thread. Uses NVML through nvidia-ml-py (module
``pynvml``), imported lazily so a missing package only disables the GPU
section ("Install NVML.bat" installs it into the embedded Python).

Sensor keys (each has a status_panel_gpu_<key> visibility switch):
    temp, hotspot, util, vram, fan, core_clock, mem_clock, power, voltage

A sensor whose first read fails (NVML reports it unsupported, or the
installed pynvml has no call for it) is added to ``unsupported`` and never
polled again, so the panel can hide its row. A later failure on a sensor that
already worked is treated as transient: that tick reads None.

NVML has no public reading for hotspot temperature or core voltage on
GeForce cards, so those two are always unsupported here; their rows and
switches exist so the panel lays out the same for every sensor.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

SENSOR_KEYS = (
    "temp", "hotspot", "util", "vram", "fan",
    "core_clock", "mem_clock", "power", "voltage",
)

STATUS_NOT_STARTED = "not_started"
STATUS_NO_NVML = "no_nvml"
STATUS_NO_GPU = "no_gpu"
STATUS_OK = "ok"


def nvml_installed() -> bool:
    """True if nvidia-ml-py (module ``pynvml``) can be imported, without importing it."""
    import importlib.util
    try:
        return importlib.util.find_spec("pynvml") is not None
    except (ImportError, ValueError):
        return False


class GpuMonitor:
    """Reads NVML sensors for GPU 0."""

    def __init__(self) -> None:
        self.status = STATUS_NOT_STARTED
        self.gpu_name = ""
        self.unsupported: set[str] = set()
        self._working: set[str] = set()
        self._nvml: Any = None
        self._handle: Any = None

    @property
    def supported(self) -> set[str]:
        """Sensors not (yet) known to be unsupported."""
        return set(SENSOR_KEYS) - self.unsupported

    def start(self) -> str:
        try:
            import pynvml
        except ImportError:
            self.status = STATUS_NO_NVML
            return self.status
        try:
            pynvml.nvmlInit()
        except Exception as exc:  # NVMLError, or the driver library is missing
            logger.info("[GPU] NVML init failed: %s", exc)
            self.status = STATUS_NO_GPU
            return self.status
        try:
            if pynvml.nvmlDeviceGetCount() < 1:
                raise RuntimeError("no NVIDIA GPU")
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            name = pynvml.nvmlDeviceGetName(self._handle)
            self.gpu_name = name.decode(errors="replace") if isinstance(name, bytes) else str(name)
        except Exception as exc:
            logger.info("[GPU] No usable NVIDIA GPU: %s", exc)
            self._shutdown(pynvml)
            self.status = STATUS_NO_GPU
            return self.status
        self._nvml = pynvml
        self.unsupported = {"hotspot", "voltage"}
        self.status = STATUS_OK
        return self.status

    def stop(self) -> None:
        if self._nvml is not None:
            self._shutdown(self._nvml)
        self._nvml = None
        self._handle = None
        self.status = STATUS_NOT_STARTED

    @staticmethod
    def _shutdown(nvml: Any) -> None:
        try:
            nvml.nvmlShutdown()
        except Exception:
            pass

    def read(self, keys=SENSOR_KEYS) -> dict[str, Any]:
        """Read the requested sensors. Unsupported or failed ones are None."""
        out: dict[str, Any] = {key: None for key in keys}
        if self.status != STATUS_OK:
            return out
        for key in keys:
            if key in self.unsupported:
                continue
            reader = getattr(self, f"_read_{key}", None)
            try:
                value = reader() if reader is not None else None
            except Exception as exc:
                if key not in self._working:
                    logger.info("[GPU] %s not supported: %s", key, exc)
                    self.unsupported.add(key)
                continue
            if value is None:
                if key not in self._working:
                    self.unsupported.add(key)
                continue
            self._working.add(key)
            out[key] = value
        return out

    # -- one NVML call per sensor ------------------------------------------
    def _read_temp(self):
        n = self._nvml
        return float(n.nvmlDeviceGetTemperature(self._handle, n.NVML_TEMPERATURE_GPU))

    def _read_util(self):
        return float(self._nvml.nvmlDeviceGetUtilizationRates(self._handle).gpu)

    def _read_vram(self):
        mem = self._nvml.nvmlDeviceGetMemoryInfo(self._handle)
        return (int(mem.used), int(mem.total))

    def _read_fan(self):
        """(rpm or None, percent or None); None when neither is available."""
        n = self._nvml
        rpm = pct = None
        try:
            pct = float(n.nvmlDeviceGetFanSpeed(self._handle))
        except Exception:
            pass
        get_rpm = getattr(n, "nvmlDeviceGetFanSpeedRPM", None)
        if get_rpm is not None:
            try:
                result = get_rpm(self._handle)
                rpm = float(getattr(result, "speed", result))
            except Exception:
                pass
        if rpm is None and pct is None:
            return None
        return (rpm, pct)

    def _read_core_clock(self):
        n = self._nvml
        return float(n.nvmlDeviceGetClockInfo(self._handle, n.NVML_CLOCK_GRAPHICS))

    def _read_mem_clock(self):
        n = self._nvml
        return float(n.nvmlDeviceGetClockInfo(self._handle, n.NVML_CLOCK_MEM))

    def _read_power(self):
        return self._nvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0


# -- compact value text for the panel rows ----------------------------------

def short_gpu_name(name: str, limit: int = 25) -> str:
    """Drop the vendor/brand prefix ("NVIDIA GeForce RTX 3060 Ti" ->
    "RTX 3060 Ti") so the name fits the panel; truncate only if still long."""
    short = name.strip()
    for prefix in ("NVIDIA ", "GeForce "):
        if short.startswith(prefix):
            short = short[len(prefix):]
    return short if len(short) <= limit else short[:limit - 1] + "…"

def format_reading(key: str, value: Any) -> str:
    if value is None:
        return "N/A"
    if key in ("temp", "hotspot"):
        return f"{value:.0f} °C"
    if key == "util":
        return f"{value:.0f} %"
    if key == "vram":
        used, total = value
        return f"{used / 1024 ** 3:.1f} / {total / 1024 ** 3:.1f} GB"
    if key == "fan":
        rpm, pct = value
        parts = []
        if rpm is not None:
            parts.append(f"{rpm:.0f} RPM")
        if pct is not None:
            parts.append(f"{pct:.0f} %")
        return " · ".join(parts) if parts else "N/A"
    if key in ("core_clock", "mem_clock"):
        return f"{value:.0f} MHz"
    if key == "power":
        return f"{value:.0f} W"
    if key == "voltage":
        return f"{value:.3f} V"
    return str(value)
