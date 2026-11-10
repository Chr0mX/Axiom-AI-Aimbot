"""NVIDIA GPU sensor reader for the Status Panel's GPU section.

Pure Python, no Qt: status_panel.py's GpuMonitorThread owns one instance and
calls read() on its own thread. Uses NVML through nvidia-ml-py (module
``pynvml``, vendored in src/python/dependencies), imported lazily so a
missing module only disables the GPU section.

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
import threading
import time
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

SENSOR_KEYS = (
    "temp", "hotspot", "util", "vram", "fan",
    "core_clock", "mem_clock", "power", "voltage",
)

STATUS_NOT_STARTED = "not_started"
STATUS_NO_NVML = "no_nvml"
STATUS_NO_GPU = "no_gpu"
STATUS_OK = "ok"
STATUS_CHECKING = "checking"      # GpuSampler only: started, no reading yet


def nvml_installed() -> bool:
    """True if the vendored nvidia-ml-py (module ``pynvml``) can be imported, without importing it."""
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


# -- Web Control sampler ------------------------------------------------------

class GpuSampler:
    """Lazy, demand-gated GPU sampler for the Web Control status (no Qt).

    The Qt panel's ``GpuMonitorThread`` only runs while that panel is visible, so Web Control
    cannot rely on its values.  This owns its *own* ``GpuMonitor`` on a daemon thread that

    * starts on the first ``snapshot()`` call (never inside a request handler: NVML init
      happens on the sampler thread),
    * reads only the sensors whose ``status_panel_gpu_<key>`` switch is on and that are
      supported, every ``status_panel_gpu_interval_ms`` (clamped 250-5000 ms),
    * stops itself (releasing NVML) after ``idle_stop_s`` without a ``snapshot()`` call, and
    * stays stopped for terminal states (no NVML / no GPU) -- nothing to poll.

    NVML ref-counts ``nvmlInit``, so this coexists with the Qt panel's own monitor.
    """

    def __init__(self, monitor_factory: Callable[[], GpuMonitor] = GpuMonitor,
                 clock: Callable[[], float] = time.monotonic,
                 idle_stop_s: float = 10.0, min_interval_s: float = 0.25) -> None:
        self._factory = monitor_factory
        self._clock = clock
        self._idle_stop_s = idle_stop_s
        self._min_interval_s = min_interval_s
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_demand = 0.0
        self._config: Any = None
        self._terminal = False
        self._state: dict[str, Any] = self._empty(STATUS_NOT_STARTED)

    @staticmethod
    def _empty(status: str, name: str = "") -> dict[str, Any]:
        return {"status": status, "name": name, "short_name": short_gpu_name(name) if name else "",
                "supported": [], "readings": {}, "text": {}}

    def snapshot(self, config: Any) -> dict[str, Any]:
        """Latest state; marks demand and starts the sampler thread if it is not running."""
        with self._lock:
            self._config = config
            self._last_demand = self._clock()
            if not self._terminal and (self._thread is None or not self._thread.is_alive()):
                if self._state["status"] in (STATUS_NOT_STARTED, STATUS_CHECKING) or not self._state["supported"]:
                    self._state = self._empty(STATUS_CHECKING)
                self._wake.clear()
                self._thread = threading.Thread(target=self._run, name="gpu-sampler", daemon=True)
                self._thread.start()
            return dict(self._state)

    def stop(self) -> None:
        self._wake.set()
        t = self._thread
        if t is not None and t.is_alive() and t is not threading.current_thread():
            t.join(timeout=2.0)

    def _publish(self, state: dict[str, Any]) -> None:
        with self._lock:
            self._state = state
            config = self._config
        if config is not None:
            try:   # lets the Visuals hint stop saying "waiting for the status panel"
                config.gpu_sensor_support = {"status": state["status"], "name": state["name"],
                                             "supported": list(state["supported"])}
            except Exception:
                pass

    def _run(self) -> None:
        monitor = self._factory()
        try:
            status = monitor.start()
            while True:
                config = self._config
                if status != STATUS_OK:
                    self._terminal = True
                    self._publish(self._empty(status))
                    return
                keys = [k for k in SENSOR_KEYS
                        if k not in monitor.unsupported
                        and getattr(config, f"status_panel_gpu_{k}", True)]
                readings = monitor.read(keys)
                text = {k: format_reading(k, v) for k, v in readings.items()}
                self._publish({
                    "status": STATUS_OK, "name": monitor.gpu_name,
                    "short_name": short_gpu_name(monitor.gpu_name),
                    "supported": sorted(monitor.supported),
                    "readings": {k: v for k, v in readings.items() if v is not None},
                    "text": text,
                })
                interval = getattr(config, "status_panel_gpu_interval_ms", 1000)
                try:
                    wait = max(self._min_interval_s, min(5000, int(interval)) / 1000.0)
                except (TypeError, ValueError):
                    wait = 1.0
                if self._wake.wait(wait):
                    return
                if self._clock() - self._last_demand > self._idle_stop_s:
                    return                      # nobody is polling: release NVML
        finally:
            monitor.stop()


_sampler: Optional[GpuSampler] = None
_sampler_lock = threading.Lock()


def get_sampler() -> GpuSampler:
    """Process-wide sampler used by web_control_server._build_status()."""
    global _sampler
    with _sampler_lock:
        if _sampler is None:
            _sampler = GpuSampler()
        return _sampler
