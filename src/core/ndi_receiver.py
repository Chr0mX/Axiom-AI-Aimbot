from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

from core.ndi_config_loader import PipelineConfig

log = logging.getLogger(__name__)

_BACKOFF = [1.0, 2.0, 4.0, 8.0, 16.0]
_MAX_ATTEMPTS = 3 * len(_BACKOFF)  # ~90s total before RuntimeError


# ---------------------------------------------------------------------------
# Cyndilib helpers — embedded to avoid importing src/core/screen_capture.py,
# which has a hard top-level `import mss` that is not required for NDI.
# These functions are pure cyndilib / numpy logic with no extra dependencies.
# ---------------------------------------------------------------------------

def _load_cyndilib_symbols() -> dict[str, Any]:
    try:
        from cyndilib.finder import Finder  # type: ignore[import-not-found]
        from cyndilib.receiver import ReceiveFrameType, Receiver  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("cyndilib is not installed") from exc

    RecvColorFormat: Any
    RecvBandwidth: Any = None
    try:
        from cyndilib.wrapper.ndi_recv import (  # type: ignore[import-not-found]
            RecvBandwidth as _RB,
            RecvColorFormat as _RCF,
        )
        RecvColorFormat, RecvBandwidth = _RCF, _RB
    except ImportError:
        from cyndilib.wrapper import RecvColorFormat as _RCF  # type: ignore[import-not-found]
        RecvColorFormat = _RCF

    VideoFrameSync: Any = None
    VideoRecvFrame: Any = None
    try:
        from cyndilib.video_frame import VideoFrameSync as _VFS  # type: ignore[import-not-found]
        VideoFrameSync = _VFS
    except ImportError:
        try:
            from cyndilib import VideoRecvFrame as _VRF  # type: ignore[import-not-found]
            VideoRecvFrame = _VRF
        except ImportError:
            pass

    return {
        "Finder": Finder,
        "Receiver": Receiver,
        "ReceiveFrameType": ReceiveFrameType,
        "RecvColorFormat": RecvColorFormat,
        "RecvBandwidth": RecvBandwidth,
        "VideoFrameSync": VideoFrameSync,
        "VideoRecvFrame": VideoRecvFrame,
    }


def _extract_ndi_source_name(source: Any) -> str:
    if isinstance(source, str):
        return source.strip()
    for attr in ("name", "source_name", "stream_name", "url"):
        v = getattr(source, attr, None)
        if isinstance(v, str) and v.strip():
            return v.strip()
    try:
        return str(source).strip()
    except Exception:
        return ""


def _extract_ndi_stream_name(source: Any) -> str:
    for attr in ("stream_name", "ndi_name", "stream", "source_name", "name"):
        v = getattr(source, attr, None)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _find_ndi_source_by_name(finder: Any, target_name: str) -> Any | None:
    target = str(target_name or "").strip()
    if not target:
        return None
    t_lo = target.lower()
    try:
        s = finder.get_source(target)
        if s is not None:
            return s
    except Exception:
        pass
    try:
        for s in finder:
            full = _extract_ndi_source_name(s).lower()
            stream = _extract_ndi_stream_name(s).lower()
            if t_lo in {full, stream} or full.endswith(f"({t_lo})") or stream.endswith(f"({t_lo})"):
                return s
    except Exception:
        pass
    return None


def _wait_for_receiver_connection(
    receiver: Any,
    frame_sync: Any | None,
    video_frame_sync: Any | None,
    receive_fn: Any | None,
    receive_video_flag: Any | None,
    attempts: int = 30,
    interval_seconds: float = 0.1,
) -> bool:
    for _ in range(max(1, int(attempts))):
        try:
            if receiver.is_connected():
                if frame_sync is not None and video_frame_sync is not None:
                    frame_sync.capture_video()
                    if int(getattr(video_frame_sync, "xres", 0) or 0) > 0:
                        return True
                elif callable(receive_fn) and receive_video_flag is not None:
                    result = receive_fn(receive_video_flag, 100)
                    if result & receive_video_flag:
                        return True
                else:
                    return True
        except Exception:
            pass
        time.sleep(interval_seconds)
    return False


def _bgra_from_cyndilib_frame(frame: Any) -> np.ndarray | None:
    import cv2  # lazy import — cv2 is in requirements-ndi.txt

    if hasattr(frame, "get_array"):
        try:
            raw = frame.get_array()
            w = int(getattr(frame, "xres", 0) or 0)
            h = int(getattr(frame, "yres", 0) or 0)
            if w <= 0 or h <= 0:
                return None
            arr = np.asarray(raw, dtype=np.uint8)
            expected = w * h * 4
            if arr.size < expected:
                return None
            return cv2.cvtColor(arr[:expected].reshape(h, w, 4), cv2.COLOR_RGBA2BGRA)
        except Exception:
            return None

    w, h = frame.get_resolution()
    if w <= 0 or h <= 0:
        return None
    raw = frame.get_array()
    if raw is None:
        return None
    raw = np.asarray(raw, dtype=np.uint8)
    expected = w * h * 4
    if raw.size < expected:
        return None
    return cv2.cvtColor(raw[:expected].reshape(h, w, 4), cv2.COLOR_RGBA2BGRA)


# ---------------------------------------------------------------------------
# Receiver
# ---------------------------------------------------------------------------

class NDIHeadlessReceiver:
    """Headless NDI frame receiver using cyndilib.

    Self-contained — does not import from screen_capture.py.
    Exposes grab() → BGRA ndarray or None on transient miss.
    Reconnect with exponential back-off is handled transparently.
    """

    def __init__(self, cfg: PipelineConfig) -> None:
        self._source_name = cfg.ndi_source_name.strip()

        symbols = _load_cyndilib_symbols()
        self._Finder = symbols["Finder"]
        self._ReceiveFrameType = symbols["ReceiveFrameType"]
        Receiver = symbols["Receiver"]
        RecvColorFormat = symbols["RecvColorFormat"]
        RecvBandwidth = symbols["RecvBandwidth"]
        VideoFrameSync = symbols["VideoFrameSync"]
        VideoRecvFrame = symbols["VideoRecvFrame"]

        self._finder: Any | None = None
        self._video_frame_sync: Any | None = None
        self._video_frame: Any | None = None
        self._source_assigned = False
        self._reconnect_failures = 0
        self._last_reconnect_attempt = 0.0

        recv_kwargs: dict[str, Any] = {"color_format": RecvColorFormat.RGBX_RGBA}
        if RecvBandwidth is not None:
            bw = getattr(RecvBandwidth, "highest", None)
            if bw is not None:
                recv_kwargs["bandwidth"] = bw

        source = self._resolve_source()
        if self._source_name and source is None:
            raise RuntimeError(f"NDI source '{self._source_name}' not found on network")
        if source is not None:
            recv_kwargs["source"] = source

        self._receiver = Receiver(**recv_kwargs)

        if VideoFrameSync is not None and getattr(self._receiver, "frame_sync", None) is not None:
            self._video_frame_sync = VideoFrameSync()
            self._receiver.frame_sync.set_video_frame(self._video_frame_sync)
            log.info("[NDI] Using VideoFrameSync path")
        elif VideoRecvFrame is not None:
            self._video_frame = VideoRecvFrame()
            self._receiver.set_video_frame(self._video_frame)
            log.info("[NDI] Using VideoRecvFrame fallback path")
        else:
            raise RuntimeError("Unsupported cyndilib version: no usable video frame API found")

        if source is not None:
            self._receiver.set_source(source)
            self._source_assigned = True
        elif not self._source_name:
            self._assign_first_available_source()

        connected = _wait_for_receiver_connection(
            self._receiver,
            getattr(self._receiver, "frame_sync", None),
            self._video_frame_sync,
            getattr(self._receiver, "receive", None),
            getattr(self._ReceiveFrameType, "recv_video", None),
            attempts=40,
            interval_seconds=0.1,
        )
        if not connected:
            raise RuntimeError("Failed to connect to NDI source — check DistroAV on main PC")

        log.info("[NDI] Receiver connected and ready")

    def grab(self) -> np.ndarray | None:
        """Return latest BGRA frame (H, W, 4) or None on a transient miss."""
        if not self._receiver.is_connected():
            self._attempt_reconnect()
            if not self._receiver.is_connected():
                return None

        frame_obj: Any | None = None
        if self._video_frame_sync is not None and getattr(self._receiver, "frame_sync", None) is not None:
            try:
                self._receiver.frame_sync.capture_video()
            except Exception:
                return None
            frame_obj = self._video_frame_sync
            if int(getattr(frame_obj, "xres", 0) or 0) <= 0:
                return None
        else:
            try:
                recv_result = self._receiver.receive(self._ReceiveFrameType.recv_video, 10)
            except Exception:
                return None
            if not (recv_result & self._ReceiveFrameType.recv_video):
                return None
            frame_obj = getattr(self._receiver, "video_frame", None) or self._video_frame

        frame = _bgra_from_cyndilib_frame(frame_obj)
        if frame is not None:
            self._reconnect_failures = 0
        return frame

    def close(self) -> None:
        for obj in (self._receiver, self._finder):
            try:
                if obj is not None:
                    obj.close()
            except Exception:
                pass

    def _resolve_source(self) -> Any | None:
        if not self._source_name:
            return None
        if self._finder is None:
            self._finder = self._Finder()
        finder = self._finder
        if not getattr(finder, "is_open", False):
            finder.open()

        source = _find_ndi_source_by_name(finder, self._source_name)
        if source is not None:
            return source

        for _ in range(8):
            try:
                changed = finder.wait_for_sources(0.5)
            except TypeError:
                changed = finder.wait_for_sources(timeout=0.5)
            if changed:
                finder.update_sources()
            source = _find_ndi_source_by_name(finder, self._source_name)
            if source is not None:
                return source
        return None

    def _assign_first_available_source(self) -> None:
        if self._finder is None:
            self._finder = self._Finder()
        finder = self._finder
        if not getattr(finder, "is_open", False):
            finder.open()

        for attempt in range(10):
            names = [n for n in finder.get_source_names() if isinstance(n, str) and n.strip()]
            if names:
                name = names[0].strip()
                with finder.notify:
                    source = finder.get_source(name)
                    self._receiver.set_source(source)
                    self._source_assigned = True
                log.info("[NDI] Auto-selected source: '%s'", name)
                return
            try:
                finder.wait_for_sources(0.5)
            except TypeError:
                finder.wait_for_sources(timeout=0.5)
            log.debug("[NDI] Waiting for source discovery (attempt %d/10)...", attempt + 1)

        log.warning("[NDI] No NDI sources discovered within timeout")

    def _attempt_reconnect(self) -> None:
        now = time.monotonic()
        backoff = _BACKOFF[min(self._reconnect_failures, len(_BACKOFF) - 1)]
        if now - self._last_reconnect_attempt < backoff:
            return

        self._last_reconnect_attempt = now
        self._reconnect_failures += 1

        if self._reconnect_failures > _MAX_ATTEMPTS:
            raise RuntimeError(
                f"[NDI] Persistent disconnect after {self._reconnect_failures} reconnect attempts"
            )

        log.warning("[NDI] Disconnected — reconnect attempt %d...", self._reconnect_failures)

        source = self._resolve_source()
        if source is None and not self._source_name:
            self._assign_first_available_source()
            return

        if source is not None:
            try:
                self._receiver.set_source(source)
                self._source_assigned = True
                _wait_for_receiver_connection(
                    self._receiver,
                    getattr(self._receiver, "frame_sync", None),
                    self._video_frame_sync,
                    getattr(self._receiver, "receive", None),
                    getattr(self._ReceiveFrameType, "recv_video", None),
                    attempts=10,
                    interval_seconds=0.1,
                )
                if self._receiver.is_connected():
                    log.info("[NDI] Reconnected successfully")
                    self._reconnect_failures = 0
            except Exception as exc:
                log.warning("[NDI] Reconnect failed: %s", exc)
