from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

from src.core.screen_capture import (
    NDICapture,
    _extract_ndi_source_name,
    _find_ndi_source_by_name,
    _load_cyndilib_symbols,
    _wait_for_receiver_connection,
)

from ..config_loader import PipelineConfig

log = logging.getLogger(__name__)

# Reconnect back-off delays (seconds): 1, 2, 4, 8, 16, 16, ...
_BACKOFF = [1.0, 2.0, 4.0, 8.0, 16.0]
_MAX_CYCLES = 3  # raise RuntimeError after this many full back-off cycles


class NDIHeadlessReceiver:
    """Headless NDI frame receiver using cyndilib.

    Wraps the cyndilib Finder + Receiver pattern from NDICapture but strips
    all GUI / Config / overlay dependencies.  Exposes a single grab() call
    that returns a full-resolution BGRA ndarray or None on a transient miss.
    Reconnect is handled transparently inside grab().
    """

    def __init__(self, cfg: PipelineConfig) -> None:
        self._cfg = cfg
        self._source_name = cfg.ndi_source_name.strip()

        symbols = _load_cyndilib_symbols()  # raises RuntimeError if not installed
        self._Finder = symbols["Finder"]
        self._ReceiveFrameType = symbols["ReceiveFrameType"]
        Receiver = symbols["Receiver"]
        RecvColorFormat = symbols["RecvColorFormat"]
        RecvBandwidth = symbols["RecvBandwidth"]
        VideoFrameSync = symbols["VideoFrameSync"]
        VideoRecvFrame = symbols["VideoRecvFrame"]

        self._VideoFrameSync_cls = VideoFrameSync
        self._VideoRecvFrame_cls = VideoRecvFrame

        self._finder: Any | None = None
        self._receiver: Any | None = None
        self._video_frame_sync: Any | None = None
        self._video_frame: Any | None = None
        self._source_assigned = False
        self._last_frame: np.ndarray | None = None

        # Build receiver kwargs
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

        # Attach video frame object
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
        self._reconnect_failures = 0
        self._last_reconnect_attempt = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

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

        frame = NDICapture._bgra_from_cyndilib_frame(frame_obj)
        if frame is None:
            return None

        self._last_frame = frame
        self._reconnect_failures = 0
        return frame

    def close(self) -> None:
        try:
            if self._receiver is not None:
                self._receiver.close()
        except Exception:
            pass
        try:
            if self._finder is not None:
                self._finder.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

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

        log.warning("[NDI] No sources discovered within timeout")

    def _attempt_reconnect(self) -> None:
        now = time.monotonic()
        backoff = _BACKOFF[min(self._reconnect_failures, len(_BACKOFF) - 1)]
        if now - self._last_reconnect_attempt < backoff:
            return

        self._last_reconnect_attempt = now
        self._reconnect_failures += 1

        if self._reconnect_failures > _MAX_CYCLES * len(_BACKOFF):
            raise RuntimeError(
                "[NDI] Persistent disconnect — failed to reconnect after "
                f"{self._reconnect_failures} attempts"
            )

        log.warning("[NDI] Disconnected. Reconnect attempt %d...", self._reconnect_failures)

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
