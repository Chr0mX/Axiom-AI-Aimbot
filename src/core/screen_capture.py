from __future__ import annotations

import colorsys
import logging
import os
import re
import shutil
import subprocess
import threading
import time

logger = logging.getLogger(__name__)
from typing import TYPE_CHECKING, Any

import cv2
import mss
import numpy as np

if TYPE_CHECKING:
    from mss.base import MSSBase

    from .config import Config


_WARNED_MESSAGES: set[str] = set()
_CAPTURE_RETRY_INTERVAL_SECONDS = 5.0
# uvc/udp reader threads run in the background and can go silent at runtime
# (device unplugged, ffmpeg subprocess died, stream stopped) without the
# backend object itself ever raising. _detect_active_capture_method() only
# does an isinstance() check, so it keeps reporting the configured method as
# "active" forever in that case — this timeout is how reinitialize_if_method_
# changed() notices the backend is alive-but-dead and forces a fresh one.
_CAPTURE_STALE_TIMEOUT_SECONDS = 3.0

_JPEG_SOF_MARKERS = frozenset({
    0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
    0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF,
})


def _jpeg_dimensions(data: bytes) -> "tuple[int, int] | None":
    """Parse a JPEG's SOF marker to get (width, height) without decoding pixels.

    Used to decide whether a reduced-resolution decode is safe for the
    current detection region before paying the cost of a full decode.
    """
    n = len(data)
    i = 2  # skip SOI (FF D8)
    while i + 3 < n:
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        if marker == 0xFF:
            i += 1
            continue
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            i += 2
            continue
        if marker == 0xD9:  # EOI
            break
        if i + 5 >= n:
            break
        seg_len = (data[i + 2] << 8) | data[i + 3]
        if marker in _JPEG_SOF_MARKERS:
            if i + 8 >= n:
                break
            height = (data[i + 5] << 8) | data[i + 6]
            width = (data[i + 7] << 8) | data[i + 8]
            if width > 0 and height > 0:
                return width, height
            return None
        if seg_len < 2:
            break
        i += 2 + seg_len
    return None

# ---------------------------------------------------------------------------
# Module-level preview frame — written by capture worker, read by GUI timer.
# ---------------------------------------------------------------------------
_preview_lock = threading.Lock()
_preview_cell: list = [None]   # [np.ndarray | None]
_preview_region_cell: list = [None]  # [dict | None]


def set_preview_frame(frame: np.ndarray) -> None:
    with _preview_lock:
        _preview_cell[0] = frame


def get_preview_frame() -> "np.ndarray | None":
    with _preview_lock:
        return _preview_cell[0]


def set_preview_region(region: "dict | None") -> None:
    with _preview_lock:
        _preview_region_cell[0] = region


def get_preview_region() -> "dict | None":
    with _preview_lock:
        return _preview_region_cell[0]


def _detect_active_capture_method(screen_capture: Any, fallback_method: str = 'mss') -> str:
    """Best-effort detection of the currently active capture backend name."""

    if screen_capture is None:
        return str(fallback_method or 'mss')

    if isinstance(screen_capture, NDICapture):
        return 'ndi'
    if isinstance(screen_capture, UVCCapture):
        return 'uvc'
    if isinstance(screen_capture, UdpCapture):
        return 'udp'

    module_name = str(getattr(type(screen_capture), '__module__', '')).lower()
    if module_name.startswith('mss') or '.mss' in module_name:
        return 'mss'
    if module_name.startswith('dxcam') or '.dxcam' in module_name:
        return 'dxcam'

    return str(fallback_method or 'mss')


def _wait_for_receiver_connection(
    receiver: Any,
    frame_sync: Any | None,
    video_frame_sync: Any | None,
    receive_fn: Any | None,
    receive_video_flag: Any | None,
    attempts: int = 30,
    interval_seconds: float = 0.1,
) -> bool:
    """Wait until a cyndilib receiver becomes connected with video-ready state."""

    for _ in range(max(1, int(attempts))):
        try:
            if receiver.is_connected():
                if frame_sync is not None and video_frame_sync is not None:
                    frame_sync.capture_video()
                    if int(getattr(video_frame_sync, 'xres', 0) or 0) > 0:
                        return True
                elif callable(receive_fn) and receive_video_flag is not None:
                    recv_result = receive_fn(receive_video_flag, 100)
                    if recv_result & receive_video_flag:
                        return True
                else:
                    return True
        except Exception:
            pass
        time.sleep(max(0.0, float(interval_seconds)))

    return bool(getattr(receiver, 'is_connected', lambda: False)())


def _load_cyndilib_symbols() -> dict[str, Any]:
    """Load cyndilib objects while supporting API differences across versions."""

    try:
        from cyndilib.finder import Finder  # type: ignore[import-not-found]
        from cyndilib.receiver import ReceiveFrameType, Receiver  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError('cyndilib is not installed') from exc

    RecvColorFormat: Any
    RecvBandwidth: Any | None = None
    try:
        from cyndilib.wrapper.ndi_recv import (  # type: ignore[import-not-found]
            RecvBandwidth as _RecvBandwidth,
            RecvColorFormat as _RecvColorFormat,
        )

        RecvColorFormat = _RecvColorFormat
        RecvBandwidth = _RecvBandwidth
    except ImportError:
        from cyndilib.wrapper import RecvColorFormat as _RecvColorFormat  # type: ignore[import-not-found]

        RecvColorFormat = _RecvColorFormat

    VideoFrameSync: Any | None = None
    VideoRecvFrame: Any | None = None
    try:
        from cyndilib.video_frame import VideoFrameSync as _VideoFrameSync  # type: ignore[import-not-found]

        VideoFrameSync = _VideoFrameSync
    except ImportError:
        try:
            from cyndilib import VideoRecvFrame as _VideoRecvFrame  # type: ignore[import-not-found]

            VideoRecvFrame = _VideoRecvFrame
        except ImportError:
            pass

    return {
        'Finder': Finder,
        'Receiver': Receiver,
        'ReceiveFrameType': ReceiveFrameType,
        'RecvColorFormat': RecvColorFormat,
        'RecvBandwidth': RecvBandwidth,
        'VideoFrameSync': VideoFrameSync,
        'VideoRecvFrame': VideoRecvFrame,
    }


# Fixed title for the UVC/NDI preview window (not user-configurable).
_UVC_WINDOW_NAME = "Axiom UVC Preview"


def _uvc_signature(config: Config) -> tuple[int, int, int, int, bool, str, str, str, str, str, int]:
    crop_mode = str(getattr(config, 'uvc_crop_mode', 'dynamic')).lower()
    return (
        int(getattr(config, 'uvc_device_index', 0)),
        int(getattr(config, 'uvc_width', 0)),
        int(getattr(config, 'uvc_height', 0)),
        int(getattr(config, 'uvc_fps', 0)),
        bool(getattr(config, 'uvc_show_window', False)),
        str(getattr(config, 'uvc_capture_method', 'dshow')).lower(),
        str(getattr(config, 'uvc_preview_scale_mode', 'scale_to_fit')).lower(),
        str(getattr(config, 'uvc_video_format', 'mjpeg')).lower(),
        str(getattr(config, 'uvc_ffmpeg_path', '') or ''),
        crop_mode,
        # Only fixed-crop mode bakes detect_range_size into the frozen crop
        # rect (ffmpeg subprocess arg, or UVCCapture's cached region) —
        # include it in the signature only then, so a live Detection Range
        # change while in fixed mode triggers the same hot-swap reinit as
        # any other UVC setting change. 0 in dynamic mode keeps this a
        # constant (no spurious reinits).
        int(getattr(config, 'detect_range_size', 0) or 0) if crop_mode == 'fixed' else 0,
    )


def _udp_signature(config: Config) -> tuple[str, int, int, float, bool]:
    return (
        str(getattr(config, 'udp_bind_ip', '0.0.0.0')),
        int(getattr(config, 'udp_bind_port', 5600)),
        int(getattr(config, 'udp_recv_buffer_size', 65536)),
        float(getattr(config, 'udp_frame_timeout', 1.0)),
        bool(getattr(config, 'udp_force_restart', False)),
    )


def _ndi_signature(config: Config) -> tuple[str, bool, str, str, bool]:
    return (
        str(getattr(config, 'ndi_source_name', '')).strip(),
        bool(getattr(config, 'uvc_show_window', False)),
        str(getattr(config, 'uvc_preview_scale_mode', 'scale_to_fit')).lower(),
        str(getattr(config, 'ndi_bandwidth', 'highest')).lower(),
        bool(getattr(config, 'ndi_force_reconnect', False)),
    )


def _extract_ndi_source_name(source: Any) -> str:
    if isinstance(source, str):
        return source.strip()
    for attr in ('name', 'source_name', 'stream_name', 'url'):
        value = getattr(source, attr, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    try:
        as_text = str(source).strip()
    except Exception:
        as_text = ''
    return as_text


def _extract_ndi_stream_name(source: Any) -> str:
    for attr in ('stream_name', 'ndi_name', 'stream', 'source_name', 'name'):
        value = getattr(source, attr, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ''


def _find_ndi_source_by_name(finder: Any, target_name: str) -> Any | None:
    """Find an NDI source by full source name or stream name."""

    target = str(target_name or '').strip()
    if not target:
        return None
    target_lower = target.lower()

    # Fast path for full-name exact match.
    try:
        source = finder.get_source(target)
        if source is not None:
            return source
    except Exception:
        pass

    # Fall back to iteration and stream-name matching (as shown in cyndilib docs).
    try:
        for source in finder:
            full_name = _extract_ndi_source_name(source)
            stream_name = _extract_ndi_stream_name(source)
            full_name_lower = full_name.lower()
            stream_name_lower = stream_name.lower()
            if (
                target_lower in {full_name_lower, stream_name_lower}
                or full_name_lower.endswith(f"({target_lower})")
                or stream_name_lower.endswith(f"({target_lower})")
            ):
                return source
    except Exception:
        pass

    return None


def list_available_ndi_sources() -> list[str]:
    """Return discovered NDI source names via cyndilib when available."""

    try:
        from cyndilib.finder import Finder  # type: ignore[import-not-found]
    except ImportError:
        return []

    def _normalize(names: list[Any]) -> list[str]:
        result: list[str] = []
        for entry in names:
            name = _extract_ndi_source_name(entry)
            if name and name not in result:
                result.append(name)
        return result

    try:
        with Finder() as finder:
            if not getattr(finder, "is_open", False):
                finder.open()
            names = _normalize(finder.get_source_names())
            if names:
                return names

            # Discovery can take a few seconds on some networks.
            for _ in range(6):
                try:
                    changed = finder.wait_for_sources(0.5)
                except TypeError:
                    changed = finder.wait_for_sources(timeout=0.5)
                if changed:
                    finder.update_sources()
                    names = _normalize(finder.get_source_names())
                    if names:
                        return names
            return _normalize(finder.get_source_names())
    except Exception:
        return []


def _format_ndi_source_label(name: str, width: int | None, height: int | None, fps: float | None) -> str:
    _ = (width, height, fps)
    return name


def _extract_ndi_source_video_meta(source: Any) -> tuple[int | None, int | None, float | None]:
    """Best-effort metadata extraction from cyndilib source objects."""

    width = height = None
    fps = None

    for key in ('width', 'xres', 'video_width', 'frame_width'):
        value = getattr(source, key, None)
        if isinstance(value, (int, float)) and int(value) > 0:
            width = int(value)
            break
    for key in ('height', 'yres', 'video_height', 'frame_height'):
        value = getattr(source, key, None)
        if isinstance(value, (int, float)) and int(value) > 0:
            height = int(value)
            break
    for key in ('frame_rate', 'framerate', 'fps', 'video_fps'):
        value = getattr(source, key, None)
        if isinstance(value, (int, float)) and float(value) > 0:
            fps = float(value)
            break
    if fps is None:
        num = getattr(source, 'frame_rate_N', None)
        den = getattr(source, 'frame_rate_D', None)
        if isinstance(num, (int, float)) and isinstance(den, (int, float)) and float(den) > 0:
            fps = float(num) / float(den)

    return width, height, fps


def _draw_detection_overlay(
    frame: np.ndarray,
    region: dict | None,
    config: Any,
    *,
    has_alpha: bool = False,
) -> np.ndarray:
    """Shared overlay renderer used by both UVCCapture and NDICapture preview threads.

    Args:
        frame:     BGR or BGRA frame to draw on (modified in place).
        region:    Detection region dict with 'left', 'top', 'width', 'height'.
        config:    Live Config object.
        has_alpha: True for BGRA frames (NDI); False for BGR frames (UVC).
                   Controls whether color tuples include a 4th alpha byte.
    """
    cfg = config
    if not bool(getattr(cfg, 'AimToggle', True)):
        return frame

    h, w = frame.shape[:2]
    region_left   = int(region.get('left',   0)) if region else 0
    region_top    = int(region.get('top',    0)) if region else 0
    region_width  = int(region.get('width',  w)) if region else w
    region_height = int(region.get('height', h)) if region else h

    cx = int(getattr(cfg, 'crosshairX', w // 2))
    cy = int(getattr(cfg, 'crosshairY', h // 2))

    def _c(b: int, g: int, r: int, a: int = 255) -> tuple:
        return (b, g, r, a) if has_alpha else (b, g, r)

    if bool(getattr(cfg, 'show_detect_range', False)):
        x1 = max(0, region_left)
        y1 = max(0, region_top)
        x2 = min(w - 1, region_left + region_width)
        y2 = min(h - 1, region_top + region_height)
        cv2.rectangle(frame, (x1, y1), (x2, y2), _c(255, 140, 0), 1, cv2.LINE_AA)

    if bool(getattr(cfg, 'show_fov', True)):
        fov = int(getattr(cfg, 'fov_size', 220))
        half = max(1, fov // 2)
        x1, y1 = cx - half, cy - half
        x2, y2 = cx + half, cy + half
        color = _c(0, 0, 255)
        if bool(getattr(cfg, 'fov_circle_filter_enabled', False)):
            cv2.circle(frame, (cx, cy), half, color, 2, cv2.LINE_AA)
        else:
            corner = max(8, min(20, fov // 6))
            cv2.line(frame, (x1, y1), (x1 + corner, y1), color, 2, cv2.LINE_AA)
            cv2.line(frame, (x1, y1), (x1, y1 + corner), color, 2, cv2.LINE_AA)
            cv2.line(frame, (x2, y1), (x2 - corner, y1), color, 2, cv2.LINE_AA)
            cv2.line(frame, (x2, y1), (x2, y1 + corner), color, 2, cv2.LINE_AA)
            cv2.line(frame, (x1, y2), (x1 + corner, y2), color, 2, cv2.LINE_AA)
            cv2.line(frame, (x1, y2), (x1, y2 - corner), color, 2, cv2.LINE_AA)
            cv2.line(frame, (x2, y2), (x2 - corner, y2), color, 2, cv2.LINE_AA)
            cv2.line(frame, (x2, y2), (x2, y2 - corner), color, 2, cv2.LINE_AA)

    if bool(getattr(cfg, 'show_boxes', True)):
        boxes       = list(getattr(cfg, 'latest_boxes',       []) or [])
        confidences = list(getattr(cfg, 'latest_confidences', []) or [])
        show_conf   = bool(getattr(cfg, 'show_confidence', True))
        _theme = {
            'cyan':   _c(255, 220, 0),
            'red':    _c(60,  60,  255),
            'yellow': _c(0,   210, 255),
            'white':  _c(255, 255, 255),
            'purple': _c(255, 60,  180),
        }
        box_color = _theme.get(str(getattr(cfg, 'box_color_theme', 'default')).lower(), _c(0, 255, 0))
        speed = float(getattr(cfg, 'chroma_box_speed', 1.0))
        hue = (time.monotonic() * speed * 60.0) % 360.0
        r_f, g_f, b_f = colorsys.hsv_to_rgb(hue / 360.0, 1.0, 1.0)
        chroma_color = _c(int(b_f * 255), int(g_f * 255), int(r_f * 255), 220)
        use_circle = bool(getattr(cfg, 'fov_circle_filter_enabled', False))
        fov_half   = float(getattr(cfg, 'fov_size', 220)) / 2.0
        for i, box in enumerate(boxes):
            try:
                x1, y1, x2, y2 = [int(v) for v in box]
            except Exception:
                continue
            if x2 <= 0 or y2 <= 0 or x1 >= w or y1 >= h:
                continue
            x1 = max(0, min(w - 1, x1))
            y1 = max(0, min(h - 1, y1))
            x2 = max(0, min(w - 1, x2))
            y2 = max(0, min(h - 1, y2))
            conf      = float(confidences[i]) if i < len(confidences) else 0.5
            thickness = max(1, min(3, 1 + round(conf * 2)))
            clen      = max(6, min(24, int(min(x2 - x1, y2 - y1) * 0.15)))
            if use_circle:
                nx = min(max(float(cx), float(x1)), float(x2))
                ny = min(max(float(cy), float(y1)), float(y2))
                in_fov = (nx - cx) ** 2 + (ny - cy) ** 2 <= fov_half * fov_half
            else:
                in_fov = (x1 < cx + fov_half and x2 > cx - fov_half and
                          y1 < cy + fov_half and y2 > cy - fov_half)
            dc = chroma_color if in_fov else box_color
            if getattr(cfg, 'box_full_rect', False):
                cv2.rectangle(frame, (x1, y1), (x2, y2), dc, thickness, cv2.LINE_AA)
            else:
                cv2.line(frame, (x1, y1), (x1 + clen, y1), dc, thickness, cv2.LINE_AA)
                cv2.line(frame, (x1, y1), (x1, y1 + clen), dc, thickness, cv2.LINE_AA)
                cv2.line(frame, (x2, y1), (x2 - clen, y1), dc, thickness, cv2.LINE_AA)
                cv2.line(frame, (x2, y1), (x2, y1 + clen), dc, thickness, cv2.LINE_AA)
                cv2.line(frame, (x1, y2), (x1 + clen, y2), dc, thickness, cv2.LINE_AA)
                cv2.line(frame, (x1, y2), (x1, y2 - clen), dc, thickness, cv2.LINE_AA)
                cv2.line(frame, (x2, y2), (x2 - clen, y2), dc, thickness, cv2.LINE_AA)
                cv2.line(frame, (x2, y2), (x2, y2 - clen), dc, thickness, cv2.LINE_AA)
            if show_conf and i < len(confidences):
                cv2.putText(frame, f"{conf * 100:.0f}%",
                            (max(0, x1 - 5), max(15, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            _c(0, 255, 255), 1, cv2.LINE_AA)

    if bool(getattr(cfg, 'show_tracer_line', False)):
        tracer_boxes = list(getattr(cfg, 'latest_boxes', []) or [])
        fov_half = max(1, int(getattr(cfg, 'fov_size', 220)) // 2)
        for box in tracer_boxes:
            try:
                x1, y1, x2, y2 = [int(v) for v in box]
            except Exception:
                continue
            bx = (x1 + x2) // 2
            by = (y1 + y2) // 2
            if abs(bx - cx) <= fov_half and abs(by - cy) <= fov_half:
                cv2.line(frame, (cx, cy), (bx, by), _c(255, 255, 255), 2, cv2.LINE_AA)

    decay_box = getattr(cfg, 'display_locked_box', None)
    if decay_box is not None and bool(getattr(cfg, 'display_locked_box_is_decaying', False)):
        try:
            x1, y1, x2, y2 = [int(v) for v in decay_box]
            x1 = max(0, min(w - 1, x1)); y1 = max(0, min(h - 1, y1))
            x2 = max(0, min(w - 1, x2)); y2 = max(0, min(h - 1, y2))
            cv2.rectangle(frame, (x1, y1), (x2, y2), _c(0, 140, 255), 1, cv2.LINE_AA)
        except Exception:
            pass

    return frame


def _render_preview_frame(window_name: str, mode: str, frame_bgr: np.ndarray) -> np.ndarray:
    """Render capture preview according to configured preview mode."""

    # Lowest-latency mode: avoid any resize/canvas composition work.
    if mode == 'low_latency':
        return frame_bgr

    if mode == 'scale_to_canvas':
        try:
            _, _, width, height = cv2.getWindowImageRect(window_name)
            if width > 0 and height > 0:
                return cv2.resize(frame_bgr, (width, height), interpolation=cv2.INTER_NEAREST)
        except Exception:
            return frame_bgr
    if mode == 'fit_to_screen':
        try:
            screen_w, screen_h = 1920, 1080
            max_w = max(320, int(screen_w * 0.9))
            max_h = max(240, int(screen_h * 0.9))
            h, w = frame_bgr.shape[:2]
            ratio = min(max_w / max(1, w), max_h / max(1, h))
            target_w = max(1, int(w * ratio))
            target_h = max(1, int(h * ratio))
            cv2.resizeWindow(window_name, target_w, target_h)
        except Exception:
            pass
        return frame_bgr

    # default: scale_to_fit
    try:
        _, _, width, height = cv2.getWindowImageRect(window_name)
        if width <= 0 or height <= 0:
            return frame_bgr
        h, w = frame_bgr.shape[:2]
        ratio = min(width / max(1, w), height / max(1, h))
        draw_w = max(1, int(w * ratio))
        draw_h = max(1, int(h * ratio))
        resized = cv2.resize(frame_bgr, (draw_w, draw_h), interpolation=cv2.INTER_NEAREST)
        channels = 1 if frame_bgr.ndim < 3 else frame_bgr.shape[2]
        canvas_shape = (height, width) if channels == 1 else (height, width, channels)
        canvas = np.zeros(canvas_shape, dtype=np.uint8)
        x = (width - draw_w) // 2
        y = (height - draw_h) // 2
        canvas[y:y + draw_h, x:x + draw_w] = resized
        return canvas
    except Exception:
        return frame_bgr


def list_available_ndi_source_details() -> list[dict[str, str | int | float | None]]:
    """Return discovered NDI sources without querying stream resolution/FPS."""

    try:
        symbols = _load_cyndilib_symbols()
    except RuntimeError:
        return []
    Finder = symbols['Finder']

    details: list[dict[str, str | int | float | None]] = []

    try:
        with Finder() as finder:
            if not getattr(finder, "is_open", False):
                finder.open()

            names = [n for n in finder.get_source_names() if isinstance(n, str) and n.strip()]
            if not names:
                for _ in range(6):
                    try:
                        changed = finder.wait_for_sources(0.5)
                    except TypeError:
                        changed = finder.wait_for_sources(timeout=0.5)
                    if changed:
                        finder.update_sources()
                        names = [n for n in finder.get_source_names() if isinstance(n, str) and n.strip()]
                        if names:
                            break
            if not names:
                return []

            for name in names:
                width: int | None = None
                height: int | None = None
                fps: float | None = None

                details.append(
                    {
                        'name': name,
                        'width': width,
                        'height': height,
                        'fps': fps,
                        'label': _format_ndi_source_label(name, width, height, fps),
                    }
                )
    except Exception:
        return []

    return details


class NDICapture:
    """NDI capture backend powered by cyndilib."""

    def __init__(self, config: Config) -> None:
        self.config = config
        # Clear reconnect flag before computing signature so the stored
        # signature reflects the settled state (no pending reconnect).
        config.ndi_force_reconnect = False
        self.config_signature = _ndi_signature(config)
        self.source_name = str(getattr(config, 'ndi_source_name', '')).strip()
        # The GUI exposes a single shared "Capture Preview Window" toggle that
        # writes uvc_show_window for both UVC and NDI backends.
        self.show_window = bool(getattr(config, 'uvc_show_window', False))
        self.window_name = str(getattr(config, 'ndi_window_name', 'Axiom NDI Preview'))
        # NDI preview prioritizes minimal display latency by default.
        ndi_preview_scale_mode = str(getattr(config, 'ndi_preview_scale_mode', '')).lower().strip()
        self.preview_scale_mode = ndi_preview_scale_mode or 'low_latency'
        self._finder: Any | None = None
        self._source_assigned = False
        self._reconnect_logged = False
        self.preview_width = int(getattr(config, 'ndi_width', getattr(config, 'width', 0)) or 0)
        self.preview_height = int(getattr(config, 'ndi_height', getattr(config, 'height', 0)) or 0)

        logger.info('[Capture][NDI] Initializing cyndilib NDI backend...')
        if self.source_name:
            logger.info("[Capture][NDI] Requested source name from config: '%s'.", self.source_name)
        else:
            logger.info('[Capture][NDI] No source name configured. First discovered source will be auto-selected.')

        try:
            symbols = _load_cyndilib_symbols()
        except RuntimeError as exc:
            raise RuntimeError('cyndilib is not installed') from exc
        self._Finder = symbols['Finder']
        self._ReceiveFrameType = symbols['ReceiveFrameType']
        Receiver = symbols['Receiver']
        RecvColorFormat = symbols['RecvColorFormat']
        RecvBandwidth = symbols['RecvBandwidth']
        VideoFrameSync = symbols['VideoFrameSync']
        VideoRecvFrame = symbols['VideoRecvFrame']

        try:
            source = self._resolve_source()
            if self.source_name and source is None:
                raise RuntimeError(f"NDI source '{self.source_name}' not found")

            _uyvy_fmt = getattr(RecvColorFormat, 'UYVY_RGBA', None)
            _bgra_fmt = getattr(RecvColorFormat, 'BGRX_BGRA', None)
            if _uyvy_fmt is not None:
                receiver_kwargs: dict[str, Any] = {'color_format': _uyvy_fmt}
                self._recv_fourcc: str = 'uyvy'
                logger.info('[Capture][NDI] Color format: UYVY_RGBA (half bandwidth, zero-copy reshape)')
            elif _bgra_fmt is not None:
                receiver_kwargs = {'color_format': _bgra_fmt}
                self._recv_fourcc = 'bgra'
                logger.info('[Capture][NDI] Color format: BGRX_BGRA (no cvtColor)')
            else:
                receiver_kwargs = {'color_format': RecvColorFormat.RGBX_RGBA}
                self._recv_fourcc = 'rgba'
                logger.info('[Capture][NDI] Color format: RGBX_RGBA (cvtColor fallback)')
            # Legacy flag kept so any external callers that check _recv_is_bgra still work
            self._recv_is_bgra: bool = self._recv_fourcc == 'bgra'
            if RecvBandwidth is not None:
                bw_pref = str(getattr(config, 'ndi_bandwidth', 'highest')).lower()
                bw_value = getattr(RecvBandwidth, bw_pref, None) or getattr(RecvBandwidth, 'highest', None)
                if bw_value is not None:
                    receiver_kwargs['bandwidth'] = bw_value
                    logger.info('[Capture][NDI] Bandwidth set to: %s', bw_pref)
            if source is not None:
                # NOTE: do NOT pass the source to the Receiver constructor.
                # Assigning it both in the ctor and again via set_source() below
                # double-assigns the source and disrupts the initial connection
                # (observed as a fall-back to MSS on the second launch, when a
                # source name is already saved in config). The auto-select path
                # — which connects reliably — only ever calls set_source() once,
                # so the configured-source path now mirrors it.
                logger.info("[Capture][NDI] Resolved source '%s'; assigning after receiver creation.",
                            _extract_ndi_source_name(source))
            self._receiver = Receiver(**receiver_kwargs)
            logger.info('[Capture][NDI] Receiver object created successfully.')

            self._video_frame_sync: Any | None = None
            self._video_frame: Any | None = None
            if VideoFrameSync is not None and getattr(self._receiver, 'frame_sync', None) is not None:
                self._video_frame_sync = VideoFrameSync()
                self._receiver.frame_sync.set_video_frame(self._video_frame_sync)
                logger.info('[Capture][NDI] Using VideoFrameSync capture path (matches gist flow).')
            elif VideoRecvFrame is not None:
                self._video_frame = VideoRecvFrame()
                self._receiver.set_video_frame(self._video_frame)
                logger.info('[Capture][NDI] Using VideoRecvFrame fallback path (legacy cyndilib compatibility).')
            else:
                raise RuntimeError('Unsupported cyndilib version: no usable video frame API found')

            self._last_reconnect_attempt = 0.0
            if source is not None:
                # Always set source explicitly; some cyndilib versions don't
                # reliably auto-connect when source is only passed in ctor.
                self._receiver.set_source(source)
                self._source_assigned = True
            elif not self.source_name:
                self._assign_first_available_source()
        except Exception as exc:
            self._teardown_receiver_and_finder()
            raise RuntimeError(f'Failed to initialize cyndilib NDI receiver: {exc}') from exc

        logger.info('[Capture][NDI] Waiting for receiver to connect and deliver first video frame (up to 6s)...')
        connected = _wait_for_receiver_connection(
            self._receiver,
            getattr(self._receiver, 'frame_sync', None),
            getattr(self, '_video_frame_sync', None),
            getattr(self._receiver, 'receive', None),
            getattr(self._ReceiveFrameType, 'recv_video', None),
            attempts=60,
            interval_seconds=0.1,
        )

        if not connected:
            self._teardown_receiver_and_finder()
            raise RuntimeError('Failed to connect to NDI source via cyndilib')
        logger.info('[Capture][NDI] Receiver connected and video stream is ready.')
        self._last_frame_time: float = time.perf_counter()

        # Shared refs for the preview thread — grab() writes, thread reads
        self._ndi_frame_lock: threading.Lock = threading.Lock()
        self._ndi_frame_ref: list = [None]    # list[np.ndarray | None]
        self._ndi_region_ref: list = [None]
        self._ndi_stop: threading.Event = threading.Event()
        # Live-measured receive rate — actual delivery can run slower than the
        # source's declared/advertised rate under network contention, so track
        # real frames/sec like the UDP backend does for the status panel.
        self._live_fps_count: int = 0
        self._live_fps_t0: float = time.perf_counter()
        # Ping-pong pair of BGRA buffers for the crop-path — eliminates per-frame .copy().
        # Alternating ensures the buffer just returned stays valid while grab() fills the other.
        self._bgra_bufs: list[np.ndarray | None] = [None, None]
        self._bgra_shapes: list[tuple] = [(), ()]
        self._bgra_idx: int = 0

        self._ndi_preview_thread: _UVCPreviewThread | None = None
        if self.show_window:
            self._ndi_preview_thread = _UVCPreviewThread(
                window_name=self.window_name,
                scale_mode=self.preview_scale_mode,
                frame_lock=self._ndi_frame_lock,
                frame_ref=self._ndi_frame_ref,
                stop_event=self._ndi_stop,
                draw_overlay_fn=self._draw_overlay,
                region_ref=self._ndi_region_ref,
                target_fps=60,
                preview_width=self.preview_width or 1920,
                preview_height=self.preview_height or 1080,
                config=self.config,
                show_cv2_window=False,  # Qt panel is the primary display; cv2 window suppressed
            )
            self._ndi_preview_thread.start()

    def _resolve_source(self, log: bool = True) -> Any | None:
        if not self.source_name:
            return None
        try:
            if self._finder is None:
                self._finder = self._Finder()
                if log:
                    logger.info('[Capture][NDI] Finder instance created.')
            finder = self._finder
            if not getattr(finder, "is_open", False):
                finder.open()
                if log:
                    logger.info('[Capture][NDI] Finder opened for network source discovery.')
            source = _find_ndi_source_by_name(finder, self.source_name)
            if source is not None:
                if log:
                    logger.info("[Capture][NDI] Matched configured source '%s'.", self.source_name)
                return source
            for _ in range(6):
                try:
                    changed = finder.wait_for_sources(0.5)
                except TypeError:
                    changed = finder.wait_for_sources(timeout=0.5)
                if changed:
                    finder.update_sources()
                    if log:
                        logger.info('[Capture][NDI] Source list changed while searching for configured source.')
                source = _find_ndi_source_by_name(finder, self.source_name)
                if source is not None:
                    if log:
                        logger.info("[Capture][NDI] Found configured source after refresh: '%s'.", self.source_name)
                    return source
            if log:
                logger.warning("[Capture][NDI] Could not find configured source '%s' after retries.", self.source_name)
            return None
        except Exception:
            return None

    def _assign_first_available_source(self) -> None:
        """Follow gist behavior: when no source is set, auto-select first discovered stream."""

        try:
            if self._finder is None:
                self._finder = self._Finder()
                logger.info('[Capture][NDI] Finder instance created for auto-select mode.')
            finder = self._finder
            if not getattr(finder, 'is_open', False):
                finder.open()
                logger.info('[Capture][NDI] Finder opened for auto-select mode.')

            for attempt in range(8):
                names = [name for name in finder.get_source_names() if isinstance(name, str) and name.strip()]
                if names:
                    selected_name = names[0].strip()
                    with finder.notify:
                        selected_source = finder.get_source(selected_name)
                        self._receiver.set_source(selected_source)
                        self._source_assigned = True
                        logger.info("[Capture][NDI] Auto-selected first available source: '%s'.", selected_name)
                    return
                try:
                    changed = finder.wait_for_sources(0.5)
                except TypeError:
                    changed = finder.wait_for_sources(timeout=0.5)
                if changed:
                    finder.update_sources()
                    logger.info('[Capture][NDI] Waiting for source discovery (attempt %d/8)...', attempt + 1)

            logger.warning('[Capture][NDI] No NDI sources discovered for auto-select within timeout window.')
        except Exception as exc:
            logger.error('[Capture][NDI] Auto-select source setup failed: %s', exc)

    def _raw_array_from_cyndilib_frame(self, frame: Any) -> tuple[np.ndarray | None, int, int]:
        """Return raw uint8 array + (width, height) with no color conversion.

        For UYVY the shape is (H, W*2) — a packed 4:2:2 plane.
        For BGRA/RGBA the shape is (H, W, 4).
        Uses the buffer protocol (zero-copy) when available, falls back to get_array().
        """
        try:
            width = int(getattr(frame, 'xres', 0) or 0)
            height = int(getattr(frame, 'yres', 0) or 0)
            if width <= 0 or height <= 0:
                return None, 0, 0
            is_uyvy = getattr(self, '_recv_fourcc', '') == 'uyvy'
            bytes_per_pixel = 2 if is_uyvy else 4
            expected = width * height * bytes_per_pixel
            # Try buffer protocol first (zero-copy)
            try:
                arr = np.frombuffer(frame, dtype=np.uint8)
            except TypeError:
                raw = frame.get_array() if hasattr(frame, 'get_array') else None
                if raw is None:
                    return None, 0, 0
                arr = np.asarray(raw, dtype=np.uint8)
            if arr.size < expected:
                return None, 0, 0
            if is_uyvy:
                return arr[:expected].reshape(height, width, 2), width, height
            return arr[:expected].reshape(height, width, 4), width, height
        except Exception:
            return None, 0, 0

    def grab(self, region: dict[str, int] | None = None, **_: Any) -> np.ndarray | None:
        now = time.perf_counter()
        if not self._receiver.is_connected():
            # is_connected() can return False transiently on healthy streams in some
            # cyndilib builds. Only reconnect when frames have genuinely stopped flowing.
            if now - self._last_frame_time > 3.0:
                if now - float(getattr(self, '_last_reconnect_attempt', 0.0) or 0.0) > 1.0:
                    self._last_reconnect_attempt = now
                    # Only log the first attempt of a disconnect episode so a
                    # persistently-unconnected receiver doesn't spam the console.
                    log = not self._reconnect_logged
                    source = self._resolve_source(log=log)
                    if source is not None:
                        try:
                            self._receiver.set_source(source)
                            self._source_assigned = True
                            if log:
                                logger.info("[Capture][NDI] Reconnecting receiver using configured source '%s'.", self.source_name)
                                self._reconnect_logged = True
                            _wait_for_receiver_connection(
                                self._receiver,
                                getattr(self._receiver, 'frame_sync', None),
                                getattr(self, '_video_frame_sync', None),
                                getattr(self._receiver, 'receive', None),
                                getattr(self._ReceiveFrameType, 'recv_video', None),
                                attempts=5,
                                interval_seconds=0.05,
                            )
                        except Exception:
                            pass
                    elif not self.source_name and not self._source_assigned:
                        self._assign_first_available_source()
                if not self._receiver.is_connected():
                    return None
            # else: is_connected() flickered but frames are recent — fall through and attempt capture
        elif self._reconnect_logged:
            # Recovered from a disconnect episode — re-arm logging for next time.
            logger.info("[Capture][NDI] Receiver connected to '%s'.", self.source_name)
            self._reconnect_logged = False

        frame_obj: Any | None = None
        if getattr(self, '_video_frame_sync', None) is not None and getattr(self._receiver, 'frame_sync', None) is not None:
            try:
                self._receiver.frame_sync.capture_video()
            except Exception:
                return None
            frame_obj = self._video_frame_sync
            try:
                _res = frame_obj.get_resolution()
                if min(_res) <= 0 or frame_obj.get_data_size() == 0:
                    return None
            except Exception:
                if int(getattr(frame_obj, 'xres', 0) or 0) <= 0 or int(getattr(frame_obj, 'yres', 0) or 0) <= 0:
                    return None
        else:
            try:
                recv_result = self._receiver.receive(self._ReceiveFrameType.recv_video, 10)
            except Exception:
                return None
            if not (recv_result & self._ReceiveFrameType.recv_video):
                return None
            frame_obj = self._receiver.video_frame or self._video_frame

        raw, frame_w, frame_h = self._raw_array_from_cyndilib_frame(frame_obj)
        if raw is None:
            return None

        self._live_fps_count += 1
        _now_fps = time.perf_counter()
        _elapsed_fps = _now_fps - self._live_fps_t0
        if _elapsed_fps >= 1.0:
            self.config.source_nominal_fps = self._live_fps_count / _elapsed_fps
            self._live_fps_count = 0
            self._live_fps_t0 = _now_fps

        if frame_w > 0 and frame_h > 0:
            self.preview_width = frame_w
            self.preview_height = frame_h
            self.config.ndi_width = frame_w
            self.config.ndi_height = frame_h

        recv_fourcc: str = getattr(self, '_recv_fourcc', 'rgba')

        def _to_bgra(arr: np.ndarray) -> np.ndarray:
            if recv_fourcc == 'uyvy':
                return cv2.cvtColor(arr, cv2.COLOR_YUV2BGRA_UYVY)
            if recv_fourcc == 'bgra':
                # raw is a zero-copy np.frombuffer() view into cyndilib's own
                # frame buffer, which gets overwritten in place on the next
                # capture_video()/receive() call — copy so callers can safely
                # hold/read this across threads without a torn-frame race.
                return arr.copy()
            return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA)

        if self.show_window:
            full_bgra = _to_bgra(raw)
            with self._ndi_frame_lock:
                self._ndi_frame_ref[0] = full_bgra
            self._ndi_region_ref[0] = region
            if region is not None:
                left   = max(0, int(region.get('left',   0)))
                top    = max(0, int(region.get('top',     0)))
                right  = min(frame_w, left + max(0, int(region.get('width',  frame_w))))
                bottom = min(frame_h, top  + max(0, int(region.get('height', frame_h))))
                if right <= left or bottom <= top:
                    return None
                frame = full_bgra[top:bottom, left:right]
            else:
                frame = full_bgra
        else:
            if region is not None:
                left   = max(0, int(region.get('left',   0)))
                top    = max(0, int(region.get('top',     0)))
                right  = min(frame_w, left + max(0, int(region.get('width',  frame_w))))
                bottom = min(frame_h, top  + max(0, int(region.get('height', frame_h))))
                if right <= left or bottom <= top:
                    return None

                if recv_fourcc == 'uyvy':
                    left  = left  & ~1
                    right = (right + 1) & ~1
                    crop_raw = raw[top:bottom, left:right, :]
                    expected_shape = (bottom - top, right - left, 4)
                    idx = self._bgra_idx
                    if self._bgra_shapes[idx] != expected_shape:
                        self._bgra_bufs[idx]   = np.empty(expected_shape, dtype=np.uint8)
                        self._bgra_shapes[idx] = expected_shape
                    cv2.cvtColor(crop_raw, cv2.COLOR_YUV2BGRA_UYVY, self._bgra_bufs[idx])
                    frame = self._bgra_bufs[idx]
                    self._bgra_idx = 1 - idx
                elif recv_fourcc == 'bgra':
                    frame = raw[top:bottom, left:right].copy()
                else:
                    crop_raw = raw[top:bottom, left:right]
                    expected_shape = (bottom - top, right - left, 4)
                    idx = self._bgra_idx
                    if self._bgra_shapes[idx] != expected_shape:
                        self._bgra_bufs[idx]   = np.empty(expected_shape, dtype=np.uint8)
                        self._bgra_shapes[idx] = expected_shape
                    cv2.cvtColor(crop_raw, cv2.COLOR_RGBA2BGRA, self._bgra_bufs[idx])
                    frame = self._bgra_bufs[idx]
                    self._bgra_idx = 1 - idx
            else:
                frame = _to_bgra(raw)

        if frame.ndim == 3 and frame.shape[2] == 4:
            self._last_frame_time = now
            return frame
        if frame.ndim == 3 and frame.shape[2] == 3:
            self._last_frame_time = now
            return cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        return None

    def _draw_overlay(self, frame_bgra: np.ndarray, region: dict[str, int] | None) -> np.ndarray:
        return _draw_detection_overlay(frame_bgra, region, self.config, has_alpha=True)

    def _teardown_receiver_and_finder(self) -> None:
        """Best-effort teardown of the receiver/finder.

        Shared by close() (normal teardown of a fully constructed instance)
        and by __init__'s failure paths (a partially constructed instance
        whose __init__ is about to raise, so no caller will ever get a
        handle to call close() with).
        """
        receiver = getattr(self, '_receiver', None)
        if receiver is not None:
            for method_name in ('disconnect', 'close', 'release', 'stop', 'shutdown'):
                method = getattr(receiver, method_name, None)
                if callable(method):
                    try:
                        method()
                    except Exception:
                        pass

        finder = getattr(self, '_finder', None)
        if finder is not None:
            for method_name in ('close', 'stop', 'shutdown'):
                method = getattr(finder, method_name, None)
                if callable(method):
                    try:
                        method()
                    except Exception:
                        pass

    def close(self) -> None:
        self._teardown_receiver_and_finder()
        # Stop the preview thread; it destroys the window on exit
        if getattr(self, '_ndi_stop', None) is not None:
            self._ndi_stop.set()
        pt = getattr(self, '_ndi_preview_thread', None)
        if pt is not None and pt.is_alive():
            pt.join(timeout=1.0)


def _list_pygrabber_device_names() -> list[str]:
    """Enumerate UVC device names via DirectShow's native COM device
    enumeration (ICreateDevEnum/IEnumMoniker) through pygrabber — the same
    API OBS and browsers use to list capture devices, instead of scraping
    ffmpeg's log output (which depends on PyAV importing cleanly; PyAV can
    fail to import in ways unrelated to whether a real UVC device is even
    reachable, e.g. a Cython/typelib packaging conflict with another
    vendored dependency). Uses SystemDeviceEnum directly rather than the
    full FilterGraph class, since name enumeration doesn't need a filter
    graph, capture-graph builder, or Windows Media profile manager — less
    COM object construction, less that can fail for this simpler query.
    """
    try:
        import comtypes
        from pygrabber.dshow_graph import SystemDeviceEnum
        from pygrabber.dshow_ids import DeviceCategories
    except Exception as exc:
        _warn_once('pygrabber_import_failed', f'[UVC] pygrabber/comtypes import failed: {exc}')
        return []

    initialized_here = False
    try:
        try:
            comtypes.CoInitialize()
            initialized_here = True
        except Exception:
            pass  # COM already initialized on this thread — fine
        try:
            return list(SystemDeviceEnum().get_available_filters(DeviceCategories.VideoInputDevice))
        except Exception as exc:
            _warn_once('pygrabber_enum_failed', f'[UVC] pygrabber device enumeration failed: {exc}')
            return []
    finally:
        if initialized_here:
            try:
                comtypes.CoUninitialize()
            except Exception:
                pass


def _list_pygrabber_device_formats(device_index: int) -> list[dict]:
    """Query a device's real supported (resolution, fps range) capabilities
    via DirectShow's IAMStreamConfig::GetStreamCaps through pygrabber — the
    same native capability query OBS uses to populate its own resolution/FPS
    lists, keyed by the same device index cv2's CAP_DSHOW backend uses (both
    ultimately enumerate the same ICreateDevEnum video-input-device category
    in the same order).
    """
    try:
        import comtypes
        from pygrabber.dshow_graph import FilterGraph
    except Exception as exc:
        _warn_once('pygrabber_import_failed', f'[UVC] pygrabber/comtypes import failed: {exc}')
        return []

    initialized_here = False
    graph = None
    try:
        try:
            comtypes.CoInitialize()
            initialized_here = True
        except Exception:
            pass
        try:
            graph = FilterGraph()
            graph.add_video_input_device(device_index)
            raw_formats = graph.get_input_device().get_formats()
        except Exception as exc:
            _warn_once('pygrabber_formats_failed', f'[UVC] pygrabber format query failed: {exc}')
            return []
        results: list[dict] = []
        for fmt in raw_formats:
            try:
                results.append({
                    'width': int(fmt['width']),
                    'height': int(fmt['height']),
                    'min_fps': float(fmt['min_framerate']),
                    'max_fps': float(fmt['max_framerate']),
                })
            except Exception:
                continue
        return results
    finally:
        if graph is not None:
            try:
                graph.remove_filters()
            except Exception:
                pass
        if initialized_here:
            try:
                comtypes.CoUninitialize()
            except Exception:
                pass


def list_supported_uvc_resolutions(
    device_index: int,
    capture_method: str = 'dshow',
) -> list[tuple[int, int]]:
    """Return the device's actual supported resolutions.

    Queries the driver directly via DirectShow's native IAMStreamConfig
    capability list through pygrabber (the same COM API OBS/browsers use),
    falling back to a guess-and-check cv2 probe against a small candidate
    set only when pygrabber is unavailable.
    """
    formats = _list_pygrabber_device_formats(device_index)
    if formats:
        sizes = {(f['width'], f['height']) for f in formats}
        if sizes:
            return sorted(sizes, key=lambda item: (item[0] * item[1], item[0]))

    backend_map = {
        'dshow': cv2.CAP_DSHOW,
        'msmf': cv2.CAP_MSMF,
        'any': cv2.CAP_ANY,
    }
    backend = backend_map.get(str(capture_method).lower(), cv2.CAP_DSHOW)
    try:
        cap = cv2.VideoCapture(int(device_index), backend)
        if not cap.isOpened():
            cap = cv2.VideoCapture(int(device_index))
    except Exception:
        # Some DirectShow driver stacks make cv2.VideoCapture() itself raise
        # instead of just failing to open (seen falling through to OpenCV's
        # internal obsensor backend on certain devices/indices).
        return []
    if not cap.isOpened():
        return []

    common_resolutions = [
        (1280, 720), (1920, 1080), (2560, 1440),
    ]
    supported: set[tuple[int, int]] = set()
    try:
        for width, height in common_resolutions:
            try:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            except Exception:
                # cap.isOpened() can lie — some broken/half-open handles
                # only reveal it when set()/get() raises a raw C++
                # exception. Skip this candidate rather than letting an
                # unguarded native exception crash the probe thread (and,
                # left unhandled, the whole app).
                continue
            if actual_w > 0 and actual_h > 0 and abs(actual_w - width) <= 8 and abs(actual_h - height) <= 8:
                supported.add((actual_w, actual_h))
    finally:
        try:
            cap.release()
        except Exception:
            pass
    return sorted(supported, key=lambda item: (item[0] * item[1], item[0]))


def list_supported_uvc_fps(
    device_index: int,
    width: int,
    height: int,
    capture_method: str = 'dshow',
) -> list[int]:
    """Return the device's actual supported FPS values at a given resolution.

    Queries the driver directly via DirectShow's native IAMStreamConfig
    capability list through pygrabber (the same COM API OBS/browsers use),
    falling back to a guess-and-check cv2 probe against a small candidate
    set only when pygrabber is unavailable.
    """
    formats = _list_pygrabber_device_formats(device_index)
    if formats:
        matching = [f for f in formats if f['width'] == width and f['height'] == height] or formats
        fps_values: set[int] = set()
        for f in matching:
            min_fps, max_fps = f['min_fps'], f['max_fps']
            fps_values.add(int(round(min_fps)))
            fps_values.add(int(round(max_fps)))
            # DirectShow drivers commonly report FPS as a continuous
            # MinFrameInterval–MaxFrameInterval range rather than one
            # discrete capability entry per step — e.g. a single 5–240
            # entry, with nothing in between. Taking only the two
            # endpoints would silently drop real, settable values like
            # 144 that fall inside that range. Fill in from the common
            # preset list wherever it's actually covered by a reported
            # range, so the dropdown stays useful without inventing
            # values the driver never actually advertised.
            for preset in (30, 60, 120, 144, 165, 240):
                if min_fps <= preset <= max_fps:
                    fps_values.add(preset)
        if fps_values:
            return sorted(fps_values)

    backend_map = {'dshow': cv2.CAP_DSHOW, 'msmf': cv2.CAP_MSMF, 'any': cv2.CAP_ANY}
    backend = backend_map.get(str(capture_method).lower(), cv2.CAP_DSHOW)
    try:
        cap = cv2.VideoCapture(int(device_index), backend)
        if not cap.isOpened():
            cap = cv2.VideoCapture(int(device_index))
    except Exception:
        # Some DirectShow driver stacks make cv2.VideoCapture() itself raise
        # instead of just failing to open (seen falling through to OpenCV's
        # internal obsensor backend on certain devices/indices).
        return [30, 60]
    if not cap.isOpened():
        return [30, 60]
    try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    except Exception:
        # cap.isOpened() can lie — some broken/half-open handles only
        # reveal it when set() raises a raw C++ exception.
        try:
            cap.release()
        except Exception:
            pass
        return [30, 60]
    common = [30, 60, 120, 144, 165, 240]
    supported: list[int] = []
    try:
        for fps in common:
            try:
                cap.set(cv2.CAP_PROP_FPS, fps)
                actual = cap.get(cv2.CAP_PROP_FPS)
            except Exception:
                continue
            if actual > 0 and abs(actual - fps) <= 2:
                supported.append(fps)
    finally:
        try:
            cap.release()
        except Exception:
            pass
    return supported or [30, 60]


def _set_window_topmost(window_name: str, topmost: bool) -> None:
    """Set an OpenCV window always-on-top via Win32 SetWindowPos."""
    try:
        import ctypes
        hwnd = ctypes.windll.user32.FindWindowW(None, window_name)
        if hwnd:
            HWND_TOPMOST, HWND_NOTOPMOST = -1, -2
            SWP_NOMOVE, SWP_NOSIZE = 0x0002, 0x0001
            ctypes.windll.user32.SetWindowPos(
                hwnd,
                HWND_TOPMOST if topmost else HWND_NOTOPMOST,
                0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE,
            )
    except Exception:
        pass


class _UVCPreviewThread(threading.Thread):
    """Dedicated thread that refreshes the UVC preview window at a fixed rate.

    Decouples preview FPS from the inference loop. The reader thread fills
    _frame_ref[0] at the camera's native FPS; this thread presents it to the
    screen independently so the preview always runs at ~target_fps regardless
    of how fast (or slow) the AI inference loop is.
    """

    def __init__(
        self,
        window_name: str,
        scale_mode: str,
        frame_lock: threading.Lock,
        frame_ref: list,
        stop_event: threading.Event,
        draw_overlay_fn,
        region_ref: list,
        target_fps: int = 60,
        preview_width: int = 1920,
        preview_height: int = 1080,
        config=None,
        show_cv2_window: bool = True,
    ) -> None:
        super().__init__(daemon=True, name='UVCPreview')
        self._window_name    = window_name
        self._scale_mode     = scale_mode
        self._lock           = frame_lock
        self._frame_ref      = frame_ref      # list[np.ndarray | None]
        self._stop           = stop_event
        self._draw_overlay   = draw_overlay_fn
        self._region_ref     = region_ref     # list[dict | None]
        self._interval       = 1.0 / max(1, target_fps)
        self._preview_width  = preview_width
        self._preview_height = preview_height
        self._config         = config
        self._show_cv2       = show_cv2_window

    def run(self) -> None:
        # cv2 GUI operations must stay on this thread.
        if self._show_cv2:
            try:
                cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(self._window_name, self._preview_width, self._preview_height)
            except Exception:
                pass

        _crop_active = False
        _topmost_active: bool | None = None

        while not self._stop.is_set():
            t0 = time.perf_counter()
            with self._lock:
                frame = self._frame_ref[0]
            if frame is not None:
                try:
                    region  = self._region_ref[0]
                    preview = self._draw_overlay(frame.copy(), region)
                    # Share full overlay-rendered frame with the Qt preview panel
                    # (before any crop so the panel can apply its own crop).
                    set_preview_frame(preview)
                    set_preview_region(region)

                    if self._show_cv2:
                        # Crop to detection region when requested so the user sees
                        # exactly what the model infers on.
                        crop = bool(region is not None and getattr(self._config, 'preview_crop_to_detection', False))
                        if crop != _crop_active:
                            _crop_active = crop
                            try:
                                if crop and region is not None:
                                    rw = max(64, int(region.get('width', self._preview_width)))
                                    rh = max(64, int(region.get('height', self._preview_height)))
                                    cv2.resizeWindow(self._window_name, rw, rh)
                                else:
                                    cv2.resizeWindow(self._window_name, self._preview_width, self._preview_height)
                            except Exception:
                                pass

                        if crop and region is not None:
                            _l = max(0, int(region.get('left', 0)))
                            _t = max(0, int(region.get('top', 0)))
                            _w = max(1, int(region.get('width', preview.shape[1])))
                            _h = max(1, int(region.get('height', preview.shape[0])))
                            _r = min(preview.shape[1], _l + _w)
                            _b = min(preview.shape[0], _t + _h)
                            if _r > _l and _b > _t:
                                preview = preview[_t:_b, _l:_r]

                        rendered = _render_preview_frame(
                            self._window_name, self._scale_mode, preview)
                        cv2.imshow(self._window_name, rendered)
                        cv2.waitKey(1)
                        topmost = bool(getattr(self._config, 'uvc_always_on_top', True))
                        if topmost != _topmost_active:
                            _topmost_active = topmost
                            _set_window_topmost(self._window_name, topmost)
                except Exception:
                    pass
            remaining = self._interval - (time.perf_counter() - t0)
            if remaining > 0.001:
                time.sleep(remaining)

        if self._show_cv2:
            try:
                cv2.destroyWindow(self._window_name)
            except Exception:
                pass


def list_uvc_device_names() -> list[str]:
    """Enumerate UVC/webcam device names in DirectShow's enumeration order.

    Used by the GUI to populate a device-name combo box instead of a bare
    numeric index. The returned order also backs ``uvc_device_index`` for
    the cv2 dshow/msmf/any capture methods, since DirectShow's own
    enumeration order is the closest available approximation of how those
    backends number devices (OpenCV doesn't expose device names itself).

    Uses pygrabber's native COM enumeration (the same
    ICreateDevEnum/IEnumMoniker API OBS and browsers use) — returns an
    empty list if pygrabber/comtypes are unavailable.
    """

    return _list_pygrabber_device_names()


def _resolve_dshow_device_name(config: Config, device_index: int) -> str:
    """Resolve a DirectShow device *name* string from config.

    ffmpeg's dshow demuxer needs ``video=<device name>``, unlike OpenCV's
    integer device index — names are enumerated via pygrabber (same as the
    Device combo box) and indexed the same way ``uvc_device_index`` selects
    among them elsewhere.
    """

    devices = list_uvc_device_names()
    if not devices:
        raise RuntimeError(
            'No DirectShow video devices found via pygrabber enumeration. '
            'The ffmpeg capture method needs a resolvable device name — '
            'pick a device from the Capture page\'s Device dropdown.'
        )
    index = int(device_index)
    if index < 0 or index >= len(devices):
        raise RuntimeError(
            f'uvc_device_index={index} out of range for {len(devices)} '
            f'enumerated device(s): {devices}.'
        )
    return devices[index]


def _resolve_ffmpeg_path(config: Config) -> str:
    """Locate an ffmpeg executable for the 'ffmpeg' capture method.

    Checked in order: an explicit ``uvc_ffmpeg_path`` override, a bundled
    copy at ``<project root>/ffmpeg/ffmpeg.exe`` (drop a build there,
    e.g. from https://www.gyan.dev/ffmpeg/builds/ — an LGPL build, not GPL,
    to avoid extra licensing obligations), then the system PATH.
    """

    override = str(getattr(config, 'uvc_ffmpeg_path', '') or '').strip()
    if override:
        return override

    # screen_capture.py lives at <project root>/src/core/screen_capture.py
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    bundled = os.path.join(project_root, 'ffmpeg', 'ffmpeg.exe')
    if os.path.isfile(bundled):
        return bundled

    found = shutil.which('ffmpeg') or shutil.which('ffmpeg.exe')
    if found:
        return found

    raise RuntimeError(
        'ffmpeg executable not found. Set "FFmpeg Path" in the Capture page, '
        f'place ffmpeg.exe at "{bundled}", or add it to your system PATH. '
        'An LGPL build (e.g. from https://www.gyan.dev/ffmpeg/builds/) is '
        'sufficient — no bundled codecs required for raw capture.'
    )


def _read_exact(stream: Any, n: int) -> bytes | None:
    """Read exactly *n* bytes from a subprocess pipe, or None on EOF/error.

    A plain ``stream.read(n)`` on a pipe is not guaranteed to return all *n*
    bytes in one call — short reads are normal, not an error — so this loops
    until the full amount has been collected. Returns None as soon as a
    read returns empty (the writing end closed/the process died), rather
    than returning a truncated frame that would desync every frame after it.
    """

    chunks: list[bytes] = []
    remaining = n
    while remaining > 0:
        chunk = stream.read(remaining)
        if not chunk:
            return None
        chunks.append(chunk)
        remaining -= len(chunk)
    return b''.join(chunks)


def _crop_nv12(buffer: np.ndarray, luma_height: int, left: int, top: int, width: int, height: int) -> np.ndarray | None:
    """Crop a raw NV12 buffer (as returned by cv2 when CAP_PROP_CONVERT_RGB
    is disabled) to the given region, without decoding pixels outside it.

    NV12 packs two planes into one array: rows [0:luma_height] are the
    full-resolution Y (luma) plane, and rows [luma_height:luma_height*3//2]
    are a half-height plane of interleaved U/V (chroma) bytes — 2:1
    subsampling in both directions. Cropping correctly means slicing both
    planes with matching coordinates, and since the UV plane only has half
    the resolution, every input coordinate must land on an even boundary
    (rounded down here — at most 1px of slop, irrelevant for aim/detection
    purposes but would misalign chroma sampling if left unaligned).
    """

    left &= ~1
    top &= ~1
    width &= ~1
    height &= ~1
    if width <= 0 or height <= 0:
        return None
    if left + width > buffer.shape[1] or top + height > luma_height:
        return None
    y_plane = buffer[top:top + height, left:left + width]
    uv_top = luma_height + top // 2
    uv_plane = buffer[uv_top:uv_top + height // 2, left:left + width]
    return np.vstack((y_plane, uv_plane))


class UVCCapture:
    """OpenCV VideoCapture backend for UVC capture cards/cameras."""

    def __init__(self, config: Config) -> None:
        self.config = config
        device_index = int(getattr(config, 'uvc_device_index', 0))
        width = int(getattr(config, 'uvc_width', 1920))
        height = int(getattr(config, 'uvc_height', 1080))
        fps = int(getattr(config, 'uvc_fps', 60))
        self._target_fps = fps  # requested rate, for _reader_worker's shortfall check
        self.show_window = bool(getattr(config, 'uvc_show_window', False))
        self.window_name = _UVC_WINDOW_NAME
        self.config_signature = _uvc_signature(config)

        capture_method = str(getattr(config, 'uvc_capture_method', 'dshow')).lower()
        self.preview_scale_mode = str(getattr(config, 'uvc_preview_scale_mode', 'scale_to_fit')).lower()
        self.is_ffmpeg = (capture_method == 'ffmpeg')
        # Only the cv2 (dshow/msmf) path below ever sets this — ffmpeg mode
        # does its own crop via -vf before Axiom ever sees a frame, so grab()
        # just uses whatever region get_capture_dimensions()'s fixed-mode
        # reporting already resolves to a no-op crop.
        self._fixed_region: dict[str, int] | None = None

        if self.is_ffmpeg:
            self._init_ffmpeg(device_index, width, height, fps)
            return

        backend_map = {
            'dshow': cv2.CAP_DSHOW,
            'msmf': cv2.CAP_MSMF,
            'any': cv2.CAP_ANY,
        }
        backend = backend_map.get(capture_method, cv2.CAP_DSHOW)

        self.cap = cv2.VideoCapture(device_index, backend)
        if not self.cap.isOpened():
            # Fallback backend when CAP_DSHOW is unavailable
            self.cap = cv2.VideoCapture(device_index)

        if not self.cap.isOpened():
            raise RuntimeError(f'UVC device open failed: index={device_index}')

        video_format = str(getattr(config, 'uvc_video_format', 'mjpeg')).lower()
        fourcc_map = {
            'mjpeg': cv2.VideoWriter_fourcc(*'MJPG'),
            'yuy2': cv2.VideoWriter_fourcc(*'YUY2'),
            'nv12': cv2.VideoWriter_fourcc(*'NV12'),
            'yuv420p': cv2.VideoWriter_fourcc(*'I420'),  # planar 4:2:0, matches ffmpeg's "yuv420p" naming
        }
        target_fourcc = fourcc_map.get(video_format, fourcc_map['mjpeg'])

        # FOURCC must be set before resolution/FPS so the driver switches codec
        # first — true for most UVC drivers, but not universal.
        try:
            self.cap.set(cv2.CAP_PROP_FOURCC, target_fourcc)
        except Exception:
            pass
        if width > 0:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height > 0:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if fps > 0:
            self.cap.set(cv2.CAP_PROP_FPS, fps)

        # If that didn't take, some drivers need the opposite order — codec
        # negotiated only after resolution/FPS are already locked in — so
        # retry once with FOURCC set last before giving up on the requested format.
        if int(self.cap.get(cv2.CAP_PROP_FOURCC)) != target_fourcc:
            try:
                self.cap.set(cv2.CAP_PROP_FOURCC, target_fourcc)
            except Exception:
                pass
        # Keep the driver queue shallow so grab() always returns the newest frame.
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        self.preview_width = max(1, int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or width or 1))
        self.preview_height = max(1, int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or height or 1))
        self.preview_fps = max(1, int(self.cap.get(cv2.CAP_PROP_FPS) or fps or 1))

        # --- Fixed (centered) crop mode ---
        # Unlike ffmpeg mode, cv2's dshow/msmf capture is already in-process
        # with no subprocess pipe, and grab() already crops before any
        # colorspace conversion — so freezing the crop rect here doesn't
        # change per-frame cost. The only real effect: a live Detection
        # Range change no longer moves the crop until capture restarts,
        # exactly like ffmpeg's fixed mode (see uvc_crop_mode's docstring
        # in config.py). Preview still renders the full frame either way —
        # only grab()'s AI-pipeline output is restricted to the fixed square.
        if str(getattr(config, 'uvc_crop_mode', 'dynamic')).lower() == 'fixed':
            crop_size = int(getattr(config, 'detect_range_size', 0) or 0) & ~1
            crop_size = max(0, min(crop_size, self.preview_width, self.preview_height))
            if crop_size > 0:
                self._fixed_region = {
                    'left': (self.preview_width - crop_size) // 2,
                    'top': (self.preview_height - crop_size) // 2,
                    'width': crop_size,
                    'height': crop_size,
                }
        # Seed with the driver-probed capability so the status panel shows
        # something before the first live measurement below completes; the
        # reader worker overwrites this every second with the actual received rate.
        config.source_nominal_fps = float(self.preview_fps)

        # Publish the actual negotiated resolution/FPS from this already-open
        # handle so the GUI ("Query Device") can read it instead of opening a
        # second competing cv2.VideoCapture to the same device index.
        config.uvc_actual_width = self.preview_width
        config.uvc_actual_height = self.preview_height
        config.uvc_actual_fps = float(self.preview_fps)

        # Verify FOURCC was actually accepted by the driver.  Without MJPEG
        # (or another compressed format), raw 1080p (e.g. YUY2) requires
        # ~237 MB/s — beyond USB 2.0 bandwidth — so the driver silently
        # throttles to 5–15 fps with no error raised.
        actual_fourcc_int = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        self.is_mjpeg = (actual_fourcc_int == target_fourcc)  # kept for back-compat; means "requested format accepted"
        self.is_expected_format = self.is_mjpeg
        if not self.is_expected_format:
            actual_str = ''.join(
                chr((actual_fourcc_int >> i) & 0xFF) for i in [0, 8, 16, 24]
            ).strip('\x00') or 'unknown'
            # "Switch to msmf" is useless advice when msmf is already active —
            # this used to always say it regardless of capture_method. If
            # msmf itself can't negotiate the requested format with this
            # device, that's a different, more likely hardware/driver-specific
            # problem: try the other Windows capture API instead, and flag
            # that this may simply be unsupported by this device rather than
            # a settings fix.
            if capture_method == 'msmf':
                suggestion = (
                    "Already using 'msmf' — this device/driver may not support "
                    f"{video_format.upper()} through Media Foundation. Try 'dshow' "
                    "instead, or this may be a hardware limitation of this capture device."
                )
            else:
                suggestion = "Switch capture method to 'msmf'."
            logging.getLogger(__name__).warning(
                "[UVC] Video format %s not accepted by driver (got '%s', capture_method='%s'). "
                "If this is a raw format at high resolution, it may exceed USB bandwidth. %s",
                video_format.upper(), actual_str, capture_method, suggestion,
            )

        # --- Raw NV12 crop-before-convert ---
        # By default cv2.VideoCapture converts every captured frame to BGR
        # internally (at the *full* negotiated resolution) before read()
        # ever returns it — even though only a small detect_range_size
        # crop of it is actually used downstream. Disabling that lets
        # read() hand back the raw NV12 buffer instead, so the (relatively
        # expensive, full-resolution) YUV->BGR colorspace conversion can
        # happen *after* cropping — over ~320x320 pixels instead of
        # ~2 million — cutting real per-frame CPU cost on the capture
        # path. Only attempted when NV12 was actually negotiated
        # (is_expected_format); gated behind a readback check since some
        # backends silently ignore CAP_PROP_CONVERT_RGB, which would
        # otherwise make grab()'s raw-buffer crop math run against an
        # already-BGR frame and produce garbage.
        self.is_raw_nv12 = False
        if video_format == 'nv12' and self.is_expected_format:
            try:
                self.cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
                self.is_raw_nv12 = (self.cap.get(cv2.CAP_PROP_CONVERT_RGB) == 0)
            except Exception:
                self.is_raw_nv12 = False
            if not self.is_raw_nv12:
                logging.getLogger(__name__).info(
                    "[UVC] Driver doesn't support disabling auto BGR conversion "
                    "(CAP_PROP_CONVERT_RGB) — NV12 crop-before-convert optimization "
                    "unavailable, falling back to full-frame conversion."
                )

        # --- Non-blocking reader thread ---
        # cap.read() blocks up to one frame period (e.g. 16 ms at 60 fps).
        # A background thread continuously reads into _latest_frame_ref so that
        # grab() can return the newest frame without blocking the inference loop.
        self._latest_frame_lock = threading.Lock()
        self._latest_frame_ref: list = [None]   # list[np.ndarray | None]
        self._region_ref: list = [None]         # list[dict | None]
        self._reader_stop = threading.Event()
        # Seeded to "now" (not None) so a device that opens but never delivers
        # a single frame within the stale timeout is caught by the same check
        # as one that goes silent after initially working — see
        # reinitialize_if_method_changed's staleness check.
        self._last_frame_perf_time = time.perf_counter()
        self._reader_thread = threading.Thread(
            target=self._reader_worker, name='UVCReader', daemon=True
        )
        self._reader_thread.start()

        # --- Dedicated preview thread ---
        # Renders the preview window (background + overlays) at camera FPS,
        # fully decoupled from inference. namedWindow/imshow/destroyWindow all
        # run on this thread so OpenCV's GUI-thread affinity requirement is met.
        self._preview_thread: _UVCPreviewThread | None = None
        if self.show_window:
            self._preview_thread = _UVCPreviewThread(
                window_name=self.window_name,
                scale_mode=self.preview_scale_mode,
                frame_lock=self._latest_frame_lock,
                frame_ref=self._latest_frame_ref,
                stop_event=self._reader_stop,
                draw_overlay_fn=self._draw_overlay,
                region_ref=self._region_ref,
                target_fps=self.preview_fps,
                preview_width=self.preview_width,
                preview_height=self.preview_height,
                config=self.config,
                show_cv2_window=False,  # Qt panel is the primary display; cv2 window suppressed
            )
            self._preview_thread.start()

    def _init_ffmpeg(self, device_index: int, width: int, height: int, fps: int) -> None:
        """Open a UVC device via an external ffmpeg.exe subprocess instead of
        cv2.VideoCapture, piping raw frames back over stdout.

        This driver has repeatedly been shown to lie about negotiated state
        through OpenCV's videoio layer (isOpened(), FOURCC acceptance,
        CAP_PROP_CONVERT_RGB) — ffmpeg's own dshow demuxer negotiates
        directly and, unlike cv2, fails loudly (non-zero exit, stderr
        message) rather than silently substituting something else when a
        requested (resolution, format, fps) combination isn't actually
        supported. No PyAV/Cython involved — this is a plain OS process
        with a pipe, sidestepping the exact packaging conflict that made
        the earlier PyAV capture method unreliable.
        """

        if width <= 0 or height <= 0:
            raise RuntimeError(
                'ffmpeg capture method requires an explicit UVC resolution '
                '(uvc_width/uvc_height) — it cannot auto-negotiate a size the '
                'way cv2.VideoCapture sometimes does.'
            )

        ffmpeg_path = _resolve_ffmpeg_path(self.config)
        device_name = _resolve_dshow_device_name(self.config, device_index)
        video_format = str(getattr(self.config, 'uvc_video_format', 'mjpeg')).lower()
        crop_mode = str(getattr(self.config, 'uvc_crop_mode', 'dynamic')).lower()

        input_args: list[str] = []
        if video_format == 'mjpeg':
            input_args += ['-vcodec', 'mjpeg']
        else:
            pixel_format_map = {'nv12': 'nv12', 'yuy2': 'yuyv422', 'yuv420p': 'yuv420p'}
            input_args += ['-pixel_format', pixel_format_map.get(video_format, 'nv12')]
        input_args += ['-video_size', f'{width}x{height}']
        if fps > 0:
            input_args += ['-framerate', str(fps)]

        # Fixed-crop mode: ffmpeg itself crops a centered detect_range_size
        # square and only that much data ever crosses the subprocess pipe.
        # Only sensible centered — matches fov_follow_mouse already being
        # forced off whenever screenshot_method == 'uvc' (capture_page.py's
        # _applyScreenshotMethodEffect), so the crosshair is always the
        # frame center already. get_capture_dimensions() (ai_loop_utils.py)
        # is taught to report (crop_size, crop_size) as the "capture size"
        # in this mode, so calculate_detection_region()'s own region math
        # naturally resolves to a full-frame no-op crop against the
        # already-cropped stream — no special-casing needed in grab().
        crop_size = 0
        if crop_mode == 'fixed':
            crop_size = int(getattr(self.config, 'detect_range_size', 0) or 0)
            if crop_size <= 0 or crop_size > min(width, height):
                raise RuntimeError(
                    f'ffmpeg fixed-crop mode needs a valid Detection Range '
                    f'(got {crop_size}) no larger than the capture resolution '
                    f'({width}x{height}).'
                )
            crop_size &= ~1  # even alignment, consistent with the rest of this codebase's crop math
            out_width, out_height = crop_size, crop_size
            vf = f'crop={crop_size}:{crop_size}:(iw-{crop_size})/2:(ih-{crop_size})/2,format=bgr24'
        else:
            out_width, out_height = width, height
            vf = 'format=bgr24'

        cmd = [
            ffmpeg_path, '-hide_banner', '-loglevel', 'warning',
            '-f', 'dshow', *input_args, '-i', f'video={device_name}',
            '-vf', vf,
            '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-an', 'pipe:1',
        ]

        self.cap = None
        creationflags = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
        try:
            self._ffmpeg_proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                creationflags=creationflags,
            )
        except Exception as exc:
            raise RuntimeError(f'Failed to launch ffmpeg ("{ffmpeg_path}"): {exc}') from exc

        # Give the process a moment to fail fast on a bad device name/format
        # combination — ffmpeg's dshow demuxer errors out (rather than
        # silently misnegotiating) when the exact requested combination
        # isn't one the driver actually advertises, unlike cv2.
        time.sleep(1.0)
        if self._ffmpeg_proc.poll() is not None:
            stderr_output = ''
            try:
                stderr_output = self._ffmpeg_proc.stderr.read().decode('utf-8', errors='replace')
            except Exception:
                pass
            raise RuntimeError(
                f'ffmpeg exited immediately (code {self._ffmpeg_proc.returncode}) — '
                f'the requested device/format/resolution/fps combination was '
                f'likely rejected. stderr:\n{stderr_output.strip()}'
            )

        self.preview_width = out_width
        self.preview_height = out_height
        self.preview_fps = max(1, fps)
        self.config.source_nominal_fps = float(self.preview_fps)
        self.config.uvc_actual_width = self.preview_width
        self.config.uvc_actual_height = self.preview_height
        self.config.uvc_actual_fps = float(self.preview_fps)
        self.is_expected_format = True  # ffmpeg fails loudly rather than silently misnegotiating
        self.is_mjpeg = self.is_expected_format  # kept for back-compat
        self.is_raw_nv12 = False  # ffmpeg always outputs plain bgr24, not raw NV12

        logging.getLogger(__name__).info(
            "[UVC][FFmpeg] Opened '%s' via ffmpeg subprocess at %dx%d @ %d fps "
            "(format=%s, crop_mode=%s).",
            device_name, out_width, out_height, self.preview_fps, video_format.upper(), crop_mode,
        )

        self._latest_frame_lock = threading.Lock()
        self._latest_frame_ref: list = [None]
        self._region_ref: list = [None]
        self._reader_stop = threading.Event()
        # See the cv2 reader thread's identical field for why this is seeded
        # to "now" rather than None.
        self._last_frame_perf_time = time.perf_counter()
        self._reader_thread = threading.Thread(
            target=self._reader_worker_ffmpeg, name='UVCReaderFFmpeg', daemon=True
        )
        self._reader_thread.start()
        self._stderr_thread = threading.Thread(
            target=self._stderr_drain_worker_ffmpeg, name='UVCFFmpegStderr', daemon=True
        )
        self._stderr_thread.start()

        self._preview_thread: _UVCPreviewThread | None = None
        if self.show_window:
            self._preview_thread = _UVCPreviewThread(
                window_name=self.window_name,
                scale_mode=self.preview_scale_mode,
                frame_lock=self._latest_frame_lock,
                frame_ref=self._latest_frame_ref,
                stop_event=self._reader_stop,
                draw_overlay_fn=self._draw_overlay,
                region_ref=self._region_ref,
                target_fps=self.preview_fps,
                preview_width=self.preview_width,
                preview_height=self.preview_height,
                config=self.config,
                show_cv2_window=False,
            )
            self._preview_thread.start()

    def _reader_worker_ffmpeg(self) -> None:
        frame_size = self.preview_width * self.preview_height * 3
        _fps_count = 0
        _fps_t0 = time.perf_counter()
        _measurement_windows = 0
        _warned_broken_pipe = False
        while not self._reader_stop.is_set():
            raw = _read_exact(self._ffmpeg_proc.stdout, frame_size)
            if raw is None:
                if not self._reader_stop.is_set() and not _warned_broken_pipe:
                    _warned_broken_pipe = True
                    logging.getLogger(__name__).error(
                        "[UVC][FFmpeg] subprocess stdout closed unexpectedly — "
                        "the ffmpeg process likely crashed or the device was "
                        "disconnected. No more frames will arrive until UVC "
                        "reinitializes."
                    )
                time.sleep(0.05)
                continue
            frame = np.frombuffer(raw, dtype=np.uint8).reshape((self.preview_height, self.preview_width, 3))
            with self._latest_frame_lock:
                self._latest_frame_ref[0] = frame
            self._last_frame_perf_time = time.perf_counter()

            _fps_count += 1
            _now = time.perf_counter()
            _elapsed = _now - _fps_t0
            if _elapsed >= 1.0:
                self.config.source_nominal_fps = _fps_count / _elapsed
                _fps_count = 0
                _fps_t0 = _now

                _measurement_windows += 1
                if _measurement_windows == 2 and self._target_fps > 0:
                    shortfall_ratio = self.config.source_nominal_fps / self._target_fps
                    if shortfall_ratio < 0.8:
                        logging.getLogger(__name__).warning(
                            "[UVC][FFmpeg] Measured capture rate %.1f fps is well "
                            "below the configured %d fps. Try a lower resolution/fps, "
                            "a different USB port/cable, or the dynamic (not fixed) "
                            "crop mode.",
                            self.config.source_nominal_fps, self._target_fps,
                        )

    def _stderr_drain_worker_ffmpeg(self) -> None:
        # Must be continuously drained or the OS pipe buffer fills and blocks
        # ffmpeg's own writes, stalling the whole subprocess.
        try:
            for line in iter(self._ffmpeg_proc.stderr.readline, b''):
                if self._reader_stop.is_set():
                    break
                text = line.decode('utf-8', errors='replace').strip()
                if text:
                    logging.getLogger(__name__).debug("[UVC][FFmpeg] %s", text)
        except Exception:
            pass

    def _reader_worker(self) -> None:
        _fps_count = 0
        _fps_t0 = time.perf_counter()
        _measurement_windows = 0
        _warned_broken_read = False
        while not self._reader_stop.is_set():
            try:
                ok, frame = self.cap.read()
            except Exception as exc:
                # cap.isOpened() can lie (seen on some DirectShow driver
                # stacks that fall through to OpenCV's internal obsensor
                # backend) — cap.read() can then raise a raw C++ exception
                # instead of just returning ok=False. Treat it the same as
                # a failed read rather than letting it kill this thread
                # (which would silently stop all UVC capture).
                if not _warned_broken_read:
                    _warned_broken_read = True
                    logging.getLogger(__name__).error(
                        "[UVC] cap.read() raised %s — the driver/backend may not "
                        "actually support this device by index. Try a different "
                        "capture method (msmf/dshow) or USB port. Retrying...",
                        exc,
                    )
                ok, frame = False, None
                time.sleep(0.05)
            if ok and frame is not None:
                with self._latest_frame_lock:
                    self._latest_frame_ref[0] = frame
                self._last_frame_perf_time = time.perf_counter()

                _fps_count += 1
                _now = time.perf_counter()
                _elapsed = _now - _fps_t0
                if _elapsed >= 1.0:
                    self.config.source_nominal_fps = _fps_count / _elapsed
                    _fps_count = 0
                    _fps_t0 = _now

                    # Same class of problem as the FOURCC check in __init__:
                    # cap.set(CAP_PROP_FPS, ...) can be silently accepted by
                    # the driver as a *setting* without the hardware actually
                    # sustaining it — the requested value just gets echoed
                    # back on cap.get(), never validated. The only way to
                    # know the real rate is to measure actual frame arrivals,
                    # which is exactly what source_nominal_fps tracks. Check
                    # once, skipping the first window (startup ramp-up can be
                    # artificially low) and not repeating (avoid log spam for
                    # a persistent, already-reported condition).
                    _measurement_windows += 1
                    if _measurement_windows == 2 and self._target_fps > 0:
                        shortfall_ratio = self.config.source_nominal_fps / self._target_fps
                        if shortfall_ratio < 0.8:
                            logging.getLogger(__name__).warning(
                                "[UVC] Measured capture rate %.1f fps is well below the "
                                "configured %d fps. If the requested video format isn't "
                                "actually active (see any 'Video format ... not accepted' "
                                "warning above), raw video bandwidth at this resolution may "
                                "not fit your USB link — try MJPEG, 'msmf' capture method, "
                                "a lower resolution, or a different USB port/cable.",
                                self.config.source_nominal_fps, self._target_fps,
                            )
            else:
                time.sleep(0.005)

    def _confirm_raw_nv12(self, frame: np.ndarray) -> bool:
        """Verify a captured frame is actually a raw single-plane NV12
        buffer, not ordinary (H, W, 3) BGR.

        The CAP_PROP_CONVERT_RGB readback check in __init__ isn't
        sufficient on its own — some DirectShow driver stacks echo back
        whatever value was set without it having any real effect (the
        same class of lie already seen from this backend for isOpened()
        and FOURCC). A raw NV12 buffer is 2-D (single channel); if a
        frame claiming to be raw NV12 is actually 3-D, self-heal by
        permanently disabling the optimization instead of letting
        cv2.cvtColor crash on the channel-count mismatch.
        """
        if frame.ndim == 2:
            return True
        self.is_raw_nv12 = False
        logging.getLogger(__name__).warning(
            "[UVC] Driver claimed CAP_PROP_CONVERT_RGB was disabled but frames "
            "are still %s-channel, not raw NV12 — disabling the crop-before-"
            "convert optimization and falling back to standard full-frame "
            "conversion.",
            frame.shape[2] if frame.ndim == 3 else frame.ndim,
        )
        return False

    def grab(self, region: dict[str, int] | None = None, **_: Any) -> np.ndarray | None:
        """Return BGRA frame cropped by region when provided.

        Always returns the most recent frame captured by the reader thread
        without blocking the caller.  UVC preview renders on the full frame
        so the preview window is independent of the AI detection crop region.
        """

        with self._latest_frame_lock:
            frame_bgr = self._latest_frame_ref[0]
        if frame_bgr is None:
            return None

        # Fixed crop mode ignores the live (Detection-Range-derived) region
        # in favor of the rect frozen at capture-start — see __init__.
        effective_region = self._fixed_region if self._fixed_region is not None else region

        # Let the preview thread know the current detection region so its
        # overlay stays in sync without requiring an extra lock or callback.
        self._region_ref[0] = effective_region

        if self.is_raw_nv12 and self._confirm_raw_nv12(frame_bgr):
            # Crop the raw NV12 buffer BEFORE converting to BGR, so the
            # (comparatively expensive) colorspace conversion only ever
            # runs over the small detection-region crop instead of the
            # full negotiated resolution — see the CAP_PROP_CONVERT_RGB
            # setup in __init__ for why this buffer isn't BGR already.
            if effective_region is None:
                return cv2.cvtColor(frame_bgr, cv2.COLOR_YUV2BGRA_NV12)
            left = max(0, int(effective_region.get('left', 0)))
            top = max(0, int(effective_region.get('top', 0)))
            width = max(0, int(effective_region.get('width', self.preview_width)))
            height = max(0, int(effective_region.get('height', self.preview_height)))
            width = min(width, self.preview_width - left)
            height = min(height, self.preview_height - top)
            cropped = _crop_nv12(frame_bgr, self.preview_height, left, top, width, height)
            if cropped is None:
                return None
            return cv2.cvtColor(cropped, cv2.COLOR_YUV2BGRA_NV12)

        if effective_region is not None:
            frame_h, frame_w = frame_bgr.shape[:2]
            left = max(0, int(effective_region.get('left', 0)))
            top = max(0, int(effective_region.get('top', 0)))
            width = max(0, int(effective_region.get('width', frame_w)))
            height = max(0, int(effective_region.get('height', frame_h)))
            right = min(frame_w, left + width)
            bottom = min(frame_h, top + height)
            if right <= left or bottom <= top:
                return None
            frame_bgr = frame_bgr[top:bottom, left:right]

        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2BGRA)

    def _draw_overlay(self, frame_bgr: np.ndarray, region: dict[str, int] | None) -> np.ndarray:
        if self.is_raw_nv12 and self._confirm_raw_nv12(frame_bgr):
            # The preview thread reads the same raw NV12 buffer grab() does
            # (shared via _latest_frame_ref) — it needs a real BGR frame to
            # draw overlays on, so convert the full frame here. Unlike
            # grab()'s crop-then-convert path, this always pays the
            # full-resolution conversion cost, but only while the preview
            # panel/window is actually enabled (self.show_window).
            frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_YUV2BGR_NV12)
        return _draw_detection_overlay(frame_bgr, region, self.config, has_alpha=False)

    def _render_preview_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        return _render_preview_frame(self.window_name, self.preview_scale_mode, frame_bgr)

    def close(self) -> None:
        self._reader_stop.set()
        if self._preview_thread is not None and self._preview_thread.is_alive():
            self._preview_thread.join(timeout=1.0)
        if self._reader_thread.is_alive():
            self._reader_thread.join(timeout=1.0)
        if getattr(self, 'is_ffmpeg', False):
            proc = getattr(self, '_ffmpeg_proc', None)
            if proc is not None:
                try:
                    proc.terminate()
                    try:
                        proc.wait(timeout=2.0)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait(timeout=2.0)
                except Exception:
                    pass
                for pipe in (proc.stdout, proc.stderr):
                    try:
                        if pipe is not None:
                            pipe.close()
                    except Exception:
                        pass
            stderr_thread = getattr(self, '_stderr_thread', None)
            if stderr_thread is not None and stderr_thread.is_alive():
                stderr_thread.join(timeout=1.0)
        elif self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
        # destroyWindow is handled by the preview thread's run() exit path


class UdpCapture:
    """UDP JPEG stream capture backend (OBS udp_stream_filter wire protocol)."""

    def __init__(self, config: Config) -> None:
        from .udp_receiver import UdpJpegReceiver

        self.config = config
        bind_ip = str(getattr(config, 'udp_bind_ip', '0.0.0.0'))
        bind_port = int(getattr(config, 'udp_bind_port', 5600))
        recv_buffer_size = int(getattr(config, 'udp_recv_buffer_size', 65536))
        frame_timeout = float(getattr(config, 'udp_frame_timeout', 1.0))
        self.window_name = _UVC_WINDOW_NAME
        self.preview_scale_mode = str(getattr(config, 'uvc_preview_scale_mode', 'scale_to_fit')).lower()
        self.config_signature = _udp_signature(config)

        self._receiver = UdpJpegReceiver(
            bind_ip=bind_ip,
            bind_port=bind_port,
            recv_buffer_size=recv_buffer_size,
            frame_timeout=frame_timeout,
        )
        self._receiver.start()
        # Clear restart flag so a queued Refresh doesn't loop forever
        config.udp_force_restart = False

        self._latest_frame_lock = threading.Lock()
        self._latest_frame_ref: list = [None]   # list[np.ndarray | None]  BGR
        self._region_ref: list = [None]
        self._stop = threading.Event()
        # Seeded to "now" (not None) so a stream that never sends a single
        # frame within the stale timeout is caught the same way as one that
        # goes silent after initially working — see
        # reinitialize_if_method_changed's staleness check.
        self._last_frame_perf_time = time.perf_counter()

        self._reader_thread = threading.Thread(
            target=self._reader_worker, name='UDPReader', daemon=True
        )
        self._reader_thread.start()

        self.preview_width = 1920
        self.preview_height = 1080
        config.source_nominal_fps = 0.0

        # Always start the preview thread so the Qt side panel receives frames
        # even when the "Capture Preview Window" toggle is off.  show_cv2_window=False
        # means no OpenCV window is ever opened; the thread only feeds set_preview_frame().
        self._preview_thread = _UVCPreviewThread(
            window_name=self.window_name,
            scale_mode=self.preview_scale_mode,
            frame_lock=self._latest_frame_lock,
            frame_ref=self._latest_frame_ref,
            stop_event=self._stop,
            draw_overlay_fn=self._draw_overlay,
            region_ref=self._region_ref,
            target_fps=60,
            preview_width=self.preview_width,
            preview_height=self.preview_height,
            config=self.config,
            show_cv2_window=False,
        )
        self._preview_thread.start()

    def _reader_worker(self) -> None:
        # Non-blocking poll with time.sleep(0) between frames.
        # On Windows, Sleep(0) yields to other runnable threads then returns
        # in < 1 ms — no 15 ms scheduler-quantum penalty that Event.wait() incurs
        # when used as a blocking wait. This allows the loop to react to a new
        # frame within microseconds of it being assembled by _recv_loop.
        _fps_count = 0
        _fps_t0 = time.perf_counter()
        _seen_id = None
        _logged_stream_dims = None

        while not self._stop.is_set():
            jpeg_bytes, frame_id = self._receiver.get_latest_frame_with_id()

            if jpeg_bytes is None or frame_id == _seen_id:
                time.sleep(0)  # GIL-releasing yield, no scheduler-quantum penalty
                continue

            _seen_id = frame_id

            # Parse the JPEG's real dimensions straight from its header (no decode
            # cost) so the actual streamed resolution is known with certainty —
            # JPEG decode throughput is dominated by entropy (Huffman) decoding of
            # the full bitstream regardless of target resolution, so a large source
            # resolution is a hard CPU ceiling on fps that no receiver-side decode
            # trick can bypass. Only the sender streaming fewer pixels helps.
            stream_dims = _jpeg_dimensions(jpeg_bytes)
            if stream_dims is not None and stream_dims != _logged_stream_dims:
                _logged_stream_dims = stream_dims
                logger.info("[UDP] stream resolution: %dx%d", stream_dims[0], stream_dims[1])

            arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
            frame_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame_bgr is None:
                continue

            h, w = frame_bgr.shape[:2]
            self._last_frame_perf_time = time.perf_counter()
            with self._latest_frame_lock:
                self._latest_frame_ref[0] = frame_bgr
                if w != self.preview_width or h != self.preview_height:
                    self.preview_width = w
                    self.preview_height = h
                    # Published so get_capture_dimensions() can size the
                    # detection region against the real, current stream
                    # resolution — the sender (e.g. an OBS udp_stream_filter
                    # crop) can change this at any time, so it can't be a
                    # fixed/user-configured value like uvc_width/uvc_height.
                    # Without this, a region computed against config.width/
                    # height (full desktop res) can fall entirely outside a
                    # smaller cropped stream, making grab() return None every
                    # frame and inference FPS drop to 0.
                    self.config.udp_width = w
                    self.config.udp_height = h

            _fps_count += 1
            _now = time.perf_counter()
            _elapsed = _now - _fps_t0
            if _elapsed >= 1.0:
                self.config.source_nominal_fps = _fps_count / _elapsed
                self.config.udp_recv_fps = self._receiver.recv_fps
                self.config.udp_dropped_fps = self._receiver.dropped_fps
                if self._receiver.dropped_fps > 0:
                    logger.warning(
                        "[UDP] %.1f incomplete frames/sec dropped (packet loss) — recv %.1f fps",
                        self._receiver.dropped_fps, self._receiver.recv_fps,
                    )
                _fps_count = 0
                _fps_t0 = _now

    def grab(self, region: dict[str, int] | None = None, **_: Any) -> np.ndarray | None:
        with self._latest_frame_lock:
            frame_bgr = self._latest_frame_ref[0]
        if frame_bgr is None:
            return None

        self._region_ref[0] = region

        if region is not None:
            frame_h, frame_w = frame_bgr.shape[:2]
            left = max(0, int(region.get('left', 0)))
            top = max(0, int(region.get('top', 0)))
            width = max(0, int(region.get('width', frame_w)))
            height = max(0, int(region.get('height', frame_h)))
            right = min(frame_w, left + width)
            bottom = min(frame_h, top + height)
            if right <= left or bottom <= top:
                return None
            frame_bgr = frame_bgr[top:bottom, left:right]

        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2BGRA)

    def _draw_overlay(self, frame_bgr: np.ndarray, region: dict[str, int] | None) -> np.ndarray:
        return _draw_detection_overlay(frame_bgr, region, self.config, has_alpha=False)

    def close(self) -> None:
        self._stop.set()
        # Stop the receiver first so its _new_frame_event fires immediately,
        # waking any _reader_worker thread blocked in get_latest_frame().
        self._receiver.stop()
        if self._preview_thread is not None and self._preview_thread.is_alive():
            self._preview_thread.join(timeout=1.0)
        if self._reader_thread.is_alive():
            self._reader_thread.join(timeout=1.0)


def _get_monitor_refresh_rate() -> int:
    """Return the primary monitor refresh rate in Hz, or 0 on failure."""
    try:
        import ctypes
        user32 = ctypes.windll.user32
        hdc = user32.GetDC(None)
        if hdc:
            gdi32 = ctypes.windll.gdi32
            VREFRESH = 116  # GetDeviceCaps constant
            rate = gdi32.GetDeviceCaps(hdc, VREFRESH)
            user32.ReleaseDC(None, hdc)
            return max(0, int(rate))
    except Exception:
        pass
    return 0


def _warn_once(key: str, message: str) -> None:
    """Emit a warning log once per process to avoid log flooding."""

    if key in _WARNED_MESSAGES:
        return
    _WARNED_MESSAGES.add(key)
    logger.warning(message)


def _initialize_dxcam_capture(config: Any | None = None) -> Any | None:
    """Initialize dxcam backend, return None when unavailable."""

    try:
        import dxcam  # type: ignore[import-not-found]
    except ImportError:
        _warn_once(
            'dxcam_import_error',
            '[Capture] DXcam backend requested but package is not installed. Falling back to MSS.',
        )
        return None

    try:
        cam = dxcam.create(output_color='BGRA')
        if cam is not None and config is not None:
            _refresh = _get_monitor_refresh_rate()
            if _refresh > 0:
                config.source_nominal_fps = float(_refresh)
        return cam
    except Exception as exc:
        _warn_once(
            'dxcam_create_error',
            f'[Capture] DXcam initialization failed with "{exc}". Falling back to MSS backend.',
        )
        return None


def _cleanup_capture(screen_capture: Any) -> None:
    """Release resources held by a screen capture backend."""

    if screen_capture is None:
        return

    # mss instances have a close() method
    close_fn = getattr(screen_capture, 'close', None)
    if callable(close_fn):
        try:
            close_fn()
        except Exception:
            pass

    # dxcam instances may expose a release() method
    release_fn = getattr(screen_capture, 'release', None)
    if callable(release_fn):
        try:
            release_fn()
        except Exception:
            pass


def initialize_screen_capture(config: Config) -> Any:
    """Initialize screen capture backend and normalize config.

    Returns ``(capture_backend, active_method_name)`` so the caller can
    track which method is currently active.
    """

    screenshot_method = getattr(config, 'screenshot_method', 'mss')
    if screenshot_method == 'dxcam':
        dxcam_capture = _initialize_dxcam_capture(config)
        if dxcam_capture is not None:
            logger.info('[Capture] DXcam backend initialized successfully (BGRA output).')
            return dxcam_capture
        _warn_once('dxcam_fallback_mss', '[Capture] DXcam backend unavailable; automatic fallback to MSS is active.')
    elif screenshot_method == 'uvc':
        try:
            uvc_capture = UVCCapture(config)
            logger.info('[Capture] UVC backend initialized via OpenCV VideoCapture.')
            return uvc_capture
        except Exception as exc:
            _warn_once(
                'uvc_fallback_mss',
                f'[Capture] UVC initialization failed with "{exc}". Falling back to MSS backend.',
            )
    elif screenshot_method == 'ndi':
        try:
            ndi_capture = NDICapture(config)
            logger.info('[Capture] NDI backend initialized via cyndilib and is now active.')
            return ndi_capture
        except Exception as exc:
            # Always log (not _warn_once): NDI is (re)connected interactively and
            # the failure reason must stay visible across repeated attempts.
            logger.error('[Capture][NDI] Initialization failed with "%s". Falling back to MSS backend.', exc)
    elif screenshot_method == 'udp':
        try:
            udp_capture = UdpCapture(config)
            logger.info('[Capture] UDP backend initialized and listening on %s:%s.',
                        getattr(config, 'udp_bind_ip', '0.0.0.0'),
                        getattr(config, 'udp_bind_port', 5600))
            return udp_capture
        except Exception as exc:
            logger.error('[Capture][UDP] Initialization failed with "%s". Falling back to MSS backend.', exc)
    elif screenshot_method != 'mss':
        _warn_once(
            'invalid_screenshot_method',
            f"[Capture] Unknown screenshot method '{screenshot_method}'. Falling back to MSS backend.",
        )

    try:
        mss_capture = mss.mss()
    except Exception as exc:
        logger.error('[Capture] MSS initialization failed with "%s".', exc)
        raise

    # For screen capture backends, report the primary monitor refresh rate as
    # the nominal source FPS so the status panel has a useful reference value.
    _refresh = _get_monitor_refresh_rate()
    if _refresh > 0:
        config.source_nominal_fps = float(_refresh)

    logger.info('[Capture] MSS backend initialized successfully.')
    return mss_capture


def _capture_backend_is_stale(
    capture_backend: Any, timeout: float = _CAPTURE_STALE_TIMEOUT_SECONDS,
) -> bool:
    """True when *capture_backend* is a live object whose reader thread has
    delivered no frames for over *timeout* seconds.

    Only uvc/udp backends set ``_last_frame_perf_time``; anything else (mss,
    dxcam, an object missing the attribute) is reported as not stale — those
    backends either aren't threaded readers or have their own recovery path.
    """
    last_frame_time = getattr(capture_backend, '_last_frame_perf_time', None)
    if last_frame_time is None:
        return False
    return (time.perf_counter() - last_frame_time) > timeout


def reinitialize_if_method_changed(
    config: Config,
    current_capture: Any,
    active_method: str,
) -> tuple[Any, str]:
    """Check whether *config.screenshot_method* has changed, or whether the
    currently active backend has silently gone dead, and reinitialize if so.

    Returns ``(capture_backend, active_method_name)``.  When there is no
    change the original objects are returned untouched.
    """

    desired = getattr(config, 'screenshot_method', 'mss')
    current_active = _detect_active_capture_method(current_capture, active_method)

    reinit_log_fn = logger.info
    reinit_log_msg = None
    reinit_log_args: tuple = ()

    if desired != current_active:
        reinit_log_msg = '[Capture] Screenshot method transition: %s -> %s. Reinitializing backend...'
        reinit_log_args = (current_active, desired)
    elif desired == 'uvc' and hasattr(current_capture, 'config_signature'):
        if getattr(current_capture, 'config_signature', None) != _uvc_signature(config):
            reinit_log_msg = '[Capture] UVC configuration changed. Reinitializing UVC backend...'
        elif _capture_backend_is_stale(current_capture):
            reinit_log_fn = logger.warning
            reinit_log_msg = (
                '[Capture] UVC backend has produced no frames for over %.0fs '
                '— treating it as dead and reinitializing...'
            )
            reinit_log_args = (_CAPTURE_STALE_TIMEOUT_SECONDS,)
    elif desired == 'ndi' and hasattr(current_capture, 'config_signature'):
        if getattr(current_capture, 'config_signature', None) != _ndi_signature(config):
            reinit_log_msg = '[Capture][NDI] NDI configuration changed. Reinitializing NDI backend...'
    elif desired == 'udp' and hasattr(current_capture, 'config_signature'):
        if getattr(current_capture, 'config_signature', None) != _udp_signature(config):
            reinit_log_msg = '[Capture][UDP] UDP configuration changed. Reinitializing UDP backend...'
        elif _capture_backend_is_stale(current_capture):
            reinit_log_fn = logger.warning
            reinit_log_msg = (
                '[Capture][UDP] UDP backend has produced no frames for over %.0fs '
                '— treating it as dead and reinitializing...'
            )
            reinit_log_args = (_CAPTURE_STALE_TIMEOUT_SECONDS,)

    if reinit_log_msg is None:
        return current_capture, current_active

    # Throttle every reinit trigger (method change, config change, or a
    # stalled backend) the same way, so a persistently-dead device retries
    # periodically instead of hammering the driver every 0.5s.
    now = time.perf_counter()
    last_attempt = float(getattr(config, '_last_capture_reinit_attempt', 0.0) or 0.0)
    if now - last_attempt < _CAPTURE_RETRY_INTERVAL_SECONDS:
        return current_capture, current_active
    setattr(config, '_last_capture_reinit_attempt', now)

    reinit_log_fn(reinit_log_msg, *reinit_log_args)

    # Release the old backend first
    _cleanup_capture(current_capture)

    new_capture = initialize_screen_capture(config)
    # Keep user's configured method in config; active backend is tracked separately.
    new_method = _detect_active_capture_method(new_capture, desired)
    return new_capture, new_method


def _to_dxcam_region(region: dict[str, int]) -> tuple[int, int, int, int]:
    """Convert mss-style region dict to dxcam-style region tuple."""

    left = int(region['left'])
    top = int(region['top'])
    right = left + int(region['width'])
    bottom = top + int(region['height'])
    return left, top, right, bottom


def capture_frame(screen_capture: Any, region: dict[str, int]) -> np.ndarray | None:
    """Capture one frame and return BGRA ndarray, or None when capture fails."""

    try:
        try:
            screenshot = screen_capture.grab(region)
        except TypeError:
            screenshot = screen_capture.grab(region=_to_dxcam_region(region))
    except mss.exception.ScreenShotError as exc:
        _warn_once('capture_screenshot_error', f"[截圖] 抓圖失敗: {exc}")
        return None
    except Exception as exc:
        _warn_once('capture_unknown_error', f"[截圖] 抓圖發生例外: {exc}")
        return None

    if screenshot is None:
        # dxcam (Desktop Duplication API) normally returns None when
        # screen content hasn't changed — this is expected, not an error.
        return None

    if isinstance(screenshot, np.ndarray):
        frame = screenshot
    else:
        frame = np.frombuffer(screenshot.bgra, dtype=np.uint8).reshape((screenshot.height, screenshot.width, 4))

    if frame.ndim != 3 or frame.shape[2] < 3:
        _warn_once('capture_invalid_frame_shape', f"[截圖] 影像格式異常: shape={getattr(frame, 'shape', None)}")
        return None

    if frame.shape[2] == 3:
        alpha = np.full((frame.shape[0], frame.shape[1], 1), 255, dtype=frame.dtype)
        frame = np.concatenate((frame, alpha), axis=2)

    if frame.size == 0:
        _warn_once('capture_empty_frame', '[截圖] 抓到空影像，已略過該幀')
        return None

    return frame
