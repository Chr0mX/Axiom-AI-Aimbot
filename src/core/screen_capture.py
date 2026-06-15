from __future__ import annotations

import colorsys
import logging
import threading
import time
from typing import TYPE_CHECKING, Any

import cv2
import mss
import numpy as np

if TYPE_CHECKING:
    from mss.base import MSSBase

    from .config import Config


_WARNED_MESSAGES: set[str] = set()
_CAPTURE_RETRY_INTERVAL_SECONDS = 5.0


def _detect_active_capture_method(screen_capture: Any, fallback_method: str = 'mss') -> str:
    """Best-effort detection of the currently active capture backend name."""

    if screen_capture is None:
        return str(fallback_method or 'mss')

    if isinstance(screen_capture, NDICapture):
        return 'ndi'
    if isinstance(screen_capture, UVCCapture):
        return 'uvc'

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


def _uvc_signature(config: Config) -> tuple[int, int, int, int, bool, str, str]:
    return (
        int(getattr(config, 'uvc_device_index', 0)),
        int(getattr(config, 'uvc_width', 0)),
        int(getattr(config, 'uvc_height', 0)),
        int(getattr(config, 'uvc_fps', 0)),
        bool(getattr(config, 'uvc_show_window', False)),
        str(getattr(config, 'uvc_capture_method', 'dshow')).lower(),
        str(getattr(config, 'uvc_preview_scale_mode', 'scale_to_fit')).lower(),
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

        print('[Capture][NDI] Initializing cyndilib NDI backend...')
        if self.source_name:
            print(f"[Capture][NDI] Requested source name from config: '{self.source_name}'.")
        else:
            print('[Capture][NDI] No source name configured. First discovered source will be auto-selected.')

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
                print('[Capture][NDI] Color format: UYVY_RGBA (half bandwidth, zero-copy reshape)')
            elif _bgra_fmt is not None:
                receiver_kwargs = {'color_format': _bgra_fmt}
                self._recv_fourcc = 'bgra'
                print('[Capture][NDI] Color format: BGRX_BGRA (no cvtColor)')
            else:
                receiver_kwargs = {'color_format': RecvColorFormat.RGBX_RGBA}
                self._recv_fourcc = 'rgba'
                print('[Capture][NDI] Color format: RGBX_RGBA (cvtColor fallback)')
            # Legacy flag kept so any external callers that check _recv_is_bgra still work
            self._recv_is_bgra: bool = self._recv_fourcc == 'bgra'
            if RecvBandwidth is not None:
                bw_pref = str(getattr(config, 'ndi_bandwidth', 'highest')).lower()
                bw_value = getattr(RecvBandwidth, bw_pref, None) or getattr(RecvBandwidth, 'highest', None)
                if bw_value is not None:
                    receiver_kwargs['bandwidth'] = bw_value
                    print(f'[Capture][NDI] Bandwidth set to: {bw_pref}')
            if source is not None:
                # NOTE: do NOT pass the source to the Receiver constructor.
                # Assigning it both in the ctor and again via set_source() below
                # double-assigns the source and disrupts the initial connection
                # (observed as a fall-back to MSS on the second launch, when a
                # source name is already saved in config). The auto-select path
                # — which connects reliably — only ever calls set_source() once,
                # so the configured-source path now mirrors it.
                print(f"[Capture][NDI] Resolved source '{_extract_ndi_source_name(source)}'; "
                      f"assigning after receiver creation.")
            self._receiver = Receiver(**receiver_kwargs)
            print('[Capture][NDI] Receiver object created successfully.')

            self._video_frame_sync: Any | None = None
            self._video_frame: Any | None = None
            if VideoFrameSync is not None and getattr(self._receiver, 'frame_sync', None) is not None:
                self._video_frame_sync = VideoFrameSync()
                self._receiver.frame_sync.set_video_frame(self._video_frame_sync)
                print('[Capture][NDI] Using VideoFrameSync capture path (matches gist flow).')
            elif VideoRecvFrame is not None:
                self._video_frame = VideoRecvFrame()
                self._receiver.set_video_frame(self._video_frame)
                print('[Capture][NDI] Using VideoRecvFrame fallback path (legacy cyndilib compatibility).')
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
            raise RuntimeError(f'Failed to initialize cyndilib NDI receiver: {exc}') from exc

        print('[Capture][NDI] Waiting for receiver to connect and deliver first video frame (up to 6s)...')
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
            raise RuntimeError('Failed to connect to NDI source via cyndilib')
        print('[Capture][NDI] Receiver connected and video stream is ready.')
        self._last_frame_time: float = time.perf_counter()

        # Shared refs for the preview thread — grab() writes, thread reads
        self._ndi_frame_lock: threading.Lock = threading.Lock()
        self._ndi_frame_ref: list = [None]    # list[np.ndarray | None]
        self._ndi_region_ref: list = [None]
        self._ndi_stop: threading.Event = threading.Event()
        # FPS caching — read once from frame metadata, then skip per-frame probe
        self._fps_cached: bool = False
        # Pre-allocated BGRA output buffer for the crop-path (avoids per-frame malloc)
        self._bgra_buf: np.ndarray | None = None
        self._bgra_shape: tuple = ()

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
            )
            self._ndi_preview_thread.start()
            print(f"[Capture][NDI] Preview window enabled: '{self.window_name}'.")

    def _resolve_source(self, log: bool = True) -> Any | None:
        if not self.source_name:
            return None
        try:
            if self._finder is None:
                self._finder = self._Finder()
                if log:
                    print('[Capture][NDI] Finder instance created.')
            finder = self._finder
            if not getattr(finder, "is_open", False):
                finder.open()
                if log:
                    print('[Capture][NDI] Finder opened for network source discovery.')
            source = _find_ndi_source_by_name(finder, self.source_name)
            if source is not None:
                if log:
                    print(f"[Capture][NDI] Matched configured source '{self.source_name}'.")
                return source
            for _ in range(6):
                try:
                    changed = finder.wait_for_sources(0.5)
                except TypeError:
                    changed = finder.wait_for_sources(timeout=0.5)
                if changed:
                    finder.update_sources()
                    if log:
                        print('[Capture][NDI] Source list changed while searching for configured source.')
                source = _find_ndi_source_by_name(finder, self.source_name)
                if source is not None:
                    if log:
                        print(f"[Capture][NDI] Found configured source after refresh: '{self.source_name}'.")
                    return source
            if log:
                print(f"[Capture][NDI] Could not find configured source '{self.source_name}' after retries.")
            return None
        except Exception:
            return None

    def _assign_first_available_source(self) -> None:
        """Follow gist behavior: when no source is set, auto-select first discovered stream."""

        try:
            if self._finder is None:
                self._finder = self._Finder()
                print('[Capture][NDI] Finder instance created for auto-select mode.')
            finder = self._finder
            if not getattr(finder, 'is_open', False):
                finder.open()
                print('[Capture][NDI] Finder opened for auto-select mode.')

            for attempt in range(8):
                names = [name for name in finder.get_source_names() if isinstance(name, str) and name.strip()]
                if names:
                    selected_name = names[0].strip()
                    with finder.notify:
                        selected_source = finder.get_source(selected_name)
                        self._receiver.set_source(selected_source)
                        self._source_assigned = True
                        print(f"[Capture][NDI] Auto-selected first available source: '{selected_name}'.")
                    return
                try:
                    changed = finder.wait_for_sources(0.5)
                except TypeError:
                    changed = finder.wait_for_sources(timeout=0.5)
                if changed:
                    finder.update_sources()
                    print(f'[Capture][NDI] Waiting for source discovery (attempt {attempt + 1}/8)...')

            print('[Capture][NDI] No NDI sources discovered for auto-select within timeout window.')
        except Exception as exc:
            print(f'[Capture][NDI] Auto-select source setup failed: {exc}')

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
                                print(f"[Capture][NDI] Reconnecting receiver using configured source '{self.source_name}'.")
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
            print(f"[Capture][NDI] Receiver connected to '{self.source_name}'.")
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

        if frame_w > 0 and frame_h > 0:
            self.preview_width = frame_w
            self.preview_height = frame_h
            self.config.ndi_width = frame_w
            self.config.ndi_height = frame_h

        if not self._fps_cached:
            _frame_fps: float = 0.0
            try:
                _frame_fps = float(frame_obj.get_frame_rate())
            except Exception:
                for _attr in ('frame_rate', 'framerate', 'fps', 'video_fps'):
                    _v = getattr(frame_obj, _attr, None)
                    if isinstance(_v, (int, float)) and float(_v) > 0:
                        _frame_fps = float(_v)
                        break
                if _frame_fps <= 0:
                    _num = getattr(frame_obj, 'frame_rate_N', None)
                    _den = getattr(frame_obj, 'frame_rate_D', None)
                    if isinstance(_num, (int, float)) and isinstance(_den, (int, float)) and float(_den) > 0:
                        _frame_fps = float(_num) / float(_den)
            if _frame_fps > 0:
                self.config.source_nominal_fps = _frame_fps
                self._fps_cached = True

        recv_fourcc: str = getattr(self, '_recv_fourcc', 'rgba')

        def _to_bgra(arr: np.ndarray) -> np.ndarray:
            if recv_fourcc == 'uyvy':
                return cv2.cvtColor(arr, cv2.COLOR_YUV2BGRA_UYVY)
            if recv_fourcc == 'bgra':
                return arr
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
                    if self._bgra_shape != expected_shape:
                        self._bgra_buf   = np.empty(expected_shape, dtype=np.uint8)
                        self._bgra_shape = expected_shape
                    cv2.cvtColor(crop_raw, cv2.COLOR_YUV2BGRA_UYVY, self._bgra_buf)
                    frame = self._bgra_buf.copy()
                elif recv_fourcc == 'bgra':
                    frame = raw[top:bottom, left:right]
                else:
                    crop_raw = raw[top:bottom, left:right]
                    expected_shape = (bottom - top, right - left, 4)
                    if self._bgra_shape != expected_shape:
                        self._bgra_buf   = np.empty(expected_shape, dtype=np.uint8)
                        self._bgra_shape = expected_shape
                    cv2.cvtColor(crop_raw, cv2.COLOR_RGBA2BGRA, self._bgra_buf)
                    frame = self._bgra_buf.copy()
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

    def close(self) -> None:
        for method_name in ('disconnect', 'close', 'release', 'stop', 'shutdown'):
            method = getattr(self._receiver, method_name, None)
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
        # Stop the preview thread; it destroys the window on exit
        if getattr(self, '_ndi_stop', None) is not None:
            self._ndi_stop.set()
        pt = getattr(self, '_ndi_preview_thread', None)
        if pt is not None and pt.is_alive():
            pt.join(timeout=1.0)


def list_supported_uvc_resolutions(
    device_index: int,
    capture_method: str = 'dshow',
) -> list[tuple[int, int]]:
    """Probe common UVC resolutions and return distinct supported entries."""

    backend_map = {
        'dshow': cv2.CAP_DSHOW,
        'msmf': cv2.CAP_MSMF,
        'any': cv2.CAP_ANY,
    }
    backend = backend_map.get(str(capture_method).lower(), cv2.CAP_DSHOW)
    cap = cv2.VideoCapture(int(device_index), backend)
    if not cap.isOpened():
        cap = cv2.VideoCapture(int(device_index))
    if not cap.isOpened():
        return []

    common_resolutions = [
        (320, 240), (640, 360), (640, 480), (800, 600), (960, 540),
        (1024, 576), (1024, 768), (1280, 720), (1280, 960), (1600, 900),
        (1920, 1080), (2560, 1440), (3840, 2160),
    ]
    supported: set[tuple[int, int]] = set()
    try:
        for width, height in common_resolutions:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            if actual_w > 0 and actual_h > 0 and abs(actual_w - width) <= 8 and abs(actual_h - height) <= 8:
                supported.add((actual_w, actual_h))
    finally:
        cap.release()
    return sorted(supported, key=lambda item: (item[0] * item[1], item[0]))


def list_supported_uvc_fps(
    device_index: int,
    width: int,
    height: int,
    capture_method: str = 'dshow',
) -> list[int]:
    """Probe common FPS values at the given resolution and return supported ones."""
    backend_map = {'dshow': cv2.CAP_DSHOW, 'msmf': cv2.CAP_MSMF, 'any': cv2.CAP_ANY}
    backend = backend_map.get(str(capture_method).lower(), cv2.CAP_DSHOW)
    cap = cv2.VideoCapture(int(device_index), backend)
    if not cap.isOpened():
        cap = cv2.VideoCapture(int(device_index))
    if not cap.isOpened():
        return [30, 60]
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    common = [24, 30, 60, 90, 120, 144, 240]
    supported: list[int] = []
    try:
        for fps in common:
            cap.set(cv2.CAP_PROP_FPS, fps)
            actual = cap.get(cv2.CAP_PROP_FPS)
            if actual > 0 and abs(actual - fps) <= 2:
                supported.append(fps)
    finally:
        cap.release()
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

    def run(self) -> None:
        # All OpenCV GUI operations for this window must happen on this thread.
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

        try:
            cv2.destroyWindow(self._window_name)
        except Exception:
            pass


class UVCCapture:
    """OpenCV VideoCapture backend for UVC capture cards/cameras."""

    def __init__(self, config: Config) -> None:
        self.config = config
        device_index = int(getattr(config, 'uvc_device_index', 0))
        width = int(getattr(config, 'uvc_width', 1920))
        height = int(getattr(config, 'uvc_height', 1080))
        fps = int(getattr(config, 'uvc_fps', 60))
        self.show_window = bool(getattr(config, 'uvc_show_window', False))
        self.window_name = _UVC_WINDOW_NAME
        self.config_signature = _uvc_signature(config)

        capture_method = str(getattr(config, 'uvc_capture_method', 'dshow')).lower()
        backend_map = {
            'dshow': cv2.CAP_DSHOW,
            'msmf': cv2.CAP_MSMF,
            'any': cv2.CAP_ANY,
            'auto': cv2.CAP_ANY,
        }
        backend = backend_map.get(capture_method, cv2.CAP_DSHOW)
        self.preview_scale_mode = str(getattr(config, 'uvc_preview_scale_mode', 'scale_to_fit')).lower()

        self.cap = cv2.VideoCapture(device_index, backend)
        if not self.cap.isOpened():
            # Fallback backend when CAP_DSHOW is unavailable
            self.cap = cv2.VideoCapture(device_index)

        if not self.cap.isOpened():
            raise RuntimeError(f'UVC device open failed: index={device_index}')

        # FOURCC must be set before resolution/FPS so the driver switches codec first.
        try:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        except Exception:
            pass
        if width > 0:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height > 0:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if fps > 0:
            self.cap.set(cv2.CAP_PROP_FPS, fps)
        # Keep the driver queue shallow so grab() always returns the newest frame.
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        self.preview_width = max(1, int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or width or 1))
        self.preview_height = max(1, int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or height or 1))
        self.preview_fps = max(1, int(self.cap.get(cv2.CAP_PROP_FPS) or fps or 1))
        # Publish nominal FPS so the status panel can display it.
        config.source_nominal_fps = float(self.preview_fps)

        # Verify FOURCC was actually accepted by the driver.  Without MJPEG,
        # 1080p raw (YUY2) requires ~237 MB/s — beyond USB 2.0 bandwidth — so
        # the driver silently throttles to 5–15 fps with no error raised.
        actual_fourcc_int = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        expected_fourcc   = cv2.VideoWriter_fourcc(*'MJPG')
        self.is_mjpeg = (actual_fourcc_int == expected_fourcc)
        if not self.is_mjpeg:
            actual_str = ''.join(
                chr((actual_fourcc_int >> i) & 0xFF) for i in [0, 8, 16, 24]
            ).strip('\x00') or 'unknown'
            logging.getLogger(__name__).warning(
                "[UVC] FOURCC MJPG not accepted by driver (got '%s'). "
                "At 1080p this limits FPS to <30. Switch backend to 'msmf'.",
                actual_str,
            )

        # --- Non-blocking reader thread ---
        # cap.read() blocks up to one frame period (e.g. 16 ms at 60 fps).
        # A background thread continuously reads into _latest_frame_ref so that
        # grab() can return the newest frame without blocking the inference loop.
        self._latest_frame_lock = threading.Lock()
        self._latest_frame_ref: list = [None]   # list[np.ndarray | None]
        self._region_ref: list = [None]         # list[dict | None]
        self._reader_stop = threading.Event()
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
            )
            self._preview_thread.start()

    def _reader_worker(self) -> None:
        while not self._reader_stop.is_set():
            ok, frame = self.cap.read()
            if ok and frame is not None:
                with self._latest_frame_lock:
                    self._latest_frame_ref[0] = frame
            else:
                time.sleep(0.005)

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

        # Let the preview thread know the current detection region so its
        # overlay stays in sync without requiring an extra lock or callback.
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

    def _render_preview_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        return _render_preview_frame(self.window_name, self.preview_scale_mode, frame_bgr)

    def close(self) -> None:
        self._reader_stop.set()
        if self._preview_thread is not None and self._preview_thread.is_alive():
            self._preview_thread.join(timeout=1.0)
        if self._reader_thread.is_alive():
            self._reader_thread.join(timeout=1.0)
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
        # destroyWindow is handled by the preview thread's run() exit path


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
    """Print warning once per process to avoid log flooding."""

    if key in _WARNED_MESSAGES:
        return
    _WARNED_MESSAGES.add(key)
    print(message)


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
            print('[Capture] DXcam backend initialized successfully (BGRA output).')
            return dxcam_capture
        _warn_once('dxcam_fallback_mss', '[Capture] DXcam backend unavailable; automatic fallback to MSS is active.')
    elif screenshot_method == 'uvc':
        try:
            uvc_capture = UVCCapture(config)
            print('[Capture] UVC backend initialized via OpenCV VideoCapture.')
            return uvc_capture
        except Exception as exc:
            _warn_once(
                'uvc_fallback_mss',
                f'[Capture] UVC initialization failed with "{exc}". Falling back to MSS backend.',
            )
    elif screenshot_method == 'ndi':
        try:
            ndi_capture = NDICapture(config)
            print('[Capture] NDI backend initialized via cyndilib and is now active.')
            return ndi_capture
        except Exception as exc:
            # Always print (not _warn_once): NDI is (re)connected interactively and
            # the failure reason must stay visible across repeated attempts.
            print(f'[Capture][NDI] Initialization failed with "{exc}". Falling back to MSS backend.')
    elif screenshot_method != 'mss':
        _warn_once(
            'invalid_screenshot_method',
            f"[Capture] Unknown screenshot method '{screenshot_method}'. Falling back to MSS backend.",
        )

    try:
        mss_capture = mss.mss()
    except Exception as exc:
        print(f'[Capture] MSS initialization failed with "{exc}".')
        raise

    # For screen capture backends, report the primary monitor refresh rate as
    # the nominal source FPS so the status panel has a useful reference value.
    _refresh = _get_monitor_refresh_rate()
    if _refresh > 0:
        config.source_nominal_fps = float(_refresh)

    print('[Capture] MSS backend initialized successfully.')
    return mss_capture


def reinitialize_if_method_changed(
    config: Config,
    current_capture: Any,
    active_method: str,
) -> tuple[Any, str]:
    """Check whether *config.screenshot_method* has changed and, if so,
    reinitialize the capture backend.

    Returns ``(capture_backend, active_method_name)``.  When there is no
    change the original objects are returned untouched.
    """

    desired = getattr(config, 'screenshot_method', 'mss')
    current_active = _detect_active_capture_method(current_capture, active_method)

    # If the user still wants a non-mss backend but we're currently running on
    # mss (due to fallback), periodically retry reinitialization.
    if desired != current_active:
        now = time.perf_counter()
        last_attempt = float(getattr(config, '_last_capture_reinit_attempt', 0.0) or 0.0)
        if now - last_attempt < _CAPTURE_RETRY_INTERVAL_SECONDS:
            return current_capture, current_active
        setattr(config, '_last_capture_reinit_attempt', now)

    if desired == current_active:
        if desired == 'uvc' and hasattr(current_capture, 'config_signature'):
            if getattr(current_capture, 'config_signature', None) != _uvc_signature(config):
                print('[Capture] UVC configuration changed. Reinitializing UVC backend...')
            else:
                return current_capture, current_active
        elif desired == 'ndi' and hasattr(current_capture, 'config_signature'):
            if getattr(current_capture, 'config_signature', None) != _ndi_signature(config):
                print('[Capture][NDI] NDI configuration changed. Reinitializing NDI backend...')
            else:
                return current_capture, current_active
        else:
            return current_capture, current_active

    print(f'[Capture] Screenshot method transition detected: {current_active} -> {desired}. Reinitializing backend...')

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
