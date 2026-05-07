from __future__ import annotations

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


def _uvc_signature(config: Config) -> tuple[int, int, int, int, bool, str, str, str]:
    return (
        int(getattr(config, 'uvc_device_index', 0)),
        int(getattr(config, 'uvc_width', 0)),
        int(getattr(config, 'uvc_height', 0)),
        int(getattr(config, 'uvc_fps', 0)),
        bool(getattr(config, 'uvc_show_window', False)),
        str(getattr(config, 'uvc_window_name', 'Axiom UVC Preview')),
        str(getattr(config, 'uvc_capture_method', 'dshow')).lower(),
        str(getattr(config, 'uvc_preview_scale_mode', 'scale_to_fit')).lower(),
    )


def _ndi_signature(config: Config) -> tuple[str, bool, str, str]:
    return (
        str(getattr(config, 'ndi_source_name', '')).strip(),
        bool(getattr(config, 'uvc_show_window', False)),
        str(getattr(config, 'uvc_preview_scale_mode', 'scale_to_fit')).lower(),
        str(getattr(config, 'ndi_bandwidth', 'highest')).lower(),
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
        self.config_signature = _ndi_signature(config)
        self.source_name = str(getattr(config, 'ndi_source_name', '')).strip()
        self.show_window = bool(getattr(config, 'uvc_show_window', False))
        self.window_name = str(getattr(config, 'ndi_window_name', 'Axiom NDI Preview'))
        # NDI preview prioritizes minimal display latency by default.
        ndi_preview_scale_mode = str(getattr(config, 'ndi_preview_scale_mode', '')).lower().strip()
        self.preview_scale_mode = ndi_preview_scale_mode or 'low_latency'
        self._finder: Any | None = None
        self._source_assigned = False
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

            receiver_kwargs: dict[str, Any] = {'color_format': RecvColorFormat.RGBX_RGBA}
            if RecvBandwidth is not None:
                bw_pref = str(getattr(config, 'ndi_bandwidth', 'highest')).lower()
                bw_value = getattr(RecvBandwidth, bw_pref, None) or getattr(RecvBandwidth, 'highest', None)
                if bw_value is not None:
                    receiver_kwargs['bandwidth'] = bw_value
                    print(f'[Capture][NDI] Bandwidth set to: {bw_pref}')
            if source is not None:
                receiver_kwargs['source'] = source
                self._source_assigned = True
                print(f"[Capture][NDI] Receiver will start with source '{_extract_ndi_source_name(source)}'.")
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

        connected = _wait_for_receiver_connection(
            self._receiver,
            getattr(self._receiver, 'frame_sync', None),
            getattr(self, '_video_frame_sync', None),
            getattr(self._receiver, 'receive', None),
            getattr(self._ReceiveFrameType, 'recv_video', None),
            attempts=30,
            interval_seconds=0.1,
        )

        if not connected:
            raise RuntimeError('Failed to connect to NDI source via cyndilib')
        print('[Capture][NDI] Receiver connected and video stream is ready.')

        if self.show_window:
            try:
                cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
                print(f"[Capture][NDI] Preview window enabled: '{self.window_name}'.")
            except Exception:
                pass

    def _resolve_source(self) -> Any | None:
        if not self.source_name:
            return None
        try:
            if self._finder is None:
                self._finder = self._Finder()
                print('[Capture][NDI] Finder instance created.')
            finder = self._finder
            if not getattr(finder, "is_open", False):
                finder.open()
                print('[Capture][NDI] Finder opened for network source discovery.')
            source = _find_ndi_source_by_name(finder, self.source_name)
            if source is not None:
                print(f"[Capture][NDI] Matched configured source '{self.source_name}'.")
                return source
            for _ in range(6):
                try:
                    changed = finder.wait_for_sources(0.5)
                except TypeError:
                    changed = finder.wait_for_sources(timeout=0.5)
                if changed:
                    finder.update_sources()
                    print('[Capture][NDI] Source list changed while searching for configured source.')
                source = _find_ndi_source_by_name(finder, self.source_name)
                if source is not None:
                    print(f"[Capture][NDI] Found configured source after refresh: '{self.source_name}'.")
                    return source
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

    @staticmethod
    def _bgra_from_cyndilib_frame(frame: Any) -> np.ndarray | None:
        if hasattr(frame, 'get_array'):
            try:
                raw = frame.get_array()
                width = int(getattr(frame, 'xres', 0) or 0)
                height = int(getattr(frame, 'yres', 0) or 0)
                if width <= 0 or height <= 0:
                    return None
                arr = np.asarray(raw, dtype=np.uint8)
                expected = width * height * 4
                if arr.size < expected:
                    return None
                arr = arr[:expected].reshape(height, width, 4)
                return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA)
            except Exception:
                return None

        width, height = frame.get_resolution()
        if width <= 0 or height <= 0:
            return None

        raw = frame.get_array()
        if raw is None:
            return None
        raw = np.asarray(raw, dtype=np.uint8)
        if raw.size == 0:
            return None

        expected = width * height * 4
        if raw.size < expected:
            return None
        rgba = raw[:expected].reshape(height, width, 4)
        return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)

    def grab(self, region: dict[str, int] | None = None, **_: Any) -> np.ndarray | None:
        if not self._receiver.is_connected():
            now = time.perf_counter()
            if now - float(getattr(self, '_last_reconnect_attempt', 0.0) or 0.0) > 1.0:
                self._last_reconnect_attempt = now
                source = self._resolve_source()
                if source is not None:
                    try:
                        self._receiver.set_source(source)
                        self._source_assigned = True
                        print(f"[Capture][NDI] Reconnecting receiver using configured source '{self.source_name}'.")
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

        frame_obj: Any | None = None
        if getattr(self, '_video_frame_sync', None) is not None and getattr(self._receiver, 'frame_sync', None) is not None:
            try:
                self._receiver.frame_sync.capture_video()
            except Exception:
                return None
            frame_obj = self._video_frame_sync
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

        frame = self._bgra_from_cyndilib_frame(frame_obj)
        if frame is None:
            return None
        full_frame = frame
        frame_h, frame_w = full_frame.shape[:2]
        if frame_w > 0 and frame_h > 0:
            self.preview_width = frame_w
            self.preview_height = frame_h
            # Expose active NDI stream resolution so detection/FOV use the same coordinate space.
            self.config.ndi_width = frame_w
            self.config.ndi_height = frame_h

        # Update nominal FPS from frame metadata when available.
        _frame_fps: float = 0.0
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

        if region is not None:
            left = max(0, int(region.get('left', 0)))
            top = max(0, int(region.get('top', 0)))
            width = max(0, int(region.get('width', frame_w)))
            height = max(0, int(region.get('height', frame_h)))
            right = min(frame_w, left + width)
            bottom = min(frame_h, top + height)
            if right <= left or bottom <= top:
                return None
            frame = frame[top:bottom, left:right]

        if self.show_window:
            try:
                preview = self._draw_overlay(full_frame.copy(), region)
                render_frame = _render_preview_frame(self.window_name, self.preview_scale_mode, preview)
                cv2.imshow(self.window_name, render_frame)
                cv2.waitKey(1)
            except Exception:
                pass

        if frame.ndim == 3 and frame.shape[2] == 4:
            return frame
        if frame.ndim == 3 and frame.shape[2] == 3:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        return None

    def _draw_overlay(self, frame_bgra: np.ndarray, region: dict[str, int] | None) -> np.ndarray:
        """Draw overlay visuals in NDI preview window using active capture coordinates."""

        cfg = self.config
        if not bool(getattr(cfg, 'AimToggle', True)):
            return frame_bgra

        h, w = frame_bgra.shape[:2]
        region_left = int(region.get('left', 0)) if region else 0
        region_top = int(region.get('top', 0)) if region else 0
        region_width = int(region.get('width', w)) if region else w
        region_height = int(region.get('height', h)) if region else h

        cx = int(getattr(cfg, 'crosshairX', w // 2))
        cy = int(getattr(cfg, 'crosshairY', h // 2))

        if bool(getattr(cfg, 'show_detect_range', False)):
            x1 = max(0, region_left)
            y1 = max(0, region_top)
            x2 = min(w - 1, region_left + region_width)
            y2 = min(h - 1, region_top + region_height)
            cv2.rectangle(frame_bgra, (x1, y1), (x2, y2), (255, 140, 0, 255), 1, cv2.LINE_AA)

        if bool(getattr(cfg, 'show_fov', True)):
            fov = int(getattr(cfg, 'fov_size', 220))
            half = max(1, fov // 2)
            x1, y1 = cx - half, cy - half
            x2, y2 = cx + half, cy + half
            corner = max(8, min(20, fov // 6))
            color = (0, 0, 255, 255)
            cv2.line(frame_bgra, (x1, y1), (x1 + corner, y1), color, 2, cv2.LINE_AA)
            cv2.line(frame_bgra, (x1, y1), (x1, y1 + corner), color, 2, cv2.LINE_AA)
            cv2.line(frame_bgra, (x2, y1), (x2 - corner, y1), color, 2, cv2.LINE_AA)
            cv2.line(frame_bgra, (x2, y1), (x2, y1 + corner), color, 2, cv2.LINE_AA)
            cv2.line(frame_bgra, (x1, y2), (x1 + corner, y2), color, 2, cv2.LINE_AA)
            cv2.line(frame_bgra, (x1, y2), (x1, y2 - corner), color, 2, cv2.LINE_AA)
            cv2.line(frame_bgra, (x2, y2), (x2 - corner, y2), color, 2, cv2.LINE_AA)
            cv2.line(frame_bgra, (x2, y2), (x2, y2 - corner), color, 2, cv2.LINE_AA)

        if bool(getattr(cfg, 'show_boxes', True)):
            boxes = list(getattr(cfg, 'latest_boxes', []) or [])
            confidences = list(getattr(cfg, 'latest_confidences', []) or [])
            show_conf = bool(getattr(cfg, 'show_confidence', True))
            # BGRA colors per theme (B, G, R, A)
            _theme_bgra = {
                'cyan':   (255, 220, 0, 255),
                'red':    (60, 60, 255, 255),
                'yellow': (0, 210, 255, 255),
                'white':  (255, 255, 255, 255),
                'purple': (255, 60, 180, 255),
            }
            theme_key = str(getattr(cfg, 'box_color_theme', 'default')).lower()
            box_color_bgra = _theme_bgra.get(theme_key, (0, 255, 0, 255))
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
                conf = float(confidences[i]) if i < len(confidences) else 0.5
                thickness = max(1, min(3, 1 + round(conf * 2)))
                corner_len = max(6, min(24, int(min(x2 - x1, y2 - y1) * 0.15)))
                cv2.line(frame_bgra, (x1, y1), (x1 + corner_len, y1), box_color_bgra, thickness, cv2.LINE_AA)
                cv2.line(frame_bgra, (x1, y1), (x1, y1 + corner_len), box_color_bgra, thickness, cv2.LINE_AA)
                cv2.line(frame_bgra, (x2, y1), (x2 - corner_len, y1), box_color_bgra, thickness, cv2.LINE_AA)
                cv2.line(frame_bgra, (x2, y1), (x2, y1 + corner_len), box_color_bgra, thickness, cv2.LINE_AA)
                cv2.line(frame_bgra, (x1, y2), (x1 + corner_len, y2), box_color_bgra, thickness, cv2.LINE_AA)
                cv2.line(frame_bgra, (x1, y2), (x1, y2 - corner_len), box_color_bgra, thickness, cv2.LINE_AA)
                cv2.line(frame_bgra, (x2, y2), (x2 - corner_len, y2), box_color_bgra, thickness, cv2.LINE_AA)
                cv2.line(frame_bgra, (x2, y2), (x2, y2 - corner_len), box_color_bgra, thickness, cv2.LINE_AA)
                if show_conf and i < len(confidences):
                    cv2.putText(
                        frame_bgra,
                        f"{conf * 100:.0f}%",
                        (max(0, x1 - 5), max(15, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )

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
                    cv2.line(frame_bgra, (cx, cy), (bx, by), (255, 255, 255, 255), 2, cv2.LINE_AA)

        return frame_bgra

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
        if self.show_window:
            try:
                cv2.destroyWindow(self.window_name)
            except Exception:
                pass


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


class UVCCapture:
    """OpenCV VideoCapture backend for UVC capture cards/cameras."""

    def __init__(self, config: Config) -> None:
        self.config = config
        device_index = int(getattr(config, 'uvc_device_index', 0))
        width = int(getattr(config, 'uvc_width', 1920))
        height = int(getattr(config, 'uvc_height', 1080))
        fps = int(getattr(config, 'uvc_fps', 60))
        self.show_window = bool(getattr(config, 'uvc_show_window', False))
        self.window_name = str(getattr(config, 'uvc_window_name', 'Axiom UVC Preview'))
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

        if self.show_window:
            try:
                cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(self.window_name, self.preview_width, self.preview_height)
            except Exception:
                pass

        # --- Non-blocking reader thread ---
        # cap.read() blocks up to one frame period (e.g. 16 ms at 60 fps).
        # A background thread continuously reads into _latest_frame so that
        # grab() can return the newest frame without blocking the inference loop.
        self._latest_frame_lock = threading.Lock()
        self._latest_frame_bgr: np.ndarray | None = None
        self._reader_stop = threading.Event()
        self._reader_thread = threading.Thread(
            target=self._reader_worker, name='UVCReader', daemon=True
        )
        self._reader_thread.start()

    def _reader_worker(self) -> None:
        while not self._reader_stop.is_set():
            ok, frame = self.cap.read()
            if ok and frame is not None:
                with self._latest_frame_lock:
                    self._latest_frame_bgr = frame

    def grab(self, region: dict[str, int] | None = None, **_: Any) -> np.ndarray | None:
        """Return BGRA frame cropped by region when provided.

        Always returns the most recent frame captured by the reader thread
        without blocking the caller.  UVC preview renders on the full frame
        so the preview window is independent of the AI detection crop region.
        """

        with self._latest_frame_lock:
            frame_bgr = self._latest_frame_bgr
        if frame_bgr is None:
            return None

        full_frame_bgr = frame_bgr

        if self.show_window:
            try:
                preview_frame = self._draw_overlay(full_frame_bgr.copy(), region)
                render_frame = self._render_preview_frame(preview_frame)
                cv2.imshow(self.window_name, render_frame)
                cv2.waitKey(1)
            except Exception:
                pass

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
        """Draw overlay.py-equivalent visuals into UVC preview window."""

        cfg = self.config
        if not bool(getattr(cfg, 'AimToggle', True)):
            return frame_bgr

        h, w = frame_bgr.shape[:2]
        region_left = int(region.get('left', 0)) if region else 0
        region_top = int(region.get('top', 0)) if region else 0
        region_width = int(region.get('width', w)) if region else w
        region_height = int(region.get('height', h)) if region else h

        cx = int(getattr(cfg, 'crosshairX', w // 2))
        cy = int(getattr(cfg, 'crosshairY', h // 2))

        if bool(getattr(cfg, 'show_detect_range', False)):
            x1 = max(0, region_left)
            y1 = max(0, region_top)
            x2 = min(w - 1, region_left + region_width)
            y2 = min(h - 1, region_top + region_height)
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (255, 140, 0), 1, cv2.LINE_AA)

        if bool(getattr(cfg, 'show_fov', True)):
            fov = int(getattr(cfg, 'fov_size', 220))
            half = max(1, fov // 2)
            x1, y1 = cx - half, cy - half
            x2, y2 = cx + half, cy + half
            corner = max(8, min(20, fov // 6))
            color = (0, 0, 255)
            # top-left
            cv2.line(frame_bgr, (x1, y1), (x1 + corner, y1), color, 2, cv2.LINE_AA)
            cv2.line(frame_bgr, (x1, y1), (x1, y1 + corner), color, 2, cv2.LINE_AA)
            # top-right
            cv2.line(frame_bgr, (x2, y1), (x2 - corner, y1), color, 2, cv2.LINE_AA)
            cv2.line(frame_bgr, (x2, y1), (x2, y1 + corner), color, 2, cv2.LINE_AA)
            # bottom-left
            cv2.line(frame_bgr, (x1, y2), (x1 + corner, y2), color, 2, cv2.LINE_AA)
            cv2.line(frame_bgr, (x1, y2), (x1, y2 - corner), color, 2, cv2.LINE_AA)
            # bottom-right
            cv2.line(frame_bgr, (x2, y2), (x2 - corner, y2), color, 2, cv2.LINE_AA)
            cv2.line(frame_bgr, (x2, y2), (x2, y2 - corner), color, 2, cv2.LINE_AA)

        if bool(getattr(cfg, 'show_boxes', True)):
            boxes = list(getattr(cfg, 'latest_boxes', []) or [])
            confidences = list(getattr(cfg, 'latest_confidences', []) or [])
            show_conf = bool(getattr(cfg, 'show_confidence', True))
            # BGR colors per theme (B, G, R)
            _theme_bgr = {
                'cyan':   (255, 220, 0),
                'red':    (60, 60, 255),
                'yellow': (0, 210, 255),
                'white':  (255, 255, 255),
                'purple': (255, 60, 180),
            }
            theme_key = str(getattr(cfg, 'box_color_theme', 'default')).lower()
            box_color_bgr = _theme_bgr.get(theme_key, (0, 255, 0))
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
                conf = float(confidences[i]) if i < len(confidences) else 0.5
                thickness = max(1, min(3, 1 + round(conf * 2)))
                corner_len = max(6, min(24, int(min(x2 - x1, y2 - y1) * 0.15)))
                cv2.line(frame_bgr, (x1, y1), (x1 + corner_len, y1), box_color_bgr, thickness, cv2.LINE_AA)
                cv2.line(frame_bgr, (x1, y1), (x1, y1 + corner_len), box_color_bgr, thickness, cv2.LINE_AA)
                cv2.line(frame_bgr, (x2, y1), (x2 - corner_len, y1), box_color_bgr, thickness, cv2.LINE_AA)
                cv2.line(frame_bgr, (x2, y1), (x2, y1 + corner_len), box_color_bgr, thickness, cv2.LINE_AA)
                cv2.line(frame_bgr, (x1, y2), (x1 + corner_len, y2), box_color_bgr, thickness, cv2.LINE_AA)
                cv2.line(frame_bgr, (x1, y2), (x1, y2 - corner_len), box_color_bgr, thickness, cv2.LINE_AA)
                cv2.line(frame_bgr, (x2, y2), (x2 - corner_len, y2), box_color_bgr, thickness, cv2.LINE_AA)
                cv2.line(frame_bgr, (x2, y2), (x2, y2 - corner_len), box_color_bgr, thickness, cv2.LINE_AA)
                if show_conf and i < len(confidences):
                    cv2.putText(
                        frame_bgr,
                        f"{conf * 100:.0f}%",
                        (max(0, x1 - 5), max(15, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )

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
                    cv2.line(frame_bgr, (cx, cy), (bx, by), (255, 255, 255), 2, cv2.LINE_AA)

        return frame_bgr

    def _render_preview_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        return _render_preview_frame(self.window_name, self.preview_scale_mode, frame_bgr)

    def close(self) -> None:
        self._reader_stop.set()
        if self._reader_thread.is_alive():
            self._reader_thread.join(timeout=1.0)
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
        if self.show_window:
            try:
                cv2.destroyWindow(self.window_name)
            except Exception:
                pass


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
            _warn_once(
                'ndi_fallback_mss',
                f'[Capture][NDI] Initialization failed with "{exc}". Falling back to MSS backend.',
            )
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
