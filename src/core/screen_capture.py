from __future__ import annotations

from typing import TYPE_CHECKING, Any

import cv2
import mss
import numpy as np

if TYPE_CHECKING:
    from mss.base import MSSBase

    from .config import Config


_WARNED_MESSAGES: set[str] = set()


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


def _ndi_signature(config: Config) -> tuple[str, bool, str]:
    return (
        str(getattr(config, 'ndi_source_name', '')).strip(),
        bool(getattr(config, 'uvc_show_window', False)),
        str(getattr(config, 'uvc_preview_scale_mode', 'scale_to_fit')).lower(),
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
    value = getattr(source, 'stream_name', None)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return ''


def _find_ndi_source_by_name(finder: Any, target_name: str) -> Any | None:
    """Find an NDI source by full source name or stream name."""

    target = str(target_name or '').strip()
    if not target:
        return None

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
            if target in {full_name, stream_name}:
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
    resolution = f"{int(width)}x{int(height)}" if width and height else "Unknown"
    fps_text = f"{fps:.2f}" if fps and fps > 0 else "Unknown"
    return f"{name} ({resolution} @ {fps_text} fps)"


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
    """Return discovered NDI sources with best-effort video metadata."""

    try:
        symbols = _load_cyndilib_symbols()
    except RuntimeError:
        return []
    Finder = symbols['Finder']
    Receiver = symbols['Receiver']
    ReceiveFrameType = symbols['ReceiveFrameType']
    RecvColorFormat = symbols['RecvColorFormat']
    VideoFrameSync = symbols['VideoFrameSync']
    VideoRecvFrame = symbols['VideoRecvFrame']

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

                source = _find_ndi_source_by_name(finder, name)
                receiver: Any | None = None
                if source is not None:
                    try:
                        width, height, fps = _extract_ndi_source_video_meta(source)
                        receiver = Receiver(source=source, color_format=RecvColorFormat.BGRX_BGRA)
                        if VideoFrameSync is not None and getattr(receiver, 'frame_sync', None) is not None:
                            video_frame = VideoFrameSync()
                            receiver.frame_sync.set_video_frame(video_frame)
                            capture_video = True
                        elif VideoRecvFrame is not None:
                            video_frame = VideoRecvFrame()
                            receiver.set_video_frame(video_frame)
                            capture_video = False
                        else:
                            video_frame = None
                            capture_video = False

                        for _ in range(10):
                            if capture_video and video_frame is not None:
                                receiver.frame_sync.capture_video()
                                frame_w = int(getattr(video_frame, 'xres', 0) or 0)
                                frame_h = int(getattr(video_frame, 'yres', 0) or 0)
                                frame_rate = float(getattr(video_frame, 'frame_rate_N', 0) or 0)
                            else:
                                recv_result = receiver.receive(ReceiveFrameType.recv_video, 350)
                                if not (recv_result & ReceiveFrameType.recv_video) or video_frame is None:
                                    continue
                                frame_w, frame_h = video_frame.get_resolution()
                                frame_rate = video_frame.get_frame_rate()

                            if frame_w and frame_h:
                                width, height = int(frame_w), int(frame_h)
                            if frame_rate:
                                fps = float(frame_rate)
                            if width and height and fps:
                                break
                    except Exception:
                        pass
                    finally:
                        if receiver is not None:
                            try:
                                receiver.disconnect()
                            except Exception:
                                pass

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
            receiver_kwargs: dict[str, Any] = {'color_format': RecvColorFormat.RGBX_RGBA}
            if RecvBandwidth is not None and hasattr(RecvBandwidth, 'highest'):
                receiver_kwargs['bandwidth'] = RecvBandwidth.highest
            self._receiver = Receiver(**receiver_kwargs)

            self._video_frame_sync: Any | None = None
            self._video_frame: Any | None = None
            if VideoFrameSync is not None and getattr(self._receiver, 'frame_sync', None) is not None:
                self._video_frame_sync = VideoFrameSync()
                self._receiver.frame_sync.set_video_frame(self._video_frame_sync)
            elif VideoRecvFrame is not None:
                self._video_frame = VideoRecvFrame()
                self._receiver.set_video_frame(self._video_frame)
            else:
                raise RuntimeError('Unsupported cyndilib version: no usable video frame API found')

            source = self._resolve_source()
            if source is not None:
                self._receiver.set_source(source)
        except Exception as exc:
            raise RuntimeError(f'Failed to initialize cyndilib NDI receiver: {exc}') from exc

        if not self._receiver.is_connected():
            for _ in range(10):
                if self._receiver.is_connected():
                    break
                try:
                    self._receiver.frame_sync.capture_video()
                    if int(getattr(self._video_frame_sync, 'xres', 0) or 0) > 0:
                        break
                except Exception:
                    pass

        if not self._receiver.is_connected():
            raise RuntimeError('Failed to connect to NDI source via cyndilib')

        if self.show_window:
            try:
                cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            except Exception:
                pass

    def _resolve_source(self) -> Any | None:
        if not self.source_name:
            return None
        try:
            if self._finder is None:
                self._finder = self._Finder()
            finder = self._finder
            if not getattr(finder, "is_open", False):
                finder.open()
            source = _find_ndi_source_by_name(finder, self.source_name)
            if source is not None:
                return source
            for _ in range(6):
                try:
                    changed = finder.wait_for_sources(0.5)
                except TypeError:
                    changed = finder.wait_for_sources(timeout=0.5)
                if changed:
                    finder.update_sources()
                source = _find_ndi_source_by_name(finder, self.source_name)
                if source is not None:
                    return source
            return None
        except Exception:
            return None

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

        if region is not None:
            frame_h, frame_w = frame.shape[:2]
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
                # Keep preview path low-latency: avoid BGRA->BGR conversion unless needed.
                preview = full_frame
                if region is not None:
                    preview = full_frame.copy()
                    frame_h, frame_w = preview.shape[:2]
                    left = max(0, int(region.get('left', 0)))
                    top = max(0, int(region.get('top', 0)))
                    width = max(0, int(region.get('width', frame_w)))
                    height = max(0, int(region.get('height', frame_h)))
                    right = min(frame_w - 1, left + width)
                    bottom = min(frame_h - 1, top + height)
                    if right > left and bottom > top:
                        cv2.rectangle(preview, (left, top), (right, bottom), (255, 140, 0, 255), 1, cv2.LINE_8)
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

        if width > 0:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height > 0:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if fps > 0:
            self.cap.set(cv2.CAP_PROP_FPS, fps)
        try:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        except Exception:
            pass
        self.preview_width = max(1, int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or width or 1))
        self.preview_height = max(1, int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or height or 1))
        self.preview_fps = max(1, int(self.cap.get(cv2.CAP_PROP_FPS) or fps or 1))
        # Keep capture queue short to reduce latency.
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        if self.show_window:
            try:
                cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(self.window_name, self.preview_width, self.preview_height)
            except Exception:
                pass

    def grab(self, region: dict[str, int] | None = None, **_: Any) -> np.ndarray | None:
        """Return BGRA frame cropped by region when provided.

        UVC preview always renders on the full capture frame so the preview
        window remains independent from the AI detection crop region.
        """

        ok, frame_bgr = self.cap.read()
        if not ok or frame_bgr is None:
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
                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2, cv2.LINE_AA)
                if show_conf and i < len(confidences):
                    conf = float(confidences[i]) * 100.0
                    cv2.putText(
                        frame_bgr,
                        f"{conf:.0f}%",
                        (max(0, x1 - 5), max(15, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )

        return frame_bgr

    def _render_preview_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        return _render_preview_frame(self.window_name, self.preview_scale_mode, frame_bgr)

    def close(self) -> None:
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


def _warn_once(key: str, message: str) -> None:
    """Print warning once per process to avoid log flooding."""

    if key in _WARNED_MESSAGES:
        return
    _WARNED_MESSAGES.add(key)
    print(message)


def _initialize_dxcam_capture() -> Any | None:
    """Initialize dxcam backend, return None when unavailable."""

    try:
        import dxcam  # type: ignore[import-not-found]
    except ImportError:
        _warn_once('dxcam_import_error', '[截圖] dxcam 未安裝，無法使用 dxcam 後端')
        return None

    try:
        return dxcam.create(output_color='BGRA')
    except Exception as exc:
        _warn_once('dxcam_create_error', f"[截圖] dxcam 初始化失敗: {exc}，將回退至 mss")
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
        dxcam_capture = _initialize_dxcam_capture()
        if dxcam_capture is not None:
            print('[截圖] 已啟用 dxcam 截圖後端')
            return dxcam_capture
        _warn_once('dxcam_fallback_mss', '[截圖] dxcam 不可用，已自動切換為 mss')
    elif screenshot_method == 'uvc':
        try:
            uvc_capture = UVCCapture(config)
            print('[截圖] 已啟用 UVC (OpenCV VideoCapture) 截圖後端')
            return uvc_capture
        except Exception as exc:
            _warn_once('uvc_fallback_mss', f"[截圖] UVC 初始化失敗: {exc}，將回退至 mss")
    elif screenshot_method == 'ndi':
        try:
            ndi_capture = NDICapture(config)
            print('[截圖] 已啟用 NDI (cyndilib) 截圖後端')
            return ndi_capture
        except Exception as exc:
            _warn_once('ndi_fallback_mss', f"[截圖] NDI 初始化失敗: {exc}，將回退至 mss")
    elif screenshot_method != 'mss':
        _warn_once('invalid_screenshot_method', f"[截圖] 未知截圖方式 '{screenshot_method}'，已改為 mss")

    try:
        mss_capture = mss.mss()
    except Exception as exc:
        print(f"[截圖] mss 初始化失敗: {exc}")
        raise

    print('[截圖] 已啟用 mss 截圖後端')
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
    if desired == active_method:
        if desired == 'uvc' and hasattr(current_capture, 'config_signature'):
            if getattr(current_capture, 'config_signature', None) != _uvc_signature(config):
                print('[截圖] 偵測到 UVC 設定變更，正在重新初始化…')
            else:
                return current_capture, active_method
        elif desired == 'ndi' and hasattr(current_capture, 'config_signature'):
            if getattr(current_capture, 'config_signature', None) != _ndi_signature(config):
                print('[截圖] 偵測到 NDI 設定變更，正在重新初始化…')
            else:
                return current_capture, active_method
        else:
            return current_capture, active_method

    print(f'[截圖] 偵測到截圖方式變更: {active_method} → {desired}，正在重新初始化…')

    # Release the old backend first
    _cleanup_capture(current_capture)

    new_capture = initialize_screen_capture(config)
    # Keep user's configured method in config; active backend is tracked separately.
    new_method = getattr(config, 'screenshot_method', 'mss')
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
