from __future__ import annotations

import os
import time
import ctypes
from typing import TYPE_CHECKING, Any

import cv2
import mss
import numpy as np

if TYPE_CHECKING:
    from mss.base import MSSBase

    from .config import Config


_WARNED_MESSAGES: set[str] = set()


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _apply_single_video_filter(frame_bgr: np.ndarray, name: str, options: dict[str, Any]) -> np.ndarray:
    normalized_name = name.strip().lower()
    if normalized_name == 'sharpen':
        strength = min(3.0, max(0.0, _safe_float(options.get('strength', 1.0), 1.0)))
        blur = cv2.GaussianBlur(frame_bgr, (0, 0), 1.0)
        return cv2.addWeighted(frame_bgr, 1.0 + strength, blur, -strength, 0)

    if normalized_name == 'denoise':
        h = min(30, max(1, _safe_int(options.get('strength', 8), 8)))
        return cv2.fastNlMeansDenoisingColored(frame_bgr, None, h, h, 7, 21)

    if normalized_name == 'deblock':
        sigma = min(120, max(10, _safe_int(options.get('strength', 35), 35)))
        return cv2.bilateralFilter(frame_bgr, 7, sigma, sigma)

    if normalized_name == 'color correction':
        gamma = min(2.5, max(0.4, _safe_float(options.get('gamma', 1.0), 1.0)))
        sat = min(2.0, max(0.5, _safe_float(options.get('saturation', 1.0), 1.0)))
        gamma_frame = np.power(frame_bgr.astype(np.float32) / 255.0, 1.0 / gamma) * 255.0
        corrected = np.clip(gamma_frame, 0, 255).astype(np.uint8)
        hsv = cv2.cvtColor(corrected, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 1] = np.clip(hsv[..., 1] * sat, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    if normalized_name == 'brightness/contrast':
        brightness = min(100, max(-100, _safe_int(options.get('brightness', 0), 0)))
        contrast = min(3.0, max(0.3, _safe_float(options.get('contrast', 1.0), 1.0)))
        return cv2.convertScaleAbs(frame_bgr, alpha=contrast, beta=brightness)

    if normalized_name == 'vibrance':
        amount = min(2.0, max(0.0, _safe_float(options.get('amount', 0.35), 0.35)))
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        sat = hsv[..., 1]
        gain = 1.0 + amount * (1.0 - (sat / 255.0))
        hsv[..., 1] = np.clip(sat * gain, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    if normalized_name in {'lanczos', 'bicubic'}:
        factor = min(2.0, max(1.05, _safe_float(options.get('factor', 1.25), 1.25)))
        interpolation = cv2.INTER_LANCZOS4 if normalized_name == 'lanczos' else cv2.INTER_CUBIC
        h, w = frame_bgr.shape[:2]
        up = cv2.resize(frame_bgr, (max(1, int(w * factor)), max(1, int(h * factor))), interpolation=interpolation)
        return cv2.resize(up, (w, h), interpolation=interpolation)

    if normalized_name == 'super resolution':
        sigma_s = min(80, max(5, _safe_int(options.get('sigma_s', 20), 20)))
        sigma_r = min(0.8, max(0.02, _safe_float(options.get('sigma_r', 0.25), 0.25)))
        detail = cv2.detailEnhance(frame_bgr, sigma_s=sigma_s, sigma_r=sigma_r)
        return cv2.GaussianBlur(detail, (0, 0), 0.4)

    return frame_bgr


def apply_configured_video_filters(frame: np.ndarray, config: Config, method: str) -> np.ndarray:
    if method not in {'uvc', 'ndi'}:
        return frame

    filter_stack = getattr(config, 'video_filters', []) or []
    if not isinstance(filter_stack, list) or not filter_stack:
        return frame

    if frame.ndim != 3 or frame.shape[2] < 3:
        return frame

    has_alpha = frame.shape[2] == 4
    alpha = frame[:, :, 3:4].copy() if has_alpha else None
    working = frame[:, :, :3].copy()
    try:
        for item in filter_stack:
            if not isinstance(item, dict):
                continue
            if not bool(item.get('enabled', True)):
                continue
            name = str(item.get('name', '')).strip()
            if not name:
                continue
            options = item.get('options', {})
            if not isinstance(options, dict):
                options = {}
            working = _apply_single_video_filter(working, name, options)
    except Exception as exc:
        _warn_once('video_filter_error', f'[截圖] 影像濾鏡套用失敗: {exc}')
        return frame

    if has_alpha and alpha is not None:
        return np.concatenate((working, alpha), axis=2)
    return working


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


def list_available_ndi_sources(timeout_ms: int = 1200) -> list[str]:
    """Try to discover NDI source names.

    Returns an empty list when NDI SDK/Python bindings are unavailable.
    You can also prefill entries via `AXIOM_NDI_SOURCES` (comma-separated).
    """

    preset_sources = [s.strip() for s in os.getenv('AXIOM_NDI_SOURCES', '').split(',') if s.strip()]
    discovered: list[str] = list(dict.fromkeys(preset_sources))

    try:
        import ndilib as ndi  # type: ignore[import-not-found]
    except ImportError:
        return discovered

    try:
        initialize = getattr(ndi, 'initialize', None)
        if callable(initialize) and not bool(initialize()):
            return discovered

        create_find = getattr(ndi, 'find_create_v2', None) or getattr(ndi, 'find_create', None)
        get_sources = getattr(ndi, 'find_get_current_sources', None)
        destroy_find = getattr(ndi, 'find_destroy', None)
        wait_for_sources = getattr(ndi, 'find_wait_for_sources', None)
        if not callable(create_find) or not callable(get_sources):
            return discovered

        find_inst = create_find()
        if find_inst is None:
            return discovered
        try:
            if callable(wait_for_sources):
                wait_for_sources(find_inst, int(timeout_ms))
            else:
                time.sleep(max(0.0, timeout_ms / 1000.0))
            sources = get_sources(find_inst) or []
            for source in sources:
                name = str(getattr(source, 'ndi_name', '') or getattr(source, 'p_ndi_name', '')).strip()
                if name and name not in discovered:
                    discovered.append(name)
        finally:
            if callable(destroy_find):
                destroy_find(find_inst)
    except Exception:
        return discovered

    return discovered


def _ndi_source_name(source: Any) -> str:
    return str(getattr(source, 'ndi_name', '') or getattr(source, 'p_ndi_name', '')).strip()


def _crop_bgra_frame(frame_bgra: np.ndarray, region: dict[str, int] | None) -> np.ndarray | None:
    if region is None:
        return frame_bgra

    frame_h, frame_w = frame_bgra.shape[:2]
    left = max(0, int(region.get('left', 0)))
    top = max(0, int(region.get('top', 0)))
    width = max(0, int(region.get('width', frame_w)))
    height = max(0, int(region.get('height', frame_h)))
    right = min(frame_w, left + width)
    bottom = min(frame_h, top + height)
    if right <= left or bottom <= top:
        return None
    return frame_bgra[top:bottom, left:right]


class NDICapture:
    """NDI receive backend using ndilib Python bindings."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.backend_name = 'ndi'
        self.find_inst: Any | None = None
        self.recv_inst: Any | None = None
        self.selected_source_name = ''
        self._pending_video_frame: Any | None = None

        try:
            import ndilib as ndi  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError('NDI Python bindings unavailable (ndilib)') from exc

        self.ndi = ndi
        initialize = getattr(ndi, 'initialize', None)
        if callable(initialize) and not bool(initialize()):
            raise RuntimeError('NDI initialization failed')

        create_find = getattr(ndi, 'find_create_v2', None) or getattr(ndi, 'find_create', None)
        if not callable(create_find):
            raise RuntimeError('NDI source discovery API unavailable')
        self.find_inst = create_find()
        if self.find_inst is None:
            raise RuntimeError('NDI source finder creation failed')

        sources = self._discover_sources(timeout_ms=1500)
        if not sources:
            raise RuntimeError('No NDI sources discovered')

        preferred_name = str(getattr(config, 'ndi_source_name', '')).strip()
        selected_source = None
        if preferred_name:
            for source in sources:
                if _ndi_source_name(source) == preferred_name:
                    selected_source = source
                    break
            if selected_source is None:
                raise RuntimeError(f"NDI source not found: '{preferred_name}'")
        else:
            selected_source = sources[0]
            config.ndi_source_name = _ndi_source_name(selected_source)

        self.selected_source_name = _ndi_source_name(selected_source)
        self.recv_inst = self._create_receiver()
        connect_fn = getattr(ndi, 'recv_connect', None) or getattr(ndi, 'RecvConnect', None)
        if not callable(connect_fn):
            raise RuntimeError('NDI receiver connect API unavailable')
        connect_fn(self.recv_inst, selected_source)

    def _discover_sources(self, timeout_ms: int) -> list[Any]:
        wait_for_sources = getattr(self.ndi, 'find_wait_for_sources', None)
        get_sources = getattr(self.ndi, 'find_get_current_sources', None)
        if not callable(get_sources):
            return []

        if callable(wait_for_sources):
            wait_for_sources(self.find_inst, int(timeout_ms))
        else:
            time.sleep(max(0.0, timeout_ms / 1000.0))
        return list(get_sources(self.find_inst) or [])

    def _create_receiver(self) -> Any:
        ndi = self.ndi
        recv_create = getattr(ndi, 'recv_create_v3', None) or getattr(ndi, 'recv_create_v2', None) or getattr(ndi, 'recv_create', None)
        if not callable(recv_create):
            raise RuntimeError('NDI receiver creation API unavailable')

        settings_factory = getattr(ndi, 'RecvCreateV3', None) or getattr(ndi, 'RecvCreateV2', None) or getattr(ndi, 'RecvCreate', None)
        settings = settings_factory() if callable(settings_factory) else None
        if settings is not None and hasattr(settings, 'color_format'):
            color_bgra = getattr(ndi, 'RECV_COLOR_FORMAT_BGRX_BGRA', None)
            if color_bgra is not None:
                settings.color_format = color_bgra

        try:
            recv_inst = recv_create(settings) if settings is not None else recv_create()
        except TypeError:
            recv_inst = recv_create()
        if recv_inst is None:
            raise RuntimeError('NDI receiver creation failed')
        return recv_inst

    def _read_video_frame(self, timeout_ms: int = 100) -> np.ndarray | None:
        ndi = self.ndi
        capture_fn = getattr(ndi, 'recv_capture_v2', None) or getattr(ndi, 'recv_capture', None)
        free_video_fn = getattr(ndi, 'recv_free_video_v2', None) or getattr(ndi, 'recv_free_video', None)
        if not callable(capture_fn) or not callable(free_video_fn):
            return None

        capture_result = capture_fn(self.recv_inst, int(timeout_ms))
        frame_type = None
        video_frame = None
        if isinstance(capture_result, tuple):
            if len(capture_result) >= 2:
                frame_type = capture_result[0]
                video_frame = capture_result[1]
        else:
            frame_type = capture_result
            video_frame = None

        frame_type_video = getattr(ndi, 'FRAME_TYPE_VIDEO', None)
        if frame_type_video is not None and frame_type != frame_type_video:
            return None
        if video_frame is None:
            return None

        try:
            return self._video_frame_to_bgra(video_frame)
        finally:
            try:
                free_video_fn(self.recv_inst, video_frame)
            except Exception:
                pass

    def _video_frame_to_bgra(self, video_frame: Any) -> np.ndarray | None:
        width = int(getattr(video_frame, 'xres', 0) or getattr(video_frame, 'width', 0) or 0)
        height = int(getattr(video_frame, 'yres', 0) or getattr(video_frame, 'height', 0) or 0)
        if width <= 0 or height <= 0:
            return None

        stride = int(getattr(video_frame, 'line_stride_in_bytes', 0) or width * 4)
        if stride <= 0:
            return None
        required = stride * height
        if required <= 0:
            return None

        data = getattr(video_frame, 'data', None) or getattr(video_frame, 'p_data', None)
        if data is None:
            return None

        flat: np.ndarray | None = None
        if isinstance(data, np.ndarray):
            flat = data.reshape(-1)
        elif isinstance(data, (bytes, bytearray, memoryview)):
            flat = np.frombuffer(data, dtype=np.uint8)
        else:
            try:
                ptr = ctypes.cast(data, ctypes.POINTER(ctypes.c_uint8))
                flat = np.ctypeslib.as_array(ptr, shape=(required,))
            except Exception:
                flat = None

        if flat is None or flat.size < required:
            return None
        rows = flat[:required].reshape((height, stride))
        pixel_bytes = rows[:, : width * 4]
        return np.ascontiguousarray(pixel_bytes.reshape((height, width, 4)))

    def grab(self, region: dict[str, int] | None = None, **_: Any) -> np.ndarray | None:
        frame_bgra = self._read_video_frame()
        if frame_bgra is None:
            return None
        frame_bgra = apply_configured_video_filters(frame_bgra, self.config, 'ndi')
        cropped = _crop_bgra_frame(frame_bgra, region)
        if cropped is None:
            return None
        self.config.ndi_width = int(frame_bgra.shape[1])
        self.config.ndi_height = int(frame_bgra.shape[0])
        return cropped

    def close(self) -> None:
        destroy_recv = getattr(self.ndi, 'recv_destroy', None)
        if callable(destroy_recv) and self.recv_inst is not None:
            try:
                destroy_recv(self.recv_inst)
            except Exception:
                pass
            self.recv_inst = None

        destroy_find = getattr(self.ndi, 'find_destroy', None)
        if callable(destroy_find) and self.find_inst is not None:
            try:
                destroy_find(self.find_inst)
            except Exception:
                pass
            self.find_inst = None

        destroy = getattr(self.ndi, 'destroy', None)
        if callable(destroy):
            try:
                destroy()
            except Exception:
                pass


class UVCCapture:
    """OpenCV VideoCapture backend for UVC capture cards/cameras."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.backend_name = 'uvc'
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

        frame_bgra = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2BGRA)
        frame_bgra = apply_configured_video_filters(frame_bgra, self.config, 'uvc')
        full_frame_bgr = frame_bgra[:, :, :3]
        frame_bgr = full_frame_bgr

        if self.show_window:
            try:
                preview_frame = self._draw_overlay(full_frame_bgr.copy(), region)
                render_frame = self._render_preview_frame(preview_frame)
                cv2.imshow(self.window_name, render_frame)
                cv2.waitKey(1)
            except Exception:
                pass

        if region is not None:
            cropped = _crop_bgra_frame(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2BGRA), region)
            if cropped is None:
                return None
            frame_bgr = cropped[:, :, :3]

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
        mode = self.preview_scale_mode
        if mode == 'scale_to_canvas':
            try:
                _, _, width, height = cv2.getWindowImageRect(self.window_name)
                if width > 0 and height > 0:
                    return cv2.resize(frame_bgr, (width, height), interpolation=cv2.INTER_LINEAR)
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
                cv2.resizeWindow(self.window_name, target_w, target_h)
            except Exception:
                pass
            return frame_bgr

        # default: scale_to_fit
        try:
            _, _, width, height = cv2.getWindowImageRect(self.window_name)
            if width <= 0 or height <= 0:
                return frame_bgr
            h, w = frame_bgr.shape[:2]
            ratio = min(width / max(1, w), height / max(1, h))
            draw_w = max(1, int(w * ratio))
            draw_h = max(1, int(h * ratio))
            resized = cv2.resize(frame_bgr, (draw_w, draw_h), interpolation=cv2.INTER_LINEAR)
            canvas = np.zeros((height, width, 3), dtype=np.uint8)
            x = (width - draw_w) // 2
            y = (height - draw_h) // 2
            canvas[y:y + draw_h, x:x + draw_w] = resized
            return canvas
        except Exception:
            return frame_bgr

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
            source = str(getattr(ndi_capture, 'selected_source_name', '')).strip()
            if source:
                print(f"[截圖] 已啟用 NDI 擷取後端: {source}")
            else:
                print('[截圖] 已啟用 NDI 擷取後端')
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

    if getattr(screen_capture, 'backend_name', '') not in {'uvc', 'ndi'}:
        config_obj = getattr(screen_capture, 'config', None)
        method = str(getattr(config_obj, 'screenshot_method', 'mss')).lower() if config_obj else 'mss'
        if method in {'uvc', 'ndi'} and config_obj is not None:
            frame = apply_configured_video_filters(frame, config_obj, method)
    return frame
