from __future__ import annotations

from typing import TYPE_CHECKING, Any

import cv2
import mss
import numpy as np

if TYPE_CHECKING:
    from mss.base import MSSBase

    from .config import Config


_WARNED_MESSAGES: set[str] = set()


def _uvc_signature(config: Config) -> tuple[int, int, int, int, bool, str]:
    return (
        int(getattr(config, 'uvc_device_index', 0)),
        int(getattr(config, 'uvc_width', 0)),
        int(getattr(config, 'uvc_height', 0)),
        int(getattr(config, 'uvc_fps', 0)),
        bool(getattr(config, 'uvc_show_window', False)),
        str(getattr(config, 'uvc_window_name', 'Axiom UVC Preview')),
    )


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

        self.cap = cv2.VideoCapture(device_index, cv2.CAP_DSHOW)
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
        """Return BGRA frame cropped by region when provided."""

        ok, frame_bgr = self.cap.read()
        if not ok or frame_bgr is None:
            return None

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

        if self.show_window:
            try:
                preview_frame = self._draw_overlay(frame_bgr.copy(), region)
                cv2.imshow(self.window_name, preview_frame)
                cv2.waitKey(1)
            except Exception:
                pass

        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2BGRA)

    def _draw_overlay(self, frame_bgr: np.ndarray, region: dict[str, int] | None) -> np.ndarray:
        """Draw overlay.py-equivalent visuals into UVC preview window."""

        cfg = self.config
        if not bool(getattr(cfg, 'AimToggle', True)):
            return frame_bgr

        h, w = frame_bgr.shape[:2]
        region_left = int(region.get('left', 0)) if region else 0
        region_top = int(region.get('top', 0)) if region else 0

        cx = int(getattr(cfg, 'crosshairX', w // 2)) - region_left
        cy = int(getattr(cfg, 'crosshairY', h // 2)) - region_top

        if bool(getattr(cfg, 'show_detect_range', False)):
            range_size = int(getattr(cfg, 'detect_range_size', min(w, h)))
            half = max(1, range_size // 2)
            x1 = max(0, cx - half)
            y1 = max(0, cy - half)
            x2 = min(w - 1, cx + half)
            y2 = min(h - 1, cy + half)
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
                x1 -= region_left
                x2 -= region_left
                y1 -= region_top
                y2 -= region_top
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

    return frame
