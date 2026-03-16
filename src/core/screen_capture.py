from __future__ import annotations

from typing import TYPE_CHECKING, Any

import mss
import numpy as np

if TYPE_CHECKING:
    from mss.base import MSSBase

    from .config import Config


_WARNED_MESSAGES: set[str] = set()


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
    elif screenshot_method != 'mss':
        _warn_once('invalid_screenshot_method', f"[截圖] 未知截圖方式 '{screenshot_method}'，已改為 mss")

    config.screenshot_method = 'mss'
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
        return current_capture, active_method

    print(f'[截圖] 偵測到截圖方式變更: {active_method} → {desired}，正在重新初始化…')

    # Release the old backend first
    _cleanup_capture(current_capture)

    new_capture = initialize_screen_capture(config)
    # initialize_screen_capture may normalise the method (e.g. fallback to mss)
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