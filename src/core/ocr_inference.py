"""
Secondary OCR inference — PaddleOCR text extraction from the active capture frame.

Reads frames from screen_capture.get_preview_frame() so it shares the capture
pipeline with the main inference and adds zero extra screen-grab overhead.

All scans (continuous and manual) always use _OCR_ROI.

Thread lifecycle:
  start(config)     — no-op if worker is already alive
  stop()            — signals the worker and joins within 2 s
  trigger_scan()    — force an immediate ROI scan on the next iteration
"""

from __future__ import annotations

import logging
import re
import threading
import time
from typing import TYPE_CHECKING

import numpy as np

_EN_FILTER = re.compile(r"[^A-Za-z0-9 .,:;/\-_%()\[\]']")

if TYPE_CHECKING:
    from .config import Config

logger = logging.getLogger(__name__)

# Absolute pixel coords within the game/capture frame (1080p reference)
_OCR_ROI: dict[str, int] = {"left": 1515, "top": 1031, "width": 314, "height": 29}

_results_lock = threading.Lock()
_ocr_results: list[str] = []

_roi_image_lock = threading.Lock()
_roi_image: np.ndarray | None = None   # last captured ROI crop (BGR/BGRA)

_stop_event: threading.Event | None = None
_worker_thread: threading.Thread | None = None
_scan_flag = threading.Event()          # set to force an immediate scan


def get_ocr_results() -> list[str]:
    """Return current OCR results as a list of formatted strings (thread-safe)."""
    with _results_lock:
        return list(_ocr_results)


def get_roi_image() -> np.ndarray | None:
    """Return the last captured ROI crop as a numpy array (thread-safe)."""
    with _roi_image_lock:
        return _roi_image if _roi_image is None else _roi_image.copy()


def trigger_scan() -> None:
    """Force an immediate ROI scan on the next worker iteration."""
    _scan_flag.set()


def start(config: Config) -> None:
    """Start the OCR worker thread. No-op if it is already running."""
    global _stop_event, _worker_thread
    if _worker_thread is not None and _worker_thread.is_alive():
        return
    _stop_event = threading.Event()
    _worker_thread = threading.Thread(
        target=_worker, args=(config, _stop_event), name="OCRWorker", daemon=True
    )
    _worker_thread.start()


def stop() -> None:
    """Signal the OCR worker to stop and wait up to 2 s for it to exit."""
    global _stop_event, _worker_thread
    if _stop_event is not None:
        _stop_event.set()
    if _worker_thread is not None and _worker_thread.is_alive():
        _worker_thread.join(timeout=2.0)
    _stop_event = None
    _worker_thread = None


def _to_rgb(frame: np.ndarray) -> np.ndarray:
    """Convert BGRA or BGR frame to RGB for PaddleOCR."""
    if frame.ndim == 3 and frame.shape[2] == 4:
        return frame[:, :, :3][:, :, ::-1]   # BGRA → RGB
    return frame[:, :, ::-1]                  # BGR  → RGB


def _crop_roi(frame: np.ndarray, log_once: list) -> np.ndarray:
    """Always crop to _OCR_ROI, clamping to frame bounds."""
    h, w = frame.shape[:2]
    l, t = _OCR_ROI["left"], _OCR_ROI["top"]
    rw, rh = _OCR_ROI["width"], _OCR_ROI["height"]
    x1, y1 = min(l, w), min(t, h)
    x2, y2 = min(l + rw, w), min(t + rh, h)
    if not log_once:
        logger.info("[OCR] Frame %dx%d → ROI crop [x:%d-%d, y:%d-%d]", w, h, x1, x2, y1, y2)
        log_once.append(True)
    return frame[y1:y2, x1:x2]


def _extract_texts(result) -> list[str]:
    """Extract raw text tokens from a PaddleOCR result (new or old API)."""
    tokens: list[str] = []
    if not result:
        return tokens
    for page in result:
        if isinstance(page, dict):
            rec_texts = page.get("rec_texts") or page.get("rec_text") or []
            if isinstance(rec_texts, str):
                rec_texts = [rec_texts]
            for text in (rec_texts or []):
                clean = _EN_FILTER.sub("", str(text)).strip()
                if clean:
                    tokens.append(clean)
        elif isinstance(page, list):
            for item in page:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    rec = item[1]
                    text = rec[0] if isinstance(rec, (list, tuple)) else str(rec)
                    clean = _EN_FILTER.sub("", str(text)).strip()
                    if clean:
                        tokens.append(clean)
    return tokens


def _parse_ocr_result(result) -> list[str]:
    """Format OCR tokens as weapon pairs: 'Weapon N: slot , name'.

    Tokens arrive in pairs: [slot, name, slot, name, ...]
    e.g. ['2', 'ALTERNATOR', '3', 'R-301'] →
         ['Weapon 1: 2 , ALTERNATOR', 'Weapon 2: 3 , R-301']

    PaddleOCR may return the name before the slot number depending on scan
    direction, so we swap any pair where the second token is purely numeric.
    """
    tokens = _extract_texts(result)
    lines: list[str] = []
    i = 0
    weapon = 1
    while i < len(tokens):
        if i + 1 < len(tokens):
            a, b = tokens[i], tokens[i + 1]
            if not a.isdigit() and b.isdigit():
                a, b = b, a
            lines.append(f"Weapon {weapon}: {a} , {b}")
            i += 2
        else:
            lines.append(f"Weapon {weapon}: {tokens[i]}")
            i += 1
        weapon += 1
    return lines


def _worker(config: Config, stop_event: threading.Event) -> None:
    from .screen_capture import get_preview_frame

    try:
        import os as _os
        _os.environ.setdefault("FLAGS_use_mkldnn", "0")
        _os.environ.setdefault("PADDLE_DISABLE_ONEDNN", "1")
        import paddle  # type: ignore[import]
        paddle.set_device("cpu")
        from paddleocr import PaddleOCR  # type: ignore[import]
        ocr = PaddleOCR(lang="en", device="cpu")
        logger.info("[OCR] PaddleOCR initialized (CPU, OneDNN disabled). ROI=%s", _OCR_ROI)
    except Exception as exc:
        logger.error("[OCR] PaddleOCR initialization failed: %s", exc)
        return

    _logged_raw = False
    _logged_crop: list = []

    while not stop_event.is_set():
        t0 = time.perf_counter()

        ocr_enabled = getattr(config, "ocr_enabled", False)
        forced = _scan_flag.is_set()

        if not ocr_enabled and not forced:
            stop_event.wait(0.1)
            continue

        if forced:
            _scan_flag.clear()

        frame = get_preview_frame()
        if frame is None:
            stop_event.wait(0.05)
            continue

        try:
            roi_crop = _crop_roi(frame, _logged_crop)

            # Store the raw ROI crop for the GUI preview (before RGB conversion)
            with _roi_image_lock:
                global _roi_image
                _roi_image = roi_crop.copy()

            img_rgb = _to_rgb(roi_crop)
            result = ocr.ocr(img_rgb)

            if not _logged_raw:
                if result and isinstance(result[0], dict):
                    logger.info("[OCR] Page dict keys: %s", list(result[0].keys()))
                    logger.info("[OCR] rec_texts: %s", repr(result[0].get("rec_texts")))
                else:
                    logger.info("[OCR] First result structure: %s", repr(result)[:500])
                _logged_raw = True

            lines = _parse_ocr_result(result)

            with _results_lock:
                _ocr_results.clear()
                _ocr_results.extend(lines)

            if forced:
                logger.info("[OCR] ROI scan found %d item(s): %s", len(lines), lines)
            elif lines:
                logger.debug("[OCR] Detected %d line(s): %s", len(lines), lines)

        except Exception as exc:
            logger.warning("[OCR] Frame error: %s", exc)

        fps = max(1, min(10, int(getattr(config, "ocr_fps", 2))))
        elapsed = time.perf_counter() - t0
        sleep_time = (1.0 / fps) - elapsed
        if sleep_time > 0:
            stop_event.wait(sleep_time)
