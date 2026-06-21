"""
Secondary OCR inference — PaddleOCR text extraction from the active capture frame.

Reads frames from screen_capture.get_preview_frame() so it shares the capture
pipeline with the main inference and adds zero extra screen-grab overhead.

ROI (absolute pixel coords within the captured frame):
  Regular OCR: crops to _OCR_ROI when the frame is large enough
  Full scan:   uses the entire preview frame

Thread lifecycle:
  start(config)        — no-op if worker is already alive
  stop()               — signals the worker and joins within 2 s
  trigger_full_scan()  — next iteration uses the full frame instead of the ROI
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

_stop_event: threading.Event | None = None
_worker_thread: threading.Thread | None = None
_full_scan_flag = threading.Event()


def get_ocr_results() -> list[str]:
    """Return current OCR results as a list of 'N , text' strings (thread-safe)."""
    with _results_lock:
        return list(_ocr_results)


def trigger_full_scan() -> None:
    """Request one full-frame OCR pass on the next worker iteration."""
    _full_scan_flag.set()


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


def _crop_roi(frame: np.ndarray) -> np.ndarray:
    """Crop to _OCR_ROI if the frame is large enough, otherwise return the full frame."""
    h, w = frame.shape[:2]
    l, t = _OCR_ROI["left"], _OCR_ROI["top"]
    rw, rh = _OCR_ROI["width"], _OCR_ROI["height"]
    if w >= l + rw and h >= t + rh:
        return frame[t:t + rh, l:l + rw]
    return frame   # frame too small for absolute ROI — use whole thing


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
    Odd leftover tokens are shown on their own line.
    """
    tokens = _extract_texts(result)
    lines: list[str] = []
    i = 0
    weapon = 1
    while i < len(tokens):
        if i + 1 < len(tokens):
            lines.append(f"Weapon {weapon}: {tokens[i]} , {tokens[i + 1]}")
            i += 2
        else:
            lines.append(f"Weapon {weapon}: {tokens[i]}")
            i += 1
        weapon += 1
    return lines


def _worker(config: Config, stop_event: threading.Event) -> None:
    import os
    # Disable OneDNN — unimplemented ops in Paddle 3.x on Windows cause frame errors
    os.environ.setdefault("FLAGS_use_mkldnn", "0")
    os.environ.setdefault("PADDLE_DISABLE_ONEDNN", "1")

    # Lower this thread's OS priority so it doesn't steal cycles from main inference
    try:
        import ctypes
        ctypes.windll.kernel32.SetThreadPriority(  # type: ignore[attr-defined]
            ctypes.windll.kernel32.GetCurrentThread(), -1  # THREAD_PRIORITY_BELOW_NORMAL
        )
    except Exception:
        pass

    from .screen_capture import get_preview_frame

    try:
        from paddleocr import PaddleOCR  # type: ignore[import]
        ocr = PaddleOCR(lang="en", device="cpu")
        logger.info("[OCR] PaddleOCR initialized (CPU, OneDNN off). ROI=%s", _OCR_ROI)
    except Exception as exc:
        logger.error("[OCR] PaddleOCR initialization failed: %s", exc)
        return

    _logged_raw = False

    while not stop_event.is_set():
        t0 = time.perf_counter()

        ocr_enabled = getattr(config, "ocr_enabled", False)
        full_scan = _full_scan_flag.is_set()

        if not ocr_enabled and not full_scan:
            stop_event.wait(0.1)
            continue

        if full_scan:
            _full_scan_flag.clear()

        frame = get_preview_frame()
        if frame is None:
            stop_event.wait(0.05)
            continue

        try:
            if full_scan:
                img_rgb = _to_rgb(frame)
                logger.info("[OCR] Full-frame scan triggered (%dx%d)...",
                            frame.shape[1], frame.shape[0])
            else:
                img_rgb = _to_rgb(_crop_roi(frame))

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

            if full_scan:
                logger.info("[OCR] Full-frame scan found %d item(s): %s", len(lines), lines)
            elif lines:
                logger.debug("[OCR] Detected %d line(s): %s", len(lines), lines)

        except Exception as exc:
            logger.warning("[OCR] Frame error: %s", exc)

        # Sleep for the remainder of the configured interval
        fps = max(1, min(10, int(getattr(config, "ocr_fps", 2))))
        elapsed = time.perf_counter() - t0
        sleep_time = (1.0 / fps) - elapsed
        if sleep_time > 0:
            stop_event.wait(sleep_time)
