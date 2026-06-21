"""
Secondary OCR inference — PaddleOCR ROI text extraction at ≤10 FPS.

Captures a fixed 1080p screen region (1500,1033 → 1820,1058), runs
PaddleOCR, and stores results as "N , text" formatted strings that the
GUI polls via get_ocr_results().

Thread lifecycle:
  start(config)  — no-op if worker is already alive
  stop()         — signals the worker and joins within 2 s

One-shot full-screen scan:
  trigger_full_scan()  — next worker iteration grabs full 1080p instead
                         of the ROI, useful for finding where text lives
"""

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING

import mss
import numpy as np

if TYPE_CHECKING:
    from .config import Config

logger = logging.getLogger(__name__)

_OCR_ROI: dict[str, int] = {"left": 1500, "top": 1033, "width": 320, "height": 25}
_FULL_SCREEN: dict[str, int] = {"left": 0, "top": 0, "width": 1920, "height": 1080}
_FPS_CAP: int = 10  # maximum frames per second

_results_lock = threading.Lock()
_ocr_results: list[str] = []

_stop_event: threading.Event | None = None
_worker_thread: threading.Thread | None = None
_full_scan_flag = threading.Event()  # set to trigger one full-screen grab


def get_ocr_results() -> list[str]:
    """Return current OCR results as a list of 'N , text' strings (thread-safe)."""
    with _results_lock:
        return list(_ocr_results)


def trigger_full_scan() -> None:
    """Request one full-screen OCR grab on the next worker iteration."""
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


def _parse_ocr_result(result) -> list[str]:
    """Parse PaddleOCR result into 'N , text' strings.

    Handles both API formats:
      Old (2.7.x): result[0] = list of [box, [text, conf]]
      New (2.9+):  result = list of page dicts; text lives in
                   page['rec_texts'] (list of strings, one per detected box)
    """
    lines: list[str] = []
    if not result:
        return lines

    idx = 1
    for page in result:
        if isinstance(page, dict):
            rec_texts = page.get("rec_texts") or page.get("rec_text") or []
            if isinstance(rec_texts, str):
                rec_texts = [rec_texts]
            for text in (rec_texts or []):
                if str(text).strip():
                    lines.append(f"{idx} , {str(text).strip()}")
                    idx += 1
        elif isinstance(page, list):
            for item in page:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    rec = item[1]
                    text = rec[0] if isinstance(rec, (list, tuple)) else str(rec)
                    if str(text).strip():
                        lines.append(f"{idx} , {str(text).strip()}")
                        idx += 1

    return lines


def _worker(config: Config, stop_event: threading.Event) -> None:
    try:
        from paddleocr import PaddleOCR  # type: ignore[import]
        ocr = PaddleOCR(lang="en")
        logger.info("[OCR] PaddleOCR initialized. ROI=%s", _OCR_ROI)
    except Exception as exc:
        logger.error("[OCR] PaddleOCR initialization failed: %s", exc)
        return

    frame_interval = 1.0 / _FPS_CAP
    _logged_raw = False

    with mss.mss() as sct:
        while not stop_event.is_set():
            t0 = time.perf_counter()

            if not getattr(config, "ocr_enabled", False) and not _full_scan_flag.is_set():
                stop_event.wait(0.1)
                continue

            try:
                full_scan = _full_scan_flag.is_set()
                if full_scan:
                    _full_scan_flag.clear()
                    region = _FULL_SCREEN
                    logger.info("[OCR] Full-screen scan triggered (1920×1080)...")
                else:
                    region = _OCR_ROI

                raw = np.array(sct.grab(region))        # BGRA uint8
                img_rgb = raw[:, :, :3][:, :, ::-1]    # BGRA → RGB

                result = ocr.ocr(img_rgb)

                if not _logged_raw:
                    if result and isinstance(result[0], dict):
                        logger.info("[OCR] Page dict keys: %s", list(result[0].keys()))
                        logger.info("[OCR] rec_texts value: %s", repr(result[0].get("rec_texts")))
                    else:
                        logger.info("[OCR] First result structure: %s", repr(result)[:500])
                    _logged_raw = True

                lines = _parse_ocr_result(result)

                with _results_lock:
                    _ocr_results.clear()
                    _ocr_results.extend(lines)

                if full_scan:
                    logger.info("[OCR] Full-screen scan found %d text item(s): %s", len(lines), lines)
                elif lines:
                    logger.debug("[OCR] Detected %d line(s): %s", len(lines), lines)

            except Exception as exc:
                logger.warning("[OCR] Frame error: %s", exc)

            elapsed = time.perf_counter() - t0
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                stop_event.wait(sleep_time)
