"""
Secondary OCR inference — PaddleOCR ROI text extraction at ≤10 FPS.

Captures a fixed 1080p screen region (1500,1033 → 1820,1058), runs
PaddleOCR, and stores results as "N , text" formatted strings that the
GUI polls via get_ocr_results().

Thread lifecycle:
  start(config)  — no-op if worker is already alive
  stop()         — signals the worker and joins within 2 s
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
_FPS_CAP: int = 10  # maximum frames per second

_results_lock = threading.Lock()
_ocr_results: list[str] = []

_stop_event: threading.Event | None = None
_worker_thread: threading.Thread | None = None


def get_ocr_results() -> list[str]:
    """Return current OCR results as a list of 'N , text' strings (thread-safe)."""
    with _results_lock:
        return list(_ocr_results)


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


def _worker(config: Config, stop_event: threading.Event) -> None:
    try:
        from paddleocr import PaddleOCR  # type: ignore[import]
        ocr = PaddleOCR(use_angle_cls=False, lang="en", show_log=False)
    except Exception as exc:
        logger.error("[OCR] PaddleOCR initialization failed: %s", exc)
        return

    frame_interval = 1.0 / _FPS_CAP

    with mss.mss() as sct:
        while not stop_event.is_set():
            t0 = time.perf_counter()

            if not getattr(config, "ocr_enabled", False):
                stop_event.wait(0.1)
                continue

            try:
                raw = np.array(sct.grab(_OCR_ROI))  # BGRA uint8
                img_rgb = raw[:, :, :3][:, :, ::-1]  # BGRA → RGB (PaddleOCR expects RGB)
                result = ocr.ocr(img_rgb, cls=False)

                lines: list[str] = []
                if result and result[0]:
                    for idx, line in enumerate(result[0], start=1):
                        text: str = line[1][0] if line[1] else ""
                        if text.strip():
                            lines.append(f"{idx} , {text.strip()}")

                with _results_lock:
                    _ocr_results.clear()
                    _ocr_results.extend(lines)

            except Exception as exc:
                logger.debug("[OCR] Frame error: %s", exc)

            elapsed = time.perf_counter() - t0
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                stop_event.wait(sleep_time)
