"""
Secondary OCR inference — PaddleOCR text extraction from the active capture frame.

PROCESS-ISOLATED DESIGN
-----------------------
PaddleOCR is CPU-heavy and does a lot of work in *Python* (pre/post-processing),
so running it in a thread inside the main process stalls the main inference loop
and the Qt UI every time it fires — the Python GIL lets only one thread run
bytecode at a time, and PaddleOCR holds it for the whole pass.

To guarantee OCR never hurts main inference or capture, the heavy work runs in a
separate child *process* (its own GIL, its own core, below-normal OS priority).
The main process only runs a tiny "feeder" thread that:
  * reads screen_capture.get_preview_frame() (shared capture, zero extra grab),
  * crops the fixed _OCR_ROI (~36 KB),
  * stores the crop for the GUI preview,
  * ships the crop to the child over a queue (keeps only the newest frame),
  * drains finished result lines back from the child.
That feeder work is microseconds per cycle, so the main loop is never blocked.

All scans (continuous and manual) always use _OCR_ROI.

Public API (unchanged):
  start(config)     — spawn the OCR process + feeder thread (no-op if running)
  stop()            — stop both and join within a few seconds
  trigger_scan()    — force an immediate ROI scan
  get_ocr_results() — current formatted result lines
  get_roi_image()   — last ROI crop (for the GUI preview)
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import queue
import re
import sys
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
_IDLE_TEARDOWN_S = 5.0  # seconds before releasing an idle child process

# ── Parent-side shared state ──────────────────────────────────────────────────
_results_lock = threading.Lock()
_ocr_results: list[str] = []

_roi_image_lock = threading.Lock()
_roi_image: np.ndarray | None = None        # last captured ROI crop (BGR/BGRA)

_scan_flag = threading.Event()              # set to force an immediate scan

_stop_event: threading.Event | None = None  # stops the feeder thread
_feeder_thread: threading.Thread | None = None

# ── Child-process plumbing ────────────────────────────────────────────────────
_proc: mp.Process | None = None
_proc_stop = None                           # mp.Event
_frame_q = None                             # mp.Queue(maxsize=1)  parent → child
_result_q = None                            # mp.Queue(maxsize=4)  child → parent


# ── Public API ────────────────────────────────────────────────────────────────

def get_ocr_results() -> list[str]:
    """Return current OCR results as a list of formatted strings (thread-safe)."""
    with _results_lock:
        return list(_ocr_results)


def get_roi_image() -> np.ndarray | None:
    """Return the last captured ROI crop as a numpy array (thread-safe)."""
    with _roi_image_lock:
        return _roi_image if _roi_image is None else _roi_image.copy()


def trigger_scan() -> None:
    """Force an immediate ROI scan on the next feeder iteration."""
    _scan_flag.set()


def _ensure_proc() -> None:
    """Spawn the OCR child process if not already alive. Called from the feeder."""
    global _proc, _proc_stop, _frame_q, _result_q
    if _proc is not None and _proc.is_alive():
        return
    try:
        ctx = mp.get_context("spawn")
        _frame_q = ctx.Queue(maxsize=1)
        _result_q = ctx.Queue(maxsize=4)
        _proc_stop = ctx.Event()
        _proc = ctx.Process(
            target=_child_main, args=(_frame_q, _result_q, _proc_stop),
            name="OCRProcess", daemon=True,
        )
        _proc.start()
        logger.info("[OCR] child process started (pid=%s)", _proc.pid)
    except Exception as exc:
        logger.error("[OCR] Failed to start OCR process: %s", exc)
        _proc = None


def _kill_proc() -> None:
    """Tear down the OCR child process, leaving the feeder thread running."""
    global _proc, _proc_stop, _frame_q, _result_q
    if _proc_stop is not None:
        try:
            _proc_stop.set()
        except Exception:
            pass
    if _frame_q is not None:
        try:
            _frame_q.put_nowait(None)  # sentinel
        except Exception:
            pass
    if _proc is not None:
        try:
            _proc.join(timeout=1.0)
            if _proc.is_alive():
                _proc.terminate()
        except Exception:
            pass
        logger.info("[OCR] child process released")
    _proc = None
    _proc_stop = None
    _frame_q = None
    _result_q = None


def start(config: Config) -> None:
    """Start the OCR feeder thread. Child process is spawned lazily on first use. No-op if running."""
    global _stop_event, _feeder_thread
    if _feeder_thread is not None and _feeder_thread.is_alive():
        return

    _stop_event = threading.Event()
    _feeder_thread = threading.Thread(
        target=_feeder, args=(config, _stop_event), name="OCRFeeder", daemon=True,
    )
    _feeder_thread.start()


def stop() -> None:
    """Stop the feeder thread and OCR child process."""
    global _stop_event, _feeder_thread
    if _stop_event is not None:
        _stop_event.set()
    _kill_proc()
    if _feeder_thread is not None and _feeder_thread.is_alive():
        _feeder_thread.join(timeout=2.0)
    _stop_event = None
    _feeder_thread = None


# ── Shared helpers (used by both parent feeder and child process) ─────────────

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


def _build_ocr():
    """Construct a PaddleOCR instance with OneDNN disabled (child process only).

    PaddleOCR 3.x (PaddleX) runs inference through the Paddle Inference
    predictor, which enables OneDNN in its own config — the global
    paddle.set_flags / FLAGS_use_mkldnn are ignored by that path, so
    enable_mkldnn=False must be passed to the constructor. The OneDNN PIR
    instruction handler crashes on this model
    (ConvertPirAttribute2RuntimeAttribute / ArrayAttribute<DoubleAttribute>),
    so disabling it is required for CPU inference to work at all.

    The doc-orientation / unwarping / textline-orientation models are also
    disabled: they are useless for a single-line weapon-slot ROI and only add
    startup and per-frame cost.

    Constructor kwargs vary across PaddleOCR versions, so we try the full set
    first and drop unknown args progressively.
    """
    try:
        from paddleocr import PaddleOCR  # type: ignore[import]
    except Exception as exc:
        logger.error("[OCR] PaddleOCR import failed: %s", exc)
        return None

    kwarg_sets = [
        # PaddleOCR 3.x (PaddleX) — OneDNN off, extra models off, single thread.
        dict(lang="en", device="cpu", enable_mkldnn=False, cpu_threads=1,
             use_doc_orientation_classify=False, use_doc_unwarping=False,
             use_textline_orientation=False),
        # 3.x minimal — OneDNN off + single thread
        dict(lang="en", device="cpu", enable_mkldnn=False, cpu_threads=1),
        # 2.7.x legacy API
        dict(lang="en", use_gpu=False, use_angle_cls=False, show_log=False,
             cpu_threads=1),
        # last-resort minimal
        dict(lang="en"),
    ]
    last_exc = None
    for kwargs in kwarg_sets:
        try:
            ocr = PaddleOCR(**kwargs)
            logger.info("[OCR] PaddleOCR initialized (CPU, mkldnn=%s). ROI=%s",
                        kwargs.get("enable_mkldnn", "n/a"), _OCR_ROI)
            return ocr
        except Exception as exc:
            last_exc = exc
            logger.debug("[OCR] init kwargs %s rejected: %s", list(kwargs), exc)
    logger.error("[OCR] PaddleOCR initialization failed: %s", last_exc)
    return None


# ── Child process entry point ─────────────────────────────────────────────────

def _child_main(frame_q, result_q, proc_stop) -> None:
    """Run in a separate process: own GIL, own core, below-normal priority.

    Receives ROI crops on frame_q, runs PaddleOCR, returns formatted lines on
    result_q. Never touches the GPU (CPU PaddlePaddle build).
    """
    # Cap math-library thread pools BEFORE paddle's DLLs load. The bundled
    # libopenblas / libomp otherwise spawn one thread per core.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

    # Defensive: ensure the shared AppData site-packages dir is importable even
    # if spawn did not carry it over in sys.path.
    la = os.environ.get("LOCALAPPDATA", "")
    if la:
        pkg = os.path.join(la, "AxiomAI", "site-packages")
        if os.path.isdir(pkg) and pkg not in sys.path:
            sys.path.insert(0, pkg)

    # Below-normal priority for the whole OCR process so the OS always schedules
    # the main app first when cores are contended.
    if sys.platform == "win32":
        try:
            import ctypes
            BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
            k32 = ctypes.windll.kernel32
            k32.SetPriorityClass(k32.GetCurrentProcess(), BELOW_NORMAL_PRIORITY_CLASS)
        except Exception:
            pass
    else:
        try:
            os.nice(10)
        except Exception:
            pass

    ocr = None
    logged_raw = False
    while not proc_stop.is_set():
        try:
            roi = frame_q.get(timeout=0.3)
        except queue.Empty:
            continue
        if roi is None:                      # sentinel → shutdown
            break
        if ocr is None:
            ocr = _build_ocr()
            if ocr is None:
                continue
        try:
            img_rgb = _to_rgb(roi)
            result = ocr.ocr(img_rgb)

            if not logged_raw:
                if result and isinstance(result[0], dict):
                    print(f"[OCR child] page keys: {list(result[0].keys())}")
                    print(f"[OCR child] rec_texts: {result[0].get('rec_texts')!r}")
                else:
                    print(f"[OCR child] first result: {repr(result)[:300]}")
                logged_raw = True

            lines = _parse_ocr_result(result)
            try:
                result_q.put_nowait(lines)
            except queue.Full:
                # drop the oldest, keep the newest
                try:
                    result_q.get_nowait()
                    result_q.put_nowait(lines)
                except Exception:
                    pass
        except Exception as exc:
            print(f"[OCR child] frame error: {exc}")


# ── Parent feeder thread ──────────────────────────────────────────────────────

def _drain(q) -> None:
    """Empty a queue without blocking."""
    try:
        while True:
            q.get_nowait()
    except queue.Empty:
        pass


def _collect_results() -> None:
    """Pull the newest result list from the child and publish it."""
    latest = None
    try:
        while True:
            latest = _result_q.get_nowait()
    except queue.Empty:
        pass
    except Exception:
        return
    if latest is not None:
        with _results_lock:
            _ocr_results[:] = latest


def _feeder(config: Config, stop_event: threading.Event) -> None:
    """Lightweight parent thread: crop the ROI and shuttle it to the child.

    Does only trivial work (a numpy slice + a ~36 KB queue put) so it never
    blocks the main inference loop on the GIL.
    """
    from .screen_capture import get_preview_frame

    global _roi_image
    log_once: list = []
    _idle_since: float | None = None

    while not stop_event.is_set():
        t0 = time.perf_counter()

        enabled = getattr(config, "second_inference_mode", "off") == "v1_ocr"
        forced = _scan_flag.is_set()
        if forced:
            _scan_flag.clear()

        active = enabled or forced

        if active:
            # Detect and log child crashes before respawning
            if _proc is not None and not _proc.is_alive():
                logger.warning("[OCR] child process died (exit=%s); respawning", _proc.exitcode)
                _kill_proc()
            _ensure_proc()
            _idle_since = None
        elif _proc is not None:
            # Release idle child after timeout
            if _idle_since is None:
                _idle_since = t0
            elif (t0 - _idle_since) > _IDLE_TEARDOWN_S:
                _kill_proc()
                _idle_since = None

        if active:
            frame = get_preview_frame()
            if frame is not None:
                try:
                    roi = _crop_roi(frame, log_once)
                    with _roi_image_lock:
                        _roi_image = roi.copy()
                    # Keep only the freshest frame in the 1-slot queue.
                    if _frame_q is not None:
                        _drain(_frame_q)
                        try:
                            _frame_q.put_nowait(roi)
                        except queue.Full:
                            pass
                except Exception as exc:
                    logger.warning("[OCR] feed error: %s", exc)

        if _result_q is not None:
            _collect_results()

        fps = max(1, min(10, int(getattr(config, "second_inference_fps", 2))))
        elapsed = time.perf_counter() - t0
        interval = 0.2 if not active else max((1.0 / fps) - elapsed, 0.05)
        stop_event.wait(interval)
