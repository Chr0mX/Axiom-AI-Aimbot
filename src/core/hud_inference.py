"""
Secondary HUD inference — YOLO11n ONNX weapon/attachment detector.

PROCESS-ISOLATED DESIGN
-----------------------
Mirrors ocr_inference.py: the ONNX Runtime session and all pre/post-processing
run in a separate child process so the main inference loop and Qt UI are never
blocked by this work.

ROI: normally cropped from the shared screen_capture preview frame using
     config.hud_roi_coords ("x1,y1,x2,y2"), defaulting to "1490,953,1870,1041"
     (Apex Legends HUD strip, 1080p reference). When config.hud_udp_enabled is
     set, that shared preview frame is bypassed entirely in favor of a
     *second*, independent UDP JPEG stream (its own UdpJpegReceiver, bound to
     hud_udp_bind_ip/hud_udp_bind_port) — for a 2PC/OBS setup where the main
     screenshot_method='udp' stream is itself already a small center crop
     (e.g. the aim FOV) that excludes the HUD region entirely, and so never
     contains the HUD pixels hud_roi_coords would otherwise crop out of it.
     In that mode, the received frame already *is* the HUD ROI (the crop
     happens OBS-side, via a second udp_stream_filter filter instance on the
     same source with its own crop rect and its own target port) —
     hud_roi_coords is not applied to it.
Model input: 320 x 320 NCHW float32, letterboxed with grey fill (114). (best.onnx
     itself uses a 1280 x 1280 input — the actual dims are always read from the
     loaded model's own input metadata, this is just the historical default.)
Output: two genuinely different shapes are handled, auto-detected per model
     (see _postprocess()): legacy raw-grid YOLO (v8/v11-style)
     [1, 4+num_classes, num_anchors], needing our own argmax/threshold/NMS; and
     Ultralytics end-to-end exports (YOLO26, or any -nms-baked-in graph)
     [1, max_det, 6], already NMS'd with each row [x1, y1, x2, y2, confidence,
     class_id] in absolute input-pixel space and nothing left to decode.

Public API (identical shape to ocr_inference.py):
  start(config)           — spawn child process + feeder thread (no-op if running)
  stop()                  — stop both and join
  trigger_hud_scan()      — force an immediate scan
  get_hud_results()       — current detection lines, e.g. ["R301: 92%"]
  get_hud_roi_image()     — last ROI crop (for GUI preview)
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import queue
import sys
import threading
import time
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .config import Config

logger = logging.getLogger(__name__)

_HUD_ROI_DEFAULT_STR = "1490,953,1870,1041"
_IDLE_TEARDOWN_S = 5.0  # seconds before releasing an idle child process
# _postprocess() used to cap this at 5 -- fine when the only consumer cared
# about "the single most likely weapon," but a categorized status readout
# (weapon + fire mode + up to 6 attachment slots, see hud_categories.py) can
# have that many simultaneous above-threshold detections in one frame, each
# in its own distinct screen position. A flat top-5-by-raw-score cap would
# silently drop a legitimate lower-scoring category (e.g. a dim Turbo icon)
# whenever enough higher-scoring ones (weapon, optic) were also present.
_MAX_RESULT_LINES = 20


def _parse_roi(coords: str) -> dict[str, int] | None:
    """Parse 'x1,y1,x2,y2' into a crop dict. Returns None on invalid input."""
    try:
        x1, y1, x2, y2 = map(int, coords.strip().split(','))
        if x2 > x1 and y2 > y1:
            return {"left": x1, "top": y1, "width": x2 - x1, "height": y2 - y1}
    except Exception:
        pass
    return None


# ── Parent-side shared state ──────────────────────────────────────────────────
_results_lock = threading.Lock()
_hud_results: list[str] = []

_boxes_lock = threading.Lock()
_hud_boxes: list[tuple] = []

_model_wh_lock = threading.Lock()
_model_input_wh: tuple[int, int] = (320, 320)

_roi_image_lock = threading.Lock()
_roi_image: np.ndarray | None = None

_scan_flag = threading.Event()

_stop_event: threading.Event | None = None
_feeder_thread: threading.Thread | None = None

# ── Child-process plumbing ────────────────────────────────────────────────────
_proc: mp.Process | None = None
_proc_stop = None
_frame_q = None     # mp.Queue(maxsize=1)  parent → child: (roi_bgra, model_path, confidence)
_result_q = None    # mp.Queue(maxsize=4)  child → parent: (list[str], list[tuple])

# ── Dedicated second UDP stream for the HUD strip (hud_udp_enabled) ──────────
# Only relevant for a 2PC/OBS setup where the *main* screenshot_method='udp'
# stream is itself already a small center crop (e.g. the 640x640/320x320 aim
# FOV) that excludes the HUD region entirely — the shared
# screen_capture.get_preview_frame() this module otherwise reads from simply
# never contains the HUD pixels in that case. When hud_udp_enabled, a second
# udp_stream_filter OBS filter instance (independent crop rect + independent
# target port, stacked on the same source) sends the HUD strip to its own
# port, and this module owns its own UdpJpegReceiver for it — completely
# independent of whatever screen_capture.py's main capture backend is doing.
_hud_udp_receiver = None            # UdpJpegReceiver | None
_hud_udp_signature: tuple | None = None   # (bind_ip, bind_port) the receiver above was built with


# ── Public API ────────────────────────────────────────────────────────────────

def get_hud_results() -> list[str]:
    with _results_lock:
        return list(_hud_results)


def get_hud_boxes() -> list[tuple]:
    """Return latest bounding boxes as (x1, y1, x2, y2, class_id, score) in model-input space."""
    with _boxes_lock:
        return list(_hud_boxes)


def get_hud_model_size() -> tuple[int, int]:
    """Return (inp_w, inp_h) of the currently loaded model."""
    with _model_wh_lock:
        return _model_input_wh


def get_hud_roi_image() -> np.ndarray | None:
    with _roi_image_lock:
        return _roi_image if _roi_image is None else _roi_image.copy()


def trigger_hud_scan() -> None:
    _scan_flag.set()


def _ensure_proc() -> None:
    """Spawn the HUD child process if not already alive. Called from the feeder."""
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
            name="HUDProcess", daemon=True,
        )
        _proc.start()
        logger.info("[HUD] child process started (pid=%s)", _proc.pid)
    except Exception as exc:
        logger.error("[HUD] Failed to start HUD process: %s", exc)
        _proc = None


def _kill_proc() -> None:
    """Tear down the HUD child process, leaving the feeder thread running."""
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
        logger.info("[HUD] child process released")
    # Each mp.Queue owns a background feeder thread in this process; leaving
    # them for GC instead of closing explicitly works but can accumulate
    # fd/semaphore churn across frequent start/stop toggling in a long
    # session. cancel_join_thread() first so close() doesn't block waiting
    # to flush a queue whose reader (the child process) is already gone.
    for q in (_frame_q, _result_q):
        if q is not None:
            try:
                q.cancel_join_thread()
                q.close()
            except Exception:
                pass
    _proc = None
    _proc_stop = None
    _frame_q = None
    _result_q = None


def _ensure_hud_udp_receiver(config: "Config") -> None:
    """(Re)build the dedicated HUD UDP receiver if not already running with
    the current bind_ip/bind_port. Called from the feeder only while
    hud_udp_enabled and HUD inference is active."""
    global _hud_udp_receiver, _hud_udp_signature
    bind_ip = str(getattr(config, "hud_udp_bind_ip", "0.0.0.0"))
    bind_port = int(getattr(config, "hud_udp_bind_port", 5601))
    signature = (bind_ip, bind_port)
    if _hud_udp_receiver is not None and _hud_udp_signature == signature:
        return
    _kill_hud_udp_receiver()
    try:
        from .udp_receiver import UdpJpegReceiver
        _hud_udp_receiver = UdpJpegReceiver(bind_ip=bind_ip, bind_port=bind_port)
        _hud_udp_receiver.start()
        _hud_udp_signature = signature
        logger.info("[HUD] dedicated UDP receiver bound to %s:%d", bind_ip, bind_port)
    except Exception as exc:
        logger.error("[HUD] Failed to start dedicated HUD UDP receiver: %s", exc)
        _hud_udp_receiver = None
        _hud_udp_signature = None


def _kill_hud_udp_receiver() -> None:
    global _hud_udp_receiver, _hud_udp_signature
    if _hud_udp_receiver is not None:
        try:
            _hud_udp_receiver.stop()
        except Exception:
            pass
    _hud_udp_receiver = None
    _hud_udp_signature = None


def start(config: "Config") -> None:
    """Start the HUD feeder thread. Child process is spawned lazily on first use. No-op if running."""
    global _stop_event, _feeder_thread
    if _feeder_thread is not None and _feeder_thread.is_alive():
        return

    _stop_event = threading.Event()
    _feeder_thread = threading.Thread(
        target=_feeder, args=(config, _stop_event), name="HUDFeeder", daemon=True,
    )
    _feeder_thread.start()


def stop() -> None:
    """Stop feeder thread, HUD child process, and the dedicated HUD UDP receiver (if any)."""
    global _stop_event, _feeder_thread
    if _stop_event is not None:
        _stop_event.set()
    _kill_proc()
    _kill_hud_udp_receiver()
    if _feeder_thread is not None and _feeder_thread.is_alive():
        _feeder_thread.join(timeout=2.0)
    _stop_event = None
    _feeder_thread = None


# ── Shared helpers ────────────────────────────────────────────────────────────

def _crop_roi(frame: np.ndarray, roi: dict[str, int], log_once: list) -> np.ndarray:
    """Crop frame to the given roi dict, clamping to frame bounds."""
    h, w = frame.shape[:2]
    l, t = roi["left"], roi["top"]
    rw, rh = roi["width"], roi["height"]
    x1, y1 = min(l, w), min(t, h)
    x2, y2 = min(l + rw, w), min(t + rh, h)
    if not log_once:
        logger.info("[HUD] Frame %dx%d → ROI [x:%d-%d, y:%d-%d]", w, h, x1, x2, y1, y2)
        log_once.append(True)
    return frame[y1:y2, x1:x2]


def _drain(q) -> None:
    try:
        while True:
            q.get_nowait()
    except queue.Empty:
        pass


def _collect_results() -> None:
    global _model_input_wh
    latest = None
    try:
        while True:
            latest = _result_q.get_nowait()
    except queue.Empty:
        pass
    except Exception:
        return
    if latest is not None:
        if isinstance(latest, tuple) and len(latest) == 4:
            lines, boxes, inp_w, inp_h = latest
            with _model_wh_lock:
                _model_input_wh = (inp_w, inp_h)
        elif isinstance(latest, tuple) and len(latest) == 2:
            lines, boxes = latest
        else:
            lines, boxes = latest, []
        with _results_lock:
            _hud_results[:] = lines
        with _boxes_lock:
            _hud_boxes[:] = boxes


# ── Child process ─────────────────────────────────────────────────────────────

def _letterbox(img_bgr: np.ndarray, inp_w: int, inp_h: int) -> np.ndarray:
    """Letterbox img_bgr into an (inp_h × inp_w) canvas with grey fill (114)."""
    import cv2
    h, w = img_bgr.shape[:2]
    scale = min(inp_w / w, inp_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((inp_h, inp_w, 3), 114, dtype=np.uint8)
    pad_x = (inp_w - new_w) // 2
    pad_y = (inp_h - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    return canvas


def _preprocess(roi: np.ndarray, inp_w: int, inp_h: int) -> np.ndarray:
    """Return NCHW float32 blob [1, 3, inp_h, inp_w] from a BGR(A) ROI crop."""
    import cv2
    if roi.ndim == 3 and roi.shape[2] == 4:
        roi = cv2.cvtColor(roi, cv2.COLOR_BGRA2BGR)
    lb = _letterbox(roi, inp_w, inp_h)
    rgb = lb[:, :, ::-1].astype(np.float32) / 255.0
    return rgb.transpose(2, 0, 1)[np.newaxis]   # [1, 3, inp_h, inp_w]


def _nms_class_agnostic(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float = 0.5) -> list[int]:
    """Greedy IoU suppression across ALL classes, not just within one.

    Ultralytics' end-to-end export runs NMS inside the graph, but per
    Ultralytics' own default that NMS is class-aware — it only suppresses
    overlapping boxes that share a class, not overlapping boxes of
    different classes. For an ambiguous small HUD icon, several distinct
    weapon/attachment classes can each cross the confidence threshold for
    the exact same region and all survive as separate, heavily-overlapping
    boxes (visible as stacked/duplicate boxes in the ROI preview, and as
    the wrong class occasionally winning "highest score" for that region).
    This collapses each such overlapping cluster down to just its single
    highest-confidence detection, regardless of which class it is.

    boxes: [N, 4] x1,y1,x2,y2. Returns kept indices (into boxes/scores),
    sorted score-descending.
    """
    if len(scores) == 0:
        return []
    order = np.argsort(scores)[::-1]
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    suppressed = np.zeros(len(order), dtype=bool)
    keep: list[int] = []
    for i in range(len(order)):
        if suppressed[i]:
            continue
        idx = order[i]
        keep.append(int(idx))
        rest = order[i + 1:]
        if len(rest) == 0:
            continue
        xx1 = np.maximum(x1[idx], x1[rest])
        yy1 = np.maximum(y1[idx], y1[rest])
        xx2 = np.minimum(x2[idx], x2[rest])
        yy2 = np.minimum(y2[idx], y2[rest])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = areas[idx] + areas[rest] - inter
        iou = np.where(union > 0, inter / union, 0.0)
        suppressed[i + 1:][iou > iou_thresh] = True
    return keep


def _postprocess(output: np.ndarray, num_classes: int, threshold: float,
                 class_names: list[str] | None,
                 is_end2end: bool = False) -> tuple[list[str], list[tuple]]:
    """Decode a HUD-detector output; return (lines, boxes).

    boxes: list of (x1, y1, x2, y2, class_id, score) in model-input space.

    Two genuinely different shapes are handled, selected by is_end2end (see
    _child_main's model-load block for how that's decided):
    - Legacy raw-grid YOLO (v8/v11-style) — [1, 4+C, anchors] or
      [1, anchors, 4+C]: per-anchor center-box (cx,cy,w,h) plus a per-class
      score block; we do our own argmax + threshold + top-K here.
    - Ultralytics end-to-end export (YOLO26, or any -nms-baked-in graph) —
      [1, max_det, 6]: NMS already ran inside the graph, so each row is
      already [x1, y1, x2, y2, confidence, class_id] in absolute
      input-pixel space. Treating this like the legacy format silently
      misreads column 5 (the real class_id) as a second "class score" and
      re-derives bogus center-box coordinates from what are already
      corner coordinates — this was a real, shipped bug (best.onnx is a
      YOLO26s end2end export) that produced a fixed, wrong num_classes and
      garbage out-of-bounds boxes.
    """
    data = output[0]

    if is_end2end:
        # [max_det, 6]: x1, y1, x2, y2, score, class_id — already NMS'd,
        # already absolute pixel coords. Some exports pad unused detection
        # slots with all-zero/near-zero-confidence rows, which the plain
        # score filter below already drops.
        if data.shape[-1] != 6 and data.shape[0] == 6:
            data = data.T
        scores = data[:, 4].astype(np.float64)
        class_ids = np.round(data[:, 5]).astype(np.int64)

        top_score = float(scores.max()) if len(scores) > 0 else 0.0
        if len(scores) > 0 and class_names:
            top_idx = np.argsort(scores)[::-1][:3]
            top_str = "  ".join(
                f"{class_names[int(class_ids[i])] if 0 <= int(class_ids[i]) < len(class_names) else class_ids[i]}="
                f"{float(scores[i]):.3f}"
                for i in top_idx
            )
            print(f"[HUD child] threshold={threshold}  top3: {top_str}")
        else:
            print(f"[HUD child] max_score={top_score:.3f} threshold={threshold} dets={len(scores)}")

        mask = scores >= threshold
        if not mask.any():
            if top_score > 0.001 and class_names:
                best_i = int(np.argmax(scores))
                best_cid = int(class_ids[best_i])
                best_name = class_names[best_cid] if 0 <= best_cid < len(class_names) else str(best_cid)
                x1, y1, x2, y2 = (float(v) for v in data[best_i, :4])
                hint_box = [(x1, y1, x2, y2, best_cid, -top_score)]
                return [f"[below threshold] best: {best_name} {top_score:.1%}  (threshold={threshold:.0%})"], hint_box
            return [], []

        passing_idx = np.where(mask)[0]
        scores_f = scores[mask]
        class_ids_f = class_ids[mask]
        boxes_f = data[passing_idx, :4]
        order = _nms_class_agnostic(boxes_f, scores_f, iou_thresh=0.5)[:_MAX_RESULT_LINES]
        lines: list[str] = []
        boxes_out: list[tuple] = []
        for i in order:
            orig = passing_idx[i]
            cid = int(class_ids_f[i])
            score = float(scores_f[i])
            x1, y1, x2, y2 = (float(v) for v in data[orig, :4])
            name = class_names[cid] if (class_names and 0 <= cid < len(class_names)) else str(cid)
            lines.append(f"{name}: {int(score * 100)}%")
            boxes_out.append((x1, y1, x2, y2, cid, score))
        return lines, boxes_out

    # ---- legacy raw-grid path ----
    # Auto-detect: if rows < cols it's [4+C, A] → transpose to [A, 4+C]
    if data.shape[0] < data.shape[1]:
        data = data.T
    # data is now [A, 4+C]

    conf = data[:, 4:4 + num_classes]
    class_ids = conf.argmax(axis=1)
    scores = conf[np.arange(len(conf)), class_ids]

    top_score = float(scores.max()) if len(scores) > 0 else 0.0

    # Log top-3 classes so we can see what the model is actually seeing
    if len(scores) > 0 and class_names:
        top_idx = np.argsort(scores)[::-1][:3]
        top_str = "  ".join(
            f"{class_names[int(class_ids[i])] if int(class_ids[i]) < len(class_names) else class_ids[i]}="
            f"{float(scores[i]):.3f}"
            for i in top_idx
        )
        print(f"[HUD child] threshold={threshold}  top3: {top_str}")
    else:
        print(f"[HUD child] max_score={top_score:.3f} threshold={threshold} anchors={len(scores)}")

    mask = scores >= threshold
    if not mask.any():
        if top_score > 0.001 and class_names:
            best_cid = int(class_ids[np.argmax(scores)])
            best_name = class_names[best_cid] if best_cid < len(class_names) else str(best_cid)
            # Return best candidate as an orange "hint" box (negative score signals below-threshold)
            best_orig = int(np.argmax(scores))
            cx, cy, bw, bh = data[best_orig, :4]
            x1, y1 = float(cx - bw / 2), float(cy - bh / 2)
            x2, y2 = float(cx + bw / 2), float(cy + bh / 2)
            hint_box = [(x1, y1, x2, y2, best_cid, -top_score)]  # negative score = below threshold
            return [f"[below threshold] best: {best_name} {top_score:.1%}  (threshold={threshold:.0%})"], hint_box
        return [], []

    passing_idx = np.where(mask)[0]
    scores_f = scores[mask]
    class_ids_f = class_ids[mask]

    order = np.argsort(scores_f)[::-1][:_MAX_RESULT_LINES]
    lines: list[str] = []
    boxes_out: list[tuple] = []
    for i in order:
        orig = passing_idx[i]
        cid = int(class_ids_f[i])
        score = float(scores_f[i])
        cx, cy, bw, bh = data[orig, :4]
        x1, y1 = float(cx - bw / 2), float(cy - bh / 2)
        x2, y2 = float(cx + bw / 2), float(cy + bh / 2)
        name = class_names[cid] if (class_names and cid < len(class_names)) else str(cid)
        lines.append(f"{name}: {int(score * 100)}%")
        boxes_out.append((x1, y1, x2, y2, cid, score))
    return lines, boxes_out


def _child_main(frame_q, result_q, proc_stop) -> None:
    """Run in a separate process: own GIL, below-normal priority."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

    # Inject AppData site-packages (for PaddleOCR/user-installed packages)
    la = os.environ.get("LOCALAPPDATA", "")
    if la:
        pkg = os.path.join(la, "AxiomAI", "site-packages")
        if os.path.isdir(pkg) and pkg not in sys.path:
            sys.path.insert(0, pkg)

    # Inject src/ and src/python/dependencies/ so cv2, numpy, onnxruntime are
    # importable in the spawned child (mirrors main.py path setup).
    _here = os.path.dirname(os.path.abspath(__file__))   # .../src/core/
    _src  = os.path.dirname(_here)                        # .../src/
    for _extra in (
        os.path.join(_src, "python", "dependencies"),
        _src,
    ):
        if os.path.isdir(_extra) and _extra not in sys.path:
            sys.path.insert(0, _extra)

    if sys.platform == "win32":
        try:
            import ctypes
            BELOW_NORMAL = 0x00004000
            k32 = ctypes.windll.kernel32
            k32.SetPriorityClass(k32.GetCurrentProcess(), BELOW_NORMAL)
        except Exception:
            pass
    else:
        try:
            os.nice(10)
        except Exception:
            pass

    try:
        import onnxruntime as ort
    except ImportError as exc:
        msg = f"[HUD Error] onnxruntime not importable: {exc}"
        print(msg)
        try:
            result_q.put_nowait([msg])
        except Exception:
            pass
        return

    session = None
    current_model_path: str = ""
    class_names: list[str] | None = None
    num_classes: int = 0
    is_end2end: bool = False
    input_name: str = ""
    inp_w: int = 320
    inp_h: int = 320
    _last_roi_shape: tuple = ()

    while not proc_stop.is_set():
        try:
            item = frame_q.get(timeout=0.3)
        except queue.Empty:
            continue
        if item is None:
            break

        roi, model_path, confidence = item

        # (Re)load session when model changes
        if model_path != current_model_path:
            session = None
            current_model_path = model_path
            class_names = None
            num_classes = 0
            is_end2end = False
            if model_path and os.path.isfile(model_path):
                try:
                    opts = ort.SessionOptions()
                    opts.intra_op_num_threads = 1
                    opts.inter_op_num_threads = 1
                    session = ort.InferenceSession(
                        model_path,
                        sess_options=opts,
                        providers=["CPUExecutionProvider"],
                    )
                    inp_meta = session.get_inputs()[0]
                    input_name = inp_meta.name
                    # Read actual model input dims from NCHW shape [1, 3, H, W]
                    try:
                        inp_h = int(inp_meta.shape[2])
                        inp_w = int(inp_meta.shape[3])
                    except Exception:
                        inp_h, inp_w = 320, 320

                    meta = session.get_modelmeta().custom_metadata_map
                    if "names" in meta:
                        import ast
                        try:
                            raw = ast.literal_eval(meta["names"])
                            if isinstance(raw, dict):
                                class_names = [raw[i] for i in range(len(raw))]
                            elif isinstance(raw, list):
                                class_names = raw
                        except Exception:
                            pass

                    out_shape = session.get_outputs()[0].shape
                    smaller_dim = None
                    if len(out_shape) >= 3:
                        d1, d2 = int(out_shape[1]), int(out_shape[2])
                        smaller_dim = min(d1, d2)
                        num_classes = smaller_dim - 4  # legacy raw-grid assumption: smaller dim = 4+C

                    # Ultralytics end-to-end exports (YOLO26, or any model with
                    # NMS baked into the graph) emit a fixed [1, max_det, 6]
                    # output — nothing like the raw [1, 4+C, anchors] grid the
                    # min(d1,d2)-4 line above assumes. Detect it from the
                    # model's own "end2end" metadata (what Ultralytics embeds),
                    # falling back to a shape heuristic (output's smaller dim
                    # is a fixed 6 while the model has way more than 2 real
                    # classes) in case that metadata key is ever stripped.
                    end2end_flag = str(meta.get("end2end", "")).strip().lower() in ("true", "1", "yes")
                    is_end2end = end2end_flag or (
                        smaller_dim == 6 and bool(class_names) and len(class_names) > 2
                    )

                    if is_end2end:
                        num_classes = len(class_names) if class_names else 0
                    elif num_classes <= 0 and class_names:
                        num_classes = len(class_names)

                    print(f"[HUD child] Loaded: {os.path.basename(model_path)}"
                          f"  classes={num_classes}  input={inp_w}×{inp_h}"
                          f"  end2end={is_end2end}")
                except Exception as exc:
                    err = f"[HUD Error] Model load failed: {exc}"
                    print(err)
                    try:
                        result_q.put_nowait([err])
                    except Exception:
                        pass
                    session = None
            else:
                err = f"[HUD Error] Model not found: {model_path!r}"
                print(err)
                try:
                    result_q.put_nowait([err])
                except Exception:
                    pass

        if session is None:
            continue

        try:
            if roi.shape != _last_roi_shape:
                _last_roi_shape = roi.shape
                print(f"[HUD child] ROI shape={roi.shape} mean={float(roi.mean()):.1f} min={int(roi.min())} max={int(roi.max())}")
            blob = _preprocess(roi, inp_w, inp_h)
            outputs = session.run(None, {input_name: blob})
            output = outputs[0]

            if not is_end2end and num_classes <= 0:
                num_classes = output.shape[1] - 4

            lines, boxes = _postprocess(output, num_classes, confidence, class_names, is_end2end)
            try:
                result_q.put_nowait((lines, boxes, inp_w, inp_h))
            except queue.Full:
                try:
                    result_q.get_nowait()
                    result_q.put_nowait((lines, boxes, inp_w, inp_h))
                except Exception:
                    pass
        except Exception as exc:
            err = f"[HUD Error] Inference: {exc}"
            print(err)
            try:
                result_q.put_nowait([err])
            except Exception:
                pass


# ── Parent feeder thread ──────────────────────────────────────────────────────

def _get_udp_hud_frame(config: "Config") -> np.ndarray | None:
    """Decode the dedicated HUD UDP receiver's latest JPEG straight to BGR.

    Unlike _crop_roi() below, no cropping is applied here — the second
    udp_stream_filter OBS instance already sends exactly the HUD strip (its
    own crop rect, its own port), so the whole decoded frame *is* the ROI.
    hud_roi_coords (a full-screen-relative rectangle) doesn't apply to it.
    """
    import cv2
    if _hud_udp_receiver is None:
        return None
    jpeg_bytes, _frame_id = _hud_udp_receiver.get_latest_frame_with_id()
    if jpeg_bytes is None:
        return None
    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _feeder(config: "Config", stop_event: threading.Event) -> None:
    from .screen_capture import get_preview_frame

    global _roi_image
    log_once: list = []
    _idle_since: float | None = None

    _src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _project_root = os.path.dirname(_src)

    while not stop_event.is_set():
        t0 = time.perf_counter()

        enabled = getattr(config, "second_inference_mode", "off") == "v2_onnx"
        hud_model_rel = getattr(config, "hud_model_path", "")
        forced = _scan_flag.is_set()
        if forced:
            _scan_flag.clear()

        active = (enabled or forced) and bool(hud_model_rel)
        udp_mode = bool(getattr(config, "hud_udp_enabled", False))

        if active:
            # Detect and log child crashes before respawning
            if _proc is not None and not _proc.is_alive():
                logger.warning("[HUD] child process died (exit=%s); respawning", _proc.exitcode)
                _kill_proc()
            _ensure_proc()
            if udp_mode:
                _ensure_hud_udp_receiver(config)
            elif _hud_udp_receiver is not None:
                # Toggled off mid-session — release it immediately rather
                # than waiting out the idle-teardown timer below.
                _kill_hud_udp_receiver()
            _idle_since = None
        else:
            # Release idle child process / UDP receiver after timeout
            if _proc is not None or _hud_udp_receiver is not None:
                if _idle_since is None:
                    _idle_since = t0
                elif (t0 - _idle_since) > _IDLE_TEARDOWN_S:
                    _kill_proc()
                    _kill_hud_udp_receiver()
                    _idle_since = None

        if active:
            model_path = (hud_model_rel if os.path.isabs(hud_model_rel)
                          else os.path.join(_project_root, hud_model_rel))
            confidence = float(getattr(config, "hud_confidence", 0.10))

            if udp_mode:
                roi = _get_udp_hud_frame(config)
            else:
                coords_str = getattr(config, "hud_roi_coords", _HUD_ROI_DEFAULT_STR) or _HUD_ROI_DEFAULT_STR
                roi_dict = _parse_roi(coords_str) or _parse_roi(_HUD_ROI_DEFAULT_STR)
                frame = get_preview_frame()
                roi = _crop_roi(frame, roi_dict, log_once) if (frame is not None and roi_dict is not None) else None

            if roi is not None:
                try:
                    with _roi_image_lock:
                        _roi_image = roi.copy()
                    if _frame_q is not None:
                        _drain(_frame_q)
                        try:
                            _frame_q.put_nowait((roi, model_path, confidence))
                        except queue.Full:
                            pass
                except Exception as exc:
                    logger.warning("[HUD] feed error: %s", exc)

        if _result_q is not None:
            _collect_results()

        fps = max(1, min(10, int(getattr(config, "second_inference_fps", 2))))
        elapsed = time.perf_counter() - t0
        interval = 0.2 if not active else max((1.0 / fps) - elapsed, 0.05)
        stop_event.wait(interval)
