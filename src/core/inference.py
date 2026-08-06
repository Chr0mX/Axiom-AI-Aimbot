# inference.py
"""AI inference module - Image preprocessing, post-processing, and PID controller"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple, Any

import cv2
import numpy as np
import numpy.typing as npt

# Pre-allocated letterbox canvases keyed by model_input_size.
# Reused across frames to eliminate per-frame np.full() allocation.
_canvas_cache: Dict[int, npt.NDArray[np.uint8]] = {}

# Minimum detections before postprocess_outputs' 'auto' box-format inference
# will conclude xyxy. Below this the test (x2>x1 and y2>y1 for every row) is
# too easily satisfied by chance on cxcywh data — see the comment at the
# check itself — so cxcywh, the far more common encoding, is assumed instead.
_XYXY_AUTO_MIN_DETECTIONS = 3

_warned_box_format = False


def _warn_once_box_format() -> None:
    """Say something the first time the box encoding is *inferred* rather
    than configured. Silent inference is how a wrong guess here stays
    invisible: it corrupts every box without erroring."""
    global _warned_box_format
    if _warned_box_format:
        return
    _warned_box_format = True
    logging.getLogger(__name__).info(
        "[Inference] Model output inferred as xyxy box format. If detections "
        "look wrong, set model_box_format to 'cxcywh' (or 'xyxy' to pin this "
        "choice) rather than relying on inference."
    )


class PIDController:
    """PID Controller - used for smooth aiming movement
    
    Implements Proportional-Integral-Derivative (PID) control algorithm for calculating mouse movement.
    Supports independent X/Y axis settings and includes dynamic P-parameter adjustment.
    
    Attributes:
        Kp: Proportional coefficient, controls reaction speed
        Ki: Integral coefficient, corrects static error
        Kd: Derivative coefficient, suppresses jitter and overshoot
    """
    
    # Loop period the historical (dt-less) tuning was implicitly written
    # against — Config.detect_interval's default. See update() for why this
    # is a reference point rather than a hardcoded assumption.
    REFERENCE_DT: float = 0.01

    def __init__(self, Kp: float, Ki: float, Kd: float) -> None:
        self.Kp = Kp  # Proportional
        self.Ki = Ki  # Integral
        self.Kd = Kd  # Derivative
        self.reset()

    def reset(self) -> None:
        """Reset controller state"""
        self.integral: float = 0.0
        self.previous_error: float = 0.0

    def update(self, error: float, dt: float | None = None) -> float:
        """
        Calculates control output based on current error.

        Args:
            error: Current error (e.g., target_x - current_x)
            dt:    Seconds since the previous update. None (or <= 0) keeps
                   the historical fixed-step behaviour.

        Returns:
            Control amount (e.g., amount mouse should move)

        Time normalization
        ------------------
        The I and D terms are scaled by ``dt / REFERENCE_DT`` rather than by
        ``dt`` directly, which is deliberate.

        The controller previously did ``integral += error`` and
        ``derivative = error - previous_error`` with no notion of elapsed
        time, so the effective Ki and Kd were a function of how fast the
        loop happened to be running. Tuning done at 100 Hz behaved
        differently at 240 Hz — and *changed underfoot at runtime*, because
        ``idle_detect_interval`` deliberately runs the loop slower while not
        aiming, so simply releasing the aim key altered the response.

        Scaling by raw ``dt`` would be the textbook fix but would silently
        invalidate every saved config: at a 10 ms step, Ki would become
        ~100x weaker and Kd ~100x stronger. Normalizing against
        REFERENCE_DT instead makes the ratio exactly 1.0 at the historical
        default rate — so existing tunings behave *identically* to before —
        while still cancelling the rate dependence everywhere else. No
        config migration is needed, which is why this is safe to enable
        unconditionally.
        """
        if dt is not None and dt > 0.0:
            # Clamp the ratio so one stalled frame (GC pause, model
            # hot-swap, a breakpoint) can't inject a huge integral step or a
            # near-zero derivative and kick the output.
            scale = max(0.25, min(4.0, dt / self.REFERENCE_DT))
        else:
            scale = 1.0

        # Integral term (with anti-windup clamping)
        self.integral += error * scale
        self.integral = max(-1000.0, min(1000.0, self.integral))

        # Derivative term — rate of change, so it divides by the step
        derivative = (error - self.previous_error) / scale

        # Adjust P parameter response curve
        adjusted_kp = self._calculate_adjusted_kp(self.Kp)

        # Calculate output
        output = (adjusted_kp * error) + (self.Ki * self.integral) + (self.Kd * derivative)

        # Update previous error
        self.previous_error = error

        return output
    
    def _calculate_adjusted_kp(self, kp: float) -> float:
        """Linear, predictable P response — effective Kp equals the configured Kp.

        The previous curve tripled the gain above 0.5 (kp=1.0 → 2.0), which made the
        upper half of the strength slider overshoot and wobble. With a clamped identity
        the response is monotonic and stable across the whole range; the GUI Kp slider
        is scaled so its full travel stays inside the proven-stable band.

        Args:
            kp: Configured P parameter value (0.0 ~ 1.0)

        Returns:
            Effective P parameter value (0.0 ~ 1.0)
        """
        return max(0.0, min(1.0, kp))


def preprocess_image(
    image: npt.NDArray[np.uint8],
    model_input_size: int,
    fast_resize: bool = False,
) -> Tuple[npt.NDArray[np.float32], float, int, int]:
    """Preprocess image for ONNX inference using letterboxing.

    Letterboxing (uniform scale + grey padding) preserves the original aspect
    ratio instead of stretching the image.  This is critical for Y-axis
    accuracy: a non-square capture region (e.g. at a screen edge) used to be
    distorted when resized to the square model input, causing the model to
    predict bounding box heights that were systematically off.  With
    letterboxing the model always sees correctly-proportioned content.

    Args:
        image: Input frame (BGR or BGRA).
        model_input_size: Square side length expected by the ONNX model (e.g. 640).

    Returns:
        (blob, scale, pad_x, pad_y) where
        - blob    : float32 tensor [1, 3, H, W] ready for model.run()
        - scale   : uniform scale factor applied (original → resized)
        - pad_x   : horizontal padding added to each side (pixels)
        - pad_y   : vertical   padding added to each side (pixels)

        Pass scale / pad_x / pad_y to postprocess_outputs() so it can
        reverse the letterbox transform and recover screen coordinates.
    """
    if image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    h, w = image.shape[:2]

    # Fast path: image is already the right square size (common for screen
    # capture with detection_size == model_input_size).
    if h == model_input_size and w == model_input_size:
        blob = cv2.dnn.blobFromImage(
            image,
            scalefactor=1.0 / 255.0,
            size=(model_input_size, model_input_size),
            swapRB=True,
            crop=False,
        )
        return np.ascontiguousarray(blob, dtype=np.float32), 1.0, 0, 0

    if fast_resize:
        # Direct nearest-neighbour resize — no grey padding canvas, no pad math.
        # Fastest path for square capture regions (detect_range_size is always
        # square, so there is no aspect-ratio distortion).
        fast_scale = model_input_size / h  # h == w for square captures
        fast_blob = cv2.dnn.blobFromImage(
            cv2.resize(image, (model_input_size, model_input_size),
                       interpolation=cv2.INTER_NEAREST),
            scalefactor=1.0 / 255.0,
            size=(model_input_size, model_input_size),
            swapRB=True,
            crop=False,
        )
        return np.ascontiguousarray(fast_blob, dtype=np.float32), fast_scale, 0, 0

    # Uniform scale so the longer side fits in model_input_size.
    scale = min(model_input_size / w, model_input_size / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Centre the resized image on a grey canvas (114 = YOLO default fill).
    pad_x = (model_input_size - new_w) // 2
    pad_y = (model_input_size - new_h) // 2
    if model_input_size not in _canvas_cache:
        _canvas_cache[model_input_size] = np.full(
            (model_input_size, model_input_size, 3), 114, dtype=np.uint8
        )
    canvas = _canvas_cache[model_input_size]
    canvas[:] = 114
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

    blob = cv2.dnn.blobFromImage(
        canvas,
        scalefactor=1.0 / 255.0,
        size=(model_input_size, model_input_size),
        swapRB=True,
        crop=False,
    )
    return np.ascontiguousarray(blob, dtype=np.float32), scale, pad_x, pad_y


def postprocess_outputs(
    outputs: List[Any],
    original_width: int,
    original_height: int,
    model_input_size: int,
    min_confidence: float,
    offset_x: int = 0,
    offset_y: int = 0,
    letterbox_scale: float = 1.0,
    letterbox_pad_x: int = 0,
    letterbox_pad_y: int = 0,
    box_format: str = 'auto',
    has_objectness: str = 'auto',
) -> Tuple[List[List[float]], List[float], List[int]]:
    """Post-process ONNX model output into screen-space bounding boxes.

    Y-axis fix
    ----------
    When preprocess_image() uses letterboxing the model predictions are in
    letterboxed coordinate space.  We must reverse the letterbox transform
    (remove padding, divide by scale) before mapping to original image space.
    Without this step, Y-axis coordinates were systematically shifted whenever
    the capture region was non-square (e.g. when the crosshair is near a screen
    edge), causing accurate X tracking but inaccurate Y tracking.

    Args:
        outputs:          Raw ONNX model outputs.
        original_width:   Width of the captured region (pixels).
        original_height:  Height of the captured region (pixels).
        model_input_size: Square side the model was run at (e.g. 640).
        min_confidence:   Detection confidence threshold (0–1).
        offset_x:         Region left edge in screen coordinates.
        offset_y:         Region top  edge in screen coordinates.
        letterbox_scale:  Scale returned by preprocess_image().
        letterbox_pad_x:  Horizontal padding returned by preprocess_image().
        letterbox_pad_y:  Vertical   padding returned by preprocess_image().

    Returns:
        (boxes, confidences) with boxes as [[x1, y1, x2, y2], …] in absolute
        screen coordinates.
    """
    raw = outputs[0][0]
    # Layout B (features × anchors): shape[0] < shape[1], needs .T to become (anchors, features)
    # Layout A (anchors × features): shape[0] > shape[1], already correct
    _is_layout_b = raw.ndim == 2 and raw.shape[0] < raw.shape[1]
    predictions = raw.T if _is_layout_b else raw

    # --- Objectness column (YOLOv5-family exports) ---
    # YOLOv5 lays out [cx, cy, w, h, objectness, cls0..clsN]; YOLOv8 drops
    # objectness entirely and uses [cx, cy, w, h, cls0..clsN]. Treating a
    # v5 export as v8 folds objectness into the class max — so confidence
    # becomes max(obj, cls...) instead of obj * cls — and shifts every class
    # id by one, because argmax then counts objectness as class 0. That id
    # shift is invisible until something actually reads class names, which
    # is exactly what the semantic filter does.
    #
    # There is no shape that distinguishes 4+1+N from 4+M, so 'auto' uses
    # the strongest available signal (v5 exports are anchors-major with a
    # class block; v8 are features-major) and can be overridden when it
    # guesses wrong.
    _objectness_mode = str(has_objectness or 'auto').lower()
    if _objectness_mode == 'yes':
        _has_obj = True
    elif _objectness_mode == 'no':
        _has_obj = False
    else:
        _has_obj = (not _is_layout_b) and predictions.shape[1] > 6

    if _has_obj:
        _obj = predictions[:, 4]
        _cls = predictions[:, 5:]
        _conf_scores = _obj * _cls.max(axis=1) if _cls.shape[1] else _obj
        _class_col_start = 5
    else:
        # Max class score (cols 4+) so any layout reports a real 0-1
        # confidence. Reading col 4 raw would yield a bbox coordinate
        # (~thousands) for Layout A models.
        _conf_scores = predictions[:, 4:].max(axis=1)
        _class_col_start = 4

    conf_mask = _conf_scores >= min_confidence
    filtered_predictions = predictions[conf_mask]

    if len(filtered_predictions) == 0:
        return [], [], []

    # --- Box encoding ---
    # Layout B models are always cxcywh after the transpose above; Layout A
    # models may export either cxcywh or xyxy.
    #
    # 'auto' infers xyxy from x2>x1 and y2>y1 holding for every detection —
    # but that test is far weaker than it looks. On cxcywh data those
    # columns are w and h, so it is really asking "is w > cx and h > cy for
    # every row", which is entirely possible by coincidence for a target
    # near the top-left with a large box. With one or two detections it is
    # close to a coin flip, and guessing wrong silently corrupts every box.
    # So 'auto' now also requires enough rows to make coincidence unlikely,
    # and defaults to cxcywh (overwhelmingly the common case) below that.
    # Anyone whose model really is xyxy can pin it instead of relying on
    # detection count.
    _box_format = str(box_format or 'auto').lower()
    if _box_format == 'xyxy':
        _treat_as_xyxy = not _is_layout_b
    elif _box_format == 'cxcywh':
        _treat_as_xyxy = False
    else:
        _treat_as_xyxy = (
            not _is_layout_b
            and len(filtered_predictions) >= _XYXY_AUTO_MIN_DETECTIONS
            and np.all(filtered_predictions[:, 2] > filtered_predictions[:, 0])
            and np.all(filtered_predictions[:, 3] > filtered_predictions[:, 1])
        )
        if _treat_as_xyxy:
            _warn_once_box_format()

    if _treat_as_xyxy:
        fp = filtered_predictions
        _cx = (fp[:, 0] + fp[:, 2]) * 0.5
        _cy = (fp[:, 1] + fp[:, 3]) * 0.5
        _w  =  fp[:, 2] - fp[:, 0]
        _h  =  fp[:, 3] - fp[:, 1]
        filtered_predictions = filtered_predictions.copy()
        filtered_predictions[:, 0] = _cx
        filtered_predictions[:, 1] = _cy
        filtered_predictions[:, 2] = _w
        filtered_predictions[:, 3] = _h

    cx = filtered_predictions[:, 0]
    cy = filtered_predictions[:, 1]
    w  = filtered_predictions[:, 2]
    h  = filtered_predictions[:, 3]

    # Reverse letterbox: remove padding offsets then undo the uniform scale.
    # This maps model-space coordinates back to original-capture-space coordinates.
    inv_scale = 1.0 / letterbox_scale if letterbox_scale > 0 else 1.0
    cx = (cx - letterbox_pad_x) * inv_scale
    cy = (cy - letterbox_pad_y) * inv_scale
    w  = w  * inv_scale
    h  = h  * inv_scale

    # Map from capture-region space to absolute screen space.
    x1 = cx - w / 2 + offset_x
    y1 = cy - h / 2 + offset_y
    x2 = cx + w / 2 + offset_x
    y2 = cy + h / 2 + offset_y

    boxes = np.stack([x1, y1, x2, y2], axis=1).tolist()
    confidences = _conf_scores[conf_mask].tolist()

    # Class IDs: argmax over the class columns only. _class_col_start is 4
    # for YOLOv8-style layouts (class 0 lives in col 4) and 5 when an
    # objectness column precedes them — counting objectness as a class is
    # what shifted every id by one on v5-family models.
    num_cols = filtered_predictions.shape[1]
    if num_cols > _class_col_start + 1:
        class_ids = filtered_predictions[:, _class_col_start:].argmax(axis=1).tolist()
        class_ids = [int(c) for c in class_ids]
    else:
        class_ids = [0] * len(boxes)

    return boxes, confidences, class_ids


def non_max_suppression(
    boxes: List[List[float]],
    confidences: List[float],
    iou_threshold: float = 0.4,
    class_ids: List[int] | None = None,
) -> Tuple[List[List[float]], List[float], List[int]]:
    """非極大值抑制 (non-maximum suppression).

    Returns ``(boxes, confidences, class_ids)``.

    ``class_ids`` is threaded through rather than left to the caller
    because NMS both **drops** boxes and **reorders** the survivors into
    descending-confidence order — so a class-id list produced alongside the
    input boxes no longer lines up with the output positionally. Callers
    previously kept using the pre-NMS list, which meant every downstream
    consumer indexing ``class_ids[i]`` by box index (notably
    ``detection_semantics.filter_detections_by_semantic_class``) read the
    class of an unrelated detection. That is invisible for single-class
    models — every id is 0 — and wrong for exactly the multi-class models
    the semantic class-name filter exists to serve.

    Passing ``class_ids=None`` yields a zero-filled list of the correct
    output length, so the return shape is uniform either way.

    Args:
        boxes: 邊界框列表
        confidences: 置信度列表
        iou_threshold: IoU 閾值
        class_ids: per-box class ids, same order/length as ``boxes``

    Returns:
        (filtered_boxes, filtered_confidences, filtered_class_ids)
    """
    if len(boxes) == 0:
        return [], [], []

    if len(boxes) == 1:
        single_ids = [int(class_ids[0])] if class_ids else [0]
        return boxes, confidences, single_ids

    boxes_arr = np.array(boxes)
    confidences_arr = np.array(confidences)
    areas = (boxes_arr[:, 2] - boxes_arr[:, 0]) * (boxes_arr[:, 3] - boxes_arr[:, 1])
    order = confidences_arr.argsort()[::-1]
    
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        
        xx1 = np.maximum(boxes_arr[i, 0], boxes_arr[order[1:], 0])
        yy1 = np.maximum(boxes_arr[i, 1], boxes_arr[order[1:], 1])
        xx2 = np.minimum(boxes_arr[i, 2], boxes_arr[order[1:], 2])
        yy2 = np.minimum(boxes_arr[i, 3], boxes_arr[order[1:], 3])
        
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        intersection = w * h
        union = areas[i] + areas[order[1:]] - intersection
        iou = intersection / np.maximum(union, 1e-6)  # 防止除零
        
        order = order[1:][iou <= iou_threshold]

    # `keep` holds original indices in descending-confidence order — reindex
    # class_ids by exactly the same list so element i of every returned list
    # describes the same detection.
    if class_ids:
        kept_class_ids = [int(class_ids[i]) if i < len(class_ids) else 0 for i in keep]
    else:
        kept_class_ids = [0] * len(keep)

    return boxes_arr[keep].tolist(), confidences_arr[keep].tolist(), kept_class_ids
