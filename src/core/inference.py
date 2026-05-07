# inference.py
"""AI inference module - Image preprocessing, post-processing, and PID controller"""

from __future__ import annotations

from typing import List, Tuple, Any

import cv2
import numpy as np
import numpy.typing as npt


class PIDController:
    """PID Controller - used for smooth aiming movement
    
    Implements Proportional-Integral-Derivative (PID) control algorithm for calculating mouse movement.
    Supports independent X/Y axis settings and includes dynamic P-parameter adjustment.
    
    Attributes:
        Kp: Proportional coefficient, controls reaction speed
        Ki: Integral coefficient, corrects static error
        Kd: Derivative coefficient, suppresses jitter and overshoot
    """
    
    def __init__(self, Kp: float, Ki: float, Kd: float) -> None:
        self.Kp = Kp  # Proportional
        self.Ki = Ki  # Integral
        self.Kd = Kd  # Derivative
        self.reset()

    def reset(self) -> None:
        """Reset controller state"""
        self.integral: float = 0.0
        self.previous_error: float = 0.0

    def update(self, error: float) -> float:
        """
        Calculates control output based on current error
        
        Args:
            error: Current error (e.g., target_x - current_x)
            
        Returns:
            Control amount (e.g., amount mouse should move)
        """
        # Integral term (with anti-windup clamping)
        self.integral += error
        self.integral = max(-1000.0, min(1000.0, self.integral))
        
        # Derivative term
        derivative = error - self.previous_error
        
        # Adjust P parameter response curve
        adjusted_kp = self._calculate_adjusted_kp(self.Kp)
        
        # Calculate output
        output = (adjusted_kp * error) + (self.Ki * self.integral) + (self.Kd * derivative)
        
        # Update previous error
        self.previous_error = error
        
        return output
    
    def _calculate_adjusted_kp(self, kp: float) -> float:
        """Calculate dynamically adjusted P parameter
        
        Implements non-linear P parameter response curve:
        - 0% ~ 50%: Linear growth, maintains original proportion
        - 50% ~ 100%: Accelerated growth, eventually scaling to 200%
        
        This design allows for smoother low sensitivity and more aggressive high sensitivity.
        
        Args:
            kp: Original P parameter value (0.0 ~ 1.0)
            
        Returns:
            Adjusted P parameter value (0.0 ~ 2.0)
        """
        if kp <= 0.5:
            return kp
        else:
            # When kp=0.5, output=0.5; when kp=1.0, output=2.0
            return 0.5 + (kp - 0.5) * 3.0


def preprocess_image(
    image: npt.NDArray[np.uint8],
    model_input_size: int,
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

    # Uniform scale so the longer side fits in model_input_size.
    scale = min(model_input_size / w, model_input_size / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Centre the resized image on a grey canvas (114 = YOLO default fill).
    pad_x = (model_input_size - new_w) // 2
    pad_y = (model_input_size - new_h) // 2
    canvas = np.full((model_input_size, model_input_size, 3), 114, dtype=np.uint8)
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
) -> Tuple[List[List[float]], List[float]]:
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
    predictions = outputs[0][0].T

    conf_mask = predictions[:, 4] >= min_confidence
    filtered_predictions = predictions[conf_mask]

    if len(filtered_predictions) == 0:
        return [], []

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
    confidences = filtered_predictions[:, 4].tolist()

    return boxes, confidences


def non_max_suppression(
    boxes: List[List[float]], 
    confidences: List[float], 
    iou_threshold: float = 0.4
) -> Tuple[List[List[float]], List[float]]:
    """
    非極大值抑制
    
    Args:
        boxes: 邊界框列表
        confidences: 置信度列表
        iou_threshold: IoU 閾值
        
    Returns:
        (filtered_boxes, filtered_confidences) 元組
    """
    if len(boxes) == 0:
        return [], []
    
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
        
    return boxes_arr[keep].tolist(), confidences_arr[keep].tolist()
