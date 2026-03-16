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


def preprocess_image(image: npt.NDArray[np.uint8], model_input_size: int) -> npt.NDArray[np.float32]:
    """
    Preprocess image to fit ONNX model
    
    Args:
        image: Input image (BGR format)
        model_input_size: Model input size
        
    Returns:
        Preprocessed tensor [1, 3, H, W]
    """
    # Optimization 1: Use cvtColor to handle BGRA -> BGR
    # This is faster than numpy slicing (image[:, :, :3]) and directly produces contiguous memory
    if image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    # 優化 2: 顯式調整大小並使用 INTER_NEAREST (最近鄰插值)
    # 當從小圖 (如 222) 放大到大圖 (如 640) 時，預設的線性插值非常耗時
    # INTER_NEAREST 速度極快，能大幅降低 pre-process 時間
    if image.shape[0] != model_input_size or image.shape[1] != model_input_size:
        image = cv2.resize(image, (model_input_size, model_input_size), interpolation=cv2.INTER_NEAREST)

    # blob: [1, 3, H, W] float32
    # 因為已經 resize 過，這裡的 resize 動作會被跳過或開銷極小
    blob = cv2.dnn.blobFromImage(
        image,
        scalefactor=1.0 / 255.0,
        size=(model_input_size, model_input_size),
        swapRB=True,
        crop=False,
    )

    # 確保連續記憶體布局（避免某些後端額外拷貝）
    return np.ascontiguousarray(blob, dtype=np.float32)


def postprocess_outputs(
    outputs: List[Any], 
    original_width: int, 
    original_height: int, 
    model_input_size: int, 
    min_confidence: float, 
    offset_x: int = 0, 
    offset_y: int = 0
) -> Tuple[List[List[float]], List[float]]:
    """
    後處理 ONNX 模型輸出
    
    Args:
        outputs: 模型輸出
        original_width: 原始圖像寬度
        original_height: 原始圖像高度
        model_input_size: 模型輸入尺寸
        min_confidence: 最小置信度閾值
        offset_x: X 軸偏移
        offset_y: Y 軸偏移
        
    Returns:
        (boxes, confidences) 元組
    """
    predictions = outputs[0][0].T
    
    # 向量化過濾：先篩選高置信度的檢測
    conf_mask = predictions[:, 4] >= min_confidence
    filtered_predictions = predictions[conf_mask]
    
    if len(filtered_predictions) == 0:
        return [], []
    
    # 向量化計算邊界框
    scale_x = original_width / model_input_size
    scale_y = original_height / model_input_size
    
    cx, cy, w, h = (filtered_predictions[:, 0], filtered_predictions[:, 1], 
                    filtered_predictions[:, 2], filtered_predictions[:, 3])
    
    x1 = (cx - w / 2) * scale_x + offset_x
    y1 = (cy - h / 2) * scale_y + offset_y
    x2 = (cx + w / 2) * scale_x + offset_x
    y2 = (cy + h / 2) * scale_y + offset_y

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
