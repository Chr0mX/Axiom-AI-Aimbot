from __future__ import annotations

import queue
from typing import TYPE_CHECKING, Dict, List, Tuple

import win32api

if TYPE_CHECKING:
    from .config import Config


def get_capture_dimensions(config: Config) -> Tuple[int, int]:
    """Get active capture dimensions based on screenshot backend."""

    if str(getattr(config, 'screenshot_method', 'mss')).lower() == 'uvc':
        cap_w = int(getattr(config, 'uvc_width', 0) or 0)
        cap_h = int(getattr(config, 'uvc_height', 0) or 0)
        if cap_w > 0 and cap_h > 0:
            return cap_w, cap_h
    return int(config.width), int(config.height)


def update_crosshair_position(config: Config, half_width: int, half_height: int) -> None:
    """Update crosshair position"""

    if config.fov_follow_mouse:
        try:
            x, y = win32api.GetCursorPos()
            config.crosshairX, config.crosshairY = x, y
        except (OSError, RuntimeError):
            config.crosshairX, config.crosshairY = half_width, half_height
    else:
        config.crosshairX, config.crosshairY = half_width, half_height


def clear_queues(boxes_queue: queue.Queue, confidences_queue: queue.Queue) -> None:
    """Clear detection queues"""

    try:
        while not boxes_queue.empty():
            boxes_queue.get_nowait()
        while not confidences_queue.empty():
            confidences_queue.get_nowait()
    except queue.Empty:
        pass
    boxes_queue.put([])
    confidences_queue.put([])


def calculate_detection_region(config: Config, crosshair_x: int, crosshair_y: int) -> Dict[str, int]:
    """Calculate detection region"""

    capture_width, capture_height = get_capture_dimensions(config)
    detection_size = int(getattr(config, 'detect_range_size', capture_height))
    detection_size = max(int(config.fov_size), min(int(capture_height), detection_size))
    half_detection_size = detection_size // 2

    region_left = max(0, crosshair_x - half_detection_size)
    region_top = max(0, crosshair_y - half_detection_size)
    region_width = max(0, min(detection_size, capture_width - region_left))
    region_height = max(0, min(detection_size, capture_height - region_top))

    return {
        'left': region_left,
        'top': region_top,
        'width': region_width,
        'height': region_height,
    }


def filter_boxes_by_fov(
    boxes: List[List[float]],
    confidences: List[float],
    crosshair_x: int,
    crosshair_y: int,
    fov_size: int,
) -> Tuple[List[List[float]], List[float]]:
    """FOV 過濾：只保留與 FOV 框有交集的人物框"""

    if not boxes:
        return [], []

    fov_half = fov_size // 2
    fov_left = crosshair_x - fov_half
    fov_top = crosshair_y - fov_half
    fov_right = crosshair_x + fov_half
    fov_bottom = crosshair_y + fov_half

    filtered_boxes = []
    filtered_confidences = []

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        if x1 < fov_right and x2 > fov_left and y1 < fov_bottom and y2 > fov_top:
            filtered_boxes.append(box)
            if i < len(confidences):
                filtered_confidences.append(confidences[i])

    return filtered_boxes, filtered_confidences


def find_closest_target(
    boxes: List[List[float]],
    confidences: List[float],
    crosshair_x: int,
    crosshair_y: int,
) -> Tuple[List[List[float]], List[float]]:
    """單目標模式 - 只保留離準心最近的一個目標"""

    if not boxes:
        return [], []

    closest_box = None
    min_distance_sq = float('inf')
    closest_confidence = 0.5

    for i, box in enumerate(boxes):
        abs_x1, abs_y1, abs_x2, abs_y2 = box
        box_center_x = (abs_x1 + abs_x2) * 0.5
        box_center_y = (abs_y1 + abs_y2) * 0.5
        dx = box_center_x - crosshair_x
        dy = box_center_y - crosshair_y
        distance_sq = dx * dx + dy * dy

        if distance_sq < min_distance_sq:
            min_distance_sq = distance_sq
            closest_box = box
            closest_confidence = confidences[i] if i < len(confidences) else 0.5

    if closest_box:
        return [closest_box], [closest_confidence]
    return [], []


def update_queues(
    overlay_boxes_queue: queue.Queue,
    overlay_confidences_queue: queue.Queue,
    boxes: List[List[float]],
    confidences: List[float],
    auto_fire_queue: queue.Queue | None = None,
) -> None:
    """更新檢測結果隊列，並向自動開火單獨佇列廣播"""

    try:
        if overlay_boxes_queue.full():
            overlay_boxes_queue.get_nowait()
        if overlay_confidences_queue.full():
            overlay_confidences_queue.get_nowait()
    except queue.Empty:
        pass

    overlay_boxes_queue.put(boxes)
    overlay_confidences_queue.put(confidences)

    if auto_fire_queue is not None:
        try:
            if auto_fire_queue.full():
                auto_fire_queue.get_nowait()
        except queue.Empty:
            pass
        auto_fire_queue.put(list(boxes))
