from __future__ import annotations

import queue
from typing import TYPE_CHECKING, Dict, List, Tuple

if TYPE_CHECKING:
    from .config import Config


def get_capture_dimensions(config: Config) -> Tuple[int, int]:
    """Get active capture dimensions based on screenshot backend."""

    screenshot_method = str(getattr(config, 'screenshot_method', 'mss')).lower()

    if screenshot_method == 'uvc':
        # "Fixed" crop mode freezes a centered detect_range_size square as
        # the capture size, for every uvc_capture_method:
        #   - ffmpeg: the subprocess itself crops+outputs only that square,
        #     so far less data crosses the subprocess pipe.
        #   - dshow/msmf: UVCCapture freezes which region grab() slices out
        #     of the in-process frame instead of recomputing it live.
        # Either way, reporting the frozen size here (same trick already
        # used for a pre-cropped UDP stream) makes calculate_detection_
        # region()'s region math naturally resolve to a full-frame no-op
        # crop against the already-cropped frame, with no special-casing
        # needed in grab().
        if str(getattr(config, 'uvc_crop_mode', 'dynamic')).lower() == 'fixed':
            crop_size = int(getattr(config, 'detect_range_size', 0) or 0) & ~1
            if crop_size > 0:
                return crop_size, crop_size
        cap_w = int(getattr(config, 'uvc_width', 0) or 0)
        cap_h = int(getattr(config, 'uvc_height', 0) or 0)
        if cap_w > 0 and cap_h > 0:
            return cap_w, cap_h
    elif screenshot_method == 'ndi':
        cap_w = int(getattr(config, 'ndi_width', 0) or 0)
        cap_h = int(getattr(config, 'ndi_height', 0) or 0)
        if cap_w > 0 and cap_h > 0:
            return cap_w, cap_h
    elif screenshot_method == 'udp':
        # Unlike uvc_width/ndi_width (user-configured), udp_width/udp_height
        # track the actual live stream resolution — the sender (e.g. an OBS
        # udp_stream_filter crop) can change it at any time. Falling back to
        # the full desktop resolution here (as if unconditionally reached)
        # would size the detection region against a canvas the stream no
        # longer matches, e.g. a stream cropped to 640x640: the region would
        # land outside the actual frame, UdpCapture.grab() would return None
        # every frame, and inference FPS would drop to 0.
        cap_w = int(getattr(config, 'udp_width', 0) or 0)
        cap_h = int(getattr(config, 'udp_height', 0) or 0)
        if cap_w > 0 and cap_h > 0:
            return cap_w, cap_h
    # getattr (not direct attribute access) so this stays safe against
    # partial/stub config objects — e.g. esp_server.py's test suite
    # deliberately calls this against a bare `class Empty: pass` to prove
    # the snapshot builder never crashes on a missing/incomplete config.
    return int(getattr(config, 'width', 1920)), int(getattr(config, 'height', 1080))


def update_crosshair_position(config: Config, half_width: int, half_height: int) -> None:
    """Update crosshair position"""

    if config.fov_follow_mouse:
        try:
            import win32api
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
    # Clamp against both dimensions, not just height — a capture source
    # narrower than tall (portrait UVC/NDI/UDP feed) would otherwise let
    # detection_size exceed capture_width, and region_width below would then
    # get clamped smaller than region_height, silently producing a
    # non-square region and defeating the square fast-preprocess path.
    detection_size = max(int(config.fov_size), min(int(capture_height), int(capture_width), detection_size))
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


def _circle_intersects_bbox(
    cx: float, cy: float, r: float,
    x1: float, y1: float, x2: float, y2: float,
) -> bool:
    # Ported from Someone_idea/fov_filter.py — true circle/AABB intersection test.
    # Finds the closest point on the rectangle to the circle centre, then checks
    # whether that point lies within the radius.
    lx, rx = (x1, x2) if x1 <= x2 else (x2, x1)
    ty, by = (y1, y2) if y1 <= y2 else (y2, y1)
    nx = min(max(cx, lx), rx)
    ny = min(max(cy, ty), by)
    return (nx - cx) ** 2 + (ny - cy) ** 2 <= r * r


def filter_boxes_by_fov(
    boxes: List[List[float]],
    confidences: List[float],
    crosshair_x: int,
    crosshair_y: int,
    fov_size: int,
    config=None,
) -> Tuple[List[List[float]], List[float]]:
    """FOV 過濾：只保留與 FOV 框有交集的人物框

    Extended to support an optional circular FOV test (Someone_idea/fov_filter.py).
    fov_circle_filter_enabled=False (default) keeps the original square behaviour.
    """

    if not boxes:
        return [], []

    use_circle = bool(getattr(config, 'fov_circle_filter_enabled', False))
    fov_half = fov_size // 2
    r = float(fov_half)
    fov_left   = crosshair_x - fov_half
    fov_top    = crosshair_y - fov_half
    fov_right  = crosshair_x + fov_half
    fov_bottom = crosshair_y + fov_half

    filtered_boxes = []
    filtered_confidences = []

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        if use_circle:
            keep = _circle_intersects_bbox(float(crosshair_x), float(crosshair_y), r, x1, y1, x2, y2)
        else:
            keep = x1 < fov_right and x2 > fov_left and y1 < fov_bottom and y2 > fov_top
        if keep:
            filtered_boxes.append(box)
            if i < len(confidences):
                filtered_confidences.append(confidences[i])

    return filtered_boxes, filtered_confidences


def find_closest_target(
    boxes: List[List[float]],
    confidences: List[float],
    crosshair_x: int,
    crosshair_y: int,
    priority_mode: str = "distance",
    confidence_weight: float = 0.5,
) -> Tuple[List[List[float]], List[float]]:
    """Single-target mode — keep the one target that wins the priority scoring."""

    if not boxes:
        return [], []

    best_box = None
    best_conf = 0.5
    best_score = float('inf')

    for i, box in enumerate(boxes):
        abs_x1, abs_y1, abs_x2, abs_y2 = box
        cx = (abs_x1 + abs_x2) * 0.5
        cy = (abs_y1 + abs_y2) * 0.5
        dx = cx - crosshair_x
        dy = cy - crosshair_y
        distance_sq = dx * dx + dy * dy
        conf = confidences[i] if i < len(confidences) else 0.5

        if priority_mode == 'confidence':
            score = 1.0 - conf
        elif priority_mode == 'composite':
            score = distance_sq * (1.0 - conf * confidence_weight)
        else:
            score = distance_sq

        if score < best_score:
            best_score = score
            best_box = box
            best_conf = conf

    if best_box:
        return [best_box], [best_conf]
    return [], []


def reduce_boxes_for_single_target(
    boxes: List[List[float]],
    confidences: List[float],
    locked_box: List[float] | None,
    locked_confidence: float,
    aimed_this_frame: bool,
    crosshair_x: int,
    crosshair_y: int,
    priority_mode: str = "distance",
    confidence_weight: float = 0.5,
) -> Tuple[List[List[float]], List[float]]:
    """single_target_mode's box-list reduction — the list auto-fire/preview/ESP
    (config.latest_boxes) see when only one target should be shown/acted on.

    Extracted from ai_loop.py so this exact selection logic — the fix for
    sticky lock being silently defeated by single_target_mode — has a home
    that's independently testable outside the threaded capture/inference loop.

    aimed_this_frame must reflect whether process_aiming() actually ran and
    updated locked_box/locked_confidence THIS frame (i.e. `is_aiming and
    boxes` was true) — not just whether boxes is non-empty. A stale
    locked_box held over from an earlier aiming frame (e.g. sticky lock still
    decaying a hold during an idle-detect frame where aiming didn't run) must
    never be reused with no fresh IOU check against the current boxes list;
    doing so previously let auto-fire/ESP/preview show a position with no
    current detection backing it. When aimed_this_frame is False, this always
    falls back to a fresh priority pick (find_closest_target) instead.
    """
    if aimed_this_frame and locked_box is not None:
        return [list(locked_box)], [locked_confidence]
    if boxes:
        return find_closest_target(
            boxes, confidences, crosshair_x, crosshair_y,
            priority_mode=priority_mode, confidence_weight=confidence_weight,
        )
    return [], []


def update_queues(
    overlay_boxes_queue: queue.Queue,
    overlay_confidences_queue: queue.Queue,
    boxes: List[List[float]],
    confidences: List[float],
    auto_fire_queue: queue.Queue | None = None,
    auto_fire_boxes: List[List[float]] | None = None,
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
        auto_fire_queue.put(list(auto_fire_boxes) if auto_fire_boxes is not None else list(boxes))
