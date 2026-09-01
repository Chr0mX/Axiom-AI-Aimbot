from __future__ import annotations

import queue
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from .config import Config
    from .ai_loop_state import LoopState


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
        # Prefer the actual negotiated resolution over the user-configured
        # request. Most UVC/webcam drivers only support a fixed set of
        # standard modes and silently negotiate to the nearest one if the
        # exact requested uvc_width/uvc_height isn't available, so the real
        # captured frame can differ from what was asked for. Using the
        # requested size here desyncs every downstream consumer of these
        # dimensions from the actual frame — most visibly the Web ESP web
        # overlay, whose "screen" w/h the browser client scales every drawn
        # box/FOV/crosshair against (esp_server.py's _build_snapshot() /
        # app.js's scale calc), but also update_crosshair_position()'s
        # crosshair-center assumption and calculate_detection_region()'s
        # actual model-input crop. Same "actual over requested" pattern the
        # udp branch below already uses for udp_width/udp_height. 0 means
        # not yet negotiated (no frame received yet); fall back to the
        # requested value only in that case.
        cap_w = int(getattr(config, 'uvc_actual_width', 0) or 0)
        cap_h = int(getattr(config, 'uvc_actual_height', 0) or 0)
        if cap_w > 0 and cap_h > 0:
            return cap_w, cap_h
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


def apply_cam_shift_deadzone(value: float, threshold: float) -> float:
    """Zero out `value` if its magnitude is below `threshold`; otherwise
    return it unchanged.

    Used by ai_loop.py's _preprocess_worker to gate what accumulates into
    state.cam_drift_x/y (the running integral of the phase-correlation-
    measured per-frame shift that ai_aiming.py's camera-drift-compensated
    prediction subtracts from the raw target position before it reaches the
    velocity predictor/Kalman filter). That integral feeds a
    frame-to-frame-differenced, then horizon-extrapolated, signal, so
    phase correlation's own measurement noise floor (quantization from the
    cam_motion_comp_size downsample, sensor/compression noise) gets
    amplified in a way the existing one-frame PID-error use of
    state.cam_shift_x/y itself (unfiltered, not differenced or
    extrapolated) never showed. Deadzoning here — only accumulating a
    shift big enough to plausibly be real motion — fixed a reported wobble
    without touching that unrelated, unfiltered PID-error path.
    """
    return value if abs(value) >= threshold else 0.0


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


def get_effective_detect_range_size(
    config: Config, capture_dims: Tuple[int, int] | None = None,
) -> int:
    """detect_range_size actually usable for detection, clamped to the
    active capture method's own live dimensions.

    config.detect_range_size is only validated (config.py's
    _validate_detect_range_size) against config.height — the full desktop
    height. For 'uvc'/'ndi'/'udp', where the live capture frame can be far
    smaller than the desktop (e.g. a small UDP crop), that leaves the raw
    field free to hold a value the actual capture frame can't satisfy.
    calculate_detection_region() has always clamped it down further, per
    frame, against get_capture_dimensions() — this factors that same
    clamp out so any other consumer (e.g. the Web ESP snapshot) reports
    the same effective size instead of the raw, potentially-too-large one.

    capture_dims lets a caller that already has (capture_width,
    capture_height) pass them through instead of triggering a second
    get_capture_dimensions() call for the same frame.
    """
    capture_width, capture_height = (
        capture_dims if capture_dims is not None else get_capture_dimensions(config)
    )
    detection_size = int(getattr(config, 'detect_range_size', capture_height))
    # getattr (not direct attribute access) — unlike calculate_detection_region()'s
    # original inline version of this formula (only ever called with a real,
    # fully-populated Config from ai_loop.py), this helper is also called from
    # esp_server.py's snapshot builder, which must tolerate a bare/incomplete
    # config object (see test_snapshot_handles_empty_and_missing).
    fov_size = int(getattr(config, 'fov_size', 0) or 0)
    # fov_height defaults to fov_size (not 0) so a config that predates
    # fov_height, or a bare stub missing it, still behaves like the square
    # FOV this clamp originally assumed.
    fov_height = int(getattr(config, 'fov_height', fov_size) or fov_size)
    # Clamp against both dimensions, not just height — a capture source
    # narrower than tall (portrait UVC/NDI/UDP feed) would otherwise let
    # detection_size exceed capture_width, and region_width below would then
    # get clamped smaller than region_height, silently producing a
    # non-square region and defeating the square fast-preprocess path.
    # The lower bound is max(fov_size, fov_height) — the square detection
    # region must contain the whole FOV rectangle, not just its width.
    return max(fov_size, fov_height, min(int(capture_height), int(capture_width), detection_size))


def compute_effective_fov(config: Config, state: LoopState, current_time: float) -> Tuple[int, int]:
    """FOV width/height actually in effect this frame, applying "Reduce FOV
    on Active Target" (fov_reduce_on_target_enabled) on top of the
    configured fov_size/fov_height.

    This is a gradual ramp, not an instant snap: once a target locks, the
    FOV starts at the full configured fov_size/fov_height and linearly
    shrinks down to fov_min_size_pct% of that over fov_min_size_duration
    seconds, then holds at that minimum for the rest of the current lock
    (fov_min_size_duration <= 0 means an instant drop straight to the
    minimum, no ramp).

    Mutates state.fov_reduce_since to track the ramp across frames: the
    None -> non-None edge of state.locked_box arms it exactly once (set to
    current_time), not every frame the target stays locked — see
    LoopState.fov_reduce_since for why that distinction matters. Ramp
    progress is then just (current_time - fov_reduce_since) / duration,
    clamped to [0, 1], so it naturally reaches 1.0 (full min-size) and
    simply stays there — no separate "expired" bookkeeping needed. Losing
    the lock resets fov_reduce_since back to 0.0 so the next acquisition
    starts its own fresh ramp from full size; the feature being off does
    the same, so re-enabling it later never inherits a stale ramp from
    however long ago it was last on.
    """
    effective_size = int(getattr(config, 'fov_size', 0) or 0)
    effective_height = int(getattr(config, 'fov_height', effective_size) or effective_size)

    if not getattr(config, 'fov_reduce_on_target_enabled', False):
        state.fov_reduce_since = 0.0
        return effective_size, effective_height

    if getattr(state, 'locked_box', None) is None:
        state.fov_reduce_since = 0.0
        return effective_size, effective_height

    if state.fov_reduce_since == 0.0:
        state.fov_reduce_since = current_time

    duration = float(getattr(config, 'fov_min_size_duration', 0.0) or 0.0)
    if duration <= 0.0:
        progress = 1.0
    else:
        progress = max(0.0, min(1.0, (current_time - state.fov_reduce_since) / duration))

    pct = max(1.0, min(100.0, float(getattr(config, 'fov_min_size_pct', 100.0) or 100.0)))
    current_pct = 100.0 - (100.0 - pct) * progress
    effective_size = max(1, int(effective_size * current_pct / 100.0))
    effective_height = max(1, int(effective_height * current_pct / 100.0))

    return effective_size, effective_height


def calculate_detection_region(config: Config, crosshair_x: int, crosshair_y: int) -> Dict[str, int]:
    """Calculate detection region"""

    capture_width, capture_height = get_capture_dimensions(config)
    detection_size = get_effective_detect_range_size(config, (capture_width, capture_height))
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


def _ellipse_intersects_bbox(
    cx: float, cy: float, a: float, b: float,
    x1: float, y1: float, x2: float, y2: float,
) -> bool:
    # True ellipse/AABB intersection test (semi-axes a, b — a == b is a
    # circle of radius a, the original Someone_idea/fov_filter.py case).
    # Scales both the ellipse centre and the box corners by (1/a, 1/b),
    # which turns the ellipse into a unit circle while keeping the box
    # axis-aligned (a per-axis scale can't tilt it) — then runs the same
    # closest-point-on-box-to-centre test the plain circle case always used,
    # just in the transformed space. That equivalence is exact: a point
    # (x,y) satisfies ((x-cx)/a)^2 + ((y-cy)/b)^2 <= 1 iff its scaled image
    # (x/a, y/b) lies within a unit circle centred at (cx/a, cy/b).
    if a <= 0 or b <= 0:
        return False
    lx, rx = (x1, x2) if x1 <= x2 else (x2, x1)
    ty, by = (y1, y2) if y1 <= y2 else (y2, y1)
    tcx, tcy = cx / a, cy / b
    tlx, trx = lx / a, rx / a
    tty, tby = ty / b, by / b
    nx = min(max(tcx, tlx), trx)
    ny = min(max(tcy, tty), tby)
    return (nx - tcx) ** 2 + (ny - tcy) ** 2 <= 1.0


def filter_boxes_by_fov(
    boxes: List[List[float]],
    confidences: List[float],
    crosshair_x: int,
    crosshair_y: int,
    fov_size: int,
    config=None,
    fov_height: Optional[float] = None,
) -> Tuple[List[List[float]], List[float]]:
    """FOV 過濾：只保留與 FOV 框有交集的人物框

    fov_size is the FOV's width; its height comes from config.fov_height,
    defaulting to fov_size (a square) if config doesn't have it — every
    existing caller that only ever set a square FOV keeps behaving exactly
    as before. Extended to support an optional elliptical FOV test
    (Someone_idea/fov_filter.py's original circle, generalized to an
    ellipse — a circle when fov_height == fov_size, same as it always was).
    fov_circle_filter_enabled=False (default) keeps the rectangular test.

    fov_height (the parameter, not config.fov_height) lets a caller override
    the height for just this call without touching config — used by "Reduce
    FOV on Active Target" (ai_loop.py) to shrink both dimensions together
    while a target is locked, since config.fov_height must stay the user's
    real configured value. Omit it (the default) to read config.fov_height
    exactly as before; every other caller is unaffected.
    """

    if not boxes:
        return [], []

    use_circle = bool(getattr(config, 'fov_circle_filter_enabled', False))
    if fov_height is None:
        fov_height = int(getattr(config, 'fov_height', fov_size) or fov_size)
    fov_half_x = fov_size / 2.0
    fov_half_y = fov_height / 2.0
    fov_left   = crosshair_x - fov_half_x
    fov_top    = crosshair_y - fov_half_y
    fov_right  = crosshair_x + fov_half_x
    fov_bottom = crosshair_y + fov_half_y

    filtered_boxes = []
    filtered_confidences = []

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        if use_circle:
            keep = _ellipse_intersects_bbox(
                float(crosshair_x), float(crosshair_y), fov_half_x, fov_half_y, x1, y1, x2, y2)
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
