from __future__ import annotations

import logging
import queue
from typing import TYPE_CHECKING, Dict, List, Tuple

if TYPE_CHECKING:
    from .ai_loop_state import LoopState
    from .config import Config

logger = logging.getLogger(__name__)


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
    # Clamp against both dimensions, not just height — a capture source
    # narrower than tall (portrait UVC/NDI/UDP feed) would otherwise let
    # detection_size exceed capture_width, and region_width below would then
    # get clamped smaller than region_height, silently producing a
    # non-square region and defeating the square fast-preprocess path.
    return max(fov_size, min(int(capture_height), int(capture_width), detection_size))


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


_unads_warned: set = set()


def apply_unads_transition(config: Config, state: LoopState, current_time: float) -> None:
    """Carry out a pending Auto Un-ADS release/re-engage transition decided by
    process_aiming() (state.unads_pending_transition), and independently
    enforce the safety-cap timeout.

    Fires a single click of the user's DEDICATED Auto Un-ADS key/button —
    config.auto_unads_makcu_button (routed through MAKCU's own command set)
    when mouse_move_method == 'makcu', otherwise config.auto_unads_key (any
    VK, via win_utils.aim_input.click_aim_key). This is deliberately a
    separate control from config.AimKeys / makcu_aim_button (the normal aim
    trigger) — it doesn't touch that key's state at all, so is_aiming's own
    computation in ai_loop.py is completely unaffected by this feature and
    needs no override to keep process_aiming() running through a release
    window.

    A single click (not a sustained press/release) is fired on BOTH the
    release and re-engage transitions, since state.unads_release_active is
    our own belief-state about whether we've toggled the game out of ADS —
    it doesn't track physical hold state the way the old (AimKeys-reusing)
    design needed to.

    Must be called unconditionally once per ai_loop.py iteration — not only
    on frames where process_aiming() ran. A full target loss (boxes empty)
    stops process_aiming() from running at all, which is exactly the
    scenario most likely to need the safety cap to still fire; relying on
    process_aiming() alone to enforce it would leave the belief-state (and
    the game's actual ADS state, if our assumption about it is right) stuck
    with no code path left to correct it.
    """
    # Lazy import: mirrors update_crosshair_position()'s local `import
    # win32api` above — keeps this module importable (and its other
    # functions collectible/testable) on non-Windows without win32api,
    # per CLAUDE.md's testing notes.
    from win_utils.aim_input import click_aim_key
    from win_utils.makcu_mouse import makcu_mouse

    transition = state.unads_pending_transition
    state.unads_pending_transition = ''

    use_makcu = getattr(config, 'mouse_move_method', '') == 'makcu'

    def _fire_click() -> bool:
        if use_makcu:
            btn = str(getattr(config, 'auto_unads_makcu_button', 'off')).lower()
            if btn == 'off':
                return False
            makcu_mouse.press_button(btn, 1)
            return True
        vk = int(getattr(config, 'auto_unads_key', 0) or 0)
        if vk == 0:
            return False
        click_aim_key(vk, str(getattr(config, 'mouse_move_method', '')))
        return True

    def _warn_unconfigured() -> None:
        key = ('makcu', ) if use_makcu else ('generic', )
        if key in _unads_warned:
            return
        _unads_warned.add(key)
        field = 'auto_unads_makcu_button' if use_makcu else 'auto_unads_key'
        logger.warning(
            "[AutoUnADS] enabled but %s is unconfigured — nothing to click.",
            field,
        )

    if transition == 'release' and not state.unads_release_active:
        if _fire_click():
            state.unads_release_active = True
            state.unads_release_start_time = current_time
            state.unads_clear_hold_start = 0.0
        else:
            _warn_unconfigured()

    elif transition == 'reengage' and state.unads_release_active:
        _fire_click()
        state.unads_release_active = False
        state.unads_release_start_time = 0.0
        state.unads_clear_hold_start = 0.0

    # Safety-cap backstop — independent of process_aiming() having run this
    # iteration (see docstring). Reuses the 'reengage' branch above via one
    # bounded recursive call rather than duplicating it; terminates after
    # exactly one extra call since unads_release_active is False by then.
    if state.unads_release_active:
        max_release_s = float(getattr(config, 'auto_unads_max_release_s', 3.0))
        if max_release_s > 0 and current_time - state.unads_release_start_time > max_release_s:
            state.unads_pending_transition = 'reengage'
            apply_unads_transition(config, state, current_time)


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
