"""Unit tests for core/ai_loop_utils.py.

Note: core.ai_loop_utils itself already defers its `import win32api` to
inside a function body rather than importing it at module scope, so simply
importing the module here is safe without win32api installed. The fixture
below still routes the import through pytest rather than a bare top-level
`import` anyway, matching the pattern used by this repo's genuinely
win32-dependent test modules (see test_mouse_methods.py) for consistency —
it isn't load-bearing for collection safety here the way it is there.
"""

import pytest


@pytest.fixture
def ai_loop_utils():
    from core import ai_loop_utils
    return ai_loop_utils


class TestFindClosestTarget:
    def test_empty_boxes_returns_empty(self, ai_loop_utils):
        boxes, confs = ai_loop_utils.find_closest_target([], [], 0, 0)
        assert boxes == [] and confs == []

    def test_distance_mode_picks_nearest_to_crosshair(self, ai_loop_utils):
        boxes = [[0, 0, 10, 10], [100, 100, 110, 110]]
        confs = [0.9, 0.99]
        picked_boxes, picked_confs = ai_loop_utils.find_closest_target(
            boxes, confs, crosshair_x=5, crosshair_y=5, priority_mode="distance")
        assert picked_boxes == [[0, 0, 10, 10]]
        assert picked_confs == [0.9]

    def test_confidence_mode_ignores_distance(self, ai_loop_utils):
        boxes = [[0, 0, 10, 10], [500, 500, 510, 510]]
        confs = [0.40, 0.95]
        picked_boxes, picked_confs = ai_loop_utils.find_closest_target(
            boxes, confs, crosshair_x=5, crosshair_y=5, priority_mode="confidence")
        # The far box wins purely on confidence despite being much farther away.
        assert picked_boxes == [[500, 500, 510, 510]]
        assert picked_confs == [0.95]

    def test_composite_mode_blends_distance_and_confidence(self, ai_loop_utils):
        # Near box has low confidence; far box has much higher confidence,
        # but only moderately farther (score = distance_sq * (1 - conf *
        # weight) — distance_sq dominates quadratically, so the near box
        # must NOT sit exactly on the crosshair (distance_sq=0 can never be
        # outweighted by any confidence factor) and the far box can't be
        # too many multiples farther or distance_sq swamps the confidence
        # term regardless of weight.
        boxes = [[5, 5, 15, 15], [25, 25, 35, 35]]  # centers (10,10), (30,30)
        confs = [0.05, 0.99]
        picked_boxes, _ = ai_loop_utils.find_closest_target(
            boxes, confs, crosshair_x=0, crosshair_y=0,
            priority_mode="composite", confidence_weight=0.95)
        assert picked_boxes == [[25, 25, 35, 35]]

    def test_missing_confidence_defaults_to_half(self, ai_loop_utils):
        # confidences shorter than boxes — index-out-of-range entries default to 0.5.
        boxes = [[0, 0, 10, 10]]
        picked_boxes, picked_confs = ai_loop_utils.find_closest_target(boxes, [], 5, 5)
        assert picked_boxes == [[0, 0, 10, 10]]
        assert picked_confs == [0.5]


class TestCalculateDetectionRegion:
    """detection_size must be clamped against both capture dimensions, not
    just height — a capture source narrower than tall (portrait UVC/NDI/UDP
    feed) would otherwise let detection_size exceed capture_width, and
    region_width (independently clamped to capture_width) would then come
    out smaller than region_height, silently producing a non-square region
    and defeating the square fast-preprocess path."""

    def test_clamps_against_narrower_dimension_for_portrait_source(self, ai_loop_utils):
        from types import SimpleNamespace
        config = SimpleNamespace(
            screenshot_method='mss', width=400, height=800,
            fov_size=100, detect_range_size=600,
        )
        region = ai_loop_utils.calculate_detection_region(config, crosshair_x=200, crosshair_y=400)
        assert region['width'] == region['height'], f"non-square region for a portrait source: {region}"
        assert region['width'] == 400

    def test_square_landscape_source_unaffected(self, ai_loop_utils):
        from types import SimpleNamespace
        config = SimpleNamespace(
            screenshot_method='mss', width=1920, height=1080,
            fov_size=100, detect_range_size=320,
        )
        region = ai_loop_utils.calculate_detection_region(config, crosshair_x=960, crosshair_y=540)
        assert region['width'] == region['height'] == 320


class TestGetEffectiveDetectRangeSize:
    """config.detect_range_size is only validated against the full desktop
    height, not the active capture method's own live dimensions — for
    'uvc'/'ndi'/'udp', where the live frame can be much smaller than the
    desktop, this leaves it free to hold a value the actual capture frame
    can't satisfy. get_effective_detect_range_size() must clamp it down to
    match what calculate_detection_region() actually uses, so any other
    consumer (e.g. the Web ESP snapshot) can't report a bigger, wrong size."""

    def test_clamps_raw_value_to_small_udp_crop(self, ai_loop_utils):
        from types import SimpleNamespace
        config = SimpleNamespace(
            screenshot_method='udp', width=1920, height=1080,
            udp_width=320, udp_height=320,
            fov_size=100, detect_range_size=900,  # valid only against the 1080-tall desktop
        )
        assert ai_loop_utils.get_effective_detect_range_size(config) == 320

    def test_matches_calculate_detection_region(self, ai_loop_utils):
        """Must agree with calculate_detection_region()'s own effective size —
        it's the same clamp, factored out."""
        from types import SimpleNamespace
        config = SimpleNamespace(
            screenshot_method='udp', width=1920, height=1080,
            udp_width=320, udp_height=320,
            fov_size=100, detect_range_size=900,
        )
        region = ai_loop_utils.calculate_detection_region(config, crosshair_x=160, crosshair_y=160)
        assert ai_loop_utils.get_effective_detect_range_size(config) == region['width'] == region['height']

    def test_accepts_precomputed_capture_dims(self, ai_loop_utils):
        """Passing capture_dims skips the internal get_capture_dimensions()
        call — must still clamp the same way."""
        from types import SimpleNamespace
        config = SimpleNamespace(fov_size=100, detect_range_size=900)
        assert ai_loop_utils.get_effective_detect_range_size(config, (320, 320)) == 320

    def test_tolerates_missing_fov_size(self, ai_loop_utils):
        """Must not crash on a stub config missing fov_size entirely —
        esp_server.py's snapshot builder calls this against partial configs."""
        from types import SimpleNamespace
        config = SimpleNamespace(detect_range_size=900)
        assert ai_loop_utils.get_effective_detect_range_size(config, (320, 320)) == 320


class TestReduceBoxesForSingleTarget:
    """Regression coverage for the single_target_mode / sticky_lock_enabled
    interaction fix: single_target_mode's box-list reduction must only trust
    state.locked_box when process_aiming() actually ran this frame
    (aimed_this_frame), never on a stale hold carried over from an earlier
    frame with no fresh IOU check against the current detections.
    """

    BOXES = [[0, 0, 10, 10], [100, 100, 110, 110]]
    CONFS = [0.9, 0.6]

    def test_aimed_this_frame_uses_locked_box_verbatim(self, ai_loop_utils):
        boxes, confs = ai_loop_utils.reduce_boxes_for_single_target(
            self.BOXES, self.CONFS,
            locked_box=[100, 100, 110, 110], locked_confidence=0.77,
            aimed_this_frame=True, crosshair_x=5, crosshair_y=5,
        )
        assert boxes == [[100, 100, 110, 110]]
        assert confs == [0.77]

    def test_locked_box_is_copied_not_aliased(self, ai_loop_utils):
        locked = [100, 100, 110, 110]
        boxes, _ = ai_loop_utils.reduce_boxes_for_single_target(
            self.BOXES, self.CONFS, locked_box=locked, locked_confidence=0.77,
            aimed_this_frame=True, crosshair_x=5, crosshair_y=5,
        )
        boxes[0][0] = 999
        assert locked[0] == 100, "mutating the returned box must not mutate state.locked_box"

    def test_not_aimed_this_frame_ignores_stale_lock(self, ai_loop_utils):
        """The core regression case: a locked_box left over from a previous
        aiming frame must NOT be reused on a frame where aiming didn't run
        (e.g. idle-detect while sticky lock is still decaying a hold) —
        it must fall back to a fresh priority pick over the current boxes."""
        boxes, confs = ai_loop_utils.reduce_boxes_for_single_target(
            self.BOXES, self.CONFS,
            locked_box=[100, 100, 110, 110], locked_confidence=0.77,
            aimed_this_frame=False, crosshair_x=5, crosshair_y=5,
        )
        # Nearest-to-crosshair box wins, not the stale locked one.
        assert boxes == [[0, 0, 10, 10]]
        assert confs == [0.9]

    def test_aimed_this_frame_but_no_lock_falls_back_to_priority_pick(self, ai_loop_utils):
        boxes, confs = ai_loop_utils.reduce_boxes_for_single_target(
            self.BOXES, self.CONFS, locked_box=None, locked_confidence=0.0,
            aimed_this_frame=True, crosshair_x=5, crosshair_y=5,
        )
        assert boxes == [[0, 0, 10, 10]]

    def test_no_boxes_returns_empty_regardless_of_lock(self, ai_loop_utils):
        boxes, confs = ai_loop_utils.reduce_boxes_for_single_target(
            [], [], locked_box=[1, 2, 3, 4], locked_confidence=0.5,
            aimed_this_frame=False, crosshair_x=0, crosshair_y=0,
        )
        assert boxes == [] and confs == []

    def test_respects_priority_mode_in_fallback_path(self, ai_loop_utils):
        boxes = [[0, 0, 10, 10], [400, 400, 410, 410]]
        confs = [0.10, 0.99]
        picked, _ = ai_loop_utils.reduce_boxes_for_single_target(
            boxes, confs, locked_box=None, locked_confidence=0.0,
            aimed_this_frame=False, crosshair_x=5, crosshair_y=5,
            priority_mode="confidence",
        )
        assert picked == [[400, 400, 410, 410]]
