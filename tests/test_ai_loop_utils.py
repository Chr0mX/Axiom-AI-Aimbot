"""Unit tests for core/ai_loop_utils.py.

Note: this module imports win32api at module scope (mouse-dispatch helpers
live alongside the pure logic tested here), so — like the rest of this
repo's win32-dependent tests (see test_mouse_methods.py) — the import is
deferred into a fixture instead of done at module top-level. A top-level
import would make a missing win32api abort collection of the *entire* test
suite; deferring it means only these tests fail individually, matching the
existing 160-failed/183-passed environment-only baseline on non-Windows.
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
