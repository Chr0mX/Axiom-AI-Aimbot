"""Unit tests for core/detection_semantics.py — the semantic false-positive
filter (class-name deny list, min-geometry thresholds, aspect-ratio heuristic).
"""

from core.detection_semantics import (
    _filter_detections_min_geometry,
    _looks_like_environment_box,
    _looks_like_vehicle,
    _semantic_keep_label,
    filter_detections_by_semantic_class,
    filter_detections_by_target_class,
)


class _FakeConfig:
    """Minimal config stand-in — only the attributes detection_semantics.py reads."""
    fov_size = 240
    detect_min_bbox_area_px = 0.0
    detect_min_bbox_short_side_px = 0.0
    detect_min_bbox_max_side_frac = 0.0
    _detect_class_names = None
    aim_target_class_ids = []


class TestSemanticKeepLabel:
    def test_deny_substring_drops(self):
        assert _semantic_keep_label("tree_01") is False
        assert _semantic_keep_label("HUD_element") is False

    def test_allow_substring_keeps(self):
        assert _semantic_keep_label("player_head") is True
        assert _semantic_keep_label("enemy_body") is True

    def test_unknown_label_defaults_to_keep(self):
        # Neither list matches -> default is to keep (avoid over-filtering
        # unrecognized custom-model class names).
        assert _semantic_keep_label("some_custom_class") is True

    def test_empty_label_keeps(self):
        assert _semantic_keep_label("") is True

    def test_deny_wins_over_allow_when_both_substrings_present(self):
        # "tree" (deny) checked before allow list — a label matching both
        # should be dropped, not kept.
        assert _semantic_keep_label("tree_target") is False


class TestLooksLikeVehicle:
    def test_wide_flat_box_is_vehicle(self):
        assert _looks_like_vehicle(bw=60, bh=30, conf=0.5) is True

    def test_narrow_tall_box_is_not_vehicle(self):
        assert _looks_like_vehicle(bw=20, bh=60, conf=0.5) is False

    def test_zero_size_is_not_vehicle(self):
        assert _looks_like_vehicle(bw=0, bh=10, conf=0.5) is False


class TestLooksLikeEnvironmentBox:
    def test_high_confidence_never_flagged(self):
        # Above the 0.62 confidence gate, nothing is treated as environment
        # regardless of shape — a real target shouldn't be dropped just
        # because it's oddly shaped if the model is confident.
        assert _looks_like_environment_box(bw=200, bh=5, conf=0.9) is False

    def test_extreme_wide_low_confidence_flagged(self):
        assert _looks_like_environment_box(bw=200, bh=20, conf=0.3) is True

    def test_normal_player_proportions_not_flagged(self):
        assert _looks_like_environment_box(bw=30, bh=60, conf=0.4) is False


class TestFilterDetectionsMinGeometry:
    def test_all_thresholds_zero_is_a_noop(self):
        """Regression guard: before this session's fix, detect_min_bbox_*
        config fields didn't exist anywhere on Config, so getattr(...) always
        fell back to 0 here and this layer was permanently a no-op. Confirms
        that default (0.0) behavior is preserved exactly."""
        boxes = [[0, 0, 2, 2], [10, 10, 15, 15]]
        confs = [0.5, 0.6]
        cids = [0, 1]
        out_b, out_c, out_i = _filter_detections_min_geometry(boxes, confs, cids, _FakeConfig())
        assert out_b == boxes
        assert out_c == confs
        assert out_i == cids

    def test_min_short_side_drops_thin_boxes(self):
        cfg = _FakeConfig()
        cfg.detect_min_bbox_short_side_px = 10.0
        boxes = [[0, 0, 2, 50], [0, 0, 20, 50]]  # first is 2px wide (thin), second 20px
        confs = [0.5, 0.5]
        out_b, out_c, out_i = _filter_detections_min_geometry(boxes, confs, [0, 0], cfg)
        assert out_b == [[0, 0, 20, 50]]

    def test_min_area_drops_tiny_boxes(self):
        cfg = _FakeConfig()
        cfg.detect_min_bbox_area_px = 5000.0
        boxes = [[0, 0, 5, 5], [0, 0, 100, 100]]  # 25px^2 vs 10000px^2
        confs = [0.5, 0.5]
        out_b, out_c, out_i = _filter_detections_min_geometry(boxes, confs, [0, 0], cfg)
        assert out_b == [[0, 0, 100, 100]]

    def test_malformed_box_is_skipped_not_crashed(self):
        # Needs a non-zero threshold, else the all-zero fast path returns
        # boxes unfiltered before the per-box unpacking loop ever runs.
        cfg = _FakeConfig()
        cfg.detect_min_bbox_short_side_px = 1.0
        boxes = [[0, 0]]  # missing coords
        out_b, out_c, out_i = _filter_detections_min_geometry(boxes, [0.5], [0], cfg)
        assert out_b == []


class TestFilterDetectionsBySemanticClass:
    def test_empty_boxes_passthrough(self):
        out_b, out_c, out_i = filter_detections_by_semantic_class([], [], [], _FakeConfig())
        assert out_b == [] and out_c == [] and out_i == []

    def test_denied_class_name_is_dropped(self):
        cfg = _FakeConfig()
        cfg._detect_class_names = {0: "tree", 1: "player_head"}
        boxes = [[0, 0, 20, 40], [50, 50, 70, 90]]
        confs = [0.8, 0.85]
        cids = [0, 1]
        out_b, out_c, out_i = filter_detections_by_semantic_class(boxes, confs, cids, cfg)
        assert out_b == [[50, 50, 70, 90]]
        assert out_i == [1]

    def test_no_class_name_map_keeps_everything_class_layer(self):
        # Without _detect_class_names, the class-name deny layer can't run —
        # only geometry/aspect-ratio heuristics apply.
        cfg = _FakeConfig()
        boxes = [[0, 0, 20, 40]]
        confs = [0.8]
        out_b, out_c, out_i = filter_detections_by_semantic_class(boxes, confs, [0], cfg)
        assert out_b == [[0, 0, 20, 40]]

    def test_geometry_layer_runs_before_class_layer(self):
        """Confirms the fixed geometry layer is actually wired into the
        combined pipeline, not just callable in isolation."""
        cfg = _FakeConfig()
        cfg.detect_min_bbox_area_px = 5000.0
        boxes = [[0, 0, 5, 5], [0, 0, 100, 100]]
        confs = [0.9, 0.9]
        out_b, out_c, out_i = filter_detections_by_semantic_class(boxes, confs, [0, 0], cfg)
        assert out_b == [[0, 0, 100, 100]]


class TestFilterDetectionsByTargetClass:
    """aim_target_class_ids — a deliberate user multi-select, independent of
    detect_semantic_filter_enabled (this function never reads that flag at
    all — see ai_loop.py, which calls it unconditionally every frame)."""

    def test_empty_selection_is_passthrough(self):
        """The default ([]) means no restriction — every class is a valid
        target, so a single-class model needs no config at all."""
        cfg = _FakeConfig()
        boxes = [[0, 0, 10, 10], [20, 20, 30, 30]]
        confs = [0.7, 0.9]
        cids = [0, 1]
        out_b, out_c, out_i = filter_detections_by_target_class(boxes, confs, cids, cfg)
        assert out_b == boxes and out_c == confs and out_i == cids

    def test_empty_boxes_passthrough(self):
        out_b, out_c, out_i = filter_detections_by_target_class([], [], [], _FakeConfig())
        assert out_b == [] and out_c == [] and out_i == []

    def test_restricts_to_selected_classes(self):
        cfg = _FakeConfig()
        cfg.aim_target_class_ids = [0]  # e.g. "enemy" only, never "teammate"
        boxes = [[0, 0, 10, 10], [20, 20, 30, 30], [40, 40, 50, 50]]
        confs = [0.7, 0.9, 0.5]
        cids = [0, 1, 0]
        out_b, out_c, out_i = filter_detections_by_target_class(boxes, confs, cids, cfg)
        assert out_b == [[0, 0, 10, 10], [40, 40, 50, 50]]
        assert out_c == [0.7, 0.5]
        assert out_i == [0, 0]

    def test_missing_attribute_defaults_to_no_restriction(self):
        """A config predating this field (or a plain stand-in that never
        set it) must behave exactly like an explicit empty list."""
        class _NoAttrConfig:
            pass
        boxes = [[0, 0, 10, 10]]
        out_b, out_c, out_i = filter_detections_by_target_class(boxes, [0.5], [7], _NoAttrConfig())
        assert out_b == boxes
        assert out_i == [7]

    def test_independent_of_semantic_filter_enabled_flag(self):
        """This function has no detect_semantic_filter_enabled gate at all —
        confirm it still restricts classes with that flag left unset/False,
        unlike filter_detections_by_semantic_class()."""
        cfg = _FakeConfig()
        cfg.aim_target_class_ids = [1]
        assert not hasattr(cfg, 'detect_semantic_filter_enabled')
        boxes = [[0, 0, 10, 10], [20, 20, 30, 30]]
        out_b, out_c, out_i = filter_detections_by_target_class(boxes, [0.5, 0.5], [0, 1], cfg)
        assert out_b == [[20, 20, 30, 30]]
        assert out_i == [1]
