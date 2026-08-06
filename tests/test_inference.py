# tests/test_inference.py
"""
AI 推理模組單元測試

測試範圍：
1. PIDController - 比例/積分/微分計算、動態 Kp 調整、reset
2. preprocess_image - 圖像預處理（BGR/BGRA 轉換、resize、blob）
3. postprocess_outputs - 模型輸出後處理
4. non_max_suppression - NMS 非極大值抑制
"""

import numpy as np
import pytest


# ============================================================
# 1. PIDController 測試
# ============================================================

class TestPIDController:
    """測試 PID 控制器"""

    def _make_pid(self, kp=0.5, ki=0.0, kd=0.0):
        from core.inference import PIDController
        return PIDController(kp, ki, kd)

    def test_initial_state(self):
        pid = self._make_pid()
        assert pid.Kp == 0.5
        assert pid.Ki == 0.0
        assert pid.Kd == 0.0
        assert pid.integral == 0.0
        assert pid.previous_error == 0.0

    def test_proportional_only(self):
        """純比例控制：output = Kp * error"""
        pid = self._make_pid(kp=0.5, ki=0.0, kd=0.0)
        output = pid.update(10.0)
        assert output == 0.5 * 10.0  # kp <= 0.5 時不調整

    def test_proportional_high_kp_is_linear(self):
        """Kp response is a clamped identity — no non-linear gain explosion above 0.5."""
        pid = self._make_pid(kp=1.0, ki=0.0, kd=0.0)
        output = pid.update(10.0)
        # kp=1.0 -> effective 1.0 (was 2.0 under the old triple-gain curve)
        assert output == 1.0 * 10.0
        # And a mid-upper value is linear too: kp=0.75 -> 0.75 (was 1.25)
        pid2 = self._make_pid(kp=0.75, ki=0.0, kd=0.0)
        assert pid2.update(10.0) == 0.75 * 10.0
        # Out-of-range is clamped to [0, 1]
        pid3 = self._make_pid(kp=1.5, ki=0.0, kd=0.0)
        assert pid3.update(10.0) == 1.0 * 10.0

    def test_integral_accumulation(self):
        """積分項累積測試"""
        pid = self._make_pid(kp=0.0, ki=0.1, kd=0.0)
        pid.update(10.0)  # integral = 10
        output = pid.update(10.0)  # integral = 20
        assert output == 0.1 * 20.0

    def test_derivative_response(self):
        """微分項回應變化測試"""
        pid = self._make_pid(kp=0.0, ki=0.0, kd=0.5)
        pid.update(0.0)   # previous_error = 0
        output = pid.update(10.0)  # derivative = 10 - 0 = 10
        assert output == 0.5 * 10.0

    def test_combined_pid(self):
        """PID 三項合併"""
        pid = self._make_pid(kp=0.3, ki=0.1, kd=0.2)
        output = pid.update(10.0)
        # kp=0.3 <= 0.5, adjusted_kp=0.3
        # P = 0.3 * 10 = 3.0
        # I = 0.1 * 10 = 1.0 (integral = 10)
        # D = 0.2 * (10 - 0) = 2.0
        assert abs(output - 6.0) < 0.001

    def test_reset(self):
        pid = self._make_pid(kp=0.5, ki=0.1, kd=0.1)
        pid.update(10.0)
        pid.update(20.0)
        pid.reset()
        assert pid.integral == 0.0
        assert pid.previous_error == 0.0

    def test_zero_error(self):
        pid = self._make_pid(kp=0.5, ki=0.0, kd=0.0)
        output = pid.update(0.0)
        assert output == 0.0

    def test_negative_error(self):
        pid = self._make_pid(kp=0.5, ki=0.0, kd=0.0)
        output = pid.update(-10.0)
        assert output == 0.5 * (-10.0)

    def test_adjusted_kp_boundary_050(self):
        """kp = 0.5 不調整"""
        pid = self._make_pid(kp=0.5)
        adjusted = pid._calculate_adjusted_kp(0.5)
        assert adjusted == 0.5

    def test_adjusted_kp_at_075(self):
        """kp = 0.75 -> 0.75 (linear identity; no triple-gain above 0.5)"""
        pid = self._make_pid()
        adjusted = pid._calculate_adjusted_kp(0.75)
        assert abs(adjusted - 0.75) < 0.001

    def test_adjusted_kp_clamped(self):
        """Out-of-range kp is clamped to [0, 1]."""
        pid = self._make_pid()
        assert pid._calculate_adjusted_kp(1.5) == 1.0
        assert pid._calculate_adjusted_kp(-0.2) == 0.0

    def test_adjusted_kp_below_050(self):
        """kp < 0.5 維持原值"""
        pid = self._make_pid()
        assert pid._calculate_adjusted_kp(0.0) == 0.0
        assert pid._calculate_adjusted_kp(0.3) == 0.3


# ============================================================
# 2. preprocess_image 測試
# ============================================================

class TestPreprocessImage:
    """測試圖像預處理"""

    def test_output_shape_bgr(self):
        from core.inference import preprocess_image
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = preprocess_image(img, 640)
        assert result.shape == (1, 3, 640, 640)
        assert result.dtype == np.float32

    def test_output_shape_bgra(self):
        """BGRA 圖像應自動轉換"""
        from core.inference import preprocess_image
        img = np.random.randint(0, 255, (480, 640, 4), dtype=np.uint8)
        result = preprocess_image(img, 640)
        assert result.shape == (1, 3, 640, 640)

    def test_output_normalized(self):
        """像素值應歸一化到 [0, 1]"""
        from core.inference import preprocess_image
        img = np.full((640, 640, 3), 255, dtype=np.uint8)
        result = preprocess_image(img, 640)
        assert result.max() <= 1.0 + 1e-6
        assert result.min() >= 0.0 - 1e-6

    def test_different_model_sizes(self):
        from core.inference import preprocess_image
        img = np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8)
        for size in [320, 416, 640]:
            result = preprocess_image(img, size)
            assert result.shape == (1, 3, size, size)

    def test_contiguous_memory(self):
        from core.inference import preprocess_image
        img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        result = preprocess_image(img, 640)
        assert result.flags['C_CONTIGUOUS']


# ============================================================
# 3. postprocess_outputs 測試
# ============================================================

class TestPostprocessOutputs:
    """測試模型輸出後處理"""

    def _make_output(self, detections):
        """
        製作模擬的模型輸出
        detections: list of (cx, cy, w, h, conf)
        """
        if len(detections) == 0:
            arr = np.zeros((1, 5, 0), dtype=np.float32)
        else:
            arr = np.array(detections, dtype=np.float32).T  # shape: (5, N)
            arr = arr.reshape(1, 5, -1)  # shape: (1, 5, N)
        return [arr]

    def test_no_detections(self):
        from core.inference import postprocess_outputs
        outputs = self._make_output([])
        boxes, confs = postprocess_outputs(outputs, 640, 640, 640, 0.5)
        assert boxes == []
        assert confs == []

    def test_single_detection(self):
        from core.inference import postprocess_outputs
        # cx=320, cy=320, w=100, h=100, conf=0.9
        outputs = self._make_output([[320, 320, 100, 100, 0.9]])
        boxes, confs = postprocess_outputs(outputs, 640, 640, 640, 0.5)
        assert len(boxes) == 1
        assert len(confs) == 1
        assert abs(confs[0] - 0.9) < 0.01


    def test_filter_low_confidence(self):
        from core.inference import postprocess_outputs
        outputs = self._make_output([
            [320, 320, 100, 100, 0.9],
            [100, 100, 50, 50, 0.1],  # 低於閾值
        ])
        boxes, confs = postprocess_outputs(outputs, 640, 640, 640, 0.5)
        assert len(boxes) == 1
        assert confs[0] >= 0.5

    def test_offset_applied(self):
        from core.inference import postprocess_outputs
        outputs = self._make_output([[320, 320, 100, 100, 0.9]])
        boxes, _ = postprocess_outputs(outputs, 640, 640, 640, 0.5, offset_x=100, offset_y=200)
        # 所有 x 座標 +100, 所有 y 座標 +200
        x1, y1, x2, y2 = boxes[0]
        assert abs(x1 - (270 + 100)) < 1
        assert abs(y1 - (270 + 200)) < 1

    def test_scale_factor(self):
        """當 original_size != model_input_size 時應正確縮放"""
        from core.inference import postprocess_outputs
        # model_input=640, original=1280x720
        outputs = self._make_output([[320, 320, 100, 100, 0.9]])
        boxes, _ = postprocess_outputs(outputs, 1280, 720, 640, 0.5)
        x1, y1, x2, y2 = boxes[0]
        # scale_x = 1280/640 = 2.0, scale_y = 720/640 = 1.125
        assert abs(x2 - x1 - 100 * 2.0) < 1
        assert abs(y2 - y1 - 100 * 1.125) < 1


# ============================================================
# 4. non_max_suppression 測試
# ============================================================

class TestNonMaxSuppression:
    """測試 NMS 非極大值抑制"""

    def test_empty_input(self):
        from core.inference import non_max_suppression
        boxes, confs, cids = non_max_suppression([], [])
        assert boxes == []
        assert confs == []
        assert cids == []

    def test_single_box(self):
        from core.inference import non_max_suppression
        boxes_in = [[10, 10, 50, 50]]
        confs_in = [0.9]
        boxes, confs, cids = non_max_suppression(boxes_in, confs_in)
        assert len(boxes) == 1
        assert cids == [0]

    def test_non_overlapping_boxes_kept(self):
        from core.inference import non_max_suppression
        boxes_in = [[10, 10, 50, 50], [200, 200, 250, 250]]
        confs_in = [0.9, 0.8]
        boxes, confs, cids = non_max_suppression(boxes_in, confs_in)
        assert len(boxes) == 2
        assert len(cids) == 2

    def test_overlapping_boxes_suppressed(self):
        from core.inference import non_max_suppression
        # 兩個幾乎完全重疊的框
        boxes_in = [[10, 10, 50, 50], [11, 11, 51, 51]]
        confs_in = [0.9, 0.8]
        boxes, confs, cids = non_max_suppression(boxes_in, confs_in, iou_threshold=0.4)
        assert len(boxes) == 1
        assert confs[0] == 0.9  # 高置信度的保留

    def test_higher_iou_threshold_keeps_more(self):
        from core.inference import non_max_suppression
        boxes_in = [[10, 10, 50, 50], [15, 15, 55, 55]]
        confs_in = [0.9, 0.85]
        # 使用高 IoU 閾值
        boxes, confs, cids = non_max_suppression(boxes_in, confs_in, iou_threshold=0.99)
        assert len(boxes) == 2  # 高閾值，兩個都保留

    def test_keeps_highest_confidence(self):
        from core.inference import non_max_suppression
        boxes_in = [[10, 10, 50, 50], [10, 10, 50, 50], [10, 10, 50, 50]]
        confs_in = [0.5, 0.9, 0.3]
        boxes, confs, cids = non_max_suppression(boxes_in, confs_in, iou_threshold=0.4)
        assert len(boxes) == 1
        assert confs[0] == 0.9

    def test_class_ids_follow_reordered_survivors(self):
        """NMS sorts survivors by descending confidence, so class_ids must be
        reindexed by the same order — not left in input order. Regression for
        the misalignment that made the semantic class-name filter read the
        class of an unrelated detection."""
        from core.inference import non_max_suppression
        # Three well-separated boxes, deliberately NOT in confidence order.
        boxes_in = [[0, 0, 20, 20], [200, 200, 220, 220], [400, 400, 420, 420]]
        confs_in = [0.3, 0.9, 0.6]
        cids_in = [7, 1, 4]  # class id travels with its own box

        boxes, confs, cids = non_max_suppression(
            boxes_in, confs_in, iou_threshold=0.4, class_ids=cids_in,
        )

        assert len(boxes) == 3  # nothing overlaps, all survive
        # Output is confidence-descending: 0.9 (box1/cls1), 0.6 (box2/cls4), 0.3 (box0/cls7)
        assert confs == [0.9, 0.6, 0.3]
        assert cids == [1, 4, 7]
        # And every triple still describes one real detection.
        for box, conf, cid in zip(boxes, confs, cids):
            i = boxes_in.index(box)
            assert confs_in[i] == conf
            assert cids_in[i] == cid

    def test_class_ids_dropped_with_suppressed_boxes(self):
        """A suppressed box must take its class id with it."""
        from core.inference import non_max_suppression
        boxes_in = [[10, 10, 50, 50], [11, 11, 51, 51], [400, 400, 420, 420]]
        confs_in = [0.5, 0.9, 0.7]
        cids_in = [2, 3, 5]

        boxes, confs, cids = non_max_suppression(
            boxes_in, confs_in, iou_threshold=0.4, class_ids=cids_in,
        )

        # Boxes 0/1 overlap → only the 0.9 one (class 3) survives, plus box 2 (class 5).
        assert len(boxes) == 2
        assert confs == [0.9, 0.7]
        assert cids == [3, 5]

    def test_class_ids_default_to_zero_when_absent(self):
        from core.inference import non_max_suppression
        boxes_in = [[0, 0, 20, 20], [200, 200, 220, 220]]
        confs_in = [0.9, 0.8]
        boxes, confs, cids = non_max_suppression(boxes_in, confs_in)
        assert cids == [0, 0]
        assert len(cids) == len(boxes)


# ============================================================
# PID time normalization (Phase 4)
# ============================================================

class TestPIDTimeNormalization:
    """The I/D terms used to depend on how fast the loop happened to run —
    and changed underfoot at runtime, since idle_detect_interval slows the
    loop while not aiming. Normalizing against REFERENCE_DT fixes the rate
    dependence while leaving behaviour at the reference rate untouched, so
    existing tunings keep their meaning."""

    def test_reference_dt_matches_legacy_fixed_step(self):
        from core.inference import PIDController
        legacy = PIDController(0.5, 0.1, 0.2)
        timed = PIDController(0.5, 0.1, 0.2)
        for err in (10.0, 8.0, 5.0, 5.0, -3.0):
            a = legacy.update(err)                                  # no dt = legacy path
            b = timed.update(err, PIDController.REFERENCE_DT)       # dt == reference
            assert a == pytest.approx(b), "reference dt must reproduce legacy output exactly"

    def test_none_dt_is_legacy_behaviour(self):
        from core.inference import PIDController
        p = PIDController(0.4, 0.05, 0.1)
        p.update(10.0, None)
        assert p.integral == pytest.approx(10.0)  # scale 1.0, not scaled by seconds

    def test_integral_accumulates_with_elapsed_time_not_call_count(self):
        """Two half-length steps must integrate the same as one full step —
        that equivalence is the entire point."""
        from core.inference import PIDController
        ref = PIDController.REFERENCE_DT
        one_step = PIDController(0.0, 1.0, 0.0)
        two_steps = PIDController(0.0, 1.0, 0.0)

        one_step.update(10.0, ref)
        two_steps.update(10.0, ref / 2)
        two_steps.update(10.0, ref / 2)

        assert one_step.integral == pytest.approx(two_steps.integral)

    def test_derivative_is_a_rate_not_a_difference(self):
        from core.inference import PIDController
        ref = PIDController.REFERENCE_DT
        slow = PIDController(0.0, 0.0, 1.0)
        fast = PIDController(0.0, 0.0, 1.0)

        slow.update(0.0, ref)
        fast.update(0.0, ref)
        # Same error change, but fast covers it in half the time -> twice the rate.
        out_slow = slow.update(10.0, ref)
        out_fast = fast.update(10.0, ref / 2)
        assert out_fast == pytest.approx(out_slow * 2)

    def test_extreme_dt_is_clamped(self):
        """A stalled frame (GC pause, model hot-swap) must not inject a huge
        integral step or a near-zero derivative."""
        from core.inference import PIDController
        ref = PIDController.REFERENCE_DT
        p = PIDController(0.0, 1.0, 0.0)
        p.update(10.0, ref * 1000)      # 10 s stall
        assert p.integral == pytest.approx(10.0 * 4.0)  # clamped at 4x, not 1000x


# ============================================================
# Model output-format overrides (Phase 4)
# ============================================================

class TestModelOutputFormat:
    def _layout_a_cxcywh(self, n=1):
        # anchors x features, [cx, cy, w, h, cls0, cls1] — no objectness.
        # cx small and w large so the naive xyxy test (x2>x1, y2>y1) is
        # satisfied, which is exactly the coincidence that used to corrupt
        # boxes.
        return np.array([[10.0, 10.0, 100.0, 120.0, 0.9, 0.1]] * n, dtype=np.float32)

    def test_auto_does_not_guess_xyxy_from_too_few_detections(self):
        """One detection satisfying x2>x1 and y2>y1 is close to a coin flip —
        cxcywh is far more common, so 'auto' must not flip on it."""
        from core.inference import postprocess_outputs
        raw = self._layout_a_cxcywh(1)
        boxes, confs, cids = postprocess_outputs([raw[None, ...]], 640, 640, 640, 0.5)
        # cxcywh reading: cx=10,w=100 -> x1 = 10 - 50 = -40
        assert boxes[0][0] == pytest.approx(-40.0)

    def test_explicit_cxcywh_overrides_auto(self):
        from core.inference import postprocess_outputs
        raw = self._layout_a_cxcywh(5)  # enough rows that auto WOULD infer xyxy
        boxes, _, _ = postprocess_outputs(
            [raw[None, ...]], 640, 640, 640, 0.5, box_format='cxcywh')
        assert boxes[0][0] == pytest.approx(-40.0)

    def test_explicit_xyxy_is_honoured(self):
        from core.inference import postprocess_outputs
        raw = self._layout_a_cxcywh(1)
        boxes, _, _ = postprocess_outputs(
            [raw[None, ...]], 640, 640, 640, 0.5, box_format='xyxy')
        # xyxy reading: x1=10, x2=100 -> centre 55, width 90 -> x1 back to 10
        assert boxes[0][0] == pytest.approx(10.0)

    def test_objectness_yes_multiplies_and_shifts_class_ids(self):
        """YOLOv5 layout: [cx,cy,w,h,obj,cls0,cls1]. Confidence is obj*cls,
        and class ids must not count objectness as class 0."""
        from core.inference import postprocess_outputs
        raw = np.array([[100.0, 100.0, 20.0, 20.0, 0.8, 0.1, 0.9]], dtype=np.float32)
        boxes, confs, cids = postprocess_outputs(
            [raw[None, ...]], 640, 640, 640, 0.5,
            box_format='cxcywh', has_objectness='yes')
        assert confs[0] == pytest.approx(0.8 * 0.9)
        assert cids[0] == 1   # cls1 wins; objectness is not a class

    def test_objectness_no_uses_class_max(self):
        from core.inference import postprocess_outputs
        raw = np.array([[100.0, 100.0, 20.0, 20.0, 0.6, 0.9]], dtype=np.float32)
        boxes, confs, cids = postprocess_outputs(
            [raw[None, ...]], 640, 640, 640, 0.5,
            box_format='cxcywh', has_objectness='no')
        assert confs[0] == pytest.approx(0.9)
        assert cids[0] == 1
