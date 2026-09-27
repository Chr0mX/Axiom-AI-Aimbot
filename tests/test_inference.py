# tests/test_inference.py
"""
AI 推理模組單元測試

測試範圍：
1. PIDController - 比例/積分/微分計算、動態 Kp 調整、reset
2. preprocess_image - 圖像預處理（BGR/BGRA 轉換、resize、blob）
3. postprocess_outputs - 模型輸出後處理
4. non_max_suppression - NMS 非極大值抑制
"""

import os

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
        result, _, _, _ = preprocess_image(img, 640)
        assert result.shape == (1, 3, 640, 640)
        assert result.dtype == np.float32

    def test_output_shape_bgra(self):
        """BGRA 圖像應自動轉換"""
        from core.inference import preprocess_image
        img = np.random.randint(0, 255, (480, 640, 4), dtype=np.uint8)
        result, _, _, _ = preprocess_image(img, 640)
        assert result.shape == (1, 3, 640, 640)

    def test_output_normalized(self):
        """像素值應歸一化到 [0, 1]"""
        from core.inference import preprocess_image
        img = np.full((640, 640, 3), 255, dtype=np.uint8)
        result, _, _, _ = preprocess_image(img, 640)
        assert result.max() <= 1.0 + 1e-6
        assert result.min() >= 0.0 - 1e-6

    def test_different_model_sizes(self):
        from core.inference import preprocess_image
        img = np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8)
        for size in [320, 416, 640]:
            result, _, _, _ = preprocess_image(img, size)
            assert result.shape == (1, 3, size, size)

    def test_contiguous_memory(self):
        from core.inference import preprocess_image
        img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        result, _, _, _ = preprocess_image(img, 640)
        assert result.flags['C_CONTIGUOUS']

    def test_bgr_to_rgb_channel_order(self):
        """The blob is RGB (swapRB) — a pure-blue BGR pixel lands in channel 2."""
        from core.inference import preprocess_image
        img = np.zeros((640, 640, 4), dtype=np.uint8)
        img[..., 0] = 255  # B in BGRA
        blob, _, _, _ = preprocess_image(img, 640)
        assert blob[0, 2].min() == pytest.approx(1.0)
        assert blob[0, 0].max() == pytest.approx(0.0)

    def test_letterbox_params_for_non_square_frame(self):
        """A 640x320 frame scales by 1.0 and is padded 160px top/bottom."""
        from core.inference import preprocess_image
        img = np.zeros((320, 640, 3), dtype=np.uint8)
        _, scale, pad_x, pad_y = preprocess_image(img, 640)
        assert scale == pytest.approx(1.0)
        assert (pad_x, pad_y) == (0, 160)

    def test_fast_resize_square_scale(self):
        from core.inference import preprocess_image
        img = np.zeros((320, 320, 3), dtype=np.uint8)
        blob, scale, pad_x, pad_y = preprocess_image(img, 640, fast_resize=True)
        assert blob.shape == (1, 3, 640, 640)
        assert scale == pytest.approx(2.0)
        assert (pad_x, pad_y) == (0, 0)


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
        boxes, confs, cids = postprocess_outputs(outputs, 640, 640, 640, 0.5)
        assert boxes == []
        assert confs == []
        assert cids == []

    def test_single_detection(self):
        """One anchor gives a (5, 1) grid — fewer anchors than features, which
        the layout check used to misread as anchors-first and crash on."""
        from core.inference import postprocess_outputs
        # cx=320, cy=320, w=100, h=100, conf=0.9
        outputs = self._make_output([[320, 320, 100, 100, 0.9]])
        boxes, confs, cids = postprocess_outputs(outputs, 640, 640, 640, 0.5)
        assert len(boxes) == 1
        assert len(confs) == 1
        assert abs(confs[0] - 0.9) < 0.01
        assert cids == [0]

    def test_filter_low_confidence(self):
        from core.inference import postprocess_outputs
        outputs = self._make_output([
            [320, 320, 100, 100, 0.9],
            [100, 100, 50, 50, 0.1],  # 低於閾值
        ])
        boxes, confs, _ = postprocess_outputs(outputs, 640, 640, 640, 0.5)
        assert len(boxes) == 1
        assert confs[0] >= 0.5

    def test_offset_applied(self):
        from core.inference import postprocess_outputs
        outputs = self._make_output([[320, 320, 100, 100, 0.9]])
        boxes, _, _ = postprocess_outputs(outputs, 640, 640, 640, 0.5, offset_x=100, offset_y=200)
        # 所有 x 座標 +100, 所有 y 座標 +200
        x1, y1, x2, y2 = boxes[0]
        assert abs(x1 - (270 + 100)) < 1
        assert abs(y1 - (270 + 200)) < 1

    def test_letterbox_reversed(self):
        """Model-space coords are un-padded then divided by the letterbox scale
        (uniform — aspect ratio is preserved, never stretched per axis)."""
        from core.inference import postprocess_outputs
        # A 640x320 capture letterboxed into 640: scale 1.0, pad_y 160.
        outputs = self._make_output([[320, 320, 100, 100, 0.9]])
        boxes, _, _ = postprocess_outputs(
            outputs, 640, 320, 640, 0.5,
            letterbox_scale=1.0, letterbox_pad_x=0, letterbox_pad_y=160,
        )
        x1, y1, x2, y2 = boxes[0]
        assert (x1, y1, x2, y2) == pytest.approx((270, 110, 370, 210))

    def test_scale_factor(self):
        """A 320px capture upscaled 2x to a 640 model maps back at 1/2."""
        from core.inference import postprocess_outputs
        outputs = self._make_output([[320, 320, 100, 100, 0.9]])
        boxes, _, _ = postprocess_outputs(outputs, 320, 320, 640, 0.5, letterbox_scale=2.0)
        x1, y1, x2, y2 = boxes[0]
        assert x2 - x1 == pytest.approx(50)
        assert y2 - y1 == pytest.approx(50)
        assert ((x1 + x2) / 2, (y1 + y2) / 2) == pytest.approx((160, 160))


class TestEnd2EndOutputs:
    """Ultralytics end-to-end exports: [1, max_det, 6] rows of
    [x1, y1, x2, y2, confidence, class_id]."""

    @staticmethod
    def _rows(*rows, max_det=300):
        arr = np.zeros((1, max_det, 6), dtype=np.float32)
        for i, r in enumerate(rows):
            arr[0, i] = r
        return [arr]

    def test_class_id_is_not_read_as_confidence(self):
        """Regression: through the generic path max(conf, class_id) became the
        score, so a class-1 row with conf 0.001 reported 1.0 and passed any
        threshold — a phantom target on every frame."""
        from core.inference import postprocess_outputs
        outputs = self._rows([10, 10, 50, 90, 0.001, 1], [100, 100, 140, 180, 0.95, 0])
        boxes, confs, cids = postprocess_outputs(outputs, 640, 640, 640, 0.8, is_end2end=True)
        assert confs == [pytest.approx(0.95)]
        assert cids == [0]
        assert boxes == [pytest.approx([100, 100, 140, 180])]

    def test_class_ids_and_letterbox_mapping(self):
        from core.inference import postprocess_outputs
        outputs = self._rows([100, 260, 200, 460, 0.9, 1])
        boxes, confs, cids = postprocess_outputs(
            outputs, 640, 320, 640, 0.5, offset_x=10, offset_y=20,
            letterbox_scale=2.0, letterbox_pad_x=0, letterbox_pad_y=160,
            is_end2end=True,
        )
        assert cids == [1]
        assert boxes[0] == pytest.approx([60, 70, 110, 170])

    def test_empty_and_all_below_threshold(self):
        from core.inference import postprocess_outputs
        assert postprocess_outputs(self._rows(), 640, 640, 640, 0.5, is_end2end=True) == ([], [], [])
        outputs = self._rows([0, 0, 10, 10, 0.2, 0])
        assert postprocess_outputs(outputs, 640, 640, 640, 0.5, is_end2end=True) == ([], [], [])


class _FakeSession:
    def __init__(self, meta, out_shape):
        self._meta = meta
        self._out_shape = out_shape

    def get_modelmeta(self):
        class _M:
            pass
        m = _M()
        m.custom_metadata_map = self._meta
        return m

    def get_outputs(self):
        class _O:
            pass
        o = _O()
        o.shape = self._out_shape
        return [o]


class TestDetectEnd2EndOutput:
    def test_metadata_true(self):
        from core.inference import detect_end2end_output
        assert detect_end2end_output(_FakeSession({"end2end": "True"}, [1, 300, 6])) is True

    def test_metadata_false_wins_over_shape(self):
        from core.inference import detect_end2end_output
        assert detect_end2end_output(_FakeSession({"end2end": "False"}, [1, 300, 6])) is False

    def test_shape_fallback(self):
        from core.inference import detect_end2end_output
        assert detect_end2end_output(_FakeSession({}, [1, 300, 6])) is True
        # A raw 2-class grid also has 6 features, but as [1, 6, anchors].
        assert detect_end2end_output(_FakeSession({}, [1, 6, 8400])) is False
        assert detect_end2end_output(_FakeSession({}, [1, 5, 8400])) is False

    def test_dynamic_shape_is_not_end2end(self):
        from core.inference import detect_end2end_output
        assert detect_end2end_output(_FakeSession({}, [1, "N", 6])) is False


_MULTICLASS_E2E_MODEL = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "Model", "Apex_8k_9t2v2_Y26s.onnx",
)


@pytest.mark.skipif(not os.path.isfile(_MULTICLASS_E2E_MODEL), reason="shipped 2-class end2end model not present")
def test_real_multiclass_end2end_model_has_no_phantom_detections():
    """Frames with no real target (max true confidence << 0.8) must produce
    zero detections at the default 0.8 threshold. Before the fix this model
    returned ~30 class-1 phantoms per frame, all reported at confidence 1.0."""
    ort = pytest.importorskip("onnxruntime")
    cv2 = pytest.importorskip("cv2")
    from core.inference import detect_end2end_output, postprocess_outputs, preprocess_image

    sess = ort.InferenceSession(_MULTICLASS_E2E_MODEL, providers=["CPUExecutionProvider"])
    size = int(sess.get_inputs()[0].shape[2])
    assert detect_end2end_output(sess) is True
    rng = np.random.default_rng(1)
    for _ in range(5):
        img = np.full((size, size, 3), rng.integers(0, 255, 3), np.uint8)
        for _ in range(12):
            x, y = (int(v) for v in rng.integers(0, size - 40, 2))
            w, h = (int(v) for v in rng.integers(20, 200, 2))
            cv2.rectangle(img, (x, y), (x + w, y + h), tuple(int(v) for v in rng.integers(0, 255, 3)), -1)
        blob, sc, px, py = preprocess_image(img, size, fast_resize=True)
        out = sess.run(None, {sess.get_inputs()[0].name: blob})
        true_max = float(out[0][0][:, 4].max())
        boxes, confs, _ = postprocess_outputs(out, size, size, size, 0.8, is_end2end=True)
        assert all(c <= true_max + 1e-6 for c in confs)
        if true_max < 0.8:
            assert boxes == []


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
