# tests/test_smart_tracker.py
"""
SmartTracker 智慧追蹤器單元測試

測試範圍：
1. 初始化與 reset
2. update - 平滑、變向偵測、靜止歸零
3. is_in_deadzone - 位置死區判斷
4. get_corrected_move - 修正移動量
5. get_predicted_position - 預測未來位置
"""

import math
import numpy as np
import pytest


def _make_tracker(**kwargs):
    from core.smart_tracker import SmartTracker
    return SmartTracker(**kwargs)


# ============================================================
# 1. 初始化與 Reset 測試
# ============================================================

class TestSmartTrackerInit:
    """測試追蹤器初始化"""

    def test_default_params(self):
        t = _make_tracker()
        assert t.alpha == 0.5
        assert t.stop_threshold == 20.0
        assert t.position_deadzone == 4.0

    def test_custom_params(self):
        t = _make_tracker(smoothing_factor=0.8, stop_threshold=5.0, position_deadzone=2.0)
        assert t.alpha == 0.8
        assert t.stop_threshold == 5.0
        assert t.position_deadzone == 2.0

    def test_initial_state_uninitialized(self):
        t = _make_tracker()
        assert t.initialized is False
        assert t.last_x is None
        assert t.last_y is None
        assert t.vx == 0.0
        assert t.vy == 0.0

    def test_reset(self):
        t = _make_tracker()
        t.update(100, 200, 0.016)
        t.update(110, 210, 0.016)
        t.reset()
        assert t.initialized is False
        assert t.last_x is None
        assert t.last_y is None
        assert t.vx == 0.0
        assert t.vy == 0.0


# ============================================================
# 2. Update 測試
# ============================================================

class TestSmartTrackerUpdate:
    """測試追蹤器更新"""

    def test_first_update_initializes(self):
        t = _make_tracker()
        x, y, vx, vy = t.update(100, 200, 0.016)
        assert t.initialized is True
        assert x == 100
        assert y == 200
        assert vx == 0.0
        assert vy == 0.0

    def test_second_update_computes_velocity(self):
        t = _make_tracker(smoothing_factor=0.0, stop_threshold=0.0)
        t.update(100, 200, 0.016)
        x, y, vx, vy = t.update(110, 200, 0.016)
        # raw_vx = (110-100) / 0.016 = 625
        # alpha=0.0: vx = 0 * 0.0 + 625 * 1.0 = 625
        assert x == 110
        assert abs(vx - 625.0) < 1.0

    def test_dt_zero_resets(self):
        t = _make_tracker()
        t.update(100, 200, 0.016)
        x, y, vx, vy = t.update(110, 210, 0)  # dt <= 0
        assert vx == 0.0
        assert vy == 0.0

    def test_smoothing_reduces_jitter(self):
        """高 alpha 值應該減少速度抖動"""
        t = _make_tracker(smoothing_factor=0.9, stop_threshold=0.0)
        t.update(100, 100, 0.016)
        t.update(110, 100, 0.016)  # vx 應較大
        _, _, vx1, _ = t.update(111, 100, 0.016)  # 微小移動

        t2 = _make_tracker(smoothing_factor=0.0, stop_threshold=0.0)
        t2.update(100, 100, 0.016)
        t2.update(110, 100, 0.016)
        _, _, vx2, _ = t2.update(111, 100, 0.016)

        # 高 alpha 的速度變化更平滑（保留更多舊速度）
        # 無法直接比較大小，但可以驗證兩者不同
        # 高 alpha 時 vx 會更接近前一幀的速度
        assert vx1 != vx2

    def test_direction_change_resets_velocity(self):
        """方向變化應重置速度（不平滑）"""
        t = _make_tracker(smoothing_factor=0.9, stop_threshold=0.0)
        t.update(100, 100, 0.016)
        t.update(110, 100, 0.016)  # 向右移動
        _, _, vx, _ = t.update(105, 100, 0.016)  # 突然向左
        # dot_product < 0，速度應直接採用新值
        assert vx < 0  # 速度方向改變

    def test_stop_threshold_zeroes_small_velocity(self):
        """低速歸零"""
        t = _make_tracker(smoothing_factor=0.0, stop_threshold=100.0)
        t.update(100, 100, 0.016)
        _, _, vx, vy = t.update(100.5, 100.5, 0.016)
        # raw_vx = 0.5 / 0.016 = 31.25 < threshold(100)
        assert vx == 0
        assert vy == 0


# ============================================================
# 3. is_in_deadzone 測試
# ============================================================

class TestSmartTrackerDeadzone:
    """測試位置死區"""

    def test_within_deadzone(self):
        t = _make_tracker(position_deadzone=10.0)
        assert t.is_in_deadzone(102, 102, 100, 100) == True

    def test_outside_deadzone(self):
        t = _make_tracker(position_deadzone=2.0)
        assert t.is_in_deadzone(110, 110, 100, 100) == False

    def test_exact_boundary(self):
        t = _make_tracker(position_deadzone=5.0)
        # 距離 = 5.0，等於 deadzone
        result = t.is_in_deadzone(103, 104, 100, 100)
        # distance = sqrt(9+16) = 5.0, 不小於 5.0
        assert result == False

    def test_zero_deadzone(self):
        t = _make_tracker(position_deadzone=0.0)
        assert t.is_in_deadzone(100, 100, 100, 100) == False

    def test_same_position(self):
        t = _make_tracker(position_deadzone=5.0)
        assert t.is_in_deadzone(100, 100, 100, 100) == True


# ============================================================
# 4. get_corrected_move 測試
# ============================================================

class TestSmartTrackerCorrectedMove:
    """測試修正移動量"""

    def test_normal_correction(self):
        t = _make_tracker(position_deadzone=2.0)
        dx, dy = t.get_corrected_move(110, 120, 100, 100)
        assert dx == 10
        assert dy == 20

    def test_in_deadzone_returns_zero(self):
        t = _make_tracker(position_deadzone=10.0)
        dx, dy = t.get_corrected_move(101, 101, 100, 100)
        assert dx == 0.0
        assert dy == 0.0

    def test_exactly_at_target(self):
        t = _make_tracker(position_deadzone=5.0)
        dx, dy = t.get_corrected_move(100, 100, 100, 100)
        assert dx == 0.0
        assert dy == 0.0


# ============================================================
# 5. get_predicted_position 測試
# ============================================================

class TestSmartTrackerPrediction:
    """測試位置預測"""

    def test_not_initialized_returns_zero(self):
        t = _make_tracker()
        x, y = t.get_predicted_position(0.1)
        assert x == 0.0
        assert y == 0.0

    def test_stationary_prediction(self):
        """靜止物件預測位置不變"""
        t = _make_tracker(stop_threshold=0.0)
        t.update(100, 200, 0.016)
        x, y = t.get_predicted_position(0.1)
        # 只初始化一幀，vx=vy=0
        assert x == 100
        assert y == 200

    def test_moving_prediction(self):
        """移動物件預測未來位置"""
        t = _make_tracker(smoothing_factor=0.0, stop_threshold=0.0)
        t.update(100, 100, 0.016)
        t.update(200, 100, 1.0)  # vx = 100/1.0 = 100 px/s
        x, y = t.get_predicted_position(0.5)
        # pred_x = 200 + 100 * 0.5 = 250
        assert abs(x - 250.0) < 1.0
        assert abs(y - 100.0) < 1.0
