"""Opt-in aim math: offsets, predictors, paths, adaptive lead."""

import sys
from unittest.mock import MagicMock

sys.modules.setdefault("win32api", MagicMock())
sys.modules.setdefault("win32con", MagicMock())
sys.modules.setdefault("cv2", MagicMock())

from core.aim_paths import adaptive_lead_s, ema_step, scale_path_step, shape_movement
from win_utils.makcu_protocol import merge_relative_move
from core.ai_aiming import calculate_aim_target
from core.target_predictor import EmaPredictor, RollingVelocityPredictor


class _Cfg:
    aim_x_offset_frac = 0.0
    aim_y_offset_frac = 0.0


def test_offsets_shift_the_aim_point():
    box = [0.0, 0.0, 100.0, 200.0]
    cfg = _Cfg()
    x0, y0 = calculate_aim_target(box, "center", 0.26, cfg)
    cfg.aim_x_offset_frac = 0.10
    cfg.aim_y_offset_frac = 0.05
    x1, y1 = calculate_aim_target(box, "center", 0.26, cfg)
    assert abs((x1 - x0) - 10.0) < 1e-6
    assert abs((y1 - y0) - 10.0) < 1e-6


def test_zero_offsets_match_the_unshifted_point():
    box = [10.0, 20.0, 50.0, 80.0]
    bare = calculate_aim_target(box, "center", 0.26, None)
    same = calculate_aim_target(box, "center", 0.26, _Cfg())
    assert bare == same


def test_ema_predictor_leads_a_constant_velocity():
    pred = EmaPredictor(alpha=1.0)
    pred.update(0, 0, 0.0, 0.1)
    pred.update(10, 0, 0.1, 0.1)
    x, y = pred.update(20, 0, 0.2, 0.1)
    assert x > 20
    assert y == 0


def test_rolling_predictor_uses_frame_lead():
    pred = RollingVelocityPredictor(history_len=5)
    pred.update(0, 0, 0.0, 2)
    x, _y = pred.update(10, 0, 0.1, 2)
    assert abs(x - 30) < 1e-6


def test_linear_path_travels_a_fraction_of_the_error():
    x, y = shape_movement(100, 0, "linear", 0.80, 0.0, 0.0)
    assert abs(x - 20) < 1e-6
    assert y == 0


def test_exponential_path_is_large_enough_to_move():
    x, _y = shape_movement(100, 0, "exponential", 0.80, 0.0, 0.0)
    assert abs(x - 6.4) < 1e-6


def test_fast_ticks_sum_to_one_60hz_step():
    full, _ = shape_movement(100, 0, "linear", 0.80, 0.0, 0.0)
    quarter, _ = scale_path_step(full, 0.0, 1.0 / 240.0)
    assert abs(quarter * 4 - full) < 1e-6


def test_makcu_path_steps_accumulate_and_pid_steps_replace():
    assert merge_relative_move(10, 4, 10, -3, accumulate=True) == (20, 1)
    assert merge_relative_move(10, 4, 10, -3, accumulate=False) == (10, -3)


def test_bezier_with_curve_is_not_the_straight_fraction():
    straight = shape_movement(100, 0, "linear", 0.70, 0.0, 0.0)
    curved = shape_movement(100, 0, "bezier", 0.70, 0.2, 0.0)
    assert curved[0] != straight[0] or curved[1] != straight[1]


def test_adaptive_lead_grows_when_the_mouse_is_slow_and_clamps():
    base = 0.10
    slow = adaptive_lead_s(base, 50.0)
    assert slow > base
    assert adaptive_lead_s(base, 0.0) == base
    assert 0.02 <= adaptive_lead_s(base, 1.5) <= 0.30


def test_ema_step_blends():
    assert ema_step(0.0, 10.0, 0.5) == 5.0
