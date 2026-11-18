"""Advanced Aim Tuning: offsets, predictors, output EMA, adaptive lead."""

import sys
from unittest.mock import MagicMock

sys.modules.setdefault("win32api", MagicMock())
sys.modules.setdefault("win32con", MagicMock())
sys.modules.setdefault("cv2", MagicMock())

from core.aim_paths import adaptive_lead_s, ema_step
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





def test_makcu_path_steps_accumulate_and_pid_steps_replace():
    assert merge_relative_move(10, 4, 10, -3, accumulate=True) == (20, 1)
    assert merge_relative_move(10, 4, 10, -3, accumulate=False) == (10, -3)



def test_adaptive_lead_grows_when_the_mouse_is_slow_and_clamps():
    base = 0.10
    slow = adaptive_lead_s(base, 50.0)
    assert slow > base
    assert adaptive_lead_s(base, 0.0) == base
    assert 0.02 <= adaptive_lead_s(base, 1.5) <= 0.30


def test_ema_step_blends():
    assert ema_step(0.0, 10.0, 0.5) == 5.0


# --- Velocity-estimate refinements (all opt-in) ---

from core.target_predictor import VelocityPredictor


def _feed(pred, points, horizon=0.1):
    out = None
    for x, t in points:
        out = pred.update(x, 0.0, t, horizon)
    return out


def test_endpoint_default_unchanged():
    pred = VelocityPredictor(history_len=3)
    x, _ = _feed(pred, [(0, 0.0), (10, 0.1), (20, 0.2)])
    assert abs(x - 30.0) < 1e-6  # 100 px/s * 0.1 s lead


def test_lsq_fit_is_steadier_than_endpoints_on_a_jittery_middle_point():
    pts = [(0, 0.0), (20, 0.1), (20, 0.2)]  # endpoints say 100 px/s
    plain = VelocityPredictor(history_len=3)
    lsq = VelocityPredictor(history_len=3)
    lsq.lsq_fit = True
    assert abs(_feed(plain, pts)[0] - 30.0) < 1e-6
    assert abs(_feed(lsq, pts)[0] - 30.0) < 1e-6  # LSQ slope is also 100 here
    pts2 = [(0, 0.0), (0, 0.1), (20, 0.2)]  # one late jump
    lsq.reset()
    assert _feed(lsq, pts2)[0] < 20 + 100 * 0.1 + 1e-6


def test_lsq_fit_handles_identical_timestamps():
    pred = VelocityPredictor(history_len=3)
    pred.lsq_fit = True
    assert _feed(pred, [(5, 1.0), (6, 1.0)]) == (6, 0.0)


def test_velocity_deadzone_zeroes_slow_and_reduces_fast_motion():
    slow = VelocityPredictor(history_len=2)
    slow.velocity_deadzone = 50.0
    assert _feed(slow, [(0, 0.0), (2, 0.1)]) == (2, 0.0)  # 20 px/s -> no lead
    fast = VelocityPredictor(history_len=2)
    fast.velocity_deadzone = 50.0
    x, _ = _feed(fast, [(0, 0.0), (20, 0.1)])  # 200 px/s -> 150 px/s kept
    assert abs(x - (20 + 150 * 0.1)) < 1e-6


def test_ego_moves_only_count_once_their_frame_shows_them():
    from core.ai_aiming import settle_ego_moves
    from core.ai_loop_state import LoopState
    state = LoopState()
    state.ego_pending.append((1.0, 5.0, -2.0))
    state.ego_pending.append((2.0, 3.0, 0.0))
    settle_ego_moves(state, 1.5)
    assert (state.ego_drift_x, state.ego_drift_y) == (5.0, -2.0)
    assert len(state.ego_pending) == 1
    settle_ego_moves(state, 2.0)
    assert state.ego_drift_x == 8.0 and not state.ego_pending


def test_new_prediction_options_are_persisted_and_clamped():
    from types import SimpleNamespace
    from core.config import _FIELD_MAP, _validate_aim_math
    for key in ("prediction_capture_timestamps", "prediction_lsq_fit",
                "prediction_velocity_deadzone_px_s", "prediction_ego_motion_comp",
                "prediction_ego_motion_scale"):
        assert _FIELD_MAP[key].startswith("tracking.prediction.")
    cfg = SimpleNamespace(prediction_velocity_deadzone_px_s=9999, prediction_ego_motion_scale=0.01)
    _validate_aim_math(cfg)
    assert cfg.prediction_velocity_deadzone_px_s == 500.0
    assert cfg.prediction_ego_motion_scale == 0.1
