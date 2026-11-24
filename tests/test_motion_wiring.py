"""process_aiming / config wiring of Auto lead and Strafe response."""
from __future__ import annotations

import math
import sys
import types
from types import SimpleNamespace

import pytest

T0 = 1000.0
FPS = 60.0


@pytest.fixture
def sent_moves(monkeypatch):
    moves = []

    def _send(dx, dy, method=None):
        moves.append((dx, dy, method))

    stub = types.ModuleType("win_utils")
    stub.send_mouse_move = _send
    monkeypatch.setitem(sys.modules, "win_utils", stub)
    import core.ai_aiming as mod
    monkeypatch.setattr(mod, "send_mouse_move", _send)
    for name in ("_predictor", "_kalman"):
        monkeypatch.setattr(mod, name, None)
    return moves


def make_config(**over):
    cfg = SimpleNamespace(
        aim_part="center", head_height_ratio=0.26, aim_custom_y_pct=50.0, aim_adaptive_ratio_enabled=False,
        aim_posture_aware_enabled=False, sticky_lock_enabled=True, lock_persistence_ms=250.0,
        lost_target_timeout_ms=1500.0, reacquire_radius_px=150.0, min_lock_duration_ms=0.0,
        target_switch_threshold=0.0, min_confidence=0.5, tracking_confidence=0.0,
        target_priority_mode="distance", target_priority_confidence_weight=0.5,
        kalman_enabled=False, prediction_enabled=True, prediction_horizon_ms=10.0,
        prediction_max_velocity=5000.0, prediction_history_len=3, prediction_method="auto",
        prediction_adaptive_lead=False, prediction_capture_timestamps=False, cam_motion_comp_enabled=False,
        anchor_enabled=False, anchor_mode="adaptive", anchor_strength=1.0, anchor_velocity_smoothing=0.4,
        anchor_accel_comp=0.35, anchor_confidence_threshold=0.35, anchor_history_len=6,
        anchor_max_distance_pct=100.0, fov_size=800, fov_height=800,
        aim_y_reduce_enabled=False, aim_y_reduce_delay=0.0, aim_y_reduce_settle_px=0.0, aim_y_reduce_floor=0.1,
        aim_y_reduce_ramp=0.0, aim_y_vel_restore_px_s=0.0, humanization=None, max_move_per_frame_px=0.0,
        aim_ema_smoothing_enabled=False, aim_ema_smoothing=0.5, mouse_move_method="sendinput",
        strafe_response=0.0, auto_lead_max_ms=60.0,
    )
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


class Rig:
    """A fixed crosshair, a target on a path, ``frame()`` mimicking ai_loop's per-frame call."""

    def __init__(self, monkeypatch, sent_moves, kp=1.0, ki=0.0, age=0.025, **cfg):
        import core.ai_aiming as mod
        from core.ai_loop_state import LoopState
        from core.inference import PIDController
        self.mod, self.moves = mod, sent_moves
        self.cfg = make_config(**cfg)
        self.state = LoopState()
        self.pid_x, self.pid_y = PIDController(kp, ki, 0), PIDController(kp, ki, 0)
        self.clock = {"t": 0.0}
        self.age = age
        monkeypatch.setattr(mod.time, "perf_counter", lambda: T0 + self.clock["t"])
        self.i = 0

    def _tick(self, t_cap):
        self.clock["t"] = t_cap + self.age
        self.state.frame_capture_t = T0 + t_cap
        self.state.aiming_start_time = T0
        self.moves.clear()

    def frame(self, t_cap, cx, cy=500.0, crosshair=(500, 500)):
        self._tick(t_cap)
        ret = self.mod.process_aiming(
            self.cfg, [[cx - 20, cy - 40, cx + 20, cy + 40]], crosshair[0], crosshair[1], self.pid_x, self.pid_y,
            "sendinput", self.state, T0 + t_cap + self.age, confidences=[0.9])
        return ret, list(self.moves)


def run_moving(rig, frames=40, speed=350.0, x0=300.0):
    last = None
    for i in range(frames):
        t = i / FPS
        last = rig.frame(t, x0 + speed * t)
    return last


# ── Auto lead ────────────────────────────────────────────────────────────────────────────────────────

def test_auto_leads_a_moving_target(sent_moves, monkeypatch):
    rig = Rig(monkeypatch, sent_moves, prediction_method="auto")
    _, moves = run_moving(rig)
    base = Rig(monkeypatch, sent_moves, prediction_enabled=False)
    _, moves0 = run_moving(base)
    assert moves[0][0] > moves0[0][0] + 5          # led further along than the raw point


def test_auto_horizon_follows_the_measured_pipeline_age(sent_moves, monkeypatch):
    dx = {}
    for age in (0.005, 0.025, 0.055):
        rig = Rig(monkeypatch, sent_moves, age=age, prediction_method="auto")
        _, moves = run_moving(rig, frames=60)
        dx[age] = moves[0][0]
    assert dx[0.005] < dx[0.025] < dx[0.055]
    # roughly velocity * (age + 10 ms) more than the raw point
    raw = Rig(monkeypatch, sent_moves, prediction_enabled=False)
    _, m0 = run_moving(raw, frames=60)
    assert dx[0.025] - m0[0][0] == pytest.approx(350.0 * 0.035, rel=0.35)


def test_auto_respects_the_horizon_cap(sent_moves, monkeypatch):
    big = Rig(monkeypatch, sent_moves, age=0.2, prediction_method="auto", auto_lead_max_ms=60.0)
    _, m_big = run_moving(big, frames=60)
    capped = Rig(monkeypatch, sent_moves, age=0.06, prediction_method="auto", auto_lead_max_ms=60.0)
    _, m_cap = run_moving(capped, frames=60)
    assert m_big[0][0] == pytest.approx(m_cap[0][0], rel=0.05)


def test_auto_ignores_the_fixed_horizon_setting(sent_moves, monkeypatch):
    first = run_moving(Rig(monkeypatch, sent_moves, prediction_method="auto", prediction_horizon_ms=5.0))[1][0][0]
    second = run_moving(Rig(monkeypatch, sent_moves, prediction_method="auto", prediction_horizon_ms=50.0))[1][0][0]
    assert first == pytest.approx(second, abs=1.0)


def test_unknown_method_still_falls_back_to_velocity(sent_moves, monkeypatch):
    rig = Rig(monkeypatch, sent_moves, prediction_method="bogus")
    run_moving(rig)
    assert rig.state.aim_anchor.samples == 0       # the aim track is untouched by the velocity fallback


def test_velocity_method_leaves_the_aim_track_alone(sent_moves, monkeypatch):
    rig = Rig(monkeypatch, sent_moves, prediction_method="velocity")
    run_moving(rig)
    assert rig.state.aim_anchor.samples == 0


def test_anchor_reset_clears_the_strafe_marker(sent_moves, monkeypatch):
    from core.ai_loop_utils import anchor_reset
    rig = Rig(monkeypatch, sent_moves)
    rig.state.reversal_seen_t = 5.0
    anchor_reset(rig.cfg, rig.state)
    assert rig.state.reversal_seen_t == 0.0


# ── strafe response (output side) ────────────────────────────────────────────────────────────────────

def _strafe_path(t, speed=350.0, period=1.2):
    u = (t % period) / period
    return 400.0 + (speed * period / 2.0) * (2 * u if u < 0.5 else 2 * (1 - u))


def test_reversal_resets_the_pid_integral_once_and_bypasses_step_smoothing(sent_moves, monkeypatch):
    rig = Rig(monkeypatch, sent_moves, ki=0.05, prediction_method="auto", strafe_response=0.8,
              aim_ema_smoothing_enabled=True, aim_ema_smoothing=0.3)
    seen = []
    orig = rig.pid_x.reset_integral

    def spy():
        seen.append(rig.clock["t"])
        orig()

    rig.pid_x.reset_integral = spy
    for i in range(int(1.6 * FPS)):
        t = i / FPS
        rig.frame(t, _strafe_path(t))
    assert 1 <= len(seen) <= 3                      # the turn at t = 0.6 s (and maybe a second at 1.2 s), once each
    assert rig.state.reversal_seen_t > 0.0


def test_strafe_response_off_never_touches_the_output_stages(sent_moves, monkeypatch):
    rig = Rig(monkeypatch, sent_moves, ki=0.05, prediction_method="auto", strafe_response=0.0)
    seen = []
    rig.pid_x.reset_integral = lambda: seen.append(1)
    for i in range(int(1.6 * FPS)):
        t = i / FPS
        rig.frame(t, _strafe_path(t))
    assert seen == [] and rig.state.reversal_seen_t == 0.0


def test_reversal_resets_the_kalman_filter(sent_moves, monkeypatch):
    rig = Rig(monkeypatch, sent_moves, prediction_method="auto", strafe_response=0.8, kalman_enabled=True)
    resets = []
    for i in range(int(1.6 * FPS)):
        t = i / FPS
        k = rig.mod._kalman
        if k is not None and not getattr(k, "_spied", False):
            o = k.reset
            k.reset = lambda o=o: (resets.append(rig.clock["t"]), o())[1]
            k._spied = True
        rig.frame(t, _strafe_path(t))
    assert any(0.6 <= r <= 0.9 for r in resets)


# ── config ───────────────────────────────────────────────────────────────────────────────────────────

def _config():
    from unittest.mock import patch
    with patch("core.config._get_screen_size", return_value=(1920, 1080)):
        from core.config import Config
        return Config()


def test_defaults_keep_everything_off():
    c = _config()
    assert (c.strafe_response, c.prediction_method) == (0.0, "velocity")
    assert c.auto_lead_max_ms == 60.0


def test_new_fields_persist_and_are_preset_eligible():
    from core.config import _FIELD_MAP
    from core.config_manager import _AIM_PRESET_FIELDS
    for k in ("strafe_response", "auto_lead_max_ms"):
        assert k in _FIELD_MAP and k in _AIM_PRESET_FIELDS
        assert _FIELD_MAP[k].startswith("tracking.")


def test_round_trip_and_old_files_load():
    c = _config()
    c.prediction_method = "auto"
    c.strafe_response = 0.7
    c.auto_lead_max_ms = 45.0
    d = c.to_dict()
    c2 = _config()
    c2.from_dict(d)
    assert (c2.prediction_method, c2.strafe_response, c2.auto_lead_max_ms) == ("auto", 0.7, 45.0)
    old = _config()
    old.from_dict({"config_version": 3, "tracking": {"prediction": {"method": "anchor"}}})
    assert old.strafe_response == 0.0


def test_validators_clamp_and_reject_garbage():
    from core.config import _validate_aim_math
    c = _config()
    c.strafe_response, c.auto_lead_max_ms = 9.0, 5.0
    _validate_aim_math(c)
    assert (c.strafe_response, c.auto_lead_max_ms) == (1.0, 20.0)
    c.strafe_response, c.auto_lead_max_ms = float("nan"), "x"
    _validate_aim_math(c)
    assert (c.strafe_response, c.auto_lead_max_ms) == (0.0, 60.0)
    c.prediction_method = "auto"
    _validate_aim_math(c)
    assert c.prediction_method == "auto"
    c.prediction_method = "rolling"
    _validate_aim_math(c)
    assert c.prediction_method == "velocity"


def test_all_finite_outputs(sent_moves, monkeypatch):
    rig = Rig(monkeypatch, sent_moves, strafe_response=1.0, prediction_method="auto")
    for i in range(120):
        t = i / FPS
        rig.frame(t, _strafe_path(t, period=0.5))
        for dx, dy, _ in rig.moves:
            assert math.isfinite(dx) and math.isfinite(dy)
