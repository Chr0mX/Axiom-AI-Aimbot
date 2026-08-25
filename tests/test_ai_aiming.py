# tests/test_ai_aiming.py
"""Regression tests for src/core/ai_aiming.py's process_aiming() fixes.

ai_aiming.py imports `win_utils` at module level (`from win_utils import
send_mouse_move`), and win_utils/__init__.py transitively imports win32api
via mouse_move.py — this fails at *collection* on a non-Windows box (see
CLAUDE.md). Every test here defers the `core.ai_aiming` import to inside the
test function, after stubbing sys.modules['win_utils'], matching the pattern
already used in test_makcu_mouse.py/test_screen_capture.py for other
win_utils-dependent modules.
"""

import sys
import types
from types import SimpleNamespace

import pytest


@pytest.fixture
def sent_moves(monkeypatch):
    """Stub win_utils.send_mouse_move and capture every move sent by
    process_aiming() during the test.

    `from win_utils import send_mouse_move` (ai_aiming.py's module top)
    only ever executes once per process — after the first test imports
    core.ai_aiming, it stays cached in sys.modules, and its `send_mouse_move`
    name stays bound to whichever stub was live at that first import.
    Patching sys.modules['win_utils'] alone is therefore only enough for
    that first import; every test additionally patches the already-imported
    (or freshly-imported) core.ai_aiming module's own bound name directly,
    so each test's own capture list is actually the one process_aiming()
    calls into, regardless of test/import order.
    """
    moves = []

    def _fake_send_mouse_move(dx, dy, method=None):
        moves.append((dx, dy, method))

    stub = types.ModuleType("win_utils")
    stub.send_mouse_move = _fake_send_mouse_move
    monkeypatch.setitem(sys.modules, "win_utils", stub)

    import core.ai_aiming as _aiming_mod
    monkeypatch.setattr(_aiming_mod, "send_mouse_move", _fake_send_mouse_move)
    return moves


def _make_config(**overrides):
    cfg = SimpleNamespace(
        aim_part="center", head_height_ratio=0.26, aim_custom_y_pct=50.0,
        aim_adaptive_ratio_enabled=False, aim_posture_aware_enabled=False,
        sticky_lock_enabled=False, lock_iou_threshold=0.3, lock_decay_frames=15,
        target_priority_mode="distance", target_priority_confidence_weight=0.5,
        kalman_enabled=False, prediction_enabled=False,
        prediction_horizon_ms=10.0, prediction_max_velocity=1200.0, prediction_history_len=3,
        cam_motion_comp_enabled=False, aim_deadzone_enabled=False,
        aim_y_reduce_enabled=False, aim_y_reduce_delay=0.0,
        aim_y_reduce_settle_px=0.0, aim_y_reduce_floor=0.1, aim_y_reduce_ramp=0.0,
        aim_y_vel_restore_px_s=0.0, humanization=None, max_move_per_frame_px=0.0,
        mouse_move_method="sendinput",
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


class TestYRecoilTargetSwapReset:
    """A same-frame target swap (boxes stays non-empty, so the full
    target-loss reset in ai_loop.py never runs) must not let a spurious
    cross-target Y velocity spike wrongly disable Y-recoil suppression."""

    def test_suppression_survives_a_target_swap(self, sent_moves):
        from core.ai_aiming import process_aiming
        from core.ai_loop_state import LoopState
        from core.inference import PIDController

        config = _make_config(
            aim_y_reduce_enabled=True, aim_y_reduce_delay=0.0,
            aim_y_reduce_settle_px=0.0, aim_y_reduce_floor=0.1,
            aim_y_reduce_ramp=0.0, aim_y_vel_restore_px_s=500.0,
        )
        state = LoopState()
        state.aiming_start_time = 1000.0 - 10.0
        pid_x, pid_y = PIDController(1, 0, 0), PIDController(1, 0, 0)

        box_a = [480.0, 480.0, 520.0, 520.0]  # center (500, 500) — on crosshair
        process_aiming(config, [box_a], 500, 500, pid_x, pid_y, "sendinput", state,
                        current_time=1000.0, confidences=[0.9])
        assert state.locked_box == box_a
        assert state.aim_y_last_target_t == 1000.0

        # 50ms later: a DIFFERENT target (IOU with box_a is 0), far away in Y.
        box_b = [480.0, 700.0, 520.0, 740.0]  # center (500, 720)
        sent_moves.clear()
        process_aiming(config, [box_b], 500, 500, pid_x, pid_y, "sendinput", state,
                        current_time=1000.05, confidences=[0.9])

        assert len(sent_moves) == 1
        dx, dy, _ = sent_moves[0]
        # errorY=220, Kp=1 -> raw PID output 220. Suppression (floor=0.1) must
        # still apply (~22) — the pre-fix bug computed vy=(720-500)/0.05=4400
        # px/s, blew past vel_restore=500, and bypassed suppression entirely
        # (dy stayed the full unsuppressed ~220).
        assert 15 <= dy <= 30, f"Y-suppression was bypassed by a spurious velocity spike: dy={dy}"

    def test_suppression_stays_active_for_a_continuous_track(self, sent_moves):
        """Same two-frame shape, but confirms the assertion band above isn't
        trivially satisfied — a genuinely continuous track (no swap) must
        also land in the same suppressed range, not just "always small"."""
        from core.ai_aiming import process_aiming
        from core.ai_loop_state import LoopState
        from core.inference import PIDController

        config = _make_config(
            aim_y_reduce_enabled=True, aim_y_reduce_delay=0.0,
            aim_y_reduce_settle_px=0.0, aim_y_reduce_floor=0.1,
            aim_y_reduce_ramp=0.0, aim_y_vel_restore_px_s=500.0,
        )
        state = LoopState()
        state.aiming_start_time = 2000.0 - 10.0
        pid_x, pid_y = PIDController(1, 0, 0), PIDController(1, 0, 0)

        box_a = [480.0, 480.0, 520.0, 520.0]
        box_b = [480.0, 700.0, 520.0, 740.0]
        process_aiming(config, [box_a], 500, 500, pid_x, pid_y, "sendinput", state,
                        current_time=2000.0, confidences=[0.9])
        sent_moves.clear()
        process_aiming(config, [box_b], 500, 500, pid_x, pid_y, "sendinput", state,
                        current_time=2000.05, confidences=[0.9])

        dx, dy, _ = sent_moves[0]
        assert 15 <= dy <= 30


class TestVelocityPrediction:
    """Velocity prediction must extrapolate the aim point ahead of the raw
    detected position by prediction_horizon_ms, not just track it exactly.
    Restored after being merged into Kalman-only smoothing — see CLAUDE.md /
    Aiming_Pipeline_Audit history for why."""

    def test_moving_target_is_extrapolated_ahead_of_raw_position(self, sent_moves, monkeypatch):
        from core.ai_aiming import process_aiming
        from core.ai_loop_state import LoopState
        from core.inference import PIDController
        import core.ai_aiming as aiming_mod

        config = _make_config(
            prediction_enabled=True, prediction_horizon_ms=100.0,
            prediction_history_len=3, prediction_max_velocity=5000.0,
        )
        state = LoopState()
        pid_x, pid_y = PIDController(1, 0, 0), PIDController(1, 0, 0)

        # process_aiming's prediction stage reads time.perf_counter() directly
        # (not the current_time kwarg), so drive it deterministically here.
        perf_times = iter([0.0, 0.1])
        monkeypatch.setattr(aiming_mod.time, "perf_counter", lambda: next(perf_times))

        box_a = [480.0, 480.0, 520.0, 520.0]  # center (500, 500) — on crosshair
        process_aiming(config, [box_a], 500, 500, pid_x, pid_y, "sendinput", state,
                        current_time=1000.0, confidences=[0.9])

        # 100ms later, target moved +10px in X -> vx = 100 px/s.
        box_b = [490.0, 480.0, 530.0, 520.0]  # center (510, 500)
        sent_moves.clear()
        process_aiming(config, [box_b], 500, 500, pid_x, pid_y, "sendinput", state,
                        current_time=1000.1, confidences=[0.9])

        assert len(sent_moves) == 1
        dx, dy, _ = sent_moves[0]
        # Raw detected error is 510-500=10. Predicted error (horizon=100ms,
        # vx=100px/s) = 10 + 100*0.1 = 20 — prediction must extrapolate past
        # the raw detected position, not just reproduce it.
        assert dx > 10, f"Prediction did not extrapolate ahead of raw position: dx={dx}"

    def test_disabled_prediction_tracks_raw_position_exactly(self, sent_moves, monkeypatch):
        """Sanity check for the test above: with prediction_enabled=False the
        same motion must NOT be extrapolated — dx should equal the raw error."""
        from core.ai_aiming import process_aiming
        from core.ai_loop_state import LoopState
        from core.inference import PIDController
        import core.ai_aiming as aiming_mod

        config = _make_config(prediction_enabled=False)
        state = LoopState()
        pid_x, pid_y = PIDController(1, 0, 0), PIDController(1, 0, 0)

        perf_times = iter([0.0, 0.1])
        monkeypatch.setattr(aiming_mod.time, "perf_counter", lambda: next(perf_times))

        box_a = [480.0, 480.0, 520.0, 520.0]
        process_aiming(config, [box_a], 500, 500, pid_x, pid_y, "sendinput", state,
                        current_time=1000.0, confidences=[0.9])

        box_b = [490.0, 480.0, 530.0, 520.0]  # center (510, 500)
        sent_moves.clear()
        process_aiming(config, [box_b], 500, 500, pid_x, pid_y, "sendinput", state,
                        current_time=1000.1, confidences=[0.9])

        assert len(sent_moves) == 1
        dx, dy, _ = sent_moves[0]
        assert dx == 10, f"Raw tracking should not extrapolate when prediction is disabled: dx={dx}"


class TestDeadzonePreviousErrorFreshness:
    """previous_error must not go stale while the deadzone suppresses
    movement, or the derivative term sees a multi-frame-old error (and
    produces a one-frame Kd kick) the moment the target exits the deadzone."""

    def test_previous_error_reset_while_suppressed(self, sent_moves, monkeypatch):
        from core.ai_aiming import process_aiming
        from core.ai_loop_state import LoopState
        from core.inference import PIDController
        import core.ai_aiming as aiming_mod

        config = _make_config(aim_deadzone_enabled=True)
        state = LoopState()
        pid_x, pid_y = PIDController(Kp=0.0, Ki=0.0, Kd=1.0), PIDController(Kp=0.0, Ki=0.0, Kd=1.0)
        monkeypatch.setattr(aiming_mod, "_apply_adaptive_deadzone", lambda ex, ey, bh, cfg: (0.0, 0.0))

        # Simulate "was tracking, error was nonzero" before entering the deadzone.
        pid_x.update(50.0)
        pid_y.update(50.0)
        assert pid_x.previous_error == 50.0

        box_c = [480.0, 480.0, 520.0, 520.0]
        process_aiming(config, [box_c], 500, 500, pid_x, pid_y, "sendinput", state,
                        current_time=1.0, confidences=[0.9])

        assert sent_moves == []  # deadzone early-return, no movement sent
        assert pid_x.previous_error == 0.0
        assert pid_y.previous_error == 0.0


class TestIdleMicroJitter:
    """apply_idle_micro_jitter() — Humanization's optional "aim key held, no
    target this frame" tremor (ai_loop.py calls this from its `else` branch
    instead of process_aiming(), which never runs without boxes). Off by
    default via humanization.micro_jitter_idle_enabled — a separate opt-in
    from micro_jitter_enabled (which only gates the on-target case)."""

    @staticmethod
    def _make_hcfg(**overrides):
        from core.humanization import HumanizationConfig
        hcfg = HumanizationConfig()
        for k, v in overrides.items():
            setattr(hcfg, k, v)
        return hcfg

    def test_noop_when_humanization_missing(self, sent_moves):
        from core.ai_aiming import apply_idle_micro_jitter
        from core.ai_loop_state import LoopState

        config = _make_config(humanization=None)
        apply_idle_micro_jitter(config, LoopState(), "sendinput")
        assert sent_moves == []

    def test_noop_when_humanization_disabled(self, sent_moves):
        from core.ai_aiming import apply_idle_micro_jitter
        from core.ai_loop_state import LoopState

        hcfg = self._make_hcfg(enabled=False, micro_jitter_enabled=True, micro_jitter_idle_enabled=True)
        config = _make_config(humanization=hcfg)
        apply_idle_micro_jitter(config, LoopState(), "sendinput")
        assert sent_moves == []

    def test_noop_when_micro_jitter_feature_off(self, sent_moves):
        from core.ai_aiming import apply_idle_micro_jitter
        from core.ai_loop_state import LoopState

        hcfg = self._make_hcfg(enabled=True, micro_jitter_enabled=False, micro_jitter_idle_enabled=True)
        config = _make_config(humanization=hcfg)
        apply_idle_micro_jitter(config, LoopState(), "sendinput")
        assert sent_moves == []

    def test_noop_when_idle_toggle_off_by_default(self, sent_moves):
        """The whole point of the opt-in: Humanization + Micro-Jitter both on
        (the shipped default) must NOT fire idle jitter until the new toggle
        is explicitly enabled too."""
        from core.ai_aiming import apply_idle_micro_jitter
        from core.ai_loop_state import LoopState

        hcfg = self._make_hcfg(enabled=True, micro_jitter_enabled=True)
        assert hcfg.micro_jitter_idle_enabled is False  # default
        config = _make_config(humanization=hcfg)
        apply_idle_micro_jitter(config, LoopState(), "sendinput")
        assert sent_moves == []

    def test_fires_and_carries_subpixel_remainder_when_fully_enabled(self, sent_moves, monkeypatch):
        from core.ai_aiming import apply_idle_micro_jitter
        from core.ai_loop_state import LoopState
        import core.humanization as humanization_mod

        hcfg = self._make_hcfg(
            enabled=True, intensity=1.0,
            micro_jitter_enabled=True, micro_jitter_idle_enabled=True,
            micro_jitter_base=2.0, micro_jitter_scale=0.0,
            motion_variation_enabled=False, speed_shaping_enabled=False,
            micro_stutter_enabled=False, reaction_variability_enabled=False,
        )
        config = _make_config(humanization=hcfg)
        state = LoopState()

        # apply_humanization's Micro-Jitter step draws random.uniform(-amp, amp)
        # independently for dx then dy — pin both draws to +amp deterministically.
        monkeypatch.setattr(humanization_mod.random, "uniform", lambda a, b: b)

        apply_idle_micro_jitter(config, state, "sendinput")

        assert len(sent_moves) == 1
        dx, dy, method = sent_moves[0]
        assert method == "sendinput"
        # amp = micro_jitter_base * intensity = 2.0 (scale term is 0 at zero
        # magnitude either way) -> both axes land exactly on 2px, no carry left.
        assert dx == 2 and dy == 2
        assert state.aim_carry_x == 0.0 and state.aim_carry_y == 0.0

    def test_skipped_on_reaction_variability_frame_skip(self, sent_moves, monkeypatch):
        from core.ai_aiming import apply_idle_micro_jitter
        from core.ai_loop_state import LoopState
        import core.humanization as humanization_mod

        hcfg = self._make_hcfg(
            enabled=True, intensity=1.0,
            micro_jitter_enabled=True, micro_jitter_idle_enabled=True,
            reaction_variability_enabled=True, reaction_skip_prob=1.0,  # always skip
        )
        config = _make_config(humanization=hcfg)
        # random.random() < reaction_skip_prob * intensity -> skip this frame.
        monkeypatch.setattr(humanization_mod.random, "random", lambda: 0.0)

        apply_idle_micro_jitter(config, LoopState(), "sendinput")
        assert sent_moves == []
