"""The Anchor features' redundant options were removed (docs/ANCHOR_PREDICTION.md, ANCHOR_FOV.md).

Old configs, presets and scripts must keep working: every retired value is folded into its
replacement at load, with the same behaviour.
"""
import math
import random
from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest

from core import anchor_fov as af
from core import config as cfgmod
from core.anchor_prediction import MODES_UI, AnchorParams, AnchorTrack, aim_uses_anchor, anchor_active


def new_config():
    with patch("core.config._get_screen_size", return_value=(1920, 1080)):
        return cfgmod.Config()


def load(data):
    c = new_config()
    c.from_dict(data)
    return c


# ── Anchor Box Prediction: Disabled and Linear are gone ──────────────────────────
class TestModeMigration:
    def test_offered_modes_are_the_three_that_differ(self):
        assert MODES_UI == ("velocity_accel", "smoothed", "adaptive")

    def test_linear_becomes_velocity_with_zero_acceleration(self):
        c = load({"tracking": {"anchor": {"mode": "linear", "accel_comp": 0.6, "enabled": True}}})
        assert (c.anchor_mode, c.anchor_accel_comp, c.anchor_enabled) == ("velocity_accel", 0.0, True)

    def test_disabled_turns_the_fov_layer_off(self):
        c = load({"tracking": {"anchor": {"mode": "disabled", "enabled": True}}})
        assert c.anchor_enabled is False and c.anchor_mode == "adaptive"
        assert anchor_active(c) is False

    def test_disabled_with_the_aim_lead_method_keeps_meaning_no_lead(self):
        c = load({"tracking": {"anchor": {"mode": "disabled"}, "prediction": {"enabled": True, "method": "anchor"}}})
        assert c.prediction_method == "velocity" and aim_uses_anchor(c) is False

    def test_disabled_leaves_other_prediction_methods_alone(self):
        c = load({"tracking": {"anchor": {"mode": "disabled"}, "prediction": {"method": "velocity"}}})
        assert c.prediction_method == "velocity"

    @pytest.mark.parametrize("mode", MODES_UI)
    def test_current_modes_are_untouched(self, mode):
        c = load({"tracking": {"anchor": {"mode": mode, "accel_comp": 0.5, "enabled": True}}})
        assert (c.anchor_mode, c.anchor_accel_comp, c.anchor_enabled) == (mode, 0.5, True)

    def test_validate_config_migrates_in_memory_values_too(self):
        c = new_config()
        c.anchor_mode = "linear"
        c.anchor_accel_comp = 0.9
        cfgmod.validate_config(c)
        assert (c.anchor_mode, c.anchor_accel_comp) == ("velocity_accel", 0.0)
        c.anchor_mode = "nonsense"
        cfgmod.validate_config(c)
        assert c.anchor_mode == "adaptive"

    def test_migration_is_idempotent(self):
        c = new_config()
        c.anchor_mode, c.anchor_enabled = "disabled", True
        cfgmod.migrate_anchor_settings(c)
        snapshot = (c.anchor_mode, c.anchor_enabled, c.anchor_accel_comp)
        cfgmod.migrate_anchor_settings(c)
        assert snapshot == (c.anchor_mode, c.anchor_enabled, c.anchor_accel_comp)


def _sequence(kind, n=90, fps=60.0, seed=4):
    rng = random.Random(seed)
    out = []
    for i in range(n):
        t = i / fps
        if kind == "constant":
            x, y = 400 + 600 * t, 500
        elif kind == "accel":
            x, y = 400 + 0.5 * 1800 * t * t, 500 + 40 * t
        elif kind == "reversal":
            x, y = (400 + 500 * t if t < 0.7 else 400 + 350 - 500 * (t - 0.7)), 500
        else:  # noisy
            x, y = 500 + 120 * t + rng.gauss(0, 1.0), 500 + rng.gauss(0, 1.0)
        out.append((1000.0 + t, (x - 20, y - 40, x + 20, y + 40)))
    return out


@pytest.mark.parametrize("kind", ["constant", "accel", "reversal", "noisy"])
def test_linear_is_exactly_velocity_with_zero_acceleration(kind):
    """The estimator only used 'linear' to zero the acceleration term, so the migration
    (linear -> velocity_accel + accel_comp 0) must give identical predictions."""
    def track(mode, accel):
        p = AnchorParams()
        p.mode, p.accel_comp, p.horizon_s, p.max_speed_px_s = mode, accel, 0.1, 1e6
        return AnchorTrack(p)

    old, new = track("linear", 0.9), track("velocity_accel", 0.0)
    for t, box in _sequence(kind):
        old.update(t, box, 0.9)
        new.update(t, box, 0.9)
        a, b = old.predict(0.1, t), new.predict(0.1, t)
        assert (a.valid, a.pcx, a.pcy, a.px1, a.py1, a.px2, a.py2) == (b.valid, b.pcx, b.pcy, b.px1, b.py1, b.px2, b.py2)


# ── Anchor FOV: custom, manual X/Y, strength, radius, min radius ─────────────────
class TestAnchorFovMigration:
    def test_custom_position_is_the_centre(self):
        assert load({"aim": {"fov_anchor": {"position": "custom"}}}).afov_position == "center"
        box = [100, 200, 140, 280]
        assert af.anchor_point(box, "center", 5, -7) == (125.0, 233.0)       # offsets added in every case

    def test_manual_xy_folds_into_the_offsets_for_the_manual_source(self):
        c = load({"aim": {"fov_anchor": {"source": "manual", "manual_x": 120, "manual_y": -40}}})
        assert (c.afov_offset_x, c.afov_offset_y) == (120, -40)

    def test_manual_xy_never_overrides_offsets_the_user_already_set(self):
        c = load({"aim": {"fov_anchor": {"source": "manual", "manual_x": 120, "manual_y": -40,
                                         "offset_x": 7, "offset_y": 0}}})
        assert (c.afov_offset_x, c.afov_offset_y) == (7, 0)

    def test_manual_xy_is_ignored_for_the_other_sources(self):
        c = load({"aim": {"fov_anchor": {"source": "tracker", "manual_x": 120, "manual_y": -40}}})
        assert (c.afov_offset_x, c.afov_offset_y) == (0, 0)

    @pytest.mark.parametrize("time_ms,strength,want", [(80, 1.5, 120), (100, 0.5, 50), (300, 2.0, 500),
                                                       (80, 0.0, 0), (80, 1.0, 80)])
    def test_strength_is_folded_into_the_look_ahead_time(self, time_ms, strength, want):
        c = load({"aim": {"fov_anchor": {"predict": {"time_ms": time_ms, "strength": strength}}}})
        assert c.afov_predict_time_ms == want

    def test_the_lead_is_numerically_unchanged_by_the_fold(self):
        t_ms, strength = 90, 1.4
        c = load({"aim": {"fov_anchor": {"predict": {"enabled": True, "time_ms": t_ms, "strength": strength}}}})
        p = af.AnchorFovParams()
        p.load(c)
        assert p.lead_s == pytest.approx(t_ms * strength / 1000.0, abs=0.0006)      # old: lead_s * strength

    def test_retired_keys_are_not_attributes_and_never_written_back(self):
        c = load({"aim": {"fov_anchor": {"radius": {"default": 250, "min": 20}, "manual_x": 5,
                                         "predict": {"strength": 1.5}}}})
        for k in ("afov_radius", "afov_min_radius", "afov_manual_x", "afov_manual_y", "afov_predict_strength"):
            assert not hasattr(c, k), k
        assert "manual_x" not in str(c.to_dict()) and "strength" not in str(c.to_dict()["aim"]["fov_anchor"])

    def test_old_main_build_file_still_loads_everything_else(self):
        c = load({"aim": {"fov_size": 260, "fov_anchor": {"source": "tracker", "follow_speed": 20,
                                                           "radius": {"default": 100, "max": 280, "dynamic": True}}}})
        assert (c.afov_source, c.afov_follow_speed, c.afov_max_radius, c.afov_dynamic_radius) == \
            ("tracker", 20.0, 280, True)


class TestAnchorFovSize:
    @staticmethod
    def params(**kw):
        c = NS(fov_size=200, fov_height=200, afov_max_radius=300)
        for k, v in kw.items():
            setattr(c, k, v)
        p = af.AnchorFovParams()
        p.load(c)
        return p

    def test_radius_is_half_the_fov_width(self):
        assert self.params(fov_size=200).radius == 100.0          # same as the old default of 100
        assert self.params(fov_size=340).radius == 170.0

    def test_maximum_radius_is_headroom_only(self):
        p = self.params(fov_size=500, afov_max_radius=120)
        assert p.radius == 250.0 and p.max_radius == 250.0
        assert self.params(fov_size=200, afov_max_radius=300).max_radius == 300.0

    def test_floor_and_garbage_fov_width(self):
        assert self.params(fov_size=4).radius == af.MIN_RADIUS_PX
        assert self.params(fov_size=float("nan")).radius == af.DEFAULT_RADIUS_PX
        assert math.isfinite(self.params(fov_size=None).radius)


# ── FOV look-ahead: one slider (0 = off) instead of a switch plus a time ─────────
class TestLookAheadMerge:
    @pytest.mark.parametrize("predict,want", [
        ({"enabled": False, "time_ms": 120}, 0),                 # off stays off
        ({"enabled": True, "time_ms": 120}, 120),                # on keeps the tuned time
        ({"enabled": True}, 80),                                 # on without a time = the old default time
        ({"enabled": False}, 0),
        ({"enabled": True, "time_ms": 80, "strength": 1.5}, 120),   # strength fold still applies first
    ])
    def test_legacy_files_fold_into_the_time(self, predict, want):
        assert load({"aim": {"fov_anchor": {"predict": predict}}}).afov_predict_time_ms == want

    def test_new_files_have_no_switch_and_keep_their_time(self):
        c = new_config()
        c.afov_predict_time_ms = 90
        d = c.to_dict()["aim"]["fov_anchor"]["predict"]
        assert d == {"time_ms": 90}
        assert load(c.to_dict()).afov_predict_time_ms == 90        # no 'enabled' key -> nothing is zeroed

    def test_default_is_off_and_the_switch_is_gone(self):
        c = new_config()
        assert c.afov_predict_time_ms == 0 and not hasattr(c, "afov_predict_enabled")
        assert af.predict_active(c) is False

    def test_params_follow_the_time(self):
        p = af.AnchorFovParams()
        p.load(NS(fov_size=200, fov_height=200, afov_predict_time_ms=150))
        assert p.predict is True and p.lead_s == pytest.approx(0.15)
        p.load(NS(fov_size=200, fov_height=200, afov_predict_time_ms=0))
        assert p.predict is False
