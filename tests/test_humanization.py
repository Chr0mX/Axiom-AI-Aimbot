"""
Tests for src/core/humanization.py

Coverage:
  - HumanizationConfig defaults and field types
  - apply_humanization: disabled / zero-intensity passthrough
  - Feature 1: micro-jitter — zero-mean bias, magnitude scaling
  - Feature 2: motion variation — mean-preserving, bounded
  - Feature 3: reaction variability — None return, zero prob
  - Feature 4: speed shaping — low/high/boundary magnitudes
  - Feature 5: micro-stutter — magnitude reduction guarantee
  - Determinism with fixed random seed
  - Direction preservation for large movements
  - Config serialisation round-trip (to_dict / from_dict)
"""

from __future__ import annotations

import math
import random
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cfg(**kwargs):
    """Build a HumanizationConfig with all features off, then apply overrides."""
    from core.humanization import HumanizationConfig
    base = HumanizationConfig(
        enabled=True,
        intensity=1.0,
        micro_jitter_enabled=False,
        motion_variation_enabled=False,
        speed_shaping_enabled=False,
        micro_stutter_enabled=False,
        reaction_variability_enabled=False,
    )
    for k, v in kwargs.items():
        setattr(base, k, v)
    return base


def _apply(dx, dy, **kwargs):
    from core.humanization import apply_humanization
    return apply_humanization(dx, dy, _cfg(**kwargs))


def _make_config():
    with patch("core.config._get_screen_size", return_value=(1920, 1080)):
        from core.config import Config
        return Config()


# ---------------------------------------------------------------------------
# HumanizationConfig defaults
# ---------------------------------------------------------------------------

class TestHumanizationConfigDefaults:
    def test_enabled_by_default(self):
        from core.humanization import HumanizationConfig
        assert HumanizationConfig().enabled is True

    def test_intensity_in_range(self):
        from core.humanization import HumanizationConfig
        cfg = HumanizationConfig()
        assert 0.0 <= cfg.intensity <= 1.0

    def test_reaction_variability_off_by_default(self):
        from core.humanization import HumanizationConfig
        assert HumanizationConfig().reaction_variability_enabled is False

    def test_micro_stutter_off_by_default(self):
        from core.humanization import HumanizationConfig
        assert HumanizationConfig().micro_stutter_enabled is False

    def test_speed_shaping_thresholds_ordered(self):
        from core.humanization import HumanizationConfig
        cfg = HumanizationConfig()
        assert cfg.speed_shaping_low < cfg.speed_shaping_high

    def test_stutter_bounds_ordered(self):
        from core.humanization import HumanizationConfig
        cfg = HumanizationConfig()
        assert 0.0 < cfg.micro_stutter_min <= cfg.micro_stutter_max < 1.0


# ---------------------------------------------------------------------------
# Passthrough cases
# ---------------------------------------------------------------------------

class TestPassthrough:
    def test_disabled_returns_input_unchanged(self):
        from core.humanization import HumanizationConfig, apply_humanization
        cfg = HumanizationConfig(enabled=False)
        assert apply_humanization(7.3, -4.1, cfg) == (7.3, -4.1)

    def test_zero_intensity_returns_input_unchanged(self):
        from core.humanization import HumanizationConfig, apply_humanization
        cfg = HumanizationConfig(intensity=0.0)
        assert apply_humanization(7.3, -4.1, cfg) == (7.3, -4.1)

    def test_all_features_off_returns_input_unchanged(self):
        result = _apply(5.0, -3.0)
        assert result == (5.0, -3.0)

    def test_zero_movement_all_features_off(self):
        result = _apply(0.0, 0.0)
        assert result == (0.0, 0.0)


# ---------------------------------------------------------------------------
# Feature 1 · Micro-jitter
# ---------------------------------------------------------------------------

class TestMicroJitter:
    def test_output_differs_from_input(self):
        random.seed(99)
        result = _apply(10.0, 10.0, micro_jitter_enabled=True)
        assert result != (10.0, 10.0)

    def test_zero_mean_bias_x(self):
        """Over many samples E[noise_x] ≈ 0 (symmetric uniform)."""
        random.seed(0)
        total = sum(
            _apply(0.0, 0.0, micro_jitter_enabled=True)[0]
            for _ in range(20_000)
        )
        assert abs(total / 20_000) < 0.02, "X-axis bias too large"

    def test_zero_mean_bias_y(self):
        random.seed(1)
        total = sum(
            _apply(0.0, 0.0, micro_jitter_enabled=True)[1]
            for _ in range(20_000)
        )
        assert abs(total / 20_000) < 0.02, "Y-axis bias too large"

    def test_amplitude_scales_with_magnitude(self):
        """Larger movement → larger noise amplitude."""
        random.seed(42)
        N = 5000
        noise_small = [
            abs(_apply(1.0, 0.0, micro_jitter_enabled=True)[0] - 1.0)
            for _ in range(N)
        ]
        random.seed(42)
        noise_large = [
            abs(_apply(50.0, 0.0, micro_jitter_enabled=True)[0] - 50.0)
            for _ in range(N)
        ]
        assert sum(noise_large) / N > sum(noise_small) / N

    def test_noise_bounded_at_default_params(self):
        """Jitter should never exceed a few pixels at default settings."""
        from core.humanization import HumanizationConfig, apply_humanization
        cfg = HumanizationConfig(
            enabled=True, intensity=1.0,
            micro_jitter_enabled=True,
            motion_variation_enabled=False,
            speed_shaping_enabled=False,
            micro_stutter_enabled=False,
            reaction_variability_enabled=False,
        )
        random.seed(7)
        for _ in range(2000):
            result = apply_humanization(20.0, 0.0, cfg)
            assert result is not None
            # Noise amplitude = base + scale*mag = 0.20 + 0.025*20 = 0.70 px max
            assert abs(result[0] - 20.0) <= 1.0, "Jitter exceeded expected bound"


# ---------------------------------------------------------------------------
# Feature 2 · Motion variation
# ---------------------------------------------------------------------------

class TestMotionVariation:
    def test_output_differs_from_input(self):
        random.seed(5)
        result = _apply(10.0, 10.0, motion_variation_enabled=True)
        assert result != (10.0, 10.0)

    def test_mean_preserving_over_many_frames(self):
        """E[output] ≈ E[input] — no bias accumulates."""
        random.seed(2)
        N = 20_000
        dx_sum = sum(
            _apply(10.0, 0.0, motion_variation_enabled=True)[0]
            for _ in range(N)
        )
        mean = dx_sum / N
        assert abs(mean - 10.0) < 0.1, f"Motion variation introduced bias: {mean}"

    def test_scale_bounded(self):
        """Output stays within (1 ± range) * input."""
        from core.humanization import HumanizationConfig, apply_humanization
        rng = 0.06
        cfg = HumanizationConfig(
            enabled=True, intensity=1.0,
            micro_jitter_enabled=False,
            motion_variation_enabled=True,
            motion_variation_range=rng,
            speed_shaping_enabled=False,
            micro_stutter_enabled=False,
            reaction_variability_enabled=False,
        )
        random.seed(3)
        for _ in range(1000):
            result = apply_humanization(10.0, 0.0, cfg)
            assert result is not None
            assert 10.0 * (1.0 - rng) - 1e-9 <= result[0] <= 10.0 * (1.0 + rng) + 1e-9

    def test_zero_movement_untouched(self):
        """Zero input stays zero (branch: magnitude == 0 skips variation)."""
        random.seed(9)
        for _ in range(100):
            result = _apply(0.0, 0.0, motion_variation_enabled=True)
            assert result == (0.0, 0.0)


# ---------------------------------------------------------------------------
# Feature 3 · Reaction variability
# ---------------------------------------------------------------------------

class TestReactionVariability:
    def test_certain_skip_returns_none(self):
        """prob=1.0 → every call returns None."""
        cfg = _cfg(
            reaction_variability_enabled=True,
            reaction_skip_prob=1.0,
        )
        from core.humanization import apply_humanization
        random.seed(0)
        for _ in range(100):
            assert apply_humanization(5.0, 5.0, cfg) is None

    def test_zero_prob_never_skips(self):
        """prob=0.0 → never returns None."""
        cfg = _cfg(
            reaction_variability_enabled=True,
            reaction_skip_prob=0.0,
        )
        from core.humanization import apply_humanization
        random.seed(0)
        for _ in range(1000):
            assert apply_humanization(5.0, 5.0, cfg) is not None

    def test_default_prob_is_low(self):
        """At default settings, skip rate is very low (< 5%)."""
        from core.humanization import HumanizationConfig, apply_humanization
        cfg = HumanizationConfig(
            enabled=True, intensity=1.0,
            micro_jitter_enabled=False,
            motion_variation_enabled=False,
            speed_shaping_enabled=False,
            micro_stutter_enabled=False,
            reaction_variability_enabled=True,
        )
        random.seed(11)
        skips = sum(
            1 for _ in range(10_000)
            if apply_humanization(5.0, 5.0, cfg) is None
        )
        assert skips / 10_000 < 0.05

    def test_feature_disabled_never_returns_none(self):
        """When disabled the feature is entirely bypassed."""
        random.seed(0)
        result = _apply(5.0, 5.0, reaction_variability_enabled=False)
        assert result is not None


# ---------------------------------------------------------------------------
# Feature 4 · Speed shaping
# ---------------------------------------------------------------------------

class TestSpeedShaping:
    def test_below_low_threshold_compressed(self):
        """Movement below low threshold is scaled down."""
        result = _apply(
            2.0, 0.0,
            speed_shaping_enabled=True,
            speed_shaping_low=10.0,
            speed_shaping_high=30.0,
            speed_shaping_low_factor=0.8,
        )
        assert result is not None
        assert result[0] < 2.0, "Sub-threshold movement should be compressed"

    def test_above_high_threshold_passthrough(self):
        """Movement above high threshold is not scaled."""
        result = _apply(
            50.0, 0.0,
            speed_shaping_enabled=True,
            speed_shaping_low=4.0,
            speed_shaping_high=22.0,
            speed_shaping_low_factor=0.88,
        )
        assert result is not None
        assert result[0] == pytest.approx(50.0, abs=1e-9)

    def test_at_boundary_low_gets_low_factor(self):
        """Movement exactly at low threshold gets low_factor scaling."""
        low = 5.0
        low_f = 0.7
        result = _apply(
            low, 0.0,
            speed_shaping_enabled=True,
            speed_shaping_low=low,
            speed_shaping_high=20.0,
            speed_shaping_low_factor=low_f,
        )
        assert result is not None
        assert result[0] == pytest.approx(low * low_f, abs=1e-9)

    def test_mid_range_is_between_bounds(self):
        """Movement between thresholds is between low_factor and 1.0 scaling."""
        low, high, low_f = 4.0, 20.0, 0.8
        mid = (low + high) / 2  # 12.0
        result = _apply(
            mid, 0.0,
            speed_shaping_enabled=True,
            speed_shaping_low=low,
            speed_shaping_high=high,
            speed_shaping_low_factor=low_f,
        )
        assert result is not None
        lo_bound = mid * low_f
        hi_bound = mid * 1.0
        assert lo_bound <= result[0] <= hi_bound

    def test_direction_preserved(self):
        """Speed shaping must not flip the sign of dx or dy."""
        for mag in [1.0, 5.0, 15.0, 40.0]:
            result = _apply(
                mag, -mag,
                speed_shaping_enabled=True,
                speed_shaping_low=4.0,
                speed_shaping_high=22.0,
                speed_shaping_low_factor=0.88,
            )
            assert result is not None
            assert result[0] > 0, "dx sign flipped"
            assert result[1] < 0, "dy sign flipped"

    def test_scale_reduces_with_intensity(self):
        """Lower intensity → compression is closer to 1.0 (less effect)."""
        from core.humanization import HumanizationConfig, apply_humanization

        def shaped(intensity):
            cfg = HumanizationConfig(
                enabled=True, intensity=intensity,
                micro_jitter_enabled=False, motion_variation_enabled=False,
                speed_shaping_enabled=True, micro_stutter_enabled=False,
                reaction_variability_enabled=False,
                speed_shaping_low=10.0, speed_shaping_high=30.0,
                speed_shaping_low_factor=0.5,
            )
            return apply_humanization(5.0, 0.0, cfg)[0]

        assert shaped(1.0) < shaped(0.5) < shaped(0.1)


# ---------------------------------------------------------------------------
# Feature 5 · Micro-stutter
# ---------------------------------------------------------------------------

class TestMicroStutter:
    def test_certain_stutter_reduces_magnitude(self):
        """prob=1.0 → magnitude always reduced."""
        result = _apply(
            10.0, 0.0,
            micro_stutter_enabled=True,
            micro_stutter_prob=1.0,
            micro_stutter_min=0.5,
            micro_stutter_max=0.8,
        )
        assert result is not None
        assert math.hypot(*result) < 10.0

    def test_stutter_factor_bounded(self):
        """Output magnitude always in [min, max] * input when stuttered."""
        mn, mx = 0.6, 0.9
        random.seed(0)
        cfg = _cfg(
            micro_stutter_enabled=True,
            micro_stutter_prob=1.0,
            micro_stutter_min=mn,
            micro_stutter_max=mx,
        )
        from core.humanization import apply_humanization
        for _ in range(500):
            result = apply_humanization(10.0, 0.0, cfg)
            assert result is not None
            assert 10.0 * mn - 1e-9 <= result[0] <= 10.0 * mx + 1e-9

    def test_zero_prob_never_stutters(self):
        """prob=0 → output always equals input (no stutter)."""
        random.seed(0)
        for _ in range(500):
            result = _apply(
                10.0, 0.0,
                micro_stutter_enabled=True,
                micro_stutter_prob=0.0,
            )
            assert result is not None
            assert result == (10.0, 0.0)

    def test_zero_movement_skips_stutter(self):
        """Stutter branch is skipped when magnitude == 0."""
        result = _apply(
            0.0, 0.0,
            micro_stutter_enabled=True,
            micro_stutter_prob=1.0,
        )
        assert result == (0.0, 0.0)


# ---------------------------------------------------------------------------
# Cross-feature: determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_seed_same_output(self):
        from core.humanization import HumanizationConfig, apply_humanization
        cfg = HumanizationConfig(enabled=True, intensity=0.7)
        random.seed(999)
        r1 = apply_humanization(12.5, -7.3, cfg)
        random.seed(999)
        r2 = apply_humanization(12.5, -7.3, cfg)
        assert r1 == r2

    def test_different_seed_different_output(self):
        from core.humanization import HumanizationConfig, apply_humanization
        cfg = HumanizationConfig(enabled=True, intensity=1.0)
        random.seed(1)
        r1 = apply_humanization(10.0, 10.0, cfg)
        random.seed(2)
        r2 = apply_humanization(10.0, 10.0, cfg)
        # Extremely unlikely to be identical with independent seeds
        assert r1 != r2


# ---------------------------------------------------------------------------
# Cross-feature: direction preservation for large movements
# ---------------------------------------------------------------------------

class TestDirectionPreservation:
    def test_positive_dx_never_reversed(self):
        """All features combined must not reverse a large positive dx."""
        from core.humanization import HumanizationConfig, apply_humanization
        cfg = HumanizationConfig(
            enabled=True, intensity=0.5,
            micro_jitter_enabled=True,
            motion_variation_enabled=True,
            speed_shaping_enabled=True,
            micro_stutter_enabled=True,
            reaction_variability_enabled=False,  # avoid None returns in loop
        )
        random.seed(42)
        for _ in range(2000):
            result = apply_humanization(30.0, 0.0, cfg)
            if result is not None:
                assert result[0] > 0, "Direction reversed for large dx=30"

    def test_intensity_zero_exact_passthrough(self):
        from core.humanization import HumanizationConfig, apply_humanization
        cfg = HumanizationConfig(intensity=0.0)
        assert apply_humanization(-5.5, 8.8, cfg) == (-5.5, 8.8)


# ---------------------------------------------------------------------------
# make_humanization_config preset helper
# ---------------------------------------------------------------------------

class TestMakeHumanizationConfig:
    def test_zero_intensity_disabled(self):
        from core.humanization import make_humanization_config
        cfg = make_humanization_config(0.0)
        assert cfg.enabled is False

    def test_full_intensity_enabled(self):
        from core.humanization import make_humanization_config
        cfg = make_humanization_config(1.0)
        assert cfg.enabled is True
        assert cfg.intensity == 1.0
        assert cfg.micro_stutter_enabled is True
        assert cfg.reaction_variability_enabled is True

    def test_mid_intensity_partial_features(self):
        from core.humanization import make_humanization_config
        cfg = make_humanization_config(0.5)
        assert cfg.micro_jitter_enabled is True
        assert cfg.motion_variation_enabled is True
        assert cfg.speed_shaping_enabled is True
        assert cfg.micro_stutter_enabled is True      # >= 0.3
        assert cfg.reaction_variability_enabled is False  # < 0.6

    def test_clamps_out_of_range(self):
        from core.humanization import make_humanization_config
        cfg = make_humanization_config(1.5)
        assert cfg.intensity == 1.0
        cfg2 = make_humanization_config(-0.5)
        assert cfg2.enabled is False


# ---------------------------------------------------------------------------
# Config integration
# ---------------------------------------------------------------------------

class TestConfigIntegration:
    def test_config_has_humanization_attribute(self):
        from core.humanization import HumanizationConfig
        c = _make_config()
        assert hasattr(c, 'humanization')
        assert isinstance(c.humanization, HumanizationConfig)

    def test_to_dict_includes_humanization_as_dict(self):
        c = _make_config()
        d = c.to_dict()
        assert 'humanization' in d
        assert isinstance(d['humanization'], dict)
        assert 'enabled' in d['humanization']
        assert 'intensity' in d['humanization']

    def test_to_dict_humanization_fields_match_dataclass(self):
        import dataclasses
        from core.humanization import HumanizationConfig
        c = _make_config()
        d = c.to_dict()['humanization']
        field_names = {f.name for f in dataclasses.fields(HumanizationConfig)}
        assert field_names == set(d.keys())

    def test_from_dict_restores_enabled_false(self):
        c = _make_config()
        c.from_dict({'humanization': {'enabled': False}})
        assert c.humanization.enabled is False

    def test_from_dict_restores_intensity(self):
        c = _make_config()
        c.from_dict({'humanization': {'intensity': 0.9}})
        assert c.humanization.intensity == pytest.approx(0.9)

    def test_from_dict_ignores_unknown_humanization_keys(self):
        """Unknown keys in the humanization dict must not raise."""
        c = _make_config()
        c.from_dict({'humanization': {'nonexistent_key': 42}})  # should not raise

    def test_from_dict_without_humanization_key_uses_defaults(self):
        """If JSON has no 'humanization' key the defaults are unchanged."""
        from core.humanization import HumanizationConfig
        c = _make_config()
        original_enabled = c.humanization.enabled
        c.from_dict({'pid_kp_x': 0.5})  # no 'humanization' key
        assert c.humanization.enabled == original_enabled

    def test_round_trip_serialisation(self):
        """to_dict → from_dict preserves all humanization fields."""
        c1 = _make_config()
        c1.humanization.enabled = False
        c1.humanization.intensity = 0.3
        c1.humanization.micro_stutter_enabled = True
        c1.humanization.micro_stutter_prob = 0.07

        d = c1.to_dict()

        c2 = _make_config()
        c2.from_dict(d)

        assert c2.humanization.enabled is False
        assert c2.humanization.intensity == pytest.approx(0.3)
        assert c2.humanization.micro_stutter_enabled is True
        assert c2.humanization.micro_stutter_prob == pytest.approx(0.07)
