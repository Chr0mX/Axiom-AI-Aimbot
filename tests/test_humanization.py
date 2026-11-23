"""
Tests for src/core/humanization.py

Coverage:
  - HumanizationConfig defaults and field types
  - apply_humanization: disabled / zero-intensity passthrough
  - Feature 1: micro-jitter — zero-mean bias, magnitude scaling
  - Feature 2: motion variation — mean-preserving, bounded
  - Feature 4: speed shaping — low/high/boundary magnitudes
  - Determinism with fixed random seed
  - Direction preservation for large movements
  - Config serialisation round-trip (to_dict / from_dict)
"""

from __future__ import annotations

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
    def test_off_by_default(self):
        from core.humanization import HumanizationConfig
        assert HumanizationConfig().enabled is False

    def test_intensity_in_range(self):
        from core.humanization import HumanizationConfig
        cfg = HumanizationConfig()
        assert 0.0 <= cfg.intensity <= 1.0

    def test_speed_shaping_thresholds_ordered(self):
        from core.humanization import HumanizationConfig
        cfg = HumanizationConfig()
        assert cfg.speed_shaping_low < cfg.speed_shaping_high


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
                speed_shaping_enabled=True,
                speed_shaping_low=10.0, speed_shaping_high=30.0,
                speed_shaping_low_factor=0.5,
            )
            return apply_humanization(5.0, 0.0, cfg)[0]

        assert shaped(1.0) < shaped(0.5) < shaped(0.1)

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
        c = _make_config()
        original_enabled = c.humanization.enabled
        c.from_dict({'pid_kp_x': 0.5})  # no 'humanization' key
        assert c.humanization.enabled == original_enabled

    def test_round_trip_serialisation(self):
        """to_dict → from_dict preserves all humanization fields."""
        c1 = _make_config()
        c1.humanization.enabled = False
        c1.humanization.intensity = 0.3
        c1.humanization.motion_variation_enabled = False
        c1.humanization.micro_jitter_base = 0.07

        d = c1.to_dict()

        c2 = _make_config()
        c2.from_dict(d)

        assert c2.humanization.enabled is False
        assert c2.humanization.intensity == pytest.approx(0.3)
        assert c2.humanization.motion_variation_enabled is False
        assert c2.humanization.micro_jitter_base == pytest.approx(0.07)


class TestRemovedFeatures:
    """Micro-Stutter and Reaction Variability were removed: old config files carrying them must still load."""

    def test_old_keys_are_ignored_and_never_returned(self):
        c = _make_config()
        c.from_dict({"humanization": {"enabled": True, "intensity": 0.4, "micro_stutter_enabled": True,
                                      "micro_stutter_prob": 1.0, "reaction_variability_enabled": True,
                                      "reaction_skip_prob": 1.0}})
        assert c.humanization.enabled is True and c.humanization.intensity == pytest.approx(0.4)
        assert not hasattr(c.humanization, "micro_stutter_enabled")
        assert not hasattr(c.humanization, "reaction_variability_enabled")
        from core.humanization import apply_humanization
        for _ in range(200):
            assert apply_humanization(5.0, 5.0, c.humanization) is not None      # a frame is never skipped now

    def test_new_installs_start_with_humanization_off(self):
        assert _make_config().humanization.enabled is False
