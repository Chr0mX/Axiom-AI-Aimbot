"""FOV mode: one choice for fov_reduce_on_target_enabled + fov_shrink_mode (core/fov_mode.py)."""
from types import SimpleNamespace as NS

import pytest

from core import fov_mode as fm


@pytest.mark.parametrize("on,saved,want", [
    (False, "fixed", fm.OFF), (False, "adaptive", fm.OFF), (False, "dynamic", fm.OFF),
    (True, "fixed", fm.FIXED), (True, "adaptive", fm.ADAPTIVE), (True, "dynamic", fm.DYNAMIC),
    (True, "anchor", fm.ANCHOR), (False, "anchor", fm.ANCHOR),          # Anchor never needed the switch
    (True, "bogus", fm.ADAPTIVE),
])
def test_every_saved_combination_maps_to_one_choice(on, saved, want):
    assert fm.mode_from_config(NS(fov_reduce_on_target_enabled=on, fov_shrink_mode=saved)) == want


def test_bare_config_is_off():
    assert fm.mode_from_config(NS()) == fm.OFF


@pytest.mark.parametrize("mode", fm.MODES)
def test_apply_round_trips_from_every_starting_point(mode):
    for on in (True, False):
        for saved in ("fixed", "adaptive", "dynamic", "anchor"):
            c = NS(fov_reduce_on_target_enabled=on, fov_shrink_mode=saved)
            assert fm.apply_mode(c, mode) == mode
            assert fm.mode_from_config(c) == mode


def test_off_keeps_the_saved_shrink_mode_so_switching_back_restores_it():
    c = NS(fov_reduce_on_target_enabled=True, fov_shrink_mode="dynamic")
    fm.apply_mode(c, fm.OFF)
    assert c.fov_reduce_on_target_enabled is False and c.fov_shrink_mode == "dynamic"
    fm.apply_mode(c, fm.DYNAMIC)
    assert c.fov_reduce_on_target_enabled is True and c.fov_shrink_mode == "dynamic"


def test_off_from_anchor_falls_back_to_adaptive_instead_of_staying_anchor():
    c = NS(fov_reduce_on_target_enabled=True, fov_shrink_mode="anchor")
    fm.apply_mode(c, fm.OFF)
    assert c.fov_shrink_mode == "adaptive" and fm.mode_from_config(c) == fm.OFF


def test_anchor_does_not_touch_the_switch():
    c = NS(fov_reduce_on_target_enabled=False, fov_shrink_mode="adaptive")
    fm.apply_mode(c, fm.ANCHOR)
    assert c.fov_reduce_on_target_enabled is False and c.fov_shrink_mode == "anchor"


def test_unknown_mode_means_off():
    c = NS(fov_reduce_on_target_enabled=True, fov_shrink_mode="fixed")
    assert fm.apply_mode(c, "nonsense") == fm.OFF


def test_shrinking_only_for_the_three_shrink_modes():
    assert fm.shrinking(NS(fov_reduce_on_target_enabled=True, fov_shrink_mode="fixed")) is True
    assert fm.shrinking(NS(fov_reduce_on_target_enabled=True, fov_shrink_mode="anchor")) is False
    assert fm.shrinking(NS(fov_reduce_on_target_enabled=False, fov_shrink_mode="fixed")) is False
