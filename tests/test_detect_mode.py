"""'When Not Aiming': one choice for keep_detecting + idle_detect_enabled (core/detect_mode.py)."""
from types import SimpleNamespace as NS

import pytest

from core import detect_mode as dm


@pytest.mark.parametrize("keep,idle,want", [
    (True, False, dm.FULL), (True, True, dm.REDUCED),
    (False, False, dm.STOP), (False, True, dm.STOP),
])
def test_every_saved_combination_maps_to_one_choice(keep, idle, want):
    assert dm.mode_from_config(NS(keep_detecting=keep, idle_detect_enabled=idle)) == want


def test_defaults_of_a_bare_config_are_full_speed():
    assert dm.mode_from_config(NS()) == dm.FULL


@pytest.mark.parametrize("mode", dm.MODES)
def test_apply_round_trips(mode):
    for keep in (True, False):
        for idle in (True, False):
            c = NS(keep_detecting=keep, idle_detect_enabled=idle)
            assert dm.apply_mode(c, mode) == mode
            assert dm.mode_from_config(c) == mode


def test_stop_keeps_the_users_reduced_rate_preference():
    c = NS(keep_detecting=True, idle_detect_enabled=True)
    dm.apply_mode(c, dm.STOP)
    assert c.keep_detecting is False and c.idle_detect_enabled is True
    dm.apply_mode(c, dm.REDUCED)
    assert (c.keep_detecting, c.idle_detect_enabled) == (True, True)
    dm.apply_mode(c, dm.FULL)
    assert (c.keep_detecting, c.idle_detect_enabled) == (True, False)


def test_unknown_mode_means_full_speed():
    c = NS(keep_detecting=False, idle_detect_enabled=True)
    assert dm.apply_mode(c, "bogus") == dm.FULL
    assert (c.keep_detecting, c.idle_detect_enabled) == (True, False)


@pytest.mark.parametrize("always_aim,always_fire,ignored", [
    (False, False, False), (True, False, True), (False, True, True), (True, True, True)])
def test_reduced_rate_note_only_when_it_cannot_apply(always_aim, always_fire, ignored):
    c = NS(keep_detecting=True, idle_detect_enabled=True, always_aim=always_aim, always_auto_fire=always_fire)
    assert dm.reduced_rate_ignored(c) is ignored
    c.idle_detect_enabled = False                       # full speed: nothing to ignore
    assert dm.reduced_rate_ignored(c) is False


def test_matches_what_ai_loop_does():
    """ai_loop.py: idle is off whenever always-aim / always-auto-fire is on, else the saved switch."""
    def runtime_idle(c):
        enabled = getattr(c, "idle_detect_enabled", True)
        if getattr(c, "always_aim", False) or getattr(c, "always_auto_fire", False):
            enabled = False
        return enabled
    c = NS(keep_detecting=True, idle_detect_enabled=True, always_aim=True, always_auto_fire=False)
    assert runtime_idle(c) is False and dm.reduced_rate_ignored(c) is True


class TestWebDetectModeField:
    """The web client sets it through the derived `detect_mode` schema field."""

    def test_read_write_and_legacy_aliases(self):
        from core import web_control_settings as wcs
        c = NS(keep_detecting=True, idle_detect_enabled=False, always_aim=False, always_auto_fire=False)
        assert wcs.get_tab_settings(c, "inference")["detect_mode"] == "full"
        assert wcs.apply_tab_settings(c, "inference", {"detect_mode": "reduced"})["ok"] is True
        assert (c.keep_detecting, c.idle_detect_enabled) == (True, True)
        assert wcs.apply_tab_settings(c, "inference", {"detect_mode": "stop"})["ok"] is True
        assert wcs.get_tab_settings(c, "inference")["detect_mode"] == "stop"
        # the two old fields still accept writes from existing scripts
        assert wcs.apply_tab_settings(c, "inference", {"keep_detecting": True, "idle_detect_enabled": False})["ok"]
        assert wcs.get_tab_settings(c, "inference")["detect_mode"] == "full"

    def test_note_flag_follows_always_aim_and_auto_fire(self):
        from core import web_control_settings as wcs
        c = NS(keep_detecting=True, idle_detect_enabled=True, always_aim=False, always_auto_fire=True)
        assert wcs.get_tab_settings(c, "inference")["reduced_rate_ignored"] is True
        c.always_auto_fire = False
        assert wcs.get_tab_settings(c, "inference")["reduced_rate_ignored"] is False

    def test_bad_choice_is_rejected_without_writing(self):
        from core import web_control_settings as wcs
        c = NS(keep_detecting=True, idle_detect_enabled=False)
        assert wcs.apply_tab_settings(c, "inference", {"detect_mode": "bogus"})["ok"] is False
        assert (c.keep_detecting, c.idle_detect_enabled) == (True, False)
