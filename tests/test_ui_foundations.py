"""Pure (Qt-free) rules behind the glass interface: accent palette, motion table, column layout,
and the new persisted look fields (accent colour / animations / nav dock)."""
import json
import os
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from core import ui_accent, ui_layout, ui_motion  # noqa: E402


class TestAccent:
    def test_parse_hex_variants_and_fallback(self):
        assert ui_accent.parse_hex("#7c6cf0") == (124, 108, 240)
        assert ui_accent.parse_hex("7C6CF0") == (124, 108, 240)
        assert ui_accent.parse_hex("nonsense") == ui_accent.parse_hex(ui_accent.DEFAULT_ACCENT)
        assert ui_accent.parse_hex(None) == ui_accent.parse_hex(ui_accent.DEFAULT_ACCENT)

    def test_to_hex_round_trip_and_clamp(self):
        assert ui_accent.to_hex((124, 108, 240)) == "#7c6cf0"
        assert ui_accent.to_hex((-5, 300, 0)) == "#00ff00"

    def test_palette_is_lighter_hi_darker_lo(self):
        for _label, hx in ui_accent.PRESET_ACCENTS:
            pal = ui_accent.derive(hx)
            assert ui_accent.luminance(pal.hi) > ui_accent.luminance(pal.base) > ui_accent.luminance(pal.lo)
            assert pal.base == ui_accent.parse_hex(hx)

    def test_text_on_accent_is_readable(self):
        assert ui_accent.derive("#101030").on_accent == (255, 255, 255)
        assert ui_accent.derive("#ffe066").on_accent != (255, 255, 255)

    def test_glow_is_stronger_in_dark_mode(self):
        assert ui_accent.derive("#7c6cf0", dark=True).glow[3] > ui_accent.derive("#7c6cf0", dark=False).glow[3]

    def test_set_accent_normalises_and_stores(self):
        try:
            assert ui_accent.set_accent("E05FB0") == "#e05fb0"
            assert ui_accent.get_accent() == "#e05fb0"
            assert ui_accent.set_accent("bad") == ui_accent.DEFAULT_ACCENT
        finally:
            ui_accent.set_accent(ui_accent.DEFAULT_ACCENT)


class TestMotion:
    def teardown_method(self):
        ui_motion.set_animations_enabled(True)

    def test_duration_follows_the_switch(self):
        assert ui_motion.duration(ui_motion.DUR) == ui_motion.DUR
        ui_motion.set_animations_enabled(False)
        assert ui_motion.duration(ui_motion.DUR) == 0
        assert ui_motion.animations_enabled() is False

    def test_easing_is_monotone_and_bounded(self):
        prev = -1.0
        for i in range(0, 101):
            v = ui_motion.ease_out_cubic(i / 100)
            assert 0.0 <= v <= 1.0 and v >= prev
            prev = v
        assert ui_motion.ease_out_cubic(-1) == 0.0 and ui_motion.ease_out_cubic(2) == 1.0
        assert ui_motion.ease_in_out_cubic(0.5) == pytest.approx(0.5)

    def test_lerp_color_blends_every_channel(self):
        assert ui_motion.lerp_color((0, 0, 0, 0), (200, 100, 50, 255), 0.5) == (100, 50, 25, 128)
        assert ui_motion.lerp_color((10, 20, 30), (10, 20, 30), 0.7) == (10, 20, 30)

    def test_durations_are_ordered(self):
        assert ui_motion.DUR_FAST < ui_motion.DUR < ui_motion.DUR_SLOW


class TestColumns:
    def test_columns_for_width(self):
        assert ui_layout.columns_for_width(600) == 1
        assert ui_layout.columns_for_width(900) == 2          # (900-20)/2 = 440
        assert ui_layout.columns_for_width(899) == 1
        assert ui_layout.columns_for_width(2000, max_columns=3) == 3
        assert ui_layout.columns_for_width(0) == 1

    def test_assign_balances_heights(self):
        cols = ui_layout.assign_columns([300, 100, 100, 100, 200], 2)
        totals = [0, 0]
        for c, h in zip(cols, [300, 100, 100, 100, 200]):
            totals[c] += h
        assert abs(totals[0] - totals[1]) <= 300   # greedy: never worse than the tallest item
        assert cols[0] == 0                                     # first item opens the left column

    def test_single_column_and_empty(self):
        assert ui_layout.assign_columns([5, 5, 5], 1) == [0, 0, 0]
        assert ui_layout.assign_columns([], 2) == []

    def test_hidden_groups_do_not_break_the_order(self):
        assert ui_layout.assign_columns([0, 0, 50, 50], 2) == [0, 0, 0, 1]

    def test_split_bands(self):
        assert ui_layout.split_bands(["group", "group", "span", "group"]) == [[0, 1], [2], [3]]
        assert ui_layout.split_bands(["span", "span"]) == [[0], [1]]
        assert ui_layout.split_bands([]) == []


@pytest.fixture
def cfgmod():
    from core import config as c
    return c


def _new_config(cfgmod):
    with patch.object(cfgmod, "_get_screen_size", return_value=(1920, 1080)):
        return cfgmod.Config()


class TestLookFields:
    def test_defaults(self, cfgmod):
        c = _new_config(cfgmod)
        assert (c.accent_color, c.ui_animations, c.nav_dock) == ("#7c6cf0", True, "top")

    def test_fields_are_persisted_under_ui(self, cfgmod):
        assert cfgmod._FIELD_MAP["accent_color"] == "ui.accent_color"
        assert cfgmod._FIELD_MAP["ui_animations"] == "ui.animations"
        assert cfgmod._FIELD_MAP["nav_dock"] == "ui.nav_dock"

    def test_round_trip(self, cfgmod):
        c = _new_config(cfgmod)
        c.accent_color, c.ui_animations, c.nav_dock = "#e05fb0", False, "left"
        data = json.loads(json.dumps(c.to_dict()))
        assert data["ui"]["accent_color"] == "#e05fb0" and data["ui"]["nav_dock"] == "left"
        d = _new_config(cfgmod)
        d.from_dict(data)
        cfgmod.validate_config(d)
        assert (d.accent_color, d.ui_animations, d.nav_dock) == ("#e05fb0", False, "left")

    @pytest.mark.parametrize("bad", ["", "red", "#12345", "#gggggg", None, 7])
    def test_bad_accent_falls_back(self, cfgmod, bad):
        c = _new_config(cfgmod)
        c.accent_color = bad
        cfgmod.validate_config(c)
        assert c.accent_color == cfgmod.DEFAULT_ACCENT_COLOR

    def test_accent_without_hash_is_normalised(self, cfgmod):
        assert cfgmod.normalize_accent_color("E05FB0") == "#e05fb0"

    @pytest.mark.parametrize("bad", ["bottom", "", None, 3])
    def test_bad_dock_falls_back(self, cfgmod, bad):
        c = _new_config(cfgmod)
        c.nav_dock = bad
        cfgmod.validate_config(c)
        assert c.nav_dock == "top"

    def test_old_config_without_the_keys_loads_with_defaults(self, cfgmod):
        c = _new_config(cfgmod)
        c.from_dict({"config_version": 3, "ui": {"dark_mode": True}})
        cfgmod.validate_config(c)
        assert c.dark_mode is True and (c.accent_color, c.ui_animations, c.nav_dock) == ("#7c6cf0", True, "top")


class TestStatusPillLogic:
    def test_cpu_formatting(self):
        from core import ui_status as s
        assert s.format_cpu(31.4) == "31%" and s.format_cpu(0) == "0%" and s.format_cpu(150) == "100%"
        assert s.format_cpu(None) == "—" and s.format_cpu("x") == "—" and s.format_cpu(float("nan")) == "—"

    def test_slots_are_padded_and_truncated(self):
        from core import ui_status as s
        assert s.normalize_slots(["a"]) == ["a", "", "", "", ""]
        assert s.normalize_slots(None) == [""] * 5
        assert len(s.normalize_slots(list("abcdefgh"))) == 5

    def test_active_and_assigned_flags(self):
        from core import ui_status as s
        st = s.slot_states(["Fast", "", "Slow"], active_name="Slow")
        assert [x["assigned"] for x in st] == [True, False, True, False, False]
        assert [x["active"] for x in st] == [False, False, True, False, False]
        assert st[0]["label"] == "1" and st[4]["index"] == 4

    def test_unassigned_slot_is_never_active(self):
        from core import ui_status as s
        assert not any(x["active"] for x in s.slot_states(["", ""], active_name=""))

    def test_active_label_priority_and_truncation(self):
        from core import ui_status as s
        assert s.active_label("P", "C") == "P" and s.active_label("", "C") == "C"
        assert s.active_label("", "") == "None"
        long = s.active_label("x" * 40)
        assert len(long) == 22 and long.endswith("…")
