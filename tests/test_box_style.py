"""Detection-box line style: thickness and corner-bracket length."""
from types import SimpleNamespace

from core.box_style import box_thickness, corner_length


def cfg(**kw):
    return SimpleNamespace(**kw)


def test_auto_thickness_matches_the_original_confidence_rule():
    c = cfg(box_line_thickness=0)
    assert box_thickness(c, 0.0) == 1
    assert box_thickness(c, 0.5) == 2
    assert box_thickness(c, 1.0) == 3


def test_fixed_thickness_ignores_confidence_and_is_capped():
    assert box_thickness(cfg(box_line_thickness=5), 0.1) == 5
    assert box_thickness(cfg(box_line_thickness=99), 0.9) == 10


def test_missing_field_behaves_as_auto():
    assert box_thickness(SimpleNamespace(), 1.0) == 3
    assert corner_length(SimpleNamespace(), 100, 200) == 15


def test_auto_corner_length_matches_the_original_rule():
    c = cfg(box_corner_length_pct=0)
    assert corner_length(c, 100, 300) == 15   # 15% of the shorter side
    assert corner_length(c, 20, 20) == 6      # floor
    assert corner_length(c, 400, 400) == 24   # ceiling


def test_percentage_corner_length_scales_and_never_overlaps():
    assert corner_length(cfg(box_corner_length_pct=30), 100, 300) == 30
    assert corner_length(cfg(box_corner_length_pct=50), 40, 40) == 20  # half the side
    assert corner_length(cfg(box_corner_length_pct=5), 10, 10) == 2    # minimum


def test_box_style_fields_are_persisted_and_clamped():
    from core.config import _FIELD_MAP, _validate_box_style
    assert _FIELD_MAP["box_line_thickness"] == "display.box_line_thickness"
    assert _FIELD_MAP["box_corner_length_pct"] == "display.box_corner_length_pct"
    c = cfg(box_line_thickness=42, box_corner_length_pct=-3)
    _validate_box_style(c)
    assert (c.box_line_thickness, c.box_corner_length_pct) == (10, 0)
    c = cfg(box_line_thickness="junk", box_corner_length_pct=None)
    _validate_box_style(c)
    assert (c.box_line_thickness, c.box_corner_length_pct) == (0, 0)


def test_web_control_schema_exposes_both_fields():
    from core.web_control_settings import _SCHEMA
    vis = _SCHEMA["visuals"]
    assert vis["box_line_thickness"]["max"] == 10
    assert vis["box_corner_length_pct"]["max"] == 50


def test_highlight_outline_follows_a_fixed_thickness_only():
    from core.box_style import highlight_thickness
    assert highlight_thickness(cfg(box_line_thickness=0), 1) == 1
    assert highlight_thickness(cfg(box_line_thickness=7), 1) == 7
    assert highlight_thickness(cfg(box_line_thickness=50), 2) == 10


# ── FOV / Detection Range / Tracer / Predicted Aim Point line widths ───────
import pytest

from core import box_style as bs

LINE_FIELDS = {
    "fov_line_thickness": (bs.fov_thickness, 2),
    "detect_range_line_thickness": (bs.detect_range_thickness, 1),
    "tracer_line_thickness": (bs.tracer_thickness, 2),
    "aim_marker_line_thickness": (bs.aim_marker_thickness, 2),
}


@pytest.mark.parametrize("attr", LINE_FIELDS)
def test_defaults_are_the_widths_the_overlays_always_drew(attr):
    fn, default = LINE_FIELDS[attr]
    assert fn(SimpleNamespace()) == default            # config predating the field
    assert fn(cfg(**{attr: default})) == default


@pytest.mark.parametrize("attr", LINE_FIELDS)
@pytest.mark.parametrize("raw,want", [(1, 1), (5, 5), (10, 10), (99, 10)])
def test_line_widths_are_clamped_to_10(attr, raw, want):
    fn, _ = LINE_FIELDS[attr]
    assert fn(cfg(**{attr: raw})) == want


@pytest.mark.parametrize("attr", LINE_FIELDS)
@pytest.mark.parametrize("raw", [0, -3, None, "junk", float("nan")])
def test_unusable_values_fall_back_to_the_default(attr, raw):
    fn, default = LINE_FIELDS[attr]
    assert fn(cfg(**{attr: raw})) == default


def test_line_width_fields_are_persisted_validated_and_exposed():
    from core.config import _FIELD_MAP, _validate_box_style
    from core.web_control_settings import _SCHEMA
    for attr, (_, default) in LINE_FIELDS.items():
        assert _FIELD_MAP[attr] == f"display.{attr}"
        assert _SCHEMA["visuals"][attr] == {"type": "int", "min": 1, "max": 10}
        c = cfg(**{attr: 99})
        _validate_box_style(c)
        assert getattr(c, attr) == 10
        c = cfg(**{attr: "junk"})
        _validate_box_style(c)
        assert getattr(c, attr) == default
        c = cfg(**{attr: 0})
        _validate_box_style(c)
        assert getattr(c, attr) == default             # no "Auto" mode: 0 means unset -> default


def test_old_config_without_the_fields_loads_with_the_old_look():
    from unittest.mock import patch
    with patch("core.config._get_screen_size", return_value=(1920, 1080)):
        from core.config import Config
        c = Config()
    before = c.to_dict()
    data = c.to_dict()
    for attr in LINE_FIELDS:
        data["display"].pop(attr, None)
    with patch("core.config._get_screen_size", return_value=(1920, 1080)):
        c2 = Config()
    c2.from_dict(data)
    for attr, (_, default) in LINE_FIELDS.items():
        assert getattr(c2, attr) == default
    assert c2.to_dict() == before
