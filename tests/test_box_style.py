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
