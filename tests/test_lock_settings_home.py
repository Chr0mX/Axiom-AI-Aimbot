"""One home for the lock timing settings: the Aim page / Aim tab, next to Sticky Target Lock.

Min Lock Duration, Tracking Confidence, Target Switch Threshold, Reacquisition Radius and Lost Target
Timeout (and Gap Extrapolation) used to be on the Inference page. Qt pages can't be imported here (no PyQt6),
so the table that drives them is read from the source with `ast`.
"""
import ast
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
AIM_PAGE = SRC / "gui" / "fluent_app" / "pages" / "aim_page.py"
INFERENCE_PAGE = SRC / "gui" / "fluent_app" / "pages" / "inference_page.py"
INDEX = SRC / "web_control_client" / "index.html"
ENGLISH = SRC / "core" / "language_data" / "English_English.json"

LOCK_FIELDS = ("lost_target_timeout_ms", "min_lock_duration_ms", "target_switch_threshold",
               "reacquire_radius_px", "tracking_confidence")


def _lock_table():
    tree = ast.parse(AIM_PAGE.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "_LOCK_CARDS" for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError("_LOCK_CARDS not found in aim_page.py")


def test_table_covers_the_five_lock_fields():
    assert tuple(row[8] for row in _lock_table()) == LOCK_FIELDS


@pytest.mark.parametrize("row", _lock_table(), ids=lambda r: r[0])
def test_every_row_is_a_real_persisted_field_with_a_matching_default(row):
    from core.config import _FIELD_MAP, Config
    from unittest.mock import patch
    attr, key, title, desc, icon, lo, hi, suffix, field, scale, default = row
    assert field in _FIELD_MAP
    with patch("core.config._get_screen_size", return_value=(1920, 1080)):
        cfg = Config()
    assert getattr(cfg, field) == pytest.approx(default)
    assert lo <= round(default * scale) <= hi
    assert scale in (1, 100)


def test_english_pack_matches_the_table_defaults():
    """One default text per key everywhere (tests/test_i18n_keys.py checks literals; the table is data)."""
    pack = json.loads(ENGLISH.read_text(encoding="utf-8"))
    for attr, key, title, desc, *_ in _lock_table():
        assert pack[key] == title, key
        assert pack[key + "_desc"] == desc, key


def test_inference_page_no_longer_builds_the_lock_cards():
    src = INFERENCE_PAGE.read_text(encoding="utf-8")
    for needle in ("minLockCard", "trackConfCard", "switchThresholdCard", "reacquireRadiusCard",
                   "lostTimeoutCard", "acqTrackGroup", "min_lock_duration_ms", "lost_target_timeout_ms"):
        assert needle not in src, needle


def test_gap_extrapolation_sits_with_the_lock_cards():
    src = AIM_PAGE.read_text(encoding="utf-8")
    block = re.search(r"self\.stickyGapCard = SwitchSettingCard\(.*?\n        \)\n", src, re.S).group(0)
    assert "parent=self.trackingGroup" in block
    assert "self.trackingGroup.addSettingCard(self.stickyGapCard)" in src


def test_lock_cards_are_advanced_and_follow_sticky_lock():
    src = AIM_PAGE.read_text(encoding="utf-8")
    for attr in ("lostTimeoutCard", "minLockCard", "switchThresholdCard", "reacquireRadiusCard", "trackConfCard"):
        assert attr in _lock_table().__repr__()
    assert "lostTimeoutCard minLockCard switchThresholdCard reacquireRadiusCard trackConfCard" in src
    vis = src[src.index("def _updateAimTuningVisibility"):]
    assert "for _attr, *_rest in _LOCK_CARDS:" in vis and "setVisible(sticky_on)" in vis


def test_web_page_has_the_cards_on_the_aim_tab_only():
    html = INDEX.read_text(encoding="utf-8")
    for f in LOCK_FIELDS + ("sticky_gap_extrapolate", "sticky_gap_frames"):
        assert 'id="aim-%s"' % f in html, f
        assert 'id="inf-%s"' % f not in html, f


def test_web_order_follows_the_plan():
    html = INDEX.read_text(encoding="utf-8")
    order = ["sticky_lock_enabled", "lock_persistence_ms", "lost_target_timeout_ms", "min_lock_duration_ms",
             "target_switch_threshold", "reacquire_radius_px", "tracking_confidence", "lock_iou_threshold",
             "sticky_gap_extrapolate", "sticky_gap_frames"]
    pos = [html.index('id="aim-%s"' % f) for f in order]
    assert pos == sorted(pos)


@pytest.mark.parametrize("page", [AIM_PAGE, INFERENCE_PAGE], ids=lambda p: p.name)
def test_cards_are_built_before_they_are_added_to_a_group(page):
    """Moving a card between groups once added `self.stickyGapCard` to a group before the line that builds it,
    which raises AttributeError when the page is constructed (Qt can't run here, so check the source order)."""
    tree = ast.parse(page.read_text(encoding="utf-8"))
    for fn in (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)):
        assigned = {}
        for node in ast.walk(fn):
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id == "self":
                        assigned.setdefault(t.attr, node.lineno)
        for node in ast.walk(fn):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr in ("addSettingCard", "addContent")):
                for a in node.args:
                    if isinstance(a, ast.Attribute) and isinstance(a.value, ast.Name) and a.value.id == "self" \
                            and a.attr in assigned:
                        assert assigned[a.attr] < node.lineno, (page.name, fn.name, a.attr, node.lineno)
