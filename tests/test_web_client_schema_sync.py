"""The Web Control page (index.html) and the server schema (web_control_settings._SCHEMA) must agree.

Static, no browser: every `data-key` control has a server field in its own tab, and the
`data-advanced` markers of the "Show advanced settings" switch only name real fields.
"""
import pathlib
import re

import pytest

from core import web_control_settings as wcs

CLIENT = pathlib.Path(__file__).resolve().parents[1] / "src" / "web_control_client"
HTML = (CLIENT / "index.html").read_text(encoding="utf-8")
JS = (CLIENT / "app.js").read_text(encoding="utf-8")
CSS = (CLIENT / "styles.css").read_text(encoding="utf-8")


def controls():
    """(tab, key, tag_text) for every element with a data-key, by the panel that contains it."""
    panels = [(m.start(), m.group(1)) for m in re.finditer(r'<div class="panel[^"]*" id="panel-([a-z]+)"', HTML)]
    out = []
    for m in re.finditer(r"<(?:input|select|textarea|button)\b[^>]*\bdata-key=\"([^\"]+)\"[^>]*>", HTML):
        tab = [name for pos, name in panels if pos < m.start()][-1]
        out.append((tab, m.group(1), m.group(0)))
    return out


CONTROLS = controls()


def test_the_scan_found_the_page():
    assert len(CONTROLS) > 150
    assert {t for t, _, _ in CONTROLS} >= {"model", "capture", "inference", "aim", "keys", "visuals", "trigger"}


@pytest.mark.parametrize("tab,key", sorted({(t, k) for t, k, _ in CONTROLS}))
def test_every_control_has_a_server_field_in_its_tab(tab, key):
    assert key in wcs._SCHEMA.get(tab, {}), f"{tab}: data-key={key} is not in the server schema"


def test_advanced_markers_name_real_controls():
    marked = [(t, k) for t, k, tag in CONTROLS if "data-advanced" in tag]
    assert len(marked) >= 60
    for tab, key in marked:
        assert key in wcs._SCHEMA[tab], (tab, key)


def test_the_switch_is_a_server_field_and_not_a_generic_data_key():
    assert wcs._SCHEMA["visuals"]["show_advanced_settings"]["type"] == "bool"
    assert 'id="adv-toggle"' in HTML
    assert not any(k == "show_advanced_settings" for _, k, _ in CONTROLS)   # it is wired by hand, once, in the sidebar


def test_status_carries_the_flag_the_client_reads():
    server = (pathlib.Path(__file__).resolve().parents[1] / "src/core/web_control_server.py").read_text(encoding="utf-8")
    assert '"show_advanced_settings"' in server and "s.show_advanced_settings" in JS


def test_asset_versions_move_together():
    css_v = re.search(r"--axiom-css:\s*(\d+);", CSS).group(1)
    assert f"styles.css?v={css_v}" in HTML and f"app.js?v={css_v}" in HTML
    assert f'cssVer !== "{css_v}"' in JS


def test_advanced_cards_hide_by_css_not_by_the_conditional_hidden_class():
    assert "body:not(.show-advanced) .main .field-card[data-advanced]" in CSS
    assert 'classList.toggle("show-advanced"' in JS


def test_number_bounds_in_the_page_match_the_schema():
    """min/max on each number input are the schema's bounds in display units (the server clamps to
    them, so a different attribute would let the page offer values the host silently changes)."""
    problems = []
    for tab, key, tag in CONTROLS:
        if 'type="number"' not in tag:
            continue
        spec = wcs._SCHEMA[tab][key]
        for attr in ("min", "max"):
            want = spec.get(attr)
            got = re.search(rf'\b{attr}="([^"]+)"', tag)
            if want is None:
                continue
            if got is None or abs(float(got.group(1)) - float(want)) > 1e-9:
                problems.append((tab, key, attr, got.group(1) if got else None, want))
    assert not problems, problems


# Fields the Qt app shows as a percentage (the saved value is a fraction): the page must use the same unit.
PERCENT_IN_QT = {
    "aim": ["head_width_ratio", "head_height_ratio", "body_width_ratio", "target_priority_confidence_weight",
            "aim_ema_smoothing", "aim_x_offset_frac", "aim_y_offset_frac",
            "humanization.micro_jitter_scale", "humanization.motion_variation_range"],
}


@pytest.mark.parametrize("tab,key", [(t, k) for t, ks in PERCENT_IN_QT.items() for k in ks])
def test_percent_fields_use_the_apps_unit(tab, key):
    from types import SimpleNamespace as NS
    spec = wcs._SCHEMA[tab][key]
    assert spec["scale"] == 100.0
    cfg = NS(head_width_ratio=0.07, head_height_ratio=0.07, body_width_ratio=0.07, target_priority_confidence_weight=0.07,
             aim_ema_smoothing=0.07, aim_x_offset_frac=0.07, aim_y_offset_frac=0.07,
             humanization=NS(micro_jitter_scale=0.07, motion_variation_range=0.07))
    # 0.07 * 100 must read exactly 7.0, not 7.000000000000001
    assert wcs.get_tab_settings(cfg, tab)[key] == 7.0


def test_percent_round_trip_stores_the_fraction():
    from types import SimpleNamespace as NS
    cfg = NS(head_width_ratio=0.3)
    res = wcs.apply_tab_settings(cfg, "aim", {"head_width_ratio": 45})
    assert res["ok"] and cfg.head_width_ratio == pytest.approx(0.45)
    res = wcs.apply_tab_settings(cfg, "aim", {"head_width_ratio": 5})          # below the app's 10 % minimum
    assert cfg.head_width_ratio == pytest.approx(0.10)


def test_every_control_outside_a_field_card_has_an_accessible_name():
    """Controls inside a .field-card are named at runtime from the card's title (initUiEnhancements); every
    other interactive control — sidebar, decks, toolbars — must carry its own name in the markup."""
    from html.parser import HTMLParser

    problems = []

    class Scan(HTMLParser):
        def __init__(self):
            super().__init__()
            self.stack, self.labels_for, self.controls, self.btn_text = [], set(), [], []
            self.open_btn = None

        def handle_starttag(self, tag, attrs):
            a = dict(attrs)
            in_card = any(t == "div" and "field-card" in (c or "") for t, c in self.stack)
            if tag == "label" and a.get("for"):
                self.labels_for.add(a["for"])
            if tag in ("input", "select", "textarea", "button") and a.get("type") not in ("hidden", "file", "radio"):
                self.controls.append((tag, a, in_card))
                if tag == "button":
                    self.open_btn = len(self.controls) - 1
                    self.btn_text.append("")
            if tag not in ("input", "br", "img", "meta", "link"):
                self.stack.append((tag, a.get("class")))

        def handle_endtag(self, tag):
            if tag == "button":
                self.open_btn = None
            for i in range(len(self.stack) - 1, -1, -1):
                if self.stack[i][0] == tag:
                    del self.stack[i:]
                    break

        def handle_data(self, data):
            if self.open_btn is not None:
                self.btn_text[-1] += data

    scan = Scan()
    scan.feed(HTML)
    btn_i = 0
    for tag, a, in_card in scan.controls:
        text = ""
        if tag == "button":
            text = scan.btn_text[btn_i].strip()
            btn_i += 1
        if in_card or a.get("data-key"):
            continue
        named = a.get("aria-label") or a.get("aria-labelledby") or a.get("title") or a.get("placeholder") or text \
            or a.get("id") in scan.labels_for
        if not named:
            problems.append((tag, a.get("id") or a.get("name")))
    assert not problems, problems
