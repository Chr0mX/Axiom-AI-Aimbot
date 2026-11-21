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
    assert len(marked) >= 80
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
