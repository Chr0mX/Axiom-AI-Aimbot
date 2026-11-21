"""Static checks on which cards each page puts behind "Show advanced settings"
(no Qt needed: the pages' source is parsed)."""
import pathlib
import re

import pytest

PAGES = pathlib.Path(__file__).resolve().parents[1] / "src/gui/fluent_app/pages"
PAGE_FILES = ["aim_page", "inference_page", "capture_page", "keys_page", "visuals_page", "other_page"]

# Needed to understand or troubleshoot the app: status/readout rows, warnings, the controls
# that connect a device or turn a feature on. Never advanced.
NEVER_ADVANCED = {
    "trtStatusCard", "trtVersionCard", "dmlStatusCard", "dmlDllCard", "makcuHwStatusCard",
    "makcuAimStatusCard", "makcuConnectionCard", "makcuConnectBtnCard", "keyConflictCard",
    "uvcHwInfoCard", "ndiHwInfoCard", "ocrResultCard", "webControlEnableCard", "webControlTokenCard",
    "webControlConnectCard", "webEspEnableCard", "webEspConnectCard", "aimKey1Card", "toggleKeyCard",
    "alwaysAimCard", "predictionEnableCard", "stickyLockCard", "anchorEnableCard", "anchorModeCard",
    "anchorHorizonCard", "screenshotMethodCard", "mouseMoveCard", "aimPartCard", "fovCard", "fovHeightCard",
    "confidenceCard", "detectModeCard", "pidPxCard", "pidIxCard", "pidDxCard", "pidPyCard", "pidIyCard",
    "pidDyCard", "languageCard", "exitSaveCard", "showConsoleCard",
}


def advanced_names(page):
    src = (PAGES / f"{page}.py").read_text(encoding="utf-8")
    m = re.search(r'self\.markAdvancedAttrs\(\*"""(.*?)""".split\(\)\)', src, flags=re.S)
    assert m, f"{page}: no _markAdvancedCards() list"
    return src, m.group(1).split()


@pytest.mark.parametrize("page", PAGE_FILES)
def test_every_listed_card_exists_and_is_listed_once(page):
    src, names = advanced_names(page)
    assert names, page
    assert len(names) == len(set(names)), f"{page}: duplicates"
    for n in names:
        assert re.search(rf"self\.{n}\s*=", src) or f'"{n}"' in src, f"{page}: {n} is never created"


@pytest.mark.parametrize("page", PAGE_FILES)
def test_status_rows_and_core_controls_are_never_advanced(page):
    _, names = advanced_names(page)
    assert not (set(names) & NEVER_ADVANCED), set(names) & NEVER_ADVANCED


@pytest.mark.parametrize("page", PAGE_FILES)
def test_no_listed_card_is_toggled_with_show_hide_or_sethidden(page):
    """The gate shadows setVisible; show()/hide()/setHidden() would bypass it."""
    src, names = advanced_names(page)
    for n in names:
        assert not re.search(rf"self\.{n}\.(show|hide|setHidden)\(", src), f"{page}: {n}"


@pytest.mark.parametrize("page", PAGE_FILES)
def test_the_page_builds_the_switch_after_registering(page):
    src = (PAGES / f"{page}.py").read_text(encoding="utf-8")
    assert src.index("self._markAdvancedCards()") < src.index("self.addAdvancedSwitch()")


def test_other_pages_without_advanced_cards_have_no_switch():
    for page in ("model_page", "trigger_page", "configs_page", "convert_page"):
        assert "addAdvancedSwitch" not in (PAGES / f"{page}.py").read_text(encoding="utf-8")
