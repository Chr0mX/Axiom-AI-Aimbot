"""Enforcement for the GUI invariants CLAUDE.md documents as recurring bug
sources.

Both are rules a reviewer has to remember, with nothing checking them —
which is precisely why they keep getting broken. These tests are static
source analysis: they parse the GUI files as text, so they need no PyQt6,
no display, and no numpy, and therefore actually run in CI and in the
sandbox where the rest of the GUI suite cannot.

Both are pinned to a **baseline of known existing violations** rather than
asserting zero. That is deliberate. The current tree has 71 untranslated
keys and 38 hardcoded colours; auditing each one to decide whether it is a
genuine bug or a legitimate exception (dialog text, one-shot status
strings, deliberately theme-independent chrome) requires running the GUI,
which cannot be done here. Locking the numbers stops the counts growing
while leaving the existing backlog visible and reducible — and a test that
fails the moment someone adds violation number 72 is worth far more than
one that has been commented out because it was red on arrival.

**Lowering a baseline when you fix something is expected**, and the failure
message says so.
"""
import re
from pathlib import Path

import pytest

_GUI = Path(__file__).resolve().parent.parent / "src" / "gui"
_PAGES = _GUI / "fluent_app" / "pages"

# --- Baselines -------------------------------------------------------------
# Per-file counts of *existing* violations at the time this test was added.
# Fix violations and lower the number; never raise one to make a test pass.

# Keys passed to t() while building widgets that are never re-applied in the
# page's retranslateUi(), so the text goes stale after the first language
# switch. See CLAUDE.md's i18n section.
_STALE_TRANSLATION_BASELINE = {
    "aim_page.py": 20,
    "capture_page.py": 0,
    "configs_page.py": 13,
    "convert_page.py": 6,
    "inference_page.py": 0,
    "keys_page.py": 7,
    "model_page.py": 5,
    "other_page.py": 19,
    "trigger_page.py": 0,
    "visuals_page.py": 1,
}

# setStyleSheet() calls containing a literal hex/rgb colour instead of a
# ThemeColors.*.get() reference — these don't adapt to a light/dark toggle.
_HARDCODED_COLOUR_BASELINE = {
    "components/capture_preview.py": 2,
    "pages/capture_page.py": 1,
    "pages/keys_page.py": 5,
    "pages/other_page.py": 18,
    "setup_wizard.py": 11,
    "window.py": 1,
}

_T_CALL = re.compile(r"""\bt\(\s*["']([^"']+)["']""")
_COLOUR_LITERAL = re.compile(r"#[0-9a-fA-F]{3,8}\b|rgba?\(\s*\d+")


def _split_retranslate(src: str):
    """Return (construction_source, retranslateUi_source) for a page file."""
    match = re.search(r"\n    def retranslateUi\(self.*?\n(?=    def |\Z)", src, re.S)
    retranslate = match.group(0) if match else ""
    return src.replace(retranslate, ""), retranslate


def _stale_keys(path: Path) -> set:
    construction, retranslate = _split_retranslate(path.read_text(encoding="utf-8"))
    return set(_T_CALL.findall(construction)) - set(_T_CALL.findall(retranslate))


def _setstylesheet_calls_with_literal_colour(src: str) -> int:
    """Count setStyleSheet(...) calls whose argument contains a colour literal.

    Brace-matches the call rather than scanning fixed-width windows, so a
    multi-line f-string stylesheet is measured as one call and a colour in
    unrelated neighbouring code isn't attributed to it.
    """
    count = 0
    for match in re.finditer(r"setStyleSheet\s*\(", src):
        depth, i = 1, match.end()
        while i < len(src) and depth:
            if src[i] == "(":
                depth += 1
            elif src[i] == ")":
                depth -= 1
            i += 1
        if _COLOUR_LITERAL.search(src[match.end():i]):
            count += 1
    return count


@pytest.mark.parametrize("filename", sorted(_STALE_TRANSLATION_BASELINE))
def test_no_new_stale_translation_keys(filename):
    """Every t()-wrapped widget label must be re-applied in retranslateUi(),
    or it shows the previously-selected language forever after a switch."""
    path = _PAGES / filename
    assert path.exists(), f"{filename} moved or was renamed — update this test"

    stale = _stale_keys(path)
    baseline = _STALE_TRANSLATION_BASELINE[filename]

    assert len(stale) <= baseline, (
        f"{filename} now has {len(stale)} translation keys used at construction "
        f"but never re-applied in retranslateUi() (baseline {baseline}).\n"
        f"New/unaccounted keys: {sorted(stale)}\n\n"
        "Widgets built with t() need the SAME call repeated inside "
        "retranslateUi(), or their text goes stale after the first language "
        "switch. Grep a correct card in the same file for the exact "
        "titleLabel/contentLabel API."
    )

    if len(stale) < baseline:
        pytest.fail(
            f"{filename} improved: {len(stale)} stale keys, baseline says {baseline}. "
            f"Lower _STALE_TRANSLATION_BASELINE['{filename}'] to {len(stale)} to "
            "lock the improvement in."
        )


@pytest.mark.parametrize("relpath", sorted(_HARDCODED_COLOUR_BASELINE))
def test_no_new_hardcoded_stylesheet_colours(relpath):
    """setStyleSheet() with a literal colour won't follow a light/dark
    toggle — colours belong in ThemeColors."""
    path = _GUI / "fluent_app" / relpath
    assert path.exists(), f"{relpath} moved or was renamed — update this test"

    found = _setstylesheet_calls_with_literal_colour(path.read_text(encoding="utf-8"))
    baseline = _HARDCODED_COLOUR_BASELINE[relpath]

    assert found <= baseline, (
        f"{relpath} now has {found} setStyleSheet() calls containing a literal "
        f"colour (baseline {baseline}).\n\n"
        "Add a ThemeColors entry and use ThemeColors.X.get() instead — a literal "
        "hex/rgb won't adapt when the user toggles light/dark."
    )

    if found < baseline:
        pytest.fail(
            f"{relpath} improved: {found} hardcoded colours, baseline says {baseline}. "
            f"Lower _HARDCODED_COLOUR_BASELINE['{relpath}'] to {found} to lock it in."
        )


def test_every_page_defines_retranslate_ui():
    """A page with no retranslateUi() at all goes fully stale on a language
    switch — this one is unconditional, not baselined."""
    missing = [
        p.name for p in sorted(_PAGES.glob("*.py"))
        if not p.name.startswith("__")
        and "def retranslateUi" not in p.read_text(encoding="utf-8")
    ]
    assert not missing, f"pages without retranslateUi(): {missing}"


# ---------------------------------------------------------------------------
# MAKCU: never hold _lock across a sleep
# ---------------------------------------------------------------------------

_WIN_UTILS = Path(__file__).resolve().parent.parent / "src" / "win_utils"


def _lock_blocks_containing_sleep(src: str) -> list:
    """Line numbers of `with self._lock:` blocks whose body sleeps.

    Body extent is determined by indentation, which is what actually
    delimits a Python block — a regex window would either miss a long block
    or falsely capture the statement after a short one.
    """
    lines = src.splitlines()
    offenders = []
    for i, line in enumerate(lines):
        if not re.match(r"\s*with self\._lock\s*:", line):
            continue
        indent = len(line) - len(line.lstrip())
        for body in lines[i + 1:]:
            if not body.strip():
                continue
            if len(body) - len(body.lstrip()) <= indent:
                break  # dedented out of the with-block
            if "time.sleep" in body:
                offenders.append(i + 1)
                break
    return offenders


def test_makcu_never_sleeps_while_holding_the_serial_lock():
    """Holding _lock across a sleep blocks the inference thread's move()/
    click() for the duration — the reason connect()/_try_open() are written
    as a sequence of short locked sections around unlocked sleeps.

    Unconditional, not baselined: makcu_mouse.py is currently clean and must
    stay that way.
    """
    path = _WIN_UTILS / "makcu_mouse.py"
    offenders = _lock_blocks_containing_sleep(path.read_text(encoding="utf-8"))
    assert not offenders, (
        f"makcu_mouse.py holds self._lock across a time.sleep() at line(s) "
        f"{offenders}. Release the lock before sleeping — the inference thread "
        "calls move() on this same lock and will stall for the sleep's duration."
    )


def test_makcu_binary_variant_violation_is_still_present_and_documented():
    """makcu_mouse_binary.py is the unused V2 binary-protocol variant.
    CLAUDE.md says not to wire it in without first fixing its
    lock-across-sleep violation — this asserts that caveat is still true, so
    the note doesn't quietly become false and get deleted as stale.

    If this fails because the file was fixed, delete this test and add
    makcu_mouse_binary.py to the unconditional check above instead.
    """
    path = _WIN_UTILS / "makcu_mouse_binary.py"
    if not path.exists():
        pytest.skip("makcu_mouse_binary.py removed")
    offenders = _lock_blocks_containing_sleep(path.read_text(encoding="utf-8"))
    assert offenders, (
        "makcu_mouse_binary.py no longer holds its lock across a sleep — the "
        "CLAUDE.md warning about wiring it in is now out of date. Update that "
        "note and move this file to the unconditional check."
    )
