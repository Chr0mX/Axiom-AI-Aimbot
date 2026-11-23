"""Translation files: every language pack parses, and every GUI string that has an English default is in
the English file too (other packs fall back to English per key, but the English pack is the template a
translator starts from — a key that only lives in a ``t("key", "default")`` call can never be translated)."""
import ast
import glob
import json
import pathlib

import pytest

SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
LANG_DIR = SRC / "core" / "language_data"
ENGLISH = LANG_DIR / "English_English.json"


def _source_files():
    for sub in ("gui", "core"):
        for f in glob.glob(str(SRC / sub / "**" / "*.py"), recursive=True):
            yield pathlib.Path(f)


def _keys_with_default():
    """{key: [(file, default)]} for every t("key", <string>) call (string constants resolved)."""
    found = {}
    for path in _source_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        consts = {}
        for n in tree.body:
            if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
                try:
                    consts[n.targets[0].id] = ast.literal_eval(n.value)
                except Exception:
                    pass
        for n in ast.walk(tree):
            if not (isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "t"
                    and len(n.args) >= 2 and isinstance(n.args[0], ast.Constant) and isinstance(n.args[0].value, str)):
                continue
            arg = n.args[1]
            try:
                default = ast.literal_eval(arg)
            except Exception:
                default = consts.get(arg.id) if isinstance(arg, ast.Name) else None
            if isinstance(default, str):
                found.setdefault(n.args[0].value, []).append((path.name, default))
    return found


@pytest.mark.parametrize("pack", sorted(LANG_DIR.glob("*.json")), ids=lambda p: p.stem)
def test_every_language_pack_parses_into_flat_strings(pack):
    data = json.loads(pack.read_text(encoding="utf-8"))
    assert isinstance(data, dict) and data
    assert all(isinstance(k, str) and isinstance(v, str) for k, v in data.items())


def test_every_default_text_key_exists_in_the_english_pack():
    english = json.loads(ENGLISH.read_text(encoding="utf-8"))
    missing = sorted(k for k in _keys_with_default() if k not in english)
    assert not missing, f"add these to English_English.json (their code default is the English text): {missing}"


def test_one_key_never_has_two_different_default_texts():
    """A key used in two places must say the same thing in both: a language switch re-runs t() with
    the second default and the label used to flip (thread_priority_desc did)."""
    clashes = {k: sorted({d for _, d in v}) for k, v in _keys_with_default().items() if len({d for _, d in v}) > 1}
    assert not clashes, clashes


def test_aim_tuning_retranslation_table_matches_the_cards():
    """aim_page.retranslateUi() refreshes the Advanced Aim Tuning cards from _TUNING_TEXT; each row must
    name a card that exists and carry the same key/English text the card is built with."""
    import re
    src = (SRC / "gui" / "fluent_app" / "pages" / "aim_page.py").read_text(encoding="utf-8")
    table_src = re.search(r"_TUNING_TEXT = \((.*?)\n\)\n", src, re.S).group(1)
    rows = ast.literal_eval("(" + table_src + ")")
    assert len(rows) >= 8
    for attr, title_key, title, desc_key, desc in rows:
        ctor = re.search(rf"\n        self\.{attr} = \w+\((.*?)\n        \)", src, re.S)
        assert ctor, f"{attr} is not built in aim_page"
        text = re.sub(r"\s+", " ", ctor.group(1))
        assert f't("{title_key}", "{title}")' in text, (attr, title_key)
        if desc_key:
            assert f't("{desc_key}", "{desc}")' in text, (attr, desc_key)
