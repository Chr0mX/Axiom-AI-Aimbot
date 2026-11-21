"""notify(): one InfoBar helper for results/validation messages (a QMessageBox is for confirmations only)."""
import importlib.util
import pathlib
import sys
import types
from unittest.mock import MagicMock

SPEC = importlib.util.spec_from_file_location(
    "notify_mod", pathlib.Path(__file__).resolve().parents[1] / "src/gui/fluent_app/components/notify.py")
notify_mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(notify_mod)             # imports qfluentwidgets lazily: no Qt needed here


def test_errors_stay_longest_and_every_kind_has_a_duration():
    d = notify_mod.DURATION_MS
    assert set(d) == {"success", "info", "warning", "error"}
    assert d["success"] < d["info"] < d["warning"] < d["error"]


def _fake_qfluentwidgets(monkeypatch):
    mod = types.ModuleType("qfluentwidgets")
    mod.InfoBar = MagicMock()
    mod.InfoBarPosition = types.SimpleNamespace(TOP="TOP")
    monkeypatch.setitem(sys.modules, "qfluentwidgets", mod)
    return mod


def test_notify_uses_the_matching_infobar_kind_position_and_duration(monkeypatch):
    qf = _fake_qfluentwidgets(monkeypatch)
    page = object()
    notify_mod.notify(page, "warning", "Title", "Body")
    qf.InfoBar.warning.assert_called_once_with("Title", "Body", duration=6000, isClosable=True, position="TOP", parent=page)


def test_unknown_kind_falls_back_to_info(monkeypatch):
    qf = _fake_qfluentwidgets(monkeypatch)
    notify_mod.notify(object(), "bogus", "T")
    assert qf.InfoBar.info.called
