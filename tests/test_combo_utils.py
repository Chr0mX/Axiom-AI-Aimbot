"""reset_items(): rebuilding a combo must never emit (and so never write config)."""
import importlib.util
import pathlib

SPEC = importlib.util.spec_from_file_location(
    "combo_utils", pathlib.Path(__file__).resolve().parents[1] / "src/gui/fluent_app/components/combo_utils.py")
combo_utils = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(combo_utils)            # pure Python: no Qt needed


class FakeCombo:
    """Emits on clear/addItems/setCurrentIndex like a QComboBox, unless signals are blocked."""

    def __init__(self, items=("a", "b", "c"), index=1):
        self.items, self.index, self.blocked, self.emitted = list(items), index, False, []

    def blockSignals(self, flag):
        self.blocked = flag

    def currentIndex(self):
        return self.index

    def _emit(self):
        if not self.blocked:
            self.emitted.append(self.index)

    def clear(self):
        self.items, self.index = [], -1
        self._emit()

    def addItems(self, items):
        first = not self.items
        self.items += list(items)
        if first and self.items:
            self.index = 0                      # Qt selects the first item and emits
            self._emit()

    def setCurrentIndex(self, i):
        self.index = i
        self._emit()


def test_nothing_is_emitted_and_the_index_is_kept():
    c = FakeCombo(index=2)
    combo_utils.reset_items(c, ["x", "y", "z"])
    assert c.items == ["x", "y", "z"] and c.index == 2 and c.emitted == [] and c.blocked is False


def test_unblocked_rebuild_would_have_emitted_the_first_item():
    """What the helper prevents: the same sequence without it emits index 0 (a config write)."""
    c = FakeCombo(index=2)
    c.clear()
    c.addItems(["x", "y", "z"])
    assert c.emitted and c.emitted[-1] == 0


def test_selection_restored_by_text_when_still_present():
    c = FakeCombo(items=["none", "COM3", "COM4"], index=2)
    combo_utils.reset_items(c, ["none", "COM4", "COM5"], text="COM4")
    assert c.index == 1 and c.emitted == []


def test_text_missing_falls_back_to_the_old_index_and_bad_index_to_the_first():
    c = FakeCombo(items=["a", "b", "c"], index=2)
    combo_utils.reset_items(c, ["a", "b", "c"], text="zzz")
    assert c.index == 2
    c = FakeCombo(index=5)
    combo_utils.reset_items(c, ["only"])
    assert c.index == 0 and c.emitted == []


def test_signals_are_unblocked_even_if_a_step_raises():
    class Broken(FakeCombo):
        def addItems(self, items):
            raise RuntimeError("boom")
    c = Broken()
    try:
        combo_utils.reset_items(c, ["x"])
    except RuntimeError:
        pass
    assert c.blocked is False
