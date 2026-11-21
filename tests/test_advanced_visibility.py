"""AdvancedVisibility: cards behind the global "Show advanced settings" switch."""
import importlib.util
import pathlib


SPEC = importlib.util.spec_from_file_location(
    "advanced", pathlib.Path(__file__).resolve().parents[1] / "src/gui/fluent_app/components/advanced.py")
advanced = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(advanced)               # pure Python: no Qt needed


class FakeCard:
    """setVisible/isHidden like a QWidget (explicit visibility only)."""

    def __init__(self, hidden=False):
        self.hidden, self.calls = hidden, []

    def setVisible(self, visible=True):
        self.calls.append(bool(visible))
        self.hidden = not visible

    def isHidden(self):
        return self.hidden


class Switch:
    def __init__(self, on):
        self.on = on

    def __call__(self):
        return self.on


def make(on=False, n=2):
    sw = Switch(on)
    gate = advanced.AdvancedVisibility(sw)
    cards = [FakeCard() for _ in range(n)]
    gate.register(*cards)
    return sw, gate, cards


def test_switch_off_hides_and_the_page_rule_cannot_unhide():
    sw, gate, (a, b) = make(on=False)
    gate.apply()
    assert a.hidden and b.hidden
    a.setVisible(True)                           # the page's own rule says "show"
    assert a.hidden                              # ...but the switch wins
    assert gate.wanted(a) is True                # and the rule's wish is remembered


def test_switching_on_restores_exactly_what_the_rules_wanted():
    sw, gate, (a, b) = make(on=False)
    a.setVisible(True)
    b.setVisible(False)                          # a rule hides b (e.g. its parent feature is off)
    assert a.hidden and b.hidden
    sw.on = True
    gate.apply()
    assert a.hidden is False and b.hidden is True


def test_switch_on_is_transparent_to_the_rules():
    sw, gate, (a, b) = make(on=True)
    a.setVisible(False)
    assert a.hidden
    a.setVisible(True)
    assert not a.hidden
    assert gate.wanted(a) is True


def test_toggle_round_trip_is_stable():
    sw, gate, (a, b) = make(on=True)
    b.setVisible(False)
    for _ in range(3):
        sw.on = False
        gate.apply()
        assert a.hidden and b.hidden
        sw.on = True
        gate.apply()
        assert not a.hidden and b.hidden


def test_a_card_hidden_before_registration_stays_what_the_rules_wanted():
    sw = Switch(True)
    gate = advanced.AdvancedVisibility(sw)
    pre_hidden = FakeCard(hidden=True)
    gate.register(pre_hidden)
    assert gate.wanted(pre_hidden) is False
    gate.apply()
    assert pre_hidden.hidden


def test_registering_twice_or_none_is_harmless():
    sw, gate, (a, b) = make()
    gate.register(a, None, b)
    assert len(gate) == 2
    a.setVisible(True)
    assert a.calls[-1] is False                  # still exactly one shadow, no recursion


def test_register_names_skips_missing_attributes():
    class Page:
        def __init__(self):
            self.cardA = FakeCard()

    gate = advanced.AdvancedVisibility(Switch(False))
    page = Page()
    gate.register_names(page, ["cardA", "doesNotExist"])
    assert len(gate) == 1


def test_it_never_reads_or_writes_a_setting():
    """The gate only owns visibility: nothing but setVisible is ever called on a card."""
    class Strict(FakeCard):
        def __getattr__(self, name):
            raise AssertionError(f"unexpected access: {name}")

    sw = Switch(False)
    gate = advanced.AdvancedVisibility(sw)
    c = Strict()
    gate.register(c)
    gate.apply()
    sw.on = True
    gate.apply()
    c.setVisible(False)
