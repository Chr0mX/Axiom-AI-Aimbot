# tests/test_slider_sync.py
"""Regression: config/preset load must move qfluentwidgets slider handles.

qfluentwidgets' Slider only repositions its handle on valueChanged. Our
Slider*Card.setValue() blocks signals (to avoid feedback loops), which used
to leave the thumb at the previous config's position while the SpinBox/label
showed the new value.
"""

from __future__ import annotations

import ast
import os


def _slider_spin_card_source() -> str:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, "src", "gui", "fluent_app", "components", "slider_spin_card.py")
    with open(path, encoding="utf-8") as f:
        return f.read()


def test_setvalue_methods_refresh_handle_after_unblocking_signals():
    src = _slider_spin_card_source()
    tree = ast.parse(src)

    setvalue_fns = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "setValue":
            setvalue_fns.append(node)
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "setValue":
            setvalue_fns.append(node)

    assert len(setvalue_fns) == 3, "expected SliderSpin/Double/Label setValue"

    for fn in setvalue_fns:
        body_src = ast.get_source_segment(src, fn) or ""
        assert "blockSignals(False)" in body_src
        # Either an inline _adjustHandlePos call (SliderLabelCard) or the
        # shared _syncSliderHandle() helper (Spin/Double cards) — both end
        # by refreshing the qfluentwidgets handle after blocked setValue.
        assert "_adjustHandlePos" in body_src or "_syncSliderHandle" in body_src
    assert "self.slider.update()" in src


def test_spin_changed_also_refreshes_handle():
    src = _slider_spin_card_source()
    assert src.count("def _onSpinChanged") == 2
    # Both integer and float spin paths must sync the handle (the helper itself now lives once in
    # the shared base class; the label card's setValue uses it too).
    assert src.count("def _syncSliderHandle") == 1
    assert src.count("self._syncSliderHandle()") >= 4


# ---------------------------------------------------------------------------------------------
# Layout (scaling) regressions. PyQt6 is not importable in CI, so these check the source and run
# the pure width helpers against a fake font-metrics object.
# ---------------------------------------------------------------------------------------------

def _helpers_namespace():
    """exec the module's constants + width helpers with a stand-in QFontMetrics (10 px per char)."""
    src = _slider_spin_card_source()
    tree = ast.parse(src)
    wanted = {"text_width", "spin_min_width", "label_min_width"}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    consts = [n for n in tree.body if isinstance(n, ast.Assign)
              and any(isinstance(t, ast.Name) and t.id.startswith("_") and t.id.isupper() or
                      isinstance(t, ast.Name) and t.id.startswith("_") for t in n.targets)]

    class FakeMetrics:
        def __init__(self, font):
            pass

        def horizontalAdvance(self, text):
            return 10 * len(text)

    ns = {"QFontMetrics": FakeMetrics, "QWidget": object}
    mod = ast.Module(body=consts + nodes, type_ignores=[])
    exec(compile(ast.fix_missing_locations(mod), "slider_spin_card_helpers", "exec"), ns)
    return ns


class _FakeSpin:
    def __init__(self, lo, hi, prefix="", suffix=""):
        self._lo, self._hi, self._p, self._s = lo, hi, prefix, suffix

    def minimum(self):
        return self._lo

    def maximum(self):
        return self._hi

    def prefix(self):
        return self._p

    def suffix(self):
        return self._s

    def textFromValue(self, v):
        return str(v)

    def font(self):
        return None


def test_spin_width_covers_text_and_spin_buttons():
    ns = _helpers_namespace()
    spin = _FakeSpin(0, 1000, suffix=" ms")
    w = ns["spin_min_width"](spin)
    widest = ns["text_width"](spin, "1000 ms")
    # The vendored stylesheet reserves ~92 px for padding + the two spin buttons: never go below it.
    assert ns["_SPIN_CHROME_W"] >= 92
    assert w >= widest + 92
    # An explicit floor still wins when the content is tiny.
    assert ns["spin_min_width"](_FakeSpin(0, 9), floor=200) == 200


def test_label_width_fits_every_sample():
    ns = _helpers_namespace()
    label = _FakeSpin(0, 1)
    w = ns["label_min_width"](label, ["0.00", "100 %", "3000 px/s"], floor=45)
    assert w >= ns["text_width"](label, "3000 px/s")
    assert ns["label_min_width"](label, ["1"], floor=70) == 70


def test_cards_have_no_hard_minimum_from_call_site_widths():
    """slider_width / spin_width / label_width used to be hard minimums that could never shrink."""
    src = _slider_spin_card_source()
    for bad in ("setMinimumWidth(slider_width)", "setMinimumWidth(spin_width)", "setMinimumWidth(label_width)"):
        assert bad not in src
    assert src.count("self._installControls(") == 3
    # slider stays flexible within a bounded range; value widget is fixed to its computed width
    assert "slider.setMaximumWidth(_SLIDER_MAX_W)" in src and "setFixedWidth(value_width)" in src


def test_main_window_has_minimum_size_and_screen_clamp():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "src", "gui", "fluent_app", "window.py"), encoding="utf-8") as f:
        src = f.read()
    assert "self.setMinimumSize(" in src and "availableGeometry()" in src
