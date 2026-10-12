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
    # Both integer and float spin paths must sync the handle.
    assert src.count("self._syncSliderHandle()") >= 4
