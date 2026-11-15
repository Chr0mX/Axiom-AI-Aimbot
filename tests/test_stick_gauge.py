import importlib.util
import os

_spec = importlib.util.spec_from_file_location(
    "stick_gauge", os.path.join(os.path.dirname(__file__), "..", "src", "core", "stick_gauge.py"))
sg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sg)


def test_center_bottom_right():
    assert sg.gauge_center(1000, 800) == (1000 - 24 - 70, 800 - 24 - 70)


def test_dot_offset_and_rings():
    g = sg.stick_gauge_geometry(0.5, -0.5, 0.1, 0.2, radius=100)
    assert g["dot_dx"] == 50 and g["dot_dy"] == -50
    assert g["dz_r"] == 10 and g["game_dz_r"] == 20 and g["deflected"]


def test_dot_clamped_to_ring():
    g = sg.stick_gauge_geometry(1.0, 1.0, 0, 0, radius=100)
    assert abs((g["dot_dx"] ** 2 + g["dot_dy"] ** 2) ** 0.5 - 100) < 1e-6


def test_centered_and_bad_input():
    assert not sg.stick_gauge_geometry(0, 0, 0.02, 0)["deflected"]
    g = sg.stick_gauge_geometry(float("nan"), None, 0, 0)
    assert g["dot_dx"] == 0 and g["dot_dy"] == 0
