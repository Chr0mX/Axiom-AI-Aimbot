"""auto_fire imports send_mouse_button from win_utils at module import time.

That name lives in mouse_click.py. If it is missing from win_utils/__init__.py,
starting the AI threads raises ImportError and the app exits before the GUI
comes up. Importing win_utils here would pull in win32api, so this checks the
export via the AST instead.
"""

import ast
import os


_SRC = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "src")


def _parse(rel):
    with open(os.path.join(_SRC, rel), encoding="utf-8") as f:
        return ast.parse(f.read())


def _imported_from(tree, module):
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == module:
            names.update(alias.name for alias in node.names)
    return names


def _all_names(tree):
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "__all__":
                return {elt.value for elt in node.value.elts if isinstance(elt, ast.Constant)}
    return set()


def test_win_utils_exports_send_mouse_button():
    tree = _parse(os.path.join("win_utils", "__init__.py"))
    imported = _imported_from(tree, "mouse_click")
    assert "send_mouse_button" in imported
    assert "send_mouse_button" in _all_names(tree)


def test_auto_fire_imports_the_exported_name():
    tree = _parse(os.path.join("core", "auto_fire.py"))
    imported = _imported_from(tree, "win_utils")
    assert "send_mouse_button" in imported
    assert "send_mouse_click" in imported
