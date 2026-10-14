# tests/conftest.py
"""
Pytest 全域配置 - 確保 src 目錄在 Python 路徑中
"""

import sys
import os

# 確保 src 目錄在路徑中，使所有測試都能正確 import
src_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)


import pytest


@pytest.fixture(autouse=True)
def _isolated_cwd(tmp_path, monkeypatch):
    """Run every test from its own empty temp directory.

    ``core.config.save_config()``/``load_config()`` default to the relative
    path ``"config.json"`` (and ``state.json``/``presets/``/``configs/`` are
    CWD-relative too), so a test that reaches the real defaults used to read
    and *write* the developer's actual files in the directory pytest was
    started from. Concretely: ``tests/test_app_controller.py`` truncated a
    real ``config.json`` to 0 bytes on every run. ``src/`` is put on
    ``sys.path`` by absolute path above, so nothing here depends on the CWD.
    """
    monkeypatch.chdir(tmp_path)
