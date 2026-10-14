# tests/test_model_rollback.py
"""Failed TensorRT conversion must roll the model selection back to the model
the inference loop is actually running (and persist that), and hot-swap load
failures must be published for the GUI / Web UI."""

from __future__ import annotations

import os
import sys
import types
from types import SimpleNamespace

import pytest

from core import model_selection


def _cfg(model_path="Model/new.onnx", fp16=True, loaded="Model/old.onnx", loaded_fp16=False):
    return SimpleNamespace(
        model_path=model_path, trt_fp16_enabled=fp16,
        loaded_model_path=loaded, loaded_model_fp16=loaded_fp16,
    )


@pytest.fixture
def saved(monkeypatch):
    calls = []
    fake = types.ModuleType("core.config")
    fake.save_config = lambda c: calls.append((c.model_path, c.trt_fp16_enabled))
    monkeypatch.setitem(sys.modules, "core.config", fake)
    return calls


def test_same_model_path_handles_forms(tmp_path):
    root = str(tmp_path)
    assert model_selection.same_model_path("Model/a b.onnx", os.path.join(root, "Model", "a b.onnx"), root)
    assert model_selection.same_model_path("Model\\a.onnx", "Model/a.onnx", root)
    assert not model_selection.same_model_path("Model/a.onnx", "Model/b.onnx", root)
    assert not model_selection.same_model_path("", "Model/b.onnx", root)


def test_rollback_restores_running_model_and_saves(saved, tmp_path):
    c = _cfg()
    restored = model_selection.rollback_failed_selection(
        c, os.path.join(str(tmp_path), "Model", "new.onnx"), str(tmp_path))
    assert restored == "Model/old.onnx"
    assert c.model_path == "Model/old.onnx"
    assert c.trt_fp16_enabled is False
    assert saved == [("Model/old.onnx", False)]


def test_rollback_skipped_when_user_picked_another_model(saved, tmp_path):
    c = _cfg(model_path="Model/third.onnx")
    assert model_selection.rollback_failed_selection(c, "Model/new.onnx", str(tmp_path)) is None
    assert c.model_path == "Model/third.onnx" and saved == []


def test_rollback_skipped_when_no_session_loaded(saved, tmp_path):
    c = _cfg(loaded="")
    assert model_selection.rollback_failed_selection(c, "Model/new.onnx", str(tmp_path)) is None
    assert c.model_path == "Model/new.onnx"


def test_rollback_reverts_precision_for_same_model(saved, tmp_path):
    c = _cfg(model_path="Model/old.onnx", fp16=True, loaded="Model/old.onnx", loaded_fp16=False)
    assert model_selection.rollback_failed_selection(c, "Model/old.onnx", str(tmp_path)) == "Model/old.onnx"
    assert c.trt_fp16_enabled is False


def test_rollback_noop_when_selection_matches_running(saved, tmp_path):
    c = _cfg(model_path="Model/old.onnx", fp16=False)
    assert model_selection.rollback_failed_selection(c, "Model/old.onnx", str(tmp_path)) is None
    assert saved == []


def test_failed_remote_conversion_rolls_back(saved, tmp_path, monkeypatch):
    from core import app_controller

    (tmp_path / "Model").mkdir()
    onnx = tmp_path / "Model" / "new.onnx"
    onnx.write_bytes(b"")

    class _FailProc:
        returncode = 1
        stdout = iter(["boom\n"])

        def wait(self):
            return 1

    monkeypatch.setattr(app_controller, "project_root", str(tmp_path))
    monkeypatch.setattr(app_controller, "_src_dir", str(tmp_path / "src"))
    monkeypatch.setattr(app_controller.subprocess, "Popen", lambda *a, **k: _FailProc())

    c = _cfg()
    c.inference_paused = True
    app_controller._run_conversion_worker(c, str(onnx), str(tmp_path / "trt_cache"), True, 2048)

    assert c.model_path == "Model/old.onnx"
    assert c.trt_fp16_enabled is False
    assert c.inference_paused is False
    status = app_controller.get_conversion_status(0)
    assert status["done"] and status["success"] is False
    assert "Reverted selection" in status["message"]


def test_hot_swap_publishes_loaded_model_and_load_errors():
    """ai_loop isn't importable here (win32api) — assert the contract in source."""
    src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "core", "ai_loop.py")
    text = open(src, encoding="utf-8").read()
    assert text.count("config.model_load_error = (") == 2   # invalid path + load failure
    assert 'config.model_load_error = ""' in text
    assert text.count("config.loaded_model_path = ") == 2    # initial + successful swap
