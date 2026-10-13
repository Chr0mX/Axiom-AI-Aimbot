# tests/test_model_selection_pipeline.py
"""Regression tests for ONNX selection → TensorRT cache/hot-swap data flow.

Covers the bugs where:
1. Hot-swap reverted config.model_path when an engine was still missing,
   so Convert's post-build write (and UI selection) never stuck in config.json.
2. find_trt_engine_cache() prefix-matched a shorter stem onto a longer model's
   engine (Foo vs Foo_v2).
3. Untagged engines were invisible under fp16/fp32 filters forever.
4. list_models()/resolve_model_path() rejected .ONNX casing.
"""

from __future__ import annotations

import os
import sys
import types
from unittest.mock import MagicMock

import pytest


def _import_session_utils(monkeypatch):
    """session_utils imports onnxruntime at module top level — fake it so
    the pure cache helpers remain testable in this sandbox."""
    if "onnxruntime" not in sys.modules:
        fake = types.ModuleType("onnxruntime")
        fake.get_available_providers = lambda: ["CPUExecutionProvider"]
        fake.SessionOptions = object
        fake.GraphOptimizationLevel = types.SimpleNamespace(ORT_ENABLE_ALL=99)
        fake.ExecutionMode = types.SimpleNamespace(ORT_SEQUENTIAL=0)
        monkeypatch.setitem(sys.modules, "onnxruntime", fake)
    # Drop a previously-failed partial import so `from core import session_utils`
    # re-runs against the faked onnxruntime.
    sys.modules.pop("core.session_utils", None)
    from core import session_utils
    return session_utils


def test_find_trt_engine_cache_rejects_longer_stem_collision(tmp_path, monkeypatch):
    session_utils = _import_session_utils(monkeypatch)

    short = tmp_path / "Foo.onnx"
    (tmp_path / "Foo_v2_sm75_fp16.engine").write_bytes(b"v2")
    assert session_utils.find_trt_engine_cache(str(short), str(tmp_path), fp16=True) is None

    long = tmp_path / "Foo_v2.onnx"
    found = session_utils.find_trt_engine_cache(str(long), str(tmp_path), fp16=True)
    assert found is not None
    assert os.path.basename(found) == "Foo_v2_sm75_fp16.engine"


def test_find_trt_engine_cache_accepts_untagged_engine_as_precision_fallback(tmp_path, monkeypatch):
    session_utils = _import_session_utils(monkeypatch)

    onnx_path = tmp_path / "Roblox_8n.onnx"
    untagged = tmp_path / "Roblox_8n_deadbeef.engine"
    untagged.write_bytes(b"untagged")

    assert session_utils.find_trt_engine_cache(str(onnx_path), str(tmp_path), fp16=True) == str(untagged)
    assert session_utils.find_trt_engine_cache(str(onnx_path), str(tmp_path), fp16=False) == str(untagged)


def test_find_trt_engine_cache_prefers_tagged_over_untagged(tmp_path, monkeypatch):
    session_utils = _import_session_utils(monkeypatch)

    onnx_path = tmp_path / "Roblox_8n.onnx"
    (tmp_path / "Roblox_8n_deadbeef.engine").write_bytes(b"untagged")
    tagged = tmp_path / "Roblox_8n_sm75_fp16.engine"
    tagged.write_bytes(b"fp16")

    assert session_utils.find_trt_engine_cache(str(onnx_path), str(tmp_path), fp16=True) == str(tagged)


def test_list_models_includes_uppercase_onnx_extension(tmp_path, monkeypatch):
    from core import app_controller

    model_dir = tmp_path / "Model"
    model_dir.mkdir()
    (model_dir / "lower.onnx").write_bytes(b"")
    (model_dir / "UPPER.ONNX").write_bytes(b"")
    (model_dir / "notes.txt").write_bytes(b"")
    monkeypatch.setattr(app_controller, "project_root", str(tmp_path))

    assert app_controller.list_models() == ["UPPER.ONNX", "lower.onnx"]


def test_resolve_model_path_accepts_uppercase_extension(tmp_path, monkeypatch):
    from core import app_controller

    (tmp_path / "Model").mkdir()
    model_file = tmp_path / "Model" / "Real.ONNX"
    model_file.write_bytes(b"")
    monkeypatch.setattr(app_controller, "project_root", str(tmp_path))

    resolved, reason = app_controller.resolve_model_path("Model/Real.ONNX")
    assert reason is None
    assert resolved == os.path.normpath(str(model_file))


def test_hot_swap_source_does_not_revert_model_path_on_missing_engine():
    """ai_loop isn't importable here (win32api), so assert the contract in source:
    the needs_trt_build skip branch must keep config.model_path (no assignment
    back to current_model_path).
    """
    src = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "src", "core", "ai_loop.py",
    )
    text = open(src, encoding="utf-8").read()
    # Locate the needs_trt_build skip block and ensure it no longer reverts.
    idx = text.find("if needs_trt_build(config, abs_model_path):")
    assert idx > 0
    block = text[idx:idx + 1200]
    assert "Keeping selected model_path=" in block
    assert "config.model_path = current_model_path" not in block


def test_portable_model_path_normalization_uses_forward_slashes(tmp_path, monkeypatch):
    """Convert success path rewrite must persist Model/<basename> portably."""
    from core import app_controller

    model_dir = tmp_path / "Model"
    model_dir.mkdir()
    onnx = model_dir / "Picked Model.onnx"
    onnx.write_bytes(b"")

    config = MagicMock()
    config.inference_paused = True
    config.trt_fp16_enabled = False
    config.model_path = "Model/old.onnx"

    class _FakeProc:
        returncode = 0
        stdout = iter(["ok\n"])

        def wait(self):
            return 0

    monkeypatch.setattr(app_controller, "project_root", str(tmp_path))
    monkeypatch.setattr(app_controller, "_src_dir", str(tmp_path / "src"))
    monkeypatch.setattr(app_controller.subprocess, "Popen", lambda *a, **k: _FakeProc())
    # Don't blanket-patch exists — leave real filesystem checks for Model/.

    saved = {}

    def _fake_save(cfg):
        saved["model_path"] = cfg.model_path

    fake_cfg_mod = types.ModuleType("core.config")
    fake_cfg_mod.save_config = _fake_save
    monkeypatch.setitem(__import__("sys").modules, "core.config", fake_cfg_mod)

    # Reset shared convert state so this doesn't pollute other tests.
    with app_controller._convert_lock:
        app_controller._convert_state.update(
            running=False, done=False, success=False, message="",
            model_path="", log=[],
        )

    app_controller._run_conversion_worker(
        config, str(onnx), str(tmp_path / "trt_cache"), True, 2048,
    )

    assert config.model_path == "Model/Picked Model.onnx"
    assert "\\" not in config.model_path
    assert saved.get("model_path") == "Model/Picked Model.onnx"

    with app_controller._convert_lock:
        app_controller._convert_state.update(
            running=False, done=False, success=False, message="",
            model_path="", log=[],
        )


def test_slider_spin_card_syncs_handle_after_blocked_setvalue():
    """slider_spin_card must call _adjustHandlePos after blocked setValue —
    otherwise qfluentwidgets leaves the thumb stuck on config load.
    """
    src = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "src", "gui", "fluent_app", "components", "slider_spin_card.py",
    )
    text = open(src, encoding="utf-8").read()
    assert "_syncSliderHandle" in text
    assert '_adjustHandlePos' in text
    # All three card types must refresh the handle after setValue.
    assert text.count("_syncSliderHandle()") + text.count('_adjustHandlePos"') >= 2
    assert "self.slider.update()" in text


def test_schema_includes_parity_fields():
    from core import web_control_settings as wcs

    assert "second_inference_mode" in wcs._SCHEMA["inference"]
    assert "hud_confidence" in wcs._SCHEMA["inference"]
    assert "hud_udp_enabled" in wcs._SCHEMA["capture"]
    assert "second_inference_fps" in wcs._SCHEMA["capture"]
    assert "show_aim_prediction_marker" in wcs._SCHEMA["visuals"]
    assert "status_panel_show_aim_toggle" in wcs._SCHEMA["visuals"]
    assert "status_panel_show_preset_config" in wcs._SCHEMA["visuals"]
    intensity = wcs._SCHEMA["aim"]["humanization.intensity"]
    assert intensity.get("scale") == 100.0
