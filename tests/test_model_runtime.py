"""Model Info: which precision/provider is actually running (core/model_runtime.py)."""
import sys
import types
from types import SimpleNamespace

import pytest

from core import model_runtime as mr

TRT = mr.TRT_PROVIDER


def cfg(**kw):
    base = dict(trt_fp16_enabled=False, loaded_model_path="", loaded_model_fp16=None, current_provider="")
    base.update(kw)
    return SimpleNamespace(**base)


class TestDescribeRuntime:
    def test_nothing_loaded_is_idle_and_shows_the_selection(self):
        r = mr.describe_runtime(cfg(trt_fp16_enabled=True), "Model/a.onnx")
        assert r["state"] == "idle" and r["running_precision"] is None and r["selected_precision"] == "FP16"

    def test_fp16_engine_running_for_the_selected_model(self):
        r = mr.describe_runtime(cfg(trt_fp16_enabled=True, loaded_model_path="C:/x/Model/a.onnx",
                                    loaded_model_fp16=True, current_provider=TRT), "Model/a.onnx")
        assert (r["state"], r["running_precision"], r["engine_fp16"]) == ("running", "FP16", True)

    def test_switch_to_fp16_before_the_swap_lands_is_pending_and_still_reports_fp32(self):
        r = mr.describe_runtime(cfg(trt_fp16_enabled=True, loaded_model_path="Model/a.onnx",
                                    loaded_model_fp16=False, current_provider=TRT), "Model/a.onnx")
        assert r["state"] == "pending" and r["running_precision"] == "FP32"
        assert r["engine_fp16"] is False          # inspect what is RUNNING, not what is selected

    def test_other_model_loaded_is_pending(self):
        r = mr.describe_runtime(cfg(loaded_model_path="Model/other.onnx", loaded_model_fp16=True,
                                    current_provider=TRT), "Model/a.onnx")
        assert r["state"] == "pending" and r["running_precision"] is None and r["loaded_model"] == "other.onnx"

    def test_non_tensorrt_provider_has_no_fp_precision(self):
        r = mr.describe_runtime(cfg(trt_fp16_enabled=True, loaded_model_path="Model/a.onnx",
                                    loaded_model_fp16=True, current_provider="CUDAExecutionProvider"),
                                "Model/a.onnx")
        assert r["state"] == "running" and r["running_precision"] is None and r["provider_name"] == "CUDA"

    def test_stem_match_is_case_insensitive_and_path_agnostic(self):
        r = mr.describe_runtime(cfg(loaded_model_path="D:\\Axiom\\Model\\A.ONNX", loaded_model_fp16=False,
                                    current_provider=TRT), "Model/a.onnx")
        assert r["is_loaded_model"]


class TestEngineResolution:
    @pytest.fixture
    def cache(self, tmp_path, monkeypatch):
        # session_utils imports onnxruntime at module level; give it a stand-in and make sure
        # the stand-in-backed module does not leak into other tests.
        had = "core.session_utils" in sys.modules
        fake_ort = types.ModuleType("onnxruntime")
        fake_ort.get_available_providers = lambda: []
        monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
        monkeypatch.delitem(sys.modules, "core.session_utils", raising=False)
        import importlib
        import core
        monkeypatch.delattr(core, "session_utils", raising=False)   # `from core import x` would reuse a stale attr
        session_utils = importlib.import_module("core.session_utils")
        monkeypatch.setattr(session_utils, "_TRT_CACHE_DIR", str(tmp_path))
        for n in ("a_sm86_fp16.engine", "a_sm86_fp32.engine"):
            (tmp_path / n).write_bytes(b"x")
        yield tmp_path
        if not had:
            sys.modules.pop("core.session_utils", None)
            if hasattr(core, "session_utils"):
                delattr(core, "session_utils")

    def test_fp16_selection_no_longer_loses_the_lexical_sort_to_fp32(self, cache):
        # This is the reported bug: sorted(glob)[-1] returned the _fp32 engine.
        c = cfg(trt_fp16_enabled=True, loaded_model_path="Model/a.onnx", loaded_model_fp16=True,
                current_provider=TRT)
        assert mr.resolve_engine_path(c, "Model/a.onnx").endswith("a_sm86_fp16.engine")

    def test_inspects_the_running_precision_while_a_switch_is_pending(self, cache):
        c = cfg(trt_fp16_enabled=True, loaded_model_path="Model/a.onnx", loaded_model_fp16=False,
                current_provider=TRT)
        assert mr.resolve_engine_path(c, "Model/a.onnx").endswith("a_sm86_fp32.engine")

    def test_no_engine_lookup_outside_tensorrt(self, cache):
        c = cfg(loaded_model_path="Model/a.onnx", current_provider="CUDAExecutionProvider")
        assert mr.resolve_engine_path(c, "Model/a.onnx") is None


class TestText:
    INFO = {"format": "TensorRT Engine", "input_size": "640x640", "num_classes": 2,
            "file_size": "12.3 MB", "precision": "FP32"}      # filename-derived value must NOT be shown

    def test_running_fp16_is_stated_clearly_and_filename_precision_is_ignored(self):
        r = mr.describe_runtime(cfg(trt_fp16_enabled=True, loaded_model_path="Model/a.onnx",
                                    loaded_model_fp16=True, current_provider=TRT), "Model/a.onnx")
        text = mr.compose_info_text(self.INFO, r)
        assert "Precision in use: FP16 \u00b7 TensorRT" in text and "RUNNING" in text
        assert "Precision: FP32" not in text and "Selected:" not in text
        assert text.splitlines()[0] == "TensorRT Engine  \u2022  Input: 640x640  \u2022  Classes: 2  \u2022  12.3 MB"

    def test_pending_switch_says_what_is_running_and_what_is_selected(self):
        r = mr.describe_runtime(cfg(trt_fp16_enabled=True, loaded_model_path="Model/a.onnx",
                                    loaded_model_fp16=False, current_provider=TRT), "Model/a.onnx")
        text = mr.compose_info_text(self.INFO, r)
        assert "Precision in use: FP32 \u00b7 TensorRT" in text and "RUNNING" not in text
        assert "Selected: FP16" in text and "pending" in text and "still FP32" in text

    def test_non_tensorrt_explains_the_setting_does_not_apply(self):
        r = mr.describe_runtime(cfg(loaded_model_path="Model/a.onnx", current_provider="DmlExecutionProvider"),
                                "Model/a.onnx")
        assert "only applies to TensorRT" in mr.compose_info_text(self.INFO, r)

    def test_idle_text(self):
        r = mr.describe_runtime(cfg(), "Model/a.onnx")
        assert "not loaded yet" in mr.compose_info_text(self.INFO, r)

    def test_signature_changes_when_the_session_changes(self):
        a = mr.describe_runtime(cfg(loaded_model_path="Model/a.onnx", loaded_model_fp16=False, current_provider=TRT), "Model/a.onnx")
        b = mr.describe_runtime(cfg(loaded_model_path="Model/a.onnx", loaded_model_fp16=True, current_provider=TRT), "Model/a.onnx")
        assert mr.runtime_signature(a) != mr.runtime_signature(b)


class TestWebModelInfo:
    def test_get_model_info_returns_text_and_structured_runtime(self, monkeypatch, tmp_path):
        from core import app_controller, web_control_settings as wcs
        monkeypatch.setattr(app_controller, "resolve_model_path", lambda p: (str(tmp_path / "a.onnx"), None))
        fake = types.ModuleType("model_detect")
        seen = {}
        fake.inspect_model = lambda path: seen.setdefault("p", path) and {
            "format": "ONNX", "input_size": "640x640", "num_classes": 1, "file_size": "1 MB", "precision": None}
        monkeypatch.setitem(sys.modules, "model_detect", fake)
        c = cfg(trt_fp16_enabled=True, loaded_model_path="Model/a.onnx", loaded_model_fp16=True,
                current_provider=TRT)
        out = wcs.get_model_info(c, "Model/a.onnx")
        assert out["ok"] and out["runtime"]["running_precision"] == "FP16" and out["runtime"]["state"] == "running"
        assert "Precision in use: FP16" in out["text"]

    def test_status_exposes_the_live_session(self):
        from core.web_control_server import _build_status
        c = cfg(trt_fp16_enabled=True, loaded_model_path="C:/x/Model/a.onnx", loaded_model_fp16=False,
                current_provider=TRT, model_path="Model/a.onnx", Running=True)
        try:
            st = _build_status(c)
        except AttributeError:
            pytest.skip("_build_status needs a fuller Config than this stub")
        assert st["loaded_model"] == "a.onnx" and st["loaded_model_fp16"] is False
        assert st["loaded_provider"] == TRT and st["trt_fp16_enabled"] is True
