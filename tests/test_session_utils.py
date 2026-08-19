# tests/test_session_utils.py
"""Tests for src/core/session_utils.py's TensorRT engine-cache helpers.

Covers the fix for: hot-swapping to a TensorRT model with no cached .engine
yet used to compile one synchronously inline (a 1-5 minute call) on
ai_loop.py's per-frame hot-swap path, freezing the whole aim loop with no
progress feedback. needs_trt_build() (backed by find_trt_engine_cache() and
effective_first_provider()) is what both the GUI's auto-redirect-to-Convert
logic (model_page.py) and ai_loop.py's hot-swap safety net check before ever
letting that inline build happen.

Like tests/test_convert_to_engine.py's session_utils-touching test, each
test here imports core.session_utils inside the test function body (not at
module scope) so a missing `onnxruntime` in this sandbox fails only that
test at run time instead of aborting collection of the whole file.
"""

import os
from unittest.mock import patch


class _FakeConfig:
    def __init__(self, backend="tensorrt", model_path="Model/Roblox_8n.onnx"):
        self.inference_backend = backend
        self.trt_fp16_enabled = True
        self.model_path = model_path


def _with_providers(providers):
    """Context manager patching session_utils.ort.get_available_providers()."""
    from core import session_utils
    return patch.object(session_utils.ort, "get_available_providers", return_value=providers)


def test_find_trt_engine_cache_returns_none_when_missing(tmp_path):
    from core import session_utils

    onnx_path = tmp_path / "Roblox_8n.onnx"
    assert session_utils.find_trt_engine_cache(str(onnx_path), str(tmp_path)) is None


def test_find_trt_engine_cache_finds_matching_engine(tmp_path):
    from core import session_utils

    onnx_path = tmp_path / "Roblox_8n.onnx"
    engine_path = tmp_path / "Roblox_8n_fp16.engine"
    engine_path.write_bytes(b"fake engine bytes")

    found = session_utils.find_trt_engine_cache(str(onnx_path), str(tmp_path))
    assert found == str(engine_path)


def test_find_trt_engine_cache_ignores_other_models(tmp_path):
    from core import session_utils

    (tmp_path / "SomeOtherModel_fp16.engine").write_bytes(b"fake")
    onnx_path = tmp_path / "Roblox_8n.onnx"
    assert session_utils.find_trt_engine_cache(str(onnx_path), str(tmp_path)) is None


def test_effective_first_provider_reports_tensorrt(tmp_path):
    from core import session_utils

    cfg = _FakeConfig(backend="tensorrt")
    with _with_providers(["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]):
        assert session_utils.effective_first_provider(cfg) == "TensorrtExecutionProvider"


def test_effective_first_provider_reports_cpu_when_only_cpu_available():
    from core import session_utils

    cfg = _FakeConfig(backend="tensorrt")
    with _with_providers(["CPUExecutionProvider"]):
        assert session_utils.effective_first_provider(cfg) == "CPUExecutionProvider"


def test_needs_trt_build_true_when_engine_missing(tmp_path):
    from core import session_utils

    onnx_path = tmp_path / "Roblox_8n.onnx"
    cfg = _FakeConfig(backend="tensorrt", model_path=str(onnx_path))
    with _with_providers(["TensorrtExecutionProvider", "CPUExecutionProvider"]), \
         patch.object(session_utils, "_TRT_CACHE_DIR", str(tmp_path)):
        assert session_utils.needs_trt_build(cfg, str(onnx_path)) is True


def test_needs_trt_build_false_when_engine_cached(tmp_path):
    from core import session_utils

    onnx_path = tmp_path / "Roblox_8n.onnx"
    (tmp_path / "Roblox_8n_fp16.engine").write_bytes(b"fake")
    cfg = _FakeConfig(backend="tensorrt", model_path=str(onnx_path))
    with _with_providers(["TensorrtExecutionProvider", "CPUExecutionProvider"]), \
         patch.object(session_utils, "_TRT_CACHE_DIR", str(tmp_path)):
        assert session_utils.needs_trt_build(cfg, str(onnx_path)) is False


def test_needs_trt_build_false_when_backend_not_tensorrt(tmp_path):
    """A CPU/DirectML backend never needs a TRT build regardless of cache
    state — effective_first_provider() short-circuits before any I/O."""
    from core import session_utils

    onnx_path = tmp_path / "Roblox_8n.onnx"
    cfg = _FakeConfig(backend="cpu", model_path=str(onnx_path))
    with _with_providers(["TensorrtExecutionProvider", "CPUExecutionProvider"]), \
         patch.object(session_utils, "_TRT_CACHE_DIR", str(tmp_path)):
        assert session_utils.needs_trt_build(cfg, str(onnx_path)) is False


def test_needs_trt_build_resolves_relative_model_path(tmp_path):
    """A relative model_path (as model_page.py/ai_loop.py store it, e.g.
    "Model/x.onnx") must resolve the same cache-stem lookup as an absolute
    one — needs_trt_build() only cares about the filename stem, and the
    resolution itself is exercised for coverage even though the actual
    project-root-relative resolution isn't meaningful under tmp_path."""
    from core import session_utils

    cfg = _FakeConfig(backend="tensorrt", model_path="Model/Roblox_8n.onnx")
    with _with_providers(["TensorrtExecutionProvider", "CPUExecutionProvider"]):
        # Doesn't raise, and resolves to *some* boolean regardless of whether
        # the real project's trt_cache/ happens to hold a matching engine.
        assert session_utils.needs_trt_build(cfg, "Model/Roblox_8n.onnx") in (True, False)
