# tests/test_convert_to_engine.py
"""Tests for src/core/convert_to_engine.py's CLI engine-prefix default.

Covers the fix for: pre-building a TensorRT engine via the documented
default CLI invocation (no --engine-prefix) used to leave the ORT cache
prefix unset, while session_utils.py's build_provider_list() always keys
its runtime lookup by the model filename stem — so the pre-built engine
landed under a cache key the app never looked for and got silently
rebuilt from scratch on first launch. --engine-prefix's argparse default
must now resolve to the model stem when the flag isn't passed, matching
session_utils.py's derivation exactly (os.path.splitext(os.path.basename(...))[0]).
"""

import os
from unittest.mock import patch


def test_engine_prefix_defaults_to_model_stem(tmp_path):
    from core import convert_to_engine as cte

    onnx_path = tmp_path / "Roblox_8n.onnx"
    onnx_path.write_bytes(b"fake onnx content")

    captured = {}

    def fake_build_engine_via_ort(onnx_path, cache_dir, fp16=True, workspace_mb=2048, engine_prefix=""):
        captured["engine_prefix"] = engine_prefix
        return True

    with patch.object(cte, "build_engine_via_ort", side_effect=fake_build_engine_via_ort), \
         patch("sys.argv", ["convert_to_engine.py", "--model", str(onnx_path)]):
        cte.main()

    assert captured["engine_prefix"] == "Roblox_8n"


def test_engine_prefix_explicit_value_is_respected(tmp_path):
    from core import convert_to_engine as cte

    onnx_path = tmp_path / "Roblox_8n.onnx"
    onnx_path.write_bytes(b"fake onnx content")

    captured = {}

    def fake_build_engine_via_ort(onnx_path, cache_dir, fp16=True, workspace_mb=2048, engine_prefix=""):
        captured["engine_prefix"] = engine_prefix
        return True

    with patch.object(cte, "build_engine_via_ort", side_effect=fake_build_engine_via_ort), \
         patch("sys.argv", ["convert_to_engine.py", "--model", str(onnx_path), "--engine-prefix", "custom_prefix"]):
        cte.main()

    assert captured["engine_prefix"] == "custom_prefix"


def test_engine_prefix_explicit_empty_string_disables_prefixing(tmp_path):
    """An explicit empty string must stay empty, not fall back to the model
    stem — the argparse default is None specifically so this is
    distinguishable from "flag not passed"."""
    from core import convert_to_engine as cte

    onnx_path = tmp_path / "Roblox_8n.onnx"
    onnx_path.write_bytes(b"fake onnx content")

    captured = {}

    def fake_build_engine_via_ort(onnx_path, cache_dir, fp16=True, workspace_mb=2048, engine_prefix=""):
        captured["engine_prefix"] = engine_prefix
        return True

    with patch.object(cte, "build_engine_via_ort", side_effect=fake_build_engine_via_ort), \
         patch("sys.argv", ["convert_to_engine.py", "--model", str(onnx_path), "--engine-prefix", ""]):
        cte.main()

    assert captured["engine_prefix"] == ""


def test_default_engine_prefix_matches_session_utils_runtime_lookup(tmp_path):
    """The CLI's default-prefix derivation must produce the exact same
    trt_engine_cache_prefix session_utils.py's build_provider_list() uses at
    runtime — that match is the entire point of the fix; any drift silently
    reintroduces the cache-miss bug. Exercises the real runtime function,
    not a reimplementation of its formula, with TensorRT availability forced
    on since this sandbox has no real TRT-capable ORT build."""
    from core import convert_to_engine as cte
    from core import session_utils

    onnx_path = tmp_path / "Some.Model_v2.onnx"
    onnx_path.write_bytes(b"fake onnx content")
    model_stem = os.path.splitext(os.path.basename(str(onnx_path)))[0]

    # CLI side: what --engine-prefix defaults to when omitted.
    captured = {}

    def fake_build_engine_via_ort(onnx_path, cache_dir, fp16=True, workspace_mb=2048, engine_prefix=""):
        captured["engine_prefix"] = engine_prefix
        return True

    with patch.object(cte, "build_engine_via_ort", side_effect=fake_build_engine_via_ort), \
         patch("sys.argv", ["convert_to_engine.py", "--model", str(onnx_path)]):
        cte.main()

    # Runtime side: what session_utils.py actually keys its cache lookup by.
    class _FakeConfig:
        model_path = str(onnx_path)
        inference_backend = "tensorrt"
        trt_fp16_enabled = True

    with patch.object(
        session_utils.ort, "get_available_providers",
        return_value=["TensorrtExecutionProvider", "CPUExecutionProvider"],
    ):
        providers = session_utils.build_provider_list(_FakeConfig())

    trt_entry = next(p for p in providers if p[0] == "TensorrtExecutionProvider")
    runtime_prefix = trt_entry[1]["trt_engine_cache_prefix"]

    assert captured["engine_prefix"] == runtime_prefix == model_stem == "Some.Model_v2"
