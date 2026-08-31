# tests/test_model_detect.py
"""Tests for model_detect.py's TensorRT Runtime/Logger caching.

TensorRT's own logger is a process-wide singleton (nvinfer1::getLogger()),
registered the first time any builder/runtime/refitter is constructed
anywhere in the process. Constructing a fresh trt.Runtime(trt.Logger(...))
on every inspect_engine() call — as this module used to do — passes a
*new* Python object each time, which never matches whatever got
registered first, so TensorRT logs "The logger passed into
createInferRuntime differs from one already registered..." and ignores
it, on every single call (this was reported as console-spamming
behavior when the Model page's periodic Config-sync re-triggers a model
info check once a second). These tests confirm inspect_engine() now
reuses one cached Runtime instance instead of constructing a new one
each call.

`tensorrt` isn't installed in this sandbox, so a fake module is injected
into sys.modules — the same technique tests/test_web_control_settings.py
already uses for model_detect itself.
"""

import os
import sys
import types

import pytest

import model_detect


class _FakeLogger:
    WARNING = "WARNING"

    def __init__(self, severity):
        self.severity = severity


class _FakeEngine:
    num_io_tensors = 0  # empty loop body in inspect_engine() — fine, only Runtime reuse is under test


class _FakeRuntime:
    instances_created = 0

    def __init__(self, logger):
        self.logger = logger
        _FakeRuntime.instances_created += 1

    def deserialize_cuda_engine(self, data):
        return _FakeEngine()


def _install_fake_tensorrt(monkeypatch):
    _FakeRuntime.instances_created = 0
    fake_trt = types.ModuleType("tensorrt")
    fake_trt.Logger = _FakeLogger
    fake_trt.Runtime = _FakeRuntime
    monkeypatch.setitem(sys.modules, "tensorrt", fake_trt)
    return fake_trt


@pytest.fixture(autouse=True)
def _reset_cached_runtime():
    """The module-level Runtime cache must never leak between tests."""
    model_detect._trt_runtime = None
    yield
    model_detect._trt_runtime = None


class TestTrtRuntimeCaching:
    def test_get_trt_runtime_constructs_once(self, monkeypatch):
        fake_trt = _install_fake_tensorrt(monkeypatch)
        first = model_detect._get_trt_runtime(fake_trt)
        second = model_detect._get_trt_runtime(fake_trt)
        assert first is second
        assert _FakeRuntime.instances_created == 1

    def test_inspect_engine_reuses_runtime_across_calls(self, monkeypatch, tmp_path):
        _install_fake_tensorrt(monkeypatch)
        engine_a = tmp_path / "a.engine"
        engine_b = tmp_path / "b.engine"
        engine_a.write_bytes(b"fake-engine-bytes-a")
        engine_b.write_bytes(b"fake-engine-bytes-b")

        model_detect.inspect_engine(str(engine_a))
        model_detect.inspect_engine(str(engine_b))

        # Two different engine files inspected, but only one Runtime built —
        # the exact fix for the repeated-logger-mismatch console spam.
        assert _FakeRuntime.instances_created == 1

    def test_inspect_engine_returns_expected_shape(self, monkeypatch, tmp_path):
        _install_fake_tensorrt(monkeypatch)
        engine_path = tmp_path / "model_fp16.engine"
        engine_path.write_bytes(b"fake-engine-bytes")

        info = model_detect.inspect_engine(str(engine_path))

        assert info["format"] == "TensorRT Engine"
        # Precision falls back to the filename convention when the (empty,
        # in this fake) tensor loop never sets it from a binding dtype.
        assert info["precision"] == "FP16"
