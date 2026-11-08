"""What precision/provider is the model actually *running* with?

The Model Info card used to derive "Precision" from the inspected file only: an ONNX
shows none, and a cached TensorRT engine was picked with ``sorted(glob)[-1]`` -- so an
FP32 engine always out-sorted the FP16 one (``_fp32`` > ``_fp16``), untagged engines
defaulted to FP32, and nothing refreshed after the precision was switched.  The
selection (``trt_fp16_enabled``) and the live session (``loaded_model_fp16`` /
``loaded_model_path`` / ``current_provider``, published by ai_loop) are separate
facts; this module keeps them apart and says which one the card is showing.

Pure Python apart from the engine lookup (``session_utils.find_trt_engine_cache``).
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

TRT_PROVIDER = "TensorrtExecutionProvider"

_PROVIDER_NAMES = {
    TRT_PROVIDER: "TensorRT",
    "CUDAExecutionProvider": "CUDA",
    "DmlExecutionProvider": "DirectML",
    "CPUExecutionProvider": "CPU",
}


def provider_name(provider: str) -> str:
    """Short human name for an ONNX Runtime execution provider string."""
    if not provider:
        return ""
    return _PROVIDER_NAMES.get(provider, provider.replace("ExecutionProvider", "") or provider)


def _basename(path: str) -> str:
    """Basename that understands both Windows and POSIX separators (config paths use either)."""
    return (path or "").replace("\\", "/").rsplit("/", 1)[-1]


def _stem(path: str) -> str:
    return os.path.splitext(_basename(path))[0].lower()


def describe_runtime(config, model_path: str) -> Dict[str, Any]:
    """Facts about the model shown in the card (``model_path``) versus the live session.

    ``state``: ``running`` (this model is loaded at the selected precision), ``pending``
    (selected but not what is loaded yet -- e.g. waiting for a TensorRT build) or
    ``idle`` (nothing loaded / inference not started).  ``engine_fp16`` is the precision
    whose cached engine should be inspected: the *running* one when this model is
    loaded, else the selected one.
    """
    selected_fp16 = bool(getattr(config, "trt_fp16_enabled", False))
    loaded_path = str(getattr(config, "loaded_model_path", "") or "")
    loaded_fp16 = getattr(config, "loaded_model_fp16", None)
    provider = str(getattr(config, "current_provider", "") or "")
    is_trt = provider == TRT_PROVIDER

    is_loaded_model = bool(loaded_path) and _stem(loaded_path) == _stem(model_path)
    loaded_fp16_known = loaded_fp16 is not None
    running_fp16: Optional[bool] = bool(loaded_fp16) if (is_loaded_model and loaded_fp16_known) else None

    if not loaded_path:
        state = "idle"
    elif is_loaded_model and (not is_trt or running_fp16 == selected_fp16):
        state = "running"
    else:
        state = "pending"

    if is_loaded_model and is_trt and running_fp16 is not None:
        running_precision: Optional[str] = "FP16" if running_fp16 else "FP32"
    else:
        running_precision = None

    return {
        "model": _basename(model_path),
        "provider": provider,
        "provider_name": provider_name(provider),
        "is_tensorrt": is_trt,
        "selected_fp16": selected_fp16,
        "selected_precision": "FP16" if selected_fp16 else "FP32",
        "is_loaded_model": is_loaded_model,
        "loaded_model": _basename(loaded_path),
        "running_precision": running_precision,
        "state": state,
        # Which engine's precision tag to look for: what is running if this model is, else the selection.
        "engine_fp16": running_fp16 if running_fp16 is not None else selected_fp16,
    }


def resolve_engine_path(config, model_path: str, runtime: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """Cached TensorRT engine for ``model_path`` at the precision in use, or None.

    Only meaningful while TensorRT is the active provider.  Uses
    ``find_trt_engine_cache(..., fp16=...)`` so an FP16 selection no longer picks the
    FP32 engine by lexical order.
    """
    runtime = runtime or describe_runtime(config, model_path)
    if not runtime["is_tensorrt"]:
        return None
    try:
        from .session_utils import find_trt_engine_cache
        return find_trt_engine_cache(model_path, fp16=runtime["engine_fp16"])
    except Exception:
        return None


def precision_lines(runtime: Dict[str, Any]) -> list:
    """The 'what is used right now' lines appended below the file facts."""
    lines = []
    prov = runtime["provider_name"]
    if runtime["state"] == "idle":
        lines.append(f"Selected: {runtime['selected_precision']} — not loaded yet (inference not running)")
        return lines
    if runtime["is_tensorrt"] and runtime["running_precision"]:
        badge = "  ● RUNNING" if runtime["state"] == "running" else ""
        lines.append(f"Precision in use: {runtime['running_precision']} · TensorRT{badge}")
    elif runtime["is_loaded_model"]:
        lines.append(
            f"Precision in use: native model precision on {prov or 'ONNX Runtime'}  ● RUNNING"
            "  (the FP32/FP16 setting only applies to TensorRT)")
    else:
        lines.append(f"Running now: {runtime['loaded_model'] or 'another model'}"
                     + (f" ({prov})" if prov else ""))
    if runtime["state"] == "pending":
        if runtime["is_tensorrt"] and runtime["is_loaded_model"]:
            lines.append(
                f"Selected: {runtime['selected_precision']} — not running yet "
                f"(engine build / swap pending; still {runtime['running_precision']})")
        else:
            lines.append(
                f"Selected: {runtime['model']} at {runtime['selected_precision']} — not running yet "
                "(build / swap pending)")
    return lines


def compose_info_text(info: Dict[str, Any], runtime: Dict[str, Any]) -> str:
    """File facts (format / input / classes / size) + the precision-in-use lines.

    The file's own ``precision`` is deliberately NOT shown: it is derived from a filename
    tag and says nothing about the live session.
    """
    parts = []
    if info.get("format"):
        parts.append(info["format"])
    parts.append(f"Input: {info['input_size']}")
    if info.get("num_classes"):
        parts.append(f"Classes: {info['num_classes']}")
    if info.get("file_size"):
        parts.append(info["file_size"])
    return "\n".join(["  •  ".join(parts)] + precision_lines(runtime))


def runtime_signature(runtime: Dict[str, Any]) -> tuple:
    """Everything that changes the card's text besides the inspected file."""
    return (runtime["state"], runtime["selected_fp16"], runtime["running_precision"],
            runtime["provider"], runtime["loaded_model"])
