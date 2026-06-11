"""model_detect.py — inspect .onnx and .engine model files.

Usage:
    python src/model_detect.py <path/to/model.onnx>
    python src/model_detect.py <path/to/model.engine>
"""

from __future__ import annotations

import argparse
import os
import sys

# ── resolve path so this script can be run from anywhere ────────────────────
_SRC_DIR     = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SRC_DIR)

# ── if running with system Python, re-launch with the embedded interpreter ──
# The embedded Python at src/python/python.exe already has onnxruntime and
# its native DLLs wired up correctly — sys.path injection alone is not enough
# for native extension modules (.pyd) that depend on co-located DLLs.
_embedded_python = os.path.join(_SRC_DIR, "python", "python.exe")
if (
    os.name == "nt"
    and os.path.exists(_embedded_python)
    and os.path.abspath(sys.executable) != os.path.abspath(_embedded_python)
):
    import subprocess
    result = subprocess.run([_embedded_python, os.path.abspath(__file__)] + sys.argv[1:])
    sys.exit(result.returncode)

# ── from here we are running under the embedded Python (or non-Windows) ─────

# Inject embedded site-packages (belt-and-suspenders for edge cases)
_embedded_pkgs = os.path.join(_SRC_DIR, "python", "Lib", "site-packages")
if os.path.isdir(_embedded_pkgs) and _embedded_pkgs not in sys.path:
    sys.path.insert(0, _embedded_pkgs)

# Inject AxiomAI AppData packages (TensorRT lives here)
_localappdata = os.environ.get("LOCALAPPDATA", "")
if _localappdata:
    _axiom_pkgs = os.path.join(_localappdata, "AxiomAI", "site-packages")
    if os.path.isdir(_axiom_pkgs) and _axiom_pkgs not in sys.path:
        sys.path.insert(0, _axiom_pkgs)
    # Also add TRT DLL directory so Windows can find nvinfer*.dll
    _trt_libs = os.path.join(_axiom_pkgs, "tensorrt_libs")
    if os.path.isdir(_trt_libs) and hasattr(os, "add_dll_directory"):
        try:
            os.add_dll_directory(_trt_libs)
        except Exception:
            pass


# ── helpers ──────────────────────────────────────────────────────────────────

def _infer_classes(output_shape: tuple) -> int | None:
    """Infer number of classes from a YOLO output tensor shape.

    Handles two common layouts:
      Layout A  (batch, anchors, 4+cls)  — YOLOv8 default
      Layout B  (batch, 4+cls, anchors)  — transposed export
    Returns None when the shape is unrecognisable.
    """
    if len(output_shape) < 3:
        return None
    # Treat any None/dynamic dim as -1
    dims = [d if isinstance(d, int) and d is not None else -1 for d in output_shape]

    # Layout A: last dim is 4+cls, middle dim is anchors (large number)
    if dims[2] > 0 and dims[1] > 0 and dims[2] < dims[1]:
        cls = dims[2] - 4
        if 1 <= cls <= 1000:
            return cls

    # Layout B: second dim is 4+cls, last dim is anchors
    if dims[1] > 0 and dims[2] > 0 and dims[1] < dims[2]:
        cls = dims[1] - 4
        if 1 <= cls <= 1000:
            return cls

    # Fallback: try subtracting 5 (older YOLOv5 format with objectness score)
    for idx in (2, 1):
        if dims[idx] > 0:
            cls = dims[idx] - 5
            if 1 <= cls <= 1000:
                return cls

    return None


def _file_size_str(path: str) -> str:
    try:
        size = os.path.getsize(path)
        if size >= 1_000_000:
            return f"{size / 1_000_000:.1f} MB"
        return f"{size / 1_000:.1f} KB"
    except OSError:
        return "unknown"


# ── inspectors ───────────────────────────────────────────────────────────────

def inspect_onnx(path: str) -> dict:
    """Inspect an ONNX model file using onnxruntime."""
    try:
        import onnxruntime as ort
    except ImportError:
        raise RuntimeError("onnxruntime is not installed.")

    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    inp  = sess.get_inputs()[0]
    out  = sess.get_outputs()[0]

    input_shape  = tuple(inp.shape)
    output_shape = tuple(out.shape)

    # H and W are the last two dims of the input tensor
    h = input_shape[2] if len(input_shape) >= 4 else None
    w = input_shape[3] if len(input_shape) >= 4 else None
    input_size = f"{h}×{w}" if (h and w) else str(input_shape)

    # Anchors: the larger spatial dimension in the output
    num_anchors = None
    if len(output_shape) >= 3:
        d1, d2 = output_shape[1], output_shape[2]
        if isinstance(d1, int) and isinstance(d2, int):
            num_anchors = max(d1, d2)

    num_classes = _infer_classes(output_shape)

    return {
        "format":       "ONNX",
        "input_size":   input_size,
        "input_shape":  str(input_shape),
        "output_shape": str(output_shape),
        "num_anchors":  num_anchors,
        "num_classes":  num_classes,
        "precision":    None,
        "file_size":    _file_size_str(path),
    }


def inspect_engine(path: str) -> dict:
    """Inspect a TensorRT engine file."""
    try:
        import tensorrt as trt  # noqa: F401
    except ImportError:
        raise RuntimeError(
            "TensorRT is not installed or not found in AppData. "
            "Cannot inspect .engine files without TensorRT."
        )

    logger  = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)

    with open(path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())

    if engine is None:
        raise RuntimeError("Failed to deserialize engine — file may be corrupt or built with a different TRT version.")

    input_shape  = None
    output_shape = None
    precision    = "FP32"

    # TRT 8.5+ uses get_tensor_* API; fall back to binding API for older versions
    if hasattr(engine, "num_io_tensors"):
        for i in range(engine.num_io_tensors):
            name  = engine.get_tensor_name(i)
            shape = tuple(engine.get_tensor_shape(name))
            dtype = engine.get_tensor_dtype(name)
            mode  = engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT and input_shape is None:
                input_shape = shape
            elif mode == trt.TensorIOMode.OUTPUT and output_shape is None:
                output_shape = shape
                if dtype == trt.DataType.HALF:
                    precision = "FP16"
                elif dtype == trt.DataType.INT8:
                    precision = "INT8"
    else:
        for i in range(engine.num_bindings):
            shape = tuple(engine.get_binding_shape(i))
            dtype = engine.get_binding_dtype(i)
            if engine.binding_is_input(i) and input_shape is None:
                input_shape = shape
            elif not engine.binding_is_input(i) and output_shape is None:
                output_shape = shape
                if dtype == trt.DataType.HALF:
                    precision = "FP16"
                elif dtype == trt.DataType.INT8:
                    precision = "INT8"

    h = input_shape[2] if input_shape and len(input_shape) >= 4 else None
    w = input_shape[3] if input_shape and len(input_shape) >= 4 else None
    input_size = f"{h}×{w}" if (h and w) else str(input_shape)

    num_anchors = None
    if output_shape and len(output_shape) >= 3:
        d1, d2 = output_shape[1], output_shape[2]
        if isinstance(d1, int) and isinstance(d2, int):
            num_anchors = max(d1, d2)

    num_classes = _infer_classes(output_shape) if output_shape else None

    return {
        "format":       "TensorRT Engine",
        "input_size":   input_size,
        "input_shape":  str(input_shape),
        "output_shape": str(output_shape),
        "num_anchors":  num_anchors,
        "num_classes":  num_classes,
        "precision":    precision,
        "file_size":    _file_size_str(path),
    }


def inspect_model(path: str) -> dict:
    """Dispatch to the appropriate inspector based on file extension."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    ext = os.path.splitext(path)[1].lower()
    if ext == ".onnx":
        return inspect_onnx(path)
    elif ext == ".engine":
        return inspect_engine(path)
    else:
        raise ValueError(f"Unsupported format: '{ext}'  (expected .onnx or .engine)")


# ── printer ──────────────────────────────────────────────────────────────────

def print_model_info(path: str, info: dict) -> None:
    SEP = "━" * 48
    print(SEP)
    print(f"  Model Inspector")
    print(f"  File      : {os.path.basename(path)}")
    print(f"  Format    : {info['format']}")
    print(SEP)
    print(f"  Input size    : {info['input_size']}")
    print(f"  Input shape   : {info['input_shape']}")
    print(f"  Output shape  : {info['output_shape']}")
    if info["num_anchors"] is not None:
        print(f"  Anchors       : {info['num_anchors']}")
    if info["num_classes"] is not None:
        print(f"  Classes       : {info['num_classes']}  (inferred)")
    else:
        print(f"  Classes       : unknown")
    if info["precision"]:
        print(f"  Precision     : {info['precision']}")
    print(f"  File size     : {info['file_size']}")
    print(SEP)


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect an .onnx or .engine model file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=("Examples:\n"
                "  python src/model_detect.py Model/apex_8n.onnx\n"
                "  python src/model_detect.py trt_cache/model.engine"),
    )
    parser.add_argument("model", help="Path to .onnx or .engine file")
    args = parser.parse_args()

    # Resolve relative to CWD first, then to project root
    path = args.model
    if not os.path.isabs(path) and not os.path.exists(path):
        alt = os.path.join(_PROJECT_DIR, path)
        if os.path.exists(alt):
            path = alt

    try:
        info = inspect_model(path)
        print_model_info(path, info)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
