"""Convert ONNX models to TensorRT engine files.

Usage examples
--------------
# Build with ORT TRT EP (recommended — matches exactly what the app uses):
    src\\python\\python.exe src\\core\\convert_to_engine.py --model Model\\Roblox_8n.onnx

# Build with the tensorrt Python API directly:
    src\\python\\python.exe src\\core\\convert_to_engine.py --model Model\\Roblox_8n.onnx --method trt

# Print equivalent trtexec shell command and exit:
    src\\python\\python.exe src\\core\\convert_to_engine.py --model Model\\Roblox_8n.onnx --print-trtexec

# Custom output directory, FP32:
    src\\python\\python.exe src\\core\\convert_to_engine.py --model Model\\CS2_8n.onnx --output trt_cache\\ --no-fp16

Notes
-----
- The 'ort' method (default) is the safest choice: it builds the engine the same
  way ORT's TensorrtExecutionProvider does at runtime, so the cached file is
  immediately reusable by the app without a second build pass.

- The 'trt' method uses the tensorrt Python API directly and gives more control
  over builder flags.  Use it for debugging or when the ORT EP build fails.

- First build takes 1–5 minutes depending on GPU speed and optimization level.
  Subsequent runs load the cached .engine file in under a second.
"""

from __future__ import annotations

import argparse
import os
import sys

# Allow running as a standalone script from either the project root or src/core/
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_SCRIPT_DIR)
_PROJECT_ROOT = os.path.dirname(_SRC_DIR)
for _d in (_SRC_DIR, _PROJECT_ROOT):
    if _d not in sys.path:
        sys.path.insert(0, _d)


# ── DLL pre-registration (must happen before any tensorrt / onnxruntime import) ──

def _register_dll_dirs() -> None:
    """Register CUDA and TensorRT DLL directories on Windows."""
    if sys.platform != "win32":
        return
    try:
        import site
        site_dirs: list[str] = list(site.getsitepackages())
        try:
            site_dirs.append(site.getusersitepackages())
        except (AttributeError, NotImplementedError):
            pass

        for sp in site_dirs:
            # TensorRT DLLs (tensorrt-cu12-libs wheel)
            trt_libs = os.path.join(sp, "tensorrt_libs")
            if os.path.isdir(trt_libs):
                os.environ["PATH"] = f"{trt_libs};{os.environ.get('PATH', '')}"
                try:
                    os.add_dll_directory(trt_libs)
                except (AttributeError, OSError):
                    pass

            # CUDA runtime DLLs (nvidia-*-cu12 wheels)
            for sub in ("cuda_runtime", "cublas", "cudnn", "cufft", "curand", "cusolver", "cusparse"):
                bin_dir = os.path.join(sp, "nvidia", sub, "bin")
                if os.path.isdir(bin_dir):
                    os.environ["PATH"] = f"{bin_dir};{os.environ.get('PATH', '')}"
                    try:
                        os.add_dll_directory(bin_dir)
                    except (AttributeError, OSError):
                        pass
    except Exception as exc:
        print(f"[WARN] DLL pre-registration failed: {exc}")


_register_dll_dirs()


# ── Build methods ─────────────────────────────────────────────────────────────

def build_engine_via_trt_api(
    onnx_path: str,
    output_path: str,
    fp16: bool = True,
    workspace_mb: int = 2048,
    input_name: str = "images",
    input_shape: tuple[int, ...] = (1, 3, 640, 640),
) -> bool:
    """Build a TRT engine using the tensorrt Python API.

    Produces a single serialized .engine file that can be loaded by
    trt.Runtime or inspected with Polygraphy / trtexec.

    Args:
        onnx_path:    Source .onnx file path.
        output_path:  Destination .engine file path.
        fp16:         Enable FP16 precision (recommended for RTX GPUs).
        workspace_mb: Builder GPU memory budget in MiB.
        input_name:   ONNX input tensor name (default "images" for YOLO).
        input_shape:  Fixed input shape (batch, C, H, W).

    Returns:
        True on success.
    """
    try:
        import tensorrt as trt
    except ImportError:
        print("[ERROR] tensorrt package is not installed.")
        print("        Run: src\\python\\python.exe src\\install_tensorrt_local.py")
        return False

    print(f"[TRT] TensorRT {trt.__version__}")
    print(f"[TRT] Source : {onnx_path}")
    print(f"[TRT] Output : {output_path}")
    print(f"[TRT] FP16   : {fp16}   Workspace: {workspace_mb} MiB")

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"[TRT] Parse error {i}: {parser.get_error(i)}")
            return False

    print(f"[TRT] ONNX parsed — {network.num_layers} layers")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, workspace_mb * 1024 * 1024
    )

    if fp16:
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("[TRT] FP16 enabled")
        else:
            print("[WARN] FP16 requested but platform_has_fast_fp16=False — falling back to FP32")

    # Static-shape profile (required even for fixed-size networks in TRT 8+)
    profile = builder.create_optimization_profile()
    profile.set_shape(input_name, input_shape, input_shape, input_shape)
    config.add_optimization_profile(profile)

    print("[TRT] Building engine — this may take 1–5 minutes...")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        print("[TRT] build_serialized_network returned None — check for ONNX op compatibility errors above")
        return False

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(serialized)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[TRT] Engine saved: {output_path}  ({size_mb:.1f} MiB)")
    return True


def build_engine_via_ort(
    onnx_path: str,
    cache_dir: str,
    fp16: bool = True,
    workspace_mb: int = 2048,
) -> bool:
    """Trigger TRT engine build through ORT's TensorrtExecutionProvider.

    ORT caches the engine automatically on first session creation.  This
    function forces that build upfront by creating a session and running one
    dummy inference — identical to what happens on app startup.

    Args:
        onnx_path:   Source .onnx file path.
        cache_dir:   Directory where ORT writes the .engine cache files.
        fp16:        Enable FP16 precision.
        workspace_mb: Builder GPU memory budget in MiB.

    Returns:
        True on success.
    """
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError as exc:
        print(f"[ERROR] Missing dependency: {exc}")
        return False

    available = ort.get_available_providers()
    if "TensorrtExecutionProvider" not in available:
        print("[ERROR] TensorrtExecutionProvider is not available in this ORT build.")
        print(f"        Available providers: {available}")
        print("        Run: src\\python\\python.exe src\\install_tensorrt_local.py")
        return False

    os.makedirs(cache_dir, exist_ok=True)

    providers = [
        (
            "TensorrtExecutionProvider",
            {
                "trt_fp16_enable": fp16,
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": cache_dir,
                "trt_timing_cache_enable": True,
                "trt_timing_cache_path": cache_dir,
                "trt_max_workspace_size": workspace_mb * 1024 * 1024,
                "trt_builder_optimization_level": 3,
                "trt_auxiliary_streams": -1,
            },
        ),
        (
            "CUDAExecutionProvider",
            {
                "cudnn_conv_algo_search": "HEURISTIC",
                "do_copy_in_default_stream": True,
            },
        ),
        "CPUExecutionProvider",
    ]

    print(f"[ORT] Loading {onnx_path} with TensorrtExecutionProvider")
    print(f"[ORT] Cache dir : {cache_dir}")
    print(f"[ORT] FP16      : {fp16}   Workspace: {workspace_mb} MiB")
    print("[ORT] Building TRT engine — this may take 1–5 minutes...")

    try:
        sess = ort.InferenceSession(onnx_path, providers=providers)
    except Exception as exc:
        print(f"[ERROR] InferenceSession creation failed: {exc}")
        return False

    # One dummy inference to confirm the engine is functional
    inp = sess.get_inputs()[0]
    shape = tuple(d if isinstance(d, int) and d > 0 else 1 for d in inp.shape)
    dummy = np.zeros(shape, dtype=np.float32)
    try:
        sess.run(None, {inp.name: dummy})
    except Exception as exc:
        print(f"[WARN] Dummy inference raised an exception (engine may still be cached): {exc}")

    actual_provider = (sess.get_providers() or ["unknown"])[0]
    print(f"[ORT] Active provider : {actual_provider}")
    print(f"[ORT] Engine files written to : {cache_dir}")

    if actual_provider != "TensorrtExecutionProvider":
        print("[WARN] ORT fell back to a non-TRT provider.  Check CUDA/TRT installation.")
        return False

    return True


# ── trtexec reference command ─────────────────────────────────────────────────

def trtexec_command(
    onnx_path: str,
    output_path: str,
    fp16: bool = True,
    workspace_mb: int = 2048,
) -> str:
    """Return the equivalent trtexec shell command as a string."""
    parts = [
        "trtexec",
        f"--onnx={onnx_path}",
        f"--saveEngine={output_path}",
        f"--workspace={workspace_mb}",
        "--best",
    ]
    if fp16:
        parts.append("--fp16")
    return " ".join(parts)


# ── CLI entry point ───────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Convert an ONNX model to a TensorRT engine cache.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--model", required=True, help="Path to the .onnx model file")
    ap.add_argument(
        "--output", default=None,
        help="Output .engine file (trt method) or cache directory (ort method). "
             "Default: trt_cache/ in project root",
    )
    ap.add_argument("--no-fp16", action="store_true", help="Disable FP16, use FP32")
    ap.add_argument(
        "--workspace", type=int, default=2048,
        help="Builder GPU memory budget in MiB (default: 2048)",
    )
    ap.add_argument(
        "--method", choices=["ort", "trt"], default="ort",
        help="'ort' uses ORT TensorrtExecutionProvider (default); "
             "'trt' uses the tensorrt Python API directly",
    )
    ap.add_argument(
        "--print-trtexec", action="store_true",
        help="Print equivalent trtexec shell command and exit",
    )
    args = ap.parse_args()

    onnx_path = os.path.abspath(args.model)
    if not os.path.isfile(onnx_path):
        print(f"[ERROR] Model not found: {onnx_path}")
        sys.exit(1)

    fp16 = not args.no_fp16
    model_stem = os.path.splitext(os.path.basename(onnx_path))[0]
    precision_tag = "fp16" if fp16 else "fp32"
    default_cache = os.path.join(_PROJECT_ROOT, "trt_cache")

    if args.output:
        if args.output.endswith(".engine"):
            output_engine = os.path.abspath(args.output)
            output_dir = os.path.dirname(output_engine)
        else:
            output_dir = os.path.abspath(args.output)
            output_engine = os.path.join(output_dir, f"{model_stem}_{precision_tag}.engine")
    else:
        output_dir = default_cache
        output_engine = os.path.join(output_dir, f"{model_stem}_{precision_tag}.engine")

    if args.print_trtexec:
        print(trtexec_command(onnx_path, output_engine, fp16=fp16, workspace_mb=args.workspace))
        sys.exit(0)

    print(f"[CONV] Model    : {onnx_path}")
    print(f"[CONV] Output   : {output_engine if args.method == 'trt' else output_dir}")
    print(f"[CONV] FP16     : {fp16}")
    print(f"[CONV] Method   : {args.method}")
    print(f"[CONV] Workspace: {args.workspace} MiB")
    print()

    if args.method == "trt":
        ok = build_engine_via_trt_api(
            onnx_path, output_engine, fp16=fp16, workspace_mb=args.workspace
        )
    else:
        ok = build_engine_via_ort(
            onnx_path, output_dir, fp16=fp16, workspace_mb=args.workspace
        )

    if not ok:
        sys.exit(1)

    print()
    print("[CONV] Done.")
    print(f"[CONV] The app will load the cached engine from '{output_dir}' on next start.")


if __name__ == "__main__":
    main()
