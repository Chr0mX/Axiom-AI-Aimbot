from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np

from core.convert_to_engine import build_engine_via_trt_api
from core.ndi_config_loader import PipelineConfig

log = logging.getLogger(__name__)


class TRTEngineError(RuntimeError):
    pass


class TRTEngine:
    """TensorRT FP16 inference engine.

    On first init, builds the engine from the ONNX model and caches the
    .engine file.  Subsequent inits deserialize the cached file (~1 second).
    Uses pycuda for synchronous H2D → execute → D2H buffer management.
    """

    def __init__(self, cfg: PipelineConfig) -> None:
        try:
            import tensorrt as trt  # type: ignore[import-not-found]
            import pycuda.autoinit  # type: ignore[import-not-found]  # noqa: F401
            import pycuda.driver as cuda  # type: ignore[import-not-found]
        except ImportError as exc:
            raise TRTEngineError(f"TensorRT/pycuda not installed: {exc}") from exc

        self._trt = trt
        self._cuda = cuda

        # __file__ is src/core/trt_engine.py → 3 dirnames = project root
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        onnx_path = os.path.normpath(os.path.join(project_root, cfg.model_path))
        cache_dir = os.path.normpath(os.path.join(project_root, cfg.trt_cache_dir))
        os.makedirs(cache_dir, exist_ok=True)

        model_stem = os.path.splitext(os.path.basename(cfg.model_path))[0]
        engine_path = os.path.join(cache_dir, f"{model_stem}_fp16.engine")

        if not os.path.exists(engine_path):
            if not os.path.exists(onnx_path):
                raise TRTEngineError(f"ONNX model not found: {onnx_path}")
            log.info("[TRT] Building engine from %s — this takes 1–5 min on first run...", onnx_path)
            ok = build_engine_via_trt_api(
                onnx_path=onnx_path,
                output_path=engine_path,
                fp16=True,
                workspace_mb=cfg.trt_workspace_mb,
                input_name="images",
                input_shape=(1, 3, cfg.model_input_size, cfg.model_input_size),
            )
            if not ok:
                raise TRTEngineError(f"TRT engine build failed for {onnx_path}")
            log.info("[TRT] Engine saved to %s", engine_path)

        # Deserialize engine
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with open(engine_path, "rb") as f:
            engine_bytes = f.read()
        self._engine = runtime.deserialize_cuda_engine(engine_bytes)
        if self._engine is None:
            raise TRTEngineError(f"Failed to deserialize TRT engine: {engine_path}")

        self._context = self._engine.create_execution_context()

        # Identify input/output tensor names (TRT 8+ uses named tensors)
        self._input_name: str | None = None
        self._output_name: str | None = None
        for i in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(i)
            mode = self._engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self._input_name = name
            else:
                self._output_name = name

        if self._input_name is None or self._output_name is None:
            raise TRTEngineError("Could not identify input/output tensors in engine")

        # Allocate pinned host + device buffers
        in_shape = tuple(self._engine.get_tensor_shape(self._input_name))
        out_shape = tuple(self._engine.get_tensor_shape(self._output_name))

        self._h_input = cuda.pagelocked_empty(int(np.prod(in_shape)), dtype=np.float32)
        self._h_output = cuda.pagelocked_empty(int(np.prod(out_shape)), dtype=np.float32)
        self._d_input = cuda.mem_alloc(self._h_input.nbytes)
        self._d_output = cuda.mem_alloc(self._h_output.nbytes)
        self._out_shape = (1,) + out_shape[1:]  # e.g. (1, 84, 8400) for YOLOv8n

        self._stream = cuda.Stream()
        log.info("[TRT] Engine loaded. Input: %s  Output: %s", in_shape, out_shape)

    def run(self, blob: np.ndarray) -> list[np.ndarray]:
        """Run inference.

        Args:
            blob: float32 ndarray [1, 3, H, W].

        Returns:
            List with one element — raw output ndarray matching ORT output format.
        """
        cuda = self._cuda

        np.copyto(self._h_input, blob.ravel())
        cuda.memcpy_htod_async(self._d_input, self._h_input, self._stream)

        self._context.set_tensor_address(self._input_name, int(self._d_input))
        self._context.set_tensor_address(self._output_name, int(self._d_output))
        self._context.execute_async_v3(stream_handle=self._stream.handle)

        cuda.memcpy_dtoh_async(self._h_output, self._d_output, self._stream)
        self._stream.synchronize()

        return [self._h_output.reshape(self._out_shape)]
