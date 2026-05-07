from __future__ import annotations

import logging
import os

import numpy as np

from ..config_loader import PipelineConfig

log = logging.getLogger(__name__)


class OnnxCudaError(RuntimeError):
    pass


class OnnxFallback:
    """ONNX Runtime CUDA fallback inference engine.

    Provides the same run() interface as TRTEngine.  Uses CUDAExecutionProvider
    only — never falls back to CPU.  Raises OnnxCudaError if CUDA EP is
    unavailable so the caller can handle it explicitly.
    """

    def __init__(self, cfg: PipelineConfig) -> None:
        try:
            import onnxruntime as ort  # type: ignore[import-not-found]
        except ImportError as exc:
            raise OnnxCudaError(f"onnxruntime not installed: {exc}") from exc

        available = ort.get_available_providers()
        if "CUDAExecutionProvider" not in available:
            raise OnnxCudaError(
                f"CUDAExecutionProvider not available. Available: {available}. "
                "Install onnxruntime-gpu and ensure CUDA drivers are present."
            )

        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        model_path = os.path.normpath(os.path.join(project_root, "..", cfg.model_path))
        if not os.path.exists(model_path):
            raise OnnxCudaError(f"ONNX model not found: {model_path}")

        providers = [
            (
                "CUDAExecutionProvider",
                {
                    "cudnn_conv_algo_search": "HEURISTIC",
                    "do_copy_in_default_stream": True,
                },
            )
        ]

        log.info("[ONNX] Loading model with CUDAExecutionProvider: %s", model_path)
        self._session = ort.InferenceSession(model_path, providers=providers)

        actual = self._session.get_providers()
        if actual and actual[0] != "CUDAExecutionProvider":
            raise OnnxCudaError(
                f"ORT fell back to '{actual[0]}' instead of CUDAExecutionProvider"
            )

        self._input_name: str = self._session.get_inputs()[0].name
        log.info("[ONNX] Session ready. Input: '%s'", self._input_name)

    def run(self, blob: np.ndarray) -> list[np.ndarray]:
        """Run inference.

        Args:
            blob: float32 ndarray [1, 3, H, W].

        Returns:
            List with one element — raw output ndarray matching TRTEngine format.
        """
        return self._session.run(None, {self._input_name: blob})
