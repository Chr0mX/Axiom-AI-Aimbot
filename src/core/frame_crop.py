from __future__ import annotations

import numpy as np

from core.inference import preprocess_image
from core.ndi_config_loader import PipelineConfig


class FrameCropper:
    """Center-crop a full-resolution frame and prepare the model input blob.

    The crop origin is fixed at init time from config — no per-frame recompute.
    The raw crop (e.g. 960×960) is passed directly to preprocess_image so that
    the correct letterbox scale is returned (scale = model_input_size / crop_size).
    Do NOT pre-resize to model_input_size before calling preprocess_image: that
    would trigger the fast path and return scale=1.0, breaking coord translation.
    """

    def __init__(self, cfg: PipelineConfig) -> None:
        self._crop_size = cfg.crop_size
        self._model_input_size = cfg.model_input_size
        self._offset_x = cfg.crop_offset_x  # (screen_width  - crop_size) // 2
        self._offset_y = cfg.crop_offset_y  # (screen_height - crop_size) // 2

    def process(
        self,
        frame: np.ndarray,
    ) -> tuple[np.ndarray, int, int, float, int, int]:
        """Crop center region and return blob + offset/scale metadata.

        Args:
            frame: Full-resolution BGRA ndarray (H, W, 4).

        Returns:
            (blob, offset_x, offset_y, lb_scale, lb_pad_x, lb_pad_y)
            - blob      : float32 [1, 3, model_input_size, model_input_size]
            - offset_x  : crop left edge in screen coords
            - offset_y  : crop top  edge in screen coords
            - lb_scale  : model_input_size / crop_size  (e.g. 640/960 ≈ 0.667)
            - lb_pad_x  : always 0 for square-to-square crops
            - lb_pad_y  : always 0 for square-to-square crops
        """
        ox, oy = self._offset_x, self._offset_y
        cs = self._crop_size

        crop = frame[oy : oy + cs, ox : ox + cs]

        # Pass raw crop directly — preprocess_image computes the correct scale.
        blob, lb_scale, lb_pad_x, lb_pad_y = preprocess_image(crop, self._model_input_size)

        return blob, ox, oy, lb_scale, lb_pad_x, lb_pad_y
