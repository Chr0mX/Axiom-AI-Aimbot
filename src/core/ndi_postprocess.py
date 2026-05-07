from __future__ import annotations

from typing import Any

import numpy as np

from core.inference import non_max_suppression, postprocess_outputs
from core.ndi_config_loader import PipelineConfig


# Inlined from src/core/ai_aiming.py — avoids pulling in the full win_utils
# import chain (send_mouse_move, etc.) which that module loads at import time.
def _calculate_aim_target(
    box: list[float],
    aim_part: str,
    head_height_ratio: float,
) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    cx = x1 + bw * 0.5
    if aim_part == "head":
        ty = y1 + bh * head_height_ratio * 0.5
    else:
        head_h = bh * head_height_ratio
        ty = (y1 + head_h + y2) * 0.5
    return cx, ty


class Postprocessor:
    """Parse raw model output → (dx, dy) relative to screen center."""

    def __init__(self, cfg: PipelineConfig) -> None:
        self._cfg = cfg
        self._screen_cx = cfg.screen_width // 2   # 960 for 1920×1080
        self._screen_cy = cfg.screen_height // 2  # 540

    def compute(
        self,
        outputs: list[Any],
        offset_x: int,
        offset_y: int,
        lb_scale: float,
    ) -> tuple[int, int] | None:
        """Translate raw inference outputs to a relative (dx, dy) move.

        Args:
            outputs:  Raw engine output list from TRTEngine.run() or OnnxFallback.run().
            offset_x: Crop left edge in full-screen coords.
            offset_y: Crop top  edge in full-screen coords.
            lb_scale: Letterbox scale from preprocess_image (crop_size→model_input_size).

        Returns:
            (dx, dy) integers — relative movement from screen center — or None if
            no detections pass confidence threshold after NMS.
        """
        cfg = self._cfg
        boxes, confidences = postprocess_outputs(
            outputs,
            original_width=cfg.crop_size,
            original_height=cfg.crop_size,
            model_input_size=cfg.model_input_size,
            min_confidence=cfg.confidence_threshold,
            offset_x=offset_x,
            offset_y=offset_y,
            letterbox_scale=lb_scale,
            letterbox_pad_x=0,
            letterbox_pad_y=0,
        )

        if not boxes:
            return None

        boxes, confidences = non_max_suppression(boxes, confidences, cfg.nms_iou_threshold)

        if not boxes:
            return None

        # Select target closest to screen center
        best_box = min(
            boxes,
            key=lambda b: (
                (_calculate_aim_target(b, cfg.aim_part, cfg.head_height_ratio)[0] - self._screen_cx) ** 2
                + (_calculate_aim_target(b, cfg.aim_part, cfg.head_height_ratio)[1] - self._screen_cy) ** 2
            ),
        )

        tx, ty = _calculate_aim_target(best_box, cfg.aim_part, cfg.head_height_ratio)
        dx = int(round(tx)) - self._screen_cx
        dy = int(round(ty)) - self._screen_cy

        # Clamp to int16 range (MAKCU limit)
        dx = max(-32768, min(32767, dx))
        dy = max(-32768, min(32767, dy))

        return dx, dy
