from __future__ import annotations

import dataclasses
import os
from typing import Any

import yaml


@dataclasses.dataclass
class PipelineConfig:
    ndi_source_name: str = ""
    crop_size: int = 960
    model_input_size: int = 640
    model_path: str = "Model/apex_8n.onnx"
    trt_cache_dir: str = "ndi_pipeline/trt_cache"
    trt_workspace_mb: int = 2048
    confidence_threshold: float = 0.20
    nms_iou_threshold: float = 0.45
    com_port: str = "COM3"
    baud_rate_initial: int = 115200
    baud_rate_target: int = 4000000
    screen_width: int = 1920
    screen_height: int = 1080
    aim_part: str = "head"
    head_height_ratio: float = 0.26
    enable_latency_log: bool = False
    latency_log_interval_s: float = 1.0

    # Derived fields — set in __post_init__, not read from yaml
    crop_offset_x: int = dataclasses.field(init=False)
    crop_offset_y: int = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        self.crop_offset_x = (self.screen_width - self.crop_size) // 2
        self.crop_offset_y = (self.screen_height - self.crop_size) // 2
        self._validate()

    def _validate(self) -> None:
        if self.crop_size < self.model_input_size:
            raise ValueError(
                f"crop_size ({self.crop_size}) must be >= model_input_size ({self.model_input_size})"
            )
        if not (0.0 < self.confidence_threshold < 1.0):
            raise ValueError(f"confidence_threshold must be in (0, 1), got {self.confidence_threshold}")
        if self.crop_size > self.screen_width or self.crop_size > self.screen_height:
            raise ValueError(
                f"crop_size ({self.crop_size}) exceeds screen dimensions "
                f"({self.screen_width}x{self.screen_height})"
            )
        if self.aim_part not in ("head", "body"):
            raise ValueError(f"aim_part must be 'head' or 'body', got '{self.aim_part}'")


def load_config(path: str = "ndi_pipeline/config.yaml") -> PipelineConfig:
    """Load config.yaml and return a validated PipelineConfig."""
    abs_path = os.path.abspath(path)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"Config file not found: {abs_path}")

    with open(abs_path, "r", encoding="utf-8") as f:
        data: dict[str, Any] = yaml.safe_load(f) or {}

    # Extract only known fields; ignore extras silently
    known = {field.name for field in dataclasses.fields(PipelineConfig) if field.init}
    filtered = {k: v for k, v in data.items() if k in known}

    return PipelineConfig(**filtered)
