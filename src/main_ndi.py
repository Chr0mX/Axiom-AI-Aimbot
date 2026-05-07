"""NDI CV Bot Pipeline — headless entry point.

Usage (from project root):
    src\\python\\python.exe src\\main_ndi.py [--config src\\config_ndi.yaml]

Runs on the inference PC (GTX 1650).  No GUI, no imshow.
"""

from __future__ import annotations

import argparse
import logging
import logging.handlers
import os
import signal
import sys
import time

# src/ is added to sys.path automatically when run as src\main_ndi.py,
# but add it explicitly for direct invocation from the project root.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from core.logging_config import setup_logging
from core.ndi_config_loader import PipelineConfig, load_config
from core.ndi_receiver import NDIHeadlessReceiver
from core.frame_crop import FrameCropper
from core.trt_engine import TRTEngine, TRTEngineError
from core.onnx_fallback import OnnxFallback, OnnxCudaError
from core.ndi_postprocess import Postprocessor
from win_utils.makcu_serial import MakcuSerial

log = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NDI CV Bot Pipeline")
    p.add_argument(
        "--config",
        default=os.path.join(_THIS_DIR, "config_ndi.yaml"),
        help="Path to config_ndi.yaml (default: src/config_ndi.yaml)",
    )
    p.add_argument("--log-file", default=None, help="Optional log file path")
    p.add_argument("--log-level", default="INFO", help="Logging level (default: INFO)")
    return p.parse_args()


def _build_engine(cfg: PipelineConfig):
    """Try TRT first; fall back to ONNX CUDA on failure."""
    try:
        engine = TRTEngine(cfg)
        log.info("[Main] Using TensorRT FP16 engine")
        return engine
    except TRTEngineError as exc:
        log.warning("[Main] TRT unavailable (%s) — falling back to ONNX CUDA", exc)

    try:
        engine = OnnxFallback(cfg)
        log.info("[Main] Using ONNX Runtime CUDA fallback")
        return engine
    except OnnxCudaError as exc:
        raise RuntimeError(
            f"Neither TensorRT nor ONNX CUDA is available: {exc}\n"
            "Inference must run on GPU — CPU fallback is not permitted."
        ) from exc


def main() -> None:
    args = _parse_args()
    setup_logging(level=args.log_level)

    # Optional rotating log file
    if args.log_file:
        os.makedirs(os.path.dirname(os.path.abspath(args.log_file)), exist_ok=True)
        fh = logging.handlers.RotatingFileHandler(
            args.log_file, maxBytes=10 * 1024 * 1024, backupCount=3, encoding="utf-8"
        )
        fh.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        logging.getLogger().addHandler(fh)

    log.info("=" * 60)
    log.info("NDI CV Bot Pipeline starting")
    log.info("Config: %s", args.config)

    cfg = load_config(args.config)
    log.info(
        "crop=%dx%d  model=%s  com=%s  baud=%d",
        cfg.crop_size, cfg.crop_size,
        os.path.basename(cfg.model_path),
        cfg.com_port,
        cfg.baud_rate_target,
    )

    # ── Module init ────────────────────────────────────────────────────
    log.info("[Main] Connecting to NDI source '%s'...", cfg.ndi_source_name or "(auto)")
    receiver = NDIHeadlessReceiver(cfg)

    cropper = FrameCropper(cfg)

    log.info("[Main] Loading inference engine...")
    engine = _build_engine(cfg)

    postproc = Postprocessor(cfg)

    log.info("[Main] Connecting to MAKCU on %s...", cfg.com_port)
    mouse = MakcuSerial(cfg)
    if not mouse.connect():
        log.error("[Main] MAKCU connect failed — output disabled; inference will continue")

    # ── Shutdown handler ───────────────────────────────────────────────
    _running = [True]

    def _shutdown(sig, frame):  # noqa: ANN001
        log.info("[Main] Shutdown signal received")
        _running[0] = False

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # ── Latency EMA state ──────────────────────────────────────────────
    ema_alpha = 0.1
    ema_total = ema_grab = ema_crop = ema_infer = ema_post = 0.0
    last_log_time = time.perf_counter()
    frame_count = 0

    log.info("[Main] Entering main loop")

    # ── Hot loop ───────────────────────────────────────────────────────
    while _running[0]:
        t0 = time.perf_counter()

        # Stage 1: NDI grab
        frame = receiver.grab()
        if frame is None:
            time.sleep(0.001)
            continue
        t1 = time.perf_counter()

        # Stage 2: Crop + preprocess
        blob, offset_x, offset_y, lb_scale, _, _ = cropper.process(frame)
        t2 = time.perf_counter()

        # Stage 3: Inference (with hot-swap fallback)
        try:
            outputs = engine.run(blob)
        except Exception as exc:
            log.warning("[Main] Inference error (%s) — switching to ONNX fallback", exc)
            try:
                engine = OnnxFallback(cfg)
                outputs = engine.run(blob)
            except Exception as exc2:
                log.error("[Main] Fallback also failed: %s — skipping frame", exc2)
                continue
        t3 = time.perf_counter()

        # Stage 4: Postprocess → (dx, dy)
        result = postproc.compute(outputs, offset_x, offset_y, lb_scale)
        t4 = time.perf_counter()

        # Stage 5: MAKCU output
        if result is not None:
            if not mouse.is_connected():
                log.warning("[Main] MAKCU disconnected — attempting reconnect")
                mouse.reconnect()
            mouse.move(*result)

        # ── Latency tracking ────────────────────────────────────────────
        if cfg.enable_latency_log:
            total_ms  = (t4 - t0) * 1000.0
            grab_ms   = (t1 - t0) * 1000.0
            crop_ms   = (t2 - t1) * 1000.0
            infer_ms  = (t3 - t2) * 1000.0
            post_ms   = (t4 - t3) * 1000.0

            ema_total  = ema_alpha * total_ms  + (1 - ema_alpha) * ema_total
            ema_grab   = ema_alpha * grab_ms   + (1 - ema_alpha) * ema_grab
            ema_crop   = ema_alpha * crop_ms   + (1 - ema_alpha) * ema_crop
            ema_infer  = ema_alpha * infer_ms  + (1 - ema_alpha) * ema_infer
            ema_post   = ema_alpha * post_ms   + (1 - ema_alpha) * ema_post
            frame_count += 1

            now = time.perf_counter()
            if now - last_log_time >= cfg.latency_log_interval_s:
                fps = frame_count / (now - last_log_time)
                log.info(
                    "[Latency EMA] total=%.1fms  grab=%.1f  crop=%.1f  "
                    "infer=%.1f  post=%.1f  fps=%.1f",
                    ema_total, ema_grab, ema_crop, ema_infer, ema_post, fps,
                )
                last_log_time = now
                frame_count = 0

    # ── Cleanup ────────────────────────────────────────────────────────
    log.info("[Main] Shutting down...")
    mouse.disconnect()
    receiver.close()
    log.info("[Main] Done")


if __name__ == "__main__":
    main()
