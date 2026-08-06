#!/usr/bin/env python3
"""
test_directshow_capture.py — launchable smoke test for the 'directshow'
capture backend (src/core/screen_capture.py's DirectShowCapture), wired to
the vendored directshow_capture.dll (chr0mx/DirectShow-Capture-DLL).

This is deliberately NOT a copy of that repo's own python/test_capture.py —
that one exercises the raw DLL/reference binding in isolation and already
confirms the DLL itself works. This script instead goes through Axiom's
actual integration point: a real Config instance and the real
DirectShowCapture class, so it also catches integration-layer mistakes the
raw binding wouldn't (wrong config field name, wrong pixel_format mapping,
region-crop math, BGRA-vs-BGR contract, ...). Run this — not just the DLL
repo's own test script — before trusting the 'directshow' screenshot_method
in production.

Requires: Windows, opencv-python, numpy, and the vendored
src/core/native/directshow_capture.dll (already committed — no build step
needed unless you changed the DLL's C++ source).

Usage (from the repo root):
    python scripts/test_directshow_capture.py --list
    python scripts/test_directshow_capture.py --device-index 0
    python scripts/test_directshow_capture.py --device-index 0 --format nv12 --seconds 10
    python scripts/test_directshow_capture.py --device-substr "USB Video" \\
        --width 1920 --height 1080 --fps 60 --buffers 4 --save snapshot.png
"""
from __future__ import annotations

import argparse
import os
import sys
import time

# Mirrors tests/conftest.py's sys.path setup — no install step needed.
_SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--list", action="store_true", help="List connected DirectShow devices and exit")
    p.add_argument("--device-index", type=int, default=None, help="Device index from --list")
    p.add_argument("--device-substr", type=str, default=None, help="Case-insensitive friendly-name substring instead of an index")
    p.add_argument("--format", choices=["mjpeg", "nv12"], default="mjpeg", help="Pixel format (default: mjpeg)")
    p.add_argument("--width", type=int, default=0, help="Requested width (0 = device default)")
    p.add_argument("--height", type=int, default=0, help="Requested height (0 = device default)")
    p.add_argument("--fps", type=int, default=0, help="Requested fps (0 = device default)")
    p.add_argument("--buffers", type=int, default=4, help="Allocator buffer count (default: 4 — run DirectShow-Capture-DLL's benchmark/ to tune per device)")
    p.add_argument("--seconds", type=float, default=5.0, help="Capture duration (default: 5)")
    p.add_argument("--region", type=str, default=None,
                    help="Optional 'left,top,width,height' crop to exercise the region-crop path "
                         "(the same crop calculate_detection_region() would hand grab() at runtime)")
    p.add_argument("--save", type=str, default=None, help="Save the last captured frame to this path (e.g. snapshot.png)")
    return p.parse_args()


def _parse_region(text: str) -> "dict[str, int] | None":
    if not text:
        return None
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 4:
        raise ValueError(f"--region must be 'left,top,width,height', got: {text!r}")
    left, top, width, height = (int(p) for p in parts)
    return {"left": left, "top": top, "width": width, "height": height}


def print_device_list(names: "list[str]") -> None:
    if not names:
        print("No DirectShow devices found. Check the device is connected, its driver is "
              "installed, and it isn't already open in another application.")
        return
    print(f"{len(names)} device(s) found:")
    for i, name in enumerate(names):
        print(f"  [{i}] {name}")


def run_capture(args) -> int:
    from core.config import Config
    from core.screen_capture import DirectShowCapture

    config = Config()
    config.directshow_device_index = args.device_index if args.device_index is not None else -1
    config.directshow_device_substr = args.device_substr or ""
    config.directshow_pixel_format = args.format
    config.directshow_width = args.width
    config.directshow_height = args.height
    config.directshow_fps = args.fps
    config.directshow_buffer_count = args.buffers

    region = _parse_region(args.region) if args.region else None

    print(f"Opening device "
          f"{repr(args.device_substr) if args.device_substr else (args.device_index if args.device_index is not None else '(first available)')} "
          f"— format={args.format}, res={args.width or 'auto'}x{args.height or 'auto'}, "
          f"fps={args.fps or 'auto'}, buffers={args.buffers}, region={region or '(full frame)'}")

    try:
        backend = DirectShowCapture(config)
    except Exception as exc:
        print(f"FAILED to open device: {exc}")
        return 1

    print("Opened. Capturing (BGRA, matching Axiom's capture-backend contract)...")

    frame_count = 0
    first_frame_wait_start = time.perf_counter()
    first_frame_latency = None
    last_frame = None
    last_shape = None
    deltas = []
    last_t = None

    deadline = time.perf_counter() + args.seconds
    try:
        while time.perf_counter() < deadline:
            frame = backend.grab(region)
            now = time.perf_counter()

            if frame is None:
                time.sleep(0.001)
                continue

            if frame_count == 0:
                first_frame_latency = now - first_frame_wait_start
                last_shape = frame.shape

            if last_t is not None:
                deltas.append(now - last_t)
            last_t = now

            frame_count += 1
            last_frame = frame
    except Exception as exc:
        print(f"ERROR during capture: {exc}")
        return 1
    finally:
        backend.close()

    print()
    print("--- Result ---")
    if frame_count == 0:
        print("No frames were received in the capture window. Check the device isn't in use by "
              "another application, and that the requested format/resolution/fps is actually one "
              "it supports (run DirectShow-Capture-DLL's benchmark/ --list-devices and a format sweep).")
        return 1

    print(f"Frames captured : {frame_count}")
    print(f"Resolution      : {last_shape[1]}x{last_shape[0]} (BGRA, {last_shape[2]} channels)")
    print(f"First frame in  : {first_frame_latency * 1000:.1f} ms after open")
    if deltas:
        avg_dt = sum(deltas) / len(deltas)
        measured_fps = 1.0 / avg_dt if avg_dt > 0 else 0.0
        max_gap_ms = max(deltas) * 1000
        print(f"Measured fps    : {measured_fps:.1f} (host-observed, includes grab() polling overhead)")
        print(f"Max gap between frames: {max_gap_ms:.1f} ms")

    if last_frame.shape[2] != 4:
        print(f"WARNING: frame has {last_frame.shape[2]} channels, expected 4 (BGRA) — "
              "this violates the capture-backend contract every other Axiom backend follows.")

    if args.save and last_frame is not None:
        import cv2
        ok = cv2.imwrite(args.save, last_frame)
        if ok:
            print(f"Snapshot saved  : {args.save}")
        else:
            print(f"Snapshot save FAILED (check the path/extension): {args.save}")

    return 0


def main() -> int:
    args = parse_args()

    try:
        from core.screen_capture import list_directshow_device_names
    except ImportError as exc:
        print(f"ERROR: missing dependency ({exc}). This script needs numpy/opencv-python installed "
              "the same way the rest of Axiom does.")
        return 1

    names = list_directshow_device_names()
    if args.list:
        print_device_list(names)
        return 0

    if args.device_index is None and args.device_substr is None:
        print_device_list(names)
        if len(names) == 1:
            print("\nExactly one device found - using it.\n")
            args.device_index = 0
        elif len(names) == 0:
            print(
                "\nNo devices enumerated (or not running on Windows / DLL not found) — "
                "pass --device-index/--device-substr to try opening one anyway, or check "
                "src/core/native/README.md if the DLL itself is the problem."
            )
            return 1
        else:
            print("\nMultiple devices found - pick one with --device-index N or --device-substr.")
            return 1

    return run_capture(args)


if __name__ == "__main__":
    sys.exit(main())
