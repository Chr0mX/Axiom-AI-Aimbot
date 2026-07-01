"""
Standalone UDP receive pipeline benchmark.

Isolates each stage of the UDP capture path so you can tell whether a low
observed FPS (e.g. "UDP Stream FPS 76" while the sender is configured for
120) is caused by:

  1. Real packet loss / network conditions   -> run in `live` mode while the
     real sender is running and watch `dropped_fps` in the log / status panel.
  2. JPEG decode CPU cost on this machine     -> run `decode-bench`.
  3. Python's raw ability to assemble frames
     from UDP packets (parsing + dict ops),
     independent of the network and decode    -> run `loopback-bench`.

Usage (run from repo root, or `python src/core/bench_udp.py <mode>`):

    python src/core/bench_udp.py decode-bench --width 1280 --height 720 --quality 80
    python src/core/bench_udp.py loopback-bench --fps 120 --duration 5
    python src/core/bench_udp.py loopback-bench --fps 120 --duration 5 --width 1280 --height 720

`decode-bench` prints the max frames/sec this CPU can run cv2.imdecode at,
with no networking involved at all — this is the hard ceiling for the
decode stage.

`loopback-bench` spins up a real UdpJpegReceiver bound to 127.0.0.1, then
fires synthetic frames at it (chunked exactly like the real sender protocol)
as fast as `--fps` requests, over the real loopback UDP stack, and reports
the achieved assembled-frame rate (recv_fps) and any drops. This isolates
Python/OS UDP-socket throughput from real-world WAN/Wi-Fi packet loss.

Compare the two numbers against your live `UDP Stream FPS` (recv_fps) and
`udp_dropped_fps` from the running app:
  - loopback-bench achieves your target fps with 0 drops, but live shows
    drops           -> real network packet loss (Wi-Fi, switch, driver).
  - loopback-bench itself can't reach your target fps                 -> Python/OS
    socket throughput ceiling on this machine (rare below a few hundred fps).
  - Both benches hit target fine, but decode-bench is close to your
    observed source_nominal_fps                                        -> JPEG
    decode is the bottleneck, not the network.
"""
from __future__ import annotations

import argparse
import socket
import struct
import sys
import time
from pathlib import Path

# ── sys.path so `core` is importable when run as a script ────────────────────
_HERE = Path(__file__).resolve().parent   # src/core/
_SRC = _HERE.parent                       # src/
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from core.udp_receiver import HEADER_FORMAT, HEADER_SIZE, UdpJpegReceiver  # noqa: E402


def _make_test_jpeg(width: int, height: int, quality: int) -> bytes:
    import cv2
    import numpy as np

    img = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return buf.tobytes()


def decode_bench(width: int, height: int, quality: int, duration: float) -> None:
    import cv2
    import numpy as np

    jpeg_bytes = _make_test_jpeg(width, height, quality)
    print(f"[decode-bench] {width}x{height} q={quality} -> {len(jpeg_bytes)} bytes/frame")
    print(f"[decode-bench] running cv2.imdecode for {duration:.1f}s ...")

    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    count = 0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < duration:
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError("decode failed")
        count += 1
    elapsed = time.perf_counter() - t0
    fps = count / elapsed
    print(f"[decode-bench] decoded {count} frames in {elapsed:.2f}s -> {fps:.1f} fps ceiling "
          f"(single-threaded, this CPU)")


def _send_frame(sock: socket.socket, addr, frame_id: int, jpeg_bytes: bytes, chunk_payload: int) -> None:
    total_size = len(jpeg_bytes)
    total_chunks = max(1, (total_size + chunk_payload - 1) // chunk_payload)
    for idx in range(total_chunks):
        start = idx * chunk_payload
        payload = jpeg_bytes[start:start + chunk_payload]
        header = struct.pack(HEADER_FORMAT, frame_id, total_size, idx, total_chunks, len(payload))
        sock.sendto(header + payload, addr)


def loopback_bench(width: int, height: int, quality: int, fps: float, duration: float, port: int) -> None:
    jpeg_bytes = _make_test_jpeg(width, height, quality)
    chunk_payload = 1400 - HEADER_SIZE  # matches typical MTU-safe chunk size
    n_chunks = max(1, (len(jpeg_bytes) + chunk_payload - 1) // chunk_payload)
    print(f"[loopback-bench] {width}x{height} q={quality} -> {len(jpeg_bytes)} bytes/frame, "
          f"{n_chunks} chunks/frame")
    print(f"[loopback-bench] target {fps:.1f} fps for {duration:.1f}s over 127.0.0.1:{port}")

    receiver = UdpJpegReceiver(bind_ip="127.0.0.1", bind_port=port)
    receiver.start()

    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    addr = ("127.0.0.1", port)

    interval = 1.0 / fps
    frame_id = 0
    t0 = time.perf_counter()
    next_send = t0
    while time.perf_counter() - t0 < duration:
        now = time.perf_counter()
        if now < next_send:
            time.sleep(max(0.0, next_send - now))
        _send_frame(sender, addr, frame_id, jpeg_bytes, chunk_payload)
        frame_id += 1
        next_send += interval

    # let the receiver drain in-flight packets
    time.sleep(0.5)
    elapsed = time.perf_counter() - t0
    sent_fps = frame_id / elapsed
    print(f"[loopback-bench] sent {frame_id} frames in {elapsed:.2f}s -> {sent_fps:.1f} fps requested")
    print(f"[loopback-bench] receiver assembled-frame rate (recv_fps): {receiver.recv_fps:.1f} fps")
    print(f"[loopback-bench] receiver dropped-frame rate (dropped_fps): {receiver.dropped_fps:.1f} fps")

    receiver.stop()
    sender.close()

    if receiver.recv_fps >= sent_fps * 0.95 and receiver.dropped_fps < 1.0:
        print("[loopback-bench] RESULT: local socket/assembly path keeps up fine — "
              "if your live UDP Stream FPS is still low, the bottleneck is real "
              "network loss, not this machine's Python/OS UDP stack.")
    else:
        print("[loopback-bench] RESULT: even on loopback (no real network), the "
              "receiver could not keep up with the target rate — this machine's "
              "CPU/OS UDP handling is the bottleneck, not network loss.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)

    p_decode = sub.add_parser("decode-bench", help="Measure raw JPEG decode fps ceiling (no networking)")
    p_decode.add_argument("--width", type=int, default=640)
    p_decode.add_argument("--height", type=int, default=640)
    p_decode.add_argument("--quality", type=int, default=80)
    p_decode.add_argument("--duration", type=float, default=3.0)

    p_loop = sub.add_parser("loopback-bench", help="Measure UDP assembly fps ceiling over 127.0.0.1")
    p_loop.add_argument("--width", type=int, default=640)
    p_loop.add_argument("--height", type=int, default=640)
    p_loop.add_argument("--quality", type=int, default=80)
    p_loop.add_argument("--fps", type=float, default=120.0)
    p_loop.add_argument("--duration", type=float, default=5.0)
    p_loop.add_argument("--port", type=int, default=5601)

    args = parser.parse_args()

    if args.mode == "decode-bench":
        decode_bench(args.width, args.height, args.quality, args.duration)
    elif args.mode == "loopback-bench":
        loopback_bench(args.width, args.height, args.quality, args.fps, args.duration, args.port)


if __name__ == "__main__":
    main()
