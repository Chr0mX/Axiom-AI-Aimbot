"""Standalone sender subprocess used by bench_udp.py's loopback-bench.

Runs as a genuinely separate OS process (not a thread in the same
interpreter) so it doesn't share a GIL with the receiver or the simulated
capture/inference contender threads -- matching the real deployment where
the OBS sender is an independent process from Axiom's Python receiver.
Not meant to be run directly.

BENCHMARK HARNESS — subprocess entry point for bench_udp.py.
"""
import socket
import struct
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

from core.udp_receiver import HEADER_FORMAT, HEADER_SIZE  # noqa: E402


def main() -> None:
    port = int(sys.argv[1])
    fps = float(sys.argv[2])
    duration = float(sys.argv[3])
    width = int(sys.argv[4])
    height = int(sys.argv[5])
    quality = int(sys.argv[6])
    image_path = sys.argv[7] if len(sys.argv) > 7 and sys.argv[7] else None

    import cv2
    import numpy as np

    if image_path:
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"failed to load image: {image_path}")
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
    else:
        base = np.zeros((height, width, 3), dtype=np.uint8)
        base[:, :, 0] = np.linspace(0, 255, width, dtype=np.uint8)[None, :]
        base[:, :, 1] = np.linspace(0, 255, height, dtype=np.uint8)[:, None]
        base[:, :, 2] = 128
        noise = np.random.randint(-20, 20, (height, width, 3))
        img = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    jpeg_bytes = buf.tobytes()

    chunk_payload = 1400 - HEADER_SIZE
    total_size = len(jpeg_bytes)
    n_chunks = max(1, (total_size + chunk_payload - 1) // chunk_payload)
    payloads = [jpeg_bytes[i * chunk_payload:(i + 1) * chunk_payload] for i in range(n_chunks)]

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    addr = ("127.0.0.1", port)

    interval = 1.0 / fps
    frame_id = 0
    t0 = time.perf_counter()
    next_send = t0
    while time.perf_counter() - t0 < duration:
        now = time.perf_counter()
        if now < next_send:
            time.sleep(max(0.0, next_send - now))
        for idx, payload in enumerate(payloads):
            header = struct.pack(HEADER_FORMAT, frame_id, total_size, idx, n_chunks, len(payload))
            sock.sendto(header + payload, addr)
        frame_id += 1
        next_send += interval

    elapsed = time.perf_counter() - t0
    print(f"SENT frame_id_count={frame_id} elapsed={elapsed:.2f} sent_fps={frame_id/elapsed:.1f} "
          f"bytes_per_frame={total_size} chunks_per_frame={n_chunks}", flush=True)


if __name__ == "__main__":
    main()
