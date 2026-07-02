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

IMPORTANT: by default `--image` is unset and this generates a smooth
gradient + light noise as test content, NOT random noise. Pure random
noise is the worst possible case for JPEG (near-incompressible, forces
maximum-length Huffman codes) and decodes several times slower than real
game/desktop footage — using it overstates the real decode cost badly
(e.g. ~50fps for noise vs ~100-200+fps for realistic content at the same
1920x1080 resolution on the same CPU). Pass `--image <path to a real
screenshot>` for the most accurate reading against your actual content.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import threading
import time
from pathlib import Path

# ── sys.path so `core` is importable when run as a script ────────────────────
_HERE = Path(__file__).resolve().parent   # src/core/
_SRC = _HERE.parent                       # src/
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from core.udp_receiver import UdpJpegReceiver  # noqa: E402


def _make_test_jpeg(width: int, height: int, quality: int, image_path: "str | None" = None) -> bytes:
    """Build a test JPEG. Pure random noise is the WORST case for JPEG (near-
    incompressible, forces maximum-length Huffman codes) and decodes several
    times slower than real footage — using it alone badly overstates the real
    decode-time bottleneck. Default here is smooth gradient + light noise,
    much closer to real game/desktop content. Pass --image for a real
    screenshot from your actual game for the most accurate measurement."""
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
    return buf.tobytes()


def decode_bench(width: int, height: int, quality: int, duration: float,
                  image_path: "str | None" = None) -> None:
    import cv2
    import numpy as np

    jpeg_bytes = _make_test_jpeg(width, height, quality, image_path)
    src = image_path or "synthetic gradient+noise (NOT random noise — see module docstring)"
    print(f"[decode-bench] source: {src}")
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


def _contender_worker(stop_evt: threading.Event) -> None:
    """Mimics real ai_loop capture/preprocess/inference threads: cv2/numpy
    calls that release the GIL during their C implementation (unlike a
    pure-Python busy loop, which would badly overstate contention)."""
    import cv2
    import numpy as np

    frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    while not stop_evt.is_set():
        small = cv2.resize(frame, (320, 320), interpolation=cv2.INTER_LINEAR)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray.astype(np.float32).mean()


def loopback_bench(width: int, height: int, quality: int, fps: float, duration: float, port: int,
                    image_path: "str | None" = None, contenders: int = 0) -> None:
    # The sender runs as a SEPARATE OS PROCESS (_bench_udp_sender.py), not a
    # thread in this interpreter -- it must not share a GIL with the receiver
    # or the contender threads below, matching real deployment where OBS is
    # an independent process from Axiom's Python receiver. Running it as an
    # in-process thread here would let contention starve the sender too,
    # confounding the very effect this bench is trying to isolate.
    sender_script = str(Path(__file__).resolve().parent / "_bench_udp_sender.py")

    print(f"[loopback-bench] target {fps:.1f} fps for {duration:.1f}s over 127.0.0.1:{port}"
          + (f", with {contenders} simulated capture/inference threads competing for CPU"
             if contenders else ""))

    receiver = UdpJpegReceiver(bind_ip="127.0.0.1", bind_port=port)
    receiver.start()

    stop_evt = threading.Event()
    contender_threads = [threading.Thread(target=_contender_worker, args=(stop_evt,), daemon=True)
                          for _ in range(contenders)]
    for t in contender_threads:
        t.start()

    proc = subprocess.run(
        [sys.executable, sender_script, str(port), str(fps), str(duration),
         str(width), str(height), str(quality), image_path or ""],
        capture_output=True, text=True, timeout=duration + 10,
    )

    time.sleep(0.5)  # let the receiver drain in-flight packets
    stop_evt.set()
    for t in contender_threads:
        t.join(timeout=2.0)

    if proc.returncode != 0:
        print(f"[loopback-bench] sender process failed:\n{proc.stderr}")
        receiver.stop()
        return

    print(f"[loopback-bench] {proc.stdout.strip()}")
    sent_fps = fps
    for tok in proc.stdout.split():
        if tok.startswith("sent_fps="):
            sent_fps = float(tok.split("=", 1)[1])
    print(f"[loopback-bench] receiver assembled-frame rate (recv_fps): {receiver.recv_fps:.1f} fps")
    print(f"[loopback-bench] receiver dropped-frame rate (dropped_fps): {receiver.dropped_fps:.1f} fps")

    receiver.stop()

    keep_up_ratio = receiver.recv_fps / sent_fps if sent_fps > 0 else 0.0
    if receiver.dropped_fps < 1.0 and keep_up_ratio >= 0.98:
        print("[loopback-bench] RESULT: local socket/assembly path keeps up fully — "
              "if your live UDP Stream FPS is still low, the bottleneck is real "
              "network loss, not this machine's Python/OS UDP stack.")
    elif keep_up_ratio >= 0.90:
        print(f"[loopback-bench] RESULT: mostly keeping up ({keep_up_ratio*100:.0f}% of target, "
              f"{receiver.dropped_fps:.1f} dropped_fps) — minor contention effects, not a hard "
              "bottleneck. If live fps is well below this, look at real network loss instead.")
    else:
        print(f"[loopback-bench] RESULT: falling significantly behind ({keep_up_ratio*100:.0f}% of "
              f"target, {receiver.dropped_fps:.1f} dropped_fps) — this machine's CPU/OS UDP "
              "handling under this load level is a real bottleneck. "
              + ("Try --contenders 0 to isolate whether it's the simulated capture/"
                 "inference load causing this." if contenders else
                 "Try --contenders 3 to check if this gets WORSE under load similar "
                 "to real capture+inference threads running."))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)

    p_decode = sub.add_parser("decode-bench", help="Measure raw JPEG decode fps ceiling (no networking)")
    p_decode.add_argument("--width", type=int, default=640)
    p_decode.add_argument("--height", type=int, default=640)
    p_decode.add_argument("--quality", type=int, default=80)
    p_decode.add_argument("--duration", type=float, default=3.0)
    p_decode.add_argument("--image", type=str, default=None,
                           help="Path to a real screenshot to use instead of synthetic content "
                                "(most accurate — random noise or even synthetic gradients don't "
                                "match real game footage compressibility)")

    p_loop = sub.add_parser("loopback-bench", help="Measure UDP assembly fps ceiling over 127.0.0.1")
    p_loop.add_argument("--width", type=int, default=640)
    p_loop.add_argument("--height", type=int, default=640)
    p_loop.add_argument("--quality", type=int, default=80)
    p_loop.add_argument("--fps", type=float, default=120.0)
    p_loop.add_argument("--duration", type=float, default=5.0)
    p_loop.add_argument("--port", type=int, default=5601)
    p_loop.add_argument("--image", type=str, default=None)
    p_loop.add_argument("--contenders", type=int, default=0,
                         help="Number of simulated capture/preprocess/inference-style "
                              "threads (real cv2/numpy work, not busy-loops) to run "
                              "concurrently, to reproduce CPU contention like the real "
                              "app under load. Try --contenders 3 to match ai_loop's "
                              "capture+preprocess+inference threads.")

    args = parser.parse_args()

    if args.mode == "decode-bench":
        decode_bench(args.width, args.height, args.quality, args.duration, args.image)
    elif args.mode == "loopback-bench":
        loopback_bench(args.width, args.height, args.quality, args.fps, args.duration, args.port,
                        args.image, args.contenders)


if __name__ == "__main__":
    main()
