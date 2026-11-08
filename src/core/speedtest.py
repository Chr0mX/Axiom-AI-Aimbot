"""LAN speed-test helpers for the Web Control server (browser <-> Axiom host).

Pure Python (no FastAPI / Qt), so the sizing, payload and statistics rules are
testable anywhere; ``web_control_server`` only wires them to routes.
"""
from __future__ import annotations

import os
import statistics
from typing import Iterator, Sequence

MIB = 1024 * 1024
CHUNK_BYTES = 64 * 1024
DEFAULT_DOWNLOAD_MB = 8
MIN_DOWNLOAD_MB = 1
MAX_DOWNLOAD_MB = 64
UPLOAD_CAP_BYTES = 64 * MIB          # a longer body is refused (HTTP 413)

# One incompressible block, generated once: repeating it costs nothing per
# request and no transport/compression layer can shrink it.
_BLOCK = os.urandom(CHUNK_BYTES)


def clamp_mb(value, default: int = DEFAULT_DOWNLOAD_MB) -> int:
    """Requested download size in MiB, bounded to [MIN, MAX]; junk -> default."""
    try:
        mb = int(float(value))
    except (TypeError, ValueError, OverflowError):
        return default
    return max(MIN_DOWNLOAD_MB, min(MAX_DOWNLOAD_MB, mb))


def iter_payload(total_bytes: int) -> Iterator[bytes]:
    """Yield exactly ``total_bytes`` bytes in <= CHUNK_BYTES pieces."""
    remaining = max(0, int(total_bytes))
    while remaining > 0:
        n = CHUNK_BYTES if remaining >= CHUNK_BYTES else remaining
        yield _BLOCK if n == CHUNK_BYTES else _BLOCK[:n]
        remaining -= n


def mbps(num_bytes: float, seconds: float) -> float:
    """Megabits per second; 0.0 for a non-positive or non-finite duration."""
    try:
        if not (seconds > 0.0) or seconds == float("inf"):
            return 0.0
        return max(0.0, float(num_bytes)) * 8.0 / seconds / 1_000_000.0
    except (TypeError, ValueError):
        return 0.0


def summarize_pings(samples_ms: Sequence[float]) -> dict:
    """min / median / max / jitter of round-trip times in ms.

    Jitter is the mean absolute difference of successive samples (RFC 3550
    style).  Non-finite and negative samples are ignored; no samples -> zeros
    and ``count`` 0.
    """
    clean = [float(s) for s in samples_ms
             if isinstance(s, (int, float)) and s == s and 0.0 <= s < float("inf")]
    if not clean:
        return {"count": 0, "min": 0.0, "median": 0.0, "max": 0.0, "jitter": 0.0}
    diffs = [abs(b - a) for a, b in zip(clean, clean[1:])]
    return {
        "count": len(clean),
        "min": min(clean),
        "median": statistics.median(clean),
        "max": max(clean),
        "jitter": (sum(diffs) / len(diffs)) if diffs else 0.0,
    }
