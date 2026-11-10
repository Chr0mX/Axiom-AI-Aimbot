"""
UDP JPEG stream receiver — matches the wire protocol used by the OBS
udp_stream_filter plugin.

Wire format per packet (big-endian):
    frame_id     : uint32  (4 bytes)  - increments per source frame
    total_size   : uint32  (4 bytes)  - total JPEG size across all chunks
    chunk_index  : uint16  (2 bytes)  - 0-based index of this chunk
    total_chunks : uint16  (2 bytes)  - total chunks for this frame
    chunk_size   : uint16  (2 bytes)  - payload bytes in this packet
    -- 14 bytes header total --
    payload      : chunk_size bytes of raw JPEG data

A frame is only emitted once every chunk for its frame_id has arrived.
Incomplete frames are dropped after `frame_timeout` seconds (handles lost
UDP packets so a missing chunk doesn't leak memory forever).
"""

import os
import collections
import socket
import struct
import threading
import time

HEADER_FORMAT = ">IIHHH"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)  # 14 bytes

# Eviction (time.time() + full dict scan) throttled to this interval instead
# of running on every packet — at 120fps with dozens of chunks/frame, running
# it per-packet burns CPU on the recv thread that's needed to keep up with
# incoming sockets, which is a direct contributor to dropped/incomplete frames.
_EVICT_INTERVAL = 0.25

# How far behind the published frame a completed frame may be and still count
# as a late straggler (discarded). Stragglers are bounded by frame_timeout x
# fps (~120 at the 1s default / 120fps); anything further back is the sender
# having restarted its counter, which must be accepted, not discarded — else
# every new frame reads as "older" until the new counter overtakes the old.
_REORDER_WINDOW = 512


def _is_newer_frame_id(frame_id, latest_id):
    """Whether *frame_id* should replace the published *latest_id*."""
    if latest_id is None:
        return True
    # Signed 32-bit difference, so a uint32 wrap reads as newer.
    delta = (frame_id - latest_id) & 0xFFFFFFFF
    if delta >= 0x80000000:
        delta -= 0x100000000
    return delta > 0 or delta < -_REORDER_WINDOW


def _boost_thread_priority() -> None:
    """Best-effort: raise this thread's OS priority above normal so a busy
    GPU/inference process doesn't starve the UDP receive thread of CPU time
    and cause the OS socket buffer to overflow (packet loss)."""
    if os.name != "nt":
        return
    try:
        import ctypes
        THREAD_PRIORITY_HIGHEST = 2
        ctypes.windll.kernel32.SetThreadPriority(
            ctypes.windll.kernel32.GetCurrentThread(), THREAD_PRIORITY_HIGHEST
        )
    except Exception:
        pass


_STATS_WINDOW = 120          # frames kept for snapshot_stats()
_STATS_STALE_S = 3.0         # no frame for this long -> the stream counts as stopped
_MAX_PLAUSIBLE_GAP = 1000    # a frame-id jump bigger than this is a sender restart, not loss


class UdpJpegReceiver:
    def __init__(self, bind_ip="0.0.0.0", bind_port=5600, recv_buffer_size=65536,
                 frame_timeout=1.0):
        self.bind_ip = bind_ip
        self.bind_port = bind_port
        self.recv_buffer_size = recv_buffer_size
        self.frame_timeout = frame_timeout

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # 4 MB — a 1 MB kernel buffer can overflow during brief scheduling
        # stalls at 120fps (bursts of dozens of ~1.4 KB chunks/frame), which
        # silently drops packets and shows up as incomplete/dropped frames.
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
        self.sock.settimeout(1.0)  # ensures recvfrom wakes up periodically on Windows
        self.sock.bind((self.bind_ip, self.bind_port))

        self._partial_frames = {}  # frame_id -> dict(chunks, total_chunks, first_seen)
        self._running = False
        self._thread = None

        self._lock = threading.Lock()
        self._latest_frame = None      # most recent completed JPEG bytes
        self._latest_frame_id = None
        self._new_frame_event = threading.Event()
        self.recv_fps: float = 0.0     # assembled frames/sec from sender
        self.dropped_fps: float = 0.0  # incomplete frames evicted/sec (packet loss)
        # Frames that assembled successfully but arrived after a newer frame
        # had already been published, so they were discarded rather than
        # rewinding the consumer to older content. Monotonic; a steadily
        # climbing value means chunks are being reordered or delayed enough
        # that frames finish out of order.
        self.stale_frames_dropped: int = 0
        self._dfps_count = 0
        self._dfps_t0 = time.time()
        # Per-frame arrival record for the Speed Test's "UDP stream" block (see snapshot_stats()):
        # (perf_counter arrival, assembly ms, whole frames lost before it, arrived out of order).
        self._frame_log = collections.deque(maxlen=_STATS_WINDOW)

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        # Wake any thread blocked in get_latest_frame(block=True) immediately.
        self._new_frame_event.set()
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            self.sock.close()
        except OSError:
            pass
        if self._thread:
            self._thread.join(timeout=2.0)

    def get_latest_frame_with_id(self, block=False, timeout=None):
        """Return (jpeg_bytes, frame_id) atomically. Both None if no frame yet.

        If block=True, waits up to *timeout* seconds for _new_frame_event before
        reading. The event is cleared after wait() returns so the next call will
        block again until the next assembled frame.
        """
        if block:
            self._new_frame_event.wait(timeout)
            self._new_frame_event.clear()
        with self._lock:
            return self._latest_frame, self._latest_frame_id

    def _recv_loop(self):
        _boost_thread_priority()
        _rfps_count = 0
        _rfps_t0 = time.time()
        _last_evict = time.time()
        # Even the throttled eviction check still called time.time() on every
        # single packet just to compare against the interval — at ~35k
        # packets/sec (e.g. 1080p JPEGs split into hundreds of chunks/frame)
        # that syscall alone is measurable CPU. Only actually check wall-clock
        # time every _EVICT_CHECK_EVERY packets; the 250ms throttle doesn't
        # need sub-millisecond precision.
        _EVICT_CHECK_EVERY = 32
        _packets_since_evict_check = 0
        while self._running:
            try:
                packet, _addr = self.sock.recvfrom(self.recv_buffer_size)
            except TimeoutError:
                continue  # 1-second poll timeout — check _running and loop
            except OSError:
                break  # socket closed or shut down

            if len(packet) < HEADER_SIZE:
                continue

            # unpack_from reads directly from `packet` without first slicing out
            # a copy of the header bytes (struct.unpack needed that slice).
            frame_id, total_size, chunk_index, total_chunks, chunk_size = struct.unpack_from(
                HEADER_FORMAT, packet
            )
            payload = packet[HEADER_SIZE:HEADER_SIZE + chunk_size]

            entry = self._partial_frames.get(frame_id)
            if entry is None:
                entry = {
                    "chunks": {},
                    "total_chunks": total_chunks,
                    "total_size": total_size,
                    "first_seen": time.time(),
                }
                self._partial_frames[frame_id] = entry

            entry["chunks"][chunk_index] = payload

            if len(entry["chunks"]) == entry["total_chunks"]:
                try:
                    jpeg_bytes = b"".join(
                        entry["chunks"][i] for i in range(entry["total_chunks"])
                    )
                except KeyError:
                    # Duplicate/out-of-range chunk indices satisfied the count
                    # while leaving a hole in the sequence (corrupt/malformed
                    # packet) — drop this frame instead of crashing the thread.
                    del self._partial_frames[frame_id]
                    continue
                # Only publish if this frame is actually newer than what's
                # already there. Frames don't necessarily complete in order:
                # if frame N is missing a chunk, N+1 can assemble first, and
                # then N's straggler arrives and completes N — publishing it
                # unconditionally would hand the consumer an *older* frame
                # than the one it already has, i.e. visible backwards jitter.
                #
                # frame_id is a uint32 that wraps, and resets to 0 whenever
                # the OBS sender restarts — see _is_newer_frame_id().
                is_newer = _is_newer_frame_id(frame_id, self._latest_frame_id)
                self._record_frame(frame_id, entry["first_seen"], is_newer)

                del self._partial_frames[frame_id]

                if is_newer:
                    with self._lock:
                        self._latest_frame = jpeg_bytes
                        self._latest_frame_id = frame_id
                    self._new_frame_event.set()
                else:
                    self.stale_frames_dropped += 1

                # Counts assembly rate, which is what recv_fps claims to
                # measure — so a late-but-complete frame still counts here
                # even though it was too old to publish.
                _rfps_count += 1
                _now = time.time()
                _elapsed = _now - _rfps_t0
                if _elapsed >= 1.0:
                    self.recv_fps = _rfps_count / _elapsed
                    _rfps_count = 0
                    _rfps_t0 = _now

            _packets_since_evict_check += 1
            if _packets_since_evict_check >= _EVICT_CHECK_EVERY:
                _packets_since_evict_check = 0
                now_t = time.time()
                if now_t - _last_evict >= _EVICT_INTERVAL:
                    _last_evict = now_t
                    self._evict_stale_frames(now_t)

    def _record_frame(self, frame_id, first_seen, is_newer, now=None) -> None:
        """Log one assembled frame (once per frame, not per packet). Must run before
        ``_latest_frame_id`` is updated: the id gap to the previous frame is the number of whole
        frames the sender produced that never arrived."""
        lost = 0
        last = self._latest_frame_id
        if is_newer and last is not None:
            gap = (frame_id - last - 1) & 0xFFFFFFFF
            if gap < _MAX_PLAUSIBLE_GAP:        # a huge jump is a sender restart, not loss
                lost = gap
        self._frame_log.append((time.perf_counter() if now is None else now,
                                max(0.0, (time.time() - first_seen) * 1000.0),
                                lost, not is_newer))

    def snapshot_stats(self, now=None):
        """Quality of the UDP video stream over the last ~120 frames, in ms / percent.

        The UDP analogue of a ping test: how evenly frames arrive (``jitter_ms``), how spread out
        one frame's packets are (``assembly_ms``), and how many frames were lost or reordered.
        These are *relative* timings measured on this host only; one-way delay needs synchronised
        clocks and is not claimed.  ``{"active": False}`` while no frames are arriving."""
        now = time.perf_counter() if now is None else now
        try:
            log = list(self._frame_log)
        except RuntimeError:
            return {"active": False}
        if len(log) < 3 or now - log[-1][0] > _STATS_STALE_S:
            return {"active": False}
        ordered = [e for e in log if not e[3]]
        intervals = [(b[0] - a[0]) * 1000.0 for a, b in zip(ordered, ordered[1:])]
        if not intervals:
            return {"active": False}
        srt = sorted(intervals)
        mid = len(srt) >> 1
        median = srt[mid] if len(srt) % 2 else (srt[mid - 1] + srt[mid]) / 2.0
        diffs = [abs(b - a) for a, b in zip(intervals, intervals[1:])]
        lost = sum(e[2] for e in log)
        assembly = sorted(e[1] for e in log)
        return {
            "active": True,
            "frames": len(log),
            "interval_ms": round(median, 2),
            "interval_min_ms": round(srt[0], 2),
            "interval_max_ms": round(srt[-1], 2),
            "jitter_ms": round(sum(diffs) / len(diffs), 2) if diffs else 0.0,
            "assembly_ms": round(assembly[len(assembly) >> 1], 2),
            "assembly_max_ms": round(assembly[-1], 2),
            "lost_frames": lost,
            "loss_pct": round(100.0 * lost / (lost + len(log)), 2),
            "reordered": sum(1 for e in log if e[3]),
        }

    def _evict_stale_frames(self, now=None):
        now = now if now is not None else time.time()
        stale = [
            fid for fid, e in self._partial_frames.items()
            if now - e["first_seen"] > self.frame_timeout
        ]
        self._dfps_count += len(stale)
        _elapsed = now - self._dfps_t0
        if _elapsed >= 1.0:
            self.dropped_fps = self._dfps_count / _elapsed
            self._dfps_count = 0
            self._dfps_t0 = now
        for fid in stale:
            del self._partial_frames[fid]
