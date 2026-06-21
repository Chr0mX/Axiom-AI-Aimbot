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

import socket
import struct
import threading
import time

HEADER_FORMAT = ">IIHHH"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)  # 14 bytes


class UdpJpegReceiver:
    def __init__(self, bind_ip="0.0.0.0", bind_port=5600, recv_buffer_size=65536,
                 frame_timeout=1.0):
        self.bind_ip = bind_ip
        self.bind_port = bind_port
        self.recv_buffer_size = recv_buffer_size
        self.frame_timeout = frame_timeout

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024)
        self.sock.settimeout(1.0)  # ensures recvfrom wakes up periodically on Windows
        self.sock.bind((self.bind_ip, self.bind_port))

        self._partial_frames = {}  # frame_id -> dict(chunks, total_chunks, first_seen)
        self._running = False
        self._thread = None

        self._lock = threading.Lock()
        self._latest_frame = None      # most recent completed JPEG bytes
        self._latest_frame_id = None
        self._new_frame_event = threading.Event()

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

    def get_latest_frame(self, block=False, timeout=None):
        """
        Returns the most recently completed JPEG frame as raw bytes, or None
        if nothing has arrived yet. If block=True, waits (up to `timeout`
        seconds) for a *new* frame rather than returning a stale one.
        """
        if block:
            if self._new_frame_event.wait(timeout):
                self._new_frame_event.clear()
        with self._lock:
            return self._latest_frame

    def _recv_loop(self):
        while self._running:
            try:
                packet, _addr = self.sock.recvfrom(self.recv_buffer_size)
            except TimeoutError:
                continue  # 1-second poll timeout — check _running and loop
            except OSError:
                break  # socket closed or shut down

            if len(packet) < HEADER_SIZE:
                continue

            frame_id, total_size, chunk_index, total_chunks, chunk_size = struct.unpack(
                HEADER_FORMAT, packet[:HEADER_SIZE]
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
                jpeg_bytes = b"".join(
                    entry["chunks"][i] for i in range(entry["total_chunks"])
                )
                with self._lock:
                    self._latest_frame = jpeg_bytes
                    self._latest_frame_id = frame_id
                self._new_frame_event.set()
                del self._partial_frames[frame_id]

            self._evict_stale_frames()

    def _evict_stale_frames(self):
        now = time.time()
        stale = [
            fid for fid, e in self._partial_frames.items()
            if now - e["first_seen"] > self.frame_timeout
        ]
        for fid in stale:
            del self._partial_frames[fid]
