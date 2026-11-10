"""Integration tests for core/udp_receiver.py — the MJPEG-over-UDP chunk
reassembly used by the 'udp' capture backend (OBS udp_stream_filter wire
protocol). Uses real UDP sockets (bound to 127.0.0.1:0, an OS-assigned free
port) rather than mocking, mirroring the WS-handshake test style already
used in test_esp_server.py.
"""

import socket
import struct
import time

import pytest

from core.udp_receiver import HEADER_FORMAT, HEADER_SIZE, UdpJpegReceiver, _is_newer_frame_id


def _make_packet(frame_id, total_size, chunk_index, total_chunks, payload):
    header = struct.pack(HEADER_FORMAT, frame_id, total_size, chunk_index, total_chunks, len(payload))
    return header + payload


@pytest.fixture
def receiver():
    r = UdpJpegReceiver(bind_ip="127.0.0.1", bind_port=0, frame_timeout=1.0)
    r.start()
    yield r
    r.stop()


def _client_socket():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    return s


def _send_frame(sock, addr, frame_id, chunks):
    """chunks: list of payload bytes; sent as one packet per chunk."""
    total = len(chunks)
    total_size = sum(len(c) for c in chunks)
    for i, c in enumerate(chunks):
        sock.sendto(_make_packet(frame_id, total_size, i, total, c), addr)


class TestHeaderFormat:
    def test_header_is_14_bytes(self):
        assert HEADER_SIZE == 14


class TestIsNewerFrameId:
    def test_first_frame_and_forward_steps(self):
        assert _is_newer_frame_id(5, None)
        assert _is_newer_frame_id(6, 5)
        assert not _is_newer_frame_id(5, 5)

    def test_recent_straggler_rejected(self):
        assert not _is_newer_frame_id(99, 100)
        # ~2.5s late at 120fps — still a straggler, well inside the window.
        assert not _is_newer_frame_id(1000 - 300, 1000)

    def test_uint32_wrap_is_forward(self):
        assert _is_newer_frame_id(2, 0xFFFFFFFE)

    def test_counter_reset_is_forward(self):
        assert _is_newer_frame_id(0, 50_000)
        assert _is_newer_frame_id(0, 2**31 - 1)


class TestFrameReassembly:
    def test_single_chunk_frame_reassembles(self, receiver):
        addr = ("127.0.0.1", receiver.sock.getsockname()[1])
        client = _client_socket()
        try:
            payload = b"\xff\xd8fake-jpeg-bytes\xff\xd9"
            _send_frame(client, addr, frame_id=1, chunks=[payload])
            jpeg, fid = receiver.get_latest_frame_with_id(block=True, timeout=2.0)
            assert fid == 1
            assert jpeg == payload
        finally:
            client.close()

    def test_multi_chunk_frame_reassembles_in_order(self, receiver):
        addr = ("127.0.0.1", receiver.sock.getsockname()[1])
        client = _client_socket()
        try:
            chunks = [b"AAAA", b"BBBB", b"CCCC"]
            _send_frame(client, addr, frame_id=7, chunks=chunks)
            jpeg, fid = receiver.get_latest_frame_with_id(block=True, timeout=2.0)
            assert fid == 7
            assert jpeg == b"AAAABBBBCCCC"
        finally:
            client.close()

    def test_out_of_order_chunks_still_reassemble_correctly(self, receiver):
        """Chunks may arrive out of network order; assembly must use
        chunk_index, not arrival order."""
        addr = ("127.0.0.1", receiver.sock.getsockname()[1])
        client = _client_socket()
        try:
            total = 3
            total_size = 12
            client.sendto(_make_packet(9, total_size, 2, total, b"CCCC"), addr)
            client.sendto(_make_packet(9, total_size, 0, total, b"AAAA"), addr)
            client.sendto(_make_packet(9, total_size, 1, total, b"BBBB"), addr)
            jpeg, fid = receiver.get_latest_frame_with_id(block=True, timeout=2.0)
            assert fid == 9
            assert jpeg == b"AAAABBBBCCCC"
        finally:
            client.close()


class TestMalformedChunkRecovery:
    def test_out_of_range_chunk_index_does_not_crash_receiver_thread(self, receiver):
        """Regression test for the KeyError fix.

        total_chunks=3, but the three packets received carry indices
        {0, 1, 5} instead of {0, 1, 2} — index 5 is out of range and index 2
        never arrives. len(entry["chunks"]) still reaches 3 (three distinct
        dict keys), which satisfies the "frame complete" check, but
        assembling via entry["chunks"][i] for i in range(3) then hits the
        missing key 2. Before the fix this raised an unhandled KeyError that
        permanently killed the daemon receive thread; now the corrupt frame
        is dropped and the thread keeps running.
        """
        addr = ("127.0.0.1", receiver.sock.getsockname()[1])
        client = _client_socket()
        try:
            total = 3
            total_size = 12
            client.sendto(_make_packet(12, total_size, 0, total, b"AAAA"), addr)
            client.sendto(_make_packet(12, total_size, 1, total, b"BBBB"), addr)
            client.sendto(_make_packet(12, total_size, 5, total, b"ZZZZ"), addr)  # out-of-range, not index 2
            time.sleep(0.2)

            # The real regression guard: the receive thread must still be
            # alive and able to process a subsequent well-formed frame — a
            # crashed thread would leave get_latest_frame_with_id returning
            # stale/None data forever after.
            assert receiver._thread.is_alive()
            _send_frame(client, addr, frame_id=13, chunks=[b"still-alive"])
            jpeg, fid = receiver.get_latest_frame_with_id(block=True, timeout=2.0)
            assert fid == 13
            assert jpeg == b"still-alive"
        finally:
            client.close()

    def test_undersized_packet_is_ignored(self, receiver):
        """A packet shorter than the 14-byte header must be dropped, not
        crash struct.unpack_from."""
        addr = ("127.0.0.1", receiver.sock.getsockname()[1])
        client = _client_socket()
        try:
            client.sendto(b"\x00\x01\x02", addr)  # 3 bytes, way under HEADER_SIZE
            time.sleep(0.2)
            assert receiver._thread.is_alive()
            # Receiver should still work normally afterwards.
            _send_frame(client, addr, frame_id=20, chunks=[b"ok"])
            jpeg, fid = receiver.get_latest_frame_with_id(block=True, timeout=2.0)
            assert fid == 20
            assert jpeg == b"ok"
        finally:
            client.close()


class TestFrameOrdering:
    """A frame that assembles late must not rewind the consumer to older
    content than it already has."""

    def test_late_completing_older_frame_does_not_replace_newer(self, receiver):
        addr = ("127.0.0.1", receiver.sock.getsockname()[1])
        client = _client_socket()
        try:
            # Frame 100 arrives incomplete: chunk 0 of 2, no chunk 1 yet.
            client.sendto(_make_packet(100, 8, 0, 2, b"OLD-"), addr)

            # Frame 101 arrives complete and is published.
            _send_frame(client, addr, frame_id=101, chunks=[b"NEW"])
            jpeg, fid = receiver.get_latest_frame_with_id(block=True, timeout=2.0)
            assert fid == 101 and jpeg == b"NEW"

            # Frame 100's straggler now lands, completing the OLDER frame.
            client.sendto(_make_packet(100, 8, 1, 2, b"DATA"), addr)
            time.sleep(0.3)

            # The consumer must still see frame 101, not the stale 100.
            jpeg, fid = receiver.get_latest_frame_with_id()
            assert fid == 101, "stale frame 100 replaced newer frame 101"
            assert jpeg == b"NEW"
            assert receiver.stale_frames_dropped == 1
        finally:
            client.close()

    def test_sender_restart_frame_id_reset_is_accepted(self, receiver):
        """A restarted OBS sender resets frame_id to 0. Treating that as
        'older' would strand the receiver on the last pre-restart frame, so
        the wraparound-aware comparison must accept it."""
        addr = ("127.0.0.1", receiver.sock.getsockname()[1])
        client = _client_socket()
        try:
            _send_frame(client, addr, frame_id=4_000_000_000, chunks=[b"before"])
            jpeg, fid = receiver.get_latest_frame_with_id(block=True, timeout=2.0)
            assert fid == 4_000_000_000

            # Sender restarts, counter back to 0. Unsigned delta would be a
            # huge positive number; the signed-32-bit reading makes it a
            # forward step, which is what we want.
            _send_frame(client, addr, frame_id=0, chunks=[b"after"])
            jpeg, fid = receiver.get_latest_frame_with_id(block=True, timeout=2.0)
            assert fid == 0
            assert jpeg == b"after"
        finally:
            client.close()

    def test_sender_restart_from_a_realistic_counter_is_accepted(self, receiver):
        """The case that actually happens: OBS restarted after minutes/hours
        of streaming (counter far below 2**31). The signed difference alone
        reads 0 as ~50k frames *older*, so every post-restart frame used to
        be dropped until the new counter caught back up."""
        addr = ("127.0.0.1", receiver.sock.getsockname()[1])
        client = _client_socket()
        try:
            _send_frame(client, addr, frame_id=50_000, chunks=[b"before"])
            jpeg, fid = receiver.get_latest_frame_with_id(block=True, timeout=2.0)
            assert fid == 50_000

            for new_id in (0, 1, 2):
                _send_frame(client, addr, frame_id=new_id, chunks=[b"after%d" % new_id])
                jpeg, fid = receiver.get_latest_frame_with_id(block=True, timeout=2.0)
                assert fid == new_id
                assert jpeg == b"after%d" % new_id
            assert receiver.stale_frames_dropped == 0
        finally:
            client.close()

    def test_duplicate_frame_id_is_not_republished(self, receiver):
        addr = ("127.0.0.1", receiver.sock.getsockname()[1])
        client = _client_socket()
        try:
            _send_frame(client, addr, frame_id=7, chunks=[b"first"])
            jpeg, fid = receiver.get_latest_frame_with_id(block=True, timeout=2.0)
            assert fid == 7 and jpeg == b"first"

            # Same id again with different content — equal is not newer.
            _send_frame(client, addr, frame_id=7, chunks=[b"again"])
            time.sleep(0.3)
            jpeg, fid = receiver.get_latest_frame_with_id()
            assert jpeg == b"first"
            assert receiver.stale_frames_dropped == 1
        finally:
            client.close()


class TestStreamStats:
    """snapshot_stats(): the Speed Test's UDP 'ping' (interval jitter, assembly, loss, reorder)."""

    @staticmethod
    def _feed(rx, frames):
        """frames: (frame_id, arrival time on the perf_counter clock, assembly ms)."""
        import time as _t
        for fid, at, asm_ms in frames:
            is_newer = _is_newer_frame_id(fid, rx._latest_frame_id)
            rx._record_frame(fid, _t.time() - asm_ms / 1000.0, is_newer, now=at)
            if is_newer:
                rx._latest_frame_id = fid

    def test_steady_stream_has_zero_jitter_and_loss(self, receiver):
        self._feed(receiver, [(i, 100.0 + i * 0.01, 1.0) for i in range(10)])
        s = receiver.snapshot_stats(now=100.0 + 9 * 0.01 + 0.1)
        assert s["active"] and s["frames"] == 10
        assert s["interval_ms"] == pytest.approx(10.0, abs=0.05)
        assert s["jitter_ms"] == pytest.approx(0.0, abs=0.05)
        assert s["loss_pct"] == 0.0 and s["lost_frames"] == 0 and s["reordered"] == 0
        assert s["assembly_ms"] == pytest.approx(1.0, abs=0.5)

    def test_uneven_arrivals_show_jitter(self, receiver):
        t, frames = 50.0, []
        for i, gap in enumerate([0, 10, 30, 10, 30, 10]):
            t += gap / 1000.0
            frames.append((i, t, 1.0))
        self._feed(receiver, frames)
        s = receiver.snapshot_stats(now=t + 0.05)
        assert s["jitter_ms"] > 15.0
        assert s["interval_max_ms"] == pytest.approx(30.0, abs=0.5)

    def test_missing_frame_ids_count_as_loss(self, receiver):
        # ids 0,1,2,(3,4 lost),5,6 -> 2 lost of 7
        self._feed(receiver, [(i, 10.0 + n * 0.01, 1.0) for n, i in enumerate([0, 1, 2, 5, 6])])
        s = receiver.snapshot_stats(now=10.0 + 0.05)
        assert s["lost_frames"] == 2
        assert s["loss_pct"] == pytest.approx(100 * 2 / 7, abs=0.1)

    def test_id_wrap_is_not_loss_and_restart_is_not_loss(self, receiver):
        self._feed(receiver, [(0xFFFFFFFE, 1.0, 1.0), (0xFFFFFFFF, 1.01, 1.0), (0, 1.02, 1.0), (1, 1.03, 1.0)])
        assert receiver.snapshot_stats(now=1.04)["lost_frames"] == 0
        self._feed(receiver, [(7_000_000, 1.05, 1.0)])      # sender restarted far away: not 7M lost frames
        assert receiver.snapshot_stats(now=1.06)["lost_frames"] == 0

    def test_late_frame_is_reordered_not_an_interval(self, receiver):
        self._feed(receiver, [(0, 1.00, 1), (1, 1.01, 1), (3, 1.02, 1), (2, 1.03, 1), (4, 1.04, 1)])
        s = receiver.snapshot_stats(now=1.05)
        assert s["reordered"] == 1 and s["lost_frames"] == 1

    def test_inactive_when_few_frames_or_stream_stopped(self, receiver):
        assert receiver.snapshot_stats() == {"active": False}
        self._feed(receiver, [(i, 5.0 + i * 0.01, 1.0) for i in range(5)])
        assert receiver.snapshot_stats(now=5.04 + 10.0) == {"active": False}

    def test_real_packets_populate_the_log(self, receiver):
        import time as _t
        addr = ("127.0.0.1", receiver.sock.getsockname()[1])
        sock = _client_socket()
        for fid in range(5):
            _send_frame(sock, addr, fid, [b"ab", b"cd"])
            _t.sleep(0.01)
        end = _t.time() + 2
        while _t.time() < end and len(receiver._frame_log) < 5:
            _t.sleep(0.01)
        assert len(receiver._frame_log) == 5
        assert receiver.snapshot_stats()["frames"] == 5
