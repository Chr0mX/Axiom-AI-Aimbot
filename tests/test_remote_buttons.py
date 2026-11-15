"""Tests for core.remote_buttons (UDP mouse-button receiver) — pure stdlib."""

import hashlib
import socket
import struct
import time

from core import remote_buttons as rb


def _pkt(mask, seq=1, token="", version=1, magic=b"AXMB"):
    tag = hashlib.sha256(token.encode()).digest()[:4] if token else b"\0\0\0\0"
    return struct.pack("<4sBBH4s", magic, version, mask, seq, tag)


class _Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def _listener(token="", clock=None):
    return rb.RemoteButtonListener("127.0.0.1", 0, token, clock or time.monotonic)


class TestParse:
    def test_valid(self):
        assert rb.parse_packet(_pkt(0x02, 7)) == (7, 0x02)

    def test_bad_magic(self):
        assert rb.parse_packet(_pkt(1, magic=b"XXXX")) is None

    def test_bad_version(self):
        assert rb.parse_packet(_pkt(1, version=2)) is None

    def test_short(self):
        assert rb.parse_packet(b"AXMB") is None

    def test_wrong_token(self):
        assert rb.parse_packet(_pkt(1, token="a"), "b") is None

    def test_token_ok(self):
        assert rb.parse_packet(_pkt(1, token="a"), "a") == (1, 1)

    def test_token_required_when_sender_has_none(self):
        assert rb.parse_packet(_pkt(1), "a") is None

    def test_mask_limited_to_five_bits(self):
        assert rb.parse_packet(_pkt(0xFF))[1] == 0x1F


class TestState:
    def test_held_per_button(self):
        lst = _listener()
        try:
            lst.feed(_pkt(0x02 | 0x10))
            assert lst.held("rmb") and lst.held("side2")
            assert not lst.held("lmb") and not lst.held("off") and not lst.held("bogus")
        finally:
            lst.stop()

    def test_stale_reads_zero(self):
        clk = _Clock()
        lst = _listener(clock=clk)
        try:
            lst.feed(_pkt(0x02))
            assert lst.held("rmb")
            clk.t += rb.STALE_AFTER_S + 0.01
            assert not lst.held("rmb")
        finally:
            lst.stop()

    def test_old_seq_ignored_and_wrap_accepted(self):
        lst = _listener()
        try:
            assert lst.feed(_pkt(0x01, 10))
            assert not lst.feed(_pkt(0x00, 9))
            assert lst.held("lmb")
            assert lst.feed(_pkt(0x00, 11))
            assert not lst.feed(_pkt(0x02, 0xFFFF))  # far "behind" 11, not a wrap
            assert lst.feed(_pkt(0x02, 0x7000))
            assert lst.feed(_pkt(0x02, 0xE000))
            assert lst.feed(_pkt(0x02, 0xFFFE))
            assert lst.feed(_pkt(0x04, 2))  # wrapped past 0xFFFF
            assert lst.held("mmb")
        finally:
            lst.stop()

    def test_sender_restart_after_stale_accepts_low_seq(self):
        clk = _Clock()
        lst = _listener(clock=clk)
        try:
            lst.feed(_pkt(0x02, 5000))
            clk.t += 5
            assert lst.feed(_pkt(0x02, 1))
            assert lst.held("rmb")
        finally:
            lst.stop()


class TestSocket:
    def test_loopback(self):
        lst = _listener()
        lst.start()
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.sendto(_pkt(0x02, 1), ("127.0.0.1", lst.port))
            s.close()
            deadline = time.time() + 2
            while time.time() < deadline and not lst.held("rmb"):
                time.sleep(0.01)
            assert lst.held("rmb")
        finally:
            lst.stop()


class TestConfig:
    def test_defaults_and_validation(self):
        from core.config import Config, _FIELD_MAP
        for k in ("remote_btn_enabled", "remote_btn_bind_ip", "remote_btn_bind_port",
                  "remote_btn_token", "remote_aim_button"):
            assert k in _FIELD_MAP
        c = Config()
        assert c.remote_btn_enabled is False and c.remote_btn_bind_port == 5602
        c.remote_aim_button = "bogus"
        c.remote_btn_bind_port = 99999
        c.validate_config()
        assert c.remote_aim_button == "rmb" and c.remote_btn_bind_port == 5602

    def test_token_excluded_from_snapshots(self):
        from core.config_manager import _SNAPSHOT_EXCLUDED_FIELDS
        assert "remote_btn_token" in _SNAPSHOT_EXCLUDED_FIELDS
