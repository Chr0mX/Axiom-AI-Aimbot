"""Tests for core.remote_buttons (mouse-button receiver on GamepadEnhancer's Network API, TCP) - pure stdlib."""

import socket
import time

from core import remote_buttons as rb


class _Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def _listener(token="secret", clock=None):
    return rb.RemoteButtonListener("127.0.0.1", 0, token, clock or time.monotonic)


def _authed(lst):
    s = rb._Session()
    assert lst.handle_line("auth secret", s) == ("OK" + rb.PROMPT, False)
    return s


class TestProtocol:
    def test_auth_required_first(self):
        lst = _listener()
        try:
            s = rb._Session()
            assert lst.handle_line("km.mouse(2)", s) == ("ERR auth", True)
            assert lst.mask() == 0
        finally:
            lst.stop()

    def test_wrong_and_empty_token_rejected(self):
        lst = _listener()
        empty = _listener(token="")
        try:
            assert lst.handle_line("auth nope", rb._Session()) == ("ERR auth", True)
            assert empty.handle_line("auth ", rb._Session()) == ("ERR auth", True)
            assert empty.handle_line("auth anything", rb._Session()) == ("ERR auth", True)
        finally:
            lst.stop()
            empty.stop()

    def test_mouse_mask_set_and_query(self):
        lst = _listener()
        try:
            s = _authed(lst)
            assert lst.handle_line("km.mouse(18)", s) == (None, False)
            assert lst.held("rmb") and lst.held("side2")
            assert not lst.held("lmb") and not lst.held("off") and not lst.held("bogus")
            assert lst.handle_line("km.mouse()", s) == ("18" + rb.PROMPT, False)
        finally:
            lst.stop()

    def test_bad_mask_rejected(self):
        lst = _listener()
        try:
            s = _authed(lst)
            for bad in ("km.mouse(32)", "km.mouse(-1)", "km.mouse(x)", "km.mouse(1,2)", "nonsense"):
                assert lst.handle_line(bad, s) == ("ERR", False), bad
            assert lst.mask() == 0
        finally:
            lst.stop()

    def test_pad_commands_are_accepted_and_ignored(self):
        lst = _listener()
        try:
            s = _authed(lst)
            assert lst.handle_line("km.controller(14,16384,500)", s) == (None, False)
            assert lst.handle_line("km.version()", s)[0].startswith(rb.VERSION_TEXT)
            lst.handle_line("km.mouse(1)", s)
            assert lst.handle_line("km.controller_clear()", s) == (None, False)
            assert lst.mask() == 0          # clear releases the buttons
        finally:
            lst.stop()

    def test_overlong_line_closes(self):
        lst = _listener()
        try:
            assert lst.handle_line("x" * 300, _authed(lst)) == ("ERR", True)
        finally:
            lst.stop()


class TestState:
    def test_stale_reads_zero(self):
        clk = _Clock()
        lst = _listener(clock=clk)
        try:
            lst.set_mask(0x02)
            assert lst.held("rmb")
            clk.t += rb.STALE_AFTER_S + 0.01
            assert not lst.held("rmb")
        finally:
            lst.stop()

    def test_mask_limited_to_five_bits(self):
        lst = _listener()
        try:
            lst.set_mask(0xFF)
            assert lst.mask() == 0x1F
        finally:
            lst.stop()


class TestSocket:
    def _recv_until(self, sock, marker, timeout=2.0):
        sock.settimeout(timeout)
        data = b""
        while marker not in data:
            chunk = sock.recv(256)
            if not chunk:
                break
            data += chunk
        return data

    def test_loopback_and_release_on_disconnect(self):
        lst = _listener()
        lst.start()
        try:
            c = socket.create_connection(("127.0.0.1", lst.port), timeout=2)
            c.sendall(b"auth secret\n")
            assert self._recv_until(c, b">>> ").startswith(b"OK")
            c.sendall(b"km.controller(15,100)\nkm.mouse(2)\n")
            deadline = time.time() + 2
            while time.time() < deadline and not lst.held("rmb"):
                time.sleep(0.01)
            assert lst.held("rmb")
            c.close()
            deadline = time.time() + 2
            while time.time() < deadline and lst.held("rmb"):
                time.sleep(0.01)
            assert not lst.held("rmb")      # a dropped sender never leaves aim engaged
        finally:
            lst.stop()

    def test_wrong_token_over_socket_is_refused(self):
        lst = _listener()
        lst.start()
        try:
            c = socket.create_connection(("127.0.0.1", lst.port), timeout=2)
            c.sendall(b"auth wrong\n")
            assert b"ERR" in self._recv_until(c, b"\r\n")
            c.sendall(b"km.mouse(2)\n")
            time.sleep(0.1)
            assert not lst.held("rmb")
            c.close()
        finally:
            lst.stop()

    def test_newest_client_replaces_the_old_one(self):
        lst = _listener()
        lst.start()
        try:
            a = socket.create_connection(("127.0.0.1", lst.port), timeout=2)
            a.sendall(b"auth secret\n")
            self._recv_until(a, b">>> ")
            b = socket.create_connection(("127.0.0.1", lst.port), timeout=2)
            b.sendall(b"auth secret\n")
            self._recv_until(b, b">>> ")
            a.settimeout(2)
            assert a.recv(16) == b""        # the old connection was closed
            b.close()
            a.close()
        finally:
            lst.stop()


class TestConfig:
    def test_defaults_and_validation(self):
        from core.config import Config, _FIELD_MAP
        for k in ("remote_btn_enabled", "remote_btn_bind_ip", "remote_btn_bind_port",
                  "remote_btn_token", "remote_aim_button"):
            assert k in _FIELD_MAP
        c = Config()
        assert c.remote_btn_enabled is False and c.remote_btn_bind_port == 5555
        c.remote_aim_button = "bogus"
        c.remote_btn_bind_port = 99999
        c.validate_config()
        assert c.remote_aim_button == "rmb" and c.remote_btn_bind_port == 5555

    def test_token_excluded_from_snapshots(self):
        from core.config_manager import _SNAPSHOT_EXCLUDED_FIELDS
        assert "remote_btn_token" in _SNAPSHOT_EXCLUDED_FIELDS
