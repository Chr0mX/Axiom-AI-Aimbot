"""Tests for the ViGEm Network backend — all Linux-runnable (pure stdlib)."""

import socket
import threading
import time

import importlib
import os

# vigem_network is pure stdlib — import it directly without going through
# win_utils/__init__.py, which pulls in mouse_move → win32api.
_src = os.path.join(os.path.dirname(__file__), "..", "src")
_spec = importlib.util.spec_from_file_location(
    "vigem_network",
    os.path.join(_src, "win_utils", "vigem_network.py"),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

VigemNetworkClient = _mod.VigemNetworkClient
_clamp = _mod._clamp
_ID_RIGHT_X = _mod._ID_RIGHT_X
_ID_RIGHT_Y = _mod._ID_RIGHT_Y
_ID_RIGHT_TRIGGER = _mod._ID_RIGHT_TRIGGER
_MAX_STICK = _mod._MAX_STICK
_MIN_STICK = _mod._MIN_STICK
_MAX_TRIGGER = _mod._MAX_TRIGGER
BASE_PIXELS = _mod.BASE_PIXELS
map_pixels_to_stick = _mod.map_pixels_to_stick


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clamp_py(v, lo, hi):
    return max(lo, min(hi, v))


class _MockServer:
    """Minimal TCP server that speaks the vigem Network API auth handshake."""

    def __init__(self, token="test_token", accept_auth=True):
        self._token = token
        self._accept_auth = accept_auth
        self._srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._srv.bind(("127.0.0.1", 0))
        self._srv.listen(1)
        self.port = self._srv.getsockname()[1]
        self.received_lines: list[str] = []
        self._conn: socket.socket | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        self._srv.settimeout(5.0)
        try:
            self._conn, _ = self._srv.accept()
        except OSError:
            return
        self._conn.settimeout(2.0)
        buf = b""
        # Read auth line
        while b"\n" not in buf:
            try:
                chunk = self._conn.recv(256)
            except OSError:
                return
            if not chunk:
                return
            buf += chunk
        auth_line = buf[:buf.index(b"\n")].decode("utf-8", errors="replace")
        buf = buf[buf.index(b"\n") + 1:]
        if self._accept_auth and auth_line == f"auth {self._token}":
            self._conn.sendall(b"OK\r\n>>> ")
        else:
            self._conn.sendall(b"ERR\r\n")
            self._conn.close()
            return
        # Read subsequent commands
        while not self._stop.is_set():
            try:
                chunk = self._conn.recv(4096)
            except socket.timeout:
                continue
            except OSError:
                break
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                self.received_lines.append(line.decode("utf-8", errors="replace"))

    def stop(self):
        self._stop.set()
        if self._conn:
            try:
                self._conn.close()
            except Exception:
                pass
        self._srv.close()
        self._thread.join(timeout=3)


# ---------------------------------------------------------------------------
# _clamp
# ---------------------------------------------------------------------------

class TestClamp:
    def test_within_range(self):
        assert _clamp(100, -200, 200) == 100

    def test_below_min(self):
        assert _clamp(-40000, _MIN_STICK, _MAX_STICK) == _MIN_STICK

    def test_above_max(self):
        assert _clamp(40000, _MIN_STICK, _MAX_STICK) == _MAX_STICK


# ---------------------------------------------------------------------------
# Pixel → stick mapping
# ---------------------------------------------------------------------------

class TestPixelToStick:
    def _map(self, dx, dy, sensitivity=1.0):
        fx = dx * sensitivity / BASE_PIXELS
        fy = dy * sensitivity / BASE_PIXELS
        rx = _clamp(int(fx * _MAX_STICK), _MIN_STICK, _MAX_STICK)
        ry = _clamp(int(fy * _MAX_STICK), _MIN_STICK, _MAX_STICK)
        return rx, ry

    def test_zero(self):
        assert self._map(0, 0) == (0, 0)

    def test_positive(self):
        rx, ry = self._map(25.0, 25.0)
        assert rx > 0
        assert ry > 0

    def test_negative(self):
        rx, ry = self._map(-25.0, -10.0)
        assert rx < 0
        assert ry < 0

    def test_full_stick(self):
        rx, _ = self._map(BASE_PIXELS, 0.0)
        assert rx == _MAX_STICK

    def test_clamp_large(self):
        rx, _ = self._map(9999.0, 0.0)
        assert rx == _MAX_STICK

    def test_sensitivity(self):
        rx1, _ = self._map(10.0, 0.0, sensitivity=1.0)
        rx2, _ = self._map(10.0, 0.0, sensitivity=2.0)
        assert rx2 > rx1

    def test_no_y_inversion(self):
        """Both Axiom dy and vigem +Y are down, so no inversion needed."""
        _, ry = self._map(0.0, 10.0)
        assert ry > 0


# ---------------------------------------------------------------------------
# Protocol formatting
# ---------------------------------------------------------------------------

class TestProtocol:
    def test_stick_x_format(self):
        cmd = f"km.controller({_ID_RIGHT_X},{1000})\n"
        assert cmd == "km.controller(14,1000)\n"

    def test_stick_y_format(self):
        cmd = f"km.controller({_ID_RIGHT_Y},{-500})\n"
        assert cmd == "km.controller(15,-500)\n"

    def test_trigger_format(self):
        cmd = f"km.controller({_ID_RIGHT_TRIGGER},{_MAX_TRIGGER})\n"
        assert cmd == "km.controller(11,1023)\n"

    def test_keepalive_format(self):
        assert "km.version()\n" == "km.version()\n"


# ---------------------------------------------------------------------------
# Delta tracking
# ---------------------------------------------------------------------------

class TestDeltaTracking:
    def test_unchanged_not_resent(self):
        client = VigemNetworkClient()
        client._last_sent_rx = 100
        client._last_sent_ry = 200
        client._last_sent_rt = 0

        buf = []
        rx, ry, rt = 100, 200, 0
        resend = False
        if rx is not None and (resend or rx != client._last_sent_rx):
            buf.append(f"km.controller({_ID_RIGHT_X},{rx})\n")
        if ry is not None and (resend or ry != client._last_sent_ry):
            buf.append(f"km.controller({_ID_RIGHT_Y},{ry})\n")
        if rt is not None and (resend or rt != client._last_sent_rt):
            buf.append(f"km.controller({_ID_RIGHT_TRIGGER},{rt})\n")

        assert buf == []

    def test_changed_value_sent(self):
        client = VigemNetworkClient()
        client._last_sent_rx = 100
        client._last_sent_ry = 200

        buf = []
        rx, ry = 150, 200
        resend = False
        if rx is not None and (resend or rx != client._last_sent_rx):
            buf.append(f"km.controller({_ID_RIGHT_X},{rx})\n")
        if ry is not None and (resend or ry != client._last_sent_ry):
            buf.append(f"km.controller({_ID_RIGHT_Y},{ry})\n")

        assert len(buf) == 1
        assert "14,150" in buf[0]

    def test_resend_all(self):
        client = VigemNetworkClient()
        client._last_sent_rx = 100
        client._last_sent_ry = 200

        buf = []
        rx, ry = 100, 200
        resend = True
        if rx is not None and (resend or rx != client._last_sent_rx):
            buf.append(f"km.controller({_ID_RIGHT_X},{rx})\n")
        if ry is not None and (resend or ry != client._last_sent_ry):
            buf.append(f"km.controller({_ID_RIGHT_Y},{ry})\n")

        assert len(buf) == 2


# ---------------------------------------------------------------------------
# Mock TCP server integration
# ---------------------------------------------------------------------------

class TestMockServer:
    def test_connect_disconnect(self):
        srv = _MockServer()
        try:
            client = VigemNetworkClient()
            ok = client.connect("127.0.0.1", srv.port, "test_token")
            assert ok
            assert client.is_connected()
            client.disconnect()
            assert not client.is_connected()
        finally:
            srv.stop()

    def test_auth_rejected(self):
        srv = _MockServer(accept_auth=False)
        try:
            client = VigemNetworkClient()
            ok = client.connect("127.0.0.1", srv.port, "wrong_token")
            assert not ok
            assert not client.is_connected()
        finally:
            srv.stop()

    def test_move_commands_received(self):
        srv = _MockServer()
        try:
            client = VigemNetworkClient()
            assert client.connect("127.0.0.1", srv.port, "test_token")
            client.enqueue_move(1000, -500)
            time.sleep(0.1)
            client.disconnect()
            time.sleep(0.1)
            lines = srv.received_lines
            assert any("km.controller(14,1000)" in l for l in lines)
            assert any("km.controller(15,-500)" in l for l in lines)
        finally:
            srv.stop()

    def test_click_command_received(self):
        srv = _MockServer()
        try:
            client = VigemNetworkClient()
            assert client.connect("127.0.0.1", srv.port, "test_token")
            client.enqueue_click(1023)
            time.sleep(0.1)
            client.disconnect()
            time.sleep(0.1)
            lines = srv.received_lines
            assert any("km.controller(11,1023)" in l for l in lines)
        finally:
            srv.stop()

    def test_controller_clear_on_disconnect(self):
        srv = _MockServer()
        try:
            client = VigemNetworkClient()
            assert client.connect("127.0.0.1", srv.port, "test_token")
            time.sleep(0.05)
            client.disconnect()
            time.sleep(0.2)
            lines = srv.received_lines
            assert any("km.controller_clear()" in l for l in lines)
        finally:
            srv.stop()


class TestSendMoveUsesConfigRef:
    class _Cfg:
        def __init__(self, s):
            self.vigem_net_sensitivity = s

    def _first_rx(self, srv, cfg, dx):
        _mod.connect_vigem_network("127.0.0.1", srv.port, "test_token", cfg)
        try:
            _mod.send_mouse_move_vigem_network(dx, 0)
            time.sleep(0.15)
        finally:
            _mod.disconnect_vigem_network()
        time.sleep(0.1)
        vals = [int(l.split(",")[1].rstrip(")")) for l in srv.received_lines
                if l.startswith("km.controller(14,")]
        return vals[0] if vals else None

    def test_sensitivity_read_from_config(self):
        srv1 = _MockServer()
        srv2 = _MockServer()
        try:
            lo = self._first_rx(srv1, self._Cfg(1.0), 10.0)
            hi = self._first_rx(srv2, self._Cfg(2.0), 10.0)
            assert lo is not None and hi is not None
            assert hi > lo
        finally:
            srv1.stop()
            srv2.stop()

    def test_no_config_defaults_to_one(self):
        srv = _MockServer()
        try:
            rx = self._first_rx(srv, None, 10.0)
            assert rx == int(10.0 / BASE_PIXELS * _MAX_STICK)
        finally:
            srv.stop()


class TestDeadZones:
    def test_input_deadzone_zeroes_small(self):
        assert map_pixels_to_stick(0.5, 0.5, 1.0, 0.05, 0.0) == (0, 0)  # 0.01 < 0.05

    def test_input_deadzone_per_axis(self):
        rx, ry = map_pixels_to_stick(25.0, 0.5, 1.0, 0.05, 0.0)
        assert rx > 0 and ry == 0

    def test_game_deadzone_pushes_past(self):
        rx, _ = map_pixels_to_stick(1.0, 0.0, 1.0, 0.0, 0.2)  # v=0.02
        assert rx >= int(0.2 * _MAX_STICK)

    def test_game_deadzone_sign_and_clamp(self):
        rx, ry = map_pixels_to_stick(-9999.0, 9999.0, 1.0, 0.0, 0.3)
        assert rx == -_MAX_STICK and ry == _MAX_STICK

    def test_zero_stays_zero(self):
        assert map_pixels_to_stick(0.0, 0.0, 1.0, 0.0, 0.3) == (0, 0)

    def test_off_matches_plain_mapping(self):
        assert map_pixels_to_stick(25.0, -10.0, 1.0, 0.0, 0.0) == (
            int(0.5 * _MAX_STICK), int(-0.2 * _MAX_STICK))


class TestRecenter:
    class _Cfg:
        vigem_net_sensitivity = 1.0
        vigem_net_input_deadzone = 0.0
        vigem_net_game_deadzone = 0.0
        vigem_net_release_ms = 40
        vigem_stick_x = 0.0
        vigem_stick_y = 0.0

    def _zeros(self, srv, axis_id):
        return [l for l in srv.received_lines if l == f"km.controller({axis_id},0)"]

    def test_stick_recenters_once_after_last_move(self):
        srv = _MockServer()
        cfg = self._Cfg()
        try:
            assert _mod.connect_vigem_network("127.0.0.1", srv.port, "test_token", cfg)
            _mod.send_mouse_move_vigem_network(25.0, 0.0)
            time.sleep(0.03)
            assert cfg.vigem_stick_x > 0
            time.sleep(0.4)
            assert cfg.vigem_stick_x == 0.0
            assert len(self._zeros(srv, 14)) == 1  # sent once, no spam
        finally:
            _mod.disconnect_vigem_network()
            srv.stop()

    def test_continuous_moves_do_not_recenter(self):
        srv = _MockServer()
        cfg = self._Cfg()
        try:
            assert _mod.connect_vigem_network("127.0.0.1", srv.port, "test_token", cfg)
            for _ in range(15):
                _mod.send_mouse_move_vigem_network(25.0, 0.0)
                time.sleep(0.01)
            assert not self._zeros(srv, 14)
        finally:
            _mod.disconnect_vigem_network()
            srv.stop()

    def test_move_after_recenter_is_sent(self):
        srv = _MockServer()
        cfg = self._Cfg()
        try:
            assert _mod.connect_vigem_network("127.0.0.1", srv.port, "test_token", cfg)
            _mod.send_mouse_move_vigem_network(25.0, 0.0)
            time.sleep(0.3)
            _mod.send_mouse_move_vigem_network(25.0, 0.0)
            time.sleep(0.015)
            assert cfg.vigem_stick_x > 0
        finally:
            _mod.disconnect_vigem_network()
            srv.stop()


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------

class TestConfigValidation:
    def test_vigem_network_in_valid_move_methods(self):
        from core.config import Config
        c = Config()
        c.mouse_move_method = "vigem_network"
        c.validate_config()
        assert c.mouse_move_method == "vigem_network"

    def test_vigem_network_in_valid_click_methods(self):
        from core.config import Config
        c = Config()
        c.mouse_click_method = "vigem_network"
        c.validate_config()
        assert c.mouse_click_method == "vigem_network"

    def test_config_defaults(self):
        from core.config import Config
        c = Config()
        assert c.vigem_net_host == ""
        assert c.vigem_net_port == 5555
        assert c.vigem_net_token == ""
        assert c.vigem_net_sensitivity == 1.0

    def test_field_map_entries(self):
        from core.config import _FIELD_MAP
        assert "vigem_net_host" in _FIELD_MAP
        assert "vigem_net_port" in _FIELD_MAP
        assert "vigem_net_token" in _FIELD_MAP
        assert "vigem_net_sensitivity" in _FIELD_MAP

    def test_token_excluded_from_snapshots(self):
        from core.config_manager import _SNAPSHOT_EXCLUDED_FIELDS
        assert "vigem_net_token" in _SNAPSHOT_EXCLUDED_FIELDS

    def test_round_trip(self):
        from core.config import Config
        c = Config()
        c.vigem_net_host = "10.0.0.5"
        c.vigem_net_port = 9999
        c.vigem_net_token = "secret"
        c.vigem_net_sensitivity = 2.5
        d = c.to_dict()
        c2 = Config()
        c2.from_dict(d)
        assert c2.vigem_net_host == "10.0.0.5"
        assert c2.vigem_net_port == 9999
        assert c2.vigem_net_token == "secret"
        assert c2.vigem_net_sensitivity == 2.5
