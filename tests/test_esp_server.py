"""Unit tests for the Web ESP server (core/esp_server.py).

Covers:
1. The state snapshot serializes to valid JSON from a Config-like object.
2. The WebSocket text-frame encoder produces a correct RFC 6455 header.
3. The WS handshake completes without authentication.
"""

import json
import socket

import pytest

from core import esp_server


class _FakeConfig:
    """Minimal stand-in carrying the fields the snapshot reads."""
    width = 1920
    height = 1080
    crosshairX = 960
    crosshairY = 540
    fov_size = 200
    detect_range_size = 320
    fov_circle_filter_enabled = False
    show_fov = True
    show_boxes = True
    box_full_rect = False
    show_detect_range = True
    show_confidence = True
    show_tracer_line = True
    show_crosshair = False
    crosshair_style = "dot"
    crosshair_size = 4
    crosshair_color_r = 255
    crosshair_color_g = 255
    crosshair_color_b = 255
    box_color_theme = "default"
    chroma_box_speed = 1.0
    aim_part = "head"
    head_height_ratio = 0.26
    min_confidence = 0.5
    latest_boxes = [[100, 200, 150, 320], [10.0, 20.0, 30.0, 40.0]]
    latest_confidences = [0.91, 0.4]
    latest_all_boxes = [[100, 200, 150, 320], [10.0, 20.0, 30.0, 40.0]]
    latest_all_confidences = [0.91, 0.4]
    display_locked_box = [100, 200, 150, 320]
    display_locked_box_is_decaying = False
    AimToggle = True
    web_esp_http_port = 8080
    web_esp_ws_port = 8765
    web_esp_fps = 60


def test_snapshot_serializes_to_json():
    esp_server._config = _FakeConfig()
    snap = esp_server._build_snapshot()
    # Round-trips through JSON without error
    encoded = json.dumps(snap)
    decoded = json.loads(encoded)
    assert decoded["screen"] == {"w": 1920, "h": 1080}
    assert decoded["boxes"] == [[100, 200, 150, 320], [10, 20, 30, 40]]
    assert decoded["confidences"] == [0.91, 0.4]
    assert decoded["settings"]["aim_part"] == "head"
    assert decoded["locked_box"] == [100, 200, 150, 320]
    assert decoded["active"] is True


def test_snapshot_handles_empty_and_missing():
    class Empty:
        pass
    esp_server._config = Empty()
    snap = esp_server._build_snapshot()
    json.dumps(snap)  # must not raise
    assert snap["boxes"] == []
    assert snap["locked_box"] is None


def test_ws_encode_text_short_frame():
    frame = esp_server._ws_encode_text("hi")
    assert frame[0] == 0x81          # FIN + text opcode
    assert frame[1] == 2             # length (no mask bit — server frames unmasked)
    assert frame[2:] == b"hi"


def test_ws_encode_text_medium_frame():
    payload = "x" * 200
    frame = esp_server._ws_encode_text(payload)
    assert frame[0] == 0x81
    assert frame[1] == 126           # 16-bit extended length marker
    assert int.from_bytes(frame[2:4], "big") == 200


def _do_handshake(request_bytes):
    a, b = socket.socketpair()
    try:
        b.sendall(request_bytes)
        ok = esp_server._ws_handshake(a)
        return ok
    finally:
        a.close()
        b.close()


def test_handshake_accepts_connection():
    req = (
        b"GET / HTTP/1.1\r\n"
        b"Upgrade: websocket\r\n"
        b"Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n\r\n"
    )
    assert _do_handshake(req) is True
