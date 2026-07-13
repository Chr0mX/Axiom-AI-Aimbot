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


def test_snapshot_boxes_unaffected_by_single_target_mode_reduction():
    """Regression guard for the Web ESP / single_target_mode fix.

    single_target_mode narrows config.latest_boxes down to just the locked
    target (for auto-fire/preview), but the Web ESP feed must keep showing
    every detection — the same set the in-game overlay draws — by reading
    config.latest_all_boxes/latest_all_confidences instead. The existing
    _FakeConfig fixture happens to set latest_boxes and latest_all_boxes to
    identical values, which can't actually distinguish "reads the all-boxes
    field" from "reads the reduced field" — this test deliberately makes
    them different (simulating single_target_mode being on) so a regression
    that reverts esp_server.py to reading latest_boxes would fail here.
    """
    class SingleTargetConfig(_FakeConfig):
        latest_boxes = [[100, 200, 150, 320]]  # single_target_mode-reduced
        latest_confidences = [0.91]
        latest_all_boxes = [
            [100, 200, 150, 320], [10.0, 20.0, 30.0, 40.0], [500.0, 500.0, 550.0, 600.0],
        ]
        latest_all_confidences = [0.91, 0.4, 0.7]

    esp_server._config = SingleTargetConfig()
    snap = esp_server._build_snapshot()
    assert len(snap["boxes"]) == 3
    assert snap["boxes"] == [[100, 200, 150, 320], [10, 20, 30, 40], [500, 500, 550, 600]]
    assert snap["confidences"] == [0.91, 0.4, 0.7]


def test_snapshot_udp_boxes_shifted_to_desktop_coordinates():
    """Regression guard for the UDP spatial-crop offset fix.

    A UDP stream is fed by an OBS filter that spatially crops a small
    sub-region out of the user's real desktop — latest_all_boxes'
    coordinates are expressed in that small crop's own 0-udp_width /
    0-udp_height space, not desktop space. Since there's no way to learn
    the crop's real position on the desktop from the stream itself, the
    snapshot assumes the crop is centered on the desktop (matching the
    aim logic's own implicit assumption — see ai_loop.py/ai_loop_utils.py)
    and shifts box/center coordinates by that centered offset, while
    "screen" reports the real desktop resolution. This is what lets the
    Web ESP overlay align with the user's actual full-screen game view
    instead of being stretched/misaligned relative to it.
    """
    class UdpStreamConfig(_FakeConfig):
        screenshot_method = "udp"
        udp_width = 320
        udp_height = 320
        crosshairX = 160  # centered within the 320x320 crop
        crosshairY = 160
        # Box coordinates in the small crop's own 0-320 coordinate space.
        latest_all_boxes = [[50, 60, 90, 140]]
        latest_all_confidences = [0.8]
        display_locked_box = [50, 60, 90, 140]

    esp_server._config = UdpStreamConfig()
    snap = esp_server._build_snapshot()
    # offset = ((1920-320)/2, (1080-320)/2) = (800, 380)
    assert snap["screen"] == {"w": 1920, "h": 1080}
    assert snap["center"] == {"x": 960, "y": 540}
    assert snap["boxes"] == [[850, 440, 890, 520]]
    assert snap["locked_box"] == [850, 440, 890, 520]


def test_snapshot_udp_no_offset_before_first_frame():
    """Before any UDP frame has arrived, udp_width/udp_height are 0
    (unset) — get_capture_dimensions() falls back to the desktop
    resolution itself, so the computed offset is exactly 0 (no shift)
    and "screen" is the desktop resolution, not a bogus 0x0 or a
    stale/wrong offset applied to boxes that don't exist yet anyway."""
    class UdpStreamNoFrameYetConfig(_FakeConfig):
        screenshot_method = "udp"
        udp_width = 0
        udp_height = 0
        crosshairX = 960
        crosshairY = 540

    esp_server._config = UdpStreamNoFrameYetConfig()
    snap = esp_server._build_snapshot()
    assert snap["screen"] == {"w": 1920, "h": 1080}
    assert snap["center"] == {"x": 960, "y": 540}


def test_snapshot_non_udp_backends_get_no_offset():
    """uvc/ndi/mss/dxcam backends have no separate 'real desktop the crop
    was taken from' concept — the captured frame IS what's being viewed —
    so they must never get an offset applied, even if their own capture
    resolution differs from the desktop."""
    class UvcConfig(_FakeConfig):
        screenshot_method = "uvc"
        uvc_width = 1280
        uvc_height = 720
        crosshairX = 640
        crosshairY = 360
        latest_all_boxes = [[10, 10, 20, 20]]
        latest_all_confidences = [0.5]

    esp_server._config = UvcConfig()
    snap = esp_server._build_snapshot()
    assert snap["screen"] == {"w": 1280, "h": 720}
    assert snap["center"] == {"x": 640, "y": 360}
    assert snap["boxes"] == [[10, 10, 20, 20]]


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
