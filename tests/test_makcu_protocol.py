"""MAK_API framing helpers — no serial hardware required.

Loaded by file path so importing it does not execute win_utils/__init__.py
(that package imports win32api).
"""

import importlib.util
import socket
import struct
import threading
from pathlib import Path

_path = Path(__file__).resolve().parents[1] / "src" / "win_utils" / "makcu_protocol.py"
_spec = importlib.util.spec_from_file_location("makcu_protocol_under_test", _path)
mp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mp)


def test_move_frame_clamps_and_round_trips():
    raw = mp.move_frame(40000, -40000)
    assert raw[:2] == mp.SYNC
    assert raw[4] == mp.CMD_MOVE
    dx, dy = struct.unpack_from("<hh", raw, 5)
    assert (dx, dy) == (32767, -32768)


def test_rejection_and_stream_enable():
    assert mp.is_rejection(b"\xff") is True
    assert mp.is_rejection(b"\x01") is False
    mouse = mp.stream_enable_frame(mp.STREAM_KIND_MOUSE, True)
    keys = mp.stream_enable_frame(mp.STREAM_KIND_KEYBOARD, False)
    assert mouse[4] == mp.CMD_BUTTONS and mouse[5] == 1
    assert keys[4] == mp.CMD_KEY_KEYS and keys[5] == 0


def test_parse_km_device_separates_v4_from_v3():
    parsed = mp.parse_km_device(">>> \r\nR:3;M:8uf;K:8uf;C:0uf\r\n>>> ")
    assert parsed["R"] == "3"
    assert parsed["M"] == "8uf"
    assert mp.parse_km_device("mouse\r\n>>> ") is None
    assert mp.parse_km_device("keyboard") is None
    assert mp.usb_period_to_hz("8uf") == 1000.0
    assert mp.usb_period_to_hz("0uf") is None
    assert mp.describe_device_kinds(0x03) == "mouse+keyboard"


def test_settings_reply_ignores_topology_notice():
    assert mp.parse_settings_reply(bytes([mp.TOPOLOGY_NOTICE, 0, 0])) is None
    payload = bytes([mp.SETTINGS_RECORD, mp.SETTINGS_OP_INFO, 0]) + struct.pack(
        "<BBBBBIH", 1, mp.SETTINGS_SECTION_MOUSE, 1, 0, 0, 9, 400)
    op, status, body = mp.parse_settings_reply(payload)
    assert (op, status) == (mp.SETTINGS_OP_INFO, 0)
    info = mp.parse_settings_info(body)
    assert info["revision"] == 9
    assert info["image_bytes"] == 400
    assert info["sections"] & mp.SETTINGS_SECTION_MOUSE


def test_vk_to_hid_covers_letters_and_either_shift():
    assert mp.VK_TO_HID[0x41] == (0x04,)  # A
    assert mp.VK_TO_HID[0x10] == (0xE1, 0xE5)  # generic Shift, either side
    assert mp.MOUSE_VK_TO_BIT[0x04] == 0x04  # middle


def test_udp_adapter_reassembles_datagrams():
    server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server.bind(("127.0.0.1", 0))
    port = server.getsockname()[1]
    adapter = mp.UdpSerialAdapter("127.0.0.1", port)
    try:
        adapter.write(b"ping")
        data, addr = server.recvfrom(64)
        assert data == b"ping"
        server.sendto(b"ab", addr)
        server.sendto(b"cd", addr)
        deadline = threading.Event()
        got = b""
        for _ in range(50):
            if adapter.in_waiting >= 4:
                got = adapter.read(4)
                break
            deadline.wait(0.01)
        assert got == b"abcd"
    finally:
        adapter.close()
        server.close()
