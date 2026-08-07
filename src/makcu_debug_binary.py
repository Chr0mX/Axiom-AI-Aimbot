"""
MAKCU V2 Binary Protocol — interactive debug REPL.

Run directly:
    python src/makcu_debug_binary.py [COM_PORT] [BAUD]

If no COM port is given the highest-numbered available port is auto-selected.
"""

import os
import re
import struct
import sys
import time

_src_dir = os.path.dirname(os.path.abspath(__file__))
_deps_dir = os.path.join(_src_dir, "python", "dependencies")
sys.path.insert(0, _deps_dir)
sys.path.insert(0, _src_dir)

from win_utils.makcu_mouse_binary import (
    MakcuMouseBinary, _build_frame, _sorted_ports,
    _CMD_VERSION, _CMD_INFO, _CMD_MOUSE_STREAM, _CMD_BUTTONS_STREAM,
    _CMD_BAUD, _DEFAULT_BAUD,
)
from makcu_binary_decoder import BinaryDecoder, format_frame

import serial.tools.list_ports


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _list_ports():
    ports = _sorted_ports(serial.tools.list_ports.comports())
    if not ports:
        print("[WARN] No COM ports found.")
    else:
        print("Available COM ports (highest first):")
        for p in ports:
            print(f"  {p.device:10s} — {p.description}")
    return ports


def _print_response(raw: bytes, cmd: int | None = None) -> None:
    """Decode raw bytes and pretty-print each frame."""
    if not raw:
        print("  (no response)")
        return
    dec = BinaryDecoder()
    frames = dec.feed(raw)
    if frames:
        for f in frames:
            print(" ", format_frame(f))
    else:
        print(f"  raw ({len(raw)} bytes): {raw.hex(' ')}")


def _parse_hex(token: str) -> int:
    return int(token, 16) if token.startswith(("0x", "0X")) else int(token, 16)


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

def _cmd_move(dev: MakcuMouseBinary, args: list) -> None:
    if len(args) < 2:
        print("[USAGE] move <dx> <dy>")
        return
    dx, dy = int(args[0]), int(args[1])
    ok = dev.move(dx, dy)
    print(f"  move({dx}, {dy}) → {'OK' if ok else 'ERR'}")


def _cmd_click(dev: MakcuMouseBinary, args: list) -> None:
    if len(args) < 2:
        print("[USAGE] click <left|right|middle|side1|side2> <down|up|1|0>")
        return
    btn = args[0].lower()
    state_str = args[1].lower()
    state = 1 if state_str in ("down", "1", "press") else 0
    ok = dev.click(btn, state)
    print(f"  click({btn}, {'down' if state else 'up'}) → {'OK' if ok else 'ERR'}")


def _cmd_scroll(dev: MakcuMouseBinary, args: list) -> None:
    if not args:
        print("[USAGE] scroll <delta>")
        return
    ok = dev.scroll(int(args[0]))
    print(f"  scroll({args[0]}) → {'OK' if ok else 'ERR'}")


def _cmd_raw(dev: MakcuMouseBinary, args: list) -> None:
    if not args:
        print("[USAGE] raw <CMD_HEX> [PAYLOAD_HEX...]")
        return
    cmd = _parse_hex(args[0])
    payload = bytes.fromhex(''.join(args[1:])) if len(args) > 1 else b''
    raw = dev.send_raw(cmd, payload)
    _print_response(raw, cmd)


def _cmd_version(dev: MakcuMouseBinary) -> None:
    raw = dev.send_raw(_CMD_VERSION)
    _print_response(raw, _CMD_VERSION)


def _cmd_info(dev: MakcuMouseBinary) -> None:
    raw = dev.send_raw(_CMD_INFO)
    _print_response(raw, _CMD_INFO)


def _cmd_baud(dev: MakcuMouseBinary, args: list) -> None:
    if not args:
        print("[USAGE] baud <rate>  e.g. baud 4000000")
        return
    rate = int(args[0])
    raw = dev.send_raw(_CMD_BAUD, struct.pack('<I', rate))
    _print_response(raw, _CMD_BAUD)
    print(f"  [NOTE] Reconnect at {rate} baud to continue at new speed.")


def _cmd_stream(dev: MakcuMouseBinary, args: list) -> None:
    """stream mouse|buttons start|stop [period_ms]"""
    if len(args) < 2:
        print("[USAGE] stream <mouse|buttons> <start|stop> [period_ms]")
        return

    stream_type = args[0].lower()
    action      = args[1].lower()
    period      = int(args[2]) if len(args) > 2 else 10

    if stream_type == "mouse":
        cmd = _CMD_MOUSE_STREAM
    elif stream_type in ("buttons", "btn"):
        cmd = _CMD_BUTTONS_STREAM
    else:
        print(f"[ERROR] Unknown stream type '{stream_type}'. Use 'mouse' or 'buttons'.")
        return

    mode = 1 if action in ("start", "on", "1") else 0
    payload = struct.pack('<BH', mode, period if mode else 0)
    raw = dev.send_raw(cmd, payload)
    _print_response(raw, cmd)

    if mode == 0:
        return

    print(f"  Streaming {stream_type}… (Ctrl-C to stop)")
    dec = BinaryDecoder()
    try:
        while True:
            raw = dev.send_raw.__func__(dev)  # type: ignore[attr-defined]
            # stream data arrives continuously; read in a loop
            # Use _serial directly for low-latency polling
            ser = dev._serial  # type: ignore[attr-defined]
            if ser is None:
                break
            waiting = ser.in_waiting
            if waiting:
                chunk = ser.read(waiting)
                frames = dec.feed(chunk)
                for f in frames:
                    print(" ", format_frame(f))
            else:
                time.sleep(0.005)
    except KeyboardInterrupt:
        print()
    finally:
        # Stop stream
        stop_payload = struct.pack('<BH', 0, 0)
        try:
            dev.send_raw(cmd, stop_payload)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Stream read loop (separate from send_raw)
# ---------------------------------------------------------------------------

def _stream_loop(dev: MakcuMouseBinary, cmd: int, period: int) -> None:
    payload = struct.pack('<BH', 1, period)
    dev.send_raw(cmd, payload)

    dec = BinaryDecoder()
    ser = dev._serial  # type: ignore[attr-defined]
    print("  Streaming… (Ctrl-C to stop)")
    try:
        while True:
            if ser is None or not dev.is_connected():
                print("[ERROR] Disconnected during stream.")
                break
            waiting = ser.in_waiting
            if waiting:
                chunk = ser.read(waiting)
                for f in dec.feed(chunk):
                    print(" ", format_frame(f))
            else:
                time.sleep(0.005)
    except KeyboardInterrupt:
        print()
    finally:
        try:
            dev.send_raw(cmd, struct.pack('<BH', 0, 0))
        except Exception:
            pass


# ---------------------------------------------------------------------------
# REPL
# ---------------------------------------------------------------------------

_HELP = """
Commands:
  move <dx> <dy>                       Relative move
  click <left|right|middle|side1|side2> <down|up>
  scroll <delta>                       Wheel scroll (+up / -down)
  raw <CMD_HEX> [PAYLOAD_HEX]          Send arbitrary binary frame
  version                              Read firmware version
  info                                 Read system info
  baud <rate>                          Switch baud rate
  stream <mouse|buttons> <start|stop> [period_ms]
  help                                 Show this message
  quit / exit / q                      Disconnect and exit
""".strip()


def main() -> None:
    print("=== MAKCU Binary Protocol REPL ===")

    argv = sys.argv[1:]
    com_port = argv[0] if argv else ''
    baud     = int(argv[1]) if len(argv) > 1 else _DEFAULT_BAUD

    ports = _list_ports()
    if not ports and not com_port:
        sys.exit(1)

    dev = MakcuMouseBinary()
    print(f"\nConnecting to {com_port or ports[0].device} at {baud} baud…")

    if not dev.connect(com_port, baud):
        print("[ERROR] Connection failed. Check port and baud rate.")
        sys.exit(1)

    actual_port = com_port or ports[0].device
    print(f"Connected to {actual_port}.\n")
    print(_HELP)
    print()

    while True:
        try:
            line = input("MAKCU-BIN> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nDisconnecting…")
            dev.disconnect()
            sys.exit(0)

        if not line:
            continue

        parts = line.split()
        verb  = parts[0].lower()
        args  = parts[1:]

        if verb in ("quit", "exit", "q"):
            dev.disconnect()
            print("Disconnected.")
            sys.exit(0)

        try:
            if verb == "move":
                _cmd_move(dev, args)
            elif verb == "click":
                _cmd_click(dev, args)
            elif verb == "scroll":
                _cmd_scroll(dev, args)
            elif verb == "raw":
                _cmd_raw(dev, args)
            elif verb == "version":
                _cmd_version(dev)
            elif verb == "info":
                _cmd_info(dev)
            elif verb == "baud":
                _cmd_baud(dev, args)
            elif verb == "stream":
                if len(args) >= 2:
                    stream_type = args[0].lower()
                    action = args[1].lower()
                    period = int(args[2]) if len(args) > 2 else 10
                    cmd = (_CMD_MOUSE_STREAM if stream_type == "mouse"
                           else _CMD_BUTTONS_STREAM)
                    if action in ("start", "on", "1"):
                        _stream_loop(dev, cmd, period)
                    else:
                        dev.send_raw(cmd, struct.pack('<BH', 0, 0))
                        print("  Stream stopped.")
                else:
                    print("[USAGE] stream <mouse|buttons> <start|stop> [period_ms]")
            elif verb == "help":
                print(_HELP)
            else:
                print(f"[ERROR] Unknown command '{verb}'. Type 'help' for a list.")

        except Exception as exc:
            msg = input(f"[ERROR] {exc}\nPress Enter to continue or type 'quit' to exit: ").strip()
            if msg.lower() in ("quit", "exit", "q"):
                dev.disconnect()
                sys.exit(1)


if __name__ == '__main__':
    main()
