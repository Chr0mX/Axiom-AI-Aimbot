#!/usr/bin/env python3
"""
debug_makcu.py — MAKCU debug utility (v2)

Tests mouse button state queries and the full Misc API over a live
serial connection.  Reference: docs/MAKCU_Native_API.md

Usage:
    python debug_makcu.py [COM_PORT] [--baud 115200|4000000] [--ui]
    python debug_makcu.py COM3
    python debug_makcu.py COM3 --baud 4000000
    python debug_makcu.py COM3 --ui            # live button visualizer
    python debug_makcu.py                      # auto-detect
    python debug_makcu.py COM3 --no-poll       # skip live poll
    python debug_makcu.py COM3 --poll 30       # poll for 30s

--ui launches a full-screen ANSI terminal visualizer showing which mouse
buttons are physically held. All other tests are skipped in UI mode.
"""

import argparse
import os
import re
import sys
import threading
import time

# ── dependency path (mirrors makcu_mouse.py) ────────────────────────────────
_here = os.path.dirname(os.path.abspath(__file__))
_deps = os.path.join(_here, "src", "python", "dependencies")
if _deps not in sys.path:
    sys.path.insert(0, _deps)

import serial
import serial.tools.list_ports

# ── import MakcuMouse for connection logic ───────────────────────────────────
_src = os.path.join(_here, "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

from win_utils.makcu_mouse import MakcuMouse

# ── helpers ──────────────────────────────────────────────────────────────────

SEP  = "─" * 60
SEP2 = "· " * 30


def _auto_detect_port() -> str | None:
    candidates = list(serial.tools.list_ports.comports())
    print(f"\n[Auto-detect] Found {len(candidates)} serial port(s):")
    for p in candidates:
        print(f"  {p.device:10s}  vid=0x{p.vid or 0:04X}  pid=0x{p.pid or 0:04X}  desc={p.description}")
    for p in candidates:
        desc = (p.description or "").lower()
        hwid = (p.hwid or "").lower()
        if any(kw in desc or kw in hwid for kw in ("makcu", "km box", "ch340", "cp210")):
            print(f"[Auto-detect] Selected: {p.device}")
            return p.device
    if candidates:
        print(f"[Auto-detect] Falling back to first port: {candidates[0].device}")
        return candidates[0].device
    return None


def _send_recv(ser: serial.Serial, cmd: str | bytes, wait: float = 0.12,
               read_until: bytes | None = None) -> bytes:
    """Send a command and return the raw response bytes."""
    if isinstance(cmd, str):
        cmd = cmd.encode("ascii")
    ser.reset_input_buffer()
    ser.write(cmd)
    ser.flush()
    if read_until:
        try:
            return ser.read_until(read_until, size=1024)
        except Exception as e:
            return f"<read_until error: {e}>".encode()
    time.sleep(wait)
    n = ser.in_waiting
    return ser.read(n) if n else b""


def _decode(raw: bytes) -> str:
    return raw.decode("ascii", errors="replace").strip()


def _print_raw(label: str, raw: bytes, interp: str = ""):
    print(f"  {label}")
    print(f"    raw    : {raw!r}")
    if interp:
        print(f"    parsed : {interp}")
    print()


# ── Misc / System API ─────────────────────────────────────────────────────────

def test_misc_api(ser: serial.Serial):
    print(SEP)
    print("MISC / SYSTEM API")
    print(SEP)

    # km.version() — firmware version string (0xBF)
    print("\n[km.version()] — firmware version")
    raw = _send_recv(ser, "km.version()\r\n", wait=0.15, read_until=b">>>")
    _print_raw("km.version()", raw, _decode(raw))

    # km.device() — returns "keyboard", "mouse", or "none" (0xB3)
    print("[km.device()] — connected HID device type")
    raw = _send_recv(ser, "km.device()\r\n", wait=0.15, read_until=b">>>")
    decoded = _decode(raw)
    m = re.search(r'km\.device\((\w+)\)', decoded)
    interp = f"device = {m.group(1)!r}" if m else "(could not parse)"
    _print_raw("km.device()", raw, interp)

    # km.info() — key=value system info (0xB8)
    print("[km.info()] — system / device info")
    raw = _send_recv(ser, "km.info()\r\n", wait=0.25, read_until=b">>>")
    decoded = _decode(raw)
    print(f"  raw    : {raw!r}")
    kv_pattern = re.compile(r'([A-Z0-9_]+)=([^\r\n,]+)')
    kvs = kv_pattern.findall(decoded)
    FIELD_LABELS = {
        "MAC1": "Primary MAC", "MAC2": "Secondary MAC",
        "TEMP": "Temperature (°C)", "RAM": "Free RAM (bytes)",
        "FW": "Firmware version", "CPU": "CPU identifier",
        "UP": "Uptime (ms)", "VID": "USB Vendor ID", "PID": "USB Product ID",
        "VENDOR": "USB Vendor string", "MODEL": "Device model",
        "ORIGINAL_SERIAL": "Original serial", "SPOOFED_SERIAL": "Spoofed serial",
        "MOUSE_BINT": "Mouse bInterval (poll rate)", "KBD_BINT": "Kbd bInterval (poll rate)",
        "FAULT": "Fault flags",
    }
    if kvs:
        print("  parsed :")
        for k, v in kvs:
            print(f"    {k:22s} = {v:<20s}  {FIELD_LABELS.get(k, '')}")
    else:
        print("  (no key=value pairs found in response)")
    print()

    # km.fault() — debug info for HID parse failures (0xB5)
    print("[km.fault()] — HID parse fault debug info")
    raw = _send_recv(ser, "km.fault()\r\n", wait=0.15, read_until=b">>>")
    _print_raw("km.fault()", raw, _decode(raw))

    # km.release() — query auto-release timer (0xBC)
    print("[km.release()] — auto-release timer (GET)")
    raw = _send_recv(ser, "km.release()\r\n", wait=0.12, read_until=b">>>")
    decoded = _decode(raw)
    m = re.search(r'release\((\d+)\)', decoded)
    interp = f"timer = {m.group(1)} ms" if m else _decode(raw) or "(no data)"
    _print_raw("km.release()", raw, interp)

    # km.bypass() — bypass mode (0xB2): 0=off, 1=mouse bypass, 2=kbd bypass
    print("[km.bypass()] — USB bypass mode (GET)")
    raw = _send_recv(ser, "km.bypass()\r\n", wait=0.12, read_until=b">>>")
    decoded = _decode(raw)
    m = re.search(r'bypass\((\d+)\)', decoded)
    BYPASS_LABELS = {"0": "off", "1": "mouse bypass", "2": "kbd bypass"}
    interp = f"mode = {m.group(1)} ({BYPASS_LABELS.get(m.group(1), '?')})" if m else _decode(raw) or "(no data)"
    _print_raw("km.bypass()", raw, interp)

    # km.screen() — virtual screen dimensions for moveto() (0xBD)
    print("[km.screen()] — virtual screen dimensions (GET)")
    raw = _send_recv(ser, "km.screen()\r\n", wait=0.12, read_until=b">>>")
    decoded = _decode(raw)
    m = re.search(r'screen\((\d+),(\d+)\)', decoded)
    interp = f"{m.group(1)} × {m.group(2)}" if m else _decode(raw) or "(no data)"
    _print_raw("km.screen()", raw, interp)

    # km.hs() — USB high-speed mode (GET) (0xB7)
    print("[km.hs()] — USB high-speed compatibility (GET)")
    raw = _send_recv(ser, "km.hs()\r\n", wait=0.12, read_until=b">>>")
    decoded = _decode(raw)
    m = re.search(r'hs\((\d+)\)', decoded)
    interp = f"hs = {m.group(1)} ({'enabled' if m and m.group(1) == '1' else 'disabled'})" if m else _decode(raw) or "(no data)"
    _print_raw("km.hs()", raw, interp)

    # km.log() — log level (GET) (0xBA): 0=none, 5=debug
    print("[km.log()] — log level (GET)")
    raw = _send_recv(ser, "km.log()\r\n", wait=0.12, read_until=b">>>")
    decoded = _decode(raw)
    m = re.search(r'log\((\d+)\)', decoded)
    interp = f"level = {m.group(1)}" if m else _decode(raw) or "(no data)"
    _print_raw("km.log()", raw, interp)

    # km.serial() — USB serial number (GET) (0xBE)
    print("[km.serial()] — USB serial number (GET)")
    raw = _send_recv(ser, "km.serial()\r\n", wait=0.12, read_until=b">>>")
    _print_raw("km.serial()", raw, _decode(raw))

    # km.led() — LED state (GET) (0xB9)
    print("[km.led()] — LED state (GET)")
    raw = _send_recv(ser, "km.led()\r\n", wait=0.12, read_until=b">>>")
    _print_raw("km.led()", raw, _decode(raw))

    # km.baud() — current baud rate (GET) (0xB1)
    print("[km.baud()] — current baud rate (GET)")
    raw = _send_recv(ser, "km.baud()\r\n", wait=0.12, read_until=b">>>")
    decoded = _decode(raw)
    m = re.search(r'baud\((\d+)\)', decoded)
    interp = f"baud = {m.group(1)}" if m else _decode(raw) or "(no data)"
    _print_raw("km.baud()", raw, interp)

    # km.help() — list all commands
    print("[km.help()] — all available commands")
    raw = _send_recv(ser, "km.help()\r\n", wait=0.3, read_until=b">>>")
    decoded = _decode(raw)
    print(f"  raw    : {raw!r}")
    print(f"  decoded:\n{decoded}")
    print()


# ── Mouse button & pointer state ──────────────────────────────────────────────

def _btn_query(ser: serial.Serial, cmd: bytes, pattern: bytes, label: str) -> tuple[bytes, str]:
    raw = _send_recv(ser, cmd, wait=0.0, read_until=b">>>")
    m = re.search(pattern, raw)
    val = int(m.group(1)) if m else -1
    STATES = {0: "up / not held", 1: "raw/physical DOWN", 2: "injected (API) DOWN", 3: "physical + injected"}
    return raw, f"{val} → {STATES.get(val, 'unknown')}"


MOUSE_STATE_METHODS = [
    ("km.left()   [LMB]",         b"km.left()\r\n",   rb'km\.left\((\d)\)'),
    ("km.right()  [RMB]",         b"km.right()\r\n",  rb'km\.right\((\d)\)'),
    ("km.middle() [MMB]",         b"km.middle()\r\n", rb'km\.middle\((\d)\)'),
    ("km.side1()  [S1 / thumb]",  b"km.side1()\r\n",  rb'km\.side1\((\d)\)'),
    ("km.side2()  [S2 / dpi]",    b"km.side2()\r\n",  rb'km\.side2\((\d)\)'),
    ("km.catch_ml()[LMB catch]",  b"km.catch_ml()\r\n", rb'catch_ml\((\d)\)'),
]


def test_mouse_state(ser: serial.Serial):
    print(SEP)
    print("MOUSE BUTTON & POINTER STATE — SNAPSHOT")
    print(SEP)
    print("(hold / click buttons before reading to see non-zero values)\n")

    for label, cmd, pattern in MOUSE_STATE_METHODS:
        try:
            raw = _send_recv(ser, cmd, wait=0.0, read_until=b">>>")
            m = re.search(pattern, raw)
            val = int(m.group(1)) if m else -1
            STATES = {0: "up", 1: "PHYS_DOWN", 2: "INJECT_DOWN", 3: "BOTH_DOWN"}
            _print_raw(label, raw, f"{val} → {STATES.get(val, 'unknown')}")
        except Exception as e:
            print(f"  {label}  ERROR: {e}\n")

    # Cursor position
    raw = _send_recv(ser, b"km.getpos()\r\n", wait=0.0, read_until=b">>>")
    m = re.search(rb'getpos\((\d+),(\d+)\)', raw)
    interp = f"cursor at ({m.group(1).decode()}, {m.group(2).decode()})" if m else "(no data)"
    _print_raw("km.getpos() [cursor pos]", raw, interp)

    # Remap / axis flags
    for cmd, label in [
        (b"km.remap_axis()\r\n",  "km.remap_axis() [inv_x,inv_y,swap flags]"),
        (b"km.invert_x()\r\n",    "km.invert_x()"),
        (b"km.invert_y()\r\n",    "km.invert_y()"),
        (b"km.swap_xy()\r\n",     "km.swap_xy()"),
    ]:
        raw = _send_recv(ser, cmd, wait=0.0, read_until=b">>>")
        _print_raw(label, raw, _decode(raw))


# ── Live poll ─────────────────────────────────────────────────────────────────

def poll_buttons(ser: serial.Serial, duration: float = 15.0):
    print(SEP)
    print(f"LIVE POLL — LMB / RMB / MMB / S1 / S2 / catch_ml  ({duration:.0f}s)")
    print("Hold or click mouse buttons to see state changes. Ctrl+C to stop early.")
    print(SEP)

    QUERIES = [
        ("lmb",      b"km.left()\r\n",     rb'km\.left\((\d)\)'),
        ("rmb",      b"km.right()\r\n",    rb'km\.right\((\d)\)'),
        ("mmb",      b"km.middle()\r\n",   rb'km\.middle\((\d)\)'),
        ("s1",       b"km.side1()\r\n",    rb'km\.side1\((\d)\)'),
        ("s2",       b"km.side2()\r\n",    rb'km\.side2\((\d)\)'),
        ("catch_ml", b"km.catch_ml()\r\n", rb'catch_ml\((\d)\)'),
    ]
    labels = {0: "up", 1: "PHYS▼", 2: "INJ▼", 3: "BOTH▼"}
    prev = {k: -1 for k, _, __ in QUERIES}
    lock = threading.Lock()

    def _read_val(cmd: bytes, pat: bytes) -> int:
        with lock:
            try:
                ser.reset_input_buffer()
                ser.write(cmd)
                ser.flush()
                resp = ser.read_until(b">>>", size=256)
                m = re.search(pat, resp)
                return int(m.group(1)) if m else -1
            except Exception:
                return -1

    end = time.monotonic() + duration
    try:
        while time.monotonic() < end:
            cur = {k: _read_val(cmd, pat) for k, cmd, pat in QUERIES}
            if cur != prev:
                ts = time.strftime("%H:%M:%S")
                parts = [f"{k}={v}({labels.get(v,'?')})" for k, v in cur.items()]
                print(f"  [{ts}] " + "  ".join(parts))
                prev = dict(cur)
            time.sleep(0.02)
    except KeyboardInterrupt:
        print("\n[Poll] Stopped by user.")
    print()


# ── Terminal button visualizer UI ─────────────────────────────────────────────

# ANSI helpers
_A  = "\033["          # ESC [
_RS = "\033[0m"        # reset
_BD = "\033[1m"        # bold
_DM = "\033[2m"        # dim
_CL = "\033[2J\033[H"  # clear screen + home
_HC = "\033[?25l"      # hide cursor
_SC = "\033[?25h"      # show cursor

# Colours for button states
_COL = {
    -1: "\033[90m",    # grey  — unknown
     0: "\033[2;37m",  # dim white — released
     1: "\033[1;92m",  # bright green — physical DOWN
     2: "\033[1;93m",  # bright yellow — injected DOWN
     3: "\033[1;91m",  # bright red — both
}
_STATE_LABEL = {-1: "  ?  ", 0: " up  ", 1: "PHYS▼", 2: "INJ▼ ", 3: "BOTH▼"}


def _btn_block(val: int, name: str, width: int = 7) -> str:
    """Render a coloured button cell: '[ LMB ]' style."""
    col   = _COL.get(val, _COL[-1])
    inner = f" {name} "
    pad   = max(0, width - len(inner))
    left  = pad // 2
    right = pad - left
    return f"{col}{_BD}[{' ' * left}{inner}{' ' * right}]{_RS}"


def show_button_ui(ser: serial.Serial):
    """
    Full-screen terminal visualizer.  Draws a mouse diagram and updates
    button states in-place using ANSI cursor positioning.

    Layout:

        ┌──────────┬──────────┐
        │   LMB    │   RMB    │
        ├────┬─────┴─────┬────┤
        │    │    MMB    │    │
        │ S1 │           │ S2 │
        └────┴───────────┴────┘
         catch_ml: 0 (inactive)

    Button colour legend:
        dim white = released
        bright green  = physical DOWN (bit 0)
        bright yellow = injected DOWN (bit 1)
        bright red    = both
    """
    QUERIES = [
        ("lmb",      b"km.left()\r\n",     rb'km\.left\((\d)\)'),
        ("rmb",      b"km.right()\r\n",    rb'km\.right\((\d)\)'),
        ("mmb",      b"km.middle()\r\n",   rb'km\.middle\((\d)\)'),
        ("s1",       b"km.side1()\r\n",    rb'km\.side1\((\d)\)'),
        ("s2",       b"km.side2()\r\n",    rb'km\.side2\((\d)\)'),
        ("catch_ml", b"km.catch_ml()\r\n", rb'catch_ml\((\d)\)'),
    ]
    lock = threading.Lock()

    def _read_all() -> dict[str, int]:
        out = {}
        for key, cmd, pat in QUERIES:
            with lock:
                try:
                    ser.reset_input_buffer()
                    ser.write(cmd)
                    ser.flush()
                    resp = ser.read_until(b">>>", size=256)
                    m = re.search(pat, resp)
                    out[key] = int(m.group(1)) if m else -1
                except Exception:
                    out[key] = -1
        return out

    def _render(st: dict[str, int]) -> str:
        lmb = _btn_block(st["lmb"],      "LMB", 8)
        rmb = _btn_block(st["rmb"],      "RMB", 8)
        mmb = _btn_block(st["mmb"],      "MMB", 11)
        s1  = _btn_block(st["s1"],       "S1",  4)
        s2  = _btn_block(st["s2"],       "S2",  4)
        cm  = st["catch_ml"]
        cm_col   = "\033[1;92m" if cm else "\033[2;37m"
        cm_label = f"{cm_col}{'ACTIVE' if cm else 'off'}{_RS}"

        lines = [
            f"  {_BD}MAKCU Mouse Button Visualizer{_RS}   {_DM}Ctrl+C to stop{_RS}",
            "",
            f"  ┌{'─'*12}┬{'─'*12}┐",
            f"  │  {lmb}  │  {rmb}  │",
            f"  ├────┬{'─'*7}┴{'─'*4}┬────┤",
            f"  │{s1}│   {mmb}   │{s2}│",
            f"  └────┴{'─'*11}┴────┘",
            "",
            f"  catch_ml (LMB intercept): {cm_label}",
            "",
            f"  {_DM}Legend:{_RS}  "
            f"{_COL[0]}■{_RS} released  "
            f"{_COL[1]}■{_RS} physical DOWN  "
            f"{_COL[2]}■{_RS} injected DOWN  "
            f"{_COL[3]}■{_RS} both",
        ]
        return "\n".join(lines)

    sys.stdout.write(_HC)   # hide cursor
    sys.stdout.write(_CL)   # clear screen
    sys.stdout.flush()

    prev: dict[str, int] = {}
    try:
        while True:
            cur = _read_all()
            if cur != prev:
                # Move to top-left and redraw the block
                sys.stdout.write("\033[H")
                sys.stdout.write(_render(cur))
                sys.stdout.flush()
                prev = dict(cur)
            time.sleep(0.02)   # ~50 Hz

    except KeyboardInterrupt:
        pass
    finally:
        sys.stdout.write(_SC)   # restore cursor
        sys.stdout.write("\n")
        sys.stdout.flush()
        print("[UI] Stopped.")


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="MAKCU debug — mouse state, keyboard state & misc API")
    parser.add_argument("port", nargs="?", default=None,
                        help="COM port (e.g. COM3 or /dev/ttyUSB0)")
    parser.add_argument("--baud", type=int, default=115200,
                        choices=[115200, 4000000],
                        help="Baud rate (default 115200). 4000000 requires a capable adapter.")
    parser.add_argument("--poll", type=float, default=15.0, metavar="SECONDS",
                        help="Duration of live button poll in seconds (default 15)")
    parser.add_argument("--ui",       action="store_true",
                        help="Launch live mouse button visualizer (terminal ANSI UI). "
                             "Skips all other tests.")
    parser.add_argument("--no-misc",  action="store_true", help="Skip misc/system API tests")
    parser.add_argument("--no-mouse", action="store_true", help="Skip mouse state snapshot")
    parser.add_argument("--no-poll",  action="store_true", help="Skip live poll")
    args = parser.parse_args()

    port = args.port or _auto_detect_port()
    if not port:
        print("[ERROR] No COM port found. Plug in the MAKCU device or pass the port explicitly.")
        sys.exit(1)

    mouse = MakcuMouse()
    if not mouse.connect(port, args.baud):
        print("[ERROR] Failed to connect to MAKCU device.")
        sys.exit(1)

    ser = mouse._serial  # raw serial handle for debug commands

    try:
        if args.ui:
            show_button_ui(ser)
        else:
            if not args.no_misc:
                test_misc_api(ser)

            if not args.no_mouse:
                test_mouse_state(ser)

            if not args.no_poll:
                poll_buttons(ser, duration=args.poll)

    finally:
        mouse.disconnect()
        print("[Done] Serial port closed.")


if __name__ == "__main__":
    main()
