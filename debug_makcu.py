#!/usr/bin/env python3
"""
debug_makcu.py — MAKCU debug utility

Tests every mouse button state query method and the Misc API
(km.device(), km.info()) over a live serial connection.

Usage:
    python debug_makcu.py [COM_PORT] [--baud 115200|4000000]

Examples:
    python debug_makcu.py COM3
    python debug_makcu.py COM3 --baud 4000000
    python debug_makcu.py          # auto-detect COM port

Hold / click mouse buttons while the poll loop is running to see
each method's live response.
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

# ── helpers ──────────────────────────────────────────────────────────────────

SEP = "─" * 60

def _auto_detect_port() -> str | None:
    """Return the first USB serial port that looks like a MAKCU device."""
    candidates = list(serial.tools.list_ports.comports())
    print(f"\n[Auto-detect] Found {len(candidates)} serial port(s):")
    for p in candidates:
        print(f"  {p.device:10s}  vid=0x{p.vid or 0:04X}  pid=0x{p.pid or 0:04X}  desc={p.description}")
    # Prefer ports whose description / hwid contains known MAKCU identifiers
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
            return ser.read_until(read_until, size=512)
        except Exception as e:
            return f"<read_until error: {e}>".encode()
    time.sleep(wait)
    n = ser.in_waiting
    return ser.read(n) if n else b""


def _decode(raw: bytes) -> str:
    return raw.decode("ascii", errors="replace").strip()


# ── connection ────────────────────────────────────────────────────────────────

def connect(port: str, baud: int) -> serial.Serial:
    print(f"\n[Connect] Opening {port} @ 115200 baud (startup rate)...")
    ser = serial.Serial(port, 115200, timeout=0.3, write_timeout=0.05)
    time.sleep(0.1)
    ser.reset_input_buffer()

    # Handshake
    raw = _send_recv(ser, "km.version()\r\n", wait=0.15)
    print(f"[Handshake] km.version() → {_decode(raw)!r}")
    if not raw:
        print("[Handshake] WARNING: no response — device may not be MAKCU")

    # Disable echo to keep output clean
    _send_recv(ser, "km.echo(0)\r\n", wait=0.08)

    if baud == 4_000_000:
        print("[Connect] Switching to 4 Mbaud...")
        ser.write(b"km.baud(4000000)\r\n")
        ser.flush()
        time.sleep(0.06)
        ser.close()
        time.sleep(0.03)
        ser = serial.Serial(port, 4_000_000, timeout=0.3, write_timeout=0.05)
        time.sleep(0.06)
        ser.reset_input_buffer()
        print("[Connect] Now at 4 Mbaud.")

    print(f"[Connect] Ready on {port} @ {baud} baud.\n")
    return ser


# ── Misc API ─────────────────────────────────────────────────────────────────

def test_misc_api(ser: serial.Serial):
    print(SEP)
    print("MISC API TESTS")
    print(SEP)

    # km.device() — GET — returns connected device type (0xB3)
    print("\n[km.device()] — GET — connected HID device type")
    raw = _send_recv(ser, "km.device()\r\n", wait=0.15)
    decoded = _decode(raw)
    print(f"  Raw bytes : {raw!r}")
    print(f"  Decoded   : {decoded!r}")
    # Try to parse a numeric value out of the response
    m = re.search(r'device\((\d+)\)', decoded)
    if m:
        dev_id = int(m.group(1))
        labels = {0: "None/disconnected", 1: "Mouse", 2: "Keyboard", 3: "Both"}
        print(f"  Device ID : {dev_id} → {labels.get(dev_id, 'Unknown')}")

    # km.info() — GET — system / device info in key=value format (0xB8)
    print("\n[km.info()] — GET — system / device info (key=value)")
    raw = _send_recv(ser, "km.info()\r\n", wait=0.25, read_until=b">>>")
    decoded = _decode(raw)
    print(f"  Raw bytes : {raw!r}")
    print()
    # Parse key=value lines
    kv_pattern = re.compile(r'([A-Z0-9_]+)=([^\r\n,]+)')
    kvs = kv_pattern.findall(decoded)
    if kvs:
        print("  Parsed fields:")
        field_labels = {
            "MAC1":            "Primary MAC address",
            "MAC2":            "Secondary MAC address",
            "TEMP":            "Device temperature (°C)",
            "RAM":             "Free RAM (bytes)",
            "FW":              "Firmware version",
            "CPU":             "CPU identifier",
            "UP":              "Uptime (ms)",
            "VID":             "USB Vendor ID",
            "PID":             "USB Product ID",
            "VENDOR":          "USB Vendor string",
            "MODEL":           "Device model",
            "ORIGINAL_SERIAL": "Original USB serial number",
            "SPOOFED_SERIAL":  "Spoofed USB serial number",
            "MOUSE_BINT":      "Mouse bInterval (polling rate)",
            "KBD_BINT":        "Keyboard bInterval (polling rate)",
            "FAULT":           "Fault / error flags",
        }
        for k, v in kvs:
            label = field_labels.get(k, "")
            print(f"    {k:20s} = {v:<20s}  {label}")
    else:
        print("  (could not parse key=value pairs from response)")


# ── Mouse button state methods ────────────────────────────────────────────────

def _query_left_raw(ser: serial.Serial) -> tuple[bytes, str]:
    """km.left() — GET — returns lock state: 0=none, 1=raw, 2=injected, 3=both"""
    raw = _send_recv(ser, b"km.left()\r\n", wait=0.0, read_until=b">>>")
    m = re.search(rb'km\.left\((\d)\)', raw)
    val = int(m.group(1)) if m else -1
    labels = {0: "not held", 1: "raw/physical", 2: "injected (API)", 3: "both physical+injected"}
    return raw, f"{val} → {labels.get(val, 'unknown')}"


def _query_right_raw(ser: serial.Serial) -> tuple[bytes, str]:
    """km.right() — GET — right mouse button lock state"""
    raw = _send_recv(ser, b"km.right()\r\n", wait=0.0, read_until=b">>>")
    m = re.search(rb'km\.right\((\d)\)', raw)
    val = int(m.group(1)) if m else -1
    labels = {0: "not held", 1: "raw/physical", 2: "injected (API)", 3: "both physical+injected"}
    return raw, f"{val} → {labels.get(val, 'unknown')}"


def _query_middle_raw(ser: serial.Serial) -> tuple[bytes, str]:
    """km.middle() — GET — middle mouse button lock state"""
    raw = _send_recv(ser, b"km.middle()\r\n", wait=0.0, read_until=b">>>")
    m = re.search(rb'km\.middle\((\d)\)', raw)
    val = int(m.group(1)) if m else -1
    labels = {0: "not held", 1: "raw/physical", 2: "injected (API)", 3: "both physical+injected"}
    return raw, f"{val} → {labels.get(val, 'unknown')}"


def _query_catch_ml(ser: serial.Serial) -> tuple[bytes, str]:
    """km.catch_ml() — GET — returns whether LMB catch/intercept mode is active.
    Distinct from lock state — catch_ml controls whether MAKCU intercepts
    the raw LMB event before it reaches the OS.
    """
    raw = _send_recv(ser, b"km.catch_ml()\r\n", wait=0.0, read_until=b">>>")
    m = re.search(rb'catch_ml\((\d)\)', raw)
    val = int(m.group(1)) if m else -1
    return raw, f"catch_ml active={val}"


def _query_getpos(ser: serial.Serial) -> tuple[bytes, str]:
    """km.getpos() — GET — returns current absolute cursor position (x, y)."""
    raw = _send_recv(ser, b"km.getpos()\r\n", wait=0.0, read_until=b">>>")
    m = re.search(rb'getpos\((\d+),(\d+)\)', raw)
    if m:
        x, y = int(m.group(1)), int(m.group(2))
        return raw, f"cursor at ({x}, {y})"
    return raw, "(no position data parsed)"


def _query_mo(ser: serial.Serial) -> tuple[bytes, str]:
    """km.mo() — GET — returns mouse overlay / movement origin mode."""
    raw = _send_recv(ser, b"km.mo()\r\n", wait=0.0, read_until=b">>>")
    m = re.search(rb'km\.mo\((\d+)\)', raw)
    val = int(m.group(1)) if m else -1
    return raw, f"mo={val}"


def _query_silent(ser: serial.Serial) -> tuple[bytes, str]:
    """km.silent() — GET — returns silent mode state (suppresses hardware reports)."""
    raw = _send_recv(ser, b"km.silent()\r\n", wait=0.0, read_until=b">>>")
    m = re.search(rb'km\.silent\((\d+)\)', raw)
    val = int(m.group(1)) if m else -1
    return raw, f"silent={val}"


# All mouse-state query methods, in order of usefulness for detecting physical LMB
MOUSE_STATE_METHODS = [
    ("km.left()   [LMB lock state]",   _query_left_raw),
    ("km.right()  [RMB lock state]",   _query_right_raw),
    ("km.middle() [MMB lock state]",   _query_middle_raw),
    ("km.catch_ml()[LMB intercept]",   _query_catch_ml),
    ("km.getpos() [cursor position]",  _query_getpos),
    ("km.mo()     [movement origin]",  _query_mo),
    ("km.silent() [silent mode]",      _query_silent),
]


def test_mouse_state_methods(ser: serial.Serial):
    print(SEP)
    print("MOUSE STATE QUERY — SINGLE SNAPSHOT (all methods)")
    print(SEP)
    print("(snapshot taken now — hold/click buttons to see non-zero values)\n")

    for name, fn in MOUSE_STATE_METHODS:
        try:
            raw, interpretation = fn(ser)
            print(f"  {name}")
            print(f"    raw        : {raw!r}")
            print(f"    parsed     : {interpretation}")
        except Exception as e:
            print(f"  {name}")
            print(f"    ERROR      : {e}")
        print()


def poll_lmb_methods(ser: serial.Serial, duration: float = 15.0):
    """Poll km.left(), km.right(), and km.catch_ml() continuously and print changes."""
    print(SEP)
    print(f"LIVE POLL — km.left / km.right / km.catch_ml  ({duration:.0f}s)")
    print("Hold or click mouse buttons to see state changes.")
    print("Press Ctrl+C to stop early.")
    print(SEP)

    prev = {"left": -1, "right": -1, "middle": -1, "catch_ml": -1}
    end = time.monotonic() + duration
    lock = threading.Lock()

    def _read_val(cmd: bytes, pattern: bytes) -> int:
        with lock:
            try:
                ser.reset_input_buffer()
                ser.write(cmd)
                ser.flush()
                resp = ser.read_until(b">>>", size=256)
                m = re.search(pattern, resp)
                return int(m.group(1)) if m else -1
            except Exception:
                return -1

    try:
        while time.monotonic() < end:
            l  = _read_val(b"km.left()\r\n",     rb'km\.left\((\d)\)')
            r  = _read_val(b"km.right()\r\n",    rb'km\.right\((\d)\)')
            md = _read_val(b"km.middle()\r\n",   rb'km\.middle\((\d)\)')
            cl = _read_val(b"km.catch_ml()\r\n", rb'catch_ml\((\d)\)')

            if (l, r, md, cl) != (prev["left"], prev["right"], prev["middle"], prev["catch_ml"]):
                ts = time.strftime("%H:%M:%S")
                lmb_labels = {0: "up", 1: "PHYS_DOWN", 2: "INJECT_DOWN", 3: "BOTH_DOWN"}
                print(
                    f"  [{ts}] "
                    f"LMB={l}({lmb_labels.get(l,'?')})  "
                    f"RMB={r}({lmb_labels.get(r,'?')})  "
                    f"MMB={md}({lmb_labels.get(md,'?')})  "
                    f"catch_ml={cl}"
                )
                prev.update({"left": l, "right": r, "middle": md, "catch_ml": cl})

            time.sleep(0.02)  # ~50 Hz polling

    except KeyboardInterrupt:
        print("\n[Poll] Stopped by user.")

    print()


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MAKCU debug — mouse state & misc API")
    parser.add_argument("port", nargs="?", default=None, help="COM port (e.g. COM3 or /dev/ttyUSB0)")
    parser.add_argument("--baud", type=int, default=115200, choices=[115200, 4000000],
                        help="Baud rate (default 115200)")
    parser.add_argument("--poll", type=float, default=15.0, metavar="SECONDS",
                        help="Duration of live poll phase in seconds (default 15)")
    parser.add_argument("--no-misc", action="store_true", help="Skip km.device() / km.info() tests")
    parser.add_argument("--no-poll", action="store_true", help="Skip live poll phase")
    args = parser.parse_args()

    port = args.port or _auto_detect_port()
    if not port:
        print("[ERROR] No COM port found. Plug in the MAKCU device or pass the port explicitly.")
        sys.exit(1)

    try:
        ser = connect(port, args.baud)
    except serial.SerialException as e:
        print(f"[ERROR] Cannot open port: {e}")
        sys.exit(1)

    try:
        # 1. Misc API (km.device, km.info)
        if not args.no_misc:
            test_misc_api(ser)
            print()

        # 2. Snapshot of all mouse state methods
        test_mouse_state_methods(ser)

        # 3. Live poll
        if not args.no_poll:
            poll_lmb_methods(ser, duration=args.poll)

    finally:
        ser.close()
        print("[Done] Serial port closed.")


if __name__ == "__main__":
    main()
