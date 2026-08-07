import sys
import os
import re
import time

_src_dir = os.path.dirname(os.path.abspath(__file__))
_deps_dir = os.path.join(_src_dir, "python", "dependencies")
sys.path.insert(0, _deps_dir)
sys.path.insert(0, _src_dir)


def _sorted_ports(ports):
    def _num(p):
        m = re.search(r'(\d+)$', p.device)
        return int(m.group(1)) if m else 0
    return sorted(ports, key=_num, reverse=True)


def _pick_port():
    try:
        from serial.tools import list_ports
        ports = _sorted_ports(list_ports.comports())
        if not ports:
            return None
        for p in ports:
            print(f"  {p.device} — {p.description}")
        return ports[0].device
    except Exception as exc:
        print(f"[ERROR] Could not enumerate COM ports: {exc}")
        return None


def main():
    print("=== MAKCU Debug REPL ===")
    print("Enumerating COM ports...")
    port = _pick_port()
    if port is None:
        print("[ERROR] No COM ports found. Connect your MAKCU device and try again.")
        sys.exit(1)

    print(f"Auto-selected: {port}")

    try:
        import serial
        ser = serial.Serial(port, 115200, timeout=0.5)
    except Exception as exc:
        print(f"[ERROR] Failed to open {port}: {exc}")
        sys.exit(1)

    print(f"Connected to {port} at 115200 baud.")
    print("Type a command and press Enter. Special: info, version, quit/exit/q")
    print()

    def send(line: str):
        if not line.endswith('\r\n'):
            line = line.rstrip('\n').rstrip('\r') + '\r\n'
        ser.write(line.encode())
        time.sleep(0.2)
        response = b""
        while ser.in_waiting:
            response += ser.read(ser.in_waiting)
            time.sleep(0.05)
        if response:
            print(response.decode(errors='replace'), end='')
            if not response.endswith(b'\n'):
                print()

    while True:
        try:
            raw = input("MAKCU> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nDisconnecting...")
            ser.close()
            sys.exit(0)

        if not raw:
            continue

        if raw.lower() in ('quit', 'exit', 'q'):
            print("Disconnecting...")
            ser.close()
            sys.exit(0)

        if raw.lower() == 'info':
            raw = 'km.info()'
        elif raw.lower() == 'version':
            raw = 'km.version()'

        try:
            send(raw)
        except Exception as exc:
            msg = input(f"[ERROR] {exc}\nPress Enter to continue or type 'quit' to exit: ").strip()
            if msg.lower() in ('quit', 'exit', 'q'):
                ser.close()
                sys.exit(1)


if __name__ == '__main__':
    main()
