"""
Read-only OBS WebSocket inspector.

Connects to OBS's built-in WebSocket server (Tools > WebSocket Server
Settings in OBS, enabled by default in modern OBS, port 4455) and lists
every filter on a given source, including each filter's full settings
dict — this tells you definitively whether `udp_stream_filter` (or any
other filter) exposes its own resolution/scale/quality setting, without
hunting through OBS's UI by hand.

This script is READ-ONLY: it never modifies your OBS configuration, scenes,
sources, or filters. It only queries and prints what's already there.

Usage:
    python src/core/obs_inspect_filters.py --source "Your Source Name"
    python src/core/obs_inspect_filters.py --source "Your Source Name" --password yourpass
    python src/core/obs_inspect_filters.py --list-sources   # don't know the exact name? list them all

If you don't know your OBS WebSocket password: OBS > Tools > WebSocket
Server Settings > Show Connect Info.

DEVELOPER TOOL — run directly, not imported.
Dumps an OBS instance's filter graph over obs-websocket to confirm the
udp_stream_filter plugin is loaded and configured as expected.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import socket
import struct
from typing import Any


class _ObsWebSocketClient:
    """Minimal RFC6455 WebSocket client + obs-websocket v5 protocol, stdlib only."""

    def __init__(self, host: str, port: int) -> None:
        self.sock = socket.create_connection((host, port), timeout=5.0)
        self.sock.settimeout(5.0)
        self._handshake(host, port)

    def _handshake(self, host: str, port: int) -> None:
        key = base64.b64encode(os.urandom(16)).decode()
        request = (
            f"GET / HTTP/1.1\r\n"
            f"Host: {host}:{port}\r\n"
            f"Upgrade: websocket\r\n"
            f"Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            f"Sec-WebSocket-Version: 13\r\n\r\n"
        )
        self.sock.sendall(request.encode())
        response = b""
        while b"\r\n\r\n" not in response:
            chunk = self.sock.recv(4096)
            if not chunk:
                raise ConnectionError("OBS closed connection during WebSocket handshake")
            response += chunk
        if b"101" not in response.split(b"\r\n", 1)[0]:
            raise ConnectionError(f"WebSocket handshake failed: {response[:200]!r}")

    def send(self, payload: dict) -> None:
        data = json.dumps(payload).encode()
        n = len(data)
        mask = os.urandom(4)
        masked = bytes(b ^ mask[i % 4] for i, b in enumerate(data))
        header = bytearray([0x81])  # FIN + text opcode
        if n < 126:
            header.append(0x80 | n)  # MASK bit set (client MUST mask)
        elif n < 65536:
            header.append(0x80 | 126)
            header += n.to_bytes(2, "big")
        else:
            header.append(0x80 | 127)
            header += n.to_bytes(8, "big")
        self.sock.sendall(bytes(header) + mask + masked)

    def recv(self) -> dict:
        header = self._recv_exact(2)
        b0, b1 = header[0], header[1]
        opcode = b0 & 0x0F
        length = b1 & 0x7F
        if length == 126:
            length = struct.unpack(">H", self._recv_exact(2))[0]
        elif length == 127:
            length = struct.unpack(">Q", self._recv_exact(8))[0]
        payload = self._recv_exact(length)
        if opcode == 0x8:  # close
            raise ConnectionError("OBS closed the WebSocket connection")
        return json.loads(payload.decode())

    def _recv_exact(self, n: int) -> bytes:
        buf = b""
        while len(buf) < n:
            chunk = self.sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("connection closed unexpectedly")
            buf += chunk
        return buf

    def close(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass


def _compute_auth(password: str, challenge: str, salt: str) -> str:
    secret = base64.b64encode(
        hashlib.sha256((password + salt).encode()).digest()
    ).decode()
    return base64.b64encode(
        hashlib.sha256((secret + challenge).encode()).digest()
    ).decode()


def connect(host: str, port: int, password: "str | None") -> _ObsWebSocketClient:
    client = _ObsWebSocketClient(host, port)
    hello = client.recv()
    if hello.get("op") != 0:
        raise ConnectionError(f"expected Hello (op 0), got: {hello}")

    identify: dict[str, Any] = {"rpcVersion": 1}
    auth_info = hello.get("d", {}).get("authentication")
    if auth_info:
        if not password:
            raise ConnectionError(
                "OBS WebSocket requires a password — pass --password "
                "(OBS > Tools > WebSocket Server Settings > Show Connect Info)"
            )
        identify["authentication"] = _compute_auth(
            password, auth_info["challenge"], auth_info["salt"]
        )

    client.send({"op": 1, "d": identify})
    identified = client.recv()
    if identified.get("op") != 2:
        raise ConnectionError(f"Identify failed, OBS replied: {identified}")
    return client


def request(client: _ObsWebSocketClient, request_type: str, request_data: "dict | None" = None) -> dict:
    request_id = os.urandom(4).hex()
    payload = {"op": 6, "d": {"requestType": request_type, "requestId": request_id}}
    if request_data:
        payload["d"]["requestData"] = request_data
    client.send(payload)
    while True:
        msg = client.recv()
        if msg.get("op") == 7 and msg.get("d", {}).get("requestId") == request_id:
            d = msg["d"]
            if not d.get("requestStatus", {}).get("result", False):
                raise RuntimeError(f"{request_type} failed: {d.get('requestStatus')}")
            return d.get("responseData", {})


def list_all_sources(client: _ObsWebSocketClient) -> None:
    scenes = request(client, "GetSceneList")
    seen = set()
    print("Sources across all scenes:")
    for scene in scenes.get("scenes", []):
        scene_name = scene["sceneName"]
        items = request(client, "GetSceneItemList", {"sceneName": scene_name})
        for item in items.get("sceneItems", []):
            name = item.get("sourceName")
            if name and name not in seen:
                seen.add(name)
                print(f"  - {name}  (in scene: {scene_name})")


def inspect_source_filters(client: _ObsWebSocketClient, source_name: str) -> None:
    filters = request(client, "GetSourceFilterList", {"sourceName": source_name})
    filter_list = filters.get("filters", [])
    if not filter_list:
        print(f"No filters found on source '{source_name}'.")
        return

    print(f"Filters on '{source_name}' (in chain order):")
    for f in filter_list:
        name = f.get("filterName", "?")
        kind = f.get("filterKind", "?")
        enabled = f.get("filterEnabled", True)
        settings = f.get("filterSettings", {})
        print(f"\n  [{'ON' if enabled else 'OFF'}] {name}  (kind: {kind})")
        if settings:
            print("    settings:")
            for k, v in settings.items():
                print(f"      {k}: {v!r}")
        else:
            print("    (no settings reported in list — fetching full detail...)")
            detail = request(client, "GetSourceFilter", {"sourceName": source_name, "filterName": name})
            for k, v in detail.get("filterSettings", {}).items():
                print(f"      {k}: {v!r}")

        lname = name.lower()
        lkind = kind.lower()
        if "udp" in lname or "udp" in lkind or "stream" in lname:
            keys = [k.lower() for k in settings.keys()]
            has_res = any(x in " ".join(keys) for x in ("width", "height", "scale", "resolution", "quality"))
            if has_res:
                print("    >>> This filter DOES appear to expose a resolution/scale/quality "
                      "setting above — reduce it (not the source's own scale) to shrink the "
                      "UDP stream without touching your recording.")
            else:
                print("    >>> No resolution/scale/quality key visible in this filter's settings. "
                      "It likely always sends at the source's native resolution — you'd need the "
                      "duplicate-source-with-scale-filter workaround instead.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=4455)
    parser.add_argument("--password", default=None)
    parser.add_argument("--source", default=None, help="Exact source name to inspect filters on")
    parser.add_argument("--list-sources", action="store_true", help="List all source names instead")
    args = parser.parse_args()

    if not args.source and not args.list_sources:
        parser.error("pass --source \"Name\" or --list-sources")

    client = connect(args.host, args.port, args.password)
    try:
        if args.list_sources:
            list_all_sources(client)
        else:
            inspect_source_filters(client, args.source)
    finally:
        client.close()


if __name__ == "__main__":
    main()
