"""
MAKCU V2 Binary Protocol — frame decoder.

No serial I/O; accepts raw bytes and returns structured Python objects.

Frame format (both directions):
    [0x50] [CMD:u8] [LEN_LO:u8] [LEN_HI:u8] [PAYLOAD: LEN bytes]

Setter response:  LEN=1, payload = 0x00 (OK) or 0x01 (ERR)
Getter response:  LEN>1, payload = data bytes
Stream frames:    same header, payload = stream-specific struct
"""

import struct
from dataclasses import dataclass
from typing import List, Optional, Union

_FRAME_HEADER = 0x50


@dataclass
class StatusResponse:
    cmd: int
    ok: bool


@dataclass
class DataResponse:
    cmd: int
    payload: bytes


@dataclass
class MouseStreamFrame:
    cmd: int
    buttons: int   # u8 bitmask: bit0=L,1=R,2=M,3=S1,4=S2
    dx: int        # i16
    dy: int        # i16
    wheel: int     # i8
    pan: int       # i8
    tilt: int      # i8


@dataclass
class ButtonsStreamFrame:
    cmd: int
    mask: int      # u16: bit0=L,1=R,2=M,3=S1,4=S2


@dataclass
class DeviceType:
    name: str      # "mouse", "keyboard", "none"
    code: int      # 2=mouse, 1=keyboard, 0=none


AnyFrame = Union[StatusResponse, DataResponse, MouseStreamFrame, ButtonsStreamFrame]

# Stream CMD codes that need specialised parsing
_CMD_MOUSE_STREAM   = 0x0C
_CMD_BUTTONS_STREAM = 0x02


class BinaryDecoder:
    """Stateful incremental decoder for MAKCU binary frames."""

    def __init__(self) -> None:
        self._buf: bytearray = bytearray()

    def feed(self, data: bytes) -> List[AnyFrame]:
        """Append raw bytes and return all complete frames decoded so far."""
        self._buf.extend(data)
        results: List[AnyFrame] = []
        while True:
            frame = self._try_parse()
            if frame is None:
                break
            results.append(frame)
        return results

    def reset(self) -> None:
        self._buf.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _try_parse(self) -> Optional[AnyFrame]:
        # Resync: discard bytes before the first 0x50
        idx = self._buf.find(_FRAME_HEADER)
        if idx < 0:
            self._buf.clear()
            return None
        if idx > 0:
            del self._buf[:idx]

        # Need header(1) + cmd(1) + len_lo(1) + len_hi(1) = 4 bytes minimum
        if len(self._buf) < 4:
            return None

        cmd = self._buf[1]
        length = self._buf[2] | (self._buf[3] << 8)
        total = 4 + length

        if len(self._buf) < total:
            return None

        payload = bytes(self._buf[4:total])
        del self._buf[:total]
        return self._decode(cmd, payload)

    def _decode(self, cmd: int, payload: bytes) -> AnyFrame:
        length = len(payload)

        # Status response: single byte 0x00 or 0x01
        if length <= 1:
            ok = (payload[0] == 0x00) if length == 1 else True
            return StatusResponse(cmd=cmd, ok=ok)

        # Known stream types
        if cmd == _CMD_MOUSE_STREAM and length >= 8:
            # [btns:u8][dx:i16][dy:i16][wheel:i8][pan:i8][tilt:i8]
            btns, dx, dy, wheel, pan, tilt = struct.unpack_from('<Bhhbbb', payload)
            return MouseStreamFrame(cmd=cmd, buttons=btns, dx=dx, dy=dy,
                                    wheel=wheel, pan=pan, tilt=tilt)

        if cmd == _CMD_BUTTONS_STREAM and length >= 2:
            (mask,) = struct.unpack_from('<H', payload)
            return ButtonsStreamFrame(cmd=cmd, mask=mask)

        return DataResponse(cmd=cmd, payload=payload)


# ---------------------------------------------------------------------------
# Convenience helpers for pretty-printing decoded frames
# ---------------------------------------------------------------------------

def format_frame(frame: AnyFrame) -> str:
    if isinstance(frame, StatusResponse):
        status = "OK" if frame.ok else "ERR"
        return f"[CMD 0x{frame.cmd:02X}] {status}"

    if isinstance(frame, MouseStreamFrame):
        btn_names = []
        if frame.buttons & 0x01: btn_names.append("L")
        if frame.buttons & 0x02: btn_names.append("R")
        if frame.buttons & 0x04: btn_names.append("M")
        if frame.buttons & 0x08: btn_names.append("S1")
        if frame.buttons & 0x10: btn_names.append("S2")
        btns = "+".join(btn_names) if btn_names else "none"
        return (f"[MOUSE STREAM] btns={btns} dx={frame.dx} dy={frame.dy} "
                f"wheel={frame.wheel} pan={frame.pan} tilt={frame.tilt}")

    if isinstance(frame, ButtonsStreamFrame):
        return f"[BTN STREAM] mask=0x{frame.mask:04X}"

    if isinstance(frame, DataResponse):
        hex_str = frame.payload.hex(' ')
        try:
            text = frame.payload.decode('utf-8', errors='replace').rstrip('\r\n\x00 ')
        except Exception:
            text = ''
        if text:
            return f"[CMD 0x{frame.cmd:02X}] data: {text!r}  ({hex_str})"
        return f"[CMD 0x{frame.cmd:02X}] data: {hex_str}"

    return repr(frame)
