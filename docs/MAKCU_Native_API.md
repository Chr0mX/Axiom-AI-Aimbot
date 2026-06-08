# MAKCU Native API — Quick Reference (v3.9)

MAKCU is a USB HID interceptor device controlled over UART serial. Two APIs exist:
- **Legacy API (ASCII)** — human-readable text commands
- **V2 API (Binary)** — compact binary frames for performance

---

## Protocol Basics

### Legacy API (ASCII)

**Send (Host → Device)**
```
.command(args)\r\n
```
- Start with `.`, end with `)`. The `km.` prefix is optional.
- Example: `.move(10,5,)` or `km.move(10,5)`

**Receive (Device → Host)**
```
km.reply\r\n>>>
```
- All replies start with `km.` and end with `\r\n>>> `
- Setters echo the command as ACK (unless `echo(0)` is set)

### V2 API (Binary)

**Send frame:** `[0x50] [CMD] [LEN_LO] [LEN_HI] [PAYLOAD...]`

**Response (setter):** `[0x50] [CMD] [LEN_LO] [LEN_HI] [0x00=OK | 0x01=ERR]`

**Response (getter):** `[0x50] [CMD] [LEN_LO] [LEN_HI] [DATA...]`

All multi-byte values are **little-endian**.

---

## Mouse Commands

| Command | ASCII | Binary CMD | Description |
|---|---|---|---|
| Button state | `left([0\|1\|2])` / `right()` / `middle()` / `side1()` / `side2()` | `0x08`/`0x11`/`0x0A`/`0x12`/`0x13` | GET/SET button. state: 0=release, 1=down, 2=silent_release |
| Click schedule | `click(btn[,count[,delay_ms]])` | `0x04` | Schedule clicks. btn 1-5, delay 0=random 35-75ms |
| Turbo / rapid-fire | `turbo([btn[,delay_ms]])` | `0x17` | Rapid-fire on hold. `turbo()` query; `turbo(0)` disable all |
| Relative move | `move(dx,dy[,segs[,cx1,cy1[,cx2,cy2]]])` | `0x0D` | Move cursor relatively. Optional Bézier via segments + control points |
| Absolute move | `moveto(x,y[,segs[,...]])` | `0x0E` | Move to absolute screen position |
| Scroll wheel | `wheel(delta)` | `0x18` | delta clamped to ±1 step. Positive=up, negative=down |
| Horizontal scroll | `pan([steps])` | `0x0F` | GET pending or SET horizontal scroll steps |
| Tilt/Z scroll | `tilt([steps])` | `0x16` | GET pending or SET tilt steps |
| Get position | `getpos()` | `0x05` | Returns current `(x,y)` pointer position |
| Silent click | `silent(x,y)` | `0x14` | Move to (x,y) then silent left-click |
| Raw frame | `mo(btns,x,y,wheel,pan,tilt)` | `0x0B` | Send a complete raw mouse HID frame |

### Lock / Catch

```
lock_<target>([0|1])    # lock_mx, lock_my, lock_mw, lock_mx+, lock_mx-, lock_ml, etc.
catch_<target>([mode])  # catch_ml, catch_mr, etc. (buttons only). mode: 0=auto, 1=manual
```
Binary lock: `0x09` | Binary catch: `0x03`

Requires the corresponding lock to be set before catch works.

### Physical-Only Remap

These only affect physical input; injected API commands are unaffected.

```
remap_button([src,dst])       # 1=left 2=right 3=middle 4=side1 5=side2; dst=0 clears
remap_axis([inv_x,inv_y,swap])# set all three flags atomically; (0) resets all
invert_x([0|1])
invert_y([0|1])
swap_xy([0|1])
```
Binary: `0x10` / `0x19` / `0x06` / `0x07` / `0x15`

---

## Keyboard Commands

Keys can be specified as **HID code (u8)** or **quoted name** (`'a'`, `"ctrl"`).

| Command | ASCII | Binary CMD | Description |
|---|---|---|---|
| Key down | `down(key)` | `0xA2` | Hold key |
| Key up | `up(key)` | `0xAA` | Release key |
| Tap key | `press(key[,hold_ms[,rand_ms]])` | `0xA7` | Press+release. Default hold: random 35-75ms |
| Type string | `string("text")` | `0xA9` | Type ASCII string (max 256 chars). Auto-handles Shift |
| Clear state | `init()` | `0xA3` | Release all held keys |
| Query key state | `isdown(key)` | `0xA4` | Returns 1=down, 0=up |
| Disable key | `disable([key1,key2,...])` or `disable(key,0\|1)` | `0xA1` | Block key from being sent to host |
| Mask key | `mask(key[,0\|1])` | `0xA6` | Mask key state |
| Remap key | `remap(src,dst)` | `0xA8` | Remap keycode. dst=0 clears |

### Key Name Reference (commonly used)

| Category | Names |
|---|---|
| Letters | `'a'`–`'z'` (lowercase), `'A'`–`'Z'` (uppercase + auto-Shift) |
| Numbers | `'0'`–`'9'` |
| Modifiers | `'ctrl'`, `'shift'`, `'alt'`, `'win'`/`'cmd'`, `'rctrl'`, `'rshift'`, `'ralt'` |
| Control | `'enter'`, `'esc'`, `'backspace'`, `'tab'`, `'space'` |
| Navigation | `'up'`, `'down'`, `'left'`, `'right'`, `'home'`, `'end'`, `'pgup'`, `'pgdown'`, `'ins'`, `'del'` |
| Function | `'f1'`–`'f12'` |
| Numpad | `'kp0'`–`'kp9'`, `'kpenter'`, `'kpdivide'`, `'kpmultiply'`, `'kpplus'`, `'kpminus'` |
| Symbols | `'minus'`, `'equals'`, `'lbracket'`, `'rbracket'`, `'backslash'`, `'semicolon'`, `'quote'`, `'grave'`, `'comma'`, `'period'`, `'slash'`, `'capslock'` |

Single-char keys are **case-sensitive** (`'a'` vs `'A'`). Multi-char names are case-insensitive.

---

## Streaming

All streaming requires baud ≥ 1M. Streams only emit on new frames.

| Command | ASCII | Binary CMD | Output |
|---|---|---|---|
| Mouse stream | `mouse([mode[,period_ms]])` | `0x0C` | 8-byte: `[btns:u8][dx:i16][dy:i16][wheel:i8][pan:i8][tilt:i8]` |
| Axis stream | `axis([mode[,period_ms]])` | `0x01` | 6-byte: `[dx:i16][dy:i16][wheel:i8]` |
| Buttons stream | `buttons([mode[,period_ms]])` | `0x02` | 2-byte mask: bit0=L,1=R,2=M,3=S1,4=S2 |
| Keyboard stream | `keyboard([mode[,period_ms]])` | `0xA5` | 15-byte: `[mods:u8][keys:u8×14]` |

**mode:** 1=raw (physical), 2=constructed (after remap/mask)
**Stop:** send `(0)` or `(0,0)`

---

## Misc / System Commands

| Command | ASCII | Binary CMD | Description |
|---|---|---|---|
| Get version | `version()` | `0xBF` | Firmware version string |
| System info | `info()` | `0xB8` | MAC, temp, RAM, FW, CPU, uptime, VID/PID, etc. |
| Active device | `device()` | `0xB3` | Returns `keyboard`, `mouse`, or `none` |
| Parse fault | `fault()` | `0xB5` | Debug info for devices that fail HID parsing |
| Reboot | `reboot()` | `0xBB` | Reboots device after sending response |
| Baud rate | `baud([rate])` | `0xB1` | GET/SET baud 115200–4000000. 0=reset to 115200. Takes effect immediately |
| Echo | `echo([0\|1])` | `0xB4` | Toggle command echo. Default: on |
| Log level | `log([0-5])` | `0xBA` | 0=none, 5=debug. Auto-disables after 3 power cycles |
| LED control | `led([target[,mode]])` | `0xB9` | target: 1=device, 2=host. mode: 0=off, 1=on. Flash: `led(target,times,delay_ms)` |
| USB serial | `serial([text])` | `0xBE` | GET/SET/reset USB serial number (persistent) |
| Bypass | `bypass([0\|1\|2])` | `0xB2` | 0=off, 1=mouse bypass, 2=kbd bypass. Disables USB write, enables raw stream |
| High-speed | `hs([0\|1])` | `0xB7` | USB high-speed compatibility (persistent) |
| Screen size | `screen([W,H])` | `0xBD` | Virtual screen dimensions for `moveto()` |
| Auto-release | `release([timer_ms])` | `0xBC` | Auto-release locks/buttons/keys after timeout. 500–300000ms. Persistent |
| Help | `help()` | — | List all commands |

---

## Quick Examples

```
# ASCII — move mouse and click
.move(50,-10,)
.left(1,)
.left(0,)

# ASCII — type "Hello"
.string("Hello",)

# ASCII — press Ctrl+C
.down('ctrl',)
.press('c',)
.up('ctrl',)

# ASCII — scroll down
.wheel(-1,)

# ASCII — enable mouse stream at 10ms
.mouse(1,10,)

# Binary — relative move (dx=100, dy=50, segments=1, no bezier)
50 01 00 0D 64 00 32 00 01 00 00
# [0x50][0x0D][0x06][0x00][x:100 i16][y:50 i16][segs:1][cx1:0][cy1:0]
```

---

## Connection Notes

- Default baud: **115200**
- Connect via USB serial (COM port / `/dev/tty*`)
- After `baud()` change: host must re-open serial at new speed
- Streaming requires **baud ≥ 1,000,000**
- The `km.` prefix in commands is optional; `.command()` is the short form
- `echo(0)` suppresses ACK echoes for cleaner streaming setups
