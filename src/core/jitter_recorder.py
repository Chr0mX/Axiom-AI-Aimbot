"""
Axiom Jitter Recorder — standalone terminal script.

Run from project root:
    python src/core/jitter_recorder.py

Keys (no Enter needed for menu):
    r  — start recording mouse movement
    s  — stop recording & save to file
    l  — list saved patterns
    p  — play a pattern
    d  — delete a pattern
    q  — quit
"""
import sys
import os
import json
import time
import itertools
import threading
import msvcrt
from pathlib import Path
from datetime import datetime

# ── sys.path so win_utils is importable when run as a script ─────────────────
_HERE = Path(__file__).resolve().parent          # src/core/
_SRC  = _HERE.parent                             # src/
_DEPS = _SRC / "python" / "dependencies"
for _p in (str(_SRC), str(_DEPS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Pattern storage ───────────────────────────────────────────────────────────
_PATTERNS_DIR = _HERE / "jitter_patterns"
_PATTERNS_DIR.mkdir(exist_ok=True)


def _list_pattern_files():
    return sorted(_PATTERNS_DIR.glob("*.json"))


def _load_pattern(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _normalize_frames(frames: list) -> list:
    """Append a correction frame so the full loop cycle has zero net displacement."""
    if not frames:
        return frames
    net_dx = sum(f["dx"] for f in frames)
    net_dy = sum(f["dy"] for f in frames)
    if net_dx == 0 and net_dy == 0:
        return frames
    avg_dt = max(1, int(sum(f["dt_ms"] for f in frames) / len(frames)))
    return frames + [{"dx": -net_dx, "dy": -net_dy, "dt_ms": avg_dt}]


def list_patterns() -> list:
    """Return [{name, path, frame_count}, ...] for all saved patterns (GUI helper)."""
    result = []
    for p in _list_pattern_files():
        try:
            data = _load_pattern(p)
            result.append({
                "name": data.get("name", p.stem),
                "path": str(p),
                "frame_count": len(data.get("frames", [])),
            })
        except Exception:
            pass
    return result


def _save_pattern(name: str, frames: list) -> Path:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name) or "pattern"
    path = _PATTERNS_DIR / f"{safe}.json"
    data = {
        "name": name,
        "recorded_at": datetime.now().isoformat(timespec="seconds"),
        "frames": frames,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return path


# ── Mouse delta sender ────────────────────────────────────────────────────────

def _send(dx: int, dy: int, backend: str) -> None:
    dx = max(-32768, min(32767, dx))
    dy = max(-32768, min(32767, dy))
    if backend == "makcu":
        from win_utils.makcu_mouse import makcu_mouse
        makcu_mouse.move(dx, dy)
    else:
        from win_utils.mouse_move import send_mouse_move_sendinput
        send_mouse_move_sendinput(dx, dy)


# ── Recording via win32api polling ────────────────────────────────────────────

class _Recorder:
    """Poll GetCursorPos at ~1 ms and accumulate (dx, dy, dt_ms) delta frames."""

    def __init__(self):
        self._frames: list = []
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._frames = []
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> list:
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        return list(self._frames)

    def _run(self) -> None:
        try:
            import win32api
        except ImportError:
            print("[recorder] win32api not available — cannot record.")
            return

        prev_x, prev_y = win32api.GetCursorPos()
        prev_t = time.perf_counter()
        while self._running:
            time.sleep(0.001)
            x, y = win32api.GetCursorPos()
            now = time.perf_counter()
            dx, dy = x - prev_x, y - prev_y
            dt_ms = min(100, max(1, int((now - prev_t) * 1000)))
            if dx != 0 or dy != 0:
                self._frames.append({"dx": dx, "dy": dy, "dt_ms": dt_ms})
            prev_x, prev_y = x, y
            prev_t = now


# ── Replay ────────────────────────────────────────────────────────────────────

def _replay(frames: list, backend: str) -> None:
    """Loop through pattern frames until Enter or Ctrl-C."""
    print(f"  Replaying {len(frames)} frames via '{backend}'. Press Enter to stop.")
    stop_event = threading.Event()

    def _wait_enter():
        sys.stdin.readline()
        stop_event.set()

    threading.Thread(target=_wait_enter, daemon=True).start()

    try:
        for f in itertools.cycle(frames):
            if stop_event.is_set():
                break
            _send(f["dx"], f["dy"], backend)
            time.sleep(f["dt_ms"] / 1000)
    except KeyboardInterrupt:
        pass
    print("  Playback stopped.")


# ── Menu helpers ──────────────────────────────────────────────────────────────

def _getch() -> str:
    """Read one character from the terminal without echo (Windows msvcrt)."""
    ch = msvcrt.getch()
    try:
        return ch.decode("utf-8").lower()
    except Exception:
        return ""


def _pick_pattern(prompt: str) -> "dict | None":
    files = _list_pattern_files()
    if not files:
        print("  No patterns saved yet.")
        return None
    for i, p in enumerate(files, 1):
        try:
            meta = _load_pattern(p)
            n_frames = len(meta.get("frames", []))
            recorded = meta.get("recorded_at", "?")
            print(f"  [{i}] {meta.get('name', p.stem)}  ({n_frames} frames, {recorded})")
        except Exception:
            print(f"  [{i}] {p.stem}  (unreadable)")
    try:
        idx = int(input(f"  {prompt}")) - 1
        if 0 <= idx < len(files):
            return {"path": files[idx], "data": _load_pattern(files[idx])}
    except (ValueError, KeyboardInterrupt):
        pass
    print("  Invalid selection.")
    return None


def _pick_backend() -> str:
    print("  Backend: [s] sendinput (default)  [m] MAKCU")
    key = _getch()
    if key == "m":
        print("  → MAKCU")
        return "makcu"
    print("  → sendinput")
    return "sendinput"


# ── Menu actions ──────────────────────────────────────────────────────────────

_recorder = _Recorder()
_recording = False


def _cmd_record() -> None:
    global _recording
    if _recording:
        print("  Already recording. Press [s] to stop.")
        return
    _recording = True
    _recorder.start()
    print("  Recording… shake your mouse. Press [s] to stop & save.")


def _cmd_stop() -> None:
    global _recording
    if not _recording:
        print("  Not currently recording.")
        return
    frames = _recorder.stop()
    _recording = False
    print(f"  Captured {len(frames)} movement frames.")
    if not frames:
        print("  Nothing to save (no mouse movement detected).")
        return
    net_dx = sum(f["dx"] for f in frames)
    net_dy = sum(f["dy"] for f in frames)
    frames = _normalize_frames(frames)
    if net_dx != 0 or net_dy != 0:
        print(f"  (corrected net drift: dx={net_dx}, dy={net_dy})")
    name = input("  Pattern name: ").strip() or "jitter"
    path = _save_pattern(name, frames)
    print(f"  Saved → {path}")


def _cmd_list() -> None:
    files = _list_pattern_files()
    if not files:
        print("  No patterns saved yet.")
        return
    for i, p in enumerate(files, 1):
        try:
            meta = _load_pattern(p)
            n = len(meta.get("frames", []))
            ts = meta.get("recorded_at", "?")
            print(f"  [{i}] {meta.get('name', p.stem)}  ({n} frames, {ts})")
        except Exception:
            print(f"  [{i}] {p.stem}  (unreadable)")


def _cmd_play() -> None:
    sel = _pick_pattern("Play pattern number: ")
    if not sel:
        return
    backend = _pick_backend()
    _replay(sel["data"]["frames"], backend)


def _cmd_delete() -> None:
    sel = _pick_pattern("Delete pattern number: ")
    if not sel:
        return
    confirm = input(f"  Delete '{sel['data'].get('name', '?')}'? [y/N]: ").strip().lower()
    if confirm == "y":
        sel["path"].unlink(missing_ok=True)
        print("  Deleted.")
    else:
        print("  Cancelled.")


# ── Main loop ─────────────────────────────────────────────────────────────────

def main() -> None:
    print("\nAxiom Jitter Recorder")
    print("=" * 23)

    while True:
        rec_tag = "  [RECORDING]" if _recording else ""
        print(f"\n{rec_tag}  [r] Record  [s] Stop & save  [l] List  [p] Play  [d] Delete  [q] Quit")
        key = _getch()

        if key == "r":
            _cmd_record()
        elif key == "s":
            _cmd_stop()
        elif key == "l":
            _cmd_list()
        elif key == "p":
            _cmd_play()
        elif key == "d":
            _cmd_delete()
        elif key == "q":
            if _recording:
                _recorder.stop()
            print("Bye.")
            break


if __name__ == "__main__":
    main()
