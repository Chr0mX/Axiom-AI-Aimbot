"""Aim-lead / strafe benchmark for the prediction layer.

    python tools/bench_motion.py                 # human-readable report
    python tools/bench_motion.py --json          # machine-readable
    python tools/bench_motion.py --methods velocity,anchor   # only some estimators

Pure Python, deterministic (seeded), no GPU/Qt/win32. It feeds every estimator the same noisy,
timestamped detections of a synthetic target and scores the aim point it would produce against
where the target really is when the mouse move takes effect (capture -> output latency
``L_PIPE + L_OUT``). Estimators:

    velocity          VelocityPredictor, app defaults (history 3, horizon 10 ms)
    anchor            Anchor estimator, adaptive mode, app defaults (horizon 10 ms)
    velocity@tuned    velocity with the horizon hand-set to the true total latency (best case a user can do)
    anchor@tuned      same for anchor
    auto              Anchor estimator + latency-aware horizon + strafe response

Metrics: lead RMSE (px), time-to-recover after a reversal (ms), overshoot after a stop (px),
per-frame CPU (us).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from core.anchor_prediction import AnchorParams, AnchorTrack, predict_aim_point  # noqa: E402
from core.target_predictor import VelocityPredictor  # noqa: E402

T0 = 1000.0
W, H = 40.0, 80.0
L_PIPE = 0.025          # capture -> aim computation (measured by the app as perf_counter() - frame_capture_t)
L_OUT = 0.012           # aim computation -> the move taking effect (not measured by the app)
L_TOTAL = L_PIPE + L_OUT


def box(cx, cy):
    return (cx - W / 2, cy - H / 2, cx + W / 2, cy + H / 2)


# ── trajectories ────────────────────────────────────────────────────────────

def strafe(speed, period):
    def f(t):
        u = (t % period) / period
        x = (speed * period / 2.0) * (2 * u if u < 0.5 else 2 * (1 - u))
        return 400.0 + x, 500.0
    return f


def stop_and_go(t):
    u = t % 1.1
    d = 400.0 * min(u, 0.6)
    return 300.0 + d + 240.0 * int(t // 1.1), 500.0


SCENARIOS = {
    "constant 350 px/s": lambda t: (200.0 + 350.0 * t, 500.0),
    "fast 900 px/s": lambda t: (100.0 + 900.0 * t, 500.0 - 100.0 * t),
    "accelerating 600 px/s2": lambda t: (200.0 + 0.5 * 600.0 * min(t, 1.5) ** 2 + 900.0 * max(0.0, t - 1.5), 500.0),
    "strafe 350 px/s T=1.2s": strafe(350.0, 1.2),
    "strafe 350 px/s T=0.6s": strafe(350.0, 0.6),
    "strafe 350 px/s T=0.3s": strafe(350.0, 0.3),
    "stop-and-go": stop_and_go,
}
REVERSAL_SCENARIOS = ("strafe 350 px/s T=1.2s", "strafe 350 px/s T=0.6s")


# ── estimator adapters ──────────────────────────────────────────────────────
# update(x, y, t_cap, age, box) -> (aim_x, aim_y)

class VelocityEst:
    def __init__(self, horizon):
        self.p = VelocityPredictor(history_len=3, max_velocity_px_per_s=1200.0)
        self.h = horizon

    def update(self, x, y, t, age, b):
        return self.p.update(x, y, t, self.h)


def _anchor_params(horizon):
    p = AnchorParams()
    p.mode, p.history_len, p.horizon_s = "adaptive", 6, horizon
    p.conf_threshold, p.max_speed_px_s, p.accel_comp, p.smoothing = 0.35, 1200.0, 0.35, 0.4
    return p


class AnchorEst:
    def __init__(self, horizon):
        self.tr = AnchorTrack(_anchor_params(horizon))
        self.h = horizon

    def update(self, x, y, t, age, b):
        return predict_aim_point(self.tr, t, b, x, y, self.h, 0.9)


def _make_auto():
    """The new estimator is wired in by the same helpers the app uses (see core.ai_aiming)."""
    from core.anchor_prediction import make_auto_track, auto_aim_point

    class AutoEst:
        def __init__(self, strafe=0.6):
            self.tr = make_auto_track(strafe_response=strafe)

        def update(self, x, y, t, age, b):
            return auto_aim_point(self.tr, t, b, x, y, age, 0.9)
    return AutoEst


def build(name):
    if name == "velocity":
        return VelocityEst(0.010)
    if name == "anchor":
        return AnchorEst(0.010)
    if name == "velocity@tuned":
        return VelocityEst(L_TOTAL)
    if name == "anchor@tuned":
        return AnchorEst(L_TOTAL)
    if name == "auto":
        return _make_auto()()
    if name == "auto-nostrafe":
        return _make_auto()(strafe=0.0)
    raise KeyError(name)


def available(names):
    out = []
    for n in names:
        try:
            build(n)
            out.append(n)
        except (KeyError, ImportError):
            pass
    return out


# ── runners ─────────────────────────────────────────────────────────────────

def frames(fps, seconds, jitter, drop, rng):
    """(capture time, dropped?) per frame with timing jitter (fraction of a frame) and drop probability."""
    out = []
    for i in range(int(seconds * fps)):
        t = i / fps + rng.uniform(-jitter, jitter) / fps
        out.append((max(0.0, t), rng.random() < drop))
    return out


def lead_error(fn, name, fps=60, seconds=3.0, noise=1.0, jitter=0.0, drop=0.0, seed=1, warm=0.5):
    """RMSE (px) of the aim point vs the target at capture + L_TOTAL."""
    rng = random.Random(seed)
    est = build(name)
    se = n = 0
    for t, dropped in frames(fps, seconds, jitter, drop, rng):
        x, y = fn(t)
        nx, ny = x + rng.gauss(0, noise), y + rng.gauss(0, noise)
        age = L_PIPE + rng.gauss(0, 0.003)
        if dropped:
            continue
        px, py = est.update(nx, ny, T0 + t, age, box(nx, ny))
        if t < warm:
            continue
        tx, ty = fn(t + L_PIPE + L_OUT)
        se += (px - tx) ** 2 + (py - ty) ** 2
        n += 1
    return math.sqrt(se / max(1, n))


def reversal_recovery(fn, name, fps=60, seconds=3.0, noise=1.0, seed=1, jitter=0.0):
    """Mean time (ms) after each reversal until |aim - target| stays within max(3 px, 20 % of the lead
    distance) for 3 consecutive frames, and the mean peak error in the 150 ms after the reversal."""
    rng = random.Random(seed)
    est = build(name)
    ts, errs = [], []
    for t, _ in frames(fps, seconds, jitter, 0.0, rng):
        x, y = fn(t)
        nx, ny = x + rng.gauss(0, noise), y + rng.gauss(0, noise)
        px, py = est.update(nx, ny, T0 + t, L_PIPE, box(nx, ny))
        tx, ty = fn(t + L_TOTAL)
        ts.append(t)
        errs.append(math.hypot(px - tx, py - ty))
    # reversal instants: sign change of the true x velocity
    rev = [ts[i] for i in range(1, len(ts) - 1)
           if (fn(ts[i] + 0.001)[0] - fn(ts[i] - 0.001)[0]) * (fn(ts[i - 1] + 0.001)[0] - fn(ts[i - 1] - 0.001)[0]) < 0]
    rec, peaks = [], []
    thr = max(3.0, 0.2 * 350.0 * L_TOTAL)
    last = -1.0
    for r in rev:
        if r < 0.7 or r - last < 0.2:
            continue
        last = r
        idx = [i for i, t in enumerate(ts) if t >= r]
        if not idx:
            continue
        i0 = idx[0]
        peaks.append(max(errs[i0:i0 + max(1, int(0.15 * fps))] or [0.0]))
        ok = 0
        t_rec = None
        for i in range(i0, len(ts)):
            ok = ok + 1 if errs[i] <= thr else 0
            if ok >= 3:
                t_rec = ts[i - 2] - r
                break
            if ts[i] - r > 0.6:
                break
        rec.append(1000.0 * (t_rec if t_rec is not None else 0.6))
    if not rec:
        return float("nan"), float("nan")
    return sum(rec) / len(rec), sum(peaks) / len(peaks)


def stop_overshoot(name, fps=60, noise=1.0, seed=1):
    """Max distance (px) the aim point travels past the point where the target stopped, within 0.5 s."""
    rng = random.Random(seed)
    est = build(name)
    stop_t, stop_x = 1.0, 200.0 + 400.0 * 1.0
    fn = lambda t: (200.0 + 400.0 * min(t, stop_t), 500.0)  # noqa: E731
    worst = 0.0
    for t, _ in frames(fps, 2.0, 0.0, 0.0, rng):
        x, y = fn(t)
        nx, ny = x + rng.gauss(0, noise), y + rng.gauss(0, noise)
        px, _py = est.update(nx, ny, T0 + t, L_PIPE, box(nx, ny))
        if stop_t + 0.05 <= t <= stop_t + 0.5:
            worst = max(worst, px - stop_x)
    return worst


def cpu_us(name, frames_n=20000, repeats=5):
    paths = [(500 + 80 * math.sin(i / 20), 500 + 40 * math.sin(i / 13)) for i in range(frames_n)]
    best = float("inf")
    for _ in range(repeats):
        est = build(name)
        s = time.perf_counter()
        for i, (x, y) in enumerate(paths):
            est.update(x, y, T0 + i / 144.0, L_PIPE, box(x, y))
        best = min(best, (time.perf_counter() - s) / frames_n * 1e6)
    return best


# ── closed loop (the real process_aiming + PID + output delay) ─────────────────────────────────────────
# target -> capture -> L_PIPE -> process_aiming()/ai_loop's miss branch -> PID -> L_OUT -> crosshair.
# Needs numpy/cv2 (core.inference imports them); the rest of this tool does not.

CL_CONFIGS = {
    "off": dict(prediction_enabled=False),
    "velocity": dict(prediction_enabled=True, prediction_method="velocity"),
    "velocity@tuned": dict(prediction_enabled=True, prediction_method="velocity", prediction_horizon_ms=L_TOTAL * 1000),
    "anchor": dict(prediction_enabled=True, prediction_method="anchor"),
    "auto": dict(prediction_enabled=True, prediction_method="auto", strafe_response=0.6),
    "auto-nostrafe": dict(prediction_enabled=True, prediction_method="auto", strafe_response=0.0),
}


def _cl_config(over):
    from types import SimpleNamespace
    cfg = SimpleNamespace(
        aim_part="center", head_height_ratio=0.26, aim_custom_y_pct=50.0, aim_adaptive_ratio_enabled=False,
        aim_posture_aware_enabled=False, sticky_lock_enabled=True, lock_persistence_ms=250.0,
        lost_target_timeout_ms=1500.0, reacquire_radius_px=150.0, min_lock_duration_ms=0.0,
        target_switch_threshold=0.0, min_confidence=0.5, tracking_confidence=0.0,
        target_priority_mode="distance", target_priority_confidence_weight=0.5,
        kalman_enabled=False, kalman_process_noise=0.01, kalman_measurement_noise=0.1,
        prediction_enabled=False, prediction_horizon_ms=10.0, prediction_max_velocity=1200.0,
        prediction_history_len=3, prediction_method="velocity", prediction_adaptive_lead=False,
        prediction_capture_timestamps=False, cam_motion_comp_enabled=False,
        anchor_enabled=False, anchor_mode="adaptive", anchor_strength=1.0, anchor_velocity_smoothing=0.4,
        anchor_accel_comp=0.35, anchor_confidence_threshold=0.35, anchor_history_len=6,
        anchor_max_distance_pct=25.0, fov_size=640, fov_height=640,
        aim_y_reduce_enabled=False, aim_y_reduce_delay=0.0, aim_y_reduce_settle_px=0.0, aim_y_reduce_floor=0.1,
        aim_y_reduce_ramp=0.0, aim_y_vel_restore_px_s=0.0, humanization=None, max_move_per_frame_px=0.0,
        aim_ema_smoothing_enabled=False, aim_ema_smoothing=0.5, mouse_move_method="sendinput",
        strafe_response=0.0, auto_lead_max_ms=60.0,
    )
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


def _load_aiming():
    import types
    stub = types.ModuleType("win_utils")
    stub.send_mouse_move = lambda dx, dy, method=None: None
    sys.modules.setdefault("win_utils", stub)
    import core.ai_aiming as A
    return A


def closed_loop(name, fn, fps=60, seconds=3.0, extra=None):
    """Run the real aiming code against ``fn``; RMS pointing error (px) after a 0.7 s warm-up."""
    errs = [e for t, e in _cl_trace(name, fn, fps, seconds, extra) if t >= 0.7]
    return {"rms": math.sqrt(sum(e * e for e in errs) / max(1, len(errs)))}


def closed_loop_reversal_peak(name, fn, fps=60, seconds=4.0, extra=None):
    """Mean over reversals of the peak pointing error in the 200 ms after each one."""
    A = _load_aiming()  # noqa: F841  (ensures the stub is in place)
    peaks = []
    # re-run recording the whole error trace
    trace = _cl_trace(name, fn, fps, seconds, extra)
    rev = [t for t, _ in trace[1:-1] if (fn(t + 0.001)[0] - fn(t - 0.001)[0]) * (fn(t - 1.0 / fps + 0.001)[0] - fn(t - 1.0 / fps - 0.001)[0]) < 0]
    last = -1.0
    for r in rev:
        if r < 0.9 or r - last < 0.2:
            continue
        last = r
        seg = [e for t, e in trace if r <= t <= r + 0.2]
        if seg:
            peaks.append(max(seg))
    return sum(peaks) / len(peaks) if peaks else float("nan")


def _cl_trace(name, fn, fps, seconds, extra):
    """[(t_cap, |crosshair - target|)] for a detection-only run (same loop as closed_loop)."""
    from collections import deque
    A = _load_aiming()
    from core.ai_loop_state import LoopState
    from core.inference import PIDController
    over = dict(CL_CONFIGS[name])
    over.update(extra or {})
    cfg = _cl_config(over)
    st = LoopState()
    pid_x, pid_y = PIDController(0.26, 0.0, 0.12), PIDController(0.26, 0.0, 0.12)
    A._predictor = None
    A._kalman = None
    cross = [fn(0.0)[0], 500.0]
    fifo = deque()
    clock = {"t": 0.0}
    rng = random.Random(1)
    real_pc = time.perf_counter
    A.send_mouse_move = lambda dx, dy, method=None: fifo.append((clock["t"] + L_OUT, dx, dy))
    time.perf_counter = lambda: T0 + clock["t"]
    out = []
    try:
        for t_cap, _ in frames(fps, seconds, 0.0, 0.0, rng):
            t_proc = t_cap + L_PIPE
            clock["t"] = t_proc
            while fifo and fifo[0][0] <= t_proc:
                _, dx, dy = fifo.popleft()
                cross[0] += dx
                cross[1] += dy
            st.frame_capture_t = T0 + t_cap
            st.aiming_start_time = T0
            x, y = fn(t_cap)
            nx, ny = x + rng.gauss(0, 1.0), y + rng.gauss(0, 1.0)
            A.process_aiming(cfg, [[nx - W / 2, ny - H / 2, nx + W / 2, ny + H / 2]], cross[0], cross[1],
                             pid_x, pid_y, "sendinput", st, T0 + t_proc, confidences=[0.9])
            out.append((t_cap, abs(cross[0] - fn(t_proc)[0])))
    finally:
        time.perf_counter = real_pc
        A.send_mouse_move = lambda dx, dy, method=None: None
    return out


CL_METHODS = ("off", "velocity", "velocity@tuned", "anchor", "auto-nostrafe", "auto")


def run_closed_loop(only=None):
    global CL_METHODS
    if only:
        CL_METHODS = tuple(m for m in CL_METHODS if m in only)
    res = {"lead": {}, "rates": {}, "strafe_peak": {}, "smoothing": {}}
    for sname, fn in SCENARIOS.items():
        res["lead"][sname] = {m: closed_loop(m, fn)["rms"] for m in CL_METHODS}
    fn = SCENARIOS["strafe 350 px/s T=0.6s"]
    for f in (30, 60, 144):
        res["rates"][str(f)] = {m: closed_loop(m, fn, fps=f)["rms"] for m in CL_METHODS}
        res["strafe_peak"][str(f)] = {m: closed_loop_reversal_peak(m, SCENARIOS["strafe 350 px/s T=1.2s"], fps=f)
                                      for m in CL_METHODS}
    for label, extra in {"Aim Step Smoothing 0.5": dict(aim_ema_smoothing_enabled=True),
                         "Kalman": dict(kalman_enabled=True)}.items():
        res["smoothing"][label] = {m: closed_loop(m, fn, extra=extra)["rms"] for m in CL_METHODS}
    return res


def report_closed_loop(res):
    print("\n== CLOSED LOOP: crosshair vs target, RMS pointing error (px) -- real process_aiming + PID(0.26/0/0.12) ==")
    _table("Scenarios @60 FPS", res["lead"], CL_METHODS)
    _table("Strafe T=0.6 s by update rate", res["rates"], CL_METHODS)
    _table("Peak error within 200 ms of a reversal (strafe T=1.2 s)", res["strafe_peak"], CL_METHODS)
    _table("Strafe T=0.6 s with a smoothing stage on", res["smoothing"], CL_METHODS)


# ── report ──────────────────────────────────────────────────────────────────

def run(methods):
    res = {"methods": methods}
    res["lead_rmse_60"] = {s: {m: lead_error(fn, m) for m in methods} for s, fn in SCENARIOS.items()}
    res["lead_rmse_rates"] = {str(f): {m: lead_error(SCENARIOS["strafe 350 px/s T=0.6s"], m, fps=f, seconds=3.0)
                                       for m in methods} for f in (30, 60, 144)}
    res["lead_rmse_stress"] = {
        lab: {m: lead_error(SCENARIOS["strafe 350 px/s T=0.6s"], m, fps=60, jitter=j, drop=d, noise=nz)
              for m in methods}
        for lab, (j, d, nz) in {"jitter 50 %": (0.5, 0.0, 1.0), "drop 20 %": (0.0, 0.2, 1.0),
                                "noise 2 px": (0.0, 0.0, 2.0), "all three": (0.5, 0.2, 2.0)}.items()}
    res["reversal"] = {}
    for s in REVERSAL_SCENARIOS:
        for f in (30, 60, 144):
            res["reversal"][f"{s} @{f}"] = {m: reversal_recovery(SCENARIOS[s], m, fps=f, seconds=4.0) for m in methods}
    res["overshoot"] = {m: stop_overshoot(m) for m in methods}
    res["cpu_us"] = {m: cpu_us(m) for m in methods}
    return res


def _table(title, rows, methods, fmt="{:7.1f}"):
    print(f"\n{title}")
    print("  " + " " * 30 + "".join(f"{m:>16}" for m in methods))
    for k, v in rows.items():
        cells = ""
        for m in methods:
            x = v[m]
            cells += f"{fmt.format(x[0]) + ' / ' + fmt.format(x[1]) if isinstance(x, tuple) else fmt.format(x):>16}"
        print(f"  {k:<30}{cells}")


def report(res):
    ms = res["methods"]
    print(f"Latency model: pipeline {L_PIPE * 1000:.0f} ms (measured) + output {L_OUT * 1000:.0f} ms (unmeasured) = "
          f"{L_TOTAL * 1000:.0f} ms to effect; noise sigma 1 px unless stated")
    _table("Lead RMSE (px), 60 FPS - lower is better", res["lead_rmse_60"], ms)
    _table("Lead RMSE (px) by update rate, strafe T=0.6 s", res["lead_rmse_rates"], ms)
    _table("Lead RMSE (px) under stress, strafe T=0.6 s @60", res["lead_rmse_stress"], ms)
    _table("Reversal: mean recovery ms / mean peak error px (<=150 ms)", res["reversal"], ms)
    print("\nOvershoot past a stop (px, 0.5 s window)")
    print("  " + "".join(f"{m:>16}" for m in ms))
    print("  " + "".join(f"{res['overshoot'][m]:16.1f}" for m in ms))
    print("\nCPU per frame (us)")
    print("  " + "".join(f"{m:>16}" for m in ms))
    print("  " + "".join(f"{res['cpu_us'][m]:16.2f}" for m in ms))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--closed-loop", action="store_true", help="also run the closed-loop simulation (needs numpy/cv2)")
    ap.add_argument("--cl-methods", default="", help="restrict the closed-loop configs (e.g. when running against an older checkout)")
    ap.add_argument("--methods", default="velocity,anchor,velocity@tuned,anchor@tuned,auto-nostrafe,auto")
    args = ap.parse_args(argv)
    methods = available([m for m in args.methods.split(",") if m])
    res = run(methods)
    if args.closed_loop:
        res["closed_loop"] = run_closed_loop(set(filter(None, args.cl_methods.split(","))) or None)
    if args.json:
        print(json.dumps(res, indent=1))
    else:
        report(res)
        if args.closed_loop:
            report_closed_loop(res["closed_loop"])


if __name__ == "__main__":
    main()
