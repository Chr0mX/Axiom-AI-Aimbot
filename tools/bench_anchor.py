"""Anchor Prediction benchmark: accuracy, jitter, lock stability, CPU cost, lag.

    python tools/bench_anchor.py            # human-readable report
    python tools/bench_anchor.py --json     # machine-readable

Synthetic, deterministic scenarios (seeded noise); pure Python, no GPU/Qt needed.
It measures the prediction layer in isolation and its effect on the Adaptive /
Dynamic FOV and on lock association, always against the same run with Anchor
Prediction off (the "hold the last box" baseline).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
import tracemalloc
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from core.ai_loop_state import LoopState  # noqa: E402
from core.ai_loop_utils import anchor_on_match, anchor_ref_box, compute_effective_fov  # noqa: E402
from core.anchor_prediction import AnchorParams, AnchorTrack, MODES, predict_aim_point  # noqa: E402
from core.target_predictor import EmaPredictor, RollingVelocityPredictor, VelocityPredictor  # noqa: E402
from core.target_tracking import edge_ratio, find_continuation  # noqa: E402

T0 = 1000.0
W, H = 40.0, 80.0
CX, CY = 500.0, 500.0


def box(cx, cy):
    return (cx - W / 2, cy - H / 2, cx + W / 2, cy + H / 2)


SCENARIOS = {
    "stationary": lambda t: (500.0, 500.0),
    "slow (60 px/s)": lambda t: (300 + 60 * t, 500.0),
    "fast (900 px/s)": lambda t: (100 + 900 * t, 500 - 150 * t),
    "accelerating": lambda t: (200 + 0.5 * 1200 * t * t, 500.0),
    "sudden reversal": lambda t: (500 + 500 * t, 500.0) if t < 0.5 else (750 - 500 * (t - 0.5), 500.0),
    "zig-zag": lambda t: (500 + 90 * math.sin(4 * t), 500 + 60 * math.sin(7 * t)),
}


def make_track(mode, history=6, horizon=0.1):
    p = AnchorParams()
    p.mode, p.history_len, p.horizon_s = mode, history, horizon
    p.conf_threshold, p.max_speed_px_s, p.accel_comp, p.smoothing = 0.2, 1e6, 0.5, 0.4
    return AnchorTrack(p)


def prediction_error(fn, mode, fps=60, seconds=3.0, noise=1.0, seed=1, horizon=0.1):
    """(RMSE of the predicted centre, RMSE of the hold-last baseline, lead jitter) at ``horizon``."""
    rng = random.Random(seed)
    tr = make_track(mode, horizon=horizon)
    se_p = se_b = 0.0
    n = 0
    leads = []
    for i in range(int(seconds * fps)):
        t = i / fps
        x, y = fn(t)
        tr.update(T0 + t, box(x + rng.gauss(0, noise), y + rng.gauss(0, noise)), 0.9)
        p = tr.predict()
        if t < 0.5:
            continue
        tx, ty = fn(t + horizon)
        se_p += (p.pcx - tx) ** 2 + (p.pcy - ty) ** 2
        se_b += (x - tx) ** 2 + (y - ty) ** 2
        leads.append((p.dx, p.dy))
        n += 1
    # Lead noise with the (legitimate) trend removed: RMS of the second difference / sqrt(6).
    d2 = [(leads[i][0] - 2 * leads[i - 1][0] + leads[i - 2][0], leads[i][1] - 2 * leads[i - 1][1] + leads[i - 2][1])
          for i in range(2, len(leads))]
    jitter = math.sqrt(sum(a * a + b * b for a, b in d2) / len(d2) / 6.0)
    return math.sqrt(se_p / n), math.sqrt(se_b / n), jitter


def fov_run(mode, safeguards, anchor, kind, fps=60, n=240):
    c = SimpleNamespace(
        fov_size=300, fov_height=300, fov_reduce_on_target_enabled=True, fov_min_size_pct=40.0,
        fov_min_size_duration=0.5, fov_recovery_duration_ms=800, fov_expand_delay_ms=300,
        fov_shrink_mode=mode, fov_dynamic_safeguards=safeguards, min_confidence=0.5, tracking_confidence=0.0,
        fov_circle_filter_enabled=False, screenshot_method="mss", screen_width=1000, screen_height=1000)
    if anchor:
        c.anchor_enabled, c.anchor_mode, c.anchor_horizon_ms = True, "adaptive", 150.0
        c.anchor_accel_comp, c.anchor_confidence_threshold, c.anchor_max_distance_pct = 1.0, 0.2, 100.0
        c.anchor_fov_influence_adaptive = c.anchor_fov_influence_dynamic = 0.5
        c.prediction_max_velocity = 5000.0
    st = LoopState()
    out = []
    for i in range(n):
        t = T0 + i / fps
        s = i / fps
        if kind == "approach":
            x = CX + 10 + 0.5 * 250 * s * s if s < 0.8 else CX + 10 + 0.5 * 250 * 0.64 - 200 * (s - 0.8)
            y = CY + 20 * math.sin(2 * s)
        else:
            x, y = CX + 90 * math.sin(4 * s), CY + 60 * math.sin(7 * s)
        w, h = compute_effective_fov(c, st, t)
        b = [x - 20, y - 40, x + 20, y + 40]
        st.locked_box = b
        c.fov_effective_size, c.fov_effective_height = w, h
        st.tracker.on_match(t, b, 0.9, edge_ratio=edge_ratio(b, CX, CY, w, h, False), same_target=i > 0)
        if anchor:
            anchor_on_match(c, st, b, 0.9, t, CX, CY, w, h)
        out.append(st.fov_scale)
    return out


def fov_metrics(s):
    return (sum(abs(b - a) for a, b in zip(s, s[1:])),
            sum(abs(s[i + 1] - 2 * s[i] + s[i - 1]) for i in range(1, len(s) - 1)),
            sum(s) / len(s))


def lock_stability(fps, trials=60, seed=3):
    """Two crossing targets (800 px/s vs 300 px/s opposite way), lock follows what is
    chosen. Returns (wrong re-locks, holds) per trial for baseline and with the prediction."""
    res = {}
    for use_anchor in (False, True):
        rng = random.Random(seed)
        wrong = holds = 0
        for _ in range(trials):
            cfg = SimpleNamespace(
                anchor_enabled=use_anchor, anchor_mode="adaptive", anchor_horizon_ms=100.0,
                anchor_accel_comp=0.5, anchor_confidence_threshold=0.2, anchor_history_len=6,
                anchor_max_distance_pct=100.0, prediction_max_velocity=5000.0, fov_size=600, fov_height=600,
                screenshot_method="mss", screen_width=2000, screen_height=1000, reacquire_radius_px=150,
                lost_target_timeout_ms=1500)
            st = LoopState()
            y0 = 500 + rng.uniform(-10, 10)
            ax0, bx0 = 200 + rng.uniform(-20, 20), 1000 + rng.uniform(-20, 20)
            locked = None
            on_a = True                                   # which target the lock currently follows
            for i in range(int(0.9 * fps)):
                t = T0 + i / fps
                s = i / fps
                a = [ax0 + 800 * s - 20, y0 - 40, ax0 + 800 * s + 20, y0 + 40]
                b = [bx0 - 300 * s - 20, y0 + rng.uniform(-6, 6) - 40, bx0 - 300 * s + 20, y0 + 40]
                if locked is not None:
                    ref = anchor_ref_box(cfg, st, t) if use_anchor else None
                    idx, amb = find_continuation([a, b], locked_box=locked, tracker=st.tracker, config=cfg,
                                                 now=t, iou_thresh=0.2, ref_box=ref)
                    if amb or idx is None:
                        holds += 1
                        continue
                    if (idx == 0) != on_a:
                        wrong += 1
                    on_a = idx == 0
                    chosen = [a, b][idx]
                else:
                    chosen = a
                st.tracker.on_match(t, chosen, 0.9, edge_ratio=0.3, same_target=locked is not None)
                anchor_on_match(cfg, st, chosen, 0.9, t, 500, 500, 600, 600)
                locked = chosen
        res[use_anchor] = (wrong / trials, holds / trials)
    return res


def cpu_cost(frames=5000, repeats=7):
    """Per-frame cost of the whole FOV + tracker (+ anchor) step: best of ``repeats`` passes, because a
    single pass on a shared machine is dominated by scheduling noise."""
    cfg = SimpleNamespace(
        fov_size=300, fov_height=300, fov_reduce_on_target_enabled=True, fov_min_size_pct=40.0,
        fov_min_size_duration=0.5, fov_recovery_duration_ms=800, fov_expand_delay_ms=300, fov_shrink_mode="dynamic",
        fov_dynamic_safeguards=True, min_confidence=0.5, tracking_confidence=0.0, fov_circle_filter_enabled=False,
        screenshot_method="mss", screen_width=1000, screen_height=1000, anchor_enabled=True, anchor_mode="adaptive",
        anchor_horizon_ms=100.0, anchor_accel_comp=0.35, anchor_confidence_threshold=0.35, anchor_history_len=6,
        anchor_max_distance_pct=25.0, anchor_fov_influence_adaptive=0.5, anchor_fov_influence_dynamic=0.5,
        prediction_max_velocity=1200.0)
    paths = [(CX + 80 * math.sin(i / 20), CY + 40 * math.sin(i / 13)) for i in range(frames)]
    out = {}
    for label, enabled in (("off", False), ("on", True)):
        cfg.anchor_enabled = enabled
        best = float("inf")
        for _ in range(repeats):
            st = LoopState()
            t_start = time.perf_counter()
            for i, (x, y) in enumerate(paths):
                t = T0 + i / 144.0
                w, h = compute_effective_fov(cfg, st, t)
                b = [x - 20, y - 40, x + 20, y + 40]
                st.locked_box = b
                cfg.fov_effective_size = cfg.fov_effective_height = w
                st.tracker.on_match(t, b, 0.9, edge_ratio=0.3, same_target=i > 0)
                anchor_on_match(cfg, st, b, 0.9, t, CX, CY, w, h)
            best = min(best, (time.perf_counter() - t_start) / frames * 1e6)
        out[label] = best
    cfg.anchor_enabled = True
    st = LoopState()
    for i in range(500):                                   # warm up
        x = CX + 80 * math.sin(i / 20)
        anchor_on_match(cfg, st, [x - 20, 460, x + 20, 540], 0.9, T0 + i / 144.0, CX, CY, 300, 300)
    tracemalloc.start()
    snap_a = tracemalloc.take_snapshot()
    for i in range(500, 5500):
        x = CX + 80 * math.sin(i / 20)
        anchor_on_match(cfg, st, [x - 20, 460, x + 20, 540], 0.9, T0 + i / 144.0, CX, CY, 300, 300)
    snap_b = tracemalloc.take_snapshot()
    tracemalloc.stop()
    retained = sum(s.size_diff for s in snap_b.compare_to(snap_a, "filename") if "anchor_prediction" in str(s.traceback)
                   or "ai_loop_utils" in str(s.traceback))
    out["retained_bytes_5000_frames"] = retained
    return out


AIM_LEAD_METHODS = ("velocity", "ema", "rolling", "anchor linear", "anchor adaptive")


def _aim_lead_predictor(method, horizon, fps):
    """(update(x, y, t, box) -> (x, y)) for one aim-lead Prediction Method, configured like the app
    does at the given horizon (Rolling's lead is a frame count, so it gets horizon * fps frames)."""
    if method == "velocity":
        pr = VelocityPredictor(history_len=3, max_velocity_px_per_s=5000.0)
        return lambda x, y, t, b: pr.update(x, y, t, horizon)
    if method == "ema":
        pr = EmaPredictor()
        return lambda x, y, t, b: pr.update(x, y, t, horizon)
    if method == "rolling":
        pr = RollingVelocityPredictor()
        return lambda x, y, t, b: pr.update(x, y, t, horizon * fps)
    tr = make_track(method.split()[1])
    tr.params.max_speed_px_s = 5000.0
    return lambda x, y, t, b: predict_aim_point(tr, t, b, x, y, horizon, 0.9)


def aim_lead_error(fn, method, fps=60, seconds=3.0, noise=1.0, seed=1, horizon=0.01):
    """RMSE (px) of the led aim point against where the target really is ``horizon`` later, and the RMSE of
    not predicting at all. Same noisy detections for every method (seeded)."""
    rng = random.Random(seed)
    update = _aim_lead_predictor(method, horizon, fps)
    se_p = se_b = 0.0
    n = 0
    for i in range(int(seconds * fps)):
        t = i / fps
        x, y = fn(t)
        nx, ny = x + rng.gauss(0, noise), y + rng.gauss(0, noise)
        px, py = update(nx, ny, T0 + t, box(nx, ny))
        if t < 0.5:
            continue
        tx, ty = fn(t + horizon)
        se_p += (px - tx) ** 2 + (py - ty) ** 2
        se_b += (nx - tx) ** 2 + (ny - ty) ** 2
        n += 1
    return math.sqrt(se_p / n), math.sqrt(se_b / n)


def aim_lead_cpu(frames=20000, repeats=5):
    """Microseconds per frame of each aim-lead method's own update (best of ``repeats``)."""
    out = {}
    paths = [(CX + 80 * math.sin(i / 20), CY + 40 * math.sin(i / 13)) for i in range(frames)]
    for method in AIM_LEAD_METHODS:
        update = _aim_lead_predictor(method, 0.01, 144)
        best = float("inf")
        for _ in range(repeats):
            update = _aim_lead_predictor(method, 0.01, 144)
            t_start = time.perf_counter()
            for i, (x, y) in enumerate(paths):
                update(x, y, T0 + i / 144.0, (x - 20, y - 40, x + 20, y + 40))
            best = min(best, (time.perf_counter() - t_start) / frames * 1e6)
        out[method] = best
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    rep = {"prediction": {}, "aim_lead": {}, "fov": {}, "stability": {}, "cpu": cpu_cost(),
           "aim_lead_cpu_us": aim_lead_cpu()}

    for name, fn in SCENARIOS.items():
        rep["prediction"][name] = {}
        for mode in MODES[1:]:
            rmse, base, jit = prediction_error(fn, mode)
            rep["prediction"][name][mode] = {"rmse_px": rmse, "baseline_px": base, "lead_jitter_px": jit}
    for fps in (30, 60, 144, 240):
        rep["prediction"][f"fast @ {fps} fps"] = {}
        for mode in ("linear", "smoothed", "adaptive"):
            rmse, base, jit = prediction_error(SCENARIOS["fast (900 px/s)"], mode, fps=fps)
            rep["prediction"][f"fast @ {fps} fps"][mode] = {"rmse_px": rmse, "baseline_px": base, "lead_jitter_px": jit}
    for name, fn in SCENARIOS.items():
        rep["aim_lead"][name] = {m: aim_lead_error(fn, m)[0] for m in AIM_LEAD_METHODS}
        rep["aim_lead"][name]["baseline"] = aim_lead_error(fn, "velocity")[1]
    for mode, sg in (("adaptive", False), ("dynamic", False), ("dynamic", True)):
        for kind in ("approach", "zigzag"):
            off, on = fov_metrics(fov_run(mode, sg, False, kind)), fov_metrics(fov_run(mode, sg, True, kind))
            rep["fov"][f"{mode}{'+safeguards' if sg else ''} / {kind}"] = {
                "total_variation": (off[0], on[0]), "jerk": (off[1], on[1]), "mean_scale": (off[2], on[2])}
    for fps in (10, 20, 30, 60):
        r = lock_stability(fps)
        rep["stability"][f"{fps} fps"] = {"wrong_relocks_per_trial": (r[False][0], r[True][0]),
                                           "holds_per_trial": (r[False][1], r[True][1])}

    if args.json:
        print(json.dumps(rep, indent=2))
        return
    print("== Prediction error at a 100 ms horizon (RMSE px; baseline = hold the last box; noise sigma 1 px) ==")
    for name, modes in rep["prediction"].items():
        row = "  ".join(f"{m}: {v['rmse_px']:6.2f}" for m, v in modes.items())
        base = next(iter(modes.values()))["baseline_px"]
        jit = "  ".join(f"{m}: {v['lead_jitter_px']:5.2f}" for m, v in modes.items())
        print(f"{name:22s} baseline {base:7.2f} | {row}\n{'':22s} lead jitter (px)        | {jit}")
    print("\n== Aim-lead Prediction Method at a 10 ms horizon, 60 FPS (RMSE px of the led aim point; "
          "baseline = no prediction; noise sigma 1 px) ==")
    print(f"{'':22s} {'baseline':>9s} " + " ".join(f"{m:>15s}" for m in AIM_LEAD_METHODS))
    for name, row in rep["aim_lead"].items():
        print(f"{name:22s} {row['baseline']:9.2f} " + " ".join(f"{row[m]:15.2f}" for m in AIM_LEAD_METHODS))
    print("per-frame update cost (us): " + "  ".join(f"{m} {v:.2f}" for m, v in rep["aim_lead_cpu_us"].items()))
    print("\n== FOV (off -> on): total variation / jerk / mean scale ==")
    for k, v in rep["fov"].items():
        print(f"{k:28s} TV {v['total_variation'][0]:5.2f}->{v['total_variation'][1]:5.2f}  "
              f"jerk {v['jerk'][0]:5.3f}->{v['jerk'][1]:5.3f}  mean {v['mean_scale'][0]:.3f}->{v['mean_scale'][1]:.3f}")
    print("\n== Lock stability, two crossing targets (off -> on) ==")
    for k, v in rep["stability"].items():
        print(f"{k:8s} wrong re-locks/trial {v['wrong_relocks_per_trial'][0]:.2f}->{v['wrong_relocks_per_trial'][1]:.2f}   "
              f"holds/trial {v['holds_per_trial'][0]:.2f}->{v['holds_per_trial'][1]:.2f}")
    c = rep["cpu"]
    print(f"\n== CPU (per detection frame, whole FOV+tracker+anchor step) ==\n"
          f"off {c['off']:.2f} us   on {c['on']:.2f} us   added {c['on'] - c['off']:.2f} us   "
          f"retained memory over 5000 frames: {c['retained_bytes_5000_frames']} bytes")


if __name__ == "__main__":
    main()
