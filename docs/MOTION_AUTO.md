# Auto Lead and Strafe Response

Two opt-in additions to the aim-lead layer that share the existing motion estimator (`AnchorTrack`,
`src/core/anchor_prediction.py`):

| Feature | Setting (Aim page, Advanced) | Default |
|---|---|---|
| **Auto lead** | Aim Lead Method = **Auto**; *Auto Lead Limit* (20-150 ms) | method `velocity`, limit 60 ms |
| **Strafe response** | *Strafe Response* 0-100 % (0 = off) | 0 (off) |

Everything is off by default; with the defaults the estimator output is **bit-identical** to the previous
build (golden digests in `tests/test_motion_auto.py`, and the closed-loop numbers below for `off` / `velocity` /
`anchor` are identical on HEAD and on this branch).

> **Occlusion Coasting was tried and removed.** A third feature (keep aiming along the estimated path while the
> locked target is hidden behind cover) shipped in `6028729` and was taken out again because it did not work in
> use. A hidden target's heading is unknowable, and even in the simulation it lost to simply freezing the
> crosshair whenever the target reversed inside cover for longer than the coast window. Old configs that still
> contain `tracking.occlusion.*` simply have those keys ignored.

## 1. What the audit found

* **Lead.** The horizon was a fixed `prediction_horizon_ms` (10 ms) from the newest sample; the capture->aim
  age (`frame_capture_t` is carried per frame) was never added, and *Adaptive Lead* keys on **mouse** speed.
  Only the Anchor method estimated acceleration.
* **Strafe.** No reversal logic in the velocity predictor or Kalman; Anchor's `turn_ema` only scaled the lead
  in `adaptive` mode and decayed over ~0.3 s, so after a reversal the lead stayed damped for hundreds of ms
  (the benchmark shows 600 ms+ to recover for the stock Anchor method).

## 2. Design

One track per lock (`LoopState.aim_anchor`) owns the motion state.

* **Auto horizon** (`AnchorTrack.auto_horizon`): `H = clamp(smoothed(perf_counter() - frame_capture_t) + 10 ms,
  5 ms, limit)`. Samples are timed by capture time (`ai_aiming._track_clock`) so the latency is counted once.
  The 10 ms stands for the aim->effect delay the app cannot measure. Auto also fixes the estimator to the raw
  fit with acceleration x0.5 (no smoothing lag).
* **Strafe response** (`AnchorTrack._detect_reversal`): slope of the newest >= 35 ms vs the span just before it,
  both measured against the new sample. Opposite signs, each beyond `(4 - response)` standard errors of a slope
  (`sqrt(2)*sigma/span`, sigma = slow outlier-clipped jitter estimate, floor 1.5 px), is a reversal. Effect: cut
  the history to the post-turn samples, zero the acceleration, clear the turn signal, fit unsmoothed. On the
  output side (`process_aiming`, first frame of a reversal): reset the PID integral and the Kalman filter, and
  skip Aim Step Smoothing for 100 ms. Time-based, so it behaves the same at 30/60/144/240 Hz.

## 3. Files

`core/anchor_prediction.py` (reversal detector, `auto_horizon`, `auto_aim_point`, `make_auto_track`),
`core/ai_aiming.py` (`auto` branch, `_track_clock`, output-side strafe handling), `core/ai_loop_state.py`,
`core/ai_loop_utils.py` (`anchor_reset`), `core/inference.py` (`PIDController.reset_integral`),
`core/config.py` (2 fields + validators), `core/web_control_settings.py`, `web_control_client/*` (assets v26),
`gui/fluent_app/pages/aim_page.py`, `language_data/English_English.json`, `tools/bench_motion.py`, tests
`test_motion_auto.py` / `test_motion_wiring.py`.

## 4. Results (`python tools/bench_motion.py --closed-loop`)

Latency model: 25 ms measured pipeline + 12 ms unmeasured output; detector noise sigma 1 px; seeded, so
reproducible. "velocity@tuned" is the velocity predictor with its horizon hand-set to the true 37 ms - the best
a user can do manually. Closed loop = the real `process_aiming`, PID 0.26/0/0.12, output delay, simulated crosshair.

**Lead error against where the target really is (px RMS, 60 FPS, estimator only):**

| scenario | velocity | anchor | velocity@tuned | auto |
|---|---|---|---|---|
| constant 350 px/s | 9.9 | 10.0 | 3.6 | **2.4** |
| fast 900 px/s | 24.8 | 25.2 | 3.6 | **3.2** |
| accelerating 600 px/s2 | 22.0 | 22.7 | 3.6 | **3.4** |
| strafe T=1.2 s | 9.8 | 11.3 | 6.6 | **6.3** |
| strafe T=0.6 s | 10.0 | 12.4 | **8.4** | 8.9 |
| strafe T=0.3 s | **10.5** | 12.0 | 11.3 | 12.9 |
| stop-and-go | 8.2 | 8.9 | 4.8 | **4.5** |

Auto matches a hand-tuned horizon without being told the latency, and cuts the default's error by 75-87 % on
straight/accelerating motion. On very fast strafes (T=0.3 s, a reversal every 150 ms) it is not better than the
default: the detector needs ~35-50 ms to see a turn.

**Reversal recovery** (ms until the estimate is within tolerance of the new heading; strafe T=1.2 s):

| rate | velocity | anchor | velocity@tuned | auto (strafe 0) | auto (strafe 60 %) |
|---|---|---|---|---|---|
| 30 Hz | 600+ | 600+ | 222 | 200 | **133** |
| 60 Hz | 600+ | 600+ | 144 | 89 | **83** |
| 144 Hz | 600+ | 600+ | 213 | 54 | **54** |

(T=0.6 s: 30 Hz 274 -> 185 ms, 60 Hz 161 -> 135 ms, 144 Hz 109 -> 110 ms.) Overshoot past a stop: 1.9 px (auto) vs
2.6 (velocity) / 3.2 (anchor). At 144 Hz the fixed fit window already recovers quickly, so the detector adds
most below ~60 Hz and under noise (stress, T=0.6 s: drop 20 % 10.0 -> 9.7, noise 2 px 9.9 -> 9.9, all three
10.9 -> 10.6 px) - a modest gain, not a large one.

**Closed loop - pointing error of the crosshair vs the target (px RMS, 60 FPS):**

| scenario | off | velocity | velocity@tuned | anchor | auto |
|---|---|---|---|---|---|
| constant 350 px/s | 31.3 | 27.8 | 18.4 | 28.1 | **19.4** |
| fast 900 px/s | 80.3 | 71.3 | 47.0 | 71.8 | **49.2** |
| accelerating | 73.1 | 65.0 | 43.1 | 65.7 | **45.2** |
| strafe T=0.6 s | 24.2 | 21.7 | 15.8 | 24.1 | 17.3 |
| strafe T=0.6 s @144 Hz | 16.9 | 14.0 | 8.1 | 16.7 | 9.6 |
| strafe T=0.6 s @30 Hz | 29.9 | 28.4 | 24.6 | 30.0 | 27.2 |
| stop-and-go | 24.5 | 21.8 | 14.7 | 22.6 | 15.8 |

Auto cuts the default velocity lead's pointing error by 20-31 % at 60 and 144 Hz with no tuning, but only 4 % at 30 Hz
(27.2 vs 28.4), where one frame is already 33 ms of target travel.
Peak error right after a reversal is *not* improved at 144 Hz (auto 23.0 vs off 18.9 / velocity 16.6): a longer
horizon leads the wrong way for the ~30 ms the detector needs, an inherent cost of any lead that is not
anticipating the turn.

**Overhead** (per detection frame, pure Python): velocity 0.5 us, anchor ~13 us, auto ~16 us; no per-frame
allocation in the estimator path.

## 5. Tests

`tests/test_motion_auto.py` (pure): bit-identical defaults, reversal detection time at 30-240 Hz, no false
reversals on steady/slow/stationary targets with noise (3 rates x 4 speeds x 5 seeds), horizon clamp/non-finite/
smoothing, Auto vs a hand-tuned horizon, parameter mapping. `tests/test_motion_wiring.py`: `process_aiming` with
Auto, horizon following the measured age, output-side strafe handling (PID integral once per reversal, Kalman
reset, EMA bypass), config round trip/validators/old files/preset pickup, finite outputs. Chromium check of the
Aim-tab controls and a stubbed-Qt wiring check of the Aim page.

## 6. Limitations / outstanding

* **Synthetic only.** All numbers are from seeded synthetic trajectories and a simulated pipeline; no real game,
  capture device or MAKCU latency was measured. The 10 ms output delay in Auto's horizon is a constant.
* **Reversal detection latency** is ~35-50 ms by construction (needs data on both sides of the turn); the
  lead points the old way until then. Faster than ~0.3 s strafe periods gain nothing from it.
* Velocity/Anchor methods keep their old `prediction_horizon_ms` lead; only Auto uses the measured age.
* The Qt page was verified with stubbed widgets only (no PyQt6 here); check the new cards on Windows.
* Possible follow-ups: use the measured age in the Anchor method too; learn the aim->effect delay from MAKCU/
  mouse round-trip when available.
