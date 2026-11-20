# Anchor Prediction

An optional, additive layer that predicts where the **locked target's box** will be a
moment from now, and lets that prediction

1. nudge the **Adaptive** and **Dynamic** FOV shrink modes ahead of time, and
2. help **re-find the lock** when a fast target outruns its last box (fewer wrong
   re-locks / target switches).

The same estimator can also lead the **aim point** itself: **Anchor** is a fourth
*Prediction Method* on the Aim page (see [Aim-lead method](#aim-lead-method-prediction-method-anchor)
below), independent of the master switch.

It never replaces the existing tracker, target selection or FOV logic. With
**Enable Anchor Prediction** off (the default) nothing runs and the app behaves exactly as
before (this is enforced by a golden test against the original code).

Code: `src/core/anchor_prediction.py` (pure Python, no numpy/Qt). Hooks live in
`src/core/ai_loop_utils.py` (`anchor_on_match`, `anchor_ref_box`, `anchor_reset`,
`_fov_tightness`) and `src/core/target_tracking.py` (`lock_id`, `ref_box`).

## Where it sits in the pipeline

```
detect -> filter -> compute_effective_fov  <---- reads anchor_view (last frame's prediction)
                         |
                  process_aiming -> select_target(ref_box = predicted box)
                         |
                  tracker.on_match -> anchor_on_match -> AnchorTrack.update/predict
                                                      -> AnchorView (edge/speed/turn numbers)
```

Everything runs inline on the inference thread: pure arithmetic, no locks, no I/O, no
numpy/GPU work, so it cannot block detection and works with any ONNX/TensorRT backend.

## The estimator (`AnchorTrack`)

One track per tracked object, keyed by `TargetTracker.lock_id` (bumped for every brand-new
lock, never reused). A new id starts with an empty history, so a reassigned id can never
inherit another object's motion. `AnchorBank` holds any number of independent tracks
(`prune()` drops stale ones). The app feeds it the locked target; the bank itself supports
several simultaneous objects (tested), but nothing else is tracked by ID in this app.

Each track keeps a ring buffer (capacity 12, preallocated) of `(t, cx, cy, w, h)` and, from
the newest `anchor_history_len` samples, does a **time-based least-squares fit** (timestamps,
never a fixed FPS):

* linear fit `x(tau) = b0 + b1*tau` -> velocity (low noise);
* quadratic fit `x(tau) = b0 + b1*tau + b2*tau^2` -> acceleration `2*b2` (needs >= 4 samples);
* linear fits of width/height -> size velocity.

The linear slope is the velocity at the window's *mean* time `tau_bar`; it is moved to the
newest sample with the acceleration: `v_end = v_lin - a*tau_bar`.

**Noise gate.** Detector jitter makes the acceleration (and, less so, the speed) estimate
noisy: the error of the acceleration falls with the 4th power of the window span. So the
speed and the acceleration must each clear **2 standard errors** (ramping to full trust at
4) before they count, with the standard error derived from the track's own observed
prediction error (floor 0.5 px of jitter). Result: ±1 px of jitter on a motionless box
produces a lead of at most ~0.1 px in every mode (measured), and acceleration only
engages for brisk acceleration or a long enough history window.

## Modes

| Mode | Velocity | Acceleration | Smoothing | Strength |
|---|---|---|---|---|
| Velocity (`velocity_accel`) | raw LSQ | x `anchor_accel_comp` (gated; 0 % = velocity only) | none | fixed |
| Smoothed | EMA | EMA x `anchor_accel_comp` | time constant 20-220 ms from `anchor_velocity_smoothing` | fixed |
| Adaptive | as Smoothed | as Smoothed | as Smoothed | x (0.3 + 0.7*quality); a large innovation (sudden direction change) also drops the stale history |

Two modes used to exist and are retired as redundant (old files load unchanged, see *Retired options*):
**Linear** was Velocity with Acceleration 0 (the estimator only used it to zero the acceleration term;
the output is bit-identical, tested), and **Disabled** duplicated the master switch ("Enable Anchor
Prediction") and, for the aim-lead method, the Prediction switch and method choice.

`quality = 0.4*confidence + 0.3*stability + 0.3*direction-consistency`, where *stability* is
the overlap between the new box and the box the previous state **predicted** (so a fast but
steady target is not mistaken for an unstable one) and direction-consistency comes from an
EMA of the angle between the instantaneous and estimated heading.

## Prediction and its safeguards

`predicted box = last box + v*T + a*T^2/2` where `T` is the horizon (plus any time already
elapsed since the last sample); the size follows its own (clamped) velocity.

* **Confidence** = track age (0 -> 1 over 0.12 s) x accuracy (`1 - err/0.35`, err = EMA of the
  one-step prediction error / box size) x (0.5 + 0.5 x detection confidence). The lead is
  zero below `anchor_confidence_threshold` and fades in over a 0.25 band above it (no jump).
* **Stationary** boxes (< 0.25 box-heights/s) get no lead.
* **Deceleration** stops at standstill instead of reversing; **acceleration** may add at most
  the current speed over the horizon (a stale deceleration estimate after a hard stop would
  otherwise extrapolate backwards).
* **Max distance** `anchor_max_distance_pct` of the *configured* FOV's shorter side (the
  configured one, deliberately not the shrunk one: the prediction influences the shrink, so
  using the shrunk FOV would feed back on itself).
* **Image bounds**: the predicted box is clamped inside the capture area; size change <= +-50 %.
* **Detection loss**: a gap longer than ~2.5 normal frame intervals (min 0.1 s) coasts - velocity
  decays with a 0.15 s constant, acceleration is dropped, drift is bounded by `v*0.15 s`; longer
  than ~6 intervals (min 0.3 s) restarts the track. The thresholds scale with the track's learned
  frame interval, so a 10 FPS stream is not mistaken for missed detections.
* **ID / target changes**: a new lock id, or a measurement implying more than
  `prediction_max_velocity` (reused: same "detection jump" concept), starts a fresh track.
* NaN/inf and out-of-order timestamps are ignored.

## Interaction with the FOV systems

`fov_shrink_mode` is a single selector (Fixed / Adaptive / Dynamic), so Adaptive and Dynamic
cannot both be active; they stay independently configurable. "Both" means **Dynamic with
Dynamic Safeguards**, which already combines dynamic quality with adaptive's boundary cap, and
there each prediction influence applies to its own part.

`anchor_on_match` stores an `AnchorView` (predicted edge ratio, radial speed, current/predicted
speed, predicted turn). `FovAdjuster` turns it into three small, smoothed corrections that
`_fov_tightness()` adds to the tracker's own signals:

* **Adaptive** (`anchor_fov_influence_adaptive`): heading for the boundary (`edge_pred > 0.5`
  and `> edge_now`) adds `w * min(edge_pred - edge_now, 0.25)` to the edge ratio, so the FOV
  opens before the target is near the edge; heading inward gives a half-weight benefit; a
  prediction well inside the FOV never expands it. `w = influence x confidence`.
* **Dynamic** (`anchor_fov_influence_dynamic`): the speed signal gets `w * (speed_pred -
  speed_now + outward radial speed)` (slowing counts half) and the direction-change signal
  gets `max(0, w*turn_pred - turn_ema)` where `turn_pred` is the angle between `v` and
  `v + a*H`.
* **Dynamic + Safeguards**: the safeguard's edge cap uses the Adaptive correction.

Anti-oscillation: each correction is EMA-filtered (0.25 s), has an output dead-band, goes to 0
when the view is stale/invalid/low-confidence, and the result still passes through the FOV's
existing rate limiter (`_slew`), so the configured min/max FOV and transition speeds are
unchanged. With influence 0 (or Anchor off) the corrections are exactly 0.

**Association assist**: while the lock is matched against new detections
(`find_continuation`), a candidate's overlap is `max(IoU with the last box, IoU with the
predicted box)`. It can only add matches, never remove them, and is `None` (no change) unless
the prediction is valid.

## Aim-lead method (Prediction Method: Anchor)

Aim page -> Advanced Aim Tuning -> **Prediction Method** is Velocity / EMA / Rolling / **Anchor**.
With Anchor the aim point is led by this estimator instead of the single-purpose velocity
predictors. It is **independent of "Enable Anchor Prediction"** (that switch only governs the FOV /
lock-association layer above): picking the method is enough, and the `anchor_*` tuning is shared.

* **Own track.** `LoopState.aim_anchor` (one `AnchorTrack`) is fed the locked box shifted by the
  same camera-drift / own-mouse compensation the aim point already carries
  (`process_aiming`'s `pred_x - target_x`), so our own correction never reads as target velocity.
  Time stamps are the same as the other predictors (`prediction_capture_timestamps` when on, else
  `perf_counter`).
* **Horizon.** The aim-lead horizon (`prediction_horizon_ms`, ~10 ms, incl. Adaptive Lead) - *not*
  `anchor_horizon_ms`, which stays the FOV horizon.
* **Lead.** `predict_aim_point()` takes the predicted box's displacement; the aim point keeps its
  relative place inside the box (`aim = pcx + (aim_x - cx) * pw / w`), so a head offset follows a
  box that grows or shrinks. Kalman (if on) still runs after it; the predicted-aim marker and Web ESP
  marker read the final point unchanged.
* **Shared tuning:** `anchor_mode`, `anchor_strength`,
  `anchor_velocity_smoothing`, `anchor_accel_comp`, `anchor_confidence_threshold`,
  `anchor_history_len`, `anchor_max_distance_pct`; `prediction_max_velocity` is the jump gate.
  `anchor_horizon_ms` and the FOV influences only matter for the FOV layer. The Inference page
  shows the shared cards while Anchor Prediction is on **or** the method is Anchor.
* **Safeguards** come with the estimator: stationary dead-zone, noise-gated velocity /
  acceleration, confidence fade-in, gap coasting (a couple of missed frames keep the lead, a long
  gap restarts), restart on a teleport above `prediction_max_velocity`.
* **Acquisition guard.** For the first 24 ms of a lock the track is reset and the frame is treated as a
  fresh zero-velocity start, exactly like the other predictors. The track is also reset on genuine
  target loss (`anchor_reset`), when prediction is turned off, and when the method is switched away.
* **Not selected = no change.** Default stays `velocity`; the other methods and every FOV path are
  untouched (a stationary target gets exactly zero lead).

Which mode for the aim lead? The shared default `adaptive` smooths the velocity, which suits the FOV
but lags on rapid acceleration; for the aim lead `velocity_accel` (Velocity) responds fastest, while
`smoothed` / `adaptive` give the steadiest lead (see the benchmark).

## Settings

Stored under `tracking.anchor.*` (aim-preset eligible) and `display.show_anchor_debug`.

| Setting | Default | Range |
|---|---|---|
| `anchor_enabled` | off | - |
| `anchor_mode` | adaptive | velocity_accel (Velocity) / smoothed / adaptive |
| `anchor_horizon_ms` | 100 | 20-500 |
| `anchor_strength` | 100 % | 0-150 % |
| `anchor_velocity_smoothing` | 40 % | 0-100 % |
| `anchor_accel_comp` | 35 % | 0-100 % |
| `anchor_confidence_threshold` | 35 % | 0-100 % |
| `anchor_history_len` | 6 | 4-12 |
| `anchor_max_distance_pct` | 25 % | 0-100 % (0 = none) |
| `anchor_fov_influence_adaptive` | 50 % | 0-100 % |
| `anchor_fov_influence_dynamic` | 50 % | 0-100 % |
| `show_anchor_debug` | off | - |

Qt: Inference page -> **Anchor Prediction**; debug toggle on the Visuals page. Web Control:
Inference tab and Visuals tab. The FOV layer is separate from the classic aim-lead predictors
(`prediction_*`, ~10 ms, lead the *aim point*); they share only `prediction_max_velocity` - plus the
`anchor_*` tuning when the aim-lead method is **Anchor** (previous section).

## Debug overlay

`show_anchor_debug` draws, for the locked target: the predicted box (dashed), a dot on the current
centre, a cross on the predicted centre, a velocity arrow and `v=... px/s  conf%`, in a colour
distinct from the actual box (in-game overlay, UVC/NDI/UDP preview, Web ESP). The snapshot is only
built while the toggle is on.

## Measured behaviour (`python tools/bench_anchor.py`)

Synthetic, seeded; 1 px detector noise; 100 ms horizon; baseline = hold the last box.

| Scenario (RMSE px) | Baseline | Velocity (was Linear) | Smoothed | Adaptive |
|---|---|---|---|---|
| Stationary | 0.0 | 1.4 | 1.4 | 1.4 (measurement noise floor, no lead) |
| Slow, 60 px/s | 6.0 | 5.3 | 5.1 | 5.3 |
| Fast, 900 px/s | 91.2 | 3.1 | 1.9 | 4.6 |
| Accelerating (to 3600 px/s) | 231.8 | 12.2 | 23.0 | 34.1 |
| Sudden reversal | 50.0 | 12.5 | 16.9 | 17.2 |
| Zig-zag | 38.0 | 21.1 | 34.9 | 33.8 |

`smoothed` on the fast target stays at 1.8-2.0 px from 30 to 240 FPS. In tracking-lag terms the
fast target goes from ~100 ms behind to ~2 ms. Lead noise (trend removed) is 0.1-0.25 px in
the smoothed modes. Smoothing deliberately trades responsiveness for stability: on rapid
direction changes the smoothed modes fall back toward the baseline instead of overshooting.

FOV (Anchor off -> on): total variation, jerk and mean scale are unchanged or better in 5 of 6
scenario/mode combinations (e.g. Dynamic zig-zag: total variation 1.97 -> 1.44, jerk 0.36 ->
0.33) while the FOV opens earlier for an approaching target (Dynamic mean scale 0.49 -> 0.56).

Lock stability with two crossing targets (800 vs 300 px/s): wrong re-locks per trial
1.00 -> 0.00 at 10 and 20 FPS, 1.00 -> 0.03 at 30 FPS, 0.60 -> 0.00 at 60 FPS; holds per trial
0.48-0.83 -> 0.

Aim-lead methods at the 10 ms aim-lead horizon, 60 FPS, 1 px detector noise (RMSE px of the led aim
point against the target 10 ms later; baseline = no prediction):

| Scenario | Baseline | Velocity | EMA | Rolling | Anchor linear | Anchor adaptive |
|---|---|---|---|---|---|---|
| Stationary | 1.39 | 1.87 | 0.97 | 1.57 | 1.39 | 1.39 |
| Slow, 60 px/s | 1.46 | 1.87 | 1.30 | 1.57 | 1.50 | 1.45 |
| Fast, 900 px/s | 9.12 | 1.87 | 15.12 | 1.57 | 1.51 | 1.44 |
| Accelerating | 22.56 | 1.87 | 37.60 | 1.62 | 1.60 | 3.07 |
| Sudden reversal | 5.31 | 2.08 | 8.51 | 1.94 | 1.91 | 2.20 |
| Zig-zag | 4.19 | 1.94 | 6.65 | 1.92 | 1.89 | 3.14 |

At a 10 ms horizon every velocity-based method is limited by detector noise, so accuracy on moving
targets is on par with Velocity / Rolling (EMA lags). What Anchor adds is **no extra jitter on a
still target** (RMSE equals the baseline, where Velocity adds ~0.5 px and Rolling ~0.2 px) and the
gap / restart / confidence safeguards; `adaptive` trades some responsiveness on accelerating and
zig-zag targets for that steadiness. Cost per frame: ~11 us (`linear`) / ~15 us (`adaptive`) for the
aim-lead update itself against ~0.5 us for Velocity - about 0.2 % of one core at 144 FPS.

Cost: ~28 us added per detection frame (best of 7; the FOV+tracker step itself is ~6 us), no
memory retained per frame. That is ~0.4 % of a 144 FPS frame budget.

## Limitations

* Adaptive and Dynamic are mutually exclusive modes of one selector (see above).
* Only the locked target has a live track; the bank supports several objects but nothing else is
  tracked by ID.
* Acceleration is only used when it clears the noise floor: with the default 6-sample history
  that means roughly >= 1000-2000 px/s^2. Raise the history length for gentler accelerations
  (a longer window needs a steadier target).
* Prediction quality depends on how steady the detector's boxes are; very jittery detections
  lower the confidence and therefore the lead.
* The Qt pages and overlays are Windows-only and were not rendered in the development sandbox.

## Retired options (migrated on load)

| Retired | Why | On load |
|---|---|---|
| mode `linear` | = Velocity with Acceleration 0 | mode `velocity_accel`, `anchor_accel_comp = 0` |
| mode `disabled` | duplicates the master switch / the Prediction on-off and method choice | `anchor_enabled = False`, mode `adaptive`; an aim-lead method of Anchor becomes Velocity, so "no lead" stays "no lead" |

The Inference page now shows one FOV-influence slider: only the influence of the current shrink mode
(Adaptive or Dynamic) ever acts, so the other is hidden (both values stay in the file).
