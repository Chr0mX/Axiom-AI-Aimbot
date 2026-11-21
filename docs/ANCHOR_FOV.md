# Anchor FOV mode

A fourth value of the FOV mode selector (`fov_shrink_mode`: Fixed / Adaptive / Dynamic / **Anchor**).
The other three only change the FOV's *size*; its centre is always the crosshair. Anchor FOV moves
the **centre** to follow an anchor point and controls the **radius** independently of it. With any
other mode selected nothing in this document runs and behaviour is unchanged (the golden test in
`tests/test_anchor_fov.py` still reproduces the original FOV output).

Code: `src/core/anchor_fov.py` (pure Python). Not to be confused with *Anchor Prediction*
(`docs/ANCHOR_PREDICTION.md`), which predicts the locked target's box to improve the Adaptive/Dynamic
shrink decisions and never moves the centre.

## Where it plugs in

`ai_loop.py` computes the FOV once per detection frame. In Anchor mode it calls
`update_anchor_fov()` instead of `compute_effective_fov()`, gets `(centre_x, centre_y, width, height)`
and passes the centre to the unchanged `filter_boxes_by_fov()`. Aiming (PID error, target priority
distances, mouse output) still uses the real crosshair; only *which detections are candidates* follows
the FOV. Lock retention is disabled in this mode (there is no shrink to retain against).

## Separation of responsibilities

| Responsibility | Code |
|---|---|
| mode selection | `anchor_fov_active(config)`; the loop's `if`; `clear_anchor_fov()` on leaving the mode |
| anchor source | `resolve_detection_anchor` / `resolve_tracker_anchor` / `resolve_manual_anchor`, filling one reused `AnchorSample` |
| anchor position | `anchor_point(box, position, off_x, off_y)` |
| movement prediction | `AnchorFovController._lead` (tracker velocity, confidence weighted) |
| position smoothing | `AnchorFovController.step` (input low-pass, exponential follow, speed cap) |
| radius | `AnchorFovController._radius_target` (+ `radius_scale` hook for a future Adaptive radius) |
| lost-anchor handling | the hold / return / search branch in `step` |
| visualisation | `publish_anchor_fov` writes the controller's real state to runtime-only config fields |

Nothing re-implements detection or tracking: the Detection source reads this frame's detections and the
selection `state.locked_box`; the Tracker source and the velocity both come from the existing
`TargetTracker` (`expected_center`, `predicted_center`, `vx/vy`, `conf_ema`, `speed_ema`).

## Anchor source and position

* **Detection** - the detection overlapping the locked target (IoU >= 0.1). With no lock, the detection
  nearest the FOV centre that intersects the FOV, so the FOV can pick a target up without aiming. A lock
  whose box is not among this frame's detections counts as lost.
* **Tracker** - the tracker's box advanced by its own short extrapolation (it lags the detections by one
  frame; `predicted_center` while temporarily lost). Smoother and survives short detection gaps; ends with
  the tracker's own lost-target timeout. Needs a lock (aim the target once).
* **Manual** - the crosshair plus the `afov_offset_x/y` pixels. Never lost.

Position: box centre / top centre / bottom centre. `afov_offset_x/y` (px) are added in every case, for every
source (for Manual they are the distance from the crosshair), so there is one offset pair and no separate
"custom" position or manual X/Y.

## Movement

Frame-rate independent, never a jump (the first frame starts exactly at the crosshair):

1. input low-pass, time constant `afov_smoothing x 0.25 s` (0 = off);
2. exponential follow, `1 - exp(-dt x afov_follow_speed)` of the remaining distance per step;
3. speed cap `afov_max_speed` px/s (0 = unlimited);
4. kept inside the detection region; a gap > 1 s or non-finite/negative `dt` moves nothing.

## Look-ahead (off by default)

`afov_predict_time_ms` (**FOV Look-ahead**, 0 = off - there is no separate switch). The lead is `tracker velocity x time x gain`. The gain follows how far the velocity can be trusted
(`conf_ema`, lock age over its first 0.25 s, zero while the tracker is temporarily lost) and fades in
(0.2 s) and out (0.1 s) rather than snapping, so a confidence dip never causes a jump. No lead below
20 px/s (detector jitter); the lead never exceeds 90 % of the radius.

## Lost anchor

After the anchor has been gone for `afov_lost_delay_ms` (brief gaps are ridden out - the FOV keeps gliding
to the last anchor):

* **Hold** - stay where it is (until an anchor shows up inside the FOV).
* **Return to Centre** - glide back to the crosshair at the normal follow speed.
* **Search** - stay and expand the radius to `afov_max_radius` so the tracking system can reacquire.

Reacquisition: **Smooth** glides to the new anchor at the normal speed; **Fast** multiplies the follow rate
and speed cap by 3 for 0.4 s (still continuous). The radius always returns to its base smoothly.

## Radius

The size is the same **FOV Width / Height** every mode uses: radius = `fov_size / 2` (a 200 px FOV has radius
100), floor 25 px, height keeps the configured `fov_height / fov_size` aspect. `afov_max_radius` (default 300) is
only headroom to grow into (Dynamic Radius, Search); it never cuts the base size. The shape
(rectangle or ellipse/circle) is still the existing *FOV Circle* setting. Optional **Dynamic Radius** widens
the FOV toward the maximum with target speed (fully wide at 3 radii/s) and low confidence.

## Fit to the target box while aiming (off by default)

`afov_fit_box_enabled` (**Fit FOV to Target Box While Aiming**) and `afov_fit_padding_pct` (**Fit Padding**,
0..200 %, default 30). While the option is on **and** aim is active (aim key / toggle / Always Aim, including
the MAKCU disengage delay) **and** a target is locked **and** the anchor source has a box (Detection or
Tracker - Manual has none, so the option is hidden for it), the FOV shrinks to the locked target's detection
box:

* half width = `box width x (1 + padding) / 2`, half height = `box height x (1 + padding) / 2`, each clamped
  between the 25 px floor and the configured FOV Width / Height - the FOV takes the box's own (usually tall)
  shape and is **never larger** than the normal FOV. The centre stays on the anchor (position and offsets are
  unchanged).
* No lock yet (aim pressed, target not selected) -> full size, so acquisition still sees every candidate.
* The anchor briefly lost while fitted -> the fitted size is held through the `afov_lost_delay_ms` grace phase
  (a rival cannot slip into a suddenly larger FOV), then the normal lost behaviour (hold / return / search)
  applies and the radius expands.
* Aim released -> the FOV glides back to the normal size with the same radius smoothing (time constant 0.2 s),
  never a jump. A fitted box also beats Dynamic Radius growth while a box exists.

Overlays and Web ESP need nothing extra: they draw the published `fov_effective_size/height`, which is the
fitted size. With the option off the output is bit-identical to before (the box shape factor stays 1 x the
configured aspect).

## Visualisation

All four surfaces (in-game Qt overlay, the UVC/NDI/UDP preview overlay, Web ESP, and the box/tracer
in-FOV tests) read `fov_center()` and the published controller state, never a separate simulation:
moving FOV, the anchor point, a dotted link to the FOV centre, the predicted anchor (prediction on), and a
amber dashed FOV with a `LOST / HOLD / RETURN / SEARCH` label while the anchor is lost.
Runtime-only config fields: `fov_center_x/y`, `afov_active/has_anchor/lost/phase`, `afov_anchor_x/y`,
`afov_pred_x/y`. Web ESP snapshot keys: `fov_center`, `afov` (both `null` in every other mode).

## Settings

Stored under `aim.fov_anchor.*` (aim-preset eligible). Inference page -> FOV Shrinking -> mode **Anchor**
reveals four groups (Anchor / Movement / Lost Anchor / Size); the same in the Web Control
Inference tab. Selecting Anchor hides the shrink-only cards (and the Reduce FOV switch).

| Setting | Default | Range |
|---|---|---|
| `afov_source` | detection | detection / tracker / manual |
| `afov_position` | center | center / top / bottom |
| `afov_offset_x/y` | 0 | -1000..1000 px |
| `afov_follow_speed` | 12 /s | 0.5..100 |
| `afov_smoothing` (FOV Anchor Smoothing) | 30 % | 0..100 % |
| `afov_max_speed` | 4000 px/s | 0..50000 (0 = unlimited) |
| `afov_predict_time_ms` (FOV Look-ahead) | 0 (off) | 0..500 ms |
| `afov_fit_box_enabled` | off | - |
| `afov_fit_padding_pct` | 30 % | 0..200 % |
| `afov_lost_behavior` | hold | hold / return / search |
| `afov_lost_delay_ms` (FOV Lost Delay) | 150 | 0..5000 |
| `afov_reacquire` | smooth | smooth / fast |
| `afov_max_radius` | 300 | 1..4096 (growth headroom only) |
| `afov_dynamic_radius` | off | - |

## How it relates to the other settings

* **Lock Persistence / Lost Target Timeout** (Aim page, Target Tracking) decide how long the *lock* survives
  missed detections. **FOV Lost Delay** only delays the FOV's *own* reaction when its anchor disappears.
* **Aim Step Smoothing** (Aim page) smooths the mouse step; **Estimator Smoothing** (Aim page, Anchor Estimator)
  smooths the Anchor velocity estimate; **FOV Anchor Smoothing** low-passes the position the FOV follows.
* **FOV Look-ahead** leads the FOV centre by the tracker velocity; **Aim Lead** (Aim page) leads the aim point
  and **FOV Prediction Horizon** (Anchor Box Prediction) leads the predicted box that feeds FOV size and
  association - three layers, three time scales.

## Limitations

* The FOV can only be where detections exist: it is kept inside the detection region, and a target that
  appears outside a *held* FOV is not picked up (use Search or Return to Centre).
* The Tracker source needs a lock, which is created by aiming; the Detection source also works
  without aiming.
* Lock retention and Anchor Prediction's FOV-influence settings apply to the shrink modes only.
* Per frame: ~12 us with 8 detections, no memory retained (`tracemalloc` over 20 000 frames).
* The Qt page, overlays and browser clients were not rendered in the development sandbox (stubbed-Qt
  wiring test and `node --check` only).

## Retired options (migrated on load)

Removed as redundant; old files, presets and scripts still load with the same behaviour:

| Retired | Why | On load |
|---|---|---|
| position `custom` | identical to `center` (offsets are added in every case) | becomes `center` |
| `afov_manual_x/y` | the same job as `afov_offset_x/y` | copied into the offsets if the source is Manual and the offsets are still 0 |
| `afov_predict_enabled` | the time alone decides (0 = off) | an old file with the switch off gets time 0; switch on without a time gets the old default 80 ms |
| `afov_predict_strength` | multiplied with `afov_predict_time_ms` into one number | folded into the time (time x strength, 0..500 ms) |
| `afov_radius`, `afov_min_radius` | the size is FOV Width / 2; the minimum only clamped the radius itself | ignored (radius follows FOV Width) |
