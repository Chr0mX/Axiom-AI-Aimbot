# Anchor FOV mode

The fifth choice of the **FOV Mode** selector (Off / Fixed / Adaptive / Dynamic / **Anchor**; saved as
`fov_shrink_mode == 'anchor'`, see `src/core/fov_mode.py`). The shrink modes only change the FOV's *size*;
its centre is always the crosshair. Anchor FOV moves the **centre** to follow an anchor point and controls
the **radius** independently of it. With any other mode selected nothing in this document runs and
behaviour is unchanged (the golden test in `tests/test_anchor_fov.py` still reproduces the original FOV
output).

Code: `src/core/anchor_fov.py` (pure Python). Not to be confused with *Anchor Prediction*
(`docs/ANCHOR_PREDICTION.md`), which predicts the locked target's box to improve the Adaptive/Dynamic
shrink decisions and never moves the centre.

## Where it plugs in

`ai_loop.py` computes the FOV once per detection frame. In Anchor mode it calls
`update_anchor_fov(config, state, boxes, cross_x, cross_y, region, now)` instead of
`compute_effective_fov()`, gets `(centre_x, centre_y, width, height)` and passes the centre to the unchanged
`filter_boxes_by_fov()`. Aiming (PID error, target priority distances, mouse output) still uses the real
crosshair; only *which detections are candidates* follows the FOV. Lock retention is disabled in this mode
(there is no shrink to retain against). Anchor works without the shrink switch.

## Separation of responsibilities

| Responsibility | Code |
|---|---|
| mode selection | `anchor_fov_active(config)`; the loop's `if`; `clear_anchor_fov()` on leaving the mode |
| anchor source | `resolve_detection_anchor` / `resolve_tracker_anchor` / `resolve_manual_anchor`, filling one reused `AnchorSample` (`valid`, `x`, `y`) |
| anchor position | `anchor_point(box, position, off_x, off_y)` |
| position smoothing | `AnchorFovController.step` (input low-pass, exponential follow, speed cap) |
| radius | `AnchorFovController._radius_target` |
| lost-anchor handling | the hold / return / search branch in `step` |
| visualisation | `publish_anchor_fov` writes the controller's real state to runtime-only config fields |

Nothing re-implements detection or tracking: the Detection source reads this frame's detections and the
selection `state.locked_box`; the Tracker source comes from the existing `TargetTracker`
(`expected_center`, `predicted_center`).

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

1. input low-pass, time constant `SMOOTHING x 0.25 s` (`SMOOTHING` = 0.3, a fixed constant);
2. exponential follow, `1 - exp(-dt x afov_follow_speed)` of the remaining distance per step;
3. speed cap `MAX_SPEED_PX_S` (4000 px/s, a fixed constant);
4. kept inside the detection region; a gap > 1 s or non-finite/negative `dt` moves nothing.

## Lost anchor

After the anchor has been gone for `LOST_DELAY_S` (0.15 s, fixed - brief gaps are ridden out: the FOV keeps
gliding to the last anchor):

* **Hold** - stay where it is (until an anchor shows up inside the FOV).
* **Return to Centre** - glide back to the crosshair at the normal follow speed.
* **Search** - stay and expand the radius to `SEARCH_RADIUS_PX` (300 px, never below the base size) so the
  tracking system can reacquire.

A new anchor is always followed at the normal speed. The radius returns to its base smoothly.

## Radius

The size is the same **FOV Width / Height** every mode uses: radius = `fov_size / 2` (a 200 px FOV has radius
100), floor 25 px, height keeps the configured `fov_height / fov_size` aspect. The search radius is only
headroom to grow into; it never cuts the base size. The shape (rectangle or ellipse/circle) is still the
existing *FOV Circle* setting.

## Visualisation

All four surfaces (in-game Qt overlay, the UVC/NDI/UDP preview overlay, Web ESP, and the box/tracer
in-FOV tests) read `fov_center()` and the published controller state, never a separate simulation:
moving FOV, the anchor point, a dotted link to the FOV centre, and an amber dashed FOV with a
`LOST / HOLD / RETURN / SEARCH` label while the anchor is lost.
Runtime-only config fields: `fov_center_x/y`, `afov_active/has_anchor/lost/phase`, `afov_anchor_x/y`.
Web ESP snapshot keys: `fov_center`, `afov` (both `null` in every other mode).

## Settings

Stored under `aim.fov_anchor.*` (aim-preset eligible). Inference page -> FOV Mode -> **Anchor** reveals
three groups (Anchor / Movement / Lost Anchor); the same in the Web Control Inference tab. Selecting Anchor
hides the shrink-only cards.

| Setting | Default | Range |
|---|---|---|
| `afov_source` | detection | detection / tracker / manual |
| `afov_position` | center | center / top / bottom |
| `afov_offset_x/y` | 0 | -1000..1000 px |
| `afov_follow_speed` | 12 /s | 0.5..100 |
| `afov_lost_behavior` | hold | hold / return / search |

## How it relates to the other settings

* **Lock Persistence / Lost Target Timeout** (Aim page, Target Tracking) decide how long the *lock* survives
  missed detections. The fixed 0.15 s lost delay only delays the FOV's *own* reaction when its anchor
  disappears.
* **Aim Step Smoothing** (Aim page) smooths the mouse step; **Estimator Smoothing** (Aim page, Anchor Estimator)
  smooths the Anchor velocity estimate.
* **Aim Lead** (Aim page) leads the aim point and **FOV Prediction Horizon** (Anchor Box Prediction) leads the
  predicted box that feeds FOV size and association. The Anchor FOV centre itself does not look ahead.

## Limitations

* The FOV can only be where detections exist: it is kept inside the detection region, and a target that
  appears outside a *held* FOV is not picked up (use Search or Return to Centre).
* The Tracker source needs a lock, which is created by aiming; the Detection source also works
  without aiming.
* Lock retention and Anchor Prediction's FOV-influence settings apply to the shrink modes only.
* The Qt page, overlays and browser clients were not rendered in the development sandbox (stubbed-Qt
  wiring test and a Chromium check of the Web Control page only).

## Removed options

Removed as redundant or too fine-grained; old files, presets and scripts still load (the keys are ignored):

| Removed | Why | What replaced it |
|---|---|---|
| position `custom` | identical to `center` (offsets are added in every case) | migrated to `center` on load |
| `afov_manual_x/y` | the same job as `afov_offset_x/y` | copied into the offsets if the source is Manual and the offsets are still 0 |
| `afov_radius`, `afov_min_radius` | the size is FOV Width / 2 | radius follows FOV Width |
| Fit to target box (`afov_fit_box_enabled`, `afov_fit_padding_pct`) | opt-in, default off, a second shrink mechanism next to Fixed/Adaptive/Dynamic | removed |
| FOV Look-ahead (`afov_predict_time_ms`) | opt-in, default off; the aim point has its own Aim Lead | removed, with the predicted-anchor marker on every overlay |
| `afov_smoothing`, `afov_max_speed`, `afov_lost_delay_ms`, `afov_max_radius` | fine-tuning nobody needed to change | fixed constants (`SMOOTHING`, `MAX_SPEED_PX_S`, `LOST_DELAY_S`, `SEARCH_RADIUS_PX`) |
| `afov_reacquire` (smooth / fast), `afov_dynamic_radius` | opt-in, default off | removed |
