# Changelog

- New "Tracer From Bottom Center" toggle (Visuals, Web Control): tracer lines start at the bottom-center of the screen instead of the crosshair, on the in-game overlay, capture preview and Web ESP.

- Web Control: LAN Speed Test / Test Connection ping no longer inflated by Axiom's own threading (async ping route, 1 ms timer resolution); the result shows how long Axiom itself spent per request. New "UDP stream" line: frame interval, jitter, packet spread, lost and reordered frames of the live UDP capture stream.

- Web Control: live NVIDIA GPU tiles in the sidebar (same sensors and switches as the status panel). Model Info now shows the precision actually running (FP16/FP32 · provider · RUNNING) and what is selected but still pending.

- Web ESP overlay: HUD wording matches Web Control; the top-right badge lists the enabled Visuals (Crosshair, FOV, Boxes, Confidence, Detection Range, Tracer Line, Predicted Aim Point); draws the Predicted Aim Point; sharp on high-DPI screens; less delay (event-driven sending on new detections, 1 ms timer, per-client latest-frame sending, newest-packet-only parsing). New "Web ESP Update Rate" setting (15–240 Hz, default 60) in Visuals and Web Control; the HUD shows draw FPS and data Hz separately.

- New Line Thickness sliders (1–10 px) for FOV, Detection Range, Tracer Line and Predicted Aim Point on the in-game overlay, capture preview, Web ESP and Web Control. The Web ESP tracer is now 2 px by default (was 1 px) to match the desktop overlay.

- Web Control: LAN Speed Test on the Other tab (ping/jitter, download, upload between the browser and the Axiom host).

- Web Control client: FOV shrink mode, advanced FOV, retention and tracking settings on the Inference tab. Web ESP: retention FOV boundary and track-only lock highlight.

- Opt-in lock retention: a locked target that leaves the shrunk FOV stays tracked (not aimed) inside a retention FOV instead of being dropped or stolen; class gate for lock continuation; confidence EMA now decays during misses.

- New opt-in "Dynamic Safeguards" (FOV shrink mode Dynamic): boundary-proximity cap and settle-gated speed/turn terms.

- PID time normalisation (opt-in) now uses the exact P gain `1-(1-Kp)^k`; EMA/FOV slew ignore non-finite or stale `dt`; FOV bounds validated; radius reacquisition rejects ambiguous matches.

## [Unreleased]

Everything merged since the 6.3 release. `src/version.py` still says 6.3; bump it to publish.

- MAKCU V4.103+: device-side autorelease (5 s, kept alive by a 2 s heartbeat) releases injected buttons if Axiom crashes. No effect on V3 firmware.
- MAKCU docs now point to https://makcu.com/mcp/documentation; the old k4tech Rust library docs were removed.
- Stopping inference now restores the process priority it raised to (HIGH by default); before, Axiom stayed at HIGH until it closed.
- `performance.latency_stats.enabled` now logs its per-frame timings at INFO, so they appear in the console; they were logged at DEBUG and never shown.
- FOV shrinking has three modes (Fixed, Adaptive, Dynamic) with an expansion delay and recovery duration; target tracking gained a lock state machine, time-based Lock Persistence (replaces Lock Decay Frames; old values convert), Lost Target Timeout, Reacquisition Radius, Minimum Lock Duration, Target Switch Threshold and a separate Tracking Confidence. Configs saved with FOV reduce on keep the old fixed behaviour.
- New opt-in "Frame-Rate Independent PID" (Aim → Aim Speed): scales the P and I response by elapsed time so aim speed doesn't change with the inference FPS. Off by default; identical at 60 FPS.
- The semantic filter's short deny words (`tie`, `bin`, `box`, `car`, `star`, ...) now match whole words of a class name, so names like `entities` or `cabin` are no longer dropped by accident. Longer deny words still match as substrings, and a deny word still wins over an allow word.
- Dev scripts (`bench_udp.py`, `_bench_udp_sender.py`, `obs_inspect_filters.py`) moved from `src/core/` to `tools/`; run them as `python tools/<script>.py`.

### New Features
- **Web Control:** a LAN control panel (token-protected) that mirrors every desktop tab: model/backend switching with confirmed restart, AI start/stop, MAKCU connect, TensorRT conversion with a live log, presets and full configs, Quick Preset slots, live capture preview for UVC/NDI/UDP, settings search. Two interface styles (Studio, the default, and Instrument) chosen on the Other tab; Studio uses the full window width. Presets and configs are shown as a card list with a "Loaded" tag.
- **Presets vs. configs:** presets hold only Aim-tab settings; configs are full snapshots. Both are listed on the Configs page, with a diff preview before loading. The status panel shows the active preset/config.
- **FOV:** rectangular or elliptical FOV; "Reduce FOV on Active Target" ramps the FOV down while a target is locked.
- **Targeting:** target-class multi-select; a raw-vs-predicted aim marker; opt-in prediction refinements (capture timestamps, least-squares velocity, velocity deadzone, own-mouse compensation); Aim Start Delay.
- **Humanization:** per-feature fine-tuning sliders, Reset to Defaults, optional idle Micro-Jitter.
- **MAKCU:** V4 / MAKXD binary protocol, UDP connection, device hotkeys, physical-move lock, mouse spread, Left/Right/Middle as Always Aim buttons, and a background auto-connect retry.
- **Capture:** Capture FPS Limit; a HUD UDP Stream for weapon detection on cropped 2-PC streams; frames are only decoded or previewed when something uses them.
- **Status panel:** optional NVIDIA GPU monitoring. `nvidia-ml-py` is bundled, so `Install NVML.bat` is gone; it needs an NVIDIA GPU and driver.
- **Models:** searchable model lists, a TensorRT-cached ✓/⬇ badge, automatic redirect to Convert when an engine is missing.
- **Keys page:** hotkey conflict warning.

### Changed
- **User data moved to `user/`:** config.json, state.json, language.json, model_notes.json, theme_colors.json, presets/ and configs/ are moved there automatically on first start (never overwriting).
- **Config schema v3:** new keys/inference/weapon_detection groups; the Remote Control token and other app state live in `state.json`, so `config.json` is safe to share. Older files still load.
- The desktop GUI refreshes lightly once a second without device scans, and loading settings never writes back.
- Versioned releases, CI (ruff, tests, mypy) and audit safety fixes (atomic config saves, corrupt-file quarantine, snapshot imports can't enable remote control or point at executables).

### Removed
- Anti-Recoil (Smart Jitter), the alternative movement curves, Dynamic FOV, Third Person Mask, and the Keep Detecting toggle on the Inference page.
- Unused code: hidden installer helpers on the Model and Aim pages.

---

## [6.3] – 2026-08-16

- **PID Unsafe Mode:** lets the Kp sliders go up to 1.00.
- `onnxruntime-gpu` pinned to 1.26.0 (last CUDA 12 build).
- Web ESP box scaling fixes and audit bug fixes (BUG-02/03/05/06/07), plus a MAKCU Y-axis overshoot fix.

---

## [6.2] – 2026-06-05

### New Features

#### 2D Kalman Filter (Target Tracking)
- New **Kalman Filter** aim-point smoother in the Target Tracking group
- 2D constant-velocity filter (state: position + velocity) — smoother and more predictive than EMA for fast-moving targets
- Configurable **Process Noise** and **Measurement Noise** sliders (lower = smoother / higher = more reactive)
- **Mutually exclusive with EMA** in the UI — enabling one automatically disables the other

#### Smart Jitter (Recoil Simulation)
- New **Smart Jitter** feature in the Anti-Detection group
- Fires pixel noise only when the detected bounding box is small relative to the detection range (i.e. target is far away), not on every frame
- Three strength presets: **Low (±1 px)**, **Medium (±3 px)**, **High (±6 px)**
- **LMB gate toggle** — when enabled, jitter only activates while left mouse button is physically held (uses `km.left()` on MAKCU for hardware-accurate detection)
- Configurable **box size threshold** (1–50% of detect range)

#### MAKCU LMB State Query
- `MakcuMouse.query_lmb_state()` — queries `km.left()` via the MAKCU ASCII API to detect physical button state (0=up / 1=raw physical / 2=injected / 3=both)
- Result cached for 16 ms to keep serial traffic ≤ 60 Hz
- `lmb_held` property exposes the physical state (bit 0) for use in aim logic
- *(Since removed: `query_lmb_state()`/`query_rmb_state()` duplicated the `lmb_held`/`rmb_held` properties with no external callers and were deleted in a later cleanup pass — the properties themselves remain.)*

#### MAKCU Debug Utility (`src/makcu_debug.py`)
- Standalone script for testing MAKCU serial communication
- Tests all mouse button state query methods: `km.left()`, `km.right()`, `km.middle()`, `km.catch_ml()`, `km.getpos()`, `km.mo()`, `km.silent()`
- Tests Misc API: `km.device()` (connected HID type) and `km.info()` (MAC, firmware, temperature, USB VID/PID, serial numbers, polling rates, fault flags)
- Live ~50 Hz poll loop that prints state changes while you hold/click buttons

---

### Removed / Cleaned Up

- **Auto Match FPS** toggle removed — the feature did not function as intended
- **Skip Letterbox** toggle removed — replaced by always-on letterbox preprocessing (correct aspect-ratio handling)
- **Model Input Size** UI slider removed — the underlying config field is kept for compatibility

---

### Fixes & Improvements

- Letterbox preprocessing is now always active — fixes systematic Y-axis coordinate errors when the detection region is non-square (e.g. crosshair near screen edge)
- `preprocess_image()` no longer accepts `fast_resize` parameter (removed dead code path) — *(since re-added: `fast_resize` came back as a legitimate fast-path for square-but-non-model-size captures, derived from the frame's own shape in `ai_loop.py`; see CLAUDE.md's ONNX/TensorRT section for the current preprocess behavior)*
- `ai_loop.py` screenshot interval simplified — removed conditional branch that was never correctly triggered
- All 10 language files updated with new Kalman and Smart Jitter UI strings

---

## [6.1] – Previous Release

- TensorRT FP16 acceleration support
- CUDA IO Binding zero-copy inference
- Frame skip gate (pixel-diff threshold)
- Sticky target lock with IoU matching and decay frames
- MAKCU baud rate upgrade protocol (115200 → 4 Mbaud)
- UVC / NDI capture preview thread decoupling
- EMA aim-point smoothing
- Velocity prediction with configurable horizon and history
- Target priority modes: Distance / Confidence / Composite
