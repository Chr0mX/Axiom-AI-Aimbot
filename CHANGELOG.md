# Changelog

## [Unreleased]

Everything merged since the 6.3 release. `src/version.py` still says 6.3; bump it to publish.

### New Features
- **Web Control:** a LAN control panel (token-protected) that mirrors every desktop tab: model/backend switching with confirmed restart, AI start/stop, MAKCU connect, TensorRT conversion with a live log, presets and full configs, Quick Preset slots, live capture preview for UVC/NDI/UDP, settings search. Two interface styles (Studio, the default, and Instrument) chosen on the Other tab; Studio uses the full window width. Presets and configs are shown as a card list with a "Loaded" tag.
- **Presets vs. configs:** presets hold only Aim-tab settings; configs are full snapshots. Both are listed on the Configs page, with a diff preview before loading. The status panel shows the active preset/config.
- **FOV:** rectangular or elliptical FOV; "Reduce FOV on Active Target" ramps the FOV down while a target is locked.
- **Targeting:** target-class multi-select; a raw-vs-predicted aim marker; opt-in prediction refinements (capture timestamps, least-squares velocity, velocity deadzone, own-mouse compensation); Aim Start Delay.
- **Humanization:** per-feature fine-tuning sliders, Reset to Defaults, optional idle Micro-Jitter.
- **MAKCU:** V4 / MAKXD binary protocol, UDP connection, device hotkeys, physical-move lock, mouse spread, Left/Right/Middle as Always Aim buttons, and a background auto-connect retry.
- **Capture:** Capture FPS Limit; a HUD UDP Stream for weapon detection on cropped 2-PC streams; frames are only decoded or previewed when something uses them.
- **Status panel:** optional NVIDIA GPU monitoring (needs `Install NVML.bat`).
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
