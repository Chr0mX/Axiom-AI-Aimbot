# Changelog

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
