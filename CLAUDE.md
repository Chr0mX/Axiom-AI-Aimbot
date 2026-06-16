# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Git Workflow

- **Main is branch-protected** — never push directly to `main`. Always develop on a feature branch and open a PR via `mcp__github__create_pull_request` targeting `chr0mx/axiom-ai-aimbot`.
- Branch naming convention in use: `claude/<topic>-<id>`
- After any rebase, re-author commits: `git config user.email noreply@anthropic.com && git config user.name Claude && git rebase --exec "git commit --amend --no-edit --reset-author" origin/main`

## Running Tests

```bash
# All tests
pytest tests/

# Single test file
pytest tests/test_inference.py

# Single test
pytest tests/test_config.py::TestConfig::test_defaults
```

Tests add `src/` to `sys.path` via `tests/conftest.py` — no install step needed.

## Architecture

### Entry Point
`src/main.py` — sets DPI awareness (`SetProcessDpiAwarenessContext` Per-Monitor V2 with fallbacks), injects `src/`, `src/python/dependencies/`, and PyWin32 paths into `sys.path`, pre-loads native DLLs, initialises logging, selects ONNX backend (DirectML / CUDA / TensorRT / CPU), then starts the Qt GUI.

### Inference Pipeline (three threads)
The main loop (`src/core/ai_loop.py`, `ai_logic_loop()`) runs three concurrent workers:

1. **`_capture_worker`** — calls the active backend's `grab()` at `screenshot_interval`, writes frames into `capture_state['latest_frame']` under `frame_lock`. Also calls `set_preview_frame()` for the live preview panel. Hot-swaps capture backends every 0.5 s via `reinitialize_if_method_changed`.
2. **`_preprocess_worker`** — reads frames, letterboxes/resizes (or fast-path for square captures), builds the ONNX input tensor, pushes to `_tensor_queue`.
3. **Main inference thread** — pulls tensors, runs `model.run()` (or `run_with_iobinding()` for CUDA IO binding), calls `postprocess_outputs()` + NMS, then `process_aiming()`. Supports hot-swap of models and providers at runtime (`_try_hot_swap_model`).

Frame capture and inference are fully decoupled — capture backends must return `(H, W, 4)` uint8 BGRA.

### Screen Capture Backends (`src/core/screen_capture.py`)
Four backends share a common `grab(region) → np.ndarray | None` interface:
- **MSS** — GDI32, cross-platform fallback
- **dxcam** — Desktop Duplication API (GPU, lowest latency; returns `None` on no-change — caller reuses last valid frame)
- **UVCCapture** — OpenCV `VideoCapture` with a dedicated `_reader_thread` so `grab()` never blocks; supports probed FPS, MSMF/V4L2 backends, always-on-top pop-out preview
- **NDICapture** — cyndilib frame-sync path (UYVY → BGRA via `cv2.COLOR_YUV2BGRA_UYVY`); uses buffer protocol (`np.frombuffer`) for zero-copy frame access; auto-reconnects on source loss

The module also owns the **live preview cell**: `set_preview_frame()` / `get_preview_frame()` / `set_preview_region()` / `get_preview_region()` expose the current detection-region frame to the GUI preview panel under a single `_preview_lock`.

### Config (`src/core/config.py`)
Single `Config` class — all runtime state lives here. Persistence is driven by `_FIELD_MAP`, a module-level dict mapping every flat `Config` attribute to its dotted path in the **v2 grouped JSON schema** (e.g. `'fov_size': 'aim.fov_size'`). `to_dict()` and `from_dict()` are both generated from `_FIELD_MAP`; adding a new persisted field only requires one entry there plus the `__init__` assignment.

**Three separate persistence targets:**
- `config.json` — all fields in `_FIELD_MAP` (v2 nested schema)
- `state.json` — one-time app state in `STATE_FIELDS` (`disclaimer_agreed`, `first_run_complete`, `ndi_installer_ran_once`)
- `language.json` — current UI language selection

`config_manager.py` owns load/save lifecycle, preset management, and migration from v1 flat format.

### Aiming & Anti-Recoil (`src/core/ai_aiming.py`)
`process_aiming()` is the per-frame entry point. Key subsystems:
- **PID controller** with separate X/Y axes (`pid_kp_x/y`, `pid_ki_x/y`, `pid_kd_x/y`)
- **Y-axis recoil suppression** (`aim_y_reduce_*`) — delay, ramp, floor, settle gate, and velocity restore
- **Smart Jitter** — when a target occupies less than `smart_jitter_box_threshold_pct` of the detect range, small random or recorded movement is applied:
  - *Procedural*: random polar coords bounded by `smart_jitter_strength`
  - *Recorded pattern*: `jitter_pattern_file` points to a `jitter_patterns/*.json`; frames are cycled via `itertools.cycle` from `_jitter_pattern_cache`; the cache invalidates when the path changes
- **Humanization** — velocity-curve, Bézier smoothing, and micro-correction via `src/core/humanization.py` (`HumanizationConfig` dataclass)
- **Sticky lock** — IOU-based target persistence across frames
- **Lateral brake** — suppresses sideways over-travel
- **Deadzone** — minimum pixel gap before movement fires

### Jitter Recorder (`src/core/jitter_recorder.py`)
Standalone terminal script (run as `python src/core/jitter_recorder.py`) and importable library. Polls `win32api.GetCursorPos()` at ~1 ms. Patterns are zero-net-displacement: `_normalize_frames()` appends a correction frame `{dx: -Σdx, dy: -Σdy}` so each loop cycle returns to origin. Patterns saved as JSON to `src/core/jitter_patterns/`.

Public API consumed by the GUI:
- `list_patterns() → list[{name, path, frame_count}]` — populates the pattern combo
- `_Recorder` class — `start()` / `stop() → frames`
- `_normalize_frames(frames)`, `_save_pattern(name, frames)`

### Mouse/Device Backends (`src/win_utils/`)
All backends expose `send_mouse_move_<method>(dx, dy)` and `send_mouse_click_<method>()`. Selected at runtime from `config.mouse_move_method`. Methods:
- `sendinput` / `mouse_event` — Windows `SendInput` / `mouse_event` API
- `makcu` — MAKCU USB HID device at 4 Mbaud (`_OPERATING_BAUD`). `move()` and `click()` must never hold `self._lock` across a `time.sleep()` — lock must be released before any sleep. Scroll-wheel input excluded from aim-button detection. MAKCU click state is reported to `status_panel.py` via a dedicated callback.
- `arduino` — Arduino Leonardo HID (`arduino_mouse.py`, `arduino_spoofer.py`)
- `ddxoft` — ddxoft driver (`ddxoft_mouse.py`)
- `xbox` — Virtual Xbox 360 controller via ViGEmBus (`xbox_controller.py`, `gamepad_input.py`)

### GUI (`src/gui/fluent_app/`)
PyQt6 + PyQt6-Fluent-Widgets. `window.py` hosts a `NavigationInterface` (min 40 px, expand 150 px) with 10 pages:

| Page class | Object name | Purpose |
|---|---|---|
| `VisualsPage` | `displayInterface` | Overlay display settings |
| `ModelPage` | `modelInterface` | Model selection & notes |
| `CapturePage` | `captureInterface` | Screen capture backend |
| `InferencePage` | `inferenceInterface` | Model inference settings |
| `AimPage` | `aimInterface` | Aiming algorithm parameters, Anti-Recoil |
| `TriggerPage` | `triggerInterface` | Triggerbot / auto-fire |
| `KeysPage` | `keysInterface` | Hotkey bindings |
| `ConfigsPage` | `configInterface` | Config presets |
| `ConvertPage` | `convertInterface` | ONNX → TensorRT conversion |
| `OtherPage` | `otherInterface` | Mouse method, performance, misc |

Each page extends `BasePage` and calls `setConfig(config)` to bind to the live `Config` object. Shared widget primitives live in `fluent_app/components/slider_spin_card.py` (`SliderLabelCard`, `SliderDoubleSpinCard`).

`window.py` also mounts a **`CapturePreviewPanel`** (`fluent_app/components/capture_preview.py`) as a collapsible side panel with a `◀` arrow toggle; the panel polls `screen_capture.get_preview_frame()` and displays a live feed with FPS counter and pop-out support.

**Aim page Anti-Recoil section** contains: Smart Jitter enable toggle, LMB gate, jitter strength, box threshold, "Record Jitter" push-button (inline toggle: ● Record / ■ Stop & Save), and a "Recorded Pattern" combo that lists all `jitter_patterns/*.json` files.

### ONNX / TensorRT
Models go in `Model/`. The preprocess fast-path fires when the captured frame is already `model_input_size × model_input_size` (no resize needed). `skip_letterbox=True` skips letterboxing for square captures. TensorRT engines are built by `src/core/convert_to_engine.py` and cached in `%LOCALAPPDATA%\AxiomAI\`. `cuda_io_binding_enabled` enables zero-copy CUDA inference (CUDA provider only).

## Key Config Fields to Know

| Field | JSON path | Purpose |
|---|---|---|
| `screenshot_method` | `capture.screenshot_method` | `'mss'` / `'dxcam'` / `'uvc'` / `'ndi'` |
| `mouse_move_method` | `hardware.mouse_move_method` | `'sendinput'` / `'makcu'` / `'arduino'` / `'ddxoft'` / `'xbox'` |
| `inference_backend` | `model.backend` | `'auto'` / `'cuda'` / `'directml'` / `'tensorrt'` / `'cpu'` |
| `cuda_io_binding_enabled` | `performance.cuda_io_binding_enabled` | Zero-copy CUDA inference |
| `skip_letterbox` | `performance.skip_letterbox` | Skip letterbox for square captures |
| `makcu_disengage_delay` | `hardware.makcu.disengage_delay` | Seconds aim stays active after releasing aim button (0–20 s) |
| `makcu_aim_button` | `hardware.makcu.aim_button` | Which MAKCU button acts as the aim trigger |
| `always_aim` | `aim.always_aim` | Skip aim-key check; aim every frame |
| `keep_detecting` | `aim.keep_detecting` | Run detection even when not aiming |
| `target_priority_mode` | `aim.target_priority_mode` | `'composite'` / `'closest'` / `'confidence'` / `'size'` |
| `smart_jitter_enabled` | `aim.smart_jitter.enabled` | Enable Smart Jitter anti-recoil |
| `smart_jitter_strength` | `aim.smart_jitter.strength` | Max jitter radius (px) |
| `smart_jitter_box_threshold_pct` | `aim.smart_jitter.box_threshold_pct` | % of detect range below which jitter fires |
| `smart_jitter_lmb_gate` | `aim.smart_jitter.lmb_gate` | Only jitter while LMB held |
| `jitter_pattern_file` | `aim.smart_jitter.pattern_file` | Path to recorded `.json` pattern; empty = procedural |
| `aim_y_reduce_enabled` | `aim.y_reduce.enabled` | Enable Y-axis recoil suppression |
| `aim_y_reduce_floor` | `aim.y_reduce.floor` | Minimum Y reduction factor |
| `aim_y_reduce_ramp` | `aim.y_reduce.ramp` | Frames to ramp up suppression |
| `aim_y_reduce_settle_px` | `aim.y_reduce.settle_px` | Pixel threshold to consider recoil settled |
| `aim_y_vel_restore_px_s` | `aim.y_reduce.vel_restore_px_s` | Velocity threshold to restore Y axis |
| `sticky_lock_enabled` | `tracking.sticky_lock.enabled` | IOU-based target persistence |
| `prediction_enabled` | `tracking.prediction.enabled` | Velocity-based motion prediction |
| `uvc_show_window` | `capture.preview.enabled` | Show live capture preview panel |
| `uvc_always_on_top` | `capture.preview.always_on_top` | Preview panel always on top |
| `preview_crop_to_detection` | `capture.preview.crop_to_detection` | Crop preview to detection region |
