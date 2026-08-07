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
pytest tests/test_config.py::TestConfigInit::test_screen_dimensions
```

Tests add `src/` to `sys.path` via `tests/conftest.py` — no install step needed.

`tests/test_gui_invariants.py` is pure static source analysis (no PyQt6, no numpy, no display), so it runs everywhere the rest of the GUI suite can't. It enforces the three manual invariants below against a **baseline of known existing violations** rather than zero — fixing one and lowering its baseline is expected, and the test tells you the new number. Raising a baseline to make it pass defeats the point.

**On non-Windows (e.g. this sandbox)**: this is fundamentally a Windows app, so most of the suite fails on missing `win32api`/PyQt6/pywin32 — environment, not code. Compare failed/passed counts against a baseline rather than expecting a clean run.

If `numpy` is also absent, `test_inference.py`, `test_screen_capture.py` and `test_detection_semantics.py` fail at **collection**, which aborts the whole run. Skip them explicitly to get a usable signal:

```bash
pytest tests/ --ignore=tests/test_inference.py \
               --ignore=tests/test_screen_capture.py \
               --ignore=tests/test_detection_semantics.py
```

That subset's current baseline is **71 failed / 297 passed / 5 errors**. Note the consequence: the inference, capture and semantic-filter tests — including everything covering NMS, box formats and the PID — **cannot run in that environment at all**, so changes there need review or a real machine, not a green run here.

Any module with a top-level `import win32api` (or that transitively imports one, e.g. anything importing `win_utils`) fails at **collection**, not just at test-run time, which aborts the whole suite unless the import is deferred — the established pattern here is a pytest fixture (or an in-test `from ... import ...`) that does the import lazily, so a missing `win32api` fails just that file's tests individually instead of blocking every other test file's collection. `ai_loop_utils.py`, `ai_loop.py`, and `ai_aiming.py` (via `win_utils`) all hit this; `detection_semantics.py`, `udp_receiver.py`, and `esp_server.py` don't and are fully testable here.

## Architecture

### Entry Point
`src/main.py` — sets DPI awareness (`SetProcessDpiAwarenessContext` Per-Monitor V2 with fallbacks), injects `src/`, `src/python/dependencies/`, and PyWin32 paths into `sys.path`, pre-loads native DLLs, initialises logging, selects ONNX backend (DirectML / CUDA / TensorRT / CPU), then starts the Qt GUI.

**Shutdown** is wired to `app.aboutToQuit` (`main.py`'s `_shutdown()`): stops the AI threads, stops the Web ESP server, saves config. It has to be `aboutToQuit` and not code after `app.exec()` — the latter runs too late to stop anything. Without it the workers are daemon threads killed mid-step, which skips the capture worker's `finally` and therefore `_cleanup_capture()`; on the v2 path that means `capture_stop`/`capture_close` never issue `IMediaControl::Stop()` and the graph dies with the process. Each step is independently guarded so shutdown can't itself throw.

`src/version.py` (`__version__`) is the single source of truth for the app version — read by `src/core/updater.py` (GitHub release check) and the About/version label in `other_page.py`; don't hardcode version strings elsewhere.

### Inference Pipeline (three threads)
The main loop (`src/core/ai_loop.py`, `ai_logic_loop()`) runs three concurrent workers:

1. **`_capture_worker`** — calls the active backend's `grab()` at `screenshot_interval`, writes frames into `capture_state['latest_frame']` under `frame_lock`, and bumps `capture_state['frame_seq']` in the same locked section. Also calls `set_preview_frame()` for the live preview panel. Hot-swaps capture backends every 0.5 s via `reinitialize_if_method_changed`. Idles entirely while `capture_state['target_region']` is `None`, which the main loop clears whenever detection stops — otherwise it would keep grabbing frames nothing consumes.
2. **`_preprocess_worker`** — reads frames, letterboxes/resizes (or fast-path for square captures), builds the ONNX input tensor, pushes to `_tensor_queue`. Change detection uses `frame_seq`, **never `id(frame)`**: `id()` is an address CPython reuses the moment an object is freed, and every frame is an identically-shaped ndarray from the same allocator, so a genuinely new frame can land on the previous one's address and be skipped. The queue is `maxsize=1` and **drop-oldest** — a queued tensor the inference thread hasn't taken is by definition staler than the one just built, so it's discarded rather than blocking.
3. **Main inference thread** — pulls tensors, runs `model.run()` (or `run_with_iobinding()` for CUDA IO binding), calls `postprocess_outputs()` + NMS, then `process_aiming()`. Supports hot-swap of models and providers at runtime (`_try_hot_swap_model`).

`non_max_suppression()` takes **and returns** `class_ids` alongside boxes/confidences. It both drops detections and reorders survivors by descending confidence, so a separately-held class-id list stops matching boxes positionally — which silently fed the semantic filter the wrong class for each box. Keep them threaded together.

`postprocess_outputs()` interprets the model's output tensor. Two things are genuinely ambiguous from shape alone and each has an explicit config override (see the field table): `model_box_format` (cxcywh vs xyxy) and `model_has_objectness` (YOLOv5-family carries an objectness column that YOLOv8 doesn't). Both default to `'auto'`; a wrong inference corrupts every box or shifts every class id by one, silently, so pin them if `'auto'` guesses wrong.

Frame capture and inference are fully decoupled — capture backends must return `(H, W, 4)` uint8 BGRA.

**Latency stats** (`enable_latency_stats`) report `loop_overhead` and `tensor_wait`, not "capture" and "preprocess" — those run on their own threads and are not timed on the inference thread at all. `infer`/`postproc` are genuine.

### Screen Capture Backends (`src/core/screen_capture.py`)
Five backends share a common `grab(region) → np.ndarray | None` interface:
- **MSS** — GDI32, cross-platform fallback
- **dxcam** — Desktop Duplication API (GPU, lowest latency; returns `None` on no-change — caller reuses last valid frame)
- **UVCCapture** — OpenCV `VideoCapture` with a dedicated `_reader_thread` so `grab()` never blocks; supports probed FPS, MSMF/V4L2 backends, always-on-top pop-out preview. `uvc_capture_method == 'dshow'` additionally selects between two DirectShow implementations via `uvc_dshow_backend`: `'v1'` (default) is `cv2.VideoCapture(CAP_DSHOW)`; `'v2'` is the native `directshow_capture.dll` (`chr0mx/DirectShow-Capture-DLL`, vendored at `src/python/dependencies/`) via `src/core/dshow_capture_native.py`'s ctypes binding (`NativeDshowCapture`) — owns the capture graph's allocator/buffer count directly instead of going through OpenCV, for lower/OBS-like latency. Both MJPEG and NV12 are supported under v2 (`uvc_video_format`, restricted to those two in the GUI while v2 is active — `yuy2`/`yuv420p` have no v2 equivalent); MJPEG is decoded to BGR in `_reader_worker_native_dll`, NV12 reuses the same raw-NV12 crop-before-convert path (`_crop_nv12`) the cv2 path's `CAP_PROP_CONVERT_RGB` optimization already uses. `uvc_ffmpeg_enabled` (external `ffmpeg.exe` subprocess) is gated to `dshow` + `v1` only — ffmpeg has no MSMF demuxer on Windows, and v2 already owns the graph directly. `uvc_crop_mode == 'fixed'` (vs. the default `'dynamic'`, which recomputes the crop from the live Detection Range every frame) freezes a centered `detect_range_size` square as the capture region; on v2/NV12 this is requested as a genuine native crop via the DLL's `dsc_open_params.crop_x/y/width/height` (`_compute_fixed_uvc_crop_region()`, shared with the cv2 path) — **not** by asking the device to negotiate a smaller resolution, which carries no guarantee of being a spatial crop of the same framing rather than the whole scene rescaled. Device negotiation always requests the full `uvc_width`/`uvc_height` regardless of crop mode; the crop is a separate, later, in-DLL step. Because a DLL build predating this crop ABI addition silently ignores the trailing crop fields instead of rejecting them, `_init_native_dll` treats `capture_open()` succeeding as provisional, not proof — `_reader_worker_native_dll` confirms the crop actually took effect against the first real frame's delivered dimensions and self-heals to a software crop (`_fixed_region`) if the DLL didn't honor it. Native crop is NV12-only (MJPEG would need decode+crop+re-encode, which the DLL doesn't implement); MJPEG's fixed-crop mode always uses the software path. A **Detection Range change does not reinitialize the device**: `reinitialize_if_method_changed` first calls `UVCCapture.try_live_reconfigure()`, which moves the rectangle via `capture_set_crop()` when native crop is genuinely in force and returns False (falling through to a full reinit) for every other config delta, for the software-crop paths, when the first-frame check already found the DLL ignoring the crop, and on `set_crop` failure. Without it every tick of a slider drag rebuilt the whole DirectShow graph. `uvc_actual_width/height/fps` come from the DLL's `capture_get_negotiated_format()` where available — genuinely negotiated values, not the request echoed back (which is what they were, and what they fall back to with a warning against a DLL that predates the call).
- **NDICapture** — cyndilib frame-sync path (UYVY → BGRA via `cv2.COLOR_YUV2BGRA_UYVY`); uses buffer protocol (`np.frombuffer`) for zero-copy frame access; auto-reconnects on source loss
- **UdpCapture** — receives an MJPEG-over-UDP stream matching OBS's `udp_stream_filter` plugin wire format (`src/core/udp_receiver.py`: 14-byte chunk header, frame reassembly, stale-frame timeout); decodes JPEG chunks into BGRA frames. The reader **blocks on the receiver's `_new_frame_event`** rather than polling — an earlier spin on `time.sleep(0)` pinned a full CPU core on the theory that `Event.wait()` costs a scheduler quantum, which is true of timed sleeps but not of condition-variable signalling. Frames are published only if newer than the last (signed 32-bit `frame_id` delta, so a restarted sender resetting to 0 still reads as forward); a late-completing older frame would otherwise rewind the consumer, and is counted in `stale_frames_dropped` instead. `udp_recv_buffer_size` is clamped to ≥ 65536 by `_validate_udp_recv_buffer_size` — below the sender's 14 + 60000 ceiling, `recvfrom` **silently truncates** datagrams and the short payload still passes the chunk-count completeness check, so corruption reaches `cv2.imdecode` with nothing pointing back at the setting.

The module also owns the **live preview cell**: `set_preview_frame()` / `get_preview_frame()` / `set_preview_region()` / `get_preview_region()` expose the current detection-region frame to the GUI preview panel under a single `_preview_lock`.

### Config (`src/core/config.py`)
Single `Config` class — all runtime state lives here. Persistence is driven by `_FIELD_MAP`, a module-level dict mapping every flat `Config` attribute to its dotted path in the **v2 grouped JSON schema** (e.g. `'fov_size': 'aim.fov_size'`). `to_dict()` and `from_dict()` are both generated from `_FIELD_MAP`; adding a new persisted field only requires one entry there plus the `__init__` assignment.

**Three separate persistence targets:**
- `config.json` — all fields in `_FIELD_MAP` (v2 nested schema)
- `state.json` — one-time app state in `STATE_FIELDS` (`disclaimer_agreed`, `first_run_complete`, `ndi_installer_ran_once`)
- `language.json` — current UI language selection

`config_manager.py` owns load/save lifecycle, preset management, and migration from v1 flat format. Presets are full config snapshots saved as JSON to `config/*.json` (derived from `_FIELD_MAP`, so they can't drift from `Config`'s current field set); `src/core/presets/*.json` holds bundled built-in presets that get seeded into the user's `config/` dir on first run (never overwriting a same-named file the user already has). Preset names are sanitized (`_sanitize_config_name()`) before being interpolated into a file path — they can come from free-text GUI dialogs or an imported file's own `name` field, both untrusted.

### Aiming & Anti-Recoil (`src/core/ai_aiming.py`)
`process_aiming()` is the per-frame entry point, called from `ai_loop.py` with the **full** FOV-filtered candidate list (never pre-reduced — see single-target note below). Key subsystems:
- **Target selection** — `target_priority_mode` (`'distance'` / `'confidence'` / `'composite'`, weighted by `target_priority_confidence_weight`) scores candidates; shared logic lives in `find_closest_target()` (`ai_loop_utils.py`), used both by `process_aiming()` and the idle-detect path in `ai_loop.py`
- **Sticky lock** — IOU-based target persistence across short detection gaps (`sticky_lock_enabled`), with an adaptive IOU threshold that scales with FOV size (`_adaptive_sticky_iou`). The "is this the same target?" continuity check that gates the Y-velocity restore uses **the same threshold the lock just applied** (`_continuity_thresh`), not the raw `lock_iou_threshold` — otherwise a target the lock deliberately kept at the looser adaptive threshold could still be judged "changed", resetting the gate mid-track on exactly the small/far targets adaptivity exists to hold.
- **`state.locked_box` / `state.locked_confidence`** (`ai_loop_state.py`) — written on *every* successful selection regardless of `sticky_lock_enabled`, so other code (e.g. `single_target_mode`'s box-list reduction) can read back the real post-lock pick instead of re-deriving a lock-blind one
- **`single_target_mode`** — reduces the box list Web ESP/auto-fire/preview see to just the locked target. This reduction happens **after** `process_aiming()` runs, derived from `state.locked_box`; it must never pre-filter the candidate list before sticky lock sees it, or sticky lock can never win over the frame's raw priority winner
- **Motion prediction** (`prediction_enabled`) — `src/core/target_predictor.py` (`VelocityPredictor`) or `src/core/kalman_filter.py` (`KalmanFilter2D`, constant-velocity 2D); both `.reset()` on true target loss to avoid stale velocity carrying into a new target, and both expose `reconfigure()` for live setting changes (don't poke their private attributes — an earlier caller rebuilt the predictor's deque every single frame to pick up settings that almost never change). `KalmanFilter2D.update()` takes `dt` and patches its state-transition matrix accordingly; it previously baked `dt=1.0` into `F` and never updated it, modelling every tick as the same duration regardless of loop rate.
- **Semantic false-positive filter** (`detect_semantic_filter_enabled`) — `src/core/detection_semantics.py`; three layers (ONNX class-name allow/deny lists, aspect-ratio geometry heuristics, min bbox size) to drop vegetation/vehicle/sign/HUD misdetections before target selection
- **PID controller** with separate X/Y axes (`pid_kp_x/y`, `pid_ki_x/y`, `pid_kd_x/y`). `update()` takes a real `dt`, normalized against `PIDController.REFERENCE_DT` (0.01 s) rather than raw seconds — at that rate the scale is exactly 1.0, so **existing tunings behave identically and no config migration was needed**, while the rate dependence is cancelled everywhere else. That mattered at runtime, not just across configs: `idle_detect_interval` slows the loop while not aiming, so releasing the aim key used to change the response. Ratio clamped to `[0.25, 4.0]` so one stalled frame can't kick the output; `state.pid_last_update_t` is cleared alongside `pid.reset()` on target loss. Don't reintroduce a raw-`dt` scaling — it would make Ki ~100× weaker and Kd ~100× stronger for every saved config.
- **Y-axis recoil suppression** (`aim_y_reduce_*`) — delay, ramp, floor, settle gate, and velocity restore
- **Smart Jitter** — when a target occupies less than `smart_jitter_box_threshold_pct` of the detect range, small random or recorded movement is applied:
  - *Procedural*: random polar coords bounded by `smart_jitter_strength`
  - *Recorded pattern*: `jitter_pattern_file` points to a `jitter_patterns/*.json`; frames are cycled via `itertools.cycle` from `_jitter_pattern_cache`; the cache invalidates when the path changes. Playback carries the truncation fraction in `state.jitter_carry_x/y` — patterns are normalized to zero net displacement by `_normalize_frames`, but `sum(int(x)) != int(sum(x))`, so truncating each frame independently accumulated residue and walked the crosshair off-target over a long burst
- **Humanization** — velocity-curve, Bézier smoothing, and micro-correction via `src/core/humanization.py` (`HumanizationConfig` dataclass)
- **Lateral brake** — suppresses sideways over-travel
- **Deadzone** — minimum pixel gap before movement fires
- **Sub-pixel carry** (`state.aim_carry_x/y`) — accumulates the fractional remainder integer truncation discards each frame, applied universally across all mouse backends so micro-corrections are never silently lost

Auto-fire (`src/core/auto_fire.py`) consumes `config.latest_boxes` (the single-target-reduced list when applicable) via `update_queues(...)`.

### Secondary / Weapon Detection (process-isolated)
Both run in a separate child process so the main inference loop and Qt UI are never blocked, feeding results back via a small "feeder" thread. Selected via `second_inference_mode` (`'off'` / `'v1_ocr'` / `'v2_onnx'`) — the Inference page shows/hides the ONNX confidence threshold accordingly.
- **`src/core/ocr_inference.py`** — PaddleOCR-based weapon-name reader
- **`src/core/hud_inference.py`** — YOLO11n ONNX weapon/attachment detector. ROI read from `config.hud_roi_coords` (`"x1,y1,x2,y2"`, default `"1490,953,1870,1041"`, Apex Legends HUD strip @ 1080p), 320×320 letterboxed input. Public API: `start(config)`, `stop()`, `trigger_hud_scan()`, `get_hud_results()`, `get_hud_roi_image()`

### Web ESP Overlay (`src/core/esp_server.py` + `src/web_overlay/`)
Optional LAN-accessible browser overlay (`web_esp_enabled`), modeled as "backend streams state, frontend draws it" — the PyQt in-game overlay stays the on-device renderer. Three daemon threads:
- **HTTP server** (stdlib `ThreadingHTTPServer`) serves the static client from `src/web_overlay/` (`index.html` / `app.js` / `styles.css`); bare `/` requests without a `?ws=` query 302-redirect to the actual bound WS port so the client never needs a manually-appended port
- **WebSocket server** — hand-rolled RFC 6455 (handshake + text-frame encoding, no external dependency), auto-increments the bind port on conflict (`_actual_ws_port`)
- **Broadcast loop** — ~`web_esp_fps` Hz (re-read every tick, so the setting is live), latest-state-wins; serializes a JSON snapshot of `Config` (`_build_snapshot()`) to all connected clients, including 1 Hz-computed `capture_fps`/`inference_fps`. Client sockets carry a `_BROADCAST_SEND_TIMEOUT` (2 s) and are dropped on timeout — with a blocking `sendall` and no deadline, one client that stops draining its TCP window (an ordinary state for a backgrounded browser tab) froze the feed for **every** client. There is no auth: this is a deliberate LAN-convenience posture, not an oversight.

Reads `config.latest_all_boxes` / `latest_all_confidences` (unreduced by `single_target_mode` — same set the in-game overlay draws), not `config.latest_boxes`, so the web view always shows every detection regardless of aiming mode.

### In-Game Overlay & Status Panel (`src/gui/`)
The actual on-device renderer referenced above — plain PyQt widgets, not part of `fluent_app/`:
- **`overlay.py`** — transparent, click-through `QWidget` drawing FOV/detect-range circles, detection boxes, confidence text, tracer lines, and the aim-point X marker directly over the game window. Colors go through the `OverlayColors` static-method wrapper (`get_fov_color()`, `get_box_color()`, `get_tracer_color()`, `get_aim_marker_color()`, etc.), each backed by a `ThemeColors.OVERLAY_*` entry — never hardcode a color here directly, add a new `ThemeColors` entry and getter instead. `box_color_theme` can override the detect-box color with a named preset instead of the theme default.
- **`status_panel.py`** — the on-screen status HUD (aim state, FPS, model name, MAKCU/Xbox connection). Supports Windows Acrylic blur via the legacy `SetWindowCompositionAttribute` API; re-reads language strings on its own refresh timer rather than needing a `retranslateUi()` call.

### Jitter Recorder (`src/core/jitter_recorder.py`)
Standalone terminal script (run as `python src/core/jitter_recorder.py`) and importable library. Polls `win32api.GetCursorPos()` at ~1 ms. Patterns are zero-net-displacement: `_normalize_frames()` appends a correction frame `{dx: -Σdx, dy: -Σdy}` so each loop cycle returns to origin. Patterns saved as JSON to `src/core/jitter_patterns/`.

Public API consumed by the GUI:
- `list_patterns() → list[{name, path, frame_count}]` — populates the pattern combo
- `_Recorder` class — `start()` / `stop() → frames`
- `_normalize_frames(frames)`, `_save_pattern(name, frames)`

### Mouse/Device Backends (`src/win_utils/`)
All backends expose `send_mouse_move_<method>(dx, dy)` and `send_mouse_click_<method>()`. Selected at runtime from `config.mouse_move_method`. Methods:
- `sendinput` / `mouse_event` — Windows `SendInput` / `mouse_event` API
- `makcu` — MAKCU USB HID device at 4 Mbaud (`_OPERATING_BAUD`), ASCII protocol; see `docs/MAKCU_Native_API.md` for the wire protocol reference. `move()` and `click()` must never hold `self._lock` across a `time.sleep()` — lock must be released before any sleep (enforced by `tests/test_gui_invariants.py`). `move()` **coalesces** into a running `_pending_dx/_pending_dy` total that the write thread drains and zeroes atomically; it does *not* queue or drop commands. These are **relative** deltas, so a dropped one is travel permanently lost rather than delayed — and `process_aiming`'s sub-pixel carry assumes every move it hands over is delivered, so it cannot compensate. The accumulator is cleared on connect, since deltas banked during a USB glitch would otherwise replay as one jump long after the target moved on. Scroll-wheel input excluded from aim-button detection (masked to bits 0-4, `_BTN_BITS`). Self-healing: a background reconnect-watchdog thread retries the connection automatically on USB glitches. MAKCU click state is reported to `status_panel.py` via a dedicated callback. `makcu_mouse_binary.py` (the V2 binary-protocol variant) is unused in production today — kept in place for a future MAKCU firmware version; don't wire it in without first fixing the lock-across-sleep violation in its `connect()`/`_send_cmd()`.
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
| `KeysPage` | `keysInterface` | Hotkey bindings + MAKCU device connection (nav label: "Keys & HW") |
| `ConfigsPage` | `configInterface` | Config presets |
| `ConvertPage` | `convertInterface` | ONNX → TensorRT conversion |
| `OtherPage` | `otherInterface` | Mouse method, performance, misc |

Each page extends `BasePage` and calls `setConfig(config)` to bind to the live `Config` object. Shared widget primitives live in `fluent_app/components/slider_spin_card.py` (`SliderLabelCard`, `SliderDoubleSpinCard`).

`window.py` also mounts a **`CapturePreviewPanel`** (`fluent_app/components/capture_preview.py`) as a collapsible side panel with a `◀` arrow toggle; the panel polls `screen_capture.get_preview_frame()` and displays a live feed with FPS counter and pop-out support.

**Aim page Anti-Recoil section** contains: Smart Jitter enable toggle, LMB gate, jitter strength, box threshold, "Record Jitter" push-button (inline toggle: ● Record / ■ Stop & Save), and a "Recorded Pattern" combo that lists all `jitter_patterns/*.json` files.

**Theming** (`fluent_app/theme_colors.py`) — a single `ThemeColors` class of `ColorPair`/`ColorPairWithAlpha` descriptors (`.get()` returns a CSS color string, `.qcolor()` a `QColor`), each with light/dark defaults and an optional override from a user `theme_colors.json`. `isDarkTheme()` (qfluentwidgets) picks which side of the pair applies. Any `setStyleSheet(...)` call with a literal hex/rgb color instead of a `ThemeColors.*.get()` reference is a theming bug — it won't adapt to a light/dark toggle. (`theme_manager.py`, a separate unused QSS-generator module, was deleted as dead code — it was never invoked anywhere.)

**First-run flow**: `fluent_app/setup_wizard.py` (Welcome → Language → Theme → Acrylic → Performance → Done, gated on `state.json`'s `first_run_complete`) followed by `gui/disclaimer_dialog.py` (gated on `disclaimer_agreed`; body text loaded from a single English-only `Disclaimer.md`, deliberately not localized — translating just the button chrome around a permanently-English legal document would be a confusing half-measure). The wizard tracks its own in-progress theme preview (`self._isDark`) separately from the app's real theme — `setTheme()` isn't actually called until the wizard closes (`applyChosenTheme()`, invoked from `main.py`), so any wizard-page widget that needs to look right in both previews must read the wizard's own state, not `isDarkTheme()`.

**i18n**: `t(key, default)` (`fluent_app/language_manager.py`) looks up `key` in the active language's JSON (`src/core/language_data/*.json`, 10 languages) and falls back to `default` (always English) if the key is missing — a missing key never crashes or shows blank, just shows English regardless of the selected language. Widgets built with a `t()`-wrapped title/description at construction must have the *same* call repeated inside that page's `retranslateUi()` (called from `window.py`'s `_refreshUI()` on every language switch) or the text goes stale after the first switch — this has been a recurring bug source; when adding a new card/label, grep an existing correct one in the same file for the exact `titleLabel`/`contentLabel` API before wiring a new one in. Translation completeness varies — English has the most keys (~290); other languages are missing anywhere from ~20 to ~75 of them (pre-existing, not something a single pass has closed).

### Native DLL ABI compatibility (`directshow_capture.dll`)

The DLL is vendored as a prebuilt binary at `src/python/dependencies/directshow_capture.dll` and bound via ctypes in `dshow_capture_native.py`. Two rules keep a version skew from turning into a crash:

- **`CaptureParams` must match `dsc_open_params` byte-for-byte.** It's a raw memory-layout match, not name-based. New fields go on the **end** only — a DLL that predates them never reads that far, so they're silently ignored rather than rejected. That silence is why `capture_open()` returning OK is not proof a new field took effect; `_init_native_dll` verifies the crop against the first real frame's dimensions instead of trusting the return code.
- **New exports must be bound behind `hasattr`.** `_load_dll` resolves symbols eagerly, so an unguarded `dll.new_function.argtypes = ...` raises `AttributeError` against an older DLL and breaks loading entirely. `capture_get_negotiated_format` is the worked example: guarded at bind time and again at call time, with `negotiated_format()` returning `None` so callers degrade instead of failing.

Current split — required vs. optional:

| Export | Status |
|---|---|
| `capture_open` / `start` / `stop` / `close` / `get_latest_frame` / `default_params` / `get_device_count` / `get_device_name` / `get_qpc_frequency` / `result_to_string` / `get_last_error` | Required |
| `capture_set_crop` | Required for `uvc_crop_mode == 'fixed'` on v2/NV12 |
| `capture_get_negotiated_format` | Optional — absent ⇒ `uvc_actual_*` falls back to requested values, with a warning |
| `capture_get_short_sample_count` | Optional — DLL-side diagnostic, not bound in Axiom |

**The vendored binary lags the DLL source.** It is rebuilt and re-committed by hand, so C++ fixes in `chr0mx/DirectShow-Capture-DLL` have no effect here until someone rebuilds (`src/build.bat`) and replaces this file. Axiom runs correctly against an older build — that's what the two rules above buy — it just doesn't get the fixes.

### ONNX / TensorRT
Models go in `Model/` (`Model_Hud/` for the secondary weapon-detector model). The preprocess fast-path is automatic and backend-agnostic, driven purely by the captured frame's own shape (not a config flag): an exact `model_input_size × model_input_size` frame skips resize/letterbox entirely (`preprocess_image()`, `inference.py`); a square-but-different-size frame gets a plain nearest-neighbor resize with no padding (`fast_resize`, derived in `ai_loop.py` from `frame.shape[0] == frame.shape[1]`); only a non-square frame gets a real letterbox with padding. TensorRT engines are built by `src/core/convert_to_engine.py` and cached in `trt_cache/` at the project root (`session_utils.py`'s `_TRT_CACHE_DIR`) — **not** `%LOCALAPPDATA%\AxiomAI\`, which is only where the TensorRT *pip package itself* gets installed by `Install TensorRT.bat`, a separate concern. `cuda_io_binding_enabled` enables zero-copy CUDA inference (CUDA provider only).

## Key Config Fields to Know

| Field | JSON path | Purpose |
|---|---|---|
| `screenshot_method` | `capture.screenshot_method` | `'mss'` / `'dxcam'` / `'uvc'` / `'ndi'` / `'udp'` |
| `udp_bind_ip` / `udp_bind_port` | `capture.udp.bind_ip` / `.bind_port` | Listen address for the OBS `udp_stream_filter` MJPEG stream |
| `mouse_move_method` | `hardware.mouse_move_method` | `'sendinput'` / `'makcu'` / `'arduino'` / `'ddxoft'` / `'xbox'` |
| `inference_backend` | `model.backend` | `'auto'` / `'cuda'` / `'directml'` / `'tensorrt'` / `'cpu'` |
| `cuda_io_binding_enabled` | `performance.cuda_io_binding_enabled` | Zero-copy CUDA inference |
| `makcu_disengage_delay` | `hardware.makcu.disengage_delay` | Seconds aim stays active after releasing aim button (0–20 s) |
| `makcu_aim_button` | `hardware.makcu.aim_button` | Which MAKCU button acts as the aim trigger |
| `always_aim` | `aim.always_aim` | Skip aim-key check; aim every frame |
| `keep_detecting` | `aim.keep_detecting` | Run detection even when not aiming |
| `single_target_mode` | `aim.single_target_mode` | Reduce Web ESP/auto-fire/preview box list to the locked target only (applied *after* sticky lock resolves the pick — see Aiming section) |
| `target_priority_mode` | `tracking.target_priority.mode` | `'distance'` / `'confidence'` / `'composite'` |
| `target_priority_confidence_weight` | `tracking.target_priority.confidence_weight` | Weight of confidence vs. distance in `'composite'` mode |
| `nms_iou_threshold` | `model.nms_iou_threshold` | IoU above which NMS suppresses the lower-confidence of two overlapping boxes (was hardcoded 0.4) |
| `model_box_format` | `model.box_format` | `'auto'` / `'cxcywh'` / `'xyxy'` — ONNX box encoding. `'auto'` infers, but only with ≥3 detections (the x2>x1 test is near a coin flip below that) and logs when it does |
| `model_has_objectness` | `model.has_objectness` | `'auto'` / `'yes'` (YOLOv5-family) / `'no'` (YOLOv8-family). Getting this wrong makes confidence `max(obj,cls)` instead of `obj*cls` and shifts every class id by one |
| `detect_semantic_filter_enabled` | `aim.detect_semantic_filter_enabled` | Filter vegetation/vehicle/sign/HUD false positives before target selection |
| `second_inference_mode` | `ocr.mode` | `'off'` / `'v1_ocr'` / `'v2_onnx'` — secondary weapon-detection method |
| `hud_roi_coords` | `ocr.hud_roi_coords` | ROI `"x1,y1,x2,y2"` for the ONNX weapon/attachment HUD scanner |
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
| `prediction_enabled` | `tracking.prediction.enabled` | Velocity/Kalman-based motion prediction |
| `uvc_crop_mode` | `capture.uvc.crop_mode` | `'dynamic'` (recompute crop from live Detection Range every frame) / `'fixed'` (freeze a centered `detect_range_size` square — native DLL crop on v2/NV12, software crop otherwise) |
| `uvc_show_window` | `capture.preview.enabled` | Show live capture preview panel |
| `uvc_always_on_top` | `capture.preview.always_on_top` | Preview panel always on top (applies to the PyQt `PreviewPopOutWindow`) |
| `preview_crop_to_detection` | `capture.preview.crop_to_detection` | Crop preview to detection region |
| `web_esp_enabled` | `web_esp.enabled` | Enable the LAN browser overlay server |
| `web_esp_http_port` / `web_esp_ws_port` | `web_esp.http_port` / `.ws_port` | Static-page and WebSocket ports (WS auto-increments on bind conflict) |
| `web_esp_fps` | `web_esp.fps` | Broadcast tick rate for the Web ESP state snapshot (re-read live) |
| `udp_recv_buffer_size` | `capture.udp.recv_buffer_size` | `recvfrom` buffer; clamped to ≥ 65536 — smaller silently truncates datagrams into corrupt JPEGs |
