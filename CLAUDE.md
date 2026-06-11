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
`src/main.py` — sets DPI awareness, injects `src/` and `src/python/dependencies/` into `sys.path`, pre-loads DLLs, initialises logging, selects ONNX backend (DirectML / CUDA / TensorRT / CPU), then starts the Qt GUI.

### Inference Pipeline (three threads)
The main loop (`src/core/ai_loop.py`, `ai_logic_loop()`) runs three concurrent workers:

1. **`_capture_worker`** — calls the active backend's `grab()` at `screenshot_interval`, writes frames into `capture_state['latest_frame']` under `frame_lock`.
2. **`_preprocess_worker`** — reads frames, letterboxes/resizes, builds the ONNX input tensor, pushes to `_tensor_queue`.
3. **Main inference thread** — pulls tensors, runs `model.run()` (or `run_with_iobinding()`), calls `postprocess_outputs()` + NMS, then `process_aiming()`.

Frame capture and inference are fully decoupled — capture backends must return `(H, W, 4)` uint8 BGRA.

### Screen Capture Backends (`src/core/screen_capture.py`)
Four backends share a common `grab(region) → np.ndarray | None` interface:
- **MSS** — GDI32, cross-platform fallback
- **dxcam** — Desktop Duplication API (GPU, lowest latency; returns `None` on no-change — caller reuses last valid frame)
- **UVCCapture** — OpenCV `VideoCapture` with a dedicated `_reader_thread` so `grab()` never blocks
- **NDICapture** — cyndilib frame-sync path (UYVY_RGBA format → `cv2.COLOR_YUV2BGRA_UYVY`); uses buffer protocol (`np.frombuffer`) for zero-copy frame access

Hot-swap between backends happens every 0.5 s in the capture worker (`reinitialize_if_method_changed`).

### Config (`src/core/config.py`)
Single `Config` class — all runtime state lives here. Persisted via `to_dict()` / `from_dict()`. No dataclass decorators; fields are plain `__init__` assignments with type hints. New fields need entries in both `__init__` and `to_dict()`.

### Mouse/Device Backends (`src/win_utils/`)
All backends expose `send_mouse_move_<method>(dx, dy)` and `send_mouse_click_<method>()`. Selected at runtime from `config.mouse_move_method`. MAKCU uses a serial connection at 4 Mbaud (`_OPERATING_BAUD`); `move()` and `click()` must never hold `self._lock` across a `time.sleep()` — the lock must be released before any sleep to avoid blocking the inference thread.

### GUI (`src/gui/fluent_app/`)
PyQt6 + PyQt6-Fluent-Widgets. `window.py` hosts a `NavigationInterface` with pages in `pages/`. Each page extends `BasePage` and calls `setConfig(config)` to bind to the live `Config` object. Shared widget primitives (sliders, spinboxes) live in `fluent_app/components/slider_spin_card.py` (`SliderLabelCard`, `SliderDoubleSpinCard`).

### ONNX / TensorRT
Models go in `Model/`. The preprocess fast-path fires when the captured frame is already `model_input_size × model_input_size` (no resize needed). `fast_resize=True` skips letterboxing for square captures. TensorRT engines are built by `src/core/convert_to_engine.py` and cached in `%LOCALAPPDATA%\AxiomAI\`.

## Key Config Fields to Know

| Field | Purpose |
|---|---|
| `screenshot_method` | `'mss'` / `'dxcam'` / `'uvc'` / `'ndi'` |
| `mouse_move_method` | `'sendinput'` / `'makcu'` / `'arduino'` / `'ddxoft'` / `'xbox'` |
| `cuda_io_binding_enabled` | Zero-copy CUDA inference (CUDA provider only) |
| `makcu_disengage_delay` | Seconds aim stays active after releasing the aim button (0–20 s) |
| `always_aim` | Skip aim-key check; aim every frame |
| `keep_detecting` | Run detection even when not aiming |
