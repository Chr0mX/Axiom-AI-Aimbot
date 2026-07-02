<div align="center">

![Version](https://img.shields.io/badge/Version-6.3-green.svg)
![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![License](https://img.shields.io/badge/License-PolyForm--Noncommercial%201.0.0-blueviolet.svg)
![GitHub last commit](https://img.shields.io/github/last-commit/Chr0mX/Axiom-AI-Aimbot)
![Repo Size](https://img.shields.io/github/repo-size/Chr0mX/Axiom-AI-Aimbot)

<h1>Axiom AI</h1>
<p>Adaptive aim assistance powered by computer vision to support gamers who need it most.</p>

## <a href="https://discord.gg/DpcqaQEj5b">Discord (Support)</a>

<p>
  <img src="./guidemo.gif" alt="Demo GIF" width="720">
</p>

<p><strong>If this project helps you, please give us a ⭐ Star!</strong></p>

</div>

## About this fork

The original Axiom AI Aimbot project (by iisHong0w0) is no longer maintained/available. This repository is a **new, "vibe coded" continuation** — rebuilt and extended largely through AI-assisted development on top of the last available codebase. Expect faster, more experimental iteration than a traditional hand-maintained project; things are actively being reworked, and some corners (see **Beta** below) are genuinely unfinished.

If you're looking for the original project, it's gone — this is what's actively developed now.

## Overview

Axiom AI is a computer vision application for real-time object detection and aim assistance. Built with Python and ONNX Runtime, it supports DirectML, CUDA, and TensorRT GPU acceleration paths with automatic fallback, and ships a modern Fluent Design UI with multi-language support and a first-run setup wizard.

## What's new in 6.x

Since the original project, this fork has added:

- **TensorRT inference support** — build and run cached TensorRT engines for a significant inference FPS boost over DirectML/CUDA, via the **Convert** tab (`src/core/convert_to_engine.py`).
- **DirectML ↔ TensorRT fallback** — automatic provider fallback chain, mostly for the fun of squeezing out compatibility across more hardware.
- **Three new capture backends** — **NDI**, **UVC**, and **UDP** (JPEG stream over the OBS `udp_stream_filter` wire protocol), alongside the original dxcam/MSS. See `src/core/screen_capture.py`.
- **MAKCU improvements** — bumped to the official **4 Mbaud** connection, added **auto-reconnect**, and added support for **reading physical mouse button state** directly from the MAKCU device (`km.left()` / `km.right()` / `km.middle()`), so aim-button and click detection can be hardware-accurate instead of relying on OS-level key state.
- **Better aim prediction** — EMA smoothing, a 2D constant-velocity **Kalman filter**, and other smoothing options for leading moving targets, selectable in the Aim tab's Target Tracking group.
- **Reworked aim point** — fixed inconsistent aim-point placement, and added a manual fine-tune control so you can dial in exactly where on the target box the crosshair should settle (head/body/custom % offset).
- **Smart aim point for crouching targets** — the center-mass aim point adapts when a target's box shape indicates a crouched stance, instead of aiming at a fixed ratio that assumes standing.
- **Sticky lock** — IOU-based target persistence across frames, so a few frames of bad/missed detections (model being unreliable) don't immediately drop the lock; it remembers recent frames and holds on briefly.
- **Enhanced Model page** — more accurate model info display (input size, class list, provider, etc.) instead of guessed/static values.
- **Revamped Qt UI** — reorganized Fluent Design pages, nicer layout and visual polish across the app.
- **Detailed Other page** — expanded debugging info (MAKCU device info, TensorRT version detection, provider diagnostics, etc.).
- **Chroma visuals** — cycling hue box colors for ESP-style overlays.
- **Web ESP overlay** — stream the detection overlay to a browser source so OBS (or any browser) on a *second* PC/output can render ESP boxes without touching the primary game capture. See `src/web_overlay/` and `src/core/esp_server.py`.

## Beta features

These are early/incomplete — used at your own risk, and feedback is welcome:

- **Weapon detector (Apex Legends only, for now)**
  - **PaddleOCR-based** weapon-name reader — functional but currently not reliable/useful enough to depend on; left in the codebase but effectively unused (`src/core/ocr_inference.py`).
  - **YOLO-based** weapon HUD detector — needs a properly trained YOLO model to be useful; the inference plumbing exists (`src/core/hud_inference.py`) but no production-ready model ships yet.
  - Goal once a reliable detector lands: automatically apply the matching recoil-control pattern for the detected weapon.

Found a feature in the code that isn't listed above? Let us know rather than assuming — some things in progress aren't ready to be called a "feature" yet.

## Key Features

- **Advanced Aim Control**
  - PID controller with separate X/Y axis tuning for precise adjustments.
  - Bézier curve smoothing for natural, human-like mouse movement.
  - Customizable FOV and independent detection range.
  - Single-target mode for focusing on the nearest threat.
  - FOV follows mouse cursor for dynamic aiming.
  - Configurable head/body region ratios, plus manual fine-tune for the exact aim point.
  - Aim deadzone and lateral brake to reduce over-travel.

- **Anti-Recoil (Smart Jitter)**
  - Smart Jitter applies subtle movement to simulate natural hand shake while on target.
  - Choose between procedural random jitter or fully custom recorded patterns.
  - **Jitter Recorder** — record your own mouse shake pattern directly from the Aim Assist page (click anywhere in the live-preview window to finish & save — no separate button to reach for mid-pattern) or via the standalone terminal tool (`src/core/jitter_recorder.py`).
  - Patterns are zero-net-displacement: each loop cycle returns the cursor to its original position.
  - LMB gate: optionally activate jitter only while left mouse button is held.

- **Y-Axis Recoil Suppression**
  - Dedicated Y-axis reduction with configurable delay, ramp, floor, settle threshold, and velocity-restore.
  - Automatically scales down upward correction to compensate for in-game weapon recoil.

- **Smart Tracker (Prediction System)**
  - EMA and 2D Kalman-filter-based motion prediction for leading moving targets.
  - Adaptive smoothing with configurable prediction time.
  - Zero-lag reset on sudden direction changes or stops.
  - Sticky lock with IOU-based target persistence across frames — tolerates a few bad detection frames without dropping the target.
  - Smart center-mass aim point that adapts to crouching targets.
  - Visual prediction overlay for debugging and tuning.

- **Auto Fire (Triggerbot)**
  - Adjustable delay and fire intervals.
  - Target priority settings (Head / Body / Both).
  - Always-on mode or key-activated toggle.

- **High Performance**
  - ONNX Runtime with DirectML, CUDA, TensorRT, or CPU inference, with automatic fallback across providers.
  - Zero-copy CUDA IO binding for minimal GPU↔CPU transfer overhead.
  - Low-latency three-thread pipeline: capture → preprocess → inference, fully decoupled.
  - Performance mode with optimized queue management.
  - Idle detection throttling to reduce resource usage when not aiming.
  - **Multiple Input Methods**
    - **SendInput** (Windows API)
    - **ddxoft**
    - **Arduino Leonardo**
    - **Xbox 360 Virtual Controller**
    - **MAKCU** (USB HID, 4 Mbaud, auto-reconnect, hardware button-state reads)

- **Live Capture Preview**
  - Collapsible side panel shows the active capture feed in real-time with FPS counter.
  - Supports pop-out mode and optional crop to detection region.
  - Works with all backends: dxcam, MSS, UVC, NDI, UDP.

- **Web ESP Overlay**
  - Streams live detection boxes/state to a lightweight web page over WebSocket.
  - Point an OBS Browser Source (or any browser, on any machine on the network) at it to render ESP without capturing the main game overlay.
  - HUD shows aim on/off, aim active/idle, model, capture method, capture FPS, inference FPS, target count, and network latency — draggable and stylable from the in-page settings panel.

- **Modern Fluent Design UI**
  - Built with PyQt6 + QFluentWidgets for a native Windows 11 look.
  - Acrylic (frosted glass) window effect with configurable transparency.
  - Dark / Light theme toggle.
  - First-run Setup Wizard for quick configuration.
  - Detailed Other page for debugging device/provider state.

- **Multi-Language Interface**
  - English, 中文, Français, Deutsch, हिन्दी, 日本語, 한국어, Português, Русский, Español.

## Supported Games (Pre-trained Models)

| Model | File |
|-------|------|
| Apex Legends | `apex.onnx` |
| Counter-Strike 2 | `CS2.onnx` |
| Fortnite | `Fornite.onnx` |
| PUBG | `Pubg.onnx` |
| Roblox | `Roblox.onnx` |
| Valorant | `Valorant[PURPLE].onnx` |

> You can also train and import your own ONNX models.

## Why does Axiom exist?

Axiom is designed for gamers who are at a disadvantage compared to regular players, including but not limited to:
- Players grieving from parental loss
- Physical disabilities
- Intellectual disabilities
- Visual impairments
- Poor hand-eye coordination
- Poor FPS performance
- Hand tremors
- Parkinson's disease
- Neurological disorders
- Players with one arm/hand
- Players using feet due to hand loss
- Players using mouth due to limb loss
- Paralyzed players using brain-computer interfaces or eye trackers
- Colorblind players
- Blind players
- Players without glasses
- Elderly players
- Chronic fatigue syndrome
- Nystagmus sufferers
- Brain injury sequelae
- Spatial perception disorders
- Anxiety disorders
- ADHD
- Movement disorders
- Autism
- Sleep-deprived players
- Overconfident players
- Players prone to overthinking
- Emotionally volatile players
- Wrong DPI settings
- No mousepad users
- Limited mouse space
- Low-quality mouse users
- Cloud gaming users
- Mouse acceleration enabled
- No air conditioning in hot/humid areas
- Sweaty hands causing mouse slippage
- Poor posture or low chairs
- Very young child players
- Beginners or untrained players
- Unstable vision players
- Special controller users
- Lucid dreamers
- Sixth sense aiming players
- Religious players who consider aiming sinful
- Fatalists who believe fate decides everything
- Players seeking randomness and chaos
- Role-playing blind snipers
- Players who think crosshairs are decorative
- Players who think they're in third person
- Crosshair drift syndrome sufferers
- Voice navigation aiming players
- Schizophrenia
- Parallel world delay sync players
- Quantum state players
- Left-right hand fighting players
- Players who think right-click is fire
- Internal slow-motion animation players
- Moral players who wait for enemies to shoot first
- Hardware flip party
- Players who only aim at enemy weapons
- Players who only aim at enemy feet
- Players who only aim at enemy hands
- Players who only aim at enemy genitals
- Pixel-level instruction followers
- Players always aiming at the floor
- Feng shui players
- Players who chant before shooting
- Astrology-based FPS players
- Bad pixels on crosshair
- Screen reflection showing face in center
- Auto-sliding chairs
- Eyes-closed FPS challengers
- Left-hand-only announcement players
- No crosshair but forgot transparent crosshair stickers
- Noodle-eating players
- Extreme hypoglycemia sufferers
- Drunk players
- Extreme binocular disparity
- Sleep paralysis FPS players
- Players who believe enemies are illusions
- Players who can't distinguish directions
- Players who treat screen center as blind spot
- Severe choice paralysis
- Anti-authority players
- Performance anxiety players
- Players who don't want to harm virtual life
- Vibrating bed FPS players
- Low battery wireless mouse users
- 24FPS monitor users
- Slow reaction but fast movement players
- High altitude residents
- Quantum superposition enemy believers
- Mind's eye believers
- Wrong muscle memory players
- Projector users
- Cat-occupied mousepad players

> **Important Notice**: This software is licensed under the PolyForm Noncommercial License 1.0.0. Commercial use is strictly prohibited.

## System Requirements

### Minimum Requirements
- **OS**: Windows 10/11 (64-bit)
- **RAM**: 16 GB
- **Graphics**: GTX 1060 6 GB / RX 580 8 GB (DirectX 12 compatible)

### Recommended Requirements
- **OS**: Windows 11 (64-bit)
- **RAM**: 32 GB or higher
- **Graphics**: RTX 3060 or better

## Requirements / Acceleration

Axiom supports three ONNX Runtime acceleration paths on Windows, with automatic fallback:

- **DirectML (default)**
  - Package: `onnxruntime-directml`
  - Best first choice for broad hardware compatibility (NVIDIA / AMD / Intel GPUs that support DirectX 12).
  - This is the default in `requirements.txt`.

- **CUDA (optional, NVIDIA-only)**
  - Package: `onnxruntime-gpu`
  - Requires version-compatible **NVIDIA Driver + CUDA + cuDNN + ONNX Runtime GPU build**.
  - If these versions do not match, ONNX Runtime usually falls back to CPU (or fails to load the CUDA provider).

- **TensorRT (optional, NVIDIA-only, fastest)**
  - Builds a cached TensorRT engine from your ONNX model via the **Convert** tab (`src/core/convert_to_engine.py`).
  - Engines are cached in `%LOCALAPPDATA%\AxiomAI\` and rebuilt automatically if the source model changes.
  - Falls back to DirectML/CUDA/CPU automatically if TensorRT isn't available or engine build fails.

### Install options

- **Default (DirectML):**
  - `pip install -r requirements.txt`
  - or `pip install -r requirements-directml.txt`

- **CUDA path (NVIDIA):**
  - `pip install -r requirements-cuda.txt`

### Runtime provider selection

At startup, Axiom logs:
- `ort.get_available_providers()`
- selected backend from config (`auto` / `cuda` / `directml` / `tensorrt` / `cpu`)
- final active provider used by the loaded model session

Use these logs to verify which provider actually ended up active, or whether runtime fell back down the chain.

## Usage

### Basic Operation

1. **Launch the Application**
   - Run `啟動Launcher.bat` or `python src/main.py`.
   - On first launch, the **Setup Wizard** will guide you through initial configuration (language, theme, model selection).

2. **Configure Settings**
   - **Visuals Tab** — Customize overlays (FOV circle, bounding boxes, crosshair, status panel, chroma visuals, Web ESP).
   - **Model Tab** — Select your ONNX model and view detailed model info.
   - **Capture Tab** — Choose and configure your screen capture backend (dxcam / MSS / UVC / NDI / UDP); live preview panel.
   - **Inference Tab** — Detection sensitivity, inference provider, and performance tuning.
   - **Aim Tab** — FOV, PID tuning, prediction (EMA/Kalman), smoothing, aim-point fine-tune, Anti-Recoil (Smart Jitter + click-to-stop jitter recorder), Y-axis recoil suppression, sticky lock.
   - **Trigger Tab** — Configure auto-fire delay, interval, and target priority.
   - **Keys Tab** — Set your preferred hotkeys for toggling aim and auto-fire.
   - **Configs Tab** — Save / load configuration presets.
   - **Convert Tab** — Convert ONNX models to TensorRT engines.
   - **Other Tab** — Mouse method, Arduino, Xbox controller, MAKCU, performance options, and detailed debugging info.

3. **Start Detection**
   - Press the configured toggle key (default: `Insert`).
   - The system will begin real-time detection and overlay.

### Configuration Files

Settings are persisted to three files in the project root:

| File | Contents |
|---|---|
| `config.json` | All tunable settings in the v2 grouped schema (aim, capture, display, hardware, etc.) |
| `state.json` | One-time app state: disclaimer accepted, first-run complete, NDI installer flag |
| `language.json` | Current UI language selection |

You can also save/load named presets via the **Configs** tab. Recorded jitter patterns are stored as `.json` files in `src/core/jitter_patterns/`.

## Project Structure

```
Axiom/
├── 啟動Launcher.bat              # Quick-start launcher
├── config/                       # Saved configuration presets
├── Model/                        # ONNX model files
├── config.json                   # Runtime settings (v2 grouped schema)
├── state.json                    # One-time app state (disclaimer, first-run)
├── language.json                 # UI language selection
├── src/
│   ├── main.py                   # Application entry point
│   ├── version.py                # Version constant — single source of truth for app version
│   ├── core/                     # Core inference & control logic
│   │   ├── ai_loop.py            # Three-thread pipeline (capture / preprocess / inference)
│   │   ├── ai_loop_state.py      # Shared loop state
│   │   ├── ai_loop_utils.py      # Loop utility functions
│   │   ├── ai_aiming.py          # Aiming, PID, Smart Jitter, Y-reduce, humanization
│   │   ├── auto_fire.py          # Triggerbot logic
│   │   ├── config.py             # Config class + _FIELD_MAP (v2 schema)
│   │   ├── config_manager.py     # Load/save/preset/migration lifecycle
│   │   ├── detection_semantics.py# Semantic target filtering
│   │   ├── esp_server.py         # Web ESP WebSocket/HTTP server
│   │   ├── hud_inference.py      # Beta: YOLO-based weapon HUD detector
│   │   ├── ocr_inference.py      # Beta: PaddleOCR weapon-name reader
│   │   ├── humanization.py       # Velocity curves, Bézier smoothing, micro-corrections
│   │   ├── inference.py          # ONNX inference wrapper, NMS, postprocessing
│   │   ├── jitter_recorder.py    # Mouse jitter recording/replay terminal tool + GUI API
│   │   ├── jitter_patterns/      # Recorded jitter pattern JSON files
│   │   ├── kalman_filter.py      # Kalman filter for target smoothing
│   │   ├── key_listener.py       # Global hotkey listener
│   │   ├── language_manager.py   # Multi-language runtime support
│   │   ├── language_data/        # Per-language JSON string files
│   │   ├── screen_capture.py     # Capture backends: dxcam / MSS / UVC / NDI / UDP + preview API
│   │   ├── udp_receiver.py       # UDP JPEG stream receiver (OBS udp_stream_filter protocol)
│   │   ├── session_utils.py      # Session utilities
│   │   ├── target_predictor.py   # Velocity-based motion prediction
│   │   ├── convert_to_engine.py  # ONNX → TensorRT engine builder
│   │   ├── logging_config.py     # Logging setup
│   │   └── updater.py            # Auto-update checker
│   ├── web_overlay/               # Web ESP static client (served to the browser)
│   │   ├── index.html
│   │   ├── app.js
│   │   └── styles.css
│   ├── gui/
│   │   ├── overlay.py            # DirectX overlay (FOV, bounding boxes, crosshair)
│   │   ├── status_panel.py       # In-game status panel
│   │   └── fluent_app/           # Fluent Design main window & pages
│   │       ├── window.py         # AxiomWindow: navigation, acrylic, preview panel
│   │       ├── components/       # Shared widgets (sliders, capture preview panel)
│   │       └── pages/            # One file per settings page
│   │           ├── aim_page.py       # Aim parameters + Anti-Recoil section
│   │           ├── capture_page.py   # Capture backend selection
│   │           ├── configs_page.py   # Config preset management
│   │           ├── convert_page.py   # ONNX → TensorRT conversion
│   │           ├── inference_page.py # Inference / model settings
│   │           ├── keys_page.py      # Hotkey bindings
│   │           ├── model_page.py     # Model selection & notes
│   │           ├── other_page.py     # Mouse method, performance, misc, debugging
│   │           ├── trigger_page.py   # Triggerbot / auto-fire
│   │           └── visuals_page.py   # Overlay display settings (incl. Web ESP)
│   └── win_utils/                # Windows-specific device backends
│       ├── __init__.py           # is_key_pressed() and shared utilities
│       ├── mouse_move.py         # SendInput mouse movement
│       ├── mouse_click.py        # Mouse click simulation
│       ├── makcu_mouse.py        # MAKCU USB HID device (4 Mbaud, auto-reconnect, button reads)
│       ├── arduino_mouse.py      # Arduino Leonardo HID driver
│       ├── arduino_spoofer.py    # Arduino Leonardo spoof path
│       ├── ddxoft_mouse.py       # ddxoft driver integration
│       ├── gamepad_input.py      # Xbox 360 virtual controller input
│       ├── xbox_controller.py    # ViGEmBus Xbox 360 emulation
│       ├── key_utils.py          # Key code helpers
│       └── vk_codes.py           # Virtual key code table
└── tests/                        # Pytest test suite
    ├── conftest.py               # Adds src/ to sys.path
    ├── test_config.py
    ├── test_config_manager.py
    ├── test_humanization.py
    ├── test_inference.py
    ├── test_makcu_mouse.py
    ├── test_mouse_methods.py
    ├── test_screen_capture.py
    └── ...
```

## 📄 License

This project is licensed under the **PolyForm Noncommercial License 1.0.0**.

- ❌ **No Commercial Use** — This software cannot be used for any commercial purpose.
- ✅ **Personal Use** — Free for personal, educational, and research purposes.
- ✅ **Modification** — You may modify and distribute the software.
- ✅ **Attribution** — Must include original license and copyright notice.

For full license details, see [LICENSE.txt](LICENSE.txt) or visit [PolyForm Noncommercial License](https://polyformproject.org/licenses/noncommercial/1.0.0/).

## 📞 Contact & Support

- **Discord**: [Join for support](https://discord.gg/DpcqaQEj5b)
- **GitHub Issues**: [Chr0mX/Axiom-AI-Aimbot](https://github.com/Chr0mX/Axiom-AI-Aimbot/issues) — bug reports and feature requests for this fork.

The original project's Discord/contact channels are no longer affiliated with this repository and aren't listed here to avoid pointing people at stale/inactive links.

---

**Disclaimer**: This software is provided "as is" without warranty. Use at your own risk. The developers are not responsible for any consequences of using this software. See [Disclaimer.md](Disclaimer.md) for full terms.

**Original project by iisHong0w0 (2025-2026), now unmaintained/unavailable. This fork is an independent, community-driven "vibe coded" continuation.**
