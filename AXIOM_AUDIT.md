# AXIOM_AUDIT.md — Full Engineering Audit

**Repository:** `chr0mx/axiom-ai-aimbot` (+ companion repos `chr0mx/directshow-capture-dll`, `chr0mx/udp-stream-filter`)
**Branch audited:** `claude/directshow-dll-mjpeg-nv12-kuglu1`
**Audit date:** 2026-08-07
**Scope:** Every first-party source file across all three repositories (~90 files, ~39,000 lines). Vendored third-party code (`src/python/**`, bundled CPython + site-packages, PyQt6, PIL, etc.) is excluded except where it affects first-party behaviour.

> This document is generated incrementally. Sections are filled as each subsystem is analysed. The consolidated **Bug List / Performance / Security / Reliability** tables and the **Executive Summary** are assembled from the per-module findings once every module has been read.

---

## Finding ID & Severity Conventions

Every finding carries a stable ID so `AXIOM_ROADMAP.md` and `AXIOM_SUMMARY.md` can reference it:

| Prefix | Category |
|---|---|
| `BUG-nn` | Correctness / logic / races / leaks / undefined behaviour |
| `PERF-nn` | Performance (allocations, copies, contention, hot-loop cost) |
| `SEC-nn` | Security (input handling, injection, supply chain, network) |
| `REL-nn` | Reliability (exception safety, recovery, cleanup, state consistency) |
| `DEBT-nn` | Technical debt / code smell / dead code / duplication / docs |

**Severity:** Critical (data loss / crash / exploit in normal use) · High (wrong behaviour or leak under realistic conditions) · Medium (edge-case bug, degraded UX, or notable smell) · Low (cosmetic, minor, or defensive-improvement).

Each finding lists: severity · file:line · function · description · root cause · impact · recommended fix · effort (S ≤1 h / M ≤half-day / L ≥1 day).

---

## 1. Executive Summary

Axiom is a mature, feature-dense Windows real-time CV aim-assist application (~34k lines of first-party Python) backed by two rigorous native C++ companions (~4.9k lines): a DirectShow capture DLL and an OBS UDP-streaming plugin. This audit read **every first-party source file** across all three repositories.

**Overall health: good.** The hardest parts of the system are the best-engineered. The screen-capture subsystem is full of hard-won, well-documented mitigations for real-world driver misbehaviour; the MAKCU/xbox device backends are disciplined about their central concurrency hazard (never holding a device lock across a sleep); and the native C++ is exemplary — every COM lifetime decision is justified in-comment, reference counts are atomic, and all five documented native memory-safety fixes (leaks, short-buffer over-read, negative `biHeight`, per-poll memcpy) are present and correct on this branch.

**Where the risk actually is:**

- **A few real correctness bugs on this branch** — most notably NMS dropping `class_ids` so the semantic false-positive filter reads misaligned classes (BUG-09), and `id(frame)` reuse causing occasional frame skips (BUG-08). Several of these are already fixed on other project branches and just need forward-porting (cross-cutting finding X-4).
- **Network-facing features that trust the LAN** — the UDP receiver has an unbounded reassembly buffer (remote memory-exhaustion DoS, SEC-02) and the Web ESP broadcast can be frozen by a single slow client (REL-06); both bind all interfaces with no auth.
- **Durability of user settings** — every config/preset write is a non-atomic truncate-write, so an ill-timed crash silently wipes all settings (REL-02).
- **Structural coupling** — the single `Config` object doubles as the cross-thread message bus, mutated lock-free from 8+ threads (X-1); correctness leans on GIL atomicity that doesn't cover multi-field consistency.

**Counts:** 11 correctness bugs (1 High-conditional, 3 Medium, 7 Low), 3 performance, 3 security (1 Med-High, 1 Medium, 1 Low), 7 reliability (2 Medium), and ~21 technical-debt items. No Critical findings.

**Quick wins (High-value, small fix):** point the update checker at the shipping repo (DEBT-01); cap the UDP reassembly map (SEC-02); atomic config writes (REL-02); per-client ESP send timeout (REL-06); `frame_seq` instead of `id(frame)` (BUG-08); thread `class_ids` through NMS (BUG-09).

See `AXIOM_ROADMAP.md` for the phased plan and `AXIOM_SUMMARY.md` for the one-page executive view.

### 1a. Reconciliation against `origin/main` (read this first)

This audit was performed against the branch `claude/directshow-dll-mjpeg-nv12-kuglu1`, which **lags `main`** — its DirectShow feature work was merged (PRs #149–152) while `main` continued forward. This is cross-cutting finding **X-4**, and it is now concretely verified: several findings below are accurate for the audited branch but **already fixed on the shipping `origin/main`**. Verified directly against `origin/main` source:

**✅ Already fixed on `main` — do NOT action these from this report:**

| Finding | Evidence on `origin/main` |
|---|---|
| **BUG-09** (NMS drops `class_ids`) | `non_max_suppression` now takes **and returns** `class_ids`; call site is `boxes, confidences, class_ids = non_max_suppression(...)` |
| **BUG-08** (`id(frame)` reuse) | `capture_state['frame_seq']` bumped under `frame_lock`; preprocess dedupes on `frame_seq`, never `id()` |
| **PERF-01** (tensor queue keeps stale) | now drop-oldest (`get_nowait` the stale item, then re-`put` on `queue.Full`) |
| **BUG-04** (`import_config` crash) | `if not isinstance(config_data, dict): return None` guard present |
| **REL-06** (ESP broadcast freeze) | per-client `_BROADCAST_SEND_TIMEOUT` set; comment "Deliberately NOT settimeout(None)" |
| **BUG-01** (shutdown path) *(reworked)* | `app.aboutToQuit → _shutdown()` stops AI threads + ESP + saves config; re-verify the model-restart sub-path |
| **DEBT-04** (validation gaps) *(partial)* | added `model_box_format`/`model_has_objectness`/`nms_iou_threshold` config + `_validate_model_output_format`/`_validate_udp_recv_buffer_size`; some numeric ranges may still be unvalidated |

**❗ Verified STILL present on `main` — genuinely actionable:**

| Finding | Status on `origin/main` |
|---|---|
| **SEC-02 / REL-03** (UDP reassembly DoS) | no count/byte cap on `_partial_frames` |
| **REL-02** (non-atomic config writes) | still truncate-write; no temp+`os.replace`/backup |
| **DEBT-01** (updater repo) | `REPO_OWNER = "iishong0w0"` unchanged |
| **DEBT-12** (child-process `print` spam) | HUD child still emits per-scan `print()` |

**Not re-verified line-by-line against `main`:** the remaining Low-severity Python items and GUI/theming debt (may be partially addressed). **The native C++ findings (DirectShow DLL BUG-11/DEBT-14; OBS plugin REL-07/DEBT-15) live in separate repositories unaffected by Axiom's `main` and stand exactly as written.**

**Net effect:** on the shipping `main` branch, the actionable **High/Medium** list shrinks to **SEC-02, REL-02, DEBT-01** (plus the standing structural item DEBT-03 and the native-repo items). The larger branch-level list below remains a faithful record of the audited branch and a checklist for confirming each item's status wherever this code is deployed.

> Everything from §2 onward describes the **audited branch**. Cross-reference §1a before acting on any individual finding.

---

## 2. Repository Overview

Axiom is a **Windows-only, real-time computer-vision aim-assist application** for games. It captures the screen (or an external video feed), runs a YOLO-family ONNX object detector on each frame, selects a target, and drives the mouse toward it through one of several hardware/software mouse backends. A PyQt6 + Fluent-Widgets GUI configures every parameter; an optional in-game overlay and a LAN browser overlay visualise detections.

### Repository layout (first-party)

| Area | Path | Files | ~Lines |
|---|---|---|---|
| Entry / bootstrap | `src/main.py`, `src/version.py`, `src/model_detect.py`, `src/launch_fluent.py`, `src/install_*.py` | 7 | ~1,900 |
| Core inference/aiming | `src/core/*.py` | ~30 | ~9,500 |
| Mouse/device backends | `src/win_utils/*.py` | ~15 | ~3,000 |
| GUI | `src/gui/**` | ~20 | ~13,000 |
| Web overlay client | `src/web_overlay/*` | 3 | ~600 |
| Tests | `tests/*.py` | ~25 | ~5,700 |
| DirectShow DLL (C++) | `chr0mx/directshow-capture-dll` | ~20 | ~3,770 |
| OBS UDP plugin (C++) | `chr0mx/udp-stream-filter` | ~6 | ~1,100 |

### Companion repositories

- **DirectShow-Capture-DLL** — a native C++ DLL exposing a small C ABI for low-latency UVC capture (NV12/MJPEG) with in-DLL native cropping. Vendored into Axiom at `src/python/dependencies/directshow_capture.dll` and driven from `src/core/dshow_capture_native.py`.
- **udp-stream-filter** — an OBS Studio plugin that MJPEG-encodes the OBS output and streams it over UDP in a custom 14-byte-header wire format, consumed by Axiom's `UdpCapture` backend.

### Distribution model

The app ships as a **portable bundle**: a full embedded CPython interpreter and all site-packages live under `src/python/`, and native DLLs are vendored under `src/python/dependencies/`. There is no `pip install` step for end users; `main.py` performs extensive `sys.path` / `add_dll_directory` / `PATH` surgery at startup to wire up pywin32, ONNX Runtime, and the CUDA/TensorRT DLL stack.

---

## 3. Architecture Overview

### Execution model

```
main.py
 ├─ DPI awareness + sys.path/DLL wiring (must precede all imports)
 ├─ load Config (config.json / state.json / language.json)
 ├─ aim_toggle_key_listener  ──► daemon thread (polls hotkey @ 30 ms)
 ├─ QApplication (main/UI thread)
 │    ├─ DisclaimerDialog (first run)
 │    ├─ SetupWizard (first run)
 │    ├─ PyQtOverlay        (transparent click-through renderer)
 │    ├─ StatusPanel        (on-screen HUD)
 │    ├─ esp_server.start() (optional Web ESP: 3 daemon threads)
 │    └─ AxiomWindow        (Fluent settings GUI, 10 pages)
 └─ start_ai_threads(model_path)
      ├─ ai_logic_loop  ──► inference thread
      │     ├─ _capture_worker    (grab frames @ screenshot_interval)
      │     └─ _preprocess_worker (letterbox → tensor → queue)
      └─ auto_fire_loop ──► triggerbot thread
```

The **inference pipeline is a 3-stage producer/consumer**: capture → preprocess → inference, decoupled by locks and a bounded tensor queue (see §Inference Pipeline). Secondary weapon/HUD detection runs in a **separate child process** to keep the main loop and UI unblocked.

### Threading inventory (first-party)

- UI thread (Qt event loop)
- Inference thread (`ai_logic_loop`) + its capture/preprocess sub-workers
- Auto-fire thread
- Hotkey listener thread
- Per-backend capture reader threads (UVC/NDI/UDP each spawn their own)
- MAKCU reconnect-watchdog thread
- Web ESP: HTTP server thread + WebSocket accept thread + broadcast thread
- Secondary-inference feeder thread + child process

Shared mutable state lives almost entirely on the single `Config` object, read/written across threads mostly without locks (relies on CPython GIL atomicity of attribute assignment — see cross-cutting findings).

---

## 4. Module-by-Module Analysis

### 4.1 Bootstrap & Entry Point

**Files:** `src/main.py` (549), `src/version.py` (1), `src/core/session_utils.py` (299), `src/core/updater.py` (82), `src/core/logging_config.py` (37), `src/core/key_listener.py` (69), `src/model_detect.py` (369), `src/install_cyndilib.py` (181), `src/install_paddleocr_local.py` (228), `src/install_tensorrt_local.py` (337), `src/launch_fluent.py` (61)

**Purpose.** `main.py` is the single production entry point. Before any heavy import it (1) pins OpenMP/MKL/OpenBLAS to 1 thread, (2) sets Qt HiDPI env + `SetProcessDpiAwarenessContext`, (3) injects `src/`, vendored deps, and pywin32 onto `sys.path`, (4) reads the configured backend directly from `config.json` to decide whether to inject the AppData GPU packages *before* importing `onnxruntime`, (5) registers NVIDIA/TensorRT DLL directories, then constructs the Qt app and starts the AI/auto-fire threads. `session_utils.py` owns ORT provider-list construction, session options, and the `InferenceController` pause/stop primitive. `updater.py` is a `QThread` GitHub-release checker. `model_detect.py` is a standalone `.onnx`/`.engine` inspector CLI.

**Assessment.** The startup ordering is genuinely load-bearing and is well-commented about *why* each step must precede the next (a common source of "silently falls back to CPU" bugs). `InferenceController` is clean and correct (Event-based pause/stop with a proper `wait_while_paused` that unblocks on stop). Provider construction is defensive and filters to actually-available providers. The main weaknesses are ordering/consistency between the two thread-lifecycle paths, mixed logging discipline, and the update-checker target.

#### Findings

**DEBT-01 · Medium · `src/core/updater.py:10-12` · module scope**
Update checker points at the wrong repository. `REPO_OWNER = "iishong0w0"` / `REPO_NAME = "Axiom-AI-Aimbot"`, so the in-app "check for updates" resolves `api.github.com/repos/iishong0w0/Axiom-AI-Aimbot/releases/latest` — the *upstream* project, not `chr0mx/axiom-ai-aimbot` which is what is actually being built and pushed here.
*Root cause:* hardcoded owner not updated when the project was forked/renamed.
*Impact:* Users are offered (or denied) updates based on a different release stream than the one they installed; version comparisons can be meaningless. If intentional (tracking upstream), it should be documented.
*Fix:* Point `REPO_OWNER` at the shipping repo, or make it a config/constant with a comment stating the intent. *Effort: S.*

**BUG-01 · Medium · `src/main.py:331-340` · `start_ai_threads()`**
The model-reload/restart path stops the old inference thread by setting `config.Running = False` and joining, but — unlike `stop_ai_threads()` — it never calls `_inference_controller.request_stop()`. If the loop is currently parked in `InferenceController.wait_while_paused()` (i.e. `inference_paused`/`pause()` was in effect), nothing clears the pause, so the join blocks the full 3 s timeout and the thread is then abandoned while still alive (the warning is logged and execution continues, potentially running two inference threads briefly). Additionally, the `auto_fire_thread.join()` is nested inside `if ai_thread is not None`, so an alive auto-fire thread is never joined when `ai_thread is None`.
*Root cause:* two divergent shutdown code paths (`stop_ai_threads` vs. the inline restart in `start_ai_threads`) that don't share logic.
*Impact:* Rare double-thread / stale-session window on model or provider hot-swap while paused.
*Fix:* Have `start_ai_threads` call `stop_ai_threads(config)` for teardown (single code path), and unnest the auto-fire join. *Effort: S.*

**REL-01 · Low · `src/core/key_listener.py:63-68` · `aim_toggle_key_listener()`**
The hotkey loop wraps each 30 ms iteration in `except Exception` and calls `traceback.print_exc()`. A persistent error (e.g. a bad VK code, or a gamepad backend raising every poll) would spam stderr ~33×/s indefinitely with no backoff and no rate-limit. It also uses `print()` rather than the configured logger.
*Fix:* Log via `logger.exception(...)` with a one-shot/rate-limited guard, or a short backoff sleep after an error. *Effort: S.*

**DEBT-02 · Low · `src/main.py` (several), `src/core/key_listener.py` · cross-cutting**
Logging discipline is inconsistent: `print()` is used for user-visible status in `key_listener.py` (`[快捷鍵] ...`), for the AppData-path injection (`main.py:125`), and for DLL-preload warnings (`main.py:86`), while the rest of the codebase uses the `logging` module. `import ctypes` appears twice in `main.py` (lines 27 and 73). Comments are a mix of Traditional Chinese and English throughout core startup, raising the bar for non-Chinese-reading maintainers.
*Impact:* Log output is fragmented (some lines bypass formatting/levels/handlers); harder onboarding.
*Fix:* Route through `logging`; de-dupe the `ctypes` import; consider standardising comment language (large, optional). *Effort: S–M.*

**BUG-02 · Low · `src/main.py:258-259, 278-291, 329-404` · module globals**
`ai_thread` / `auto_fire_thread` are module globals mutated from both the Qt/GUI callback thread (`start_threads_callback` → `start_ai_threads`) and from `stop_ai_threads`/`main` with no lock. Concurrent "reload model" and "shutdown" (or two rapid reloads) can interleave the join/reassign sequence.
*Impact:* Low in practice (GUI actions are serialised on the UI thread), but the pattern is a latent race.
*Fix:* Guard the lifecycle transitions with a small lock, or funnel all start/stop through one serialised controller. *Effort: M.*

**BUG-03 · Low · `src/core/updater.py:14-31` · `parse_version()`**
A version tag with a non-numeric suffix in any dotted field (`"6.4-beta"`, `"6.4rc1"`) makes `int(part)` raise, and the `except ValueError` substitutes `0` for that whole field — so `"6.4-beta"` parses to `(6, 0, 0)`, comparing as *older* than `6.3`. Pre-release release tags would misreport "up to date".
*Fix:* Strip non-numeric trailing characters per field (regex `^\d+`) before `int()`, or use `packaging.version`. *Effort: S.*

**SEC-01 · Medium · `src/install_cyndilib.py:92-158` · `download_file()` / `install_ndi_runtime()`**
The NDI runtime is downloaded from `NDI_RUNTIME_URL` and then executed silently (`installer.exe /verysilent`) with **no checksum or signature verification**. The fallback path shells out to `powershell ... -Command "Invoke-WebRequest -Uri '{url}' -OutFile '{destination}'"`, string-interpolating `url`/`destination` into a PowerShell command line.
*Root cause:* convenience download-and-run without integrity pinning; string-built PowerShell command.
*Impact:* A MITM or a compromised/redirected download host could substitute a malicious installer that is then run silently with the user's privileges. `url` is currently a hardcoded constant so the PowerShell interpolation is not attacker-controlled today, but it is a fragile pattern.
*Fix:* Pin and verify a SHA-256 of the downloaded installer before executing; prefer passing args to PowerShell without string interpolation (or drop the PowerShell fallback). *Effort: M.*

**Positive notes.**
- `InferenceController` (`session_utils.py:84-147`) is a correct, minimal pause/stop primitive; `request_stop()` deliberately clears the pause event so a stop can't deadlock a paused waiter.
- The "read backend from config.json before importing onnxruntime" dance (`main.py:107-126`) correctly prevents the CUDA/GPU build from shadowing `DmlExecutionProvider` for DirectML users — a subtle, well-justified ordering.
- `model_detect.py` is defensive (dual TRT API support, filename-based precision override, helpful install hints) and self-relaunches under the embedded interpreter.

---

### 4.2 Config Subsystem

**Files:** `src/core/config.py` (878), `src/core/config_manager.py` (276), `src/core/presets/*.json`

**Purpose.** `config.py` defines the single `Config` class holding **all** runtime state (~180 attributes), plus module-level `save_config`/`load_config`/`save_state`/`load_state` and a family of `_validate_*` clamps. Persistence is driven by `_FIELD_MAP`, a flat-attr → dotted-JSON-path table that generates both `to_dict()` (v2 grouped schema) and `from_dict()` (nested-first, flat-key fallback = the v1→v2 migration). `config_manager.py` layers preset save/load/rename/import/export on top, seeding bundled presets from `src/core/presets/` and sanitising preset names.

**Assessment.** The `_FIELD_MAP` single-source-of-truth is a genuinely good design — presets, `config.json`, and the in-memory object can't drift because all three derive from one table. Name sanitisation against path traversal is correct and well-commented. The weaknesses are (a) `Config` doubling as both the persisted-settings store *and* the cross-thread runtime message bus, (b) non-atomic file writes, (c) a crash path in `import_config`, and (d) load-time validation that covers only a minority of numeric fields.

#### Findings

**REL-02 · Medium · `src/core/config.py:682-701` (`save_config`), `config_manager.py` (all writers) · cross-cutting**
No file write is atomic. Every persistence path opens the target with mode `'w'` (truncate) and writes JSON in place. A crash, power loss, or disk-full **during** the write leaves a truncated/half-written `config.json`; the next `load_config` then hits `json.JSONDecodeError`, returns `False`, and the app silently falls back to *all defaults* — the user's entire configuration is lost with only an INFO/ERROR log line. There is no backup copy and no temp-file+rename.
*Root cause:* direct truncate-write with no durability strategy.
*Impact:* Total settings loss on an ill-timed crash; same risk for every preset file.
*Fix:* Write to `config.json.tmp` then `os.replace()` (atomic on Windows and POSIX); optionally keep a `.bak` of the last good file. Apply to `save_config`, `save_state`, and all `ConfigManager` writers. *Effort: S–M.*

**BUG-04 · Medium · `src/core/config_manager.py:249-276` · `import_config()`**
`import_config` does `config_data = json.load(f)` then immediately `config_data.get('name', ...)`. If the imported file is syntactically valid JSON but **not an object** (e.g. a top-level array `[...]`, a string, or a number), `.get()` raises `AttributeError`, which is **not** in the `except (OSError, json.JSONDecodeError)` clause and therefore propagates uncaught to the GUI caller (crash / error dialog). (Note: the analogous non-dict guard that exists in some other branches is *not* present on this branch.)
*Root cause:* assumes top-level JSON is always a dict.
*Impact:* A malformed or hostile preset file crashes the import flow instead of being rejected gracefully.
*Fix:* After `json.load`, `if not isinstance(config_data, dict): return None`. *Effort: S.*

**DEBT-03 · Medium (architecture) · `src/core/config.py:227-586` · `Config`**
`Config` is a **god object**: it stores persisted user settings (~150 fields via `_FIELD_MAP`) *and* live inter-thread runtime state (`latest_boxes`, `latest_all_boxes`, `screenshot_frame_count`, `detection_frame_count`, `Running`, `AimToggle`, `makcu_aim_active`, `display_locked_box`, `udp_width/height`, `source_nominal_fps`, …) on the same instance. This single mutable object is read and written by the UI thread, inference thread, capture workers, auto-fire thread, hotkey thread, Web ESP threads, and the OCR feeder — almost entirely **without locks**, relying on CPython GIL atomicity of individual attribute assignments.
*Root cause:* organic growth; the config object became the convenient global bus.
*Impact:* Total coupling (every module imports and mutates `Config`); correctness depends on GIL semantics that don't guarantee consistency across *multi-field* updates (e.g. `latest_boxes` and `latest_confidences` can be observed out of sync by the overlay/ESP); very hard to test in isolation.
*Fix (incremental):* Split runtime IPC state into a separate `RuntimeState` object with explicit locks for the multi-field groups (boxes+confidences), leaving `Config` as pure persisted settings. Large; stage it. *Effort: L.*

**DEBT-04 · Low–Medium · `src/core/config.py:776-878` · `_validate_*`**
Load-time validation covers `detect_interval`, `screenshot_interval`, `idle_detect_interval`, `screenshot_method`, `mouse_method`, `inference_backend`, `thread_priority`, and `detect_range_size` — but **not** the many other externally-loadable numeric fields: `min_confidence` (should be 0–1), `fov_size` (>0, ≤ detect_range), all six PID gains, `web_esp_http_port`/`web_esp_ws_port`/`udp_bind_port` (1–65535), `acrylic_window_alpha`/`acrylic_element_alpha` (0–255), `xbox_sensitivity`/`xbox_deadzone`, `smart_jitter_*`, and `hud_roi_coords` format. GUI sliders clamp these interactively, but a hand-edited or **imported** `config.json`/preset (explicitly untrusted per project docs) bypasses the GUI and is loaded unchecked.
*Impact:* Out-of-range values silently degrade behaviour (e.g. `min_confidence=5.0` → nothing ever detected; a bad port → server bind failure).
*Fix:* Extend the `_validate_*` battery to clamp the remaining ranges, ideally table-driven alongside `_FIELD_MAP`. *Effort: M.*

**BUG-05 · Low · `src/core/config.py:614-623` · `from_dict()` type coercion**
For a `bool` field, coercion is `bool(val) if expected is bool else expected(val)`. If a persisted/edited JSON stores a **string** for a bool field (`"false"`, `"0"`, `"no"`), `bool("false")` is `True` (non-empty string) — the value flips to the opposite of what was written.
*Impact:* Only affects hand-edited/legacy/corrupt configs, but silently wrong.
*Fix:* Parse bool-like strings explicitly (`val.strip().lower() in {"1","true","yes","on"}`). *Effort: S.*

**BUG-06 · Low · `src/core/config.py:847-865` · `_validate_detect_range_size()`**
The clamp is `max(min_size, min(max_size, raw))` with `min_size = fov_size` and `max_size = height`. It never validates that `fov_size ≤ height`. If a config sets `fov_size > height`, `min_size > max_size` and the result is `min_size` (= `fov_size`), so `detect_range_size` ends up **larger than the screen height**, violating the documented invariant.
*Fix:* Clamp `min_size` to `max_size` first (`min_size = min(min_size, max_size)`). *Effort: S.*

**DEBT-05 · Low · `src/core/config.py:18-22, 704-715`**
`_get_screen_size()` calls `user32.SetProcessDPIAware()` again even though `main.py` already established Per-Monitor-V2 awareness before any import (redundant, and a *downgrade* attempt that the OS ignores once V2 is set). `_migrate_config()` is a documented no-op (migration is actually done by `from_dict`'s dual-read) — fine, but the empty function invites confusion about where migrations live.
*Fix:* Drop the redundant DPI call; leave a one-line comment on `_migrate_config` pointing at `from_dict`. *Effort: S.*

**Positive notes.**
- `_sanitize_config_name` (`config_manager.py:25-38`) correctly defends preset file paths against traversal from both GUI free-text and imported-file `name` fields (`basename` + invalid-char regex + trailing dot/space strip).
- `_seed_builtin_presets` never clobbers a user file of the same name and re-seeds deleted built-ins — sensible.
- `_FIELD_MAP`-derived preset serialisation (`_get_config_data`) means presets can't drift from `Config`'s field set, and older/smaller presets load without wiping current values (from_dict skips absent keys).

---

### 4.3 Capture Subsystem

**Files:** `src/core/screen_capture.py` (2,798), `src/core/udp_receiver.py` (216), `src/core/dshow_capture_native.py` (383)

**Purpose.** Five interchangeable capture backends behind a common `grab(region) → BGRA ndarray | None` contract: **MSS** (GDI fallback), **dxcam** (Desktop Duplication), **UVCCapture** (capture cards/cameras — with three sub-implementations: cv2 dshow/msmf, an external ffmpeg subprocess, and the native `directshow_capture.dll` v2 path), **NDICapture** (cyndilib), and **UdpCapture** (MJPEG-over-UDP from the OBS plugin). `initialize_screen_capture` builds the configured backend (falling back to MSS on any failure); `reinitialize_if_method_changed` hot-swaps backends every ~0.5 s when config changes *or* when a threaded backend goes silent (`_last_frame_perf_time` staleness). `udp_receiver.py` reassembles the chunked UDP wire format; `dshow_capture_native.py` is the ctypes binding to the C ABI.

**Assessment.** This is the most heavily engineered and best-defended subsystem in the codebase. It is full of hard-won, well-documented mitigations for real-world capture-driver misbehaviour: FOURCC/CONVERT_RGB readback verification, measured-vs-requested FPS shortfall warnings, crop-before-convert to avoid full-res colour conversion, double-buffered NDI conversion to avoid per-frame allocation, throttled reinit to avoid hammering a dead device, `_warn_once` to prevent log floods, and a self-healing native-crop confirmation against the first real frame. The defects that remain are narrow: an unbounded reassembly buffer exposed to the network, one non-atomic frame/dimension publish, and minor observability gaps.

#### Findings

**REL-03 / SEC-02 · Medium–High · `src/core/udp_receiver.py:126-201` · `_recv_loop()`**
The partial-frame reassembly map `self._partial_frames` is keyed by the attacker-controllable `frame_id` (uint32 straight off the wire) and is bounded only by time-based eviction (`_EVICT_INTERVAL = 0.25 s`, entries older than `frame_timeout = 1 s`). There is **no cap on the number of concurrent partial frames or total buffered bytes**. A malicious or malfunctioning sender on the LAN (the socket binds `0.0.0.0` by default and UDP is unauthenticated) can send a flood of packets each with a distinct `frame_id` (and/or a large `total_chunks` so frames never complete); every unique id allocates a dict entry holding its chunk payloads (up to `recv_buffer_size` each) for up to ~1 s before eviction. At the packet rates the code itself cites (~35 k pkt/s), this is a multi-hundred-MB to multi-GB transient — an out-of-memory DoS. `total_chunks`/`total_size`/`chunk_size` are also used without sanity bounds.
*Root cause:* trusts an unauthenticated network source to be well-behaved; eviction is time-based only.
*Impact:* Remote (LAN) memory-exhaustion DoS of the whole app while UDP capture is enabled.
*Fix:* Cap `len(self._partial_frames)` (evict oldest when exceeded) and/or cap total buffered bytes; validate `total_chunks`/`total_size` against a sane maximum and drop packets that violate it; consider binding to a specific interface by default. *Effort: M.*

**BUG-07 · Low–Medium · `src/core/screen_capture.py:1894-1900, 2219-2236` · `_reader_worker_native_dll()` / `UVCCapture.grab()` (native NV12)**
On the native-DLL NV12 path the reader thread publishes the frame buffer under `_latest_frame_lock` (line 1896-1897) but updates the frame's dimensions **outside** the lock immediately after (`self.preview_width = w; self.preview_height = h`, lines 1899-1900). `grab()` reads the frame under the lock, then reads `self.preview_height`/`preview_width` *outside* the lock and passes `self.preview_height` into `_crop_nv12(frame_bgr, self.preview_height, …)`. If the delivered resolution changes between two frames (e.g. exactly at the one-shot native-crop self-heal that flips delivered dims, or a mid-stream mode change), `grab()` can pair a frame with the *other* frame's height, making the NV12 plane math wrong — producing a `None` crop, a garbage crop, or a `cv2.cvtColor` shape error.
*Root cause:* frame and its dimensions are not published atomically; `grab()` trusts separately-published ints instead of the frame array's own shape.
*Impact:* Rare corrupted/dropped frame at a resolution transition on the v2 NV12 path.
*Fix:* Derive luma height/width directly from `frame_bgr.shape` in the raw-NV12 branch of `grab()` (shape is inherently consistent with the buffer), or publish `(frame, w, h)` together under the lock. *Effort: S.*

**DEBT-06 · Low · `src/core/dshow_capture_native.py:159-209` · `_load_dll()`**
The binding does not bind the DLL's `capture_get_short_sample_count` or `capture_get_negotiated_format` exports, so the native path can't surface the DLL's own short-sample/dropped-frame diagnostic or its authoritative negotiated format — the Python side reconstructs negotiated dims from the first frame instead. Not a bug (the DLL's design is latest-frame-wins and the first-frame check covers crop confirmation), but it forfeits a ready-made observability signal that would make "my fps quietly halved" diagnosable.
*Fix:* Bind both exports (hasattr-guarded for old DLLs) and log the short-sample count periodically. *Effort: S.*

**DEBT-07 · Low · `src/core/screen_capture.py:943-961` · `NDICapture.grab()` double-buffer**
The no-preview NDI crop path returns a reference into a 2-slot reusable buffer (`self._bgra_bufs[idx]`) and flips `self._bgra_idx`. A returned frame stays valid for exactly one more `grab()` before its buffer is overwritten in place by a later `cv2.cvtColor(..., dst=buf)`. This is a deliberate zero-allocation optimisation and is safe under the current "consume each frame before the next-next grab" pipeline usage, but it is an implicit lifetime contract that isn't enforced or documented at the `grab()` boundary — a future caller that holds a frame reference across grabs would see it mutate.
*Fix:* Document the one-grab validity in the `grab()` docstring, or return `.copy()` when the caller can't guarantee prompt consumption. *Effort: S.*

**Positive notes.**
- Every threaded backend seeds `_last_frame_perf_time` to "now" so a device that opens but never delivers is caught by the same staleness path as one that dies later — a subtle correctness detail done right in all three backends.
- The native-crop **self-heal** (`_reader_worker_native_dll:1845-1868`) verifies the DLL actually honoured the crop against the first frame's real dimensions and falls back to a software crop for an older DLL that silently ignored the new struct fields — exactly the right way to evolve a vendored-binary ABI.
- `_jpeg_dimensions` parses SOF markers with correct bounds checks; `_crop_nv12` rounds to even boundaries and returns `None` (never throws) on any out-of-range rectangle; `_read_exact` correctly handles short pipe reads.
- ffmpeg subprocess teardown kills the process *before* joining the reader thread, avoiding a guaranteed 1 s join timeout on every close — the kind of detail usually missed.

---

### 4.4 Inference Pipeline

**Files:** `src/core/ai_loop.py` (766), `src/core/inference.py` (326), `src/core/ai_loop_utils.py` (278), `src/core/ai_loop_state.py` (43)

**Purpose.** `ai_logic_loop()` is the master real-time loop. It spins up a **capture worker** (grabs frames into `capture_state` under `frame_lock`, hot-swaps backends every 0.5 s) and a **preprocess worker** (letterbox/resize → ONNX tensor → `_tensor_queue`), while the main thread pulls tensors, runs the ONNX session (or `run_with_iobinding` for CUDA), post-processes + NMS, runs the semantic filter, FOV-filters, and calls `process_aiming`. It also handles model/provider hot-swap, MAKCU aim-button state, disengage delay, idle-detect throttling, sticky-lock decay, and single-target reduction. `inference.py` holds `preprocess_image` (letterbox with a cached canvas), `postprocess_outputs` (layout A/B + xyxy/cxcywh detection, letterbox reversal), `non_max_suppression`, and the `PIDController`. `ai_loop_utils.py` has the region math and box-list helpers; `ai_loop_state.py` is the per-loop `LoopState` dataclass.

**Assessment.** The threading architecture is sound: three-stage decoupling with two dedicated locks, cooperative pause/stop, graceful model hot-swap that drains stale tensors, and correct letterbox coordinate reversal. Two real correctness defects exist on this branch — an `id()`-reuse frame-dedup and an NMS/`class_ids` misalignment — plus a tensor-queue staleness choice. (Note: several of these are fixed on other branches per the project's audit history, but are present on the branch under audit.)

#### Findings

**BUG-09 · High (conditional) · `src/core/ai_loop.py:625, 634-637` + `src/core/inference.py:279-326` · `ai_logic_loop` / `non_max_suppression`**
`postprocess_outputs` returns three aligned lists `(boxes, confidences, class_ids)`. NMS is then applied as `boxes, confidences = non_max_suppression(boxes, confidences)` — which both **drops boxes** and **reorders** the survivors by descending confidence (`order = argsort[::-1]` → `keep`) — but `class_ids` is **not** passed through NMS and keeps its original order and length. Immediately after, when `detect_semantic_filter_enabled` is on, `filter_detections_by_semantic_class(boxes, confidences, class_ids, config)` is called with the now-**misaligned** `class_ids`: `class_ids[i]` no longer describes `boxes[i]`, and the lists can even differ in length.
*Root cause:* NMS signature never threaded `class_ids`.
*Impact:* With the semantic false-positive filter enabled **and** a multi-class model, each detection is classified by the wrong class name, so the allow/deny/geometry decisions are applied to the wrong boxes — valid targets dropped, filtered classes kept. (Harmless for a single-class model, where all `class_ids` are 0, which is why the default Apex config doesn't expose it.)
*Fix:* Thread `class_ids` through `non_max_suppression` (return the kept indices and reindex all three lists), or run the semantic filter *before* NMS. *Effort: S–M.*

**BUG-08 · Medium · `src/core/ai_loop.py:399-406` · `_preprocess_worker()`**
New-frame detection uses `frame_id = id(frame)` and skips when `frame_id == last_frame_id`. Python reuses `id()` (object address) after an object is freed, so once the capture worker replaces `capture_state['latest_frame']`, the previous ndarray can be garbage-collected and a **new** frame allocated at the **same address**, yielding an identical `id()`. The preprocess worker then treats a genuinely new frame as already-seen and skips it, stalling the pipeline until a frame lands at a different address.
*Root cause:* `id()` is not a stable content/sequence identifier.
*Impact:* Intermittent dropped frames / added latency, hard to reproduce; worse under memory pressure where address recycling is more frequent.
*Fix:* Publish a monotonically-increasing `frame_seq` counter alongside the frame under `frame_lock` and compare that instead of `id()`. *Effort: S.*

**PERF-01 · Low–Medium · `src/core/ai_loop.py:310, 428-431` · `_tensor_queue`**
The tensor queue is `maxsize=1` and the preprocess worker does `put(..., timeout=0.05)` then `except queue.Full: pass` — i.e. when inference stalls >50 ms the **newest** tensor is discarded and the older queued one is what inference eventually consumes (drop-newest / keep-stale). For a latency-critical aimbot the opposite is preferable: always hand inference the freshest frame.
*Root cause:* Full-queue policy drops the incoming item rather than evicting the stale one.
*Impact:* Under inference load spikes, inference can act on a staler frame than necessary. Minor in steady state (queue rarely stays full a full 50 ms).
*Fix:* On `queue.Full`, `get_nowait()` the stale tensor then `put` the fresh one (evict-oldest), matching `update_queues`' pattern. *Effort: S.*

**DEBT-08 · Low · `src/core/ai_loop.py:10, 755-758` · main-loop error handler**
The top-level loop `except Exception` uses `traceback.print_exc()` (stdout, bypassing the logger) plus a blanket `time.sleep(1.0)`. A recurring exception both spams stdout outside the logging system and throttles the entire loop to 1 Hz with no diagnosis aid beyond the raw traceback.
*Fix:* `logger.exception(...)`; consider a bounded error counter that stops or escalates after N consecutive failures. *Effort: S.*

**DEBT-09 · Low · `src/core/inference.py:14, 155-161` · `_canvas_cache`**
The letterbox canvas cache is a module-global reused in place (`canvas[:] = 114; canvas[...] = resized`). It is safe only because a single thread (`_preprocess_worker`) calls `preprocess_image` in the main pipeline — but `hud_inference`/`ocr_inference` and tests also import this module, and any future second concurrent caller at the same `model_input_size` would corrupt a frame mid-build.
*Fix:* Document the single-thread assumption, or key the canvas per-thread / allocate when contended. *Effort: S.*

**Positive notes.**
- Letterbox forward (`preprocess_image`) and reverse (`postprocess_outputs`) transforms are correct and well-explained; the square fast-path and non-square letterbox path are cleanly separated, and the Y-axis distortion fix is real.
- `postprocess_outputs` robustly auto-detects YOLO output layout (A vs. B transpose) and xyxy-vs-cxcywh box encoding, and uses `cols[4:]` max as confidence so it never mistakes a coordinate for a score.
- Model hot-swap drains the tensor queue and re-syncs semantic class names, and IO-binding failure falls back to a plain `model.run` without killing the loop.
- Sticky-lock decay, disengage-delay falling-edge handling, and the single-target reduction being derived from the *post-lock* pick (not a pre-filter) are all implemented carefully and match the documented intent.

---

### 4.5 Aiming Stack

**Files:** `src/core/ai_aiming.py` (450), `src/core/target_predictor.py` (72), `src/core/kalman_filter.py` (93), `src/core/humanization.py` (219), `src/core/detection_semantics.py` (228), `src/core/auto_fire.py` (151)

**Purpose.** `process_aiming()` is the per-frame aiming entry point: compute the aim point per box (`calculate_aim_target`, with adaptive head-ratio and posture-awareness), priority-score and select a target, apply sticky lock (adaptive IOU), optional velocity prediction (`VelocityPredictor`) and Kalman smoothing (`KalmanFilter2D`), camera-motion compensation, adaptive deadzone, dual-axis PID, Y-recoil suppression, humanization (`humanization.py`), sub-pixel carry, per-frame cap, and smart jitter — then emit the mouse move. `detection_semantics.py` is the three-layer false-positive filter; `auto_fire.py` is the triggerbot that fires when the crosshair is inside a target's head/body sub-region.

**Assessment.** This is the most feature-dense subsystem and, notably, remains readable and correct in its core math: divide-by-zero guards are consistently present, prediction has a velocity sanity cap, sticky lock uses an area-scaled adaptive IOU, the deadzone keeps PID `previous_error` fresh to avoid a Kd kick, and humanization is provably zero-bias and O(1). Findings are minor tuning/robustness issues plus the cross-module NMS/`class_ids` interaction already logged as BUG-09 (whose named-class layer lives here in `filter_detections_by_semantic_class`).

#### Findings

**BUG-10 · Low · `src/core/ai_aiming.py:269-271` · `process_aiming()` continuity check**
The Y-recoil velocity-restore gate resets its timestamp when the newly-selected box "isn't a continuation" of the previous one, using the **raw** `lock_iou_threshold` (`_iou_thresh`) for that continuity test. But the sticky lock itself decided continuation using the **adaptive** area-scaled threshold (`_adaptive_sticky_iou`, often much lower, e.g. 0.34× base). The two thresholds disagree: a target sticky-lock considers the same can be judged "different" by the continuity check, spuriously zeroing `aim_y_last_target_t` and forcing `_vy = 0` next frame — defeating the velocity-restore gate exactly when tracking a small/far target.
*Fix:* Reuse the same (adaptive) threshold the lock used for the continuity comparison. *Effort: S.*

**PERF-02 · Low · `src/core/ai_aiming.py:56-68` · `_get_predictor()`**
When prediction is enabled this is called every frame and, on every call, **rebuilds the history `deque`** (`type(_predictor._history)(_predictor._history, maxlen=history_len)`) and pokes private attributes directly. That is a per-frame heap allocation and object churn for a value that changes only when the user edits `prediction_history_len`.
*Fix:* Add a `reconfigure(history_len, max_velocity)` method to `VelocityPredictor` (as `KalmanFilter2D.reconfigure` already has) that mutates in place only when a parameter actually changed. *Effort: S.*

**DEBT-10 · Low · `src/core/ai_aiming.py:433-442` · smart-jitter procedural branch**
The procedural-jitter fallback (random polar offset) is duplicated verbatim in two branches (no pattern iterator, and no pattern file). Extract to a small local helper. Also, `calculate_aim_target` carries four `TODO` feature stubs (X-offset, Y-nudge, per-class routing, confidence blend) in shipping code — track them as issues rather than inline TODOs. *Effort: S.*

**REL-04 · Low · `src/core/kalman_filter.py:86-88` · `KalmanFilter2D.update()`**
`np.linalg.inv(S)` has no guard. `S = H·P·Hᵀ + R`; with `measurement_noise = 0` (nothing validates it — see DEBT-04) `R = 0` and `S` can become singular, raising `LinAlgError` that unwinds up to the `ai_logic_loop` catch-all (1 s stall). Kalman is opt-in and defaults to 0.1, so this is latent.
*Fix:* Clamp `measurement_noise`/`process_noise` to a small positive floor (in config validation and/or `reconfigure`), or use `np.linalg.pinv`. *Effort: S.*

**Positive notes.**
- `VelocityPredictor` discards its history on a velocity spike above `max_velocity` (detection jump vs. real motion) — the right way to keep a constant-velocity model from chasing teleports.
- `detection_semantics.py` never uses `eval` — it parses YOLO `names` metadata with `json.loads` then `ast.literal_eval` fallback, and pads/clamps `class_ids` defensively per box.
- The adaptive deadzone deliberately calls `pid_x.update(0.0)`/`pid_y.update(0.0)` while suppressed so the derivative term doesn't see a stale multi-frame error and produce a spurious Kd kick when the target leaves the deadzone — a subtle, correct detail.
- `auto_fire.py`'s head/body/both hit-testing is geometrically sound and reads from the single-target-reduced box list.

---

### 4.6 Input / Device Backends

**Files:** `src/win_utils/` — `__init__.py` (246, dispatcher), `makcu_mouse.py` (564), `xbox_controller.py` (424), `arduino_mouse.py` (260), `ddxoft_mouse.py` (285), `mouse_move.py` (50), `mouse_click.py` (104), `gamepad_input.py` (219), `vk_codes.py`, `key_utils.py`, `admin.py`, `console.py`, `arduino_spoofer.py` (125)

**Purpose.** A facade (`win_utils/__init__.py`) exposes `send_mouse_move(dx, dy, method)` / `send_mouse_click(method)` dispatching to one of six backends: `sendinput`, `mouse_event`, `ddxoft` (driver DLL), `arduino` (Leonardo HID over serial), `makcu` (USB HID proxy over 4 Mbaud serial), and `xbox` (virtual ViGEmBus gamepad). MAKCU is the most complex: async write thread, button-event stream reader, and a reconnect watchdog.

**Assessment.** This subsystem is disciplined about its central hazard — never holding the serial/device lock across a `time.sleep()` — and applies it consistently in MAKCU (`connect`, `_try_open`, `click`, `_query_info`) and Xbox (`move_right_stick` sleeps outside the lock). Module-level singletons all have lazy constructors (fields set to `None`; no DLL load or hardware I/O at import), so importing `win_utils` is side-effect-free and `main.py`'s "only touch ddxoft if selected" guard genuinely holds. The MAKCU button-stream framing is verified against a real hardware capture with proper resync. Findings are minor robustness/observability items.

#### Findings

**REL-05 · Low · `src/win_utils/makcu_mouse.py:305-316, 446-458` · reconnect vs. disconnect race**
`disconnect()` sets `_write_stop`, then joins the reconnect thread with a **1 s** timeout — but a reconnect that is mid-`connect()` can take up to ~2 s (the docstring's own estimate for the DE-AD handshake path). If the join times out, `disconnect()` proceeds to `_close_locked()` while `connect()` (running in the reconnect thread) may still be opening/reopening the port, potentially leaving a live connection behind after `disconnect()` returned. `connect()`'s `_write_stop` checks mitigate but don't fully close this window.
*Fix:* Give the reconnect-thread join a timeout ≥ the worst-case handshake, or make `connect()` abort promptly and re-check `_write_stop` immediately before its final `_try_open`. *Effort: S.*

**DEBT-11 · Low · `src/win_utils/makcu_mouse.py:358-360` · stream-reader raw logging**
`_stream_reader` logs the first 20 raw byte chunks at **INFO** (`"[MAKCU] stream raw bytes: …"`). `logged_chunks` resets to 0 each time the reader thread starts, so every reconnect re-emits up to 20 INFO lines — a flapping USB device produces continuous INFO spam. This looks like leftover bring-up instrumentation.
*Fix:* Demote to DEBUG, or gate behind a one-time module flag. *Effort: S.*

**PERF-03 · Low · `src/win_utils/xbox_controller.py:173-243` · `move_right_stick()`**
Each xbox move pushes the stick, `time.sleep(self.stick_duration)` (~5 ms), then re-centres — **synchronously on the calling (inference) thread**. This caps inference throughput to roughly `1/stick_duration` (~200 Hz) whenever the xbox backend is active, unlike MAKCU which offloads writes to its own thread.
*Fix:* Offload xbox stick pulses to a dedicated writer thread (mirror the MAKCU async-write pattern). *Effort: M.*

**Positive notes.**
- Lock-across-sleep is respected everywhere it matters — the exact discipline the project's docs call out as a recurring hazard.
- MAKCU `move()` uses drop-*oldest* (evict stale, enqueue newest) so the device always gets the freshest delta; and because PID is a closed feedback loop on the crosshair→target error, a rare dropped move is self-corrected by a larger delta next frame rather than accumulating undershoot.
- MAKCU's stream reader verifies both the `km.` prefix **and** the `\r\n>>> ` suffix before trusting a frame, and resyncs on the next `km.` on any misalignment — robust against partial/corrupt serial reads.
- All device singletons are import-safe (lazy init), preserving the intended "don't load high-risk components until selected" startup behaviour.

---

### 4.7 Secondary Inference & Jitter Recorder

**Files:** `src/core/hud_inference.py` (561), `src/core/ocr_inference.py` (468), `src/core/jitter_recorder.py` (313), `src/core/convert_to_engine.py` (411)

**Purpose.** `hud_inference.py` (YOLO11n ONNX weapon detector) and `ocr_inference.py` (PaddleOCR weapon-name reader) both run their model in a **separate child process** (spawn context) fed by a parent-side "feeder" thread, so neither blocks the main inference loop or Qt UI. Selected via `second_inference_mode`. `jitter_recorder.py` records mouse-delta patterns (zero-net-displacement normalised) for the anti-recoil system. `convert_to_engine.py` builds/caches TensorRT engines.

**Assessment.** The process-isolation design is genuinely good: lazy child spawn, idle teardown after 5 s, child-crash detection + respawn with exit-code logging, frame-queue drain-to-latest, result-queue evict-oldest, below-normal child priority, and `ast.literal_eval` (never `eval`) for YOLO `names` metadata. `jitter_recorder._save_pattern` sanitises names to `[A-Za-z0-9_-]`. The one real issue is leftover debug instrumentation.

#### Findings

**DEBT-12 · Low–Medium · `src/core/hud_inference.py:295-297, 446, 470` and `src/core/ocr_inference.py:362-379` · child-process `print()` spam**
Both child processes emit `print()` diagnostics on essentially **every scan**. `hud_inference._postprocess` unconditionally prints a `top3`/`max_score` line per inference (2–10 FPS), and `_child_main` prints ROI shape/mean on change and model-load lines. This is bring-up instrumentation shipped in production: it floods stdout, bypasses the `logging` system entirely (so it ignores levels/handlers), and crosses the process boundary to whatever console the child inherits.
*Root cause:* debug prints never gated or converted to logging.
*Impact:* Continuous console spam and a small per-scan cost when the secondary detector is enabled.
*Fix:* Route through `logging` at DEBUG, or gate behind an env/config debug flag. *Effort: S.*

**DEBT-13 · Low · `src/core/hud_inference.py:130-166` · `_kill_proc()`**
After `terminate()` on a child that didn't join within 1 s, there is no follow-up `join()`/reap, so a slow-dying child can briefly linger. Minor; mp reaps daemon children eventually.
*Fix:* Add a short `join()` after `terminate()`. *Effort: S.*

**Positive notes.**
- Child processes pin `OMP/MKL/OPENBLAS` threads to 1 and drop to below-normal OS priority so the secondary model can't starve the primary loop.
- The feeder detects a dead child (`_proc.is_alive()` false), logs the exit code, and respawns — real fault tolerance, not just best-effort.
- `jitter_recorder._normalize_frames` appends a correction frame so each pattern loop returns to origin (zero net displacement) — correct for an anti-recoil replay, and names are filesystem-sanitised before path interpolation.

---

### 4.8 Web ESP Overlay Server

**Files:** `src/core/esp_server.py` (468), `src/web_overlay/app.js` (512), `index.html` (68), `styles.css` (144)

**Purpose.** An optional LAN browser overlay. Three daemon threads: a stdlib `ThreadingHTTPServer` serving the static client from `web_overlay/`, a hand-rolled RFC 6455 WebSocket accept loop, and a fixed-tick broadcaster (~`web_esp_fps` Hz) that serialises a read-only snapshot of `Config` (detection boxes, FOV/settings, FPS, model name) and pushes it to every connected client. The client renders everything to a `<canvas>`.

**Assessment.** The hand-rolled WebSocket server is careful: bounded handshake read (8 KB cap, 5 s timeout), correct `Sec-WebSocket-Accept` computation, `TCP_NODELAY` to avoid Nagle-induced overlay lag, WS port auto-increment on conflict, orphan-thread cleanup on restart, and dead-client pruning. The client is canvas-only (`fillText`/`fillRect`), so server-supplied strings (model name, capture method) can't cause DOM/XSS injection. Two issues stand out: a broadcast loop that can be frozen by a single slow client, and the deliberate no-auth LAN exposure.

#### Findings

**REL-06 · Medium · `src/core/esp_server.py:291, 353-381` · `_broadcast_loop()` / client sockets**
After the handshake, each client socket is put in **blocking** mode (`conn.settimeout(None)`, line 291), and the broadcast loop calls `conn.sendall(frame)` (line 369) with no per-send timeout. If any one client stops reading (slow device, suspended tab, malicious peer), its TCP send buffer fills and `sendall` **blocks the entire broadcast thread indefinitely** — freezing overlay updates for *every* connected client until that socket errors out. (This branch lacks the per-client send-timeout guard used elsewhere in the project's history.)
*Root cause:* blocking sends in a shared broadcast loop with no timeout and no per-client send isolation.
*Impact:* One slow/stalled/hostile LAN client stalls the ESP for all clients.
*Fix:* Set a short send timeout on client sockets (e.g. `settimeout(2.0)`) and drop a client on `timeout`; or make sends non-blocking and drop clients whose buffer would block; or send per-client on separate threads. *Effort: S–M.*

**SEC-03 · Low–Medium (by design) · `src/core/esp_server.py:14, 235, 301` · no-auth LAN exposure**
Both servers bind `0.0.0.0` with **no authentication**. Any host on the LAN can (a) fetch the static client and (b) open the WebSocket and receive the full live snapshot — detection boxes, crosshair position, model filename, capture method, and FPS. This is documented as intentional, but it is a real information-exposure surface for a tool whose users likely don't want their activity observable to co-located devices, and there is no toggle to bind loopback-only or require a token.
*Fix (optional):* Offer a bind-address setting (default `127.0.0.1`) and/or a shared-secret query token checked at WS handshake. *Effort: M.*

**Positive notes.**
- `_ws_handshake` caps the request read at 8 KB with a 5 s timeout — no unbounded-read DoS on connect.
- The server never parses post-handshake client frames, so there is no WebSocket frame-parser attack surface (one-way streaming by design).
- `start()` tears down orphaned WS/broadcast threads if a prior partial start left them running — no thread/socket leak on retry.
- Canvas-only client eliminates HTML/JS injection from server-supplied strings.

---

### 4.9 DirectShow-Capture-DLL (native C++)

**Files:** `src/src/capture_session.cpp` (350), `capture_api.cpp` (221), `capture_session.h` (123), `common/src/graph_builder.cpp` (347), `sink_filter.cpp` (391), `device_enum.cpp` (93), `media_type_utils.cpp` (39), the matching headers, `python/directshow_capture.py` (335, reference binding), `python/test_capture.py`, `benchmark/`.

**Purpose.** A purpose-built native UVC capture library exposing a small C ABI (`capture_open/start/get_latest_frame/set_crop/stop/close`), owning the DirectShow filter graph and allocator buffer count directly (the reason it exists over `cv2.VideoCapture`), with in-DLL NV12 native cropping. Consumed by Axiom via `dshow_capture_native.py`.

**Assessment.** This is the most rigorous code in the entire project. COM object lifetimes are managed exactly, with long comments documenting the precise leak each `Release()` prevents; reference counts are `std::atomic`; the latest-frame handoff is an O(1) vector swap (no per-poll memcpy); and the memory-safety-critical paths carry real bounds checks. All five documented native-side fixes are present and correct on this branch:
- **D1** (SinkFilter/SinkPin/media-type/IMemAllocator leak) — `~SinkPin` releases all COM refs; `TeardownGraph` releases the sink's own reference. ✔
- **D2** (heap over-read on a short NV12 sample) — `OnFrame` validates `len >= w*h*3/2` before cropping and counts short samples. ✔
- **D3** (negative `biHeight` top-down DIBs) — magnitude taken in both `SelectFormat` and the post-connect readback. ✔
- **D4/D5** (per-poll 3 MB memcpy) — replaced by `m_readBuffer.swap(m_captureBuffer)` with a `m_frameConsumed` gate. ✔

The handle magic-number guard (`kSessionMagic`), even-aligned crop validation, allocator suggest-then-read-back, and the `cbFormat >= sizeof(VIDEOINFOHEADER)` guards before every `VIDEOINFOHEADER*` cast are all correct.

#### Findings

**BUG-11 · Low · `common/src/sink_filter.cpp:57-67, 353-359` · `SinkFilter::State()` / `Receive()`**
`m_state` (a `FILTER_STATE` enum) is written under `m_stateLock` in `Stop`/`Pause`/`Run` but read **without** the lock in `Receive()` (via `m_filter->State()`) and in `GetState()`. This is a formal C++ data race / UB. On x86-64 an aligned enum read is atomic in practice, so the worst realistic effect is processing one extra frame right after `Stop()`.
*Fix:* Make `m_state` `std::atomic<FILTER_STATE>` (matches the already-atomic `m_refCount`). *Effort: S.*

**DEBT-14 · Low · vendored-binary staleness · `src/python/dependencies/directshow_capture.dll`**
The DLL ships as a committed binary in Axiom and can lag the C++ source (the source has no build-stamp export). Axiom's Python side self-heals only the **crop** ABI (first-frame dimension check); a vendored DLL predating the D1/D2/D3 fixes would still leak COM refs / over-read short NV12 samples with no runtime signal. There is no `capture_get_version()` export to detect which build is in place (compounding DEBT-06).
*Fix:* Add a `capture_get_version()`/build-stamp export and have `dshow_capture_native.py` log it at load, and gate risky paths on a minimum version. *Effort: S–M.*

**Positive notes.**
- Every COM-lifetime decision is justified in-comment against the specific leak or use-after-free it avoids — `TeardownGraph`'s release-ours-before-graph ordering analysis is exemplary.
- `SelectFormat` distinguishes "format not advertised" from "resolution not advertised for this format" and surfaces both to the caller as distinct `dsc_result`s.
- MJPEG native crop is correctly *refused* (`DSC_ERR_CROP_NOT_SUPPORTED`) rather than attempting something crop-shaped on compressed bytes.
- The C ABI's memory-ownership contract (pointer valid only until the next call) is documented and honoured on both sides.

---

### 4.10 udp-stream-filter (OBS plugin, native C++)

**Files:** `src/udp_stream_filter.cpp` (743), `src/plugin-support.h`, `CMakeLists.txt`, `buildspec.json`, `vcpkg.json`

**Purpose.** An OBS Studio video-filter plugin that taps the rendered frame, GPU-crops it, JPEG-encodes it on a background thread, and streams it chunked over UDP in the 14-byte-header wire format Axiom's `UdpCapture`/`udp_receiver.py` consume. It is the *sender* half of the UDP capture path.

**Assessment.** Solid, performance-aware OBS code. Locking is cleanly partitioned (`net_mtx` for socket/addr/quality, `enc_mtx` for the pending-frame handoff), the GPU→CPU readback is double-buffered to avoid a per-frame pipeline stall, the FPS gate "banks" fractional time so a requested rate isn't floored to an integer divisor of the render tick, dropped frames are counted and surfaced, and the crop is a true GPU-side sub-rect readback (cost scales with crop size, not source size). It is outbound-only and parses no untrusted input; `inet_pton` validates the destination and the port range is UI-constrained.

#### Findings

**REL-07 · Low · `src/udp_stream_filter.cpp:342-347` · `encode_thread_func()`**
`obs_source_update_properties(f->source)` is called from the **encode (background) thread**, not OBS's UI thread, to refresh the live FPS/drop status text. It's throttled to once per ~4 s, but driving a properties/UI refresh from a non-UI thread is a thread-safety grey area in OBS and could, in the worst case, race the UI while a user has the Filters dialog open.
*Fix:* Marshal the refresh to the UI thread (e.g. via an OBS task queue / `obs_queue_task(OBS_TASK_UI, …)`) instead of calling it directly from the encode thread. *Effort: S–M.*

**DEBT-15 · Low · `src/udp_stream_filter.cpp:165-187, 238, 307` · protocol/doc nits**
(1) The header comment says `frame_id` "increments per source frame", but it is actually assigned from `frames_sent` and advances **per successfully-sent frame** — dropped frames don't increment it. Harmless for reassembly (still monotonic/unique) but the comment misleads. (2) `UDP_MAX_PAYLOAD = 60000` deliberately relies on IP fragmentation, so losing any single IP fragment drops an entire 60 KB chunk; this is a reasonable LAN-only trade but worth stating as a constraint. (3) `send_jpeg_chunked` allocates a fresh `std::vector` packet buffer every frame — a reusable member buffer would remove that per-frame allocation.
*Fix:* Correct the comment; note the fragmentation trade-off; hoist the packet buffer to the filter struct. *Effort: S.*

**Positive notes.**
- The `net_mtx`/`enc_mtx` split is exactly right: the socket can be reconfigured mid-stream without the encode thread ever `sendto`-ing on a closed descriptor or a half-rewritten `sockaddr`.
- Double-buffered `gs_stage_texture`/`gs_stagesurface_map` removes the GPU readback stall that otherwise caps FPS — a real, well-reasoned optimisation.
- Drop accounting distinguishes "encoder is the bottleneck" from "capture gate is pacing us", surfaced in the properties UI.
- Clean `WSAStartup`/`WSACleanup` lifecycle and full resource teardown in `udp_stream_destroy` (thread joined, socket closed, GPU surfaces destroyed under `obs_enter_graphics`).

---

### 4.11 GUI Subsystem

**Files:** `src/gui/fluent_app/window.py` (867), `pages/*.py` (10 pages, ~7,300 lines — `aim_page.py` alone is 2,050), `components/*.py`, `theme_colors.py` (990), `setup_wizard.py` (980), `language_manager.py`, plus `src/gui/overlay.py` (400), `status_panel.py` (978), `disclaimer_dialog.py`.

**Purpose.** A PyQt6 + Fluent-Widgets configuration app. `window.py` hosts a 10-page `NavigationInterface`, applies Windows acrylic/rounded-corner effects, drives i18n refresh and the update check, and mounts the live capture-preview panel. Each page binds to the live `Config` and writes settings on widget callbacks. `overlay.py`/`status_panel.py` are the on-device renderers. `theme_colors.py` centralises light/dark colour pairs. `setup_wizard.py` is the first-run flow.

**Assessment.** For its size this subsystem is well-organised and follows correct Qt patterns. The three long-running operations (TensorRT conversion, UVC device probe, model inspection) each run in a `QThread` subclass and marshal results back to the UI thread via `pyqtSignal` — never touching widgets off-thread — and `capture_page`'s probe worker even wraps `run()` defensively because a raw exception in `QThread.run` can take the whole app down. All 10 pages implement `retranslateUi`, so language switching stays consistent (the recurring stale-text hazard the project docs warn about is handled). The notable issues are the documented theming-debt pattern and the pervasive `Config` coupling.

#### Findings

**DEBT-16 · Low–Medium · `src/gui/**` (≈26 sites) · literal-colour `setStyleSheet`**
There are ~26 `setStyleSheet(...)` calls using literal hex/`rgb()` colours instead of `ThemeColors.*.get()`. Per the project's own theming contract, each is a latent theming bug: it won't adapt when the user toggles light/dark. (The project already tracks this as a baselined GUI invariant, so this quantifies rather than newly discovers it.)
*Fix:* Route each through a `ThemeColors` entry + getter; enforce with the existing GUI-invariant test at a ratcheting baseline. *Effort: M.*

**DEBT-17 · Low (architecture) · `src/gui/fluent_app/pages/*` · direct `Config` mutation coupling**
Every page holds `self._config` and writes fields directly on each widget callback (e.g. `self._config.xbox_sensitivity = value/100.0`). This is the UI face of the `Config` god-object (DEBT-03): there is no settings-controller layer, so validation, persistence timing, and cross-field constraints are scattered across ~7,000 lines of page code and cannot be unit-tested without a live `Config`. `aim_page.py` at 2,050 lines is a maintainability outlier.
*Fix:* Introduce a thin settings-controller/binding layer between widgets and `Config`; split `aim_page` by section. Large; stage it. *Effort: L.*

**DEBT-18 · Low · i18n completeness (pre-existing) · `src/core/language_data/*.json`**
Per project docs, English has ~290 keys and other languages are missing ~20–75 each; missing keys fall back to English (never crash/blank). Not a regression, but a standing completeness gap that grows every time a new `t()` key is added without updating all 10 language files.
*Fix:* A CI check that flags keys present in English but absent elsewhere. *Effort: S (tooling) / ongoing (translation).*

**Positive notes.**
- Long operations use `QThread` + `pyqtSignal` correctly; no cross-thread widget access observed.
- `theme_colors.py` is a clean, centralised light/dark abstraction with a JSON override loaded via `json.load` (no `eval`).
- The setup wizard correctly tracks its own in-progress theme (`self._isDark`) separately from the app's real theme so both previews render right before `applyChosenTheme()` commits.
- `overlay.py` routes every colour through `OverlayColors`/`ThemeColors` getters (the intended pattern), and `status_panel.py` re-reads language strings on its own timer.

---

### 4.12 Test Suite

**Files:** `tests/` — 22 test modules (~5,550 lines) + `conftest.py`.

**Purpose.** Unit/regression coverage for config, capture, inference utils, aiming, mouse backends, ESP, UDP receiver, humanization, semantics, the DLL binding, and more. `conftest.py` injects `src/` onto `sys.path`.

**Assessment.** Where tests exist they are genuinely good: regression-oriented (each targets a specific past fix), they use realistic `SimpleNamespace` configs, and they carefully defer `win_utils`/`win32api` imports inside test bodies with `sys.modules` stubbing so a single module's missing native dep fails only its own tests instead of aborting collection for the whole suite. The gaps are (a) several pure-logic modules with no tests at all and (b) the suite's inability to run clean outside a full Windows environment.

#### Findings

**DEBT-19 · Medium (value) / Low (cost) · `tests/` · untested pure-logic modules**
No direct tests exist for `target_predictor.py` (velocity estimate + max-velocity sanity-cap reset), `kalman_filter.py` (filter update math + `reconfigure`), or `auto_fire.py` (head/body/both hit-testing geometry). These are **zero-dependency, deterministic** modules — the easiest and highest-value things in the codebase to unit-test — yet have no coverage. (Also untested, but harder: `ai_loop.py` threaded loop, `hud_inference`/`ocr_inference` child processes, `jitter_recorder`, `session_utils`.) The absence of `auto_fire` tests is notable given it directly triggers clicks.
*Fix:* Add unit tests for `target_predictor`, `kalman_filter`, and `auto_fire` first (each ~1 h). *Effort: M.*

**DEBT-20 · Low · `tests/` · off-Windows CI signal**
Most test files fail at import/collection without `win32api`/PyQt6/numpy, producing a large standing "failed" baseline (the project docs track it as a fixed number to compare against rather than expecting green). The deferred-import + `sys.modules`-stub pattern is applied in the newer tests but not uniformly, so a bare/Linux CI run gives weak signal and regressions can hide behind the baseline.
*Fix:* Extend the lazy-import/stub pattern to all `win_utils`-touching tests so the runnable subset is green on Linux; run that subset in CI. *Effort: M.*

**Positive notes.**
- `test_ai_aiming.py`, `test_makcu_mouse.py`, `test_screen_capture.py` handle the module-level `win32api` collection hazard correctly and deliberately — this is subtle and done right.
- Tests target real regressions (Y-recoil target-swap reset, sticky-lock interaction, ESP snapshot-against-empty-config robustness), not trivial getters.
- Breadth is good: the network/parse-heavy, fully-testable modules (`udp_receiver`, `esp_server`, `detection_semantics`, `inference`) all have dedicated suites.

---

## 5. Cross-Cutting Findings

These span multiple modules and are the structural themes behind many individual findings.

**X-1 · The `Config` object is both settings store and inter-thread bus (DEBT-03, DEBT-17).**
Almost all shared mutable state lives on one `Config` instance, mutated lock-free from 8+ threads. Correctness rests on CPython GIL atomicity of single attribute writes, which does **not** extend to multi-field consistency — e.g. `latest_all_boxes` and `latest_all_confidences` are written separately and can be read out of sync by the overlay and Web ESP. This is the root cause of the whole class of "publish-not-atomic" issues (also BUG-07). Highest-leverage structural fix in the codebase, but large.

**X-2 · Logging discipline is inconsistent (DEBT-02, DEBT-08, DEBT-11, DEBT-12).**
`print()` / `traceback.print_exc()` are used in hot paths (`key_listener`, `ai_loop` main loop, `auto_fire`, MAKCU stream reader, and both secondary-inference child processes) alongside the `logging` module. Result: fragmented output that ignores levels/handlers, some of it emitted every frame/scan.

**X-3 · External/loaded input is under-validated at the trust boundary (DEBT-04, BUG-04, BUG-05, SEC-01, SEC-02).**
Config/preset JSON, imported presets, downloaded installers, and UDP packets are all trusted more than they should be: numeric config ranges aren't clamped on load, `import_config` assumes a dict, the NDI installer runs without a checksum, and the UDP reassembler has no memory cap.

**X-4 · The audited branch lags the project's own fix history.**
Several defects here (NMS `class_ids` threading, `id(frame)`→`frame_seq`, atomic config writes, Web ESP send timeout, `import_config` dict guard, PID `REFERENCE_DT`, reserved-module headers) are described as fixed on other branches in the project's audit history but are **present on `claude/directshow-dll-mjpeg-nv12-kuglu1`**. Merging/forward-porting those fixes would clear a large fraction of the High/Medium list at once.

**X-5 · Startup DLL-path registration is triplicated (DEBT-21).**
The NVIDIA/TensorRT DLL-directory registration logic is implemented three times with slight variations — `main.py:_register_nvidia_dll_dirs`, `session_utils.py:_register_trt_dll_dirs`, and `model_detect.py:_inject_tensorrt_paths`. They can (and partially do) drift in which sub-packages they scan.
*Fix:* Extract one shared helper. *Effort: S.*

---

## 6. Consolidated Finding Tables

### 6.1 Bug List (correctness)

| ID | Sev | File:line | One-line |
|---|---|---|---|
| BUG-09 | High* | ai_loop.py:625,634 / inference.py:279 | NMS drops `class_ids`; semantic filter reads misaligned classes (*conditional: filter on + multi-class model) |
| BUG-08 | Medium | ai_loop.py:399-406 | `id(frame)` reuse can skip a genuinely new frame |
| BUG-01 | Medium | main.py:331-340 | Restart path skips `request_stop()`; auto-fire join nested wrong |
| BUG-04 | Medium | config_manager.py:249 | `import_config` crashes (AttributeError) on non-dict JSON |
| BUG-07 | Low-Med | screen_capture.py:1896-1900,2233 | Native NV12 frame & dims not published atomically |
| BUG-10 | Low | ai_aiming.py:269 | Y-gate continuity uses raw, not adaptive, IOU threshold |
| BUG-02 | Low | main.py:258,329 | Module-global thread handles mutated without a lock |
| BUG-03 | Low | updater.py:14 | `parse_version` zeros a field on non-numeric suffix |
| BUG-05 | Low | config.py:614 | bool coercion of string `"false"` → `True` |
| BUG-06 | Low | config.py:847 | `detect_range_size` can exceed screen height if `fov_size>height` |
| BUG-11 | Low | sink_filter.cpp:57,353 | `m_state` read without lock (benign UB on x86) |

### 6.2 Performance

| ID | Sev | File | One-line |
|---|---|---|---|
| PERF-01 | Low-Med | ai_loop.py:428 | Tensor queue drops newest (keeps stale) under load |
| PERF-02 | Low | ai_aiming.py:64 | Predictor history deque rebuilt every frame |
| PERF-03 | Low | xbox_controller.py:173 | Xbox move blocks inference thread ~5 ms/move |

### 6.3 Security

| ID | Sev | File | One-line |
|---|---|---|---|
| SEC-02 (=REL-03) | Med-High | udp_receiver.py:157 | Unbounded partial-frame map → LAN UDP memory-exhaustion DoS |
| SEC-01 | Medium | install_cyndilib.py:92-158 | NDI installer downloaded & run with no checksum; PowerShell string-interp |
| SEC-03 | Low-Med | esp_server.py:14,235 | Web ESP binds 0.0.0.0 with no auth; broadcasts app state to LAN |

### 6.4 Reliability

| ID | Sev | File | One-line |
|---|---|---|---|
| REL-02 | Medium | config.py:682 + all writers | Non-atomic writes → total settings loss on ill-timed crash |
| REL-06 | Medium | esp_server.py:369 | One slow WS client blocks the broadcast for all clients |
| REL-03 | Med-High | udp_receiver.py:157 | (see SEC-02) unbounded reassembly buffer |
| REL-01 | Low | key_listener.py:63 | Hotkey error path spams stderr with no backoff |
| REL-04 | Low | kalman_filter.py:88 | `inv(S)` unguarded if `measurement_noise=0` |
| REL-05 | Low | makcu_mouse.py:305,446 | disconnect vs in-flight reconnect race (short join timeout) |
| REL-07 | Low | udp_stream_filter.cpp:346 | OBS properties refresh called off the UI thread |

### 6.5 Technical Debt / Code Smells / Docs

DEBT-01 (Medium: updater points at `iishong0w0` not the shipping repo), DEBT-03/17 (Config god-object + page coupling), DEBT-04 (config validation gaps), DEBT-16 (~26 literal-colour `setStyleSheet`), DEBT-19 (untested pure-logic modules), DEBT-12 (child-process per-scan `print`), plus DEBT-02/05/06/07/08/09/10/11/13/14/15/18/20/21 (see per-module sections). All are Low unless noted.

### 6.6 Dead / Reserved Code

On this branch, these modules have **no first-party importers** and **no role-clarifying header** (the "RESERVED/DEV TOOL" headers from the project's history are not present here), so they read as dead code: `makcu_debug.py`, `makcu_debug_binary.py`, `obs_inspect_filters.py`, `bench_udp.py`, `_bench_udp_sender.py`, `launch_fluent.py`. `makcu_mouse_binary.py` / `makcu_binary_decoder.py` are imported only by the debug scripts and are documented in `CLAUDE.md` as reserved for a future MAKCU firmware. **Recommendation:** add a one-line header stating RESERVED / DEV-TOOL / BENCHMARK status to each (don't delete — they're intentional), so they aren't mistaken for cruft.

### 6.7 Duplicate Code

- NVIDIA/TRT DLL-dir registration triplicated (X-5 / DEBT-21).
- Procedural smart-jitter block duplicated in `ai_aiming.py` (DEBT-10).
- Region-crop/clamp math repeated across `UVCCapture.grab`, `UdpCapture.grab`, `NDICapture.grab` (candidate for a shared helper).

---

## 7. Priority Buckets

### High priority (do first — real impact, small/medium fix)

1. **BUG-09** — thread `class_ids` through NMS (or filter pre-NMS). Correctness of a shipped feature.
2. **SEC-02/REL-03** — cap the UDP reassembly map (count + bytes). Closes a remote DoS.
3. **REL-02** — atomic config/preset/state writes (`tmp`+`os.replace`, keep a `.bak`). Prevents total settings loss.
4. **REL-06** — per-client send timeout in the ESP broadcast loop. Stops one client freezing all.
5. **BUG-08** — replace `id(frame)` with a monotonic `frame_seq`. Removes intermittent frame skips.
6. **DEBT-01** — point the updater at the shipping repo (or document intent). Trivial, user-facing.
7. **SEC-01** — checksum-verify the NDI installer before executing it.

### Medium priority

8. **BUG-01** — unify `start_ai_threads` teardown through `stop_ai_threads`.
9. **BUG-04** — reject non-dict JSON in `import_config`.
10. **DEBT-04** — extend load-time validation to ports, confidences, alphas, ratios, ROI.
11. **PERF-01** — evict-oldest tensor-queue policy.
12. **DEBT-12** — route secondary-inference child diagnostics through logging.
13. **DEBT-19** — unit tests for `target_predictor`, `kalman_filter`, `auto_fire`.
14. **BUG-07** — publish native NV12 frame+dims atomically (or derive dims from the array).
15. **DEBT-16** — migrate literal-colour `setStyleSheet` calls to `ThemeColors`.
16. **DEBT-03** *(structural, stage it)* — split runtime IPC state out of `Config`.

### Low priority

All remaining BUG-02/03/05/06/10/11, PERF-02/03, REL-01/04/05/07, and DEBT-02/05/06/07/08/09/10/11/13/14/15/18/20/21, plus the SEC-03 hardening (bind-address/token) if the no-auth posture is ever revisited.

---

## 8. Risk Assessment

| Area | Risk | Rationale |
|---|---|---|
| Data integrity (user config) | **Medium** | REL-02: a crash during save silently wipes all settings; no atomic write or backup. |
| Remote/network | **Medium** | SEC-02 (UDP DoS) and REL-06 (ESP broadcast stall) are both reachable from the LAN with no auth; SEC-03 leaks app state. All gated on the optional UDP/ESP features being enabled. |
| Aim correctness | **Medium (scoped)** | BUG-09 corrupts the semantic filter's class layer for multi-class models; BUG-08/PERF-01 add occasional stale/dropped frames. Default single-class config is largely unaffected. |
| Native memory safety (C++) | **Low** | The DirectShow DLL and OBS plugin are rigorous; all documented leaks/over-reads are fixed. Only BUG-11 (benign data race) remains. |
| Concurrency | **Medium** | The lock-free `Config` bus (X-1) is a latent hazard; today's symptoms are minor (out-of-sync box/confidence lists) but the pattern doesn't scale. |
| Maintainability | **Medium** | God-object coupling (X-1), a 2,050-line page, triplicated startup logic, and inconsistent logging raise the change-risk on core files. |
| Supply chain | **Low-Medium** | SEC-01: one downloaded-and-executed installer without integrity verification (optional NDI path). |
| Crash/stability | **Low** | Hot loops catch broadly and self-heal; worst case is a 1 s throttle or a backend reinit, not a crash. |

**Overall:** the codebase is **healthy and unusually well-defended in its hardest parts** (capture drivers, native C++, device concurrency). The residual risk concentrates in (a) a handful of correctness bugs on this branch that are already fixed elsewhere, (b) network-facing features that trust the LAN, and (c) the structural `Config`-as-bus coupling.

---

## 9. Future Recommendations

1. **Forward-port the branch's known fixes (X-4).** The single highest-value action: merge the NMS/`class_ids`, `frame_seq`, atomic-write, ESP-timeout, and `import_config`-guard fixes from the project's other branches into the mainline this branch feeds.
2. **Establish a trust boundary layer.** One module that validates/clamps everything crossing in from disk, network, or subprocess (config, presets, UDP headers, downloaded binaries).
3. **Split `Config`.** Extract `RuntimeState` (boxes, counters, FPS, flags) with explicit locks for multi-field groups; leave `Config` as pure persisted settings — improves testability and removes X-1.
4. **Unify logging.** Ban `print` in `src/core`/`src/win_utils` via a lint rule; route child-process diagnostics through a queue to the parent logger.
5. **Add a native build stamp.** `capture_get_version()` in the DLL + a load-time log so vendored-binary drift (DEBT-14) becomes visible.
6. **Grow the deterministic test core.** Start with the pure-logic modules (predictor, Kalman, auto-fire, PID), then a green Linux CI subset via the existing lazy-import pattern.
7. **Optional network hardening.** Default the Web ESP to loopback with an opt-in LAN toggle + token; add a UDP source allowlist/rate cap.

---

## 10. Methodology & Coverage Notes

Every first-party source file across the three repositories was read (entry/bootstrap, config, all five capture backends + the native binding, the full inference and aiming stacks, all six device backends, both secondary-inference paths, the Web ESP server + client, the complete DirectShow C++ DLL, the OBS C++ plugin, the GUI at architecture/pattern level with spot-reads, and the test suite). Vendored third-party trees (`src/python/**`, bundled CPython/site-packages) were excluded except where they affect first-party behaviour. The test suite could not be executed in this sandbox (no `pytest`/`numpy`; the app bundles its own interpreter and is Windows-only), so test findings are from static reading. Findings are evidence-anchored to `file:line`; severities reflect realistic impact under the feature's normal enablement.
