# AXIOM_ROADMAP.md — Phased Implementation Roadmap

Companion to `AXIOM_AUDIT.md`. Every item references a finding ID defined there. Phases are ordered to **maximise impact while minimising risk**: correctness/security first, then robustness, then performance, then structural debt. Effort: **S** ≤ 1 h · **M** ≤ half-day · **L** ≥ 1 day.

Each phase is independently shippable and leaves the tree in a working state.

---

## Phase 0 — Reconcile with `main` (prerequisite, ~1 h)

The audited branch lags `origin/main` (cross-cutting **X-4**). Verification against shipping `main` (see `AXIOM_AUDIT.md` §1a) shows the following are **already fixed on `main`** and need **no work** — only confirm they're present wherever this code is actually deployed: **BUG-09** (NMS `class_ids`), **BUG-08** (`frame_seq`), **PERF-01** (drop-oldest tensor queue), **BUG-04** (`import_config` dict guard), **REL-06** (ESP send timeout), and the reworked shutdown for **BUG-01** (re-verify the model-restart sub-path only).

**Against `main`, Phase 1 therefore reduces to the three items still verified present: 1.2 (SEC-02), 1.3 (REL-02), 1.6 (DEBT-01)** — plus 1.7 (SEC-01) if the NDI path is in scope. Skip 1.1/1.4/1.5.

- [ ] Confirm the deployment target is `main` (not the audited branch); if it *is* the branch, all of Phase 1 applies as written.
- [ ] Spot-check each "already fixed" item in the actual target before crossing it off.

---

## Phase 1 — Correctness & Security (High priority) · ~1–1.5 days

Goal: eliminate the correctness bugs and the two LAN-reachable risks.

| # | Finding | Action | Effort |
|---|---|---|---|
| 1.1 | **BUG-09** | Thread `class_ids` through `non_max_suppression` (return kept indices; reindex boxes/confidences/class_ids), or run the semantic filter before NMS. Add a regression test. | M |
| 1.2 | **SEC-02/REL-03** | Cap `udp_receiver._partial_frames` by entry count **and** total buffered bytes; drop oldest on overflow; sanity-bound `total_chunks`/`total_size`. | M |
| 1.3 | **REL-02** | Atomic writes for `config.json`, `state.json`, presets: write `*.tmp` then `os.replace`; keep one `.bak`. | S–M |
| 1.4 | **REL-06** | Set a send timeout (~2 s) on ESP client sockets; drop a client on `timeout`/error instead of blocking the broadcast loop. | S |
| 1.5 | **BUG-08** | Publish a monotonic `frame_seq` with each frame under `frame_lock`; dedupe on it instead of `id(frame)`. | S |
| 1.6 | **DEBT-01** | Point `updater.REPO_OWNER/NAME` at the shipping repo (or add a comment documenting the upstream-tracking intent). | S |
| 1.7 | **SEC-01** | Verify a pinned SHA-256 of the NDI installer before executing; drop/parameterise the PowerShell fallback. | M |

**Exit criteria:** semantic filter correct on a multi-class model; UDP receiver memory bounded under a flood; killing the app mid-save never loses settings; a paused browser tab doesn't freeze the ESP.

---

## Phase 2 — Robustness & Validation (Medium priority) · ~1 day

| # | Finding | Action | Effort |
|---|---|---|---|
| 2.1 | **BUG-01** | Route `start_ai_threads` teardown through `stop_ai_threads` (single path; `request_stop()` included; un-nest the auto-fire join). | S |
| 2.2 | **BUG-04** | Reject non-dict top-level JSON in `import_config` (`return None`). | S |
| 2.3 | **DEBT-04** | Extend `_validate_*` to clamp ports (1–65535), `min_confidence` (0–1), acrylic alphas (0–255), xbox sens/deadzone, ratios, and `hud_roi_coords` format. | M |
| 2.4 | **BUG-07** | In `UVCCapture.grab` native-NV12 path, derive luma dims from `frame.shape` (or publish frame+dims together under the lock). | S |
| 2.5 | **REL-04** | Clamp Kalman `process/measurement_noise` to a positive floor (or use `pinv`). | S |
| 2.6 | **REL-05** | Give the MAKCU reconnect-thread join a timeout ≥ worst-case handshake; re-check `_write_stop` before the final open. | S |
| 2.7 | **REL-01 / DEBT-08** | Rate-limit/back off the hotkey and main-loop error handlers; route through `logger.exception`. | S |
| 2.8 | **REL-07** | Marshal the OBS `obs_source_update_properties` refresh to the UI thread. | S–M |

**Exit criteria:** hand-edited/imported configs can't crash or silently mis-behave; model hot-swap-while-paused stops cleanly; error paths don't spam.

---

## Phase 3 — Observability & Logging Hygiene (Medium/Low) · ~half-day

| # | Finding | Action | Effort |
|---|---|---|---|
| 3.1 | **DEBT-12** | Route OCR/HUD child diagnostics through `logging` at DEBUG (or gate behind a debug flag). | S |
| 3.2 | **DEBT-02 / X-2** | Replace `print`/`print_exc` in `key_listener`, `ai_loop`, `auto_fire`, MAKCU stream reader with `logging`; de-dupe the double `ctypes` import. | S–M |
| 3.3 | **DEBT-11** | Demote MAKCU raw-byte stream logging to DEBUG / one-time. | S |
| 3.4 | **DEBT-06 / DEBT-14** | Bind `capture_get_short_sample_count`/`capture_get_negotiated_format`; add a `capture_get_version()` export to the DLL and log it at load. | S–M |

**Exit criteria:** no per-frame/per-scan stdout output; a stale vendored DLL is visible in the log.

---

## Phase 4 — Performance (Low, do alongside Phase 2/3) · ~half-day

| # | Finding | Action | Effort |
|---|---|---|---|
| 4.1 | **PERF-01** | Evict-oldest tensor-queue policy (drop stale, enqueue fresh). | S |
| 4.2 | **PERF-02** | Add `VelocityPredictor.reconfigure()`; mutate in place only on change instead of rebuilding the deque per frame. | S |
| 4.3 | **PERF-03** | Offload xbox stick pulses to a dedicated writer thread (mirror MAKCU async-write). | M |
| 4.4 | **DEBT-15** | Hoist the OBS plugin's per-frame packet vector to the filter struct; fix the `frame_id` comment. | S |

---

## Phase 5 — Test Coverage (Medium value) · ~1 day

| # | Finding | Action | Effort |
|---|---|---|---|
| 5.1 | **DEBT-19** | Unit tests for `target_predictor` (velocity + sanity-cap reset), `kalman_filter` (update math + `reconfigure`), `auto_fire` (head/body/both hit-testing), `PIDController`. | M |
| 5.2 | **DEBT-20** | Apply the lazy-import/`sys.modules`-stub pattern to all `win_utils`-touching tests so the runnable subset is green on Linux; wire that subset into CI. | M |
| 5.3 | — | Add regression tests locking in the Phase 1 fixes (NMS class alignment, UDP cap, ESP timeout). | M |

---

## Phase 6 — Structural Debt (Low priority, high long-term value) · staged, ≥ several days

These are large; schedule them deliberately, not opportunistically.

| # | Finding | Action | Effort |
|---|---|---|---|
| 6.1 | **DEBT-03 / X-1** | Extract `RuntimeState` (boxes, counters, FPS, flags) from `Config` with explicit locks for multi-field groups (boxes+confidences); leave `Config` as pure persisted settings. | L |
| 6.2 | **DEBT-17** | Introduce a settings-controller/binding layer between GUI widgets and `Config`; split `aim_page.py` (2,050 lines) by section. | L |
| 6.3 | **DEBT-16** | Migrate the ~26 literal-colour `setStyleSheet` calls to `ThemeColors`; enforce with a ratcheting invariant test. | M |
| 6.4 | **X-5 / DEBT-21** | Extract one shared NVIDIA/TRT DLL-registration helper used by `main.py`, `session_utils.py`, `model_detect.py`. | S |
| 6.5 | **DEBT-06.7 (dead/reserved code)** | Add a one-line RESERVED/DEV-TOOL/BENCHMARK header to each unimported standalone module (don't delete). | S |
| 6.6 | **DEBT-10 / 6.7 dup** | Extract the duplicated procedural-jitter block and the repeated region-crop math into helpers. | S |

---

## Dependency / sequencing notes

- **Phase 0 first** — it may satisfy much of Phase 1 for free and avoids re-implementing solved problems.
- Phases 1–4 are independent of each other and of Phase 6; do 1 before shipping anything network-facing.
- Phase 5.3 depends on Phase 1 landing.
- Phase 6.1 (`RuntimeState` split) makes BUG-07 and the box/confidence sync issue disappear structurally — but is large, so the targeted fixes in Phases 1–2 should land first regardless.

## Suggested first PR (smallest set that most reduces risk)

BUG-09 (1.1) + SEC-02 (1.2) + REL-02 (1.3) + REL-06 (1.4) + DEBT-01 (1.6). All are S–M, touch disjoint files, and together close the top correctness bug and every Medium+ network/data-loss risk.
