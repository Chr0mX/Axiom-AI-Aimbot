# AXIOM_SUMMARY.md — Executive Summary

One-page companion to `AXIOM_AUDIT.md` (full detail) and `AXIOM_ROADMAP.md` (phased plan).

**Scope:** every first-party source file in `chr0mx/axiom-ai-aimbot` (~34k lines Python) plus its two native C++ companions — `directshow-capture-dll` and `udp-stream-filter` (~4.9k lines). Branch: `claude/directshow-dll-mjpeg-nv12-kuglu1`. Tests were read, not run (Windows-only app; no `pytest`/`numpy` in the audit sandbox).

> **⚠️ Reconciliation with `main`:** the audited branch **lags `origin/main`**. Verified against shipping `main`, these top findings are **already fixed**: BUG-09 (NMS `class_ids`), BUG-08 (`frame_seq`), PERF-01 (tensor queue), BUG-04 (`import_config`), REL-06 (ESP timeout), and the shutdown path (BUG-01). **Still present on `main` and actionable: SEC-02 (UDP DoS cap), REL-02 (atomic config writes), DEBT-01 (updater repo), DEBT-12 (child `print` spam).** Native C++ findings are in separate repos and stand as written. See `AXIOM_AUDIT.md` §1a for the full table.

---

## Current health: **GOOD**

Axiom is a mature, feature-dense Windows real-time computer-vision aim-assist app. Its **hardest components are its best-built**:

- **Screen capture** (5 backends) is defensively engineered against real capture-driver lies (FOURCC/CONVERT_RGB readback checks, measured-vs-requested FPS warnings, crop-before-convert, self-healing native crop, throttled reinit).
- **Native C++** (DirectShow DLL + OBS plugin) is exemplary: atomic refcounts, every COM lifetime justified in-comment, clean lock partitioning, and all documented memory-safety fixes (leaks, short-buffer over-read, negative `biHeight`, per-poll memcpy) present and correct.
- **Device backends** (MAKCU/xbox/arduino/ddxoft) consistently respect the "never hold a device lock across a sleep" rule and lazy-load hardware so startup stays safe.

**No Critical findings.** Totals: **11** correctness bugs (1 High-conditional · 3 Medium · 7 Low), **3** performance, **3** security (1 Med-High · 1 Medium · 1 Low), **7** reliability (2 Medium), **~21** technical-debt items.

---

## Highest-risk issues (act on these)

| Risk | Finding | Why it matters |
|---|---|---|
| 🟠 Semantic filter corruption | **BUG-09** | NMS drops `class_ids`, so the false-positive filter reads the wrong class per box (multi-class models). Wrong targets dropped/kept. |
| 🟠 LAN memory-exhaustion DoS | **SEC-02 / REL-03** | UDP receiver's reassembly map is unbounded; a malformed/hostile LAN sender can OOM the app. |
| 🟠 Total settings loss | **REL-02** | Every config/preset write is a non-atomic truncate-write; a crash mid-save silently resets **all** settings. |
| 🟠 ESP broadcast freeze | **REL-06** | One slow/suspended browser client blocks overlay updates for every client (blocking `sendall`, no timeout). |
| 🟡 Intermittent frame skips | **BUG-08** | `id(frame)` reuse makes the preprocess worker skip genuinely-new frames. |
| 🟡 Unverified installer exec | **SEC-01** | NDI runtime is downloaded and run with no checksum (optional path). |

> **Key context:** several of these (BUG-09, BUG-08, REL-02, REL-06) are described as **already fixed on other project branches** — they are simply present on the branch audited. Forward-porting is the single highest-value move.

---

## Quick wins (small fix, outsized value)

1. **Point the updater at the shipping repo** — `updater.py` currently checks `iishong0w0/Axiom-AI-Aimbot`, not `chr0mx/...` (DEBT-01). One-line fix.
2. **Cap the UDP reassembly map** (SEC-02) — closes a remote DoS.
3. **Atomic config writes** (REL-02) — `tmp`+`os.replace`+`.bak`.
4. **Per-client ESP send timeout** (REL-06).
5. **`frame_seq` instead of `id(frame)`** (BUG-08).
6. **Thread `class_ids` through NMS** (BUG-09).

A single PR of #1–#6 (all S–M, disjoint files) removes the top correctness bug and every Medium-or-higher network/data-loss risk.

---

## Structural themes (longer-term)

- **`Config` is a god-object doubling as the cross-thread bus** — mutated lock-free from 8+ threads; correctness rests on GIL atomicity that doesn't cover multi-field consistency (e.g. boxes vs. confidences can be read out of sync). Highest-leverage refactor; stage it.
- **Inconsistent logging** — `print`/`traceback.print_exc` in hot paths and per-scan in the OCR/HUD child processes; unify on `logging`.
- **Under-validated trust boundaries** — config/preset JSON, UDP packets, and a downloaded installer are trusted more than they should be.
- **Test gaps in the easy places** — pure-logic modules (`target_predictor`, `kalman_filter`, `auto_fire`, PID) have zero unit tests despite being deterministic and dependency-free.

---

## Recommendation

Ship **Roadmap Phase 0 (forward-port known fixes) + Phase 1 (correctness & security)** first — roughly 1.5–2 days of work that clears the entire High-priority list. Then Phases 2–3 (validation + logging hygiene) for robustness, Phase 5 (deterministic tests) to lock it in, and schedule Phase 6 (the `Config` split) deliberately. The foundation is strong; the work is targeted cleanup and forward-porting, not rearchitecture.
