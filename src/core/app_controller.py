"""Shared application-logic boundary between the Qt GUI and any other client.

Sandbox note: `stop_ai_threads`/`pause_ai_inference`/`resume_ai_inference`/
`start_ai_threads` depend (transitively, via `.ai_loop`/`.auto_fire`/
`.session_utils`) on `win32api`/`onnxruntime`/`cv2`, none of which are
installed on non-Windows dev boxes — the same constraint CLAUDE.md already
documents for `ai_loop.py` itself. Those imports are deferred to inside each
function body (not at module top level) specifically so this module stays
importable — and `set_always_aim`/the module import itself stay testable —
on a machine without those packages; only actually *calling* the
thread-lifecycle functions requires them, exactly like calling into
`ai_loop.py` already does.

Every function here is plain: it takes `config` (and whatever else it
genuinely needs) as an explicit argument, never imports anything from `gui`/
PyQt6, and never touches a QWidget. A Qt slot and a web API route handler
are both expected to call the *same* function here rather than duplicating
"decide what to do" logic on each side — see CLAUDE.md's Web Control section
for the split this was extracted from (e.g. keys_page.py's
`_onAlwaysAimChanged` used to inline both the `always_aim`/
`idle_detect_enabled` decision and the widget refresh in one method; the
decision half now lives here as `set_always_aim()`).

Threading model
---------------
`Config` itself has no lock (see config.py) — every existing single-field
read/write in this codebase (GUI slots, ai_loop.py, esp_server.py,
key_listener.py) is a raw, unsynchronized `getattr`/`setattr`, relying on
CPython's GIL to make one scalar assignment atomic. That's still correct
here and is *not* what `_multi_field_lock` below is for.

The one real risk is a *multi-field* command — writing more than one
`Config` attribute as a single logical unit — being observed half-applied
by a concurrent reader (ai_loop.py's per-frame poll, esp_server.py's
broadcast tick). `_multi_field_lock` guards exactly those functions, and
only those.

It is a **module-level** lock, not a `Config` attribute, deliberately:
`ConfigManager.preview_config_changes()` does `copy.deepcopy(config_instance)`
(see config_manager.py) to build a dry-run preview without touching the
real config, and `copy.deepcopy()` on a raw `threading.Lock` raises
`TypeError: cannot pickle '_thread.lock' object` — confirmed directly
(`copy.deepcopy` walks into every instance attribute, and a lock has no
`__deepcopy__`/`__reduce__` of its own). Since there is exactly one live
`Config` instance for the app's whole lifetime today (created once in
main.py, shared by reference into every GUI page and the AI loop), a
module-level lock here is equivalent in practice to a per-instance one,
without that landmine.
"""
from __future__ import annotations

import logging
import os
import queue
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import Config

logger = logging.getLogger(__name__)

# .../src/core/app_controller.py -> .../src/core -> .../src -> repo root.
# Mirrors main.py's own `project_root = os.path.dirname(src_dir)` derivation,
# just computed one directory deeper.
_this_dir = os.path.dirname(os.path.abspath(__file__))
_src_dir = os.path.dirname(_this_dir)
project_root = os.path.dirname(_src_dir)

# ---------------------------------------------------------------------------
# AI thread lifecycle — moved verbatim from main.py (start_ai_threads,
# stop_ai_threads, pause_ai_inference, resume_ai_inference, and the lock/
# handles they share). Behavior is unchanged; only the location moved, so
# that a web route can call these the same way main.py's own startup/
# shutdown code already does — previously nothing outside main.py could,
# since these lived as module-private functions there.
# ---------------------------------------------------------------------------

ai_thread: threading.Thread | None = None
auto_fire_thread: threading.Thread | None = None

# Serializes every read-check-join-reassign sequence that touches ai_thread/
# auto_fire_thread. Without it, a GUI "reload model" call (start_ai_threads)
# racing a concurrent stop_ai_threads() call (e.g. from an installer worker
# thread, or now a web-triggered stop) — or two rapid reloads — could
# interleave: one caller's join() could observe a thread object the other
# had already replaced, leaving a stale handle abandoned without ever being
# joined, or briefly running two inference threads at once. Held across the
# whole lifecycle transition (including model load in start_ai_threads)
# rather than split into smaller critical sections — these calls are
# infrequent, not hot-path code, so serializing the full transition is the
# correct trade.
_ai_threads_lock = threading.Lock()

# Guards multi-field Config "commands" only — see the module docstring.
_multi_field_lock = threading.Lock()

# Guards connect_makcu() against two overlapping connect attempts racing on
# the same port (e.g. a double-click, or the Qt GUI and a web client both
# clicking connect within the same ~1-2s handshake window) — MakcuMouse's
# own connect() has no such guard. Not needed for disconnect_makcu(): its
# teardown path is idempotent-safe to call repeatedly.
_makcu_connect_lock = threading.Lock()


def stop_ai_threads(config: "Config", join_timeout: float = 3.0) -> None:
    """Stop AI inference and auto-fire threads without closing the application.

    Safe to call from the GUI (e.g. before a CUDA installer runs in a worker
    thread) or from a web control route. The UI remains responsive because
    this only touches background daemon threads.

    After this call:
    - config.Running is False
    - Both AI threads have been joined (or timed out)
    - The ONNX session held by the AI thread goes out of scope and will be
      garbage-collected, releasing its GPU/CPU resources.
    """
    from .session_utils import inference_controller as _inference_controller

    global ai_thread, auto_fire_thread

    config.Running = False
    _inference_controller.request_stop()

    with _ai_threads_lock:
        if ai_thread is not None and ai_thread.is_alive():
            ai_thread.join(timeout=join_timeout)
            if ai_thread.is_alive():
                logger.warning("AI thread did not stop within %.1fs", join_timeout)

        if auto_fire_thread is not None and auto_fire_thread.is_alive():
            auto_fire_thread.join(timeout=join_timeout)

    _inference_controller.clear_stop()


def pause_ai_inference(config: "Config") -> None:
    """Pause AI inference cooperatively without stopping threads.

    The inference loop will sleep on its next iteration. Call
    resume_ai_inference() to continue. Prefer stop_ai_threads() when you
    need to release GPU resources (e.g. before a CUDA upgrade install).
    """
    from .session_utils import inference_controller as _inference_controller

    config.inference_paused = True
    _inference_controller.pause()


def resume_ai_inference(config: "Config") -> None:
    """Resume AI inference after a pause_ai_inference() call."""
    from .session_utils import inference_controller as _inference_controller

    config.inference_paused = False
    _inference_controller.resume()


def start_ai_threads(
    config: "Config",
    overlay_boxes_queue: queue.Queue,
    overlay_confidences_queue: queue.Queue,
    auto_fire_boxes_queue: queue.Queue,
    model_path: str,
) -> bool:
    """Load a model and start/restart the AI inference + auto-fire threads.

    Args:
        config: the live Config instance.
        overlay_boxes_queue / overlay_confidences_queue: feed the in-game
            overlay's detection-box rendering.
        auto_fire_boxes_queue: feeds auto_fire_loop.
        model_path: path to a .onnx model, absolute or relative to the
            project root.

    Returns:
        True if the threads started successfully.
    """
    from .session_utils import build_provider_list, optimize_onnx_session
    from .ai_loop import ai_logic_loop
    from .auto_fire import auto_fire_loop

    global ai_thread, auto_fire_thread

    with _ai_threads_lock:
        # Stop any existing threads first.
        if ai_thread is not None and ai_thread.is_alive():
            config.Running = False
            ai_thread.join(timeout=3.0)
            if auto_fire_thread is not None and auto_fire_thread.is_alive():
                auto_fire_thread.join(timeout=3.0)
            if ai_thread.is_alive():
                logger.warning("AI thread did not stop within 3s — continuing anyway")

        config.Running = True

        # Only .onnx models are supported.
        if not model_path.endswith('.onnx'):
            logger.error("Unsupported model format (expected .onnx): %s", model_path)
            return False

        if not os.path.isabs(model_path):
            model_path = os.path.join(project_root, model_path)

        if not os.path.exists(model_path):
            logger.error("Model file does not exist: %s", model_path)
            return False

        model = None
        try:
            import onnxruntime as ort

            providers = build_provider_list(config)
            logger.info("Attempting to load ONNX providers: %s", providers)

            session_options = optimize_onnx_session(config)
            if session_options:
                model = ort.InferenceSession(model_path, providers=providers, sess_options=session_options)
            else:
                model = ort.InferenceSession(model_path, providers=providers)

            actual_providers = model.get_providers()
            if actual_providers:
                config.current_provider = actual_providers[0]
                logger.info("Model loaded using provider: %s", actual_providers[0])

                requested_backend = getattr(config, "inference_backend", "auto")
                if requested_backend == "cuda" and config.current_provider != "CUDAExecutionProvider":
                    logger.warning(
                        "CUDA backend requested but actual provider is %s. Check "
                        "onnxruntime-gpu / NVIDIA driver / CUDA-cuDNN compatibility.",
                        config.current_provider,
                    )
            else:
                logger.warning("Could not determine active provider")
                config.current_provider = providers[0] if providers else 'CPUExecutionProvider'
        except Exception as e:
            logger.error("Failed to load ONNX model: %s", e)
            logger.error("Check that the matching ONNX Runtime backend (CUDA/DirectML/CPU) is installed")
            return False

        ai_thread = threading.Thread(
            target=ai_logic_loop,
            args=(config, model, 'onnx', overlay_boxes_queue, overlay_confidences_queue, auto_fire_boxes_queue),
            daemon=True,
        )
        auto_fire_thread = threading.Thread(
            target=auto_fire_loop,
            args=(config, auto_fire_boxes_queue),
            daemon=True,
        )

        ai_thread.start()
        auto_fire_thread.start()
        return True


# ---------------------------------------------------------------------------
# Single-field toggles that already have a real GUI control — exposed here
# unchanged in effect, just callable from anywhere (a web route included)
# instead of only from a Qt slot.
# ---------------------------------------------------------------------------

def set_always_aim(config: "Config", enabled: bool) -> None:
    """Enable/disable `always_aim`, with its coupled `idle_detect_enabled` reset.

    Extracted from keys_page.py's `_onAlwaysAimChanged`, which used to
    inline this same two-field write directly in a Qt slot alongside a
    widget-visibility refresh. The coupling — turning always_aim on also
    turns idle_detect_enabled off — is application logic, not GUI logic: a
    web-issued toggle needs the identical side effect, so it has to live
    here rather than staying trapped inside a Qt-only method.

    This is the first (and so far only) multi-field Config command exposed
    to a caller outside the GUI thread, so it's the one that actually needs
    `_multi_field_lock` — a single-field `setattr` elsewhere in this
    codebase relies on the GIL for atomicity and doesn't take a lock at all.
    """
    with _multi_field_lock:
        config.always_aim = bool(enabled)
        if enabled:
            config.idle_detect_enabled = False


# ---------------------------------------------------------------------------
# MAKCU device connect/disconnect — thin wrappers around already GUI-free
# win_utils.makcu_mouse functions, so a web route and keys_page.py's own
# Connect/Disconnect button share the exact same config-read/guard logic.
# ---------------------------------------------------------------------------

def connect_makcu(config: "Config") -> bool:
    """Connect to the MAKCU device using the configured port/baud.

    Reads `config.makcu_com_port`/`config.makcu_baud_rate` rather than
    taking them as parameters — a web caller has no combo boxes to read
    them from the way `keys_page.py`'s own `_onMakcuConnectToggle` does
    (`self.makcuComPortCombo`/`self.makcuBaudCombo`), so this is the one
    place both callers can share the same already-persisted values.

    `win_utils.makcu_mouse.connect_makcu()` is already GUI-free and
    documented as safe to call off the GUI thread (its own lock is never
    held across a sleep) — this wrapper exists only to own the config-read
    and the empty-port guard, and to serialize concurrent connect attempts
    via `_makcu_connect_lock`, not to add any new hardware logic.
    """
    com_port = str(getattr(config, "makcu_com_port", "") or "")
    if not com_port:
        logger.warning("[MAKCU] connect requested with no COM port configured")
        return False
    baud = int(getattr(config, "makcu_baud_rate", 4_000_000) or 4_000_000)

    with _makcu_connect_lock:
        try:
            from win_utils.makcu_mouse import connect_makcu as _connect_makcu
        except ImportError:
            logger.error("[MAKCU] win_utils.makcu_mouse not importable")
            return False
        return bool(_connect_makcu(com_port, baud))


def disconnect_makcu(config: "Config") -> None:
    """Disconnect the MAKCU device, if connected.

    `config` is accepted (and currently unused) to match this module's
    "plain functions, config first" convention and keep the signature
    stable if a future revision needs to read/write Config here too.
    """
    try:
        from win_utils.makcu_mouse import disconnect_makcu as _disconnect_makcu
    except ImportError:
        logger.error("[MAKCU] win_utils.makcu_mouse not importable")
        return
    _disconnect_makcu()
