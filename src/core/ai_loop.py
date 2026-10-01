"""Main loop for AI inference and mouse control"""

from __future__ import annotations

import ctypes
import os
import queue
import threading
import time
import traceback
from typing import TYPE_CHECKING

import logging

import cv2
import numpy as np

from win_utils import is_key_pressed

logger = logging.getLogger(__name__)

from . import ai_aiming
from .ai_aiming import process_aiming
from .ai_loop_state import LoopState
from .detection_semantics import filter_detections_by_target_class, sync_detection_class_names_from_backend
from .ai_loop_utils import (
    aim_start_delay_elapsed,
    apply_cam_shift_deadzone,
    calculate_detection_region,
    clear_queues,
    compute_effective_fov,
    filter_boxes_by_fov,
    get_capture_dimensions,
    put_latest,
    reduce_boxes_for_single_target,
    update_crosshair_position,
    update_queues,
)
from .inference import (
    PIDController,
    detect_end2end_output,
    non_max_suppression,
    postprocess_outputs,
    preprocess_image,
)
from .session_utils import inference_controller
from .screen_capture import (
    _capture_backend_is_stale,
    _cleanup_capture,
    _detect_active_capture_method,
    capture_frame,
    initialize_screen_capture,
    reinitialize_if_method_changed,
)

if TYPE_CHECKING:
    import onnxruntime as ort

    from .config import Config


# EMA smoothing factor for latency stats (internal; not user-configurable).
_LATENCY_STATS_ALPHA = 0.2

# Per-axis noise floor (screen px) below which a frame's phase-correlation
# shift is treated as measurement noise rather than real camera motion, and
# is not accumulated into state.cam_drift_x/y. See
# ai_loop_utils.apply_cam_shift_deadzone()'s docstring for why this only
# gates the drift integral and not state.cam_shift_x/y itself.
_CAM_DRIFT_DEADZONE_PX = 0.5

# Threaded backends (uvc/udp/ndi) keep handing back their last decoded frame
# after the source stops delivering. The backend-reinit watchdog only fires
# after screen_capture._CAPTURE_STALE_TIMEOUT_SECONDS; until then, re-inferring
# that frozen frame re-applies the same PID correction every iteration and
# drags the mouse. Well above any real source's frame period (a 5 fps source
# is 0.2 s), so it only trips on genuine source loss.
_SOURCE_FRAME_MAX_AGE_S = 0.5

_PREPROCESS_ERROR_LOG_INTERVAL_S = 5.0

# Identity of the ai_logic_loop() that currently owns the pipeline. config.Running
# is shared by every loop, so a loop that outlived start_ai_threads()'s join
# timeout would otherwise resume the moment the new loop sets Running back True.
_active_loop_token: object | None = None


def _makcu_button_held(mm, name: str) -> bool:
    """MAKCU stream button state for a config button name."""
    if name == 'rmb':
        return mm.rmb_held
    if name == 'mmb':
        return mm.mmb_held
    if name == 'side1':
        return mm.side1_held
    if name == 'side2':
        return mm.side2_held
    return mm.lmb_held


def _sync_makcu_features(config: Config) -> None:
    """Push config-driven MAKCU features to the device — cheap and idempotent,
    so a change from any UI (Qt, Web Control, a loaded preset) lands within
    one method-check interval."""
    try:
        from win_utils import set_device_hotkeys
        from win_utils.makcu_mouse import makcu_mouse as _mm
        use_makcu = getattr(config, 'mouse_move_method', '') == 'makcu'
        set_device_hotkeys(use_makcu and bool(getattr(config, 'makcu_device_hotkeys', False)))
        if use_makcu and getattr(config, 'makcu_mouse_spread_enabled', False):
            _mm.request_mouse_spread(int(getattr(config, 'makcu_mouse_spread', 0)))
        else:
            _mm.request_mouse_spread(None)
    except Exception as exc:
        logger.debug("[AI Loop] MAKCU feature sync failed: %s", exc)


def _set_makcu_move_lock(config: Config, active: bool) -> None:
    """Assert/release the MAKCU physical-movement lock. Must run every frame
    while active — the driver auto-releases a lock that stops being
    re-asserted."""
    wanted = (active and getattr(config, 'makcu_lock_physical_move', False)
              and getattr(config, 'mouse_move_method', '') == 'makcu')
    try:
        from win_utils.makcu_mouse import makcu_mouse as _mm
        if wanted or _mm.physical_move_locked:
            _mm.set_physical_move_lock(wanted)
    except Exception as exc:
        logger.debug("[AI Loop] MAKCU move lock failed: %s", exc)


def _probe_model_input_size(session, abs_model_path: str) -> int:
    """Return spatial H (=W) from a loaded ORT session, 0 if not determinable.

    TRT EP exposes dims as None even for static-shape models, so fall back to
    a throwaway CPUExecutionProvider session reading the ONNX file directly.
    """
    import onnxruntime as _ort
    shape = session.get_inputs()[0].shape
    if len(shape) >= 4:
        try:
            h = int(shape[2])
            if h > 0:
                return h
        except (TypeError, ValueError):
            pass
    try:
        probe = _ort.InferenceSession(abs_model_path, providers=["CPUExecutionProvider"])
        ps = probe.get_inputs()[0].shape
        if len(ps) >= 4 and isinstance(ps[2], int) and ps[2] > 0:
            return ps[2]
    except Exception:
        pass
    return 0


_hot_swap_skip_logged: tuple | None = None


def _try_hot_swap_model(
    config: Config,
    model: ort.InferenceSession,
    current_model_path: str,
    current_backend: str,
    current_dml_fallback: bool,
    current_fp16: bool,
):
    """Try hot-swapping ONNX model when model/provider related settings change."""

    config_backend = str(getattr(config, "inference_backend", "auto")).lower()
    config_dml_fallback = bool(getattr(config, "dml_cpu_fallback", True))
    config_fp16 = bool(getattr(config, "trt_fp16_enabled", False))

    should_reload = (
        config.model_path != current_model_path
        or config_backend != current_backend
        or config_dml_fallback != current_dml_fallback
        or config_fp16 != current_fp16
    )
    if not should_reload:
        return (
            model, current_model_path, model.get_inputs()[0].name,
            current_backend, current_dml_fallback, current_fp16,
        )

    new_model_path = config.model_path
    if not os.path.isabs(new_model_path):
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        abs_model_path = os.path.join(project_root, new_model_path)
    else:
        abs_model_path = new_model_path

    if not (os.path.exists(abs_model_path) and abs_model_path.endswith('.onnx')):
        logger.warning("[Model HotSwap] Invalid path or file not found: %s", abs_model_path)
        config.model_path = current_model_path
        config.trt_fp16_enabled = current_fp16
        return (
            model, current_model_path, model.get_inputs()[0].name,
            current_backend, current_dml_fallback, current_fp16,
        )

    # A TensorRT session with no cached .engine yet compiles one synchronously
    # inside InferenceSession(...) — a 1-5 minute call. This function runs
    # once per frame on the main inference thread, so doing that inline would
    # freeze the whole aim loop with zero progress feedback. The GUI's
    # model/backend selectors (model_page.py) already check this before ever
    # writing a combination like that to config, redirecting to the Convert
    # tab instead — this is a safety net for paths that bypass the GUI (a
    # loaded preset, a hand-edited config.json, a race with the Convert
    # worker) rather than the primary UX.
    from .session_utils import needs_trt_build
    if needs_trt_build(config, abs_model_path):
        global _hot_swap_skip_logged
        skip_key = (abs_model_path, config_fp16, config_backend)
        if _hot_swap_skip_logged != skip_key:
            _hot_swap_skip_logged = skip_key
            logger.warning(
                "[Model HotSwap] Skipping swap to %s (%s) — no cached TensorRT engine yet "
                "(building one inline would block the inference loop for 1-5 min). "
                "Convert it first via the Convert tab.",
                os.path.basename(abs_model_path),
                "FP16" if config_fp16 else "FP32",
            )
        config.model_path = current_model_path
        # Leave trt_fp16_enabled on the user's choice. Reverting it would
        # snap the precision combo back while the Convert tab is building
        # that precision. The loaded session stays on current_fp16 until
        # the matching engine exists and this function swaps it in.
        return (
            model, current_model_path, model.get_inputs()[0].name,
            current_backend, current_dml_fallback, current_fp16,
        )

    try:
        import onnxruntime as _ort

        from .session_utils import build_provider_list, optimize_onnx_session

        providers = build_provider_list(config)
        session_options = optimize_onnx_session(config)
        if session_options:
            new_model = _ort.InferenceSession(abs_model_path, providers=providers, sess_options=session_options)
        else:
            new_model = _ort.InferenceSession(abs_model_path, providers=providers)

        input_name = new_model.get_inputs()[0].name
        _detected_size = _probe_model_input_size(new_model, abs_model_path)
        if _detected_size:
            config.model_input_size = _detected_size
            logger.info("[Model HotSwap] Auto-detected input size: %d", _detected_size)
        actual_providers = new_model.get_providers()
        if actual_providers:
            config.current_provider = actual_providers[0]
        logger.info("[Model HotSwap] Switched to: %s / %s", os.path.basename(abs_model_path), config_backend)
        return new_model, new_model_path, input_name, config_backend, config_dml_fallback, config_fp16
    except Exception as e:
        logger.error("[Model HotSwap] Load failed: %s — continuing with current model", e)
        config.model_path = current_model_path
        config.trt_fp16_enabled = current_fp16
        return (
            model, current_model_path, model.get_inputs()[0].name,
            current_backend, current_dml_fallback, current_fp16,
        )


def _sleep_precise(seconds: float) -> None:
    """Sleep with better precision for very short intervals on Windows."""

    if seconds <= 0:
        return

    if seconds >= 0.002:
        time.sleep(seconds)
        return

    # Reduce CPU spin on sub-2ms waits:
    # 1) cooperatively yield while remaining time is still relatively large
    # 2) only busy-wait in a very small tail window for precision
    deadline = time.perf_counter() + seconds
    spin_threshold = 0.0002  # 0.2ms spin window

    while True:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            break

        if remaining > spin_threshold:
            # Yield the timeslice to avoid burning a full CPU core.
            # Keep a tiny safety margin so final timing still relies on perf_counter.
            sleep_for = max(0.0, remaining - spin_threshold)
            if sleep_for >= 0.001:
                time.sleep(sleep_for)
            else:
                time.sleep(0)


_PRIORITY_MAP = {
    "normal":        0,
    "above_normal":  1,
    "high":          2,
    "time_critical": 15,
}

# Windows PROCESS priority classes — what Task Manager shows
_PROCESS_CLASS_MAP = {
    "normal":        0x00000020,  # NORMAL_PRIORITY_CLASS
    "above_normal":  0x00008000,  # ABOVE_NORMAL_PRIORITY_CLASS
    "high":          0x00000080,  # HIGH_PRIORITY_CLASS
    "time_critical": 0x00000080,  # HIGH_PRIORITY_CLASS (REALTIME_PRIORITY_CLASS is unsafe)
}


def _set_thread_priority(level: str) -> None:
    if os.name != 'nt':
        return
    try:
        ctypes.windll.kernel32.SetThreadPriority(
            ctypes.windll.kernel32.GetCurrentThread(),
            _PRIORITY_MAP.get(level, 2),
        )
    except Exception:
        pass


def _set_process_priority(level: str) -> None:
    if os.name != 'nt':
        return
    try:
        ctypes.windll.kernel32.SetPriorityClass(
            ctypes.windll.kernel32.GetCurrentProcess(),
            _PROCESS_CLASS_MAP.get(level, 0x00000080),
        )
    except Exception:
        pass


def _set_windows_timer_resolution_1ms(enable: bool) -> bool:
    """Enable/disable 1ms timer resolution on Windows. Returns success status."""

    if os.name != 'nt':
        return False

    try:
        winmm = ctypes.WinDLL('winmm')
        if enable:
            return winmm.timeBeginPeriod(1) == 0
        return winmm.timeEndPeriod(1) == 0
    except Exception:
        return False


def ai_logic_loop(
    config: Config,
    model: ort.InferenceSession,
    model_type: str,
    overlay_boxes_queue: queue.Queue,
    overlay_confidences_queue: queue.Queue,
    auto_fire_boxes_queue: queue.Queue | None = None,
) -> None:
    """AI 推理和滑鼠控制的主要循環"""

    global _active_loop_token
    loop_token = object()
    _active_loop_token = loop_token

    def _is_active() -> bool:
        return config.Running and _active_loop_token is loop_token

    input_name = model.get_inputs()[0].name
    is_end2end = detect_end2end_output(model)
    logger.info("[AI Loop] Model output format: %s", "end-to-end [N, 6]" if is_end2end else "raw grid")

    # Auto-detect model input size from the initial session (same probe as hot-swap).
    # Ensures 320/416/448/512/640 models all work without manual config.
    _init_model_path = config.model_path
    if not os.path.isabs(_init_model_path):
        _project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        _init_model_path = os.path.join(_project_root, _init_model_path)
    _init_size = _probe_model_input_size(model, _init_model_path)
    if _init_size:
        config.model_input_size = _init_size
        logger.info("[AI Loop] Initial model input size: %d", _init_size)

    # Populate config._detect_class_names from the *initial* session too —
    # previously this only happened inside the hot-swap path below, so the
    # Aim page's Target Classes selector (and the semantic-filter's
    # class-name deny list) stayed empty until the user swapped the model
    # at least once, even though the very first model loaded already has
    # class names to read. A real, reported bug: a multi-class model
    # selected from a fresh app start showed no class selector at all.
    sync_detection_class_names_from_backend(model, config)

    pid_x = PIDController(config.pid_kp_x, config.pid_ki_x, config.pid_kd_x)
    pid_y = PIDController(config.pid_kp_y, config.pid_ki_y, config.pid_kd_y)

    state = LoopState(cached_mouse_move_method=config.mouse_move_method)
    # CUDA IO binding — set up once per model session to avoid repeated
    # host→device copies.  Recreated on model hot-swap.
    _io_binding: list = [None]

    def _setup_io_binding(m) -> object | None:
        if not getattr(config, 'cuda_io_binding_enabled', False):
            return None
        providers = m.get_providers() if hasattr(m, 'get_providers') else []
        if not any('CUDA' in p or 'Tensorrt' in p for p in providers):
            return None
        try:
            return m.io_binding()
        except Exception:
            return None

    _io_binding[0] = _setup_io_binding(model)

    _prio_level = getattr(config, 'thread_priority', 'high')
    _set_process_priority(_prio_level)
    _set_thread_priority(_prio_level)

    current_model_path = config.model_path
    current_backend = str(getattr(config, "inference_backend", "auto")).lower()
    current_dml_fallback = bool(getattr(config, "dml_cpu_fallback", True))
    current_fp16 = bool(getattr(config, "trt_fp16_enabled", False))

    ema_total = 0.0
    ema_overhead = 0.0   # this loop's per-iteration work before the tensor wait
    ema_qwait = 0.0      # time blocked waiting for _preprocess_worker's output
    ema_inf = 0.0
    ema_post = 0.0
    last_stats_print = time.perf_counter()
    last_detection_run_time = 0.0

    # Two independent locks to eliminate cross-contention:
    #   region_lock  — inference writes target_region, capture reads it
    #   frame_lock   — capture writes latest_frame/latest_region, inference reads them
    region_lock = threading.Lock()
    frame_lock  = threading.Lock()
    capture_stop_event = threading.Event()
    capture_state: dict[str, object] = {
        'latest_frame': None,
        'latest_region': None,
        'target_region': None,
        # Monotonic publish counter, bumped under frame_lock every time
        # latest_frame is replaced. The preprocess worker uses this — NOT
        # id(latest_frame) — to tell "new frame" from "same frame again".
        # id() is a memory address that CPython reuses the moment an object
        # is freed, and every captured frame is an identically-shaped
        # ndarray from the same allocator, so a genuinely new frame can land
        # on the address the previous one just vacated and be skipped as a
        # duplicate (classic ABA). A counter can't collide.
        'frame_seq': 0,
    }
    # (frame, region it was captured for) — reused when dxcam reports "no change",
    # so the frame is never republished against a region it wasn't grabbed from.
    _last_valid_frame: list = [None]  # mutable container for closure
    # Mutable containers so the capture worker can hot-swap the backend
    _capture_backend: list = [None]
    _active_method: list = [None]

    # MAKCU aim toggle state (used when makcu_aim_mode == "toggle")
    _aim_toggle_active: list = [False]
    _aim_btn_prev: list = [False]
    # MAKCU Always-Aim side-button toggle state (used when
    # makcu_always_aim_mode == "toggle") — independent of the aim-trigger
    # toggle state above, since the two buttons/modes are configured separately.
    _always_aim_toggle_active: list = [False]
    _always_aim_btn_prev: list = [False]
    # MAKCU disengage-delay state
    _disengage_time: list = [0.0]   # perf_counter timestamp when aim was released
    _was_aiming: list = [False]     # previous-frame is_aiming for falling-edge detection

    # Preprocess worker state — runs concurrently with inference to avoid
    # serializing resize+normalize and ONNX inference on the same thread.
    _tensor_queue: queue.Queue = queue.Queue(maxsize=1)
    _preprocess_stop: threading.Event = threading.Event()

    def _capture_worker() -> None:
        _set_thread_priority(getattr(config, 'thread_priority', 'high'))
        _capture_backend[0] = initialize_screen_capture(config)
        _active_method[0] = _detect_active_capture_method(
            _capture_backend[0],
            getattr(config, 'screenshot_method', 'mss'),
        )

        high_res_timer_enabled = False
        last_capture_perf = 0.0
        last_method_check = 0.0
        source_stale_logged = False

        try:
            while config.Running and not capture_stop_event.is_set():
                screenshot_interval = max(0.001, float(getattr(config, 'screenshot_interval', config.detect_interval)))
                detect_interval = float(getattr(config, 'detect_interval', screenshot_interval))
                # Windows' default Sleep()/time.sleep() granularity is ~15.6ms;
                # any throttle interval tighter than that needs the high-res
                # multimedia timer below or the configured interval is
                # silently ignored — both screenshot_interval and
                # detect_interval gate a _sleep_precise() call (this loop and
                # the main inference loop respectively), and this timer
                # resolution request is process-wide, so either interval
                # needing it is enough to request it.
                should_use_high_res_timer = min(screenshot_interval, detect_interval) < 0.015

                if should_use_high_res_timer and not high_res_timer_enabled:
                    high_res_timer_enabled = _set_windows_timer_resolution_1ms(True)
                elif high_res_timer_enabled and not should_use_high_res_timer:
                    _set_windows_timer_resolution_1ms(False)
                    high_res_timer_enabled = False

                # --- Hot-swap screenshot backend every 0.5s ---
                now_check = time.perf_counter()
                if now_check - last_method_check >= 0.5:
                    last_method_check = now_check
                    new_backend, new_method = reinitialize_if_method_changed(
                        config, _capture_backend[0], _active_method[0],
                    )
                    if new_backend is not _capture_backend[0]:
                        _capture_backend[0] = new_backend
                        _active_method[0] = new_method
                        _last_valid_frame[0] = None  # reset cached frame on backend change

                with region_lock:
                    target_region = capture_state.get('target_region')

                if target_region is None:
                    _sleep_precise(0.001)
                    continue

                now_capture = time.perf_counter()
                wait_for = screenshot_interval - (now_capture - last_capture_perf)
                if wait_for > 0:
                    _sleep_precise(wait_for)
                    continue

                last_capture_perf = time.perf_counter()
                captured_frame = capture_frame(_capture_backend[0], target_region)
                captured_region = target_region

                if captured_frame is not None and _capture_backend_is_stale(
                    _capture_backend[0], _SOURCE_FRAME_MAX_AGE_S
                ):
                    if not source_stale_logged:
                        logger.warning(
                            "[Capture] Source (%s) delivered no new frame for >%.1fs — "
                            "pausing detection instead of re-inferring the frozen frame.",
                            _active_method[0], _SOURCE_FRAME_MAX_AGE_S,
                        )
                        source_stale_logged = True
                    _last_valid_frame[0] = None
                    continue
                if source_stale_logged and captured_frame is not None:
                    logger.info("[Capture] Source (%s) is delivering frames again.", _active_method[0])
                    source_stale_logged = False

                if captured_frame is not None:
                    _last_valid_frame[0] = (captured_frame, target_region)
                elif _last_valid_frame[0] is not None:
                    # dxcam returns None when screen content hasn't changed;
                    # reuse the last valid frame so FPS isn't throttled by VSync
                    captured_frame, captured_region = _last_valid_frame[0]
                else:
                    continue

                with frame_lock:
                    capture_state['latest_frame'] = captured_frame
                    capture_state['latest_region'] = captured_region
                    capture_state['frame_seq'] = int(capture_state['frame_seq']) + 1

                config.screenshot_frame_count = int(getattr(config, 'screenshot_frame_count', 0)) + 1
        finally:
            if high_res_timer_enabled:
                _set_windows_timer_resolution_1ms(False)
            if _capture_backend[0] is not None:
                _cleanup_capture(_capture_backend[0])

    def _preprocess_worker() -> None:
        _set_thread_priority(getattr(config, 'thread_priority', 'high'))
        last_frame_seq: int = -1
        _cmc_prev: list = [None]  # previous 128×128 float32 gray frame for phase correlation
        _last_error_log: list = [float('-inf')]
        while not _preprocess_stop.is_set() and config.Running:
            frame = None
            try:
                with frame_lock:
                    frame = capture_state.get('latest_frame')
                    region = capture_state.get('latest_region')
                    frame_seq = int(capture_state['frame_seq'])
                if frame is None or region is None or frame_seq == last_frame_seq:
                    time.sleep(0.001)
                    continue
                last_frame_seq = frame_seq

                if getattr(config, 'cam_motion_comp_enabled', False):
                    cmc_size = int(getattr(config, 'cam_motion_comp_size', 128))
                    small = cv2.resize(frame[:, :, :3], (cmc_size, cmc_size), interpolation=cv2.INTER_LINEAR)
                    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
                    if _cmc_prev[0] is not None and _cmc_prev[0].shape == gray.shape:
                        shift, _ = cv2.phaseCorrelate(_cmc_prev[0], gray)
                        sx = frame.shape[1] / float(cmc_size)
                        sy = frame.shape[0] / float(cmc_size)
                        state.cam_shift_x = max(-30.0, min(30.0, float(shift[0]) * sx))
                        state.cam_shift_y = max(-30.0, min(30.0, float(shift[1]) * sy))
                        # Running integral of the per-frame shift — see
                        # LoopState.cam_drift_x/y — so process_aiming can
                        # compensate the predictor/Kalman's position history,
                        # not just this frame's PID error. Deadzoned per axis
                        # before accumulating — see apply_cam_shift_deadzone()'s
                        # docstring for why.
                        state.cam_drift_x += apply_cam_shift_deadzone(state.cam_shift_x, _CAM_DRIFT_DEADZONE_PX)
                        state.cam_drift_y += apply_cam_shift_deadzone(state.cam_shift_y, _CAM_DRIFT_DEADZONE_PX)
                    _cmc_prev[0] = gray
                else:
                    _cmc_prev[0] = None
                    state.cam_shift_x = 0.0
                    state.cam_shift_y = 0.0
                    state.cam_drift_x = 0.0
                    state.cam_drift_y = 0.0

                if getattr(config, 'third_person_mask', False):
                    frame = frame.copy()
                    from .aim_paths import apply_third_person_mask
                    apply_third_person_mask(frame)

                _frame_is_square = frame.shape[0] == frame.shape[1]
                tensor, lb_scale, lb_pad_x, lb_pad_y = preprocess_image(
                    frame, config.model_input_size, fast_resize=_frame_is_square
                )
                put_latest(_tensor_queue, (tensor, lb_scale, lb_pad_x, lb_pad_y, region), timeout=0.05)
            except Exception:
                now_err = time.perf_counter()
                if now_err - _last_error_log[0] >= _PREPROCESS_ERROR_LOG_INTERVAL_S:
                    _last_error_log[0] = now_err
                    logger.exception(
                        "[Preprocess] Failed on frame shape=%s dtype=%s model_input_size=%s",
                        getattr(frame, 'shape', None), getattr(frame, 'dtype', None),
                        getattr(config, 'model_input_size', None),
                    )
                time.sleep(0.001)

    _preprocess_thread = threading.Thread(target=_preprocess_worker, name='PreprocessWorker', daemon=True)
    _preprocess_thread.start()

    capture_thread = threading.Thread(target=_capture_worker, name='CaptureWorker', daemon=True)
    capture_thread.start()

    from .ocr_inference import start as _ocr_start, stop as _ocr_stop
    from .hud_inference import start as _hud_start, stop as _hud_stop
    _ocr_start(config)
    _hud_start(config)

    try:
        while _is_active():
            try:
                # ── Cooperative pause / stop check ───────────────────────────
                # config.inference_paused is a simple flag that UI code can set
                # before running an installer.  inference_controller provides an
                # event-based alternative usable from non-config code paths.
                if getattr(config, 'inference_paused', False) or inference_controller.should_pause:
                    if not inference_controller.wait_while_paused(check_interval=0.05):
                        break  # stop was requested while waiting
                    # Re-check the config flag after unpausing
                    if getattr(config, 'inference_paused', False):
                        time.sleep(0.05)
                        continue

                if inference_controller.should_stop:
                    break

                loop_start = time.perf_counter()
                current_time = time.time()

                prev_model = model
                (
                    model, current_model_path, input_name,
                    current_backend, current_dml_fallback, current_fp16,
                ) = _try_hot_swap_model(
                    config,
                    model,
                    current_model_path,
                    current_backend,
                    current_dml_fallback,
                    current_fp16,
                )
                if model is not prev_model:
                    _io_binding[0] = _setup_io_binding(model)
                    is_end2end = detect_end2end_output(model)
                    logger.info("[Model HotSwap] Output format: %s",
                                "end-to-end [N, 6]" if is_end2end else "raw grid")
                    # Drain tensors sized for the old model so the next inference
                    # always receives a tensor matching the new model_input_size.
                    while True:
                        try:
                            _tensor_queue.get_nowait()
                        except queue.Empty:
                            break
                    # Refresh ONNX class-name metadata for semantic FP filter (Someone_idea).
                    try:
                        sync_detection_class_names_from_backend(model, config)
                    except Exception:
                        pass

                if current_time - state.last_pid_update > state.pid_check_interval:
                    pid_x.Kp, pid_x.Ki, pid_x.Kd = config.pid_kp_x, config.pid_ki_x, config.pid_kd_x
                    pid_y.Kp, pid_y.Ki, pid_y.Kd = config.pid_kp_y, config.pid_ki_y, config.pid_kd_y
                    state.last_pid_update = current_time

                if current_time - state.last_method_check_time > state.method_check_interval:
                    new_method = config.mouse_move_method
                    if new_method != state.cached_mouse_move_method:
                        state.cached_mouse_move_method = new_method
                    state.last_method_check_time = current_time
                    _sync_makcu_features(config)

                capture_width, capture_height = get_capture_dimensions(config)
                update_crosshair_position(config, capture_width // 2, capture_height // 2)

                _makcu_btn  = getattr(config, 'makcu_aim_button', 'lmb')
                _makcu_mode = getattr(config, 'makcu_aim_mode', 'hold')
                _use_makcu  = (
                    _makcu_btn != 'off'
                    and getattr(config, 'mouse_move_method', '') == 'makcu'
                )

                # Optional MAKCU side-button-driven Always Aim — a second,
                # independent activation path alongside the plain always_aim
                # checkbox (see config.py's makcu_always_aim_button docstring).
                # Deliberately not gated on _use_makcu/makcu_aim_button: the
                # side button is its own optional feature and should work
                # even if the primary Aim Trigger Button is set to "off".
                _makcu_always_btn  = getattr(config, 'makcu_always_aim_button', 'off')
                _makcu_always_mode = getattr(config, 'makcu_always_aim_mode', 'hold')
                _use_makcu_always  = (
                    _makcu_always_btn != 'off'
                    and getattr(config, 'mouse_move_method', '') == 'makcu'
                )
                _effective_always_aim = bool(getattr(config, 'always_aim', False))
                if _use_makcu_always:
                    try:
                        from win_utils.makcu_mouse import is_makcu_connected, makcu_mouse as _mm
                        if is_makcu_connected():
                            # Bindable to any of the four MAKCU stream buttons
                            # (L/R alongside the original side1/side2) — same
                            # rising-edge hold/toggle handling regardless of
                            # which physical button is chosen.
                            always_btn_now = _makcu_button_held(_mm, _makcu_always_btn)
                            if _makcu_always_mode == 'toggle':
                                # Rising-edge detection: flip toggle on button press
                                if always_btn_now and not _always_aim_btn_prev[0]:
                                    _always_aim_toggle_active[0] = not _always_aim_toggle_active[0]
                                _always_aim_btn_prev[0] = always_btn_now
                                _effective_always_aim = _effective_always_aim or _always_aim_toggle_active[0]
                            else:
                                # Hold mode: always-aim while button is held
                                _effective_always_aim = _effective_always_aim or always_btn_now
                    except Exception:
                        pass
                config.makcu_always_aim_active = _effective_always_aim

                if _use_makcu:
                    # MAKCU mode: aim state driven purely by the stream button so
                    # that AimKeys (which may include other mouse buttons) cannot
                    # bleed through and fire aim on the wrong button.
                    is_aiming = _effective_always_aim
                else:
                    is_aiming = _effective_always_aim or any(is_key_pressed(k) for k in config.AimKeys)
                if _use_makcu:
                    try:
                        from win_utils.makcu_mouse import is_makcu_connected, makcu_mouse as _mm
                        if is_makcu_connected():
                            btn_now = _makcu_button_held(_mm, _makcu_btn)
                            if _makcu_mode == 'toggle':
                                # Rising-edge detection: flip toggle on button press
                                if btn_now and not _aim_btn_prev[0]:
                                    _aim_toggle_active[0] = not _aim_toggle_active[0]
                                _aim_btn_prev[0] = btn_now
                                is_aiming = is_aiming or _aim_toggle_active[0]
                            else:
                                # Hold mode: aim while button is held
                                is_aiming = is_aiming or btn_now
                    except Exception:
                        pass
                # Makcu disengage-delay: keep is_aiming True for up to N seconds after release
                _disengage_delay = float(getattr(config, 'makcu_disengage_delay', 0.0) or 0.0)
                _raw_is_aiming = is_aiming  # pre-delay-extension value — see _was_aiming[0] below
                if _was_aiming[0] and not is_aiming:
                    # Falling edge: user just released/toggled off aim
                    _disengage_time[0] = current_time
                elif is_aiming and _disengage_time[0] > 0.0:
                    # Re-engaged during delay window: cancel timer
                    _disengage_time[0] = 0.0
                if not is_aiming and _disengage_delay > 0.0 and _disengage_time[0] > 0.0:
                    if current_time - _disengage_time[0] < _disengage_delay:
                        is_aiming = True  # still within delay window
                    else:
                        _disengage_time[0] = 0.0  # delay expired
                # Must track the RAW (pre-delay-extension) state, not the
                # effective `is_aiming` above — storing the extended value
                # here made every subsequent frame while still in the delay
                # window look like a fresh falling edge (_was_aiming[0]=True
                # from the extension, current raw is_aiming=False), which
                # kept resetting _disengage_time[0] to "now" every single
                # frame. That meant `current_time - _disengage_time[0]` was
                # never more than one frame old, so the delay condition above
                # never actually expired — the aim status stuck at "Aiming"
                # indefinitely instead of clearing after makcu_disengage_delay
                # seconds.
                _was_aiming[0] = _raw_is_aiming

                if _use_makcu and is_aiming != config.makcu_aim_active:
                    logger.debug(
                        "[AI Loop] MAKCU aim %s (trigger=%s mode=%s raw=%s always=%s delay_hold=%s)",
                        "engaged" if is_aiming else "disengaged", _makcu_btn, _makcu_mode,
                        _raw_is_aiming, _effective_always_aim, is_aiming and not _raw_is_aiming)
                config.makcu_aim_active = is_aiming
                if is_aiming:
                    if state.aiming_start_time == 0.0:
                        state.aiming_start_time = current_time
                else:
                    state.aiming_start_time = 0.0

                if not config.AimToggle or (not config.keep_detecting and not is_aiming):
                    # Detection stops here, so every published result is about to
                    # go stale — including auto-fire's, which would otherwise keep
                    # firing on its last box list for as long as its key is held.
                    clear_queues(overlay_boxes_queue, overlay_confidences_queue,
                                 auto_fire_queue=auto_fire_boxes_queue)
                    config.latest_boxes = []
                    config.latest_confidences = []
                    config.latest_all_boxes = []
                    config.latest_all_confidences = []
                    config.display_locked_box = None
                    config.display_locked_box_is_decaying = False
                    config.aim_prediction_active = False
                    _set_makcu_move_lock(config, False)
                    time.sleep(0.05)
                    continue

                crosshair_x, crosshair_y = config.crosshairX, config.crosshairY
                region = calculate_detection_region(config, crosshair_x, crosshair_y)
                if region['width'] <= 0 or region['height'] <= 0:
                    # e.g. fov_follow_mouse with the cursor on another monitor —
                    # without a sleep this retries at 100% of a core.
                    time.sleep(0.005)
                    continue

                with region_lock:
                    capture_state['target_region'] = region

                idle_enabled = getattr(config, 'idle_detect_enabled', True)
                if _effective_always_aim or getattr(config, 'always_auto_fire', False):
                    idle_enabled = False
                if idle_enabled and not is_aiming:
                    desired_interval = getattr(config, 'idle_detect_interval', config.detect_interval)
                else:
                    desired_interval = config.detect_interval

                now_detect = time.perf_counter()
                elapsed_detect = now_detect - last_detection_run_time
                if elapsed_detect < desired_interval:
                    next_detect_wait = max(0.0, desired_interval - elapsed_detect)
                    if next_detect_wait > 0:
                        _sleep_precise(next_detect_wait)
                    continue
                last_detection_run_time = now_detect

                t0 = time.perf_counter()
                try:
                    _pre_result = _tensor_queue.get(timeout=0.02)
                except queue.Empty:
                    continue
                input_tensor, lb_scale, lb_pad_x, lb_pad_y, latest_region = _pre_result
                if input_tensor.shape[2] != config.model_input_size:
                    continue  # stale tensor from a size transition; discard silently
                t1 = time.perf_counter()
                t2 = t3 = t4 = None

                try:
                    t2 = time.perf_counter()
                    iob = _io_binding[0]
                    if iob is not None:
                        try:
                            iob.bind_cpu_input(input_name, input_tensor)
                            for out in model.get_outputs():
                                iob.bind_output(out.name)
                            model.run_with_iobinding(iob)
                            outputs = iob.copy_outputs_to_cpu()
                        except Exception:
                            _io_binding[0] = None
                            outputs = model.run(None, {input_name: input_tensor})
                    else:
                        outputs = model.run(None, {input_name: input_tensor})
                    t3 = time.perf_counter()
                    boxes, confidences, class_ids = postprocess_outputs(
                        outputs,
                        latest_region['width'],
                        latest_region['height'],
                        config.model_input_size,
                        config.min_confidence,
                        latest_region['left'],
                        latest_region['top'],
                        letterbox_scale=lb_scale,
                        letterbox_pad_x=lb_pad_x,
                        letterbox_pad_y=lb_pad_y,
                        is_end2end=is_end2end,
                    )
                    # class_ids must go through NMS with the boxes: NMS drops
                    # detections and reorders the survivors by confidence, so
                    # a separately-held class_ids list stops matching boxes
                    # positionally the moment more than one detection exists.
                    boxes, confidences, class_ids = non_max_suppression(
                        boxes, confidences, class_ids=class_ids,
                    )
                    t4 = time.perf_counter()
                    config.last_detection_time = time.time()
                    config.detection_frame_count = int(getattr(config, 'detection_frame_count', 0)) + 1
                except (RuntimeError, ValueError) as e:
                    logger.error("ONNX inference error: %s", e)
                    continue

                # --- Target class selection (multi-select which of the loaded
                # model's own classes are valid aim targets, e.g. "enemy" but
                # never "teammate"). Independent of, and always applied
                # regardless of, the semantic FP filter below — a deliberate
                # user choice, not a false-positive heuristic. Cheap no-op
                # when aim_target_class_ids is empty (the default). ---
                boxes, confidences, class_ids = filter_detections_by_target_class(
                    boxes, confidences, class_ids, config)

                # --- Semantic FP filter (new feature from Someone_idea) ---
                if getattr(config, 'detect_semantic_filter_enabled', False):
                    from .detection_semantics import filter_detections_by_semantic_class
                    boxes, confidences, class_ids = filter_detections_by_semantic_class(
                        boxes, confidences, class_ids, config)

                all_boxes, all_confidences = boxes, confidences
                # --- Reduce FOV size while actively tracking a locked target ---
                # state.locked_box reflects last frame's selection (process_aiming
                # writes it further below in this same loop) — using it here to
                # gate THIS frame's FOV is the correct causal order: "was a target
                # already locked coming into this frame" decides whether to keep
                # other, farther-out detections from ever reaching target
                # selection at all this frame. See compute_effective_fov()'s own
                # docstring for the fov_reduce_since-driven shrink ramp this
                # feeds.
                _effective_fov_size, _effective_fov_height = compute_effective_fov(
                    config, state, current_time)
                # Published every frame (not just while shrunk) so the in-game
                # overlay, the UVC/NDI/UDP preview's baked-in overlay, and the
                # Web ESP overlay all draw/reason about the FOV actually in
                # effect right now, rather than the static configured size.
                config.fov_effective_size = _effective_fov_size
                config.fov_effective_height = _effective_fov_height
                boxes, confidences = filter_boxes_by_fov(
                    boxes, confidences, crosshair_x, crosshair_y, _effective_fov_size, config,
                    fov_height=_effective_fov_height)
                # NOTE: single_target_mode's reduction to one box used to happen
                # here, before process_aiming() ever saw the candidate list. That
                # meant sticky lock's IOU search — which needs the FULL list to
                # decide whether the previously-locked target is still visible —
                # only ever got a list of 0-or-1 boxes, so it could never actually
                # prefer the old target over whatever won this frame's plain
                # priority scoring. single_target_mode silently defeated sticky
                # lock. The full list is now always passed to process_aiming();
                # the single-target reduction for auto-fire/preview/ESP purposes
                # is derived below from what process_aiming() actually selected
                # (post sticky-lock), not from a separate lock-blind pre-filter.

                # Runtime cache for UVC preview overlay rendering — full list by
                # default; narrowed further below when single_target_mode is on.
                config.latest_boxes = boxes
                config.latest_confidences = confidences
                # Unreduced set — same list the in-game overlay draws from, used
                # by Web ESP so it isn't narrowed down by single_target_mode.
                config.latest_all_boxes = all_boxes
                config.latest_all_confidences = all_confidences

                # aim_start_delay_ms: detection keeps running, but no mouse
                # movement is sent until the delay since aim engaged has passed.
                aim_engaged = is_aiming and aim_start_delay_elapsed(
                    config, state.aiming_start_time, current_time)
                aimed_this_frame = bool(aim_engaged and boxes)
                _set_makcu_move_lock(config, aimed_this_frame)
                if aimed_this_frame:
                    process_aiming(
                        config,
                        boxes,
                        crosshair_x,
                        crosshair_y,
                        pid_x,
                        pid_y,
                        state.cached_mouse_move_method,
                        state,
                        current_time,
                        confidences=confidences,
                    )
                else:
                    # Not aiming this frame — either no detections came back, or
                    # detection did find boxes but the aim key isn't held (e.g.
                    # keep_detecting is on and the user simply isn't aiming right
                    # now). Both cases hit this branch identically: aimed_this_frame
                    # is False either way. If sticky lock is enabled and a target
                    # is currently locked, hold the lock (and all PID / smoothing
                    # state) for up to lock_decay_frames frames before giving up,
                    # instead of dropping it instantly — this is what makes
                    # releasing/re-pressing the aim key on the same target not
                    # restart tracking from scratch every time.
                    sticky = getattr(config, 'sticky_lock_enabled', False)
                    holding_lock = False
                    if sticky and state.locked_box is not None:
                        decay = int(getattr(config, 'lock_decay_frames', 15))
                        state.no_detection_frames += 1
                        config.display_locked_box_is_decaying = True
                        holding_lock = state.no_detection_frames < decay

                    if (aim_engaged and not boxes and holding_lock
                            and getattr(config, 'sticky_gap_extrapolate', False)):
                        ai_aiming.process_sticky_gap(
                            config, crosshair_x, crosshair_y, pid_x, pid_y,
                            state.cached_mouse_move_method, state, current_time,
                        )

                    if not holding_lock:
                        pid_x.reset()
                        pid_y.reset()
                        state.aim_y_last_target_y = 0.0
                        state.aim_y_last_target_t = 0.0
                        state.locked_box = None
                        state.no_detection_frames = 0
                        state.aim_carry_x = 0.0
                        state.aim_carry_y = 0.0
                        state.aim_ema_x = 0.0
                        state.aim_ema_y = 0.0
                        state.sticky_last_t = 0.0
                        state.sticky_vx = 0.0
                        state.sticky_vy = 0.0
                        config.display_locked_box = None
                        config.display_locked_box_is_decaying = False
                        config.aim_prediction_active = False
                        # Target lost — clear stale prediction/Kalman state so a
                        # newly-acquired target isn't corrupted by the old one's history.
                        if ai_aiming._predictor is not None:
                            ai_aiming._predictor.reset()
                        if ai_aiming._ema_predictor is not None:
                            ai_aiming._ema_predictor.reset()
                        if ai_aiming._rolling_predictor is not None:
                            ai_aiming._rolling_predictor.reset()
                        if ai_aiming._kalman is not None:
                            ai_aiming._kalman.reset()

                    # Humanization idle Micro-Jitter — aim key held, no target
                    # this frame. Runs after the reset above so this call's own
                    # sub-pixel carry isn't wiped by it in the same frame; opt-in
                    # (see apply_idle_micro_jitter's docstring), no-ops otherwise.
                    if aim_engaged:
                        ai_aiming.apply_idle_micro_jitter(config, state, state.cached_mouse_move_method)

                if config.single_target_mode:
                    config.latest_boxes, config.latest_confidences = reduce_boxes_for_single_target(
                        boxes, confidences,
                        state.locked_box, state.locked_confidence, aimed_this_frame,
                        crosshair_x, crosshair_y,
                        priority_mode=getattr(config, 'target_priority_mode', 'distance'),
                        confidence_weight=getattr(config, 'target_priority_confidence_weight', 0.5),
                    )

                update_queues(
                    overlay_boxes_queue,
                    overlay_confidences_queue,
                    all_boxes,
                    all_confidences,
                    auto_fire_queue=auto_fire_boxes_queue,
                    auto_fire_boxes=config.latest_boxes,
                )

                if getattr(config, 'enable_latency_stats', False):
                    alpha = _LATENCY_STATS_ALPHA
                    total_ms = (time.perf_counter() - loop_start) * 1000.0
                    # Named for what they actually measure. These were
                    # previously logged as "cap" and "pre", which they never
                    # were: capture runs on _capture_worker and preprocessing
                    # on _preprocess_worker, so neither is timed on this
                    # thread at all. What t0 and t1 actually bracket is this
                    # loop's own per-iteration overhead (hot-swap check, PID
                    # refresh, aim-key polling, region math) and the wait for
                    # a tensor to appear. Mislabelling them sent anyone
                    # reading these numbers looking for a capture problem
                    # that the numbers could not have shown.
                    overhead_ms = (t0 - loop_start) * 1000.0
                    qwait_ms = (t1 - t0) * 1000.0
                    inf_ms = (t3 - t2) * 1000.0 if t3 is not None and t2 is not None else 0.0
                    post_ms = (t4 - t3) * 1000.0 if t4 is not None and t3 is not None else 0.0

                    ema_total = ema_total * (1 - alpha) + total_ms * alpha
                    ema_overhead = ema_overhead * (1 - alpha) + overhead_ms * alpha
                    ema_qwait = ema_qwait * (1 - alpha) + qwait_ms * alpha
                    ema_inf = ema_inf * (1 - alpha) + inf_ms * alpha
                    ema_post = ema_post * (1 - alpha) + post_ms * alpha

                    now = time.perf_counter()
                    if now - last_stats_print >= float(getattr(config, 'latency_stats_interval', 1.0)):
                        logger.debug(
                            "[Latency EMA] total=%.1fms loop_overhead=%.1fms "
                            "tensor_wait=%.1fms infer=%.1fms postproc=%.1fms "
                            "interval=%.0fms (capture/preprocess run on their "
                            "own threads and are not timed here)",
                            ema_total, ema_overhead, ema_qwait,
                            ema_inf, ema_post, desired_interval * 1000,
                        )
                        last_stats_print = now

            except Exception as e:
                logger.error("[AI Loop Error] %s", e)
                traceback.print_exc()
                time.sleep(1.0)
    finally:
        # HUD/OCR feeders are module-level singletons shared with whichever loop
        # replaced this one — only the loop that still owns the pipeline stops them.
        if _active_loop_token is loop_token:
            _set_makcu_move_lock(config, False)
            _hud_stop()
            _ocr_stop()
        _preprocess_stop.set()
        if _preprocess_thread.is_alive():
            _preprocess_thread.join(timeout=1.0)
        capture_stop_event.set()
        capture_thread.join(timeout=1.0)
