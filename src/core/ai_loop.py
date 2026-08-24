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
from .ai_loop_utils import (
    calculate_detection_region,
    clear_queues,
    filter_boxes_by_fov,
    get_capture_dimensions,
    reduce_boxes_for_single_target,
    update_crosshair_position,
    update_queues,
)
from .inference import PIDController, non_max_suppression, postprocess_outputs, preprocess_image
from .session_utils import inference_controller
from .screen_capture import (
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


def _try_hot_swap_model(
    config: Config,
    model: ort.InferenceSession,
    current_model_path: str,
    current_backend: str,
    current_dml_fallback: bool,
):
    """Try hot-swapping ONNX model when model/provider related settings change."""

    config_backend = str(getattr(config, "inference_backend", "auto")).lower()
    config_dml_fallback = bool(getattr(config, "dml_cpu_fallback", True))

    should_reload = (
        config.model_path != current_model_path
        or config_backend != current_backend
        or config_dml_fallback != current_dml_fallback
    )
    if not should_reload:
        return model, current_model_path, model.get_inputs()[0].name, current_backend, current_dml_fallback

    new_model_path = config.model_path
    if not os.path.isabs(new_model_path):
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        abs_model_path = os.path.join(project_root, new_model_path)
    else:
        abs_model_path = new_model_path

    if not (os.path.exists(abs_model_path) and abs_model_path.endswith('.onnx')):
        logger.warning("[Model HotSwap] Invalid path or file not found: %s", abs_model_path)
        config.model_path = current_model_path
        return model, current_model_path, model.get_inputs()[0].name, current_backend, current_dml_fallback

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
        logger.warning(
            "[Model HotSwap] Skipping swap to %s — no cached TensorRT engine yet "
            "(building one inline would block the inference loop for 1-5 min). "
            "Convert it first via the Convert tab.",
            os.path.basename(abs_model_path),
        )
        config.model_path = current_model_path
        return model, current_model_path, model.get_inputs()[0].name, current_backend, current_dml_fallback

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
        return new_model, new_model_path, input_name, config_backend, config_dml_fallback
    except Exception as e:
        logger.error("[Model HotSwap] Load failed: %s — continuing with current model", e)
        config.model_path = current_model_path
        return model, current_model_path, model.get_inputs()[0].name, current_backend, current_dml_fallback


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

    input_name = model.get_inputs()[0].name

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
    _last_valid_frame: list = [None]  # mutable container for closure
    # Mutable containers so the capture worker can hot-swap the backend
    _capture_backend: list = [None]
    _active_method: list = [None]

    # MAKCU aim toggle state (used when makcu_aim_mode == "toggle")
    _aim_toggle_active: list = [False]
    _aim_btn_prev: list = [False]
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

                if captured_frame is not None:
                    _last_valid_frame[0] = captured_frame
                elif _last_valid_frame[0] is not None:
                    # dxcam returns None when screen content hasn't changed;
                    # reuse the last valid frame so FPS isn't throttled by VSync
                    captured_frame = _last_valid_frame[0]
                else:
                    continue

                with frame_lock:
                    capture_state['latest_frame'] = captured_frame
                    capture_state['latest_region'] = target_region
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
        while not _preprocess_stop.is_set() and config.Running:
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
                    _cmc_prev[0] = gray
                else:
                    _cmc_prev[0] = None
                    state.cam_shift_x = 0.0
                    state.cam_shift_y = 0.0

                _frame_is_square = frame.shape[0] == frame.shape[1]
                tensor, lb_scale, lb_pad_x, lb_pad_y = preprocess_image(
                    frame, config.model_input_size, fast_resize=_frame_is_square
                )
                try:
                    _tensor_queue.put((tensor, lb_scale, lb_pad_x, lb_pad_y, region), timeout=0.05)
                except queue.Full:
                    pass
            except Exception:
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
        while config.Running:
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
                model, current_model_path, input_name, current_backend, current_dml_fallback = _try_hot_swap_model(
                    config,
                    model,
                    current_model_path,
                    current_backend,
                    current_dml_fallback,
                )
                if model is not prev_model:
                    _io_binding[0] = _setup_io_binding(model)
                    # Drain tensors sized for the old model so the next inference
                    # always receives a tensor matching the new model_input_size.
                    while True:
                        try:
                            _tensor_queue.get_nowait()
                        except queue.Empty:
                            break
                    # Refresh ONNX class-name metadata for semantic FP filter (Someone_idea).
                    try:
                        from .detection_semantics import sync_detection_class_names_from_backend
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

                capture_width, capture_height = get_capture_dimensions(config)
                update_crosshair_position(config, capture_width // 2, capture_height // 2)

                _makcu_btn  = getattr(config, 'makcu_aim_button', 'lmb')
                _makcu_mode = getattr(config, 'makcu_aim_mode', 'hold')
                _use_makcu  = (
                    _makcu_btn != 'off'
                    and getattr(config, 'mouse_move_method', '') == 'makcu'
                )
                if _use_makcu:
                    # MAKCU mode: aim state driven purely by the stream button so
                    # that AimKeys (which may include other mouse buttons) cannot
                    # bleed through and fire aim on the wrong button.
                    is_aiming = bool(getattr(config, 'always_aim', False))
                else:
                    is_aiming = bool(getattr(config, 'always_aim', False)) or any(is_key_pressed(k) for k in config.AimKeys)
                if _use_makcu:
                    try:
                        from win_utils.makcu_mouse import is_makcu_connected, makcu_mouse as _mm
                        if is_makcu_connected():
                            btn_now = _mm.rmb_held if _makcu_btn == 'rmb' else _mm.lmb_held
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

                config.makcu_aim_active = is_aiming
                if is_aiming:
                    if state.aiming_start_time == 0.0:
                        state.aiming_start_time = current_time
                else:
                    state.aiming_start_time = 0.0

                if not config.AimToggle or (not config.keep_detecting and not is_aiming):
                    clear_queues(overlay_boxes_queue, overlay_confidences_queue)
                    time.sleep(0.05)
                    continue

                crosshair_x, crosshair_y = config.crosshairX, config.crosshairY
                region = calculate_detection_region(config, crosshair_x, crosshair_y)
                if region['width'] <= 0 or region['height'] <= 0:
                    continue

                with region_lock:
                    capture_state['target_region'] = region

                idle_enabled = getattr(config, 'idle_detect_enabled', True)
                if getattr(config, 'always_aim', False) or getattr(config, 'always_auto_fire', False):
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

                # --- Semantic FP filter (new feature from Someone_idea) ---
                if getattr(config, 'detect_semantic_filter_enabled', False):
                    from .detection_semantics import filter_detections_by_semantic_class
                    boxes, confidences, class_ids = filter_detections_by_semantic_class(
                        boxes, confidences, class_ids, config)

                all_boxes, all_confidences = boxes, confidences
                boxes, confidences = filter_boxes_by_fov(boxes, confidences, crosshair_x, crosshair_y, config.fov_size, config)
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

                aimed_this_frame = bool(is_aiming and boxes)
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

                    if not holding_lock:
                        pid_x.reset()
                        pid_y.reset()
                        state.aim_y_last_target_y = 0.0
                        state.aim_y_last_target_t = 0.0
                        state.locked_box = None
                        state.no_detection_frames = 0
                        state.aim_carry_x = 0.0
                        state.aim_carry_y = 0.0
                        config.display_locked_box = None
                        config.display_locked_box_is_decaying = False
                        # Target lost — clear stale prediction/Kalman state so a
                        # newly-acquired target isn't corrupted by the old one's history.
                        if ai_aiming._predictor is not None:
                            ai_aiming._predictor.reset()
                        if ai_aiming._kalman is not None:
                            ai_aiming._kalman.reset()

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
        _hud_stop()
        _ocr_stop()
        _preprocess_stop.set()
        if _preprocess_thread.is_alive():
            _preprocess_thread.join(timeout=1.0)
        capture_stop_event.set()
        capture_thread.join(timeout=1.0)
