"""Main loop for AI inference and mouse control"""

from __future__ import annotations

import ctypes
import os
import queue
import threading
import time
import traceback
from typing import TYPE_CHECKING

import cv2
import numpy as np

from win_utils import is_key_pressed

from .ai_aiming import process_aiming
from .ai_loop_state import LoopState
from .ai_loop_utils import (
    calculate_detection_region,
    clear_queues,
    filter_boxes_by_fov,
    find_closest_target,
    get_capture_dimensions,
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
        print(f"[模型熱切換] 路徑無效或檔案不存在: {abs_model_path}")
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
        actual_providers = new_model.get_providers()
        if actual_providers:
            config.current_provider = actual_providers[0]
        print(f"[模型熱切換] 已切換至: {os.path.basename(abs_model_path)} / {config_backend}")
        return new_model, new_model_path, input_name, config_backend, config_dml_fallback
    except Exception as e:
        print(f"[模型熱切換] 載入失敗: {e}，繼續使用原模型")
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

    if os.name == 'nt':
        try:
            ctypes.windll.kernel32.SetThreadPriority(
                ctypes.windll.kernel32.GetCurrentThread(), 1)  # ABOVE_NORMAL
        except Exception:
            pass

    current_model_path = config.model_path
    current_backend = str(getattr(config, "inference_backend", "auto")).lower()
    current_dml_fallback = bool(getattr(config, "dml_cpu_fallback", True))

    ema_total = 0.0
    ema_capture = 0.0
    ema_pre = 0.0
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
    }
    _last_valid_frame: list = [None]  # mutable container for closure
    # Mutable containers so the capture worker can hot-swap the backend
    _capture_backend: list = [None]
    _active_method: list = [None]

    _tensor_queue: queue.Queue = queue.Queue(maxsize=1)
    _preprocess_stop: threading.Event = threading.Event()

    def _preprocess_worker() -> None:
        last_frame_id: int = -1
        while not _preprocess_stop.is_set() and config.Running:
            try:
                with frame_lock:
                    frame = capture_state.get('latest_frame')
                    region = capture_state.get('latest_region')
                    frame_id = id(frame)
                if frame is None or region is None or frame_id == last_frame_id:
                    time.sleep(0.001)
                    continue
                last_frame_id = frame_id
                tensor, lb_scale, lb_pad_x, lb_pad_y = preprocess_image(
                    frame, config.model_input_size)
                try:
                    _tensor_queue.put(
                        (tensor, lb_scale, lb_pad_x, lb_pad_y, region),
                        timeout=0.05)
                except queue.Full:
                    pass
            except Exception:
                time.sleep(0.001)

    _preprocess_thread = threading.Thread(
        target=_preprocess_worker, name='PreprocessWorker', daemon=True)
    _preprocess_thread.start()

    def _capture_worker() -> None:
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
                should_use_high_res_timer = screenshot_interval <= 0.002

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

                # Pre-resize NDI frames in the capture thread so the inference
                # thread only runs blobFromImage (fast path) instead of a full
                # letterbox resize on a potentially large crop region.
                store_region = target_region
                if (getattr(config, 'ndi_pre_resize', True)
                        and _active_method[0] == 'ndi'):
                    model_sz = int(getattr(config, 'model_input_size', 640))
                    fh, fw = captured_frame.shape[:2]
                    if fw != model_sz or fh != model_sz:
                        captured_frame = cv2.resize(
                            captured_frame, (model_sz, model_sz),
                            interpolation=cv2.INTER_AREA,
                        )
                        store_region = dict(target_region)
                        store_region['width'] = model_sz
                        store_region['height'] = model_sz

                with frame_lock:
                    capture_state['latest_frame'] = captured_frame
                    capture_state['latest_region'] = store_region

                config.last_screenshot_time = time.time()
                config.screenshot_frame_count = int(getattr(config, 'screenshot_frame_count', 0)) + 1
        finally:
            if high_res_timer_enabled:
                _set_windows_timer_resolution_1ms(False)
            if _capture_backend[0] is not None:
                _cleanup_capture(_capture_backend[0])

    capture_thread = threading.Thread(target=_capture_worker, name='CaptureWorker', daemon=True)
    capture_thread.start()

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

                is_aiming = bool(getattr(config, 'always_aim', False)) or any(is_key_pressed(k) for k in config.AimKeys)
                _makcu_btn = getattr(config, 'makcu_aim_button', 'lmb')
                if not is_aiming and _makcu_btn != 'off' \
                        and getattr(config, 'mouse_move_method', '') == 'makcu':
                    try:
                        from win_utils.makcu_mouse import is_makcu_connected, makcu_mouse as _mm
                        if is_makcu_connected():
                            _mm.lmb_cache_seconds = max(0.008, config.detect_interval)
                            is_aiming = _mm.rmb_held if _makcu_btn == 'rmb' else _mm.lmb_held
                    except Exception:
                        pass
                if is_aiming:
                    if state.aiming_start_time == 0.0:
                        state.aiming_start_time = current_time
                else:
                    state.aiming_start_time = 0.0

                if not config.AimToggle or (not config.keep_detecting and not is_aiming):
                    clear_queues(overlay_boxes_queue, overlay_confidences_queue)
                    config.tracker_has_prediction = False
                    time.sleep(0.05)
                    continue

                crosshair_x, crosshair_y = config.crosshairX, config.crosshairY
                region = calculate_detection_region(config, crosshair_x, crosshair_y)
                if region['width'] <= 0 or region['height'] <= 0:
                    continue

                with region_lock:
                    capture_state['target_region'] = region
                with frame_lock:
                    latest_frame = capture_state.get('latest_frame')
                    latest_region = capture_state.get('latest_region')

                if latest_frame is None or latest_region is None:
                    _sleep_precise(0.001)
                    continue

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

                # Frame skip gate: skip inference when the capture region
                # hasn't changed significantly (e.g. static background).
                t0 = time.perf_counter()
                try:
                    _pre_result = _tensor_queue.get(timeout=0.02)
                except queue.Empty:
                    continue
                input_tensor, lb_scale, lb_pad_x, lb_pad_y, latest_region = _pre_result
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
                    boxes, confidences = non_max_suppression(boxes, confidences)
                    t4 = time.perf_counter()
                    config.last_detection_time = time.time()
                    config.detection_frame_count = int(getattr(config, 'detection_frame_count', 0)) + 1
                except (RuntimeError, ValueError) as e:
                    print(f"ONNX 推理錯誤: {e}")
                    continue

                # --- Semantic FP filter (new feature from Someone_idea) ---
                if getattr(config, 'detect_semantic_filter_enabled', False):
                    from .detection_semantics import filter_detections_by_semantic_class
                    boxes, confidences, class_ids = filter_detections_by_semantic_class(
                        boxes, confidences, class_ids, config)

                # Snapshot all post-NMS boxes for overlay display before FOV filter
                display_boxes = list(boxes)
                display_confs = list(confidences)

                boxes, confidences = filter_boxes_by_fov(boxes, confidences, crosshair_x, crosshair_y, config.fov_size, config)

                if config.single_target_mode:
                    boxes, confidences = find_closest_target(
                        boxes, confidences, crosshair_x, crosshair_y,
                        priority_mode=getattr(config, 'target_priority_mode', 'distance'),
                        confidence_weight=getattr(config, 'target_priority_confidence_weight', 0.5),
                    )

                # Runtime cache for UVC preview overlay rendering
                config.latest_boxes = boxes
                config.latest_confidences = confidences

                if is_aiming and boxes:
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
                    config.tracker_has_prediction = False
                    pid_x.reset()
                    pid_y.reset()
                    state.smooth_x = 0.0
                    state.smooth_y = 0.0
                    state.locked_box = None
                    state.no_detection_frames = 0
                    config.display_locked_box = None
                    config.display_locked_box_is_decaying = False

                update_queues(
                    overlay_boxes_queue,
                    overlay_confidences_queue,
                    display_boxes,
                    display_confs,
                    auto_fire_queue=auto_fire_boxes_queue,
                    auto_fire_boxes=boxes,
                )

                if getattr(config, 'enable_latency_stats', False):
                    alpha = float(getattr(config, 'latency_stats_alpha', 0.2))
                    total_ms = (time.perf_counter() - loop_start) * 1000.0
                    cap_ms = (t0 - loop_start) * 1000.0
                    pre_ms = (t1 - t0) * 1000.0
                    inf_ms = (t3 - t2) * 1000.0 if t3 is not None and t2 is not None else 0.0
                    post_ms = (t4 - t3) * 1000.0 if t4 is not None and t3 is not None else 0.0

                    ema_total = ema_total * (1 - alpha) + total_ms * alpha
                    ema_capture = ema_capture * (1 - alpha) + cap_ms * alpha
                    ema_pre = ema_pre * (1 - alpha) + pre_ms * alpha
                    ema_inf = ema_inf * (1 - alpha) + inf_ms * alpha
                    ema_post = ema_post * (1 - alpha) + post_ms * alpha

                    now = time.perf_counter()
                    if now - last_stats_print >= float(getattr(config, 'latency_stats_interval', 1.0)):
                        print(
                            f"[Latency EMA] total={ema_total:.1f}ms "
                            f"cap={ema_capture:.1f}ms pre={ema_pre:.1f}ms "
                            f"inf={ema_inf:.1f}ms post={ema_post:.1f}ms "
                            f"interval={desired_interval*1000:.0f}ms"
                        )
                        last_stats_print = now

            except Exception as e:
                print(f"[AI Loop Error] {e}")
                traceback.print_exc()
                time.sleep(1.0)
    finally:
        _preprocess_stop.set()
        if _preprocess_thread.is_alive():
            _preprocess_thread.join(timeout=1.0)
        capture_stop_event.set()
        capture_thread.join(timeout=1.0)
