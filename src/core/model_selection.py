# model_selection.py
"""Pure helpers for the model-selection <-> TensorRT-conversion hand-off.

No Qt / onnxruntime / win32 imports, so it is usable from the Convert page,
app_controller's background build thread, and unit tests alike.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def same_model_path(a: str, b: str, project_root: str | None = None) -> bool:
    """True if two model paths (portable "Model/x.onnx", absolute, or Windows
    backslash form) refer to the same file."""
    if not a or not b:
        return False
    root = project_root or _PROJECT_ROOT

    def _norm(p: str) -> str:
        p = str(p).replace("\\", "/")
        if not os.path.isabs(p):
            p = os.path.join(root, p)
        return os.path.normcase(os.path.normpath(p))

    return _norm(a) == _norm(b)


def rollback_failed_selection(config, attempted_path: str, project_root: str | None = None) -> str | None:
    """After a failed TensorRT conversion, point config back at the model the
    inference loop is actually running.

    The Model page writes the user's pick to config.model_path immediately
    (so a successful build can be hot-swapped in), which means a failed build
    would otherwise leave config.json pointing at a model with no engine —
    the next launch would then try to compile it inline. Only rolls back if
    the failed model is still the current selection (the user may have picked
    something else meanwhile) and a session is actually loaded.

    Returns the restored model_path, or None if nothing was rolled back.
    """
    loaded = getattr(config, "loaded_model_path", "") or ""
    if not loaded or not same_model_path(config.model_path, attempted_path, project_root):
        return None
    loaded_fp16 = bool(getattr(config, "loaded_model_fp16", config.trt_fp16_enabled))
    path_differs = not same_model_path(loaded, config.model_path, project_root)
    if not path_differs and loaded_fp16 == bool(config.trt_fp16_enabled):
        return None  # selection already matches what is running
    config.model_path = loaded
    config.trt_fp16_enabled = loaded_fp16
    try:
        from core.config import save_config
        save_config(config)
    except Exception as exc:
        logger.warning("[Model] Could not persist rollback after failed conversion: %s", exc)
    return loaded
