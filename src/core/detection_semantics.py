"""Semantic false-positive filter for YOLO detections.

Ported from Someone_idea/detection_semantics.py.
Filters detections that match vegetation, vehicles, signs, and HUD elements
using three layers: ONNX class-name allow/deny lists, aspect-ratio geometry
heuristics, and minimum bounding-box size thresholds.
"""
from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_ALLOW_SUBSTR: tuple = (
    'head', 'helm', 'helmet', 'body', 'torso', 'chest', 'player', 'enemy',
    'opponent', 'char', 'person', 'soldier', 'operator', 'weapon', 'gun',
    'rifle', 'knife', 'sniper', 'armor', 'vest', 'target', 'bot', 'human',
    'zombie', 'agent', 'marine', 'skeleton', 'survivor', 'hunter',
)

_DENY_SUBSTR: tuple = (
    'tree', 'palm', 'vegetation', 'bush', 'house', 'building', 'window',
    'door', 'roof', 'car', 'truck', 'vehicle', 'sky', 'grass', 'rock',
    'stone', 'mountain', 'terrain', 'road', 'sign', 'board', 'billboard',
    'blackboard', 'whiteboard', 'fence', 'poster', 'table', 'chair', 'crate',
    'barrel', 'bus', 'train', 'motorcycle', 'bicycle', 'dog', 'cat', 'animal',
    'bird', 'flower', 'plant', 'helicopter', 'drone', 'quadcopter', 'rotor',
    'propeller', 'lamp', 'light', 'lantern', 'streetlight', 'trash', 'bin',
    'box', 'container', 'wall', 'gate', 'glass', 'mirror', 'water', 'cloud',
    'moon', 'star', 'pole', 'hydrant', 'bench', 'backpack', 'umbrella',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'kite',
    'baseball', 'glove', 'skateboard', 'surfboard', 'racket', 'bottle', 'cup',
    'fork', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hotdog', 'pizza', 'donut', 'cake', 'couch',
    'pottedplant', 'bed', 'diningtable', 'toilet', 'tv', 'laptop', 'remote',
    'keyboard', 'cellphone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddybear',
    'toothbrush', 'carbon', 'reload', 'xp', 'score', 'ammo', 'health',
    'shield', 'controller', 'gamepad', 'xbox', 'menu', 'guide', 'hud',
)


def _looks_like_vehicle(bw: float, bh: float, conf: float) -> bool:
    if bw <= 0 or bh <= 0:
        return False
    aspect = bw / bh
    if aspect > 1.3 and bw > 45 and bh > 25:
        return True
    if 1.1 < aspect < 1.5 and bw > 30 and bh > 20 and bw * bh < 2500:
        return True
    return False


def _looks_like_environment_box(bw: float, bh: float, conf: float) -> bool:
    bw = max(1.0, abs(float(bw)))
    bh = max(1.0, abs(float(bh)))
    ar = bw / bh
    area = bw * bh
    cf = float(conf)
    if _looks_like_vehicle(bw, bh, cf):
        return True
    if cf >= 0.62:
        return False
    if ar < 8 or area < 100:
        return False
    if ar <= 0.14 and bh >= 28:
        return True
    if ar >= 1.85 and bh >= 18:
        return True
    if ar <= 0.22 and bh >= 40 and cf < 0.52:
        return True
    if area >= 8000 and cf < 0.5:
        return True
    return False


def _semantic_keep_label(name: str) -> bool:
    if not name:
        return True
    nl = name.lower().strip()
    for deny in _DENY_SUBSTR:
        if deny in nl:
            return False
    for allow in _ALLOW_SUBSTR:
        if allow in nl:
            return True
    return True


def _filter_detections_min_geometry(
    boxes: list, confidences: list, class_ids: list, config
) -> Tuple[list, list, list]:
    min_area = float(getattr(config, 'detect_min_bbox_area_px', 0) or 0)
    min_short = float(getattr(config, 'detect_min_bbox_short_side_px', 0) or 0)
    max_side_frac = float(getattr(config, 'detect_min_bbox_max_side_frac', 0) or 0)
    roi_ref = float(max(120, min(2048, getattr(config, 'fov_size', 240) or 240)))
    min_long_px = max_side_frac * roi_ref if max_side_frac > 0 else 0

    if min_area <= 0 and min_short <= 0 and min_long_px <= 0:
        return boxes, confidences, class_ids

    cids = list(class_ids)
    if len(cids) < len(boxes):
        cids = cids + [0] * (len(boxes) - len(cids))

    out_b, out_c, out_i = [], [], []
    for i, box in enumerate(boxes):
        try:
            x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
        except (TypeError, ValueError, IndexError):
            continue
        bw = max(1.0, abs(x2 - x1))
        bh = max(1.0, abs(y2 - y1))
        short_side = min(bw, bh)
        long_side = max(bw, bh)
        area = bw * bh
        if min_short > 0 and short_side < min_short:
            continue
        if min_area > 0:
            area_thr = min_area
            if long_side < max(18, roi_ref * 0.06):
                area_thr = max(100, min_area * 0.3)
            elif long_side < max(28, roi_ref * 0.09):
                area_thr = max(160, min_area * 0.45)
            if area < area_thr:
                continue
        if min_long_px > 0 and long_side < min_long_px:
            continue
        out_b.append(box)
        out_c.append(float(confidences[i]) if i < len(confidences) else 0.0)
        out_i.append(int(cids[i]) if i < len(cids) else 0)

    return out_b, out_c, out_i


def filter_detections_by_semantic_class(
    boxes: list,
    confidences: list,
    class_ids: list,
    config,
    frame=None,
) -> Tuple[list, list, list]:
    """Filter detections by semantic class name, geometry, and aspect-ratio heuristics.

    Layers applied in order:
    1. Minimum bbox geometry (area / short-side thresholds)
    2. ONNX class-name deny list (trees, vehicles, HUD elements, etc.)
    3. Aspect-ratio environment heuristic (wide/tall low-confidence shapes)
    """
    if not boxes:
        return boxes or [], confidences or [], list(class_ids) if class_ids else []

    boxes, confidences, cids = _filter_detections_min_geometry(
        list(boxes), list(confidences), list(class_ids) if class_ids else [], config)

    nmap = getattr(config, '_detect_class_names', None)
    out_b, out_c, out_i = [], [], []

    for i, box in enumerate(boxes):
        cid = int(cids[i]) if i < len(cids) else 0
        conf = float(confidences[i]) if i < len(confidences) else 0.0

        # Named-class semantic deny
        if isinstance(nmap, dict) and cid in nmap:
            if not _semantic_keep_label(str(nmap[cid])):
                continue

        # Aspect-ratio / environment heuristic
        try:
            bw = abs(float(box[2]) - float(box[0]))
            bh = abs(float(box[3]) - float(box[1]))
            if _looks_like_environment_box(bw, bh, conf):
                continue
        except Exception:
            pass

        out_b.append(box)
        out_c.append(conf)
        out_i.append(cid)

    dropped = len(boxes) - len(out_b)
    if dropped > 0:
        logger.debug('Semantic filter: dropped %d detection(s)', dropped)

    return out_b, out_c, out_i


def sync_detection_class_names_from_backend(backend, config) -> None:
    """Read ONNX model metadata 'names' field and store in config._detect_class_names.

    Supports both JSON dict and Python-literal dict formats used by YOLO exporters.
    Called once after each model load / hot-swap.
    """
    try:
        setattr(config, '_detect_class_names', None)
    except Exception:
        return

    if backend is None:
        return

    try:
        meta = backend.get_modelmeta()
        custom = getattr(meta, 'custom_metadata_map', {}) or {}
        if 'names' not in custom:
            return
        names_raw = custom['names']
        names = None
        try:
            import json
            names = json.loads(names_raw)
        except Exception:
            import ast
            try:
                names = ast.literal_eval(names_raw)
            except Exception:
                pass
        if names is None:
            return
        if isinstance(names, dict):
            setattr(config, '_detect_class_names', {int(k): str(v) for k, v in names.items()})
        elif isinstance(names, (list, tuple)):
            setattr(config, '_detect_class_names', {i: str(n) for i, n in enumerate(names)})
    except Exception:
        pass
