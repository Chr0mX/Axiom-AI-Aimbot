"""
mot_tracker.py — Multi-Object Tracker (Kalman + Hungarian + lightweight identity signature)

Pipeline position:
    NMS output (boxes, confidences)
    → MOTTracker.update()
    → confirmed tracked boxes + confidences (predict-only during occlusion)
    → FOV filter → target selection → PID

Motion model: constant-velocity Kalman filter, state = [cx, cy, vx, vy]
Association:  Hungarian algorithm (scipy) with graceful greedy fallback
Signature:    aspect ratio + velocity direction + centroid history (no frame crops)
"""

from __future__ import annotations

import math
from collections import deque
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment as _scipy_lap
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

# Chi-squared gate threshold: 2 DOF, 95% confidence → 5.991
_CHI2_GATE = 5.991


class _Status(Enum):
    TENTATIVE = 0
    CONFIRMED = 1
    LOST = 2


class _Track:
    __slots__ = (
        "track_id", "x", "P", "w", "h", "aspect_ratio",
        "centroid_history", "hits", "age", "time_since_update",
        "status", "confidence",
    )

    def __init__(
        self,
        track_id: int,
        cx: float,
        cy: float,
        w: float,
        h: float,
        confidence: float,
    ) -> None:
        self.track_id = track_id
        # Kalman state: [cx, cy, vx, vy]
        self.x = np.array([cx, cy, 0.0, 0.0], dtype=np.float64)
        # Initial covariance — high uncertainty on velocity
        self.P = np.diag([25.0, 25.0, 100.0, 100.0]).astype(np.float64)
        # Box size tracked separately with EMA
        self.w = float(w)
        self.h = float(h)
        self.aspect_ratio = w / max(h, 1.0)
        # Last 3 observed centroids for trajectory continuity
        self.centroid_history: deque = deque([(cx, cy)], maxlen=3)
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.status = _Status.TENTATIVE
        self.confidence = float(confidence)


# ---------------------------------------------------------------------------
# Kalman matrices (shared constants — only F changes per dt)
# ---------------------------------------------------------------------------
_H = np.array([[1.0, 0.0, 0.0, 0.0],
               [0.0, 1.0, 0.0, 0.0]], dtype=np.float64)   # 2×4 observation matrix

_R = np.diag([4.0, 4.0]).astype(np.float64)               # measurement noise (≈2 px std)

# Process noise base — scaled by dt inside predict
_Q_base = np.diag([1.0, 1.0, 4.0, 4.0]).astype(np.float64)

# EMA alpha for box size updates
_SIZE_ALPHA = 0.4


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class MOTTracker:
    """
    Multi-object tracker for the Axiom AI aimbot pipeline.

    Usage (inside ai_loop after NMS):
        boxes, confidences = mot_tracker.update(boxes, confidences, current_time)

    Only confirmed tracks are returned.  During occlusion (no matching detection)
    a confirmed track continues to output its Kalman-predicted position for up to
    max_age frames before being deleted.
    """

    def __init__(
        self,
        min_hits: int = 2,
        max_age: int = 8,
        lambda_motion: float = 0.45,
        lambda_iou: float = 0.25,
        lambda_sig: float = 0.30,
        iou_min: float = 0.05,
        sig_gate: float = 0.70,
    ) -> None:
        self.min_hits = min_hits
        self.max_age = max_age
        self.lambda_motion = lambda_motion
        self.lambda_iou = lambda_iou
        self.lambda_sig = lambda_sig
        self.iou_min = iou_min
        self.sig_gate = sig_gate

        self._tracks: List[_Track] = []
        self._next_id: int = 1
        self._prev_time: Optional[float] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        boxes: List[List[float]],
        confidences: List[float],
        current_time: float,
    ) -> Tuple[List[List[float]], List[float]]:
        """
        Ingest new detections; return (boxes, confidences) for confirmed tracks.

        boxes: list of [x1, y1, x2, y2] in screen coordinates
        Returns the same format — caller may drop-in replace raw NMS output.
        """
        dt = 0.01
        if self._prev_time is not None and current_time > self._prev_time:
            dt = current_time - self._prev_time
        self._prev_time = current_time

        # Convert [x1,y1,x2,y2] → (cx, cy, w, h)
        dets = _boxes_to_centroids(boxes)

        # 1. Predict all existing tracks forward by dt
        for track in self._tracks:
            _kalman_predict(track, dt)

        # 2. Data association
        matches, unmatched_tracks, unmatched_dets = self._associate(dets)

        # 3. Update matched tracks with their paired detection
        for ti, di in matches:
            _kalman_update(self._tracks[ti], dets[di], confidences[di], self._SIZE_ALPHA if False else _SIZE_ALPHA)

        # 4. Handle unmatched tracks (occlusion / target lost)
        for ti in unmatched_tracks:
            track = self._tracks[ti]
            track.time_since_update += 1
            if track.time_since_update > self.max_age:
                track.status = _Status.LOST

        # 5. Spawn new tentative tracks for unmatched detections
        for di in unmatched_dets:
            cx, cy, w, h = dets[di]
            track = _Track(self._next_id, cx, cy, w, h, confidences[di])
            self._next_id += 1
            self._tracks.append(track)

        # 6. Promote tentative → confirmed
        for track in self._tracks:
            if track.status == _Status.TENTATIVE and track.hits >= self.min_hits:
                track.status = _Status.CONFIRMED

        # 7. Prune lost tracks
        self._tracks = [t for t in self._tracks if t.status != _Status.LOST]

        # 8. Collect confirmed-track output boxes
        out_boxes: List[List[float]] = []
        out_confs: List[float] = []
        for track in self._tracks:
            if track.status == _Status.CONFIRMED:
                cx, cy = float(track.x[0]), float(track.x[1])
                hw = track.w * 0.5
                hh = track.h * 0.5
                out_boxes.append([cx - hw, cy - hh, cx + hw, cy + hh])
                out_confs.append(track.confidence)

        return out_boxes, out_confs

    def reset(self) -> None:
        """Clear all track state.  Call when exiting aiming mode."""
        self._tracks.clear()
        self._prev_time = None

    # ------------------------------------------------------------------
    # Association
    # ------------------------------------------------------------------

    def _associate(
        self,
        dets: List[Tuple[float, float, float, float]],
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        n_t = len(self._tracks)
        n_d = len(dets)

        if n_t == 0:
            return [], [], list(range(n_d))
        if n_d == 0:
            return [], list(range(n_t)), []

        # Build n_t × n_d cost matrix; invalid pairs get 1e6
        cost = np.full((n_t, n_d), 1e6, dtype=np.float64)
        valid = np.zeros((n_t, n_d), dtype=bool)

        for ti, track in enumerate(self._tracks):
            for di, det in enumerate(dets):
                mc, m_ok = self._motion_cost(track, det)
                ic, i_ok = self._iou_cost(track, det)
                sc, s_ok = self._sig_cost(track, det)
                if m_ok and i_ok and s_ok:
                    valid[ti, di] = True
                    cost[ti, di] = (
                        self.lambda_motion * mc
                        + self.lambda_iou * ic
                        + self.lambda_sig * sc
                    )

        # Linear assignment
        if _HAS_SCIPY:
            row_idx, col_idx = _scipy_lap(cost)
            assignment = list(zip(row_idx.tolist(), col_idx.tolist()))
        else:
            assignment = _greedy_lap(cost)

        matches: List[Tuple[int, int]] = []
        matched_t: set = set()
        matched_d: set = set()

        for ti, di in assignment:
            if valid[ti, di]:
                matches.append((ti, di))
                matched_t.add(ti)
                matched_d.add(di)

        unmatched_tracks = [ti for ti in range(n_t) if ti not in matched_t]
        unmatched_dets = [di for di in range(n_d) if di not in matched_d]

        return matches, unmatched_tracks, unmatched_dets

    # ------------------------------------------------------------------
    # Cost terms
    # ------------------------------------------------------------------

    def _motion_cost(
        self, track: _Track, det: Tuple[float, float, float, float]
    ) -> Tuple[float, bool]:
        """Mahalanobis distance normalized to [0,1], gated at chi2=5.991."""
        z = np.array([det[0], det[1]], dtype=np.float64)
        innov = z - _H @ track.x
        S = _H @ track.P @ _H.T + _R
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return 1.0, False
        d2 = float(innov @ S_inv @ innov)
        if d2 > _CHI2_GATE:
            return 1.0, False
        return d2 / _CHI2_GATE, True

    def _iou_cost(
        self, track: _Track, det: Tuple[float, float, float, float]
    ) -> Tuple[float, bool]:
        """1 - IoU, gated on iou_min."""
        cx, cy, dw, dh = det
        dx1, dy1 = cx - dw * 0.5, cy - dh * 0.5
        dx2, dy2 = cx + dw * 0.5, cy + dh * 0.5

        tx, ty = float(track.x[0]), float(track.x[1])
        tx1, ty1 = tx - track.w * 0.5, ty - track.h * 0.5
        tx2, ty2 = tx + track.w * 0.5, ty + track.h * 0.5

        iw = max(0.0, min(dx2, tx2) - max(dx1, tx1))
        ih = max(0.0, min(dy2, ty2) - max(dy1, ty1))
        inter = iw * ih
        union = dw * dh + track.w * track.h - inter
        iou = inter / max(union, 1e-6)

        if iou < self.iou_min:
            return 1.0, False
        return 1.0 - iou, True

    def _sig_cost(
        self, track: _Track, det: Tuple[float, float, float, float]
    ) -> Tuple[float, bool]:
        """Lightweight identity signature: aspect ratio + velocity direction + trajectory."""
        det_cx, det_cy, det_w, det_h = det

        # 1. Aspect ratio difference (normalized 0→1)
        det_ar = det_w / max(det_h, 1.0)
        denom_ar = max(track.aspect_ratio, det_ar, 1e-6)
        ar_cost = min(abs(track.aspect_ratio - det_ar) / denom_ar, 1.0)

        # 2. Velocity direction alignment
        vx, vy = float(track.x[2]), float(track.x[3])
        vel_mag = math.hypot(vx, vy)
        if vel_mag > 2.0 and len(track.centroid_history) >= 2:
            last_cx, last_cy = track.centroid_history[-1]
            dx, dy = det_cx - last_cx, det_cy - last_cy
            d = math.hypot(dx, dy)
            if d > 1.0:
                cos_sim = (vx * dx + vy * dy) / (vel_mag * d)
                vel_cost = (1.0 - max(-1.0, min(1.0, cos_sim))) * 0.5
            else:
                vel_cost = 0.0
        else:
            vel_cost = 0.0  # stationary track — do not penalize

        # 3. Trajectory continuity: mean centroid-history → detection distance
        if track.centroid_history:
            mean_d = sum(
                math.hypot(det_cx - hx, det_cy - hy)
                for hx, hy in track.centroid_history
            ) / len(track.centroid_history)
            gate_px = max(track.w, track.h, 50.0)
            hist_cost = min(mean_d / gate_px, 1.0)
        else:
            hist_cost = 0.0

        sig = (ar_cost + vel_cost + hist_cost) / 3.0
        if sig > self.sig_gate:
            return sig, False
        return sig, True


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _kalman_predict(track: _Track, dt: float) -> None:
    F = np.array([
        [1.0, 0.0,  dt, 0.0],
        [0.0, 1.0, 0.0,  dt],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float64)
    Q = _Q_base * dt
    track.x = F @ track.x
    track.P = F @ track.P @ F.T + Q
    track.age += 1


def _kalman_update(
    track: _Track,
    det: Tuple[float, float, float, float],
    confidence: float,
    size_alpha: float,
) -> None:
    cx, cy, w, h = det
    z = np.array([cx, cy], dtype=np.float64)
    innov = z - _H @ track.x
    S = _H @ track.P @ _H.T + _R
    K = track.P @ _H.T @ np.linalg.inv(S)
    track.x = track.x + K @ innov
    track.P = (np.eye(4) - K @ _H) @ track.P

    # EMA update for box dimensions and aspect ratio
    track.w = size_alpha * w + (1.0 - size_alpha) * track.w
    track.h = size_alpha * h + (1.0 - size_alpha) * track.h
    track.aspect_ratio = size_alpha * (w / max(h, 1.0)) + (1.0 - size_alpha) * track.aspect_ratio

    track.centroid_history.append((cx, cy))
    track.hits += 1
    track.time_since_update = 0
    track.confidence = confidence


def _boxes_to_centroids(
    boxes: List[List[float]],
) -> List[Tuple[float, float, float, float]]:
    result = []
    for box in boxes:
        x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
        result.append(((x1 + x2) * 0.5, (y1 + y2) * 0.5, x2 - x1, y2 - y1))
    return result


def _greedy_lap(cost: np.ndarray) -> List[Tuple[int, int]]:
    """Greedy assignment fallback when scipy is unavailable."""
    n_rows, n_cols = cost.shape
    used_cols: set = set()
    assignment: List[Tuple[int, int]] = []
    for r in range(n_rows):
        best_c, best_v = -1, float("inf")
        for c in range(n_cols):
            if c not in used_cols and cost[r, c] < best_v:
                best_v, best_c = cost[r, c], c
        if best_c >= 0:
            assignment.append((r, best_c))
            used_cols.add(best_c)
    return assignment
