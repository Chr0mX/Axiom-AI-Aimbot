"""Anchor prediction: where a tracked box will be a short time from now.

An additive layer on top of the existing lock tracking (it never replaces or
alters ``TargetTracker``/target selection). Each tracked object has an
``AnchorTrack`` that keeps a short timestamped history of its box, estimates
velocity (and optionally acceleration) from it, and predicts the future box
(centre + size) with a confidence. ``AnchorBank`` holds several independent
tracks keyed by track id. The result feeds the Adaptive / Dynamic FOV modes
(see ``FovAdjuster``) and lock association (``predicted box`` as an extra
overlap reference); with Anchor Prediction disabled none of this runs.

Pure Python (no numpy / Qt / win32): importable and testable anywhere, no
per-frame allocation in the hot path, and driven by the caller's timestamps
(never a fixed FPS).

Estimator — time-based least squares over the last ``history_len`` samples,
relative to the newest sample (tau = t_i - t_last <= 0):

    x(tau) ~= b0 + b1*tau            -> velocity = b1                (linear)
    x(tau) ~= b0 + b1*tau + b2*tau^2 -> velocity = b1, accel = 2*b2  (needs >= 4 samples)

Modes (``MODES``):

* ``disabled``       off — nothing is tracked or predicted.
* ``linear``         raw LSQ velocity, no acceleration, no smoothing.
* ``velocity_accel`` raw LSQ velocity + acceleration * ``accel_comp``.
* ``smoothed``       as above, with a time-constant EMA on velocity/acceleration
                     (``smoothing`` picks the time constant) to cut jitter.
* ``adaptive``       smoothed, and the lead strength follows tracking quality
                     (prediction confidence, box-overlap stability, direction
                     consistency); a large innovation (sudden direction change)
                     drops the stale history so the estimate recovers quickly.

Prediction = last box moved by v*H + a*H^2/2, where H is the horizon (plus any
time already elapsed since the last sample). Safeguards: non-finite and
out-of-order samples are ignored; a gap > ``_MAX_GAP_S`` restarts the track and
one > ``_COAST_START_S`` coasts (velocity decays, acceleration dropped, bounded
drift); a measurement implying more than ``max_speed_px_s`` is treated as a new
object (id reassignment / detection jump) and restarts the track; a (near)
stationary box gets no lead; deceleration stops at standstill instead of
reversing; acceleration may add at most the current speed over the horizon;
the lead is zero below the confidence threshold and fades in over a band above
it; displacement is capped at ``max_distance_px``; the predicted box is clamped
to the image bounds and its size change limited.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Sequence

from .target_tracking import box_iou, clamp01, edge_ratio

MODES = ('disabled', 'linear', 'velocity_accel', 'smoothed', 'adaptive')
# What the settings offer and persist. 'disabled' (the master switch / prediction method already
# turn it off) and 'linear' (= velocity_accel with Acceleration 0) are only still understood by
# the estimator and migrated on load (config.migrate_anchor_settings()).
MODES_UI = ('velocity_accel', 'smoothed', 'adaptive')

_CAP = 12                       # ring-buffer capacity (max history_len)
_MIN_ACCEL_SAMPLES = 4
_MAX_GAP_S = 0.30               # gap between samples that restarts the track (floor; see _GAP_FRAMES)
_COAST_START_S = 0.10           # gap that coasts: decaying velocity, no accel (floor; see _COAST_FRAMES)
_COAST_FRAMES = 2.5             # ...but never less than this many typical frame intervals,
_GAP_FRAMES = 6.0               # so a slow inference rate (10 FPS) isn't mistaken for missed detections
_TAU_DT_S = 0.2                 # EMA weight for the typical frame interval
_COAST_DECAY_S = 0.15
_STATIONARY_BOX_PER_S = 0.25    # below this many box-heights/s the box counts as still
_SUPPORT_FULL_S = 0.12          # track age for full confidence
_TAU_ERR_S = 0.15               # prediction-error EMA time constant
_TAU_STAB_S = 0.40
_TAU_TURN_S = 0.30
_ERR_ZERO_FIT = 0.35            # normalised one-step error that zeroes "accuracy"
_CONF_RAMP = 0.25               # confidence band over which the lead fades in
_SPIKE_ERR = 0.5                # normalised innovation that signals a direction change
_MAX_SIZE_CHANGE = 0.5          # predicted w/h stays within +/-50 % of the current
_MIN_TURN_SPEED = 30.0          # px/s below which direction is meaningless
_TAU_MIN_S = 0.02
_TAU_SPAN_S = 0.20              # smoothing 0..1 -> time constant 20..220 ms
_SIGMA_FLOOR_PX = 0.5           # assumed detector jitter when the track looks perfectly clean
_SIGNIFICANCE_K = 2.0           # speed / acceleration must clear this many standard errors to count


def _alpha(dt: float, tau: float) -> float:
    return 1.0 - math.exp(-dt / tau)


# ── settings ────────────────────────────────────────────────────────────────

_PARAM_DEFAULTS = {
    'mode': 'adaptive',
    'horizon_s': 0.100,
    'strength': 1.0,
    'smoothing': 0.4,
    'accel_comp': 0.35,
    'conf_threshold': 0.35,
    'history_len': 6,
    'max_distance_px': 0.0,      # 0 = unlimited
    'max_speed_px_s': 1200.0,
    'influence_adaptive': 0.5,
    'influence_dynamic': 0.5,
}


def anchor_active(config) -> bool:
    """The single on/off gate: ``anchor_enabled``. (A stale ``disabled`` mode still reads as
    off; it is never stored any more, config.migrate_anchor_settings() turns it into
    ``anchor_enabled = False`` on load.)"""
    return bool(getattr(config, 'anchor_enabled', False)) and \
        getattr(config, 'anchor_mode', 'adaptive') in MODES[1:]


class AnchorParams:
    """Settings read by the tracks; mutated in place from ``Config`` so the
    hot loop allocates nothing."""

    __slots__ = tuple(_PARAM_DEFAULTS)

    def __init__(self) -> None:
        for name, value in _PARAM_DEFAULTS.items():
            setattr(self, name, value)

    def update_from(self, config) -> "AnchorParams":
        mode = getattr(config, 'anchor_mode', None)
        if mode in MODES:
            self.mode = mode
        self._num('horizon_s', getattr(config, 'anchor_horizon_ms', None), 0.001)
        self._num('strength', getattr(config, 'anchor_strength', None))
        self._num('smoothing', getattr(config, 'anchor_velocity_smoothing', None))
        self._num('accel_comp', getattr(config, 'anchor_accel_comp', None))
        self._num('conf_threshold', getattr(config, 'anchor_confidence_threshold', None))
        self._num('max_speed_px_s', getattr(config, 'prediction_max_velocity', None))
        self._num('influence_adaptive', getattr(config, 'anchor_fov_influence_adaptive', None))
        self._num('influence_dynamic', getattr(config, 'anchor_fov_influence_dynamic', None))
        try:
            self.history_len = max(_MIN_ACCEL_SAMPLES, min(_CAP, int(getattr(config, 'anchor_history_len', 6))))
        except (TypeError, ValueError):
            pass
        # The distance cap is a share of the *configured* FOV (never the shrunk
        # one: the prediction influences the shrink, so using the shrunk FOV
        # here would feed back on itself).
        pct = getattr(config, 'anchor_max_distance_pct', None)
        try:
            pct = float(pct)
            fw = float(getattr(config, 'fov_size', 0) or 0)
            fh = float(getattr(config, 'fov_height', fw) or fw)
            self.max_distance_px = max(0.0, pct) / 100.0 * min(fw, fh) if pct > 0.0 and fw > 0.0 else 0.0
        except (TypeError, ValueError):
            pass
        return self

    def _num(self, name: str, value, scale: float = 1.0) -> None:
        try:
            v = float(value) * scale
        except (TypeError, ValueError):
            return
        if math.isfinite(v):
            setattr(self, name, v)


class Prediction:
    """Result of ``AnchorTrack.predict`` (one reused object per track)."""

    __slots__ = ('valid', 'confidence', 'x1', 'y1', 'x2', 'y2', 'px1', 'py1', 'px2', 'py2',
                 'cx', 'cy', 'pcx', 'pcy', 'dx', 'dy', 'dist', 'vx', 'vy', 'ax', 'ay', 'speed')

    def __init__(self) -> None:
        self.clear()

    def clear(self) -> None:
        self.valid = False
        self.confidence = 0.0
        self.x1 = self.y1 = self.x2 = self.y2 = 0.0
        self.px1 = self.py1 = self.px2 = self.py2 = 0.0
        self.cx = self.cy = self.pcx = self.pcy = 0.0
        self.dx = self.dy = self.dist = 0.0
        self.vx = self.vy = self.ax = self.ay = self.speed = 0.0

    def box(self):
        return (self.x1, self.y1, self.x2, self.y2)

    def predicted_box(self):
        return (self.px1, self.py1, self.px2, self.py2)


# ── one tracked object ──────────────────────────────────────────────────────

class AnchorTrack:
    """History, motion estimate and prediction for ONE tracked object. Tracks
    share no state, so several can run side by side (see ``AnchorBank``)."""

    def __init__(self, params: Optional[AnchorParams] = None) -> None:
        self.params = params if params is not None else AnchorParams()
        self._t = [0.0] * _CAP
        self._x = [0.0] * _CAP
        self._y = [0.0] * _CAP
        self._w = [0.0] * _CAP
        self._h = [0.0] * _CAP
        self._bounds_w = self._bounds_h = 0.0
        self.pred = Prediction()
        self.reset()

    def set_bounds(self, width: float, height: float) -> None:
        """Image size the predicted box is clamped into (<= 0: no clamp)."""
        self._bounds_w, self._bounds_h = float(width), float(height)

    def reset(self) -> None:
        self._n = 0
        self._head = 0
        self.t_start = 0.0
        self.last_t = 0.0
        self.last_box = None            # (x1, y1, x2, y2) of the newest sample
        self.cx = self.cy = 0.0         # newest raw centre
        self.w = self.h = 0.0           # newest box size
        self.vx = self.vy = 0.0         # velocity used for prediction (px/s)
        self.ax = self.ay = 0.0         # acceleration estimate (px/s^2), before accel_comp
        self.vw = self.vh = 0.0         # size velocity (px/s)
        self.sig_thr = 0.0              # speed (px/s) below which the estimate is indistinguishable from jitter
        self.acc_thr = 0.0              # same for the acceleration magnitude (px/s^2)
        self._dt_typ = 0.0              # typical interval between samples (learned)
        self._coast_s = _COAST_START_S
        self._gap_s = _MAX_GAP_S
        self._have_smooth = False
        self.err_ema = 0.0              # one-step prediction error / box size
        self.stab_ema = 0.5             # frame-to-frame box overlap
        self.turn_ema = 0.0             # direction-change signal 0..1
        self.det_conf = 1.0
        self.confidence = 0.0
        self.pred.clear()

    @property
    def samples(self) -> int:
        return self._n

    # ── update ──────────────────────────────────────────────────────────
    def update(self, t: float, box: Sequence[float], det_conf: float = 1.0) -> bool:
        """Feed one observation of the box at time ``t`` (seconds). Returns
        False when the sample was ignored (non-finite / out of order)."""
        try:
            x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
        except (TypeError, ValueError, IndexError):
            return False
        if not math.isfinite(t + x1 + y1 + x2 + y2):      # one check: any nan/inf makes the sum non-finite
            return False
        if self._n > 0 and t <= self.last_t:
            return False
        w = max(1.0, abs(x2 - x1))
        h = max(1.0, abs(y2 - y1))
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        self.det_conf = clamp01(det_conf) if math.isfinite(det_conf) else 0.0
        p = self.params

        if self._n == 0:
            self._restart(t, x1, y1, x2, y2, cx, cy, w, h)
            return True
        dt = t - self.last_t
        if dt > self._gap_s:
            self._restart(t, x1, y1, x2, y2, cx, cy, w, h)
            return True
        if self._dt_typ == 0.0:                       # second sample: learn the frame interval
            self._learn_dt(dt)
        # An impossible speed means a different object (id reassignment,
        # detection jump): never fold it into this track's history.
        if math.hypot(cx - self.cx, cy - self.cy) / dt > p.max_speed_px_s:
            self._restart(t, x1, y1, x2, y2, cx, cy, w, h)
            return True

        # One-step prediction error from the state BEFORE this sample.
        size = max(4.0, 0.5 * (self.w + self.h))
        coast = dt > self._coast_s
        if coast:
            decay = math.exp(-dt / _COAST_DECAY_S)
            travel = _COAST_DECAY_S * (1.0 - decay)
            ex, ey = self.cx + self.vx * travel, self.cy + self.vy * travel
        else:
            a_c = p.accel_comp if p.mode != 'linear' else 0.0
            ex = self.cx + self.vx * dt + 0.5 * self.ax * a_c * dt * dt
            ey = self.cy + self.vy * dt + 0.5 * self.ay * a_c * dt * dt
        err = math.hypot(cx - ex, cy - ey) / size
        if self._n >= 3:
            self.err_ema += (err - self.err_ema) * _alpha(dt, _TAU_ERR_S)
        # Stability: overlap with where the previous state expected the box to be (the last box
        # moved by the expected displacement). Plain frame-to-frame overlap would fall with speed
        # alone and wrongly mark a fast, perfectly steady target as unstable.
        lb = self.last_box
        sx, sy = ex - self.cx, ey - self.cy
        self.stab_ema += (box_iou((lb[0] + sx, lb[1] + sy, lb[2] + sx, lb[3] + sy), (x1, y1, x2, y2))
                          - self.stab_ema) * _alpha(dt, _TAU_STAB_S)

        # Direction change: compare the instantaneous heading with the estimate.
        ivx, ivy = (cx - self.cx) / dt, (cy - self.cy) / dt
        sp_i, sp_e = math.hypot(ivx, ivy), math.hypot(self.vx, self.vy)
        if sp_i > _MIN_TURN_SPEED and sp_e > _MIN_TURN_SPEED:
            cos = (ivx * self.vx + ivy * self.vy) / (sp_i * sp_e)
            turn = (1.0 - max(-1.0, min(1.0, cos))) * 0.5
        else:
            turn = 0.0
        self.turn_ema += (turn - self.turn_ema) * _alpha(dt, _TAU_TURN_S)

        if coast:
            # Missed detections: keep a decayed velocity, drop acceleration and
            # the history spanning the gap.
            self.vx *= decay
            self.vy *= decay
            self.ax = self.ay = 0.0
            self.vw = self.vh = 0.0
            self._n = 0
        else:
            self._learn_dt(dt)
            if p.mode == 'adaptive' and err > _SPIKE_ERR:
                self._n = min(self._n, 1)       # sudden direction change: forget the old heading

        self._push(t, cx, cy, w, h)
        self.last_t = t
        self.cx, self.cy, self.w, self.h = cx, cy, w, h
        self.last_box = (x1, y1, x2, y2)
        self._estimate(dt)
        self._update_confidence()
        return True

    def _learn_dt(self, dt: float) -> None:
        self._dt_typ = dt if self._dt_typ == 0.0 else self._dt_typ + (dt - self._dt_typ) * _TAU_DT_S
        self._coast_s = max(_COAST_START_S, _COAST_FRAMES * self._dt_typ)
        self._gap_s = max(_MAX_GAP_S, _GAP_FRAMES * self._dt_typ)

    def miss(self, t: float) -> None:
        """No detection this frame. Nothing is stored; ``predict(now=t)``
        coasts, and the track expires on its own after ``_MAX_GAP_S``."""

    def _restart(self, t, x1, y1, x2, y2, cx, cy, w, h) -> None:
        det = self.det_conf
        self.reset()
        self.det_conf = det
        self._push(t, cx, cy, w, h)
        self.t_start = self.last_t = t
        self.cx, self.cy, self.w, self.h = cx, cy, w, h
        self.last_box = (x1, y1, x2, y2)
        self._update_confidence()

    def _push(self, t, cx, cy, w, h) -> None:
        i = self._head
        self._t[i], self._x[i], self._y[i], self._w[i], self._h[i] = t, cx, cy, w, h
        self._head = (i + 1) % _CAP
        self._n = min(self._n + 1, _CAP)

    # ── estimation ──────────────────────────────────────────────────────
    def _estimate(self, dt: float) -> None:
        p = self.params
        m = min(self._n, p.history_len)
        if m < 2:
            return                                  # keep the (decayed) previous estimate
        want_accel = p.mode != 'linear' and p.accel_comp > 0.0 and m >= _MIN_ACCEL_SAMPLES
        fit = self._fit(m, want_accel)
        if fit is None:
            return
        rvx, rvy, rax, ray, rvw, rvh, sxx, acc_var, tau_bar = fit
        # Standard error of the speed estimate: detector jitter (as observed through the
        # one-step innovation) / sqrt(Sxx) for a line fit.
        size = max(4.0, 0.5 * (self.w + self.h))
        sigma = max(_SIGMA_FLOOR_PX, self.err_ema * size)
        self.sig_thr = _SIGNIFICANCE_K * sigma / math.sqrt(max(sxx, 1e-12))
        if acc_var > 0.0:
            # Acceleration from the quadratic fit, kept only as far as it clears its own noise floor
            # (std error 2 * sigma * sqrt(det / D)); ramps 0 -> 1 over [thr, 2 thr]. The velocity is the
            # LINEAR fit's (the line's slope is the velocity at the window's mean time tau_bar), moved to
            # the newest sample by the kept acceleration -- so velocity noise stays the low linear-fit
            # noise unless the acceleration is real.
            self.acc_thr = _SIGNIFICANCE_K * 2.0 * sigma * math.sqrt(acc_var)
            keep = clamp01((math.hypot(rax, ray) - self.acc_thr) / self.acc_thr) if self.acc_thr > 0.0 else 1.0
            rax *= keep
            ray *= keep
            rvx -= rax * tau_bar
            rvy -= ray * tau_bar
        else:
            self.acc_thr = 0.0
            rax = ray = 0.0
        if p.mode in ('linear', 'velocity_accel') or not self._have_smooth:
            self.vx, self.vy, self.ax, self.ay, self.vw, self.vh = rvx, rvy, rax, ray, rvw, rvh
            self._have_smooth = p.mode in ('smoothed', 'adaptive')
        else:
            a = _alpha(dt, _TAU_MIN_S + _TAU_SPAN_S * clamp01(p.smoothing))
            self.vx += (rvx - self.vx) * a
            self.vy += (rvy - self.vy) * a
            self.ax += (rax - self.ax) * a
            self.ay += (ray - self.ay) * a
            self.vw += (rvw - self.vw) * a
            self.vh += (rvh - self.vh) * a

    def _fit(self, m: int, want_accel: bool):
        """LSQ over the newest ``m`` samples -> (vx, vy, ax, ay, vw, vh) or None."""
        head, cap = self._head, _CAP
        t_last = self._t[(head - 1) % cap]
        s1 = s2 = s3 = s4 = 0.0
        x0 = x1 = x2 = y0 = y1 = y2 = 0.0
        w0 = w1 = h0 = h1 = 0.0
        for i in range(m):
            j = (head - m + i) % cap
            tau = self._t[j] - t_last
            t2 = tau * tau
            s1 += tau
            s2 += t2
            x, y, w, h = self._x[j], self._y[j], self._w[j], self._h[j]
            x0 += x
            y0 += y
            x1 += x * tau
            y1 += y * tau
            w0 += w
            h0 += h
            w1 += w * tau
            h1 += h * tau
            if want_accel:
                s3 += t2 * tau
                s4 += t2 * t2
                x2 += x * t2
                y2 += y * t2
        s0 = float(m)
        det = s0 * s2 - s1 * s1
        if det <= 1e-12:
            return None
        sxx = det / s0
        tau_bar = s1 / s0                       # mean time of the window relative to the newest sample (<= 0)
        vlx = (s0 * x1 - s1 * x0) / det         # linear-fit slope = velocity at the window's mean time
        vly = (s0 * y1 - s1 * y0) / det
        vw = (s0 * w1 - s1 * w0) / det
        vh = (s0 * h1 - s1 * h0) / det
        if want_accel:
            d = _det3(s0, s1, s2, s1, s2, s3, s2, s3, s4)
            if abs(d) > 1e-15 * max(1e-30, s0 * s2 * s4):
                ax = 2.0 * _det3(s0, s1, x0, s1, s2, x1, s2, s3, x2) / d
                ay = 2.0 * _det3(s0, s1, y0, s1, s2, y1, s2, s3, y2) / d
                return vlx, vly, ax, ay, vw, vh, sxx, det / d, tau_bar
        return vlx, vly, 0.0, 0.0, vw, vh, sxx, 0.0, tau_bar

    def _update_confidence(self) -> None:
        support = clamp01((self.last_t - self.t_start) / _SUPPORT_FULL_S)
        accuracy = 1.0 - clamp01(self.err_ema / _ERR_ZERO_FIT)
        self.confidence = support * accuracy * (0.5 + 0.5 * self.det_conf)

    # ── query ───────────────────────────────────────────────────────────
    def predict(self, horizon_s: Optional[float] = None, now: Optional[float] = None) -> Prediction:
        """Predicted box after ``horizon_s`` (default: the configured horizon),
        measured from ``now`` (default: the newest sample). Returns the track's
        reused ``Prediction``; ``valid`` is False when there is no usable lead
        (too little history, expired, or the box is stationary)."""
        p = self.params
        r = self.pred
        r.clear()
        if self._n == 0:
            return r
        H = p.horizon_s if horizon_s is None else max(0.0, horizon_s)
        lag = 0.0 if now is None else max(0.0, now - self.last_t)
        r.cx, r.cy = self.cx, self.cy
        x1, y1, x2, y2 = self.last_box
        r.x1, r.y1, r.x2, r.y2 = x1, y1, x2, y2
        r.px1, r.py1, r.px2, r.py2 = x1, y1, x2, y2
        r.pcx, r.pcy = self.cx, self.cy
        r.vx, r.vy = self.vx, self.vy
        r.speed = math.hypot(self.vx, self.vy)
        r.confidence = self.confidence
        if lag > self._gap_s or self._n < 2:
            return r

        # Effective acceleration and the time the lead covers.
        a_c = 0.0 if p.mode == 'linear' else clamp01(p.accel_comp)
        ax, ay = self.ax * a_c, self.ay * a_c
        conf = self.confidence
        if lag > self._coast_s:
            decay = math.exp(-lag / _COAST_DECAY_S)
            span = _COAST_DECAY_S * (1.0 - decay)        # bounded coasting travel
            ax = ay = 0.0
            conf *= decay
            vx, vy, T = self.vx, self.vy, span
        else:
            vx, vy, T = self.vx, self.vy, lag + H
        r.ax, r.ay = ax, ay

        speed = math.hypot(vx, vy)
        dz = _STATIONARY_BOX_PER_S * max(1.0, min(self.w, self.h))
        if speed <= dz or T <= 0.0:
            return r
        keep = (speed - dz) / speed
        thr = self.sig_thr
        if thr > 0.0:                                      # ramp 0 -> 1 between thr and 2*thr: don't extrapolate jitter
            keep *= clamp01((speed - thr) / thr)
            if keep <= 0.0:
                return r
        dx = _axis_lead(vx * keep, ax, T)
        dy = _axis_lead(vy * keep, ay, T)

        gate = clamp01((conf - p.conf_threshold) / _CONF_RAMP) if conf > p.conf_threshold else 0.0
        scale = max(0.0, p.strength) * gate
        if p.mode == 'adaptive':
            q = (0.4 * conf + 0.3 * clamp01((self.stab_ema - 0.3) / 0.5)
                 + 0.3 * clamp01(1.0 - 2.0 * self.turn_ema))
            scale *= 0.3 + 0.7 * q
        dx *= scale
        dy *= scale
        r.confidence = conf

        dist = math.hypot(dx, dy)
        cap = p.max_distance_px
        if cap > 0.0 and dist > cap:
            f = cap / dist
            dx *= f
            dy *= f
            dist = cap

        # Predicted size, limited so noise can't blow the box up or collapse it.
        pw = min(self.w * (1.0 + _MAX_SIZE_CHANGE), max(self.w * (1.0 - _MAX_SIZE_CHANGE), self.w + self.vw * T * gate))
        ph = min(self.h * (1.0 + _MAX_SIZE_CHANGE), max(self.h * (1.0 - _MAX_SIZE_CHANGE), self.h + self.vh * T * gate))

        pcx, pcy = self.cx + dx, self.cy + dy
        bw, bh = self._bounds_w, self._bounds_h
        if bw > 0.0 and bh > 0.0:                          # keep the box inside the image
            pcx = min(max(pcx, 0.5 * pw), max(0.5 * pw, bw - 0.5 * pw))
            pcy = min(max(pcy, 0.5 * ph), max(0.5 * ph, bh - 0.5 * ph))
            dx, dy = pcx - self.cx, pcy - self.cy
            dist = math.hypot(dx, dy)
        r.pcx, r.pcy, r.dx, r.dy, r.dist = pcx, pcy, dx, dy, dist
        r.px1, r.py1, r.px2, r.py2 = pcx - 0.5 * pw, pcy - 0.5 * ph, pcx + 0.5 * pw, pcy + 0.5 * ph
        r.valid = gate > 0.0
        return r


def _det3(a, b, c, d, e, f, g, h, i) -> float:
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)


def _axis_lead(v: float, a: float, T: float) -> float:
    """Displacement over ``T`` seconds for one axis. Decelerating (``a``
    against ``v``) stops at standstill instead of reversing; speeding up is
    capped so acceleration adds at most the current speed over ``T`` (a stale
    deceleration estimate after a hard stop otherwise extrapolates backwards)."""
    if a * v < 0.0:
        if T > -v / a:
            return -v * v / (2.0 * a)
    elif T > 0.0 and a != 0.0:
        lim = abs(v) / T
        if abs(a) > lim:
            a = math.copysign(lim, a)
    return v * T + 0.5 * a * T * T


# ── several objects ─────────────────────────────────────────────────────────

class AnchorBank:
    """Independent tracks keyed by track id. A new id starts a fresh track
    (history reset), so a reassigned id never inherits another object's motion."""

    def __init__(self, params: Optional[AnchorParams] = None) -> None:
        self.params = params if params is not None else AnchorParams()
        self._tracks: Dict[int, AnchorTrack] = {}

    def __len__(self) -> int:
        return len(self._tracks)

    def get(self, track_id: int) -> Optional[AnchorTrack]:
        return self._tracks.get(track_id)

    def update(self, track_id: int, t: float, box: Sequence[float], det_conf: float = 1.0) -> AnchorTrack:
        tr = self._tracks.get(track_id)
        if tr is None:
            tr = self._tracks[track_id] = AnchorTrack(self.params)
        tr.update(t, box, det_conf)
        return tr

    def drop(self, track_id: int) -> None:
        self._tracks.pop(track_id, None)

    def prune(self, now: float, max_age_s: float = 2.0) -> int:
        """Drop tracks not updated for ``max_age_s``; returns how many."""
        stale = [k for k, tr in self._tracks.items() if now - tr.last_t > max_age_s]
        for k in stale:
            del self._tracks[k]
        return len(stale)

    def clear(self) -> None:
        self._tracks.clear()

    def set_bounds(self, width: float, height: float) -> None:
        for tr in self._tracks.values():
            tr.set_bounds(width, height)


# ── FOV integration ─────────────────────────────────────────────────────────

_TAU_ADJ_S = 0.25               # smoothing of the FOV adjustments (anti-oscillation); swept 0.15-0.5 s: 0.25 keeps
                                # ~all the benefit while cutting the added FOV jerk from x1.3-1.5 to x1.15-1.25
_ADJ_DEADBAND = 0.01
_EDGE_TOWARD_CAP = 0.25         # most an approaching target may push the edge ratio
_EDGE_TOWARD_MIN = 0.5          # no anticipatory expansion for predictions well inside the FOV


class AnchorView:
    """Per-frame facts about the locked target's prediction, computed once when
    the lock is matched (the crosshair/FOV geometry is known there) and read
    cheaply by ``FovAdjuster`` when the next frame's FOV is decided."""

    __slots__ = ('valid', 'confidence', 'edge_pred', 'radial_speed', 'speed_now',
                 'speed_pred', 'turn_pred', 't')

    def __init__(self) -> None:
        self.invalidate()

    def invalidate(self) -> None:
        self.valid = False
        self.confidence = 0.0
        self.edge_pred = 0.0
        self.radial_speed = 0.0
        self.speed_now = 0.0
        self.speed_pred = 0.0
        self.turn_pred = 0.0
        self.t = 0.0

    def fill(self, pred: Prediction, horizon_s: float, crosshair_x: float, crosshair_y: float,
             fov_w: float, fov_h: float, elliptical: bool, now: float) -> None:
        """Derive the FOV-relevant numbers from ``pred`` (valid or not)."""
        if not pred.valid:
            self.invalidate()
            return
        self.valid = True
        self.t = now
        self.confidence = pred.confidence
        self.edge_pred = edge_ratio(pred.predicted_box(), crosshair_x, crosshair_y, fov_w, fov_h, elliptical)
        vx, vy = pred.vx, pred.vy
        self.speed_now = math.hypot(vx, vy)
        ex, ey = vx + pred.ax * horizon_s, vy + pred.ay * horizon_s
        self.speed_pred = math.hypot(ex, ey)
        # outward speed: component of the velocity along crosshair -> target
        rx, ry = pred.cx - crosshair_x, pred.cy - crosshair_y
        rn = math.hypot(rx, ry)
        self.radial_speed = (vx * rx + vy * ry) / rn if rn > 1e-6 else 0.0
        if self.speed_now > _MIN_TURN_SPEED and self.speed_pred > _MIN_TURN_SPEED:
            cos = (vx * ex + vy * ey) / (self.speed_now * self.speed_pred)
            self.turn_pred = (1.0 - max(-1.0, min(1.0, cos))) * 0.5
        else:
            self.turn_pred = 0.0


class FovAdjuster:
    """Turns an ``AnchorView`` into small, smoothed additive corrections for the
    existing Adaptive / Dynamic FOV signals. All corrections are 0 whenever the
    view is invalid or stale, the confidence is low, or the influence is 0, so
    the FOV code behaves exactly as before in those cases. The internal state is
    EMA-filtered (``_TAU_ADJ_S``) and only the *output* has a dead-band, so a
    flickering prediction can't make the FOV oscillate and a slow ramp can still
    build up; the FOV's own slew limiter additionally rate-limits the result."""

    __slots__ = ('edge_adj', 'speed_adj', 'turn_adj', '_e', '_s', '_u', '_t')

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.edge_adj = self.speed_adj = self.turn_adj = 0.0
        self._e = self._s = self._u = 0.0
        self._t = 0.0

    def update(self, view: AnchorView, edge_now: float, speed_ema: float, turn_ema: float,
               fov_ref: float, infl_adaptive: float, infl_dynamic: float, now: float) -> None:
        dt = now - self._t if self._t > 0.0 else 0.0
        self._t = now
        a = _alpha(dt, _TAU_ADJ_S) if 0.0 < dt < 1.0 else (1.0 if dt >= 1.0 else 0.0)
        e_t = s_t = u_t = 0.0
        if view.valid and now - view.t < _MAX_GAP_S:
            w_a = clamp01(infl_adaptive) * view.confidence
            w_d = clamp01(infl_dynamic) * view.confidence
            d = view.edge_pred - edge_now
            if d > 0.0:                       # heading for the boundary
                e_t = w_a * min(d, _EDGE_TOWARD_CAP) if view.edge_pred > _EDGE_TOWARD_MIN else 0.0
            else:                             # heading inward: half-weight benefit, never negative edge
                e_t = max(-edge_now, 0.5 * w_a * d)
            ds = view.speed_pred - view.speed_now
            if ds < 0.0:
                ds *= 0.5                     # slowing down counts for less
            s_t = w_d * (ds + max(0.0, view.radial_speed))
            s_t = max(-speed_ema, min(1.5 * max(1.0, fov_ref), s_t))
            u_t = max(0.0, w_d * view.turn_pred - turn_ema)
        self._e += (e_t - self._e) * a
        self._s += (s_t - self._s) * a
        self._u += (u_t - self._u) * a
        self.edge_adj = self._e if abs(self._e) >= _ADJ_DEADBAND else 0.0
        self.speed_adj = self._s if abs(self._s) >= 1.0 else 0.0
        self.turn_adj = self._u if abs(self._u) >= _ADJ_DEADBAND else 0.0
