"""Anchor prediction: where the aim anchor will be a short time from now.

The "anchor" is the aim point ``process_aiming()`` steers at (the box centre
plus the configured target-area offset), not a detector anchor box — the
detector is never touched. A time-based constant-acceleration tracker follows
that point and the box size, and ``predict()`` returns the point advanced by a
horizon, scaled by how much the estimate can be trusted.

Algorithm — critically damped g-h-k (alpha-beta-gamma) filter. Per sample with
elapsed time ``dt`` (taken from the caller's timestamps, never a fixed FPS)::

    theta = exp(-dt / tau)
    g = 1 - theta^3,  h = 1.5 (1 - theta^2)(1 - theta),  k = 0.5 (1 - theta)^3
    x += v dt + a dt^2 / 2 ;  v += a dt          (predict)
    r  = z - x
    x += g r ;  v += h r / dt ;  a += 2 k r / dt^2   (correct)

``tau`` comes from the Smoothing setting, so the filter behaves the same at 30
or 300 samples per second. It is O(1), allocation-free per sample and needs no
numpy, so it is importable and testable anywhere.

FOV-based: the lead cap is a percentage of the *effective* FOV (the one the
Adaptive/Dynamic shrink modes are currently driving, so it tightens with it),
and the predicted point is clamped inside that FOV (rectangle, or ellipse when
the circular FOV filter is on) so the prediction can never aim outside the
region targets are selected from.

Safeguards: non-finite / out-of-order samples are ignored; a measurement that
implies more than ``max_speed_px_s`` is a detection jump (new target) and
restarts the filter; a gap longer than ``_COAST_START_S`` coasts with decaying
velocity and no acceleration (bounded drift), and one longer than
``_MAX_GAP_S`` restarts it; the lead is zero for a (near) stationary target,
never reverses past a standstill when decelerating, is capped at
``max_distance_px`` and fades out as the prediction confidence approaches the
threshold.

``adjust_for_mode()`` maps the existing FOV-shrink mode (adaptive / dynamic)
and the tracker's smoothed signals to a strength / horizon / agility
modulation, so the same Adaptive and Dynamic settings drive prediction
trust instead of a competing selector.
"""
from __future__ import annotations

import math

# A gap longer than this between samples makes the old state meaningless.
_MAX_GAP_S = 0.30
# A gap longer than this coasts: velocity decays, acceleration is dropped.
_COAST_START_S = 0.10
_COAST_DECAY_S = 0.15
# Smoothing 0..1 -> filter time constant (seconds).
_TAU_MIN_S = 0.02
_TAU_SPAN_S = 0.20
# Below this many box-heights/s the target counts as stationary.
_STATIONARY_BOX_PER_S = 0.25
# Seconds of tracking (since a restart) for full confidence — time-based, so it
# is the same at any sample rate.
_SUPPORT_FULL_S = 0.12
# Innovation (box-normalised) EMA time constant and the value that zeroes "fit".
_TAU_RESID_S = 0.15
_RESID_ZERO_FIT = 0.35
# Size EMA time constant.
_TAU_SIZE_S = 0.10
# Confidence ramp above the threshold, so the lead fades in instead of jumping.
_CONF_RAMP = 0.25
# Number of recent anchor positions kept for the trajectory display.
TRAIL_LEN = 16

_DEFAULTS = {
    'horizon_s': 0.010,
    'strength': 1.0,
    'smoothing': 0.4,
    'velocity_influence': 1.0,
    'accel_influence': 0.35,
    'max_distance_pct': 25.0,
    'max_distance_px': 0.0,
    'confidence_threshold': 0.35,
    'max_speed_px_s': 1200.0,
}


def _clamp01(v: float) -> float:
    return 0.0 if v < 0.0 else 1.0 if v > 1.0 else v


def _ema_alpha(dt: float, tau: float) -> float:
    return 1.0 - math.exp(-dt / tau)


class AnchorParams:
    """Settings read by the predictor. Mutated in place from Config each frame
    so the hot loop allocates nothing."""

    __slots__ = tuple(_DEFAULTS)

    def __init__(self) -> None:
        for name, value in _DEFAULTS.items():
            setattr(self, name, value)

    def update_from(self, config) -> "AnchorParams":
        """Copy the anchor_* / prediction_* settings off ``config`` (missing
        attributes keep their defaults; non-finite values are ignored)."""
        self._set('horizon_s', getattr(config, 'prediction_horizon_ms', None), 0.001)
        self._set('strength', getattr(config, 'anchor_strength', None))
        self._set('smoothing', getattr(config, 'anchor_smoothing', None))
        self._set('velocity_influence', getattr(config, 'anchor_velocity_influence', None))
        self._set('accel_influence', getattr(config, 'anchor_accel_influence', None))
        self._set('max_distance_pct', getattr(config, 'anchor_max_distance_pct', None))
        self._set('confidence_threshold', getattr(config, 'anchor_confidence_threshold', None))
        self._set('max_speed_px_s', getattr(config, 'prediction_max_velocity', None))
        # The lead cap is a share of the shorter side of the FOV in effect now.
        fw, fh = effective_fov(config)
        self.max_distance_px = (max(0.0, self.max_distance_pct) / 100.0 * min(fw, fh)
                                if self.max_distance_pct > 0.0 and fw > 0.0 else 0.0)
        return self

    def _set(self, name: str, value, scale: float = 1.0) -> None:
        try:
            v = float(value) * scale
        except (TypeError, ValueError):
            return
        if math.isfinite(v):
            setattr(self, name, v)


def effective_fov(config) -> tuple:
    """(width, height) of the FOV in effect this frame — the shrunk one while
    "Reduce FOV on Active Target" is acting, else the configured size."""
    fw = float(getattr(config, 'fov_effective_size', 0) or getattr(config, 'fov_size', 0) or 0)
    fh = float(getattr(config, 'fov_effective_height', 0) or getattr(config, 'fov_height', 0) or fw)
    return fw, fh


class ModeAdjust:
    """Per-frame modulation produced by ``adjust_for_mode`` (reused, not
    reallocated): multiplies the lead strength and horizon, and sets how
    strongly the filter tightens its time constant on a sudden change."""

    __slots__ = ('strength_scale', 'horizon_scale', 'agility')

    def __init__(self) -> None:
        self.neutral()

    def neutral(self) -> None:
        self.strength_scale = 1.0
        self.horizon_scale = 1.0
        self.agility = 0.0


def adjust_for_mode(mode: str, tracker, out: ModeAdjust) -> ModeAdjust:
    """Fill ``out`` for the FOV-shrink ``mode`` from the tracker's smoothed signals.

    * ``adaptive`` — trust follows movement consistency: steady direction,
      stable boxes and high confidence keep the full lead; a direction change
      or shaky detections cut it (never below 35 %), and it eases off towards
      the FOV edge. The horizon is unchanged.
    * ``dynamic``  — reacts quickly: the lead is muted less, the horizon
      shortens right after a direction change and stretches slightly while
      the motion is clean, and the filter tightens its time constant on large
      innovations (``agility``) to cut tracking lag.
    * anything else (``fixed``) — neutral.
    """
    out.neutral()
    if mode not in ('adaptive', 'dynamic') or tracker is None:
        return out
    turn_q = _clamp01(1.0 - 2.0 * float(tracker.turn_ema))
    stab_q = _clamp01((float(tracker.stab_ema) - 0.3) / 0.5)
    conf_q = _clamp01(float(tracker.conf_ema))
    cons = 0.35 * turn_q + 0.25 * stab_q + 0.20 * _clamp01(float(tracker.hit_ema)) + 0.20 * conf_q
    if mode == 'adaptive':
        # Near the FOV boundary a lead would push the aim out of the region
        # targets are selected from, so back off (same edge signal the
        # adaptive FOV shrink uses).
        edge_q = _clamp01((0.9 - float(tracker.edge_ratio)) / 0.45)
        out.strength_scale = (0.35 + 0.65 * cons) * (0.6 + 0.4 * edge_q)
    else:
        out.strength_scale = 0.60 + 0.40 * cons
        turn_burst = _clamp01(3.0 * float(tracker.turn_ema))
        out.horizon_scale = (1.0 - 0.5 * turn_burst) * (0.85 + 0.40 * cons)
        out.agility = 1.0
    return out


class AnchorPredictor:
    """One tracked anchor. Keep one instance per tracked object — instances
    share no state, so several can run side by side."""

    def __init__(self, params: AnchorParams | None = None) -> None:
        self.params = params if params is not None else AnchorParams()
        self._trail_x = [0.0] * TRAIL_LEN
        self._trail_y = [0.0] * TRAIL_LEN
        self._fov = None
        self.reset()

    def reset(self) -> None:
        """Forget the target (call on target loss / switch)."""
        self.n = 0                 # samples since the last restart
        self.last_t = 0.0
        self.t_start = 0.0         # timestamp of the sample that started this track
        self.x = self.y = 0.0
        self.vx = self.vy = 0.0
        self.ax = self.ay = 0.0
        self.w = self.h = 0.0
        self.resid_ema = 0.0
        self.det_conf = 1.0
        self.confidence = 0.0
        self.pred_x = self.pred_y = 0.0
        self.lead_dist = 0.0
        self._trail_n = 0
        self._trail_i = 0

    def set_fov(self, cx: float, cy: float, width: float, height: float, ellipse: bool = False) -> None:
        """FOV the predicted point is clamped into (width <= 0 disables it).
        Kept across ``reset()`` — it describes the screen, not the target."""
        self._fov = (cx, cy, width, height, ellipse) if width > 0.0 and height > 0.0 else None

    # ── update ───────────────────────────────────────────────────────────
    def update(self, x: float, y: float, t: float, w: float, h: float,
               det_conf: float = 1.0, agility: float = 0.0) -> None:
        """Feed one measurement of the anchor at time ``t`` (seconds)."""
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(t)
                and math.isfinite(w) and math.isfinite(h)):
            return
        if self.n > 0 and t <= self.last_t:
            return      # out of order / duplicate timestamp: nothing new
        w = max(1.0, w)
        h = max(1.0, h)
        self.det_conf = _clamp01(det_conf) if math.isfinite(det_conf) else 0.0

        if self.n == 0:
            self._restart(x, y, t, w, h)
            return
        dt = t - self.last_t
        if dt > _MAX_GAP_S:
            self._restart(x, y, t, w, h)
            return

        # A measurement implying an impossible speed is a different target.
        if math.hypot(x - self.x, y - self.y) / dt > self.params.max_speed_px_s:
            self._restart(x, y, t, w, h)
            return

        if dt > _COAST_START_S:
            # Missed detections: don't extrapolate acceleration across the gap.
            decay = math.exp(-dt / _COAST_DECAY_S)
            travel = _COAST_DECAY_S * (1.0 - decay)
            self.x += self.vx * travel
            self.y += self.vy * travel
            self.vx *= decay
            self.vy *= decay
            self.ax = self.ay = 0.0
        else:
            half_dt2 = 0.5 * dt * dt
            self.x += self.vx * dt + self.ax * half_dt2
            self.y += self.vy * dt + self.ay * half_dt2
            self.vx += self.ax * dt
            self.vy += self.ay * dt

        rx = x - self.x
        ry = y - self.y
        size = max(4.0, 0.5 * (self.w + self.h))
        resid = math.hypot(rx, ry) / size

        if self.n == 1:
            # Second sample: two-point velocity, so a steady target converges at once.
            self.x, self.y = x, y
            self.vx = rx / dt + self.vx
            self.vy = ry / dt + self.vy
        else:
            tau = _TAU_MIN_S + _TAU_SPAN_S * _clamp01(self.params.smoothing)
            if agility > 0.0:
                excess = _clamp01((resid - 0.10) / 0.40)
                tau /= 1.0 + 3.0 * _clamp01(agility) * excess
            theta = math.exp(-dt / tau)
            omt = 1.0 - theta
            g = 1.0 - theta * theta * theta
            hh = 1.5 * (1.0 - theta * theta) * omt
            kk = 0.5 * omt * omt * omt
            self.x += g * rx
            self.y += g * ry
            self.vx += hh * rx / dt
            self.vy += hh * ry / dt
            self.ax += 2.0 * kk * rx / (dt * dt)
            self.ay += 2.0 * kk * ry / (dt * dt)

        a_size = _ema_alpha(dt, _TAU_SIZE_S)
        self.w += (w - self.w) * a_size
        self.h += (h - self.h) * a_size
        if self.n >= 2:    # the second sample's innovation only reflects the missing velocity
            self.resid_ema += (resid - self.resid_ema) * _ema_alpha(dt, _TAU_RESID_S)
        self.n += 1
        self.last_t = t
        self._push_trail()
        self._update_confidence()

    def _restart(self, x: float, y: float, t: float, w: float, h: float) -> None:
        det = self.det_conf
        self.reset()
        self.det_conf = det
        self.x, self.y = x, y
        self.w, self.h = w, h
        self.n = 1
        self.last_t = self.t_start = t
        self._push_trail()
        self._update_confidence()

    def _push_trail(self) -> None:
        i = self._trail_i
        self._trail_x[i] = self.x
        self._trail_y[i] = self.y
        self._trail_i = (i + 1) % TRAIL_LEN
        if self._trail_n < TRAIL_LEN:
            self._trail_n += 1

    def _update_confidence(self) -> None:
        support = _clamp01((self.last_t - self.t_start) / _SUPPORT_FULL_S)
        fit = 1.0 - _clamp01(self.resid_ema / _RESID_ZERO_FIT)
        self.confidence = support * fit * (0.5 + 0.5 * self.det_conf)

    # ── query ────────────────────────────────────────────────────────────
    def predict(self, horizon_scale: float = 1.0, strength_scale: float = 1.0):
        """Predicted anchor position after the horizon; also sets ``pred_x``,
        ``pred_y``, ``lead_dist`` and keeps ``confidence`` current. Returns the
        filtered current position when there is no usable lead."""
        p = self.params
        self.pred_x, self.pred_y = self.x, self.y
        self.lead_dist = 0.0
        if self.n < 2:
            return self.pred_x, self.pred_y

        speed = math.hypot(self.vx, self.vy)
        dz = _STATIONARY_BOX_PER_S * max(1.0, min(self.w, self.h))
        if speed <= dz:
            return self.pred_x, self.pred_y
        keep = (speed - dz) / speed
        vel = max(0.0, p.velocity_influence)
        acc = max(0.0, p.accel_influence)
        horizon = max(0.0, p.horizon_s * horizon_scale)

        dx = _axis_lead(self.vx * keep * vel, self.ax * acc, horizon)
        dy = _axis_lead(self.vy * keep * vel, self.ay * acc, horizon)

        conf_gate = _clamp01((self.confidence - p.confidence_threshold) / _CONF_RAMP) \
            if self.confidence > p.confidence_threshold else 0.0
        scale = max(0.0, p.strength) * max(0.0, strength_scale) * conf_gate
        dx *= scale
        dy *= scale

        dist = math.hypot(dx, dy)
        cap = p.max_distance_px
        if cap > 0.0 and dist > cap:
            f = cap / dist
            dx *= f
            dy *= f
            dist = cap
        fov = self._fov
        if fov is not None and dist > 0.0:
            fx, fy = _clamp_to_fov(self.x + dx, self.y + dy, fov)
            # Only pull the point back when the current anchor is itself inside;
            # otherwise "back inside" would flip the lead's direction.
            if _inside_fov(self.x, self.y, fov):
                dx, dy = fx - self.x, fy - self.y
                dist = math.hypot(dx, dy)
        self.pred_x = self.x + dx
        self.pred_y = self.y + dy
        self.lead_dist = dist
        return self.pred_x, self.pred_y

    def trail(self) -> list:
        """Recent filtered anchor positions, oldest first, as ``[(x, y), ...]``
        (allocates; call only when the trajectory is being displayed)."""
        n = self._trail_n
        start = (self._trail_i - n) % TRAIL_LEN
        return [(self._trail_x[(start + i) % TRAIL_LEN], self._trail_y[(start + i) % TRAIL_LEN])
                for i in range(n)]


def _axis_lead(v: float, a: float, horizon: float) -> float:
    """Displacement over ``horizon`` for one axis.

    Decelerating (``a`` against ``v``): stops at standstill instead of
    reversing. Speeding up: the speed gained over the horizon is capped at the
    current speed, so a stale or noisy acceleration estimate (typically the
    leftover of a hard stop, seen while the velocity estimate is still near
    zero) cannot invent a large lead from near-rest.
    """
    if a * v < 0.0:
        if horizon > -v / a:
            return -v * v / (2.0 * a)
    elif horizon > 0.0 and a != 0.0:
        lim = abs(v) / horizon
        if abs(a) > lim:
            a = math.copysign(lim, a)
    return v * horizon + 0.5 * a * horizon * horizon


def _inside_fov(px: float, py: float, fov) -> bool:
    cx, cy, w, h, ellipse = fov
    nx = (px - cx) / (0.5 * w)
    ny = (py - cy) / (0.5 * h)
    return (nx * nx + ny * ny <= 1.0) if ellipse else (abs(nx) <= 1.0 and abs(ny) <= 1.0)


def _clamp_to_fov(px: float, py: float, fov):
    cx, cy, w, h, ellipse = fov
    nx = (px - cx) / (0.5 * w)
    ny = (py - cy) / (0.5 * h)
    if ellipse:
        r = math.hypot(nx, ny)
        if r > 1.0:
            nx /= r
            ny /= r
    else:
        nx = max(-1.0, min(1.0, nx))
        ny = max(-1.0, min(1.0, ny))
    return cx + nx * 0.5 * w, cy + ny * 0.5 * h
