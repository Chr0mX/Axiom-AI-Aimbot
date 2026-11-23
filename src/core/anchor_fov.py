"""Anchor FOV: a FOV whose centre follows an anchor point.

The existing FOV modes (Fixed / Adaptive / Dynamic) only ever change the FOV's
*size*; its centre is always the crosshair.  ``fov_shrink_mode == 'anchor'``
adds a fourth mode where this module controls both the FOV centre and its
radius.  It is pure Python (no numpy/Qt/win32), so it is testable anywhere.

Responsibilities are kept in separate, small pieces:

* mode selection          -- :func:`anchor_fov_active`
* anchor source           -- :func:`resolve_detection_anchor`,
                             :func:`resolve_tracker_anchor`,
                             :func:`resolve_manual_anchor` (they fill one reused
                             :class:`AnchorSample`; nothing is tracked twice --
                             detections and ``TargetTracker`` are the inputs)
* anchor position         -- :func:`anchor_point` (box centre / top / bottom
                             + X/Y offsets)
* position smoothing      -- :meth:`AnchorFovController.step` (input low-pass,
                             exponential follow, speed cap -- frame-rate
                             independent, no jumps; the smoothing and speed cap
                             are fixed values, only the follow speed is a setting)
* radius                  -- :meth:`AnchorFovController._radius_target`
                             (independent of the position; the FOV Width / 2,
                             widened to ``SEARCH_RADIUS_PX`` while searching)
* lost-anchor handling    -- the ``hold`` / ``return`` / ``search`` branch in
                             :meth:`AnchorFovController.step`
* visualisation           -- :func:`publish_anchor_fov` writes the controller's
                             real state to runtime-only config fields; the
                             overlays read those, never a simulation.

Everything runs inline on the inference thread: a handful of float operations
per frame, a single reused sample object, no locks, no I/O.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

from .target_tracking import TrackState, box_iou, clamp

SOURCES = ('detection', 'tracker', 'manual')
POSITIONS = ('center', 'top', 'bottom')
LOST_MODES = ('hold', 'return', 'search')

# Controller phases (also what the overlays show).
PH_FOLLOW = 0     # an anchor exists and is being followed
PH_GRACE = 1      # anchor just vanished, within the lost delay: keep gliding to it
PH_HOLD = 2       # lost: FOV stays where it is
PH_RETURN = 3     # lost: FOV glides back to the crosshair
PH_SEARCH = 4     # lost: FOV stays and expands so a new anchor can be found
PHASE_NAMES = ('follow', 'grace', 'hold', 'return', 'search')

MIN_RADIUS_PX = 25.0         # hard floor of the FOV radius (FOV Width 50 px)
DEFAULT_RADIUS_PX = 100.0
_MAX_DT_S = 1.0            # a longer gap between frames is a stall, not motion
_SMOOTH_MAX_S = 0.25       # input low-pass time constant at smoothing == 1
_RADIUS_TAU_S = 0.20       # radius approaches its target with this time constant
SMOOTHING = 0.3            # input low-pass strength (0-1); was a slider, the default is what everyone used
MAX_SPEED_PX_S = 4000.0    # cap on how fast the FOV centre may move (never a jump); was a slider
LOST_DELAY_S = 0.15        # the anchor must be gone this long before the lost behaviour starts; was a slider
SEARCH_RADIUS_PX = 300.0   # how far the FOV grows while searching for a lost anchor (never below its own size)
_MIN_IOU = 0.10            # locked box <-> detection match


def anchor_fov_active(config) -> bool:
    """True when the Anchor FOV mode is selected (independent of the
    "Reduce FOV on Active Target" switch, which only drives the other modes)."""
    return str(getattr(config, 'fov_shrink_mode', '')) == 'anchor'


def fov_center(config) -> tuple:
    """(x, y) the FOV is centred on: the Anchor FOV controller's centre while
    that mode is active and has published one, else the crosshair."""
    if anchor_fov_active(config):
        x = getattr(config, 'fov_center_x', None)
        y = getattr(config, 'fov_center_y', None)
        if x is not None and y is not None:
            return float(x), float(y)
    return float(getattr(config, 'crosshairX', 0)), float(getattr(config, 'crosshairY', 0))


def _finite(v) -> bool:
    return isinstance(v, (int, float)) and math.isfinite(v)


def _num(config, name: str, default: float) -> float:
    v = getattr(config, name, default)
    return float(v) if _finite(v) else float(default)


class AnchorFovParams:
    """Settings snapshot, refreshed in place each frame (no allocation)."""
    __slots__ = ('source', 'position', 'off_x', 'off_y', 'follow',
                 'lost_mode', 'radius', 'min_radius', 'max_radius', 'aspect')

    def __init__(self) -> None:
        for n in self.__slots__:
            setattr(self, n, 0.0)
        self.source = 'detection'
        self.position = 'center'
        self.lost_mode = 'hold'
        self.aspect = 1.0

    def load(self, c) -> None:
        src = str(getattr(c, 'afov_source', 'detection'))
        self.source = src if src in SOURCES else 'detection'
        pos = str(getattr(c, 'afov_position', 'center'))
        self.position = pos if pos in POSITIONS else 'center'
        self.off_x = _num(c, 'afov_offset_x', 0.0)
        self.off_y = _num(c, 'afov_offset_y', 0.0)
        self.follow = clamp(_num(c, 'afov_follow_speed', 12.0), 0.5, 100.0)
        lm = str(getattr(c, 'afov_lost_behavior', 'hold'))
        self.lost_mode = lm if lm in LOST_MODES else 'hold'
        # The FOV size is the one FOV Width every mode uses: radius = Width / 2 (the aspect still comes
        # from FOV Height / Width). The search radius is only headroom to grow into, so it can never
        # cut the base size.
        rmin = MIN_RADIUS_PX
        base = clamp(_num(c, 'fov_size', 2.0 * DEFAULT_RADIUS_PX) * 0.5, rmin, 4096.0)
        self.min_radius = rmin
        self.radius = base
        self.max_radius = clamp(SEARCH_RADIUS_PX, base, 4096.0)
        self.aspect = _aspect(c)


class AnchorSample:
    """What the source resolution found this frame (one reused instance)."""
    __slots__ = ('valid', 'x', 'y')

    def __init__(self) -> None:
        self.clear()

    def clear(self) -> None:
        self.valid = False
        self.x = self.y = 0.0


# ── anchor position ──────────────────────────────────────────────────────────

def anchor_point(box: Sequence[float], position: str, off_x: float, off_y: float):
    """Anchor coordinates from a bounding box and the configured offsets.

    ``center`` is the box centre, ``top``/``bottom`` the middle of the top /
    bottom edge.  The X/Y offsets (pixels) are added in every case.
    """
    x = (float(box[0]) + float(box[2])) * 0.5
    if position == 'top':
        y = float(min(box[1], box[3]))
    elif position == 'bottom':
        y = float(max(box[1], box[3]))
    else:
        y = (float(box[1]) + float(box[3])) * 0.5
    return x + off_x, y + off_y


# ── anchor source resolution ─────────────────────────────────────────────────

def resolve_detection_anchor(s: AnchorSample, p: AnchorFovParams, boxes, locked_box,
                             tracker, now: float, fcx: float, fcy: float,
                             half_w: float, half_h: float) -> None:
    """Detection source: the detected object that is the current selection.

    That is the detection overlapping the locked target; with no lock, the
    detection nearest the FOV centre that intersects the FOV (so the FOV can
    pick up a target without anything being aimed at).  A lock whose box is not
    among this frame's detections counts as lost.
    """
    s.clear()
    if not boxes:
        return
    best = None
    if locked_box is not None:
        best_iou = _MIN_IOU
        for b in boxes:
            v = box_iou(b, locked_box)
            if v > best_iou:
                best, best_iou = b, v
    else:
        best_d = float('inf')
        for b in boxes:
            if not (b[0] < fcx + half_w and b[2] > fcx - half_w
                    and b[1] < fcy + half_h and b[3] > fcy - half_h):
                continue
            d = ((b[0] + b[2]) * 0.5 - fcx) ** 2 + ((b[1] + b[3]) * 0.5 - fcy) ** 2
            if d < best_d:
                best, best_d = b, d
    if best is None:
        return
    x, y = anchor_point(best, p.position, p.off_x, p.off_y)
    if not (math.isfinite(x) and math.isfinite(y)):
        return
    s.valid = True
    s.x, s.y = x, y


def resolve_tracker_anchor(s: AnchorSample, p: AnchorFovParams, tracker, now: float) -> None:
    """Tracker source: the tracker's current position for the lock, advanced by its
    own short extrapolation (it lags the detections by a frame), kept through short
    detection gaps and ended by the tracker's own lost-target timeout."""
    s.clear()
    if tracker is None or not tracker.locked or tracker.last_box is None or tracker.last_center is None:
        return
    if tracker.state is TrackState.TEMPORARILY_LOST:
        ec = tracker.predicted_center(now)
    else:
        ec = tracker.expected_center(now)
    if ec is None:
        return
    dx = ec[0] - tracker.last_center[0]
    dy = ec[1] - tracker.last_center[1]
    lb = tracker.last_box
    x, y = anchor_point((lb[0] + dx, lb[1] + dy, lb[2] + dx, lb[3] + dy), p.position, p.off_x, p.off_y)
    if not (math.isfinite(x) and math.isfinite(y)):
        return
    s.valid = True
    s.x, s.y = x, y


def resolve_manual_anchor(s: AnchorSample, p: AnchorFovParams, cross_x: float, cross_y: float) -> None:
    """Manual source: a fixed point given as the X/Y offset from the crosshair (the same
    offset pair the other sources add to their anchor point)."""
    s.clear()
    x, y = cross_x + p.off_x, cross_y + p.off_y
    if math.isfinite(x) and math.isfinite(y):
        s.valid = True
        s.x, s.y = x, y


# ── the controller ───────────────────────────────────────────────────────────

class AnchorFovController:
    """Holds the FOV's actual centre/radius and moves them toward the anchor."""
    __slots__ = ('params', 'sample', 'initialised', 'cx', 'cy', 'sx', 'sy', 'radius',
                 'last_t', 'ax', 'ay', 'lost_since', 'phase', 'lost')

    def __init__(self) -> None:
        self.params = AnchorFovParams()
        self.sample = AnchorSample()
        self.reset()

    def reset(self) -> None:
        self.initialised = False
        self.cx = self.cy = 0.0        # FOV centre actually in effect
        self.sx = self.sy = 0.0        # smoothed target the centre is gliding to
        self.radius = 0.0              # FOV half width
        self.last_t = 0.0
        self.ax = self.ay = 0.0        # last resolved anchor point
        self.lost_since = 0.0
        self.phase = PH_FOLLOW
        self.lost = False

    # -- helpers ------------------------------------------------------------
    @staticmethod
    def _radius_target(p: AnchorFovParams, base_r: float, searching: bool) -> float:
        """Independent of the position: the base radius, widened to the search radius while
        looking for a lost anchor."""
        return p.max_radius if searching else base_r

    # -- main step ----------------------------------------------------------
    def step(self, now: float, cross_x: float, cross_y: float, bounds) -> None:
        """Advance one frame using ``self.sample`` (already resolved).

        ``bounds`` = (left, top, right, bottom) the FOV centre is kept inside.
        """
        p, s = self.params, self.sample
        if not self.initialised:
            self.cx = self.sx = float(cross_x)
            self.cy = self.sy = float(cross_y)
            self.radius = clamp(p.radius, p.min_radius, p.max_radius)
            self.ax, self.ay = self.cx, self.cy
            self.initialised = True
            dt = 0.0
        else:
            dt = now - self.last_t
            if not math.isfinite(dt) or dt < 0.0 or dt > _MAX_DT_S:
                dt = 0.0
        self.last_t = now

        base_r = clamp(p.radius, p.min_radius, p.max_radius)
        l, t, r, b = bounds
        searching = False

        if s.valid:
            self.lost_since = 0.0
            self.lost = False
            self.phase = PH_FOLLOW
            self.ax = clamp(s.x, l, r)
            self.ay = clamp(s.y, t, b)
            tx, ty = self.ax, self.ay
        else:
            self.lost = True
            if self.lost_since == 0.0:
                self.lost_since = now
            if now - self.lost_since < LOST_DELAY_S:
                self.phase = PH_GRACE
                tx, ty = self.sx, self.sy           # finish gliding to the last anchor
            elif p.lost_mode == 'return':
                self.phase = PH_RETURN
                tx, ty = float(cross_x), float(cross_y)
            elif p.lost_mode == 'search':
                self.phase = PH_SEARCH
                searching = True
                tx, ty = self.sx, self.sy
            else:
                self.phase = PH_HOLD
                tx, ty = self.sx, self.sy

        if dt > 0.0:
            # 1) input low-pass: removes detector jitter before it reaches the FOV
            a1 = 1.0 - math.exp(-dt / (SMOOTHING * _SMOOTH_MAX_S))
            self.sx += (tx - self.sx) * a1
            self.sy += (ty - self.sy) * a1
            # 2) exponential follow with a speed cap (never a jump)
            a2 = 1.0 - math.exp(-dt * p.follow)
            dx, dy = (self.sx - self.cx) * a2, (self.sy - self.cy) * a2
            lim = MAX_SPEED_PX_S * dt
            step = math.hypot(dx, dy)
            if step > lim:
                dx, dy = dx * lim / step, dy * lim / step
            self.cx = clamp(self.cx + dx, l, r)
            self.cy = clamp(self.cy + dy, t, b)
            # 3) size: the base radius (widened while searching)
            rt = self._radius_target(p, base_r, searching)
            self.radius += (rt - self.radius) * (1.0 - math.exp(-dt / _RADIUS_TAU_S))
        else:
            # first/stalled frame: keep the target consistent, move nothing
            self.sx, self.sy = clamp(self.sx, l, r), clamp(self.sy, t, b)
        self.radius = clamp(self.radius, p.min_radius, p.max_radius)


# ── glue used by the inference loop ──────────────────────────────────────────

def update_anchor_fov(config, state, boxes, cross_x: float, cross_y: float,
                      region: Optional[dict], now: float):
    """Resolve the anchor, advance the controller and publish its state.

    Returns ``(centre_x, centre_y, fov_width, fov_height)`` for this frame's FOV
    filter.  The width/height keep the configured FOV's aspect ratio
    (``fov_height / fov_size``) so a rectangular FOV stays rectangular.
    """
    ctrl = getattr(state, 'afov', None)
    if ctrl is None:
        ctrl = state.afov = AnchorFovController()
    p = ctrl.params
    p.load(config)
    s = ctrl.sample

    if ctrl.initialised:
        fcx, fcy = ctrl.cx, ctrl.cy
        hw = ctrl.radius
        hh = hw * p.aspect
    else:
        fcx, fcy, hw = float(cross_x), float(cross_y), p.radius
        hh = hw * p.aspect

    if p.source == 'tracker':
        resolve_tracker_anchor(s, p, getattr(state, 'tracker', None), now)
    elif p.source == 'manual':
        resolve_manual_anchor(s, p, float(cross_x), float(cross_y))
    else:
        resolve_detection_anchor(s, p, boxes, getattr(state, 'locked_box', None),
                                 getattr(state, 'tracker', None), now, fcx, fcy, hw, hh)

    if region:
        bounds = (float(region.get('left', 0)), float(region.get('top', 0)),
                  float(region.get('left', 0) + region.get('width', 0)),
                  float(region.get('top', 0) + region.get('height', 0)))
        if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
            bounds = (cross_x, cross_y, cross_x, cross_y)
    else:
        bounds = (cross_x - 4096.0, cross_y - 4096.0, cross_x + 4096.0, cross_y + 4096.0)

    ctrl.step(now, float(cross_x), float(cross_y), bounds)
    w = max(1, int(round(ctrl.radius * 2.0)))
    h = max(1, int(round(ctrl.radius * 2.0 * p.aspect)))
    publish_anchor_fov(config, ctrl, w, h)
    return ctrl.cx, ctrl.cy, w, h


def _aspect(config) -> float:
    fw = float(getattr(config, 'fov_size', 0) or 0)
    fh = float(getattr(config, 'fov_height', fw) or fw)
    return fh / fw if fw > 0 and fh > 0 else 1.0


def publish_anchor_fov(config, ctrl: AnchorFovController, w: int, h: int) -> None:
    """Expose the controller's real state to the overlays (runtime-only fields)."""
    config.fov_center_x = ctrl.cx
    config.fov_center_y = ctrl.cy
    config.fov_effective_size = float(w)
    config.fov_effective_height = float(h)
    config.afov_active = True
    config.afov_has_anchor = bool(ctrl.sample.valid)
    config.afov_lost = bool(ctrl.lost)
    config.afov_phase = ctrl.phase
    config.afov_anchor_x = ctrl.ax
    config.afov_anchor_y = ctrl.ay


def clear_anchor_fov(config, state) -> None:
    """Leave Anchor mode: forget the controller and the published state, so the
    other modes (and the overlays) are exactly as they were without it."""
    ctrl = getattr(state, 'afov', None)
    if ctrl is not None and ctrl.initialised:
        ctrl.reset()
    if getattr(config, 'afov_active', False) or getattr(config, 'fov_center_x', None) is not None:
        config.afov_active = False
        config.fov_center_x = None
        config.fov_center_y = None
        config.afov_has_anchor = False
        config.afov_lost = False
