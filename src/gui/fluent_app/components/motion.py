"""Qt animation helpers shared by every glass widget (timings/easing live in ``core.ui_motion``).

Rules these helpers enforce:

* A new request **retargets** the running tween from the value it currently shows — rapid clicks
  never jump back and never queue up animations.
* ``Config.ui_animations`` off (``ui_motion.set_animations_enabled(False)``) means every tween jumps
  to its end value at once; callers need no second code path.
* Painters own no timers: a ``Tween`` / ``HoverProgress`` repaints only the widget that owns it.
* Graphics effects exist only while a fade runs (``fade_in`` removes its effect when finished), so
  nothing keeps a page rendering through an offscreen pixmap.
"""
from __future__ import annotations

from typing import Callable, Optional

from PyQt6.QtCore import QEasingCurve, QObject, QPoint, QVariantAnimation, pyqtSignal
from PyQt6.QtWidgets import QGraphicsOpacityEffect, QWidget

from core import ui_motion
from core.ui_motion import DUR, DUR_FAST, DUR_SLOW  # noqa: F401  (re-exported for widgets)


class Tween(QObject):
    """Animated float ``value`` with ``to(target, ms)``; emits ``changed(value)`` on every step."""

    changed = pyqtSignal(float)
    finished = pyqtSignal()

    def __init__(self, parent: Optional[QObject] = None, value: float = 0.0,
                 easing: QEasingCurve.Type = QEasingCurve.Type.OutCubic):
        super().__init__(parent)
        self._value = float(value)
        self._anim = QVariantAnimation(self)
        self._anim.setEasingCurve(QEasingCurve(easing))
        self._anim.valueChanged.connect(self._onStep)
        self._anim.finished.connect(self.finished)

    @property
    def value(self) -> float:
        return self._value

    @property
    def running(self) -> bool:
        return self._anim.state() == QVariantAnimation.State.Running

    def _onStep(self, v) -> None:
        self._value = float(v)
        self.changed.emit(self._value)

    def jump(self, value: float) -> None:
        """Set the value now, cancelling any running animation."""
        self._anim.stop()
        self._value = float(value)
        self.changed.emit(self._value)

    def to(self, target: float, ms: int = DUR) -> None:
        """Animate from the *currently shown* value to ``target`` (instant when animations are off)."""
        target = float(target)
        d = ui_motion.duration(ms)
        if d <= 0 or abs(target - self._value) < 1e-6:
            self.jump(target)
            self.finished.emit()
            return
        self._anim.stop()
        self._anim.setDuration(d)
        self._anim.setStartValue(self._value)
        self._anim.setEndValue(target)
        self._anim.start()


class HoverProgress(Tween):
    """0→1 hover (and 1→… press) amount that repaints ``widget`` while it changes."""

    def __init__(self, widget: QWidget, ms: int = DUR_FAST):
        super().__init__(widget)
        self._ms = ms
        self.changed.connect(lambda _v: widget.update())

    def set(self, active: bool) -> None:
        self.to(1.0 if active else 0.0, self._ms)


def fade_in(widget: QWidget, ms: int = DUR, rise: int = 0, on_done: Optional[Callable[[], None]] = None) -> None:
    """Fade ``widget`` in (optionally rising ``rise`` px into place), then drop the effect.

    Calling it again while a fade runs restarts from the current opacity. With animations off the
    widget is simply shown as is.
    """
    d = ui_motion.duration(ms)
    if d <= 0 or not widget.isVisible():
        widget.setGraphicsEffect(None)
        if on_done:
            on_done()
        return
    old = getattr(widget, "_fadeTween", None)
    start = old.value if old is not None and old.running else 0.0
    if old is not None:      # retarget: silence the previous fade before its effect is replaced
        old._anim.stop()
        for sig in (old.changed, old.finished):
            try:
                sig.disconnect()
            except TypeError:
                pass
    effect = QGraphicsOpacityEffect(widget)
    effect.setOpacity(start)
    widget.setGraphicsEffect(effect)
    origin = widget.pos()
    tween = Tween(widget, start, QEasingCurve.Type.OutCubic)
    widget._fadeTween = tween

    def step(v: float) -> None:
        effect.setOpacity(v)
        if rise:
            widget.move(QPoint(origin.x(), origin.y() + int(round(rise * (1.0 - v)))))

    def done() -> None:
        if getattr(widget, "_fadeTween", None) is tween:
            widget.setGraphicsEffect(None)
            if rise:
                widget.move(origin)
            widget._fadeTween = None
        if on_done:
            on_done()

    tween.changed.connect(step)
    tween.finished.connect(done)
    tween.to(1.0, ms)
