"""Hide "advanced" cards behind the global *Show advanced settings* switch.

Pure Python (no Qt import): it only needs widgets with ``setVisible`` (and optionally
``isHidden``), so it is unit-tested with fakes.

How it works: each registered widget's ``setVisible`` is shadowed on that one instance. The
page's own visibility rules keep calling ``card.setVisible(<their rule>)`` exactly as before;
the shadow remembers that *wanted* state and applies ``wanted and switch_is_on``. Toggling the
switch just re-applies every remembered state. So

* a rule can never un-hide an advanced card while the switch is off,
* turning the switch on restores exactly what the rules wanted (no stale/forced visibility),
* nothing here ever reads or writes a setting: hidden cards keep their values.
"""
from __future__ import annotations

from typing import Callable, Iterable


class _Entry:
    __slots__ = ("widget", "orig", "wanted")

    def __init__(self, widget, orig, wanted: bool) -> None:
        self.widget, self.orig, self.wanted = widget, orig, wanted


class AdvancedVisibility:
    def __init__(self, enabled: Callable[[], bool]) -> None:
        self._enabled = enabled
        self._entries: list[_Entry] = []

    def register(self, *widgets) -> None:
        for w in widgets:
            if w is None or any(e.widget is w for e in self._entries):
                continue
            hidden = getattr(w, "isHidden", None)
            entry = _Entry(w, w.setVisible, wanted=not (hidden() if callable(hidden) else False))

            def set_visible(visible=True, _e=entry, _gate=self):
                _e.wanted = bool(visible)
                _e.orig(_e.wanted and bool(_gate._enabled()))

            w.setVisible = set_visible
            self._entries.append(entry)

    def register_names(self, owner, names: Iterable[str]) -> None:
        """Register ``getattr(owner, name)`` for each name that exists (tables and optional cards)."""
        self.register(*[getattr(owner, n, None) for n in names])

    def apply(self) -> None:
        """Re-apply every remembered state against the switch's current value."""
        on = bool(self._enabled())
        for e in self._entries:
            e.orig(e.wanted and on)

    def wanted(self, widget) -> bool:
        """What the page's own rules want for ``widget`` (ignoring the switch)."""
        for e in self._entries:
            if e.widget is widget:
                return e.wanted
        raise KeyError(widget)

    def __len__(self) -> int:
        return len(self._entries)
