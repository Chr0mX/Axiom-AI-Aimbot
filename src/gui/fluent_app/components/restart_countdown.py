"""A cancellable "restart in N seconds" notice shared by every page that needs a restart to apply a setting
(Model page: crossing to/from DirectML; Visuals page: switching the interface look).

Closing the notice cancels the restart — the saved setting is kept and takes effect on the next launch.
A second ``start()`` restarts the countdown instead of stacking another timer (two timers meant two
restarts). The restart itself saves the config, launches a new process with the same arguments and
quits the Qt app, so ``main._shutdown`` runs through ``aboutToQuit`` (unlike the hard-exit
``app_controller.restart_axiom`` used for the remote restart).
"""
from __future__ import annotations

import subprocess
import sys
from typing import Callable, Optional

from PyQt6.QtCore import QObject, QTimer
from PyQt6.QtWidgets import QApplication, QWidget
from qfluentwidgets import BodyLabel, InfoBar, InfoBarPosition

from ..language_manager import t

COUNTDOWN_SECONDS = 5


def restart_app(config=None) -> None:
    """Save ``config``, start a fresh Axiom process and quit this one."""
    from core.config import save_config
    if config is not None:
        save_config(config)
    subprocess.Popen([sys.executable] + sys.argv)
    QApplication.instance().quit()


class RestartCountdown(QObject):
    def __init__(self, page: QWidget, get_config: Callable[[], object], parent: Optional[QObject] = None):
        super().__init__(parent or page)
        self._page = page
        self._get_config = get_config
        self._remaining = 0
        self._bar = None
        self._timer: Optional[QTimer] = None

    def start(self) -> None:
        self.cancel()
        self._remaining = COUNTDOWN_SECONDS
        self._bar = InfoBar.info(
            t("restart_required", "Restart Required"),
            self._text(),
            duration=-1, isClosable=True, position=InfoBarPosition.TOP, parent=self._page,
        )
        self._bar.closedSignal.connect(self._onBarClosed)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(1000)

    def cancel(self) -> None:
        timer, self._timer = self._timer, None
        if timer is not None:
            timer.stop()
            timer.deleteLater()
        bar, self._bar = self._bar, None
        if bar is not None:
            try:
                bar.close()
            except RuntimeError:        # already deleted
                pass

    def _text(self) -> str:
        return f"Restarting in {self._remaining}s…"

    def _onBarClosed(self) -> None:
        if self._remaining > 0:
            self.cancel()

    def _tick(self) -> None:
        self._remaining -= 1
        if self._remaining <= 0:
            self.cancel()
            restart_app(self._get_config())
        elif self._bar:
            for lbl in self._bar.findChildren(BodyLabel):
                lbl.setText(self._text())
                break
