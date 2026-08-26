"""Unit tests for core/app_controller.py.

Covers only set_always_aim() — the AI-thread lifecycle functions
(start_ai_threads/stop_ai_threads/pause_ai_inference/resume_ai_inference)
depend transitively on win32api/onnxruntime/cv2 (via .ai_loop/.auto_fire/
.session_utils) that aren't installed in this sandbox; they're moved
verbatim from main.py with no behavior change (see the module's own
docstring), so there's nothing new to unit-test there beyond what
importing this module already exercises (see test_module_importable_
without_windows_deps below).
"""

import threading

import pytest

from core import app_controller


class _FakeConfig:
    """Minimal stand-in carrying just the fields set_always_aim touches."""
    always_aim = False
    idle_detect_enabled = True


def test_module_importable_without_windows_deps():
    """Importing app_controller must never require win32api/onnxruntime/cv2.

    Those are only needed by the AI-thread lifecycle functions, and only
    once actually *called* — see the module docstring's "Sandbox note".
    This test is really just documentation-as-a-test: the module import at
    the top of this file already either succeeded (passing) or the whole
    file would have failed collection (the exact class of bug CLAUDE.md
    describes for ai_loop.py itself).
    """
    assert hasattr(app_controller, "set_always_aim")
    assert hasattr(app_controller, "start_ai_threads")
    assert hasattr(app_controller, "stop_ai_threads")


def test_set_always_aim_enables_and_disables_idle_detect():
    config = _FakeConfig()
    config.idle_detect_enabled = True

    app_controller.set_always_aim(config, True)
    assert config.always_aim is True
    assert config.idle_detect_enabled is False


def test_set_always_aim_disable_does_not_touch_idle_detect():
    """Turning always_aim OFF must not force idle_detect back on.

    Mirrors the original keys_page.py behavior exactly: the coupling is
    one-directional (enabling always_aim forces idle-detect off so the two
    "detect when not aiming" mechanisms don't fight each other) — disabling
    it makes no claim about what idle-detect should be, so it's left alone.
    """
    config = _FakeConfig()
    config.idle_detect_enabled = False

    app_controller.set_always_aim(config, False)
    assert config.always_aim is False
    assert config.idle_detect_enabled is False  # unchanged


def test_set_always_aim_coerces_truthy_values():
    config = _FakeConfig()
    app_controller.set_always_aim(config, 1)
    assert config.always_aim is True
    assert isinstance(config.always_aim, bool)


def test_set_always_aim_is_reentrant_under_concurrent_calls():
    """Two threads calling set_always_aim concurrently must never interleave.

    Not a proof of correctness under adversarial scheduling (that's what
    _multi_field_lock's design docstring argues from first principles), but
    a smoke test that acquiring/releasing the module-level lock repeatedly
    from multiple threads doesn't deadlock or raise.
    """
    config = _FakeConfig()
    errors = []

    def worker(enabled):
        try:
            for _ in range(200):
                app_controller.set_always_aim(config, enabled)
        except Exception as exc:  # pragma: no cover - failure path only
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i % 2 == 0,)) for i in range(8)]
    for th in threads:
        th.start()
    for th in threads:
        th.join(timeout=5.0)

    assert not errors
    assert not any(th.is_alive() for th in threads)
    # Whichever call landed last, the coupling invariant must still hold:
    # always_aim True implies idle_detect_enabled False.
    if config.always_aim:
        assert config.idle_detect_enabled is False
