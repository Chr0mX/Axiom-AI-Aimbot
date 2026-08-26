"""Unit tests for core/app_controller.py.

Covers set_always_aim() and the connect_makcu()/disconnect_makcu() thin
wrappers. The AI-thread lifecycle functions (start_ai_threads/
stop_ai_threads/pause_ai_inference/resume_ai_inference) depend
transitively on win32api/onnxruntime/cv2 (via .ai_loop/.auto_fire/
.session_utils) that aren't installed in this sandbox; they're moved
verbatim from main.py with no behavior change (see the module's own
docstring), so there's nothing new to unit-test there beyond what
importing this module already exercises (see test_module_importable_
without_windows_deps below).
"""

import sys
import threading
import types

import pytest

from core import app_controller


class _FakeConfig:
    """Minimal stand-in carrying just the fields these functions touch."""
    always_aim = False
    idle_detect_enabled = True
    makcu_com_port = ""
    makcu_baud_rate = 4_000_000
    model_path = ""
    Running = False
    inference_backend = "auto"


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
    assert hasattr(app_controller, "connect_makcu")
    assert hasattr(app_controller, "disconnect_makcu")
    assert hasattr(app_controller, "resolve_model_path")
    assert hasattr(app_controller, "request_model_change")
    assert hasattr(app_controller, "list_models")


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


class TestConnectMakcu:
    """connect_makcu()'s own decide logic — the guard clause and the
    ImportError fallback are both genuinely exercisable here with no
    mocking; the true delegate-to-win_utils happy path needs sys.modules
    faked (see _fake_makcu_module below), since the real win_utils package
    transitively imports win32api, unavailable in this sandbox.
    """

    def test_empty_port_returns_false_without_importing_win_utils(self):
        """The empty-port guard must run before the deferred import at all.

        Proven, not assumed: this sandbox genuinely has no `win_utils`
        importable (confirmed elsewhere in this file), so if the guard
        clause didn't short-circuit before the import, this call would
        raise instead of returning False.
        """
        config = _FakeConfig()
        config.makcu_com_port = ""
        assert app_controller.connect_makcu(config) is False

    def test_missing_win_utils_returns_false(self):
        """Real sandbox condition, not a simulated one: win_utils.makcu_mouse
        is genuinely unimportable here (missing win32api), so this exercises
        the try/except ImportError fallback for real.
        """
        config = _FakeConfig()
        config.makcu_com_port = "COM5"
        assert app_controller.connect_makcu(config) is False

    def test_delegates_to_win_utils_makcu_mouse_when_available(self, monkeypatch):
        """Fakes win_utils/win_utils.makcu_mouse in sys.modules so the
        deferred `from win_utils.makcu_mouse import connect_makcu` resolves
        against the fake without ever executing the real win_utils/
        __init__.py (which is what actually pulls in win32api) — this is
        the technique any app_controller function with a deferred heavy
        import needs for a real happy-path test in this sandbox.
        """
        calls = []

        def _fake_connect(com_port, baud_rate):
            calls.append((com_port, baud_rate))
            return True

        fake_pkg = types.ModuleType("win_utils")
        fake_submodule = types.ModuleType("win_utils.makcu_mouse")
        fake_submodule.connect_makcu = _fake_connect
        monkeypatch.setitem(sys.modules, "win_utils", fake_pkg)
        monkeypatch.setitem(sys.modules, "win_utils.makcu_mouse", fake_submodule)

        config = _FakeConfig()
        config.makcu_com_port = "COM7"
        config.makcu_baud_rate = 115200

        assert app_controller.connect_makcu(config) is True
        assert calls == [("COM7", 115200)]

    def test_propagates_a_false_result_from_win_utils(self, monkeypatch):
        fake_pkg = types.ModuleType("win_utils")
        fake_submodule = types.ModuleType("win_utils.makcu_mouse")
        fake_submodule.connect_makcu = lambda com_port, baud_rate: False
        monkeypatch.setitem(sys.modules, "win_utils", fake_pkg)
        monkeypatch.setitem(sys.modules, "win_utils.makcu_mouse", fake_submodule)

        config = _FakeConfig()
        config.makcu_com_port = "COM7"

        assert app_controller.connect_makcu(config) is False

    def test_default_baud_used_when_config_field_is_falsy(self, monkeypatch):
        """getattr(..., 4_000_000) only supplies a default when the attribute
        is missing — an explicit 0/None/"" on Config must fall back too.
        """
        calls = []
        fake_pkg = types.ModuleType("win_utils")
        fake_submodule = types.ModuleType("win_utils.makcu_mouse")
        fake_submodule.connect_makcu = lambda com_port, baud_rate: calls.append(baud_rate) or True
        monkeypatch.setitem(sys.modules, "win_utils", fake_pkg)
        monkeypatch.setitem(sys.modules, "win_utils.makcu_mouse", fake_submodule)

        config = _FakeConfig()
        config.makcu_com_port = "COM7"
        config.makcu_baud_rate = 0

        app_controller.connect_makcu(config)
        assert calls == [4_000_000]


class TestDisconnectMakcu:
    def test_missing_win_utils_does_not_raise(self):
        """Same real-sandbox-condition test as connect_makcu's fallback —
        disconnect_makcu() must degrade to a no-op, never propagate.
        """
        config = _FakeConfig()
        assert app_controller.disconnect_makcu(config) is None

    def test_delegates_to_win_utils_makcu_mouse_when_available(self, monkeypatch):
        calls = []
        fake_pkg = types.ModuleType("win_utils")
        fake_submodule = types.ModuleType("win_utils.makcu_mouse")
        fake_submodule.disconnect_makcu = lambda: calls.append(True)
        monkeypatch.setitem(sys.modules, "win_utils", fake_pkg)
        monkeypatch.setitem(sys.modules, "win_utils.makcu_mouse", fake_submodule)

        config = _FakeConfig()
        app_controller.disconnect_makcu(config)
        assert calls == [True]


class TestResolveModelPath:
    """resolve_model_path() is pure os.path logic — no onnxruntime/win32api
    import at all — so every branch is genuinely testable here with no
    faking, unlike the rest of start_ai_threads().
    """

    def test_empty_path(self):
        assert app_controller.resolve_model_path("") == (None, "no_model_path")

    def test_wrong_extension(self):
        assert app_controller.resolve_model_path("model.txt") == (None, "invalid_model_path")

    def test_nonexistent_absolute_path(self, tmp_path):
        missing = str(tmp_path / "does_not_exist.onnx")
        assert app_controller.resolve_model_path(missing) == (None, "not_found")

    def test_existing_absolute_path(self, tmp_path):
        model_file = tmp_path / "real.onnx"
        model_file.write_bytes(b"")
        resolved, reason = app_controller.resolve_model_path(str(model_file))
        assert reason is None
        assert resolved == str(model_file)

    def test_existing_relative_path_resolved_against_project_root(self, tmp_path, monkeypatch):
        (tmp_path / "Model").mkdir()
        model_file = tmp_path / "Model" / "real.onnx"
        model_file.write_bytes(b"")
        monkeypatch.setattr(app_controller, "project_root", str(tmp_path))

        resolved, reason = app_controller.resolve_model_path("Model/real.onnx")
        assert reason is None
        assert resolved == str(model_file)

    def test_nonexistent_relative_path(self, tmp_path, monkeypatch):
        monkeypatch.setattr(app_controller, "project_root", str(tmp_path))
        assert app_controller.resolve_model_path("Model/missing.onnx") == (None, "not_found")


class TestListModels:
    """list_models() is pure os.path/glob logic — no onnxruntime/win32api
    import, no dependency on model_page.py — so every branch is genuinely
    testable here with no faking, same as TestResolveModelPath above.
    """

    def test_no_model_dir_returns_empty_list(self, tmp_path, monkeypatch):
        monkeypatch.setattr(app_controller, "project_root", str(tmp_path))
        assert app_controller.list_models() == []

    def test_empty_model_dir_returns_empty_list(self, tmp_path, monkeypatch):
        (tmp_path / "Model").mkdir()
        monkeypatch.setattr(app_controller, "project_root", str(tmp_path))
        assert app_controller.list_models() == []

    def test_returns_sorted_onnx_basenames(self, tmp_path, monkeypatch):
        model_dir = tmp_path / "Model"
        model_dir.mkdir()
        (model_dir / "zeta.onnx").write_bytes(b"")
        (model_dir / "alpha.onnx").write_bytes(b"")
        (model_dir / "mid.onnx").write_bytes(b"")
        monkeypatch.setattr(app_controller, "project_root", str(tmp_path))

        assert app_controller.list_models() == ["alpha.onnx", "mid.onnx", "zeta.onnx"]

    def test_ignores_non_onnx_files(self, tmp_path, monkeypatch):
        model_dir = tmp_path / "Model"
        model_dir.mkdir()
        (model_dir / "real.onnx").write_bytes(b"")
        (model_dir / "notes.txt").write_bytes(b"")
        (model_dir / "real.engine").write_bytes(b"")
        monkeypatch.setattr(app_controller, "project_root", str(tmp_path))

        assert app_controller.list_models() == ["real.onnx"]


class TestRequestModelChange:
    """request_model_change()'s refusal branches (not_found/invalid_model_path/
    invalid_backend/needs_restart) are reachable with zero faking — they
    return before the deferred `from .session_utils import needs_trt_build`.
    The needs_conversion/success branches fake core.session_utils in
    sys.modules, the same technique already established for
    win_utils.makcu_mouse: `from .session_utils import needs_trt_build`
    inside core/app_controller.py resolves against sys.modules
    ["core.session_utils"] without ever importing the real module (which
    needs onnxruntime, unavailable in this sandbox).
    """

    def _model_file(self, tmp_path):
        model_file = tmp_path / "real.onnx"
        model_file.write_bytes(b"")
        return str(model_file)

    def _fake_session_utils(self, monkeypatch, needs_build):
        calls = []
        fake_module = types.ModuleType("core.session_utils")

        def _fake_needs_trt_build(cfg, model_path):
            calls.append(cfg.inference_backend)
            return needs_build

        fake_module.needs_trt_build = _fake_needs_trt_build
        monkeypatch.setitem(sys.modules, "core.session_utils", fake_module)
        return calls

    def test_refuses_missing_file(self):
        config = _FakeConfig()
        result = app_controller.request_model_change(config, "does/not/exist.onnx")
        assert result == {"ok": False, "reason": "not_found"}
        assert config.model_path == ""

    def test_refuses_bad_extension(self, tmp_path):
        bad_file = tmp_path / "real.txt"
        bad_file.write_bytes(b"")
        config = _FakeConfig()
        result = app_controller.request_model_change(config, str(bad_file))
        assert result == {"ok": False, "reason": "invalid_model_path"}

    def test_refuses_invalid_backend(self, tmp_path):
        config = _FakeConfig()
        result = app_controller.request_model_change(
            config, self._model_file(tmp_path), inference_backend="rocm")
        assert result == {"ok": False, "reason": "invalid_backend"}
        assert config.model_path == ""

    def test_refuses_crossing_into_directml(self, tmp_path):
        config = _FakeConfig()
        config.inference_backend = "cuda"
        result = app_controller.request_model_change(
            config, self._model_file(tmp_path), inference_backend="directml")
        assert result == {"ok": False, "reason": "needs_restart"}
        assert config.inference_backend == "cuda"  # untouched
        assert config.model_path == ""

    def test_refuses_crossing_out_of_directml(self, tmp_path):
        config = _FakeConfig()
        config.inference_backend = "directml"
        result = app_controller.request_model_change(
            config, self._model_file(tmp_path), inference_backend="cuda")
        assert result == {"ok": False, "reason": "needs_restart"}
        assert config.inference_backend == "directml"  # untouched

    def test_allows_staying_on_directml(self, tmp_path, monkeypatch):
        """Same backend on both sides — not a crossing — must fall through
        to the (faked) needs_trt_build check instead of refusing just
        because DirectML is involved at all.
        """
        self._fake_session_utils(monkeypatch, needs_build=False)
        config = _FakeConfig()
        config.inference_backend = "directml"
        result = app_controller.request_model_change(
            config, self._model_file(tmp_path), inference_backend="directml")
        assert result["ok"] is True
        assert config.inference_backend == "directml"

    def test_refuses_when_trt_build_needed(self, tmp_path, monkeypatch):
        self._fake_session_utils(monkeypatch, needs_build=True)
        config = _FakeConfig()
        model_file = self._model_file(tmp_path)
        result = app_controller.request_model_change(config, model_file, inference_backend="tensorrt")
        assert result == {"ok": False, "reason": "needs_conversion"}
        assert config.model_path == ""
        assert config.inference_backend == "auto"  # untouched

    def test_applies_both_fields_on_success(self, tmp_path, monkeypatch):
        self._fake_session_utils(monkeypatch, needs_build=False)
        config = _FakeConfig()
        config.Running = True
        model_file = self._model_file(tmp_path)

        result = app_controller.request_model_change(config, model_file, inference_backend="cpu")

        assert result["ok"] is True
        assert result["model_path"] == model_file
        assert result["inference_backend"] == "cpu"
        assert result["applied_live"] is True
        assert config.model_path == model_file
        assert config.inference_backend == "cpu"

    def test_applied_live_false_when_not_running(self, tmp_path, monkeypatch):
        self._fake_session_utils(monkeypatch, needs_build=False)
        config = _FakeConfig()
        config.Running = False
        result = app_controller.request_model_change(config, self._model_file(tmp_path))
        assert result["applied_live"] is False

    def test_defaults_to_current_backend_when_none_requested(self, tmp_path, monkeypatch):
        calls = self._fake_session_utils(monkeypatch, needs_build=False)
        config = _FakeConfig()
        config.inference_backend = "cuda"
        app_controller.request_model_change(config, self._model_file(tmp_path))
        assert config.inference_backend == "cuda"
        assert calls == ["cuda"]

    def test_needs_trt_build_is_evaluated_against_requested_backend_not_current(self, tmp_path, monkeypatch):
        """The deepcopy passed to needs_trt_build must reflect the
        *requested* backend, not whatever config.inference_backend
        currently is — otherwise a model+backend switch in one call would
        evaluate the TRT-build question against the wrong provider.
        """
        calls = self._fake_session_utils(monkeypatch, needs_build=False)
        config = _FakeConfig()
        config.inference_backend = "cpu"
        app_controller.request_model_change(
            config, self._model_file(tmp_path), inference_backend="tensorrt")
        # The fake saw "tensorrt" (the requested backend), not "cpu" (what
        # config.inference_backend was at call time) — proving the check
        # ran against the deepcopy's simulated value.
        assert calls == ["tensorrt"]
        # Only written to the real config afterward, since needs_build=False.
        assert config.inference_backend == "tensorrt"


class _FakeEspServer:
    """Stand-in for core.esp_server with the same running-state semantics
    the real module has (module-level state toggled by start()/stop()) —
    faked via sys.modules so set_web_esp_enabled()/
    restart_web_esp_if_running() never bind a real socket/thread in this
    unit test, same "fake the deferred import" technique as win_utils.
    makcu_mouse above.
    """

    def __init__(self):
        self.running = False
        self.start_calls = []
        self.stop_calls = 0

    def start(self, config):
        self.running = True
        self.start_calls.append(config)
        return True

    def stop(self):
        self.running = False
        self.stop_calls += 1

    def is_running(self):
        return self.running


def _install_fake_esp_server(monkeypatch):
    """Fakes core.esp_server for the deferred `from core import esp_server`
    in set_web_esp_enabled()/restart_web_esp_if_running().

    Patching sys.modules["core.esp_server"] alone isn't sufficient once
    something elsewhere in a full-suite run has done a REAL `from core
    import esp_server` (test_esp_server.py does, at module level): that
    binds a real `esp_server` attribute directly on the `core` package
    object, and `from core import esp_server` resolves via a plain
    getattr(core, "esp_server") first — succeeding against that cached
    real attribute — before it would ever fall back to consulting
    sys.modules. So the package attribute has to be patched too, same
    two-level fix as vk_codes/get_serial_ports' "module unavailable" tests
    (see web_control_settings.py's TestListVkOptions/TestGetSerialPorts).
    """
    fake = _FakeEspServer()
    fake_module = types.ModuleType("core.esp_server")
    fake_module.start = fake.start
    fake_module.stop = fake.stop
    fake_module.is_running = fake.is_running
    monkeypatch.setitem(sys.modules, "core.esp_server", fake_module)
    import core
    monkeypatch.setattr(core, "esp_server", fake_module, raising=False)
    return fake


class TestSetWebEspEnabled:
    def test_enabling_writes_config_and_starts_server(self, monkeypatch):
        fake = _install_fake_esp_server(monkeypatch)
        config = _FakeConfig()
        config.web_esp_enabled = False
        result = app_controller.set_web_esp_enabled(config, True)
        assert result is True
        assert config.web_esp_enabled is True
        assert fake.start_calls == [config]
        assert fake.stop_calls == 0

    def test_disabling_writes_config_and_stops_server(self, monkeypatch):
        fake = _install_fake_esp_server(monkeypatch)
        fake.running = True
        config = _FakeConfig()
        config.web_esp_enabled = True
        result = app_controller.set_web_esp_enabled(config, False)
        assert result is False
        assert config.web_esp_enabled is False
        assert fake.stop_calls == 1
        assert fake.start_calls == []

    def test_enabled_flag_coerced_to_real_bool(self, monkeypatch):
        """A truthy non-bool (e.g. from a loosely-deserialized JSON body)
        must still be stored as an actual bool on config, not the raw
        truthy value — mirrors set_always_aim()'s own bool() coercion."""
        _install_fake_esp_server(monkeypatch)
        config = _FakeConfig()
        app_controller.set_web_esp_enabled(config, 1)
        assert config.web_esp_enabled is True


class TestRestartWebEspIfRunning:
    def test_noop_when_not_running(self, monkeypatch):
        fake = _install_fake_esp_server(monkeypatch)
        fake.running = False
        config = _FakeConfig()
        result = app_controller.restart_web_esp_if_running(config)
        assert result is False
        assert fake.stop_calls == 0
        assert fake.start_calls == []

    def test_restarts_when_running(self, monkeypatch):
        fake = _install_fake_esp_server(monkeypatch)
        fake.running = True
        config = _FakeConfig()
        result = app_controller.restart_web_esp_if_running(config)
        assert result is True
        assert fake.stop_calls == 1
        assert fake.start_calls == [config]
