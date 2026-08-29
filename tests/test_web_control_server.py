"""Regression test for core/web_control_server.py's route registration.

The real fastapi/pydantic/uvicorn packages vendored under
src/python/dependencies/ ship a Windows-only compiled pydantic_core
extension (_pydantic_core.cp311-win_amd64.pyd), so they can't actually be
imported in this sandbox — that's why web_control_server.py has had no
tests of its own until now (see CLAUDE.md's Web Control API section).

This file fakes just enough of that surface (via sys.modules) to call the
REAL web_control_server.start() and inspect its REAL registered route
handlers. That's deliberate: the bug this test guards against lives at
route-*registration* time — inside start(), when each @app.post(...)
decorator runs — not inside any handler's body, so a test that only
called the (fully mocked) handler functions directly would never see it.
"""

import inspect
import sys
import time
import types

import pytest


class _FakeBaseModel:
    """Stand-in for pydantic.BaseModel. Real field validation isn't the
    point here — only that FastAPI's route registration sees a genuine
    class for each body parameter's annotation, not an unresolved string
    or ForwardRef (see web_control_server.py's module docstring)."""

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class _RecordingFastAPI:
    """Fakes just the FastAPI surface web_control_server.start() touches:
    route-decorator registration, add_middleware(), and mount(). Records
    every (method, path, handler) triple so the test can inspect the real
    handler functions' real parameter annotations afterward — app itself
    is a local variable inside start(), never returned, so instances are
    tracked on the class for the test to reach.
    """

    instances: list["_RecordingFastAPI"] = []

    def __init__(self, *args, **kwargs):
        self.routes = []
        _RecordingFastAPI.instances.append(self)

    def add_middleware(self, *args, **kwargs):
        pass

    def _register(self, method, path, **kwargs):
        def decorator(func):
            self.routes.append((method, path, func))
            return func
        return decorator

    def get(self, path, **kwargs):
        return self._register("GET", path, **kwargs)

    def post(self, path, **kwargs):
        return self._register("POST", path, **kwargs)

    def mount(self, *args, **kwargs):
        pass


class _FakeUvicornServer:
    """Fakes uvicorn.Server — just enough for _ThreadServer(uvicorn.Server)
    to run on its background thread and report started=True promptly,
    matching start()'s own poll loop, then exit cleanly on stop()'s
    should_exit=True."""

    def __init__(self, config):
        self.config = config
        self.should_exit = False
        self.started = False

    def run(self):
        self.started = True
        while not self.should_exit:
            time.sleep(0.01)


def _install_fake_fastapi_stack(monkeypatch):
    _RecordingFastAPI.instances = []

    fastapi_mod = types.ModuleType("fastapi")
    fastapi_mod.FastAPI = _RecordingFastAPI
    fastapi_mod.Depends = lambda dependency=None: dependency
    fastapi_mod.Header = lambda default=None, **kwargs: default

    class _HTTPException(Exception):
        def __init__(self, status_code=500, detail=""):
            super().__init__(detail)
            self.status_code = status_code
            self.detail = detail

    fastapi_mod.HTTPException = _HTTPException

    middleware_mod = types.ModuleType("fastapi.middleware")
    cors_mod = types.ModuleType("fastapi.middleware.cors")
    cors_mod.CORSMiddleware = object()
    middleware_mod.cors = cors_mod
    fastapi_mod.middleware = middleware_mod

    staticfiles_mod = types.ModuleType("fastapi.staticfiles")

    class _StaticFiles:
        def __init__(self, *args, **kwargs):
            pass

    staticfiles_mod.StaticFiles = _StaticFiles
    fastapi_mod.staticfiles = staticfiles_mod

    responses_mod = types.ModuleType("fastapi.responses")

    class _FakeStreamingResponse:
        def __init__(self, content, media_type=None, **kwargs):
            self.content = content
            self.media_type = media_type

    responses_mod.StreamingResponse = _FakeStreamingResponse
    fastapi_mod.responses = responses_mod

    pydantic_mod = types.ModuleType("pydantic")
    pydantic_mod.BaseModel = _FakeBaseModel

    uvicorn_mod = types.ModuleType("uvicorn")
    uvicorn_mod.Server = _FakeUvicornServer
    uvicorn_mod.Config = lambda app, **kwargs: types.SimpleNamespace(app=app, **kwargs)

    monkeypatch.setitem(sys.modules, "fastapi", fastapi_mod)
    monkeypatch.setitem(sys.modules, "fastapi.middleware", middleware_mod)
    monkeypatch.setitem(sys.modules, "fastapi.middleware.cors", cors_mod)
    monkeypatch.setitem(sys.modules, "fastapi.staticfiles", staticfiles_mod)
    monkeypatch.setitem(sys.modules, "fastapi.responses", responses_mod)
    monkeypatch.setitem(sys.modules, "pydantic", pydantic_mod)
    monkeypatch.setitem(sys.modules, "uvicorn", uvicorn_mod)


@pytest.fixture
def started_server(monkeypatch):
    """Starts the real web_control_server against the faked fastapi/
    pydantic/uvicorn stack, yields (module, app), and always stops it."""
    _install_fake_fastapi_stack(monkeypatch)

    import importlib
    import core.web_control_server as wcserver
    importlib.reload(wcserver)  # re-run start()'s deferred imports against the fakes

    config = types.SimpleNamespace(
        web_control_token="test-token",
        web_control_port=0,
        model_path="",
        inference_backend="auto",
    )

    assert wcserver.start(config) is True
    assert _RecordingFastAPI.instances, "start() never constructed a FastAPI() app"
    app = _RecordingFastAPI.instances[-1]
    try:
        yield wcserver, app
    finally:
        wcserver.stop()


def _routes_by_path(app):
    return {path: func for (_, path, func) in app.routes}


class TestBodyModelAnnotationsResolveToRealClasses:
    """Each of these routes takes a Pydantic BaseModel body parameter that
    is defined *locally* inside start() (since `from pydantic import
    BaseModel` is itself deferred to inside start()). Before this fix,
    `from __future__ import annotations` made every one of these
    annotations an unresolvable string, which FastAPI silently
    reinterpreted as a required query parameter named "body" — every POST
    with a real JSON body then 422'd with `{"loc": ["query", "body"],
    "msg": "Field required"}` no matter what was sent.
    """

    @pytest.mark.parametrize(
        "path,model_name",
        [
            ("/api/control/always_aim", "AlwaysAimBody"),
            ("/api/control/model", "ModelChangeBody"),
            ("/api/model_notes", "ModelNotesBody"),
            ("/api/control/web_esp_enabled", "WebEspEnabledBody"),
            ("/api/configs/save", "ConfigNameBody"),
            ("/api/configs/rename", "ConfigRenameBody"),
            ("/api/configs/import", "ConfigImportBody"),
            ("/api/control/convert", "ConvertBody"),
            ("/api/preset_slots", "PresetSlotBody"),
        ],
    )
    def test_body_param_is_a_real_class_not_a_string(self, started_server, path, model_name):
        _wcserver, app = started_server
        handler = _routes_by_path(app)[path]
        annotation = inspect.signature(handler).parameters["body"].annotation

        # The core assertion: a real class object, never a bare string —
        # `isinstance(x, str)` is exactly the failure mode this regresses
        # against (an unresolved forward-reference annotation).
        assert not isinstance(annotation, str), (
            f"{path}'s body parameter annotation is still a string "
            f"({annotation!r}) — from __future__ import annotations would "
            f"make FastAPI treat it as a required query parameter instead "
            f"of the request body."
        )
        assert isinstance(annotation, type)
        assert issubclass(annotation, _FakeBaseModel)
        assert annotation.__name__ == model_name

    def test_generic_settings_body_is_still_a_plain_dict(self, started_server):
        """POST /api/settings/{tab} takes `body: dict` — dict is a builtin,
        always resolvable regardless of postponed evaluation, so this one
        was never actually broken. Included for contrast/documentation."""
        _wcserver, app = started_server
        handler = _routes_by_path(app)["/api/settings/{tab}"]
        annotation = inspect.signature(handler).parameters["body"].annotation
        assert annotation is dict


class TestPreviewStreamRoute:
    """GET /api/preview_stream — the MJPEG capture-preview stream for the
    "1PC remote-config" case (see CLAUDE.md's Web Control API section).
    The actual JPEG-encoding generator body (_preview_jpeg_frames()) needs
    cv2, which isn't importable in this sandbox (see this file's own
    module docstring) — but a generator function's body never runs until
    it's iterated, and just calling the route handler only *constructs*
    the generator without ever pulling a value from it, so registration
    and response shape are still directly testable without cv2 at all.
    """

    def test_registered_as_get_route(self, started_server):
        _wcserver, app = started_server
        assert "/api/preview_stream" in _routes_by_path(app)

    def test_handler_returns_multipart_streaming_response(self, started_server):
        _wcserver, app = started_server
        handler = _routes_by_path(app)["/api/preview_stream"]
        response = handler()
        assert response.media_type == "multipart/x-mixed-replace; boundary=frame"
        # Lazy by construction — calling the handler builds the generator
        # but never advances it, so this never touches cv2/screen_capture.
        assert inspect.isgenerator(response.content)


def _route_by_method_and_path(app, method, path):
    """_routes_by_path() keys a dict by path alone, so a GET and a POST
    sharing the same path (e.g. /api/preset_slots) collide — this looks up
    a specific (method, path) pair straight from app.routes instead."""
    for registered_method, registered_path, func in app.routes:
        if registered_method == method and registered_path == path:
            return func
    raise KeyError((method, path))


class TestPresetSlotRoutes:
    """GET/POST /api/preset_slots — the Quick Presets sidebar's 5 assignable
    shortcut slots (see CLAUDE.md's Web Control API section). Both routes
    share one path distinguished only by HTTP method, so
    _route_by_method_and_path() above is used instead of _routes_by_path()."""

    def test_get_route_registered_and_delegates_to_get_preset_slots(self, started_server, monkeypatch):
        _wcserver, app = started_server
        calls = []
        monkeypatch.setattr(
            "core.web_control_settings.get_preset_slots",
            lambda config: calls.append(config) or ["a", "", "", "", ""],
        )
        handler = _route_by_method_and_path(app, "GET", "/api/preset_slots")
        assert handler() == {"slots": ["a", "", "", "", ""]}
        assert len(calls) == 1

    def test_post_route_registered_and_delegates_to_set_preset_slot(self, started_server, monkeypatch):
        _wcserver, app = started_server
        calls = []
        monkeypatch.setattr(
            "core.web_control_settings.set_preset_slot",
            lambda config, index, name: calls.append((index, name)) or {"ok": True},
        )
        handler = _route_by_method_and_path(app, "POST", "/api/preset_slots")
        # PresetSlotBody is a local class defined inside start() (not a
        # module global), so it's only reachable via the handler's own
        # resolved parameter annotation — same technique
        # TestBodyModelAnnotationsResolveToRealClasses uses above.
        body_cls = inspect.signature(handler).parameters["body"].annotation
        body = body_cls(index=3, name="my_preset")
        assert handler(body) == {"ok": True}
        assert calls == [(3, "my_preset")]


class TestBuildStatus:
    """_build_status() is a plain module-level function — no FastAPI/
    pydantic involved at all — so it's directly testable without any of
    the fake-stack machinery above."""

    def _config(self, **kwargs):
        return types.SimpleNamespace(**kwargs)

    def test_selected_backend_tracks_config_field_not_current_provider(self):
        """Regression test: the Model panel's Backend <select> must sync
        from config.inference_backend (what model_page.py's own
        _loadFromConfig() uses), not from the live ONNX EP string exposed
        via the "inference_backend" field below — that string (e.g.
        "TensorrtExecutionProvider") never matches any of the select's
        four option values and was leaving the dropdown stuck on "Auto"
        even while TensorRT was genuinely active.
        """
        import core.web_control_server as wcserver

        config = self._config(
            inference_backend="tensorrt",
            current_provider="TensorrtExecutionProvider",
        )
        status = wcserver._build_status(config)
        assert status["selected_backend"] == "tensorrt"
        # The plain-text status readout is untouched — it still prefers
        # the live provider string, which is correct for *that* field.
        assert status["inference_backend"] == "TensorrtExecutionProvider"

    def test_selected_backend_defaults_to_auto(self):
        import core.web_control_server as wcserver

        status = wcserver._build_status(self._config())
        assert status["selected_backend"] == "auto"

    def test_selected_backend_unaffected_by_missing_current_provider(self):
        import core.web_control_server as wcserver

        config = self._config(inference_backend="cpu")
        status = wcserver._build_status(config)
        assert status["selected_backend"] == "cpu"
        assert status["inference_backend"] == "cpu"

    def test_makcu_com_port_no_longer_in_response(self):
        """The sidebar's standalone MAKCU Port stat was removed (dropped in
        favor of the Keys & HW tab's own COM Port field, which already
        shows this) — makcu_com_port is now dead weight in the response,
        confirmed gone rather than just unused."""
        import core.web_control_server as wcserver

        config = self._config(makcu_com_port="COM5")
        status = wcserver._build_status(config)
        assert "makcu_com_port" not in status
        # makcu_connected stays — the Keys & HW tab's MAKCU toggle still
        # syncs from it (see app.js's applyStatus()).
        assert "makcu_connected" in status

    def test_stream_fps_fields_reflect_config(self):
        """screenshot_method/source_fps/udp_recv_fps/udp_dropped_fps —
        added so the web client can show a "Stream FPS" stat for
        uvc/ndi/udp, mirroring status_panel.py's own source_fps_row."""
        import core.web_control_server as wcserver

        config = self._config(
            screenshot_method="udp",
            source_nominal_fps=30.0,
            udp_recv_fps=59.6,
            udp_dropped_fps=2.3,
        )
        status = wcserver._build_status(config)
        assert status["screenshot_method"] == "udp"
        assert status["source_fps"] == 30.0
        assert status["udp_recv_fps"] == 59.6
        assert status["udp_dropped_fps"] == 2.3

    def test_stream_fps_fields_default_when_missing(self):
        import core.web_control_server as wcserver

        status = wcserver._build_status(self._config())
        assert status["screenshot_method"] == "mss"
        assert status["source_fps"] == 0.0
        assert status["udp_recv_fps"] == 0.0
        assert status["udp_dropped_fps"] == 0.0
