"""Unit tests for core/web_control_settings.py.

Covers the generic get_tab_settings()/apply_tab_settings() mechanism (pure
logic, no faking needed) plus the model-notes/game-profiles/hud-models file
helpers (tmp_path-based) and the model-info/UVC-probe/NDI-sources
functions, which need sys.modules faking for their deferred heavy imports
(model_detect, core.screen_capture) — same technique already established in
test_app_controller.py for win_utils.makcu_mouse/core.session_utils.
"""

import sys
import types

import pytest

from core import web_control_settings as wcs


def _config(**kwargs):
    return types.SimpleNamespace(**kwargs)


class TestGetTabSettings:
    def test_unknown_tab_returns_empty_dict(self):
        assert wcs.get_tab_settings(_config(), "nonsense") == {}

    def test_reads_bool_int_float_str_fields(self):
        config = _config(
            fov_size=250,
            fov_follow_mouse=1,  # truthy non-bool, must coerce to real bool
            screenshot_method="dxcam",
        )
        result = wcs.get_tab_settings(config, "inference")
        assert result["fov_size"] == 250
        assert result["fov_follow_mouse"] is True

        result = wcs.get_tab_settings(config, "capture")
        assert result["screenshot_method"] == "dxcam"

    def test_applies_display_scale(self):
        # screenshot_interval is stored in seconds; scale=1000 shows ms.
        config = _config(screenshot_interval=0.01)
        result = wcs.get_tab_settings(config, "capture")
        assert result["screenshot_interval"] == pytest.approx(10.0)

    def test_missing_field_is_none(self):
        config = _config()
        result = wcs.get_tab_settings(config, "inference")
        assert result["fov_size"] is None

    def test_capture_adds_extra_computed_keys(self):
        config = _config(uvc_actual_width=1920, uvc_actual_height=1080, uvc_actual_fps=59.9)
        result = wcs.get_tab_settings(config, "capture")
        assert "system_ip" in result
        assert "bind_ip_options" in result
        assert result["bind_ip_options"][0] == "0.0.0.0"
        assert result["uvc_actual_width"] == 1920

    def test_model_tab_has_no_capture_extras(self):
        result = wcs.get_tab_settings(_config(), "model")
        assert "system_ip" not in result


class TestApplyTabSettings:
    def test_unknown_tab(self):
        assert wcs.apply_tab_settings(_config(), "nonsense", {}) == {"ok": False, "reason": "unknown_tab"}

    def test_non_dict_body(self):
        assert wcs.apply_tab_settings(_config(), "capture", "not a dict") == {
            "ok": False, "reason": "invalid_body",
        }

    def test_unknown_field_rejected_without_touching_config(self):
        config = _config(fov_size=100)
        result = wcs.apply_tab_settings(config, "inference", {"fov_size": 200, "bogus_field": 1})
        assert result == {"ok": False, "reason": "unknown_field", "field": "bogus_field"}
        # Atomic: the valid field in the same body must NOT have been
        # applied either, since the whole body failed validation.
        assert config.fov_size == 100

    def test_invalid_choice_rejected(self):
        config = _config(screenshot_method="mss")
        result = wcs.apply_tab_settings(config, "capture", {"screenshot_method": "not_a_real_method"})
        assert result == {"ok": False, "reason": "invalid_choice", "field": "screenshot_method"}
        assert config.screenshot_method == "mss"

    def test_choice_is_case_insensitive(self):
        config = _config(screenshot_method="mss")
        result = wcs.apply_tab_settings(config, "capture", {"screenshot_method": "UVC"})
        assert result["ok"] is True
        assert config.screenshot_method == "uvc"

    def test_invalid_numeric_value_rejected(self):
        config = _config(fov_size=100)
        result = wcs.apply_tab_settings(config, "inference", {"fov_size": "not_a_number"})
        assert result == {"ok": False, "reason": "invalid_value", "field": "fov_size"}
        assert config.fov_size == 100

    def test_bool_coercion(self):
        config = _config(keep_detecting=False)
        wcs.apply_tab_settings(config, "inference", {"keep_detecting": 1})
        assert config.keep_detecting is True

    def test_scale_round_trips_through_config(self):
        """A display value of 10 (ms) for screenshot_interval (scale=1000)
        must be stored on config as 0.01 (seconds)."""
        config = _config(screenshot_interval=0.0)
        wcs.apply_tab_settings(config, "capture", {"screenshot_interval": 10})
        assert config.screenshot_interval == pytest.approx(0.01)

    def test_clamps_to_min_and_max(self):
        config = _config(fov_size=100)
        wcs.apply_tab_settings(config, "inference", {"fov_size": 9999})
        assert config.fov_size == 500  # schema max
        wcs.apply_tab_settings(config, "inference", {"fov_size": -5})
        assert config.fov_size == 50  # schema min

    def test_partial_update_does_not_touch_unrelated_fields(self):
        config = _config(fov_size=100, fov_height=200, keep_detecting=True)
        wcs.apply_tab_settings(config, "inference", {"fov_size": 300})
        assert config.fov_size == 300
        assert config.fov_height == 200
        assert config.keep_detecting is True

    def test_success_returns_applied_settings(self):
        config = _config(fov_size=100)
        result = wcs.apply_tab_settings(config, "inference", {"fov_size": 150})
        assert result["ok"] is True
        assert result["applied"]["fov_size"] == 150


class TestModelNotes:
    def test_empty_model_name_returns_empty_string(self):
        assert wcs.get_model_notes("") == ""

    def test_no_saved_file_returns_default_template(self, tmp_path, monkeypatch):
        monkeypatch.setattr(wcs, "project_root", str(tmp_path))
        text = wcs.get_model_notes("my_model.onnx")
        assert "my_model.onnx" in text
        assert "Game Settings" in text

    def test_save_then_get_round_trips(self, tmp_path, monkeypatch):
        monkeypatch.setattr(wcs, "project_root", str(tmp_path))
        assert wcs.save_model_notes("my_model.onnx", "custom notes here") is True
        assert wcs.get_model_notes("my_model.onnx") == "custom notes here"

    def test_save_empty_model_name_returns_false(self, tmp_path, monkeypatch):
        monkeypatch.setattr(wcs, "project_root", str(tmp_path))
        assert wcs.save_model_notes("", "text") is False

    def test_save_preserves_other_models_notes(self, tmp_path, monkeypatch):
        monkeypatch.setattr(wcs, "project_root", str(tmp_path))
        wcs.save_model_notes("a.onnx", "notes for a")
        wcs.save_model_notes("b.onnx", "notes for b")
        assert wcs.get_model_notes("a.onnx") == "notes for a"
        assert wcs.get_model_notes("b.onnx") == "notes for b"

    def test_corrupt_notes_file_falls_back_to_default(self, tmp_path, monkeypatch):
        monkeypatch.setattr(wcs, "project_root", str(tmp_path))
        (tmp_path / "model_info.json").write_text("not valid json", encoding="utf-8")
        text = wcs.get_model_notes("x.onnx")
        assert "x.onnx" in text


class TestOpenModelFolder:
    def test_missing_model_dir_returns_false(self, tmp_path, monkeypatch):
        monkeypatch.setattr(wcs, "project_root", str(tmp_path))
        assert wcs.open_model_folder() is False

    def test_no_startfile_returns_false(self, tmp_path, monkeypatch):
        (tmp_path / "Model").mkdir()
        monkeypatch.setattr(wcs, "project_root", str(tmp_path))
        monkeypatch.delattr("os.startfile", raising=False)
        assert wcs.open_model_folder() is False


class TestGameProfiles:
    def test_missing_game_json_returns_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr(wcs, "project_root", str(tmp_path))
        assert wcs.get_game_profiles() == {"games": {}}

    def test_reads_game_json(self, tmp_path, monkeypatch):
        import json
        (tmp_path / "game.json").write_text(
            json.dumps({"Apex Legends": "1490,953,1870,1041"}), encoding="utf-8"
        )
        monkeypatch.setattr(wcs, "project_root", str(tmp_path))
        assert wcs.get_game_profiles() == {"games": {"Apex Legends": "1490,953,1870,1041"}}

    def test_corrupt_game_json_returns_empty(self, tmp_path, monkeypatch):
        (tmp_path / "game.json").write_text("not json", encoding="utf-8")
        monkeypatch.setattr(wcs, "project_root", str(tmp_path))
        assert wcs.get_game_profiles() == {"games": {}}


class TestHudModels:
    def test_no_dir_returns_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr(wcs, "project_root", str(tmp_path))
        assert wcs.get_hud_models() == []

    def test_returns_sorted_onnx_basenames(self, tmp_path, monkeypatch):
        hud_dir = tmp_path / "Model_Hud"
        hud_dir.mkdir()
        (hud_dir / "zeta.onnx").write_bytes(b"")
        (hud_dir / "alpha.onnx").write_bytes(b"")
        (hud_dir / "notes.txt").write_bytes(b"")
        monkeypatch.setattr(wcs, "project_root", str(tmp_path))
        assert wcs.get_hud_models() == ["alpha.onnx", "zeta.onnx"]


class TestGetModelInfo:
    def test_missing_file_returns_reason(self, tmp_path, monkeypatch):
        monkeypatch.setattr(wcs, "project_root", str(tmp_path))
        config = _config(current_provider="")
        result = wcs.get_model_info(config, "does/not/exist.onnx")
        assert result == {"ok": False, "reason": "not_found"}

    def test_happy_path_formats_parts_string(self, tmp_path, monkeypatch):
        model_file = tmp_path / "real.onnx"
        model_file.write_bytes(b"")
        monkeypatch.setattr(wcs, "project_root", str(tmp_path))

        fake_module = types.ModuleType("model_detect")
        fake_module.inspect_model = lambda path: {
            "format": "ONNX", "input_size": "640×640", "num_classes": 2,
            "precision": None, "file_size": "10.0 MB",
        }
        monkeypatch.setitem(sys.modules, "model_detect", fake_module)

        config = _config(current_provider="")
        result = wcs.get_model_info(config, str(model_file))
        assert result["ok"] is True
        assert result["text"] == "ONNX  •  Input: 640×640  •  Classes: 2  •  10.0 MB"

    def test_prefers_cached_trt_engine_when_tensorrt_active(self, tmp_path, monkeypatch):
        model_file = tmp_path / "real.onnx"
        model_file.write_bytes(b"")
        trt_cache = tmp_path / "trt_cache"
        trt_cache.mkdir()
        engine_file = trt_cache / "real_fp16.engine"
        engine_file.write_bytes(b"")
        monkeypatch.setattr(wcs, "project_root", str(tmp_path))

        seen_paths = []
        fake_module = types.ModuleType("model_detect")

        def _fake_inspect(path):
            seen_paths.append(path)
            return {"format": "TensorRT Engine", "input_size": "640×640",
                    "num_classes": None, "precision": "FP16", "file_size": "5.0 MB"}

        fake_module.inspect_model = _fake_inspect
        monkeypatch.setitem(sys.modules, "model_detect", fake_module)

        config = _config(current_provider="TensorrtExecutionProvider")
        result = wcs.get_model_info(config, str(model_file))
        assert result["ok"] is True
        assert seen_paths == [str(engine_file)]

    def test_model_detect_unavailable(self, tmp_path, monkeypatch):
        model_file = tmp_path / "real.onnx"
        model_file.write_bytes(b"")
        monkeypatch.setattr(wcs, "project_root", str(tmp_path))
        monkeypatch.setitem(sys.modules, "model_detect", None)  # forces ImportError on import

        config = _config(current_provider="")
        result = wcs.get_model_info(config, str(model_file))
        assert result["ok"] is False
        assert result["reason"] == "model_detect_unavailable"

    def test_inspect_failure_returns_message_as_text(self, tmp_path, monkeypatch):
        model_file = tmp_path / "real.onnx"
        model_file.write_bytes(b"")
        monkeypatch.setattr(wcs, "project_root", str(tmp_path))

        fake_module = types.ModuleType("model_detect")

        def _raise(path):
            raise RuntimeError("corrupt model file")

        fake_module.inspect_model = _raise
        monkeypatch.setitem(sys.modules, "model_detect", fake_module)

        config = _config(current_provider="")
        result = wcs.get_model_info(config, str(model_file))
        assert result == {"ok": True, "text": "corrupt model file"}


class TestProbeUvc:
    def test_screen_capture_unavailable(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "core.screen_capture", None)
        result = wcs.probe_uvc(0, "msmf", 1920, 1080)
        assert result["ok"] is False
        assert result["reason"] == "screen_capture_unavailable"

    def test_happy_path_combines_three_calls(self, monkeypatch):
        fake_module = types.ModuleType("core.screen_capture")
        fake_module.list_supported_uvc_resolutions = lambda device, method: [(1920, 1080), (1280, 720)]
        fake_module.list_supported_uvc_fps = lambda device, w, h, method: [30, 60]
        fake_module.list_uvc_device_names = lambda: ["Webcam A", "Webcam B"]
        monkeypatch.setitem(sys.modules, "core.screen_capture", fake_module)

        result = wcs.probe_uvc(0, "msmf", 1920, 1080)
        assert result["ok"] is True
        assert result["resolutions"] == [[1920, 1080], [1280, 720]]
        assert result["fps_list"] == [30, 60]
        assert result["device_names"] == ["Webcam A", "Webcam B"]

    def test_partial_failure_still_returns_ok_with_empty_lists(self, monkeypatch):
        fake_module = types.ModuleType("core.screen_capture")

        def _raise(*a, **k):
            raise RuntimeError("device busy")

        fake_module.list_supported_uvc_resolutions = _raise
        fake_module.list_supported_uvc_fps = _raise
        fake_module.list_uvc_device_names = _raise
        monkeypatch.setitem(sys.modules, "core.screen_capture", fake_module)

        result = wcs.probe_uvc(0, "msmf", 1920, 1080)
        assert result == {"ok": True, "resolutions": [], "fps_list": [], "device_names": []}


class TestGetNdiSources:
    def test_screen_capture_unavailable(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "core.screen_capture", None)
        result = wcs.get_ndi_sources()
        assert result["ok"] is False
        assert result["reason"] == "screen_capture_unavailable"

    def test_happy_path(self, monkeypatch):
        fake_module = types.ModuleType("core.screen_capture")
        fake_module.list_available_ndi_source_details = lambda: [{"name": "PC (Cam)", "label": "PC (Cam)"}]
        monkeypatch.setitem(sys.modules, "core.screen_capture", fake_module)

        result = wcs.get_ndi_sources()
        assert result == {"ok": True, "sources": [{"name": "PC (Cam)", "label": "PC (Cam)"}]}


class TestGetLocalIps:
    def test_returns_a_list(self):
        # Network-dependent — just confirm it degrades to a list (possibly
        # empty in a sandboxed/offline environment) rather than raising.
        assert isinstance(wcs.get_local_ips(), list)


class TestNestedFieldPaths:
    """_get_nested()/_set_nested() (exercised indirectly through
    get_tab_settings()/apply_tab_settings()) for dotted attribute paths
    (config.humanization.*, the "aim" tab) and list-index paths
    (config.AimKeys[N], the "keys" tab)."""

    def _humanization_config(self, **overrides):
        base = dict(
            enabled=True, intensity=0.5,
            micro_jitter_enabled=True, micro_jitter_base=0.2, micro_jitter_scale=0.025,
            micro_jitter_idle_enabled=False,
            motion_variation_enabled=True, motion_variation_range=0.06,
            speed_shaping_enabled=True, speed_shaping_low=4.0, speed_shaping_high=22.0,
            speed_shaping_low_factor=0.88,
            micro_stutter_enabled=False, micro_stutter_prob=0.03, micro_stutter_min=0.65,
            micro_stutter_max=0.90,
            reaction_variability_enabled=False, reaction_skip_prob=0.015,
        )
        base.update(overrides)
        return types.SimpleNamespace(**base)

    def test_get_reads_dotted_humanization_field(self):
        config = _config(humanization=self._humanization_config(intensity=0.7))
        result = wcs.get_tab_settings(config, "aim")
        assert result["humanization.intensity"] == 0.7

    def test_apply_writes_dotted_humanization_field_without_touching_siblings(self):
        config = _config(humanization=self._humanization_config())
        result = wcs.apply_tab_settings(config, "aim", {"humanization.intensity": 0.9})
        assert result["ok"] is True
        assert config.humanization.intensity == 0.9
        assert config.humanization.enabled is True  # untouched

    def test_get_reads_list_index_field(self):
        config = _config(AimKeys=[1, 6, 2])
        result = wcs.get_tab_settings(config, "keys")
        assert result["AimKeys.0"] == 1
        assert result["AimKeys.1"] == 6
        assert result["AimKeys.2"] == 2

    def test_apply_writes_list_index_field_without_touching_siblings(self):
        config = _config(AimKeys=[1, 6, 2])
        result = wcs.apply_tab_settings(config, "keys", {"AimKeys.1": 4})
        assert result["ok"] is True
        assert config.AimKeys == [1, 4, 2]

    def test_get_missing_intermediate_object_is_none(self):
        config = _config()  # no .humanization attribute at all
        result = wcs.get_tab_settings(config, "aim")
        assert result["humanization.intensity"] is None

    def test_apply_missing_intermediate_object_does_not_raise(self):
        config = _config()  # no .humanization
        # _set_nested() returns False for a missing intermediate object;
        # the write loop discards that return value (there's nothing to
        # meaningfully report per-field once validation has already
        # passed), so the overall apply still reports ok.
        result = wcs.apply_tab_settings(config, "aim", {"humanization.intensity": 0.9})
        assert result["ok"] is True

    def test_plain_field_unaffected_by_nested_path_generalization(self):
        """A bare (non-dotted) field must behave exactly as before —
        _get_nested()/_set_nested()'s loop runs zero iterations for a
        single-segment path, falling straight through to getattr/setattr."""
        config = _config(fov_size=100)
        result = wcs.apply_tab_settings(config, "inference", {"fov_size": 200})
        assert result["ok"] is True
        assert config.fov_size == 200


class TestIntValuedChoices:
    """"choice" fields whose choices list holds ints (cam_motion_comp_size,
    makcu_baud_rate) instead of the usual strings — must round-trip as int,
    not get coerced into a string."""

    def test_cam_motion_comp_size_round_trips_as_int(self):
        config = _config(cam_motion_comp_size=128)
        result = wcs.apply_tab_settings(config, "aim", {"cam_motion_comp_size": 256})
        assert result["ok"] is True
        assert config.cam_motion_comp_size == 256
        assert isinstance(config.cam_motion_comp_size, int)

    def test_cam_motion_comp_size_accepts_string_value(self):
        """The HTML <select>'s value attribute is always a string — the
        wire value from a real browser is "256", not the number 256."""
        config = _config(cam_motion_comp_size=128)
        result = wcs.apply_tab_settings(config, "aim", {"cam_motion_comp_size": "256"})
        assert result["ok"] is True
        assert config.cam_motion_comp_size == 256
        assert isinstance(config.cam_motion_comp_size, int)

    def test_cam_motion_comp_size_rejects_invalid_value(self):
        config = _config(cam_motion_comp_size=128)
        result = wcs.apply_tab_settings(config, "aim", {"cam_motion_comp_size": 512})
        assert result == {"ok": False, "reason": "invalid_choice", "field": "cam_motion_comp_size"}
        assert config.cam_motion_comp_size == 128

    def test_makcu_baud_rate_round_trips_as_int(self):
        config = _config(makcu_baud_rate=4_000_000)
        result = wcs.apply_tab_settings(config, "keys", {"makcu_baud_rate": 115200})
        assert result["ok"] is True
        assert config.makcu_baud_rate == 115200
        assert isinstance(config.makcu_baud_rate, int)


class TestListVkOptions:
    def test_missing_win_utils_falls_back_to_none_only(self, monkeypatch):
        """Forced-unavailable case, not a "real sandbox condition" test:
        other test files in this suite (test_makcu_mouse.py) leave a
        working win32api stand-in cached in sys.modules for the rest of
        the process once they've run, which makes the real
        win_utils.vk_codes genuinely importable again later in a full
        `pytest tests/` run — so asserting "win_utils is unimportable here"
        would be order-dependent. Force it explicitly instead."""
        # Both keys: "win_utils.vk_codes" can already be cached in
        # sys.modules independently of its parent package (e.g. from an
        # earlier real import elsewhere in the same process), so forcing
        # only the parent to None isn't sufficient to guarantee the import
        # fails here too.
        monkeypatch.setitem(sys.modules, "win_utils", None)
        monkeypatch.setitem(sys.modules, "win_utils.vk_codes", None)
        options = wcs.list_vk_options()
        assert options == [{"code": 0, "label": "None / Unbound"}]

    def test_happy_path_includes_named_entries(self, monkeypatch):
        fake_pkg = types.ModuleType("win_utils")
        fake_submodule = types.ModuleType("win_utils.vk_codes")
        fake_submodule.VK_CODE_MAP = {0x01: "Mouse Left", 0x41: "A"}
        monkeypatch.setitem(sys.modules, "win_utils", fake_pkg)
        monkeypatch.setitem(sys.modules, "win_utils.vk_codes", fake_submodule)

        options = wcs.list_vk_options()
        assert options[0] == {"code": 0, "label": "None / Unbound"}
        assert {"code": 1, "label": "Mouse Left"} in options
        assert {"code": 0x41, "label": "A"} in options
        assert len(options) == 3


class TestGetSerialPorts:
    def test_serial_unavailable(self, monkeypatch):
        """Forced-unavailable case (see TestListVkOptions's equivalent test
        for why this can't rely on ambient sandbox state across the full
        suite)."""
        monkeypatch.setitem(sys.modules, "serial", None)
        result = wcs.get_serial_ports()
        assert result["ok"] is False
        assert result["reason"] == "serial_unavailable"

    def test_happy_path(self, monkeypatch):
        class _FakePortInfo:
            def __init__(self, device):
                self.device = device

        fake_serial = types.ModuleType("serial")
        fake_tools = types.ModuleType("serial.tools")
        fake_list_ports = types.ModuleType("serial.tools.list_ports")
        fake_list_ports.comports = lambda: [_FakePortInfo("COM3"), _FakePortInfo("COM5")]
        fake_tools.list_ports = fake_list_ports
        fake_serial.tools = fake_tools
        monkeypatch.setitem(sys.modules, "serial", fake_serial)
        monkeypatch.setitem(sys.modules, "serial.tools", fake_tools)
        monkeypatch.setitem(sys.modules, "serial.tools.list_ports", fake_list_ports)

        result = wcs.get_serial_ports()
        assert result == {"ok": True, "ports": ["COM3", "COM5"]}


class TestResetHumanization:
    def test_replaces_humanization_with_fresh_defaults(self):
        stale = types.SimpleNamespace(enabled=False, intensity=0.1)
        config = _config(humanization=stale)
        result = wcs.reset_humanization(config)
        assert result == {"ok": True}
        # core.humanization is pure Python (no Qt/onnxruntime), genuinely
        # importable here — real HumanizationConfig() defaults, not faked.
        assert config.humanization is not stale
        assert config.humanization.enabled is True
        assert config.humanization.intensity == 0.5

    def test_humanization_unavailable(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "core.humanization", None)
        config = _config()
        result = wcs.reset_humanization(config)
        assert result["ok"] is False
        assert result["reason"] == "humanization_unavailable"


class TestAimAndKeysSchemaCoverage:
    """Sanity checks that the aim/keys schemas are internally consistent —
    complements the cross-file data-key-vs-schema check done by hand while
    building the frontend (every data-key in index.html resolves to a real
    schema entry, and vice versa for anything wired to a live field)."""

    def test_get_tab_settings_returns_every_schema_key(self):
        config = _config()
        result = wcs.get_tab_settings(config, "aim")
        assert set(result.keys()) >= set(wcs._SCHEMA["aim"].keys())

    def test_apply_empty_update_is_a_no_op_success(self):
        config = _config(fov_size=123)
        result = wcs.apply_tab_settings(config, "aim", {})
        assert result["ok"] is True
        assert config.fov_size == 123


def _fake_esp_server_module(monkeypatch, **attrs):
    """Fakes core.esp_server for a deferred `from core import esp_server`.

    Patching sys.modules["core.esp_server"] alone isn't sufficient once
    something elsewhere in a full-suite run has already done a REAL `from
    core import esp_server` (test_esp_server.py does, at module level):
    that binds a real `esp_server` attribute directly on the `core`
    package object, and `from core import esp_server` resolves via a
    plain getattr(core, "esp_server") first — succeeding against that
    cached real attribute — before it would ever fall back to consulting
    sys.modules. So the package attribute has to be patched too, same
    two-level fix as TestListVkOptions/TestGetSerialPorts' "module
    unavailable" tests above.
    """
    fake_module = types.ModuleType("core.esp_server")
    for key, value in attrs.items():
        setattr(fake_module, key, value)
    monkeypatch.setitem(sys.modules, "core.esp_server", fake_module)
    import core
    monkeypatch.setattr(core, "esp_server", fake_module, raising=False)
    return fake_module


def _clear_esp_server_module(monkeypatch):
    """Forces `from core import esp_server` to fail with ImportError —
    both levels again (see _fake_esp_server_module's docstring): a bare
    sys.modules[...] = None wouldn't be seen at all if core.esp_server is
    already a real cached attribute on the core package."""
    monkeypatch.setitem(sys.modules, "core.esp_server", None)
    import core
    monkeypatch.delattr(core, "esp_server", raising=False)


class TestVisualsSchema:
    """_SCHEMA["visuals"] round-trip + its read-only web_esp_* extras.
    core.esp_server has no win32api/Qt import (see CLAUDE.md's testability
    note), so it's faked via sys.modules the same way core.screen_capture
    is above — purely to avoid a real get_tab_settings() call binding an
    actual socket/thread, not because the real module is unimportable."""

    def _install_fake_esp_server(self, monkeypatch, running=False, url=""):
        _fake_esp_server_module(monkeypatch, is_running=lambda: running, connect_url=lambda: url)

    def test_get_tab_settings_returns_every_schema_key(self, monkeypatch):
        self._install_fake_esp_server(monkeypatch)
        result = wcs.get_tab_settings(_config(), "visuals")
        assert set(result.keys()) >= set(wcs._SCHEMA["visuals"].keys())

    def test_reads_bool_choice_int_fields(self, monkeypatch):
        self._install_fake_esp_server(monkeypatch)
        config = _config(
            show_fov=True, box_color_theme="cyan", chroma_box_speed=5,
            crosshair_color_r=10, crosshair_color_g=20, crosshair_color_b=30,
        )
        result = wcs.get_tab_settings(config, "visuals")
        assert result["show_fov"] is True
        assert result["box_color_theme"] == "cyan"
        assert result["chroma_box_speed"] == 5
        assert result["crosshair_color_r"] == 10

    def test_apply_rejects_invalid_box_color_theme(self, monkeypatch):
        self._install_fake_esp_server(monkeypatch)
        config = _config(box_color_theme="default")
        result = wcs.apply_tab_settings(config, "visuals", {"box_color_theme": "not_a_theme"})
        assert result == {"ok": False, "reason": "invalid_choice", "field": "box_color_theme"}
        assert config.box_color_theme == "default"

    def test_apply_clamps_crosshair_size(self, monkeypatch):
        self._install_fake_esp_server(monkeypatch)
        config = _config(crosshair_size=5)
        wcs.apply_tab_settings(config, "visuals", {"crosshair_size": 999})
        assert config.crosshair_size == 20  # schema max

    def test_web_esp_enabled_is_not_a_writable_schema_field(self):
        """Deliberately excluded — see the _SCHEMA["visuals"] comment.
        Enabling/disabling must go through the dedicated action route
        (app_controller.set_web_esp_enabled), not a plain setattr, since a
        bare Config write wouldn't start/stop the real server."""
        assert "web_esp_enabled" not in wcs._SCHEMA["visuals"]

    def test_extras_reflect_not_running(self, monkeypatch):
        self._install_fake_esp_server(monkeypatch, running=False)
        config = _config(web_esp_enabled=True)
        result = wcs.get_tab_settings(config, "visuals")
        assert result["web_esp_enabled"] is True
        assert result["web_esp_running"] is False
        assert result["web_esp_url"] == ""

    def test_extras_reflect_running_with_url(self, monkeypatch):
        self._install_fake_esp_server(monkeypatch, running=True, url="http://127.0.0.1:8080/?ws=8765")
        config = _config(web_esp_enabled=True)
        result = wcs.get_tab_settings(config, "visuals")
        assert result["web_esp_running"] is True
        assert result["web_esp_url"] == "http://127.0.0.1:8080/?ws=8765"

    def test_extras_degrade_gracefully_when_esp_server_unavailable(self, monkeypatch):
        _clear_esp_server_module(monkeypatch)
        config = _config(web_esp_enabled=False)
        result = wcs.get_tab_settings(config, "visuals")
        assert result["web_esp_running"] is False
        assert result["web_esp_url"] == ""


class TestOpenWebEspInBrowser:
    def test_returns_false_when_not_running(self, monkeypatch):
        _fake_esp_server_module(monkeypatch, is_running=lambda: False)
        assert wcs.open_web_esp_in_browser() is False

    def test_returns_false_when_url_empty(self, monkeypatch):
        _fake_esp_server_module(monkeypatch, is_running=lambda: True, connect_url=lambda: "")
        assert wcs.open_web_esp_in_browser() is False

    def test_opens_browser_when_running(self, monkeypatch):
        _fake_esp_server_module(
            monkeypatch, is_running=lambda: True,
            connect_url=lambda: "http://127.0.0.1:8080/?ws=8765",
        )
        opened = []
        monkeypatch.setattr("webbrowser.open", lambda url: opened.append(url))
        assert wcs.open_web_esp_in_browser() is True
        assert opened == ["http://127.0.0.1:8080/?ws=8765"]

    def test_esp_server_unavailable_returns_false(self, monkeypatch):
        _clear_esp_server_module(monkeypatch)
        assert wcs.open_web_esp_in_browser() is False


class TestConfigPresets:
    """Preset CRUD wrappers around core.config_manager.ConfigManager — a
    pure-Python module (no Qt/onnxruntime), so real ConfigManager instances
    are used here rather than faked, pointed at a tmp_path directory via
    monkeypatching wcs._config_manager() (the same one-ConfigManager-per-
    call factory the real routes use). src/core/presets/ currently ships no
    bundled built-in presets (the one that used to live there, "Apex MAKCU
    UDP Precision", was removed), so a fresh ConfigManager's
    _seed_builtin_presets() is a no-op and a brand-new directory starts
    genuinely empty.
    """

    @pytest.fixture(autouse=True)
    def _patch_config_manager(self, tmp_path, monkeypatch):
        from core.config_manager import ConfigManager
        self.configs_dir = tmp_path / "config"
        monkeypatch.setattr(wcs, "_config_manager", lambda: ConfigManager(str(self.configs_dir)))

    def _real_config(self):
        # Config() calls _get_screen_size(), which uses ctypes.windll —
        # unavailable outside Windows (see test_config.py's _make_config(),
        # the same established pattern for this sandbox).
        from unittest.mock import patch
        from core.config import Config
        with patch("core.config._get_screen_size", return_value=(1920, 1080)):
            return Config()

    def test_fresh_directory_has_no_presets(self):
        assert wcs.list_config_presets() == []

    def test_save_then_list(self):
        result = wcs.save_config_preset(self._real_config(), "my_preset")
        assert result == {"ok": True}
        assert "my_preset" in wcs.list_config_presets()

    def test_save_empty_name_fails(self):
        result = wcs.save_config_preset(self._real_config(), "")
        assert result == {"ok": False}

    def test_preview_identical_config_returns_no_changes(self):
        config = self._real_config()
        wcs.save_config_preset(config, "identical")
        result = wcs.preview_config_preset(config, "identical")
        assert result == {"ok": True, "changes": []}

    def test_preview_missing_preset_returns_read_failed(self):
        result = wcs.preview_config_preset(self._real_config(), "does_not_exist")
        assert result == {"ok": False, "reason": "read_failed"}

    def test_preview_reports_a_real_change(self):
        config = self._real_config()
        config.fov_size = 150
        wcs.save_config_preset(config, "wide_fov")
        config.fov_size = 50
        result = wcs.preview_config_preset(config, "wide_fov")
        assert result["ok"] is True
        assert len(result["changes"]) >= 1

    def test_load_applies_saved_values(self):
        saved = self._real_config()
        saved.fov_size = 275
        wcs.save_config_preset(saved, "custom_fov")

        target = self._real_config()
        target.fov_size = 100
        result = wcs.load_config_preset(target, "custom_fov")
        assert result == {"ok": True}
        assert target.fov_size == 275

    def test_load_missing_preset_fails(self):
        result = wcs.load_config_preset(self._real_config(), "no_such_preset")
        assert result == {"ok": False}

    def test_delete_removes_preset(self):
        wcs.save_config_preset(self._real_config(), "to_delete")
        assert wcs.delete_config_preset("to_delete") == {"ok": True}
        assert "to_delete" not in wcs.list_config_presets()

    def test_delete_missing_preset_fails(self):
        assert wcs.delete_config_preset("never_existed") == {"ok": False}

    def test_rename_updates_list(self):
        wcs.save_config_preset(self._real_config(), "old_name")
        result = wcs.rename_config_preset("old_name", "new_name")
        assert result == {"ok": True}
        presets = wcs.list_config_presets()
        assert "new_name" in presets
        assert "old_name" not in presets

    def test_rename_missing_preset_fails(self):
        result = wcs.rename_config_preset("does_not_exist", "whatever")
        assert result == {"ok": False}

    def test_open_configs_folder_missing_dir_returns_false(self):
        # The autouse fixture points configs_dir at a path ConfigManager
        # itself creates via ensure_configs_directory() — remove it first
        # so this genuinely exercises the "not os.path.isdir" guard.
        import shutil
        shutil.rmtree(self.configs_dir, ignore_errors=True)
        assert wcs.open_configs_folder() is False

    def test_open_configs_folder_no_startfile_returns_false(self, monkeypatch):
        wcs.save_config_preset(self._real_config(), "anything")  # ensures dir exists
        monkeypatch.delattr("os.startfile", raising=False)
        assert wcs.open_configs_folder() is False


class TestExportImportConfigPreset:
    @pytest.fixture(autouse=True)
    def _patch_config_manager(self, tmp_path, monkeypatch):
        from core.config_manager import ConfigManager
        self.configs_dir = tmp_path / "config"
        monkeypatch.setattr(wcs, "_config_manager", lambda: ConfigManager(str(self.configs_dir)))

    def _real_config(self):
        # Config() calls _get_screen_size(), which uses ctypes.windll —
        # unavailable outside Windows (see test_config.py's _make_config(),
        # the same established pattern for this sandbox).
        from unittest.mock import patch
        from core.config import Config
        with patch("core.config._get_screen_size", return_value=(1920, 1080)):
            return Config()

    def test_export_missing_preset_returns_not_found(self):
        result = wcs.export_config_preset_content("does_not_exist")
        assert result == {"ok": False, "reason": "not_found"}

    def test_export_returns_raw_json_content(self):
        config = self._real_config()
        config.fov_size = 321
        wcs.save_config_preset(config, "export_me")
        result = wcs.export_config_preset_content("export_me")
        assert result["ok"] is True
        assert result["name"] == "export_me"
        import json
        parsed = json.loads(result["content"])
        assert parsed["config"]["fov_size"] == 321

    def test_import_invalid_json_rejected(self):
        result = wcs.import_config_preset_content("not valid json")
        assert result == {"ok": False, "reason": "invalid_json"}

    def test_import_non_dict_json_rejected(self):
        result = wcs.import_config_preset_content("[1, 2, 3]")
        assert result == {"ok": False, "reason": "invalid_json"}

    def test_import_uses_name_field_from_content(self):
        import json
        content = json.dumps({"name": "imported_preset", "config": {"fov_size": 111}})
        result = wcs.import_config_preset_content(content)
        assert result == {"ok": True, "name": "imported_preset"}
        assert "imported_preset" in wcs.list_config_presets()

    def test_import_missing_name_falls_back_to_default(self):
        import json
        content = json.dumps({"config": {"fov_size": 111}})
        result = wcs.import_config_preset_content(content)
        assert result["ok"] is True
        assert result["name"] == "imported_config"

    def test_import_never_overwrites_existing_preset(self):
        import json
        wcs.save_config_preset(self._real_config(), "dup_name")
        content = json.dumps({"name": "dup_name", "config": {"fov_size": 999}})
        result = wcs.import_config_preset_content(content)
        assert result["ok"] is True
        assert result["name"] == "dup_name_1"
        assert "dup_name" in wcs.list_config_presets()
        assert "dup_name_1" in wcs.list_config_presets()

    def test_import_sanitizes_unsafe_name(self):
        import json
        content = json.dumps({"name": "../../evil", "config": {}})
        result = wcs.import_config_preset_content(content)
        assert result["ok"] is True
        # basename() strips the path traversal — only "evil" (or its
        # sanitized form) should ever land inside configs_dir.
        assert "/" not in result["name"]
        assert ".." not in result["name"]

    def test_export_then_import_round_trips_via_content(self):
        """The whole content-based Export -> (client downloads) -> Import
        flow, without ever touching a host-side file path."""
        config = self._real_config()
        config.fov_size = 456
        wcs.save_config_preset(config, "round_trip_source")
        exported = wcs.export_config_preset_content("round_trip_source")
        assert exported["ok"] is True

        wcs.delete_config_preset("round_trip_source")
        imported = wcs.import_config_preset_content(exported["content"])
        assert imported["ok"] is True
        assert imported["name"] == "round_trip_source"

        fresh = self._real_config()
        fresh.fov_size = 1
        wcs.load_config_preset(fresh, "round_trip_source")
        assert fresh.fov_size == 456


class TestPresetSlots:
    """get_preset_slots()/set_preset_slot() — the web control client's
    5-button "Quick Presets" sidebar. Persistence goes through
    core.config.save_config(), so it's patched to a tmp-dir config.json the
    same way test_app_controller.py patches it for other Config-writing
    functions; preset-name validation goes through wcs._config_manager(),
    patched exactly like TestConfigPresets/TestExportImportConfigPreset
    above."""

    @pytest.fixture(autouse=True)
    def _patch_config_manager(self, tmp_path, monkeypatch):
        from core.config_manager import ConfigManager
        self.configs_dir = tmp_path / "config"
        monkeypatch.setattr(wcs, "_config_manager", lambda: ConfigManager(str(self.configs_dir)))

    def _real_config(self):
        from unittest.mock import patch
        from core.config import Config
        with patch("core.config._get_screen_size", return_value=(1920, 1080)):
            return Config()

    def test_default_slots_are_five_blank_strings(self):
        config = self._real_config()
        assert wcs.get_preset_slots(config) == ["", "", "", "", ""]

    def test_get_pads_a_short_list(self):
        config = self._real_config()
        config.preset_slots = ["only_one"]
        assert wcs.get_preset_slots(config) == ["only_one", "", "", "", ""]

    def test_get_truncates_a_long_list(self):
        config = self._real_config()
        config.preset_slots = ["a", "b", "c", "d", "e", "f", "g"]
        assert wcs.get_preset_slots(config) == ["a", "b", "c", "d", "e"]

    def test_get_handles_missing_attribute(self):
        config = self._real_config()
        del config.preset_slots
        assert wcs.get_preset_slots(config) == ["", "", "", "", ""]

    def test_set_rejects_negative_index(self):
        config = self._real_config()
        result = wcs.set_preset_slot(config, -1, "")
        assert result == {"ok": False, "reason": "invalid_index"}

    def test_set_rejects_index_out_of_range(self):
        config = self._real_config()
        result = wcs.set_preset_slot(config, 5, "")
        assert result == {"ok": False, "reason": "invalid_index"}

    def test_set_rejects_non_int_index(self):
        config = self._real_config()
        result = wcs.set_preset_slot(config, "0", "")
        assert result == {"ok": False, "reason": "invalid_index"}

    def test_set_rejects_a_name_that_is_not_a_saved_preset(self):
        config = self._real_config()
        result = wcs.set_preset_slot(config, 0, "no_such_preset")
        assert result == {"ok": False, "reason": "not_found"}
        assert wcs.get_preset_slots(config)[0] == ""

    def test_set_assigns_a_real_preset_name(self, monkeypatch):
        config = self._real_config()
        wcs.save_config_preset(config, "my_aim_preset")

        saved = {}
        monkeypatch.setattr(
            "core.config.save_config",
            lambda cfg: saved.update(called=True, slots=list(cfg.preset_slots)),
        )

        result = wcs.set_preset_slot(config, 2, "my_aim_preset")
        assert result == {"ok": True}
        assert config.preset_slots[2] == "my_aim_preset"
        assert config.preset_slots == ["", "", "my_aim_preset", "", ""]
        assert saved.get("called") is True

    def test_set_empty_name_clears_a_slot_without_needing_a_real_preset(self, monkeypatch):
        config = self._real_config()
        wcs.save_config_preset(config, "my_aim_preset")
        monkeypatch.setattr("core.config.save_config", lambda cfg: None)
        wcs.set_preset_slot(config, 1, "my_aim_preset")

        result = wcs.set_preset_slot(config, 1, "")
        assert result == {"ok": True}
        assert config.preset_slots[1] == ""

    def test_set_survives_save_config_raising(self, monkeypatch):
        """A persistence failure must not surface as a route-level error —
        the in-memory assignment (and the eventual real config.json write on
        some later successful save) is what matters, matching this module's
        existing best-effort save pattern elsewhere."""
        config = self._real_config()
        wcs.save_config_preset(config, "my_aim_preset")
        monkeypatch.setattr(
            "core.config.save_config",
            lambda cfg: (_ for _ in ()).throw(RuntimeError("disk full")),
        )
        result = wcs.set_preset_slot(config, 0, "my_aim_preset")
        assert result == {"ok": True}
        assert config.preset_slots[0] == "my_aim_preset"


class TestTriggerSchema:
    def test_get_tab_settings_returns_every_schema_key(self):
        config = _config()
        result = wcs.get_tab_settings(config, "trigger")
        assert set(result.keys()) >= set(wcs._SCHEMA["trigger"].keys())

    def test_reads_all_fields(self):
        config = _config(
            auto_fire_target_part="head", always_auto_fire=True,
            mouse_click_method="makcu", auto_fire_delay=0.5, auto_fire_interval=0.2,
        )
        result = wcs.get_tab_settings(config, "trigger")
        assert result["auto_fire_target_part"] == "head"
        assert result["always_auto_fire"] is True
        assert result["mouse_click_method"] == "makcu"
        assert result["auto_fire_delay"] == 0.5
        assert result["auto_fire_interval"] == 0.2

    def test_apply_rejects_invalid_target_part(self):
        config = _config(auto_fire_target_part="both")
        result = wcs.apply_tab_settings(config, "trigger", {"auto_fire_target_part": "legs"})
        assert result == {"ok": False, "reason": "invalid_choice", "field": "auto_fire_target_part"}
        assert config.auto_fire_target_part == "both"

    def test_apply_rejects_mouse_click_method_outside_triggers_narrower_list(self):
        """The "trigger" tab's mouse_click_method entry is independently
        validated against trigger_page.py's own 6-item combo list, tighter
        than the "aim" tab's unrestricted "str" entry for the same field —
        see _SCHEMA["trigger"]'s comment."""
        config = _config(mouse_click_method="mouse_event")
        result = wcs.apply_tab_settings(config, "trigger", {"mouse_click_method": "not_a_real_method"})
        assert result == {"ok": False, "reason": "invalid_choice", "field": "mouse_click_method"}
        # The "aim" tab's own (looser) entry for the same field must be
        # completely unaffected by "trigger"'s stricter validation.
        result2 = wcs.apply_tab_settings(config, "aim", {"mouse_click_method": "not_a_real_method"})
        assert result2["ok"] is True
        assert config.mouse_click_method == "not_a_real_method"

    def test_apply_clamps_auto_fire_delay_and_interval(self):
        config = _config(auto_fire_delay=1.0, auto_fire_interval=0.5)
        wcs.apply_tab_settings(config, "trigger", {"auto_fire_delay": 99, "auto_fire_interval": 99})
        assert config.auto_fire_delay == 2.0  # schema max
        assert config.auto_fire_interval == 1.0  # schema max

    def test_always_auto_fire_bool_coercion(self):
        config = _config(always_auto_fire=False)
        wcs.apply_tab_settings(config, "trigger", {"always_auto_fire": 1})
        assert config.always_auto_fire is True


class TestConvertSchema:
    def test_get_tab_settings_returns_every_schema_key(self):
        config = _config()
        result = wcs.get_tab_settings(config, "convert")
        assert set(result.keys()) >= set(wcs._SCHEMA["convert"].keys())

    def test_reads_and_writes_trt_fp16_enabled(self):
        config = _config(trt_fp16_enabled=False)
        result = wcs.get_tab_settings(config, "convert")
        assert result["trt_fp16_enabled"] is False

        wcs.apply_tab_settings(config, "convert", {"trt_fp16_enabled": True})
        assert config.trt_fp16_enabled is True
