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
