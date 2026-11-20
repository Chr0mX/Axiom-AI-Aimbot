"""Effective-value helpers that replace GUI code which used to rewrite saved settings.

Visibility/sync code must only show or hide cards; the saved value is the user's choice and
the runtime derives what actually applies. These cover the pure helpers behind that.
"""
from types import SimpleNamespace as NS

import pytest

from core import uvc_engine, web_control_settings as wcs
from core.ai_loop_utils import cuda_io_binding_effective, fov_follow_effective
from core.config import EXTERNAL_CAPTURE_METHODS


class TestFovFollowEffective:
    @pytest.mark.parametrize("method", ["mss", "dxcam"])
    def test_desktop_capture_honours_the_setting(self, method):
        assert fov_follow_effective(NS(fov_follow_mouse=True, screenshot_method=method)) is True
        assert fov_follow_effective(NS(fov_follow_mouse=False, screenshot_method=method)) is False

    @pytest.mark.parametrize("method", EXTERNAL_CAPTURE_METHODS)
    def test_external_feeds_never_follow_the_cursor(self, method):
        cfg = NS(fov_follow_mouse=True, screenshot_method=method)
        assert fov_follow_effective(cfg) is False
        assert cfg.fov_follow_mouse is True            # the saved choice is left alone

    def test_method_match_is_case_insensitive_and_defaults_are_safe(self):
        assert fov_follow_effective(NS(fov_follow_mouse=True, screenshot_method="UVC")) is False
        assert fov_follow_effective(NS()) is False


class TestCudaIoBindingEffective:
    def test_user_switch_wins_for_any_provider(self):
        assert cuda_io_binding_effective(NS(cuda_io_binding_enabled=True), ["CUDAExecutionProvider"]) is True

    def test_tensorrt_is_always_on_without_touching_the_saved_value(self):
        cfg = NS(cuda_io_binding_enabled=False)
        assert cuda_io_binding_effective(cfg, ["TensorrtExecutionProvider", "CUDAExecutionProvider"]) is True
        assert cfg.cuda_io_binding_enabled is False

    def test_plain_cuda_and_cpu_follow_the_switch(self):
        cfg = NS(cuda_io_binding_enabled=False)
        assert cuda_io_binding_effective(cfg, ["CUDAExecutionProvider"]) is False
        assert cuda_io_binding_effective(cfg, ["CPUExecutionProvider"]) is False
        assert cuda_io_binding_effective(cfg, []) is False


class TestUvcEngine:
    @staticmethod
    def cfg(method="dshow", backend="v1", ffmpeg=False, fmt="yuy2"):
        return NS(uvc_capture_method=method, uvc_dshow_backend=backend, uvc_ffmpeg_enabled=ffmpeg,
                  uvc_video_format=fmt)

    @pytest.mark.parametrize("backend,ffmpeg,want", [
        ("v1", False, "opencv"), ("v2", False, "native"), ("v1", True, "ffmpeg"),
        ("v2", True, "native"),                        # the old illegal combination: v2 wins, as at runtime
    ])
    def test_saved_fields_map_to_one_engine(self, backend, ffmpeg, want):
        assert uvc_engine.saved_engine(self.cfg(backend=backend, ffmpeg=ffmpeg)) == want

    def test_only_directshow_has_a_choice(self):
        for method in ("msmf", "any"):
            assert uvc_engine.engine_from_config(self.cfg(method=method, backend="v2")) == "opencv"
            # ...but the saved choice survives, so going back to dshow restores it
            assert uvc_engine.saved_engine(self.cfg(method=method, backend="v2")) == "native"

    @pytest.mark.parametrize("engine", uvc_engine.ENGINES)
    def test_apply_round_trips_and_never_makes_an_illegal_combination(self, engine):
        c = self.cfg(backend="v2", ffmpeg=True)
        assert uvc_engine.apply_engine(c, engine) == engine
        assert uvc_engine.saved_engine(c) == engine
        assert not (c.uvc_dshow_backend == "v2" and c.uvc_ffmpeg_enabled)

    def test_unknown_engine_means_opencv(self):
        c = self.cfg(backend="v2")
        assert uvc_engine.apply_engine(c, "bogus") == "opencv"
        assert (c.uvc_dshow_backend, c.uvc_ffmpeg_enabled) == ("v1", False)

    def test_native_engine_offers_only_the_formats_the_dll_implements(self):
        assert uvc_engine.video_format_choices("native") == ("mjpeg", "nv12")
        assert "yuy2" in uvc_engine.video_format_choices("opencv")

    def test_effective_format_falls_back_without_rewriting_the_saved_choice(self):
        c = self.cfg(backend="v2", fmt="yuy2")
        assert uvc_engine.effective_video_format(c) == "mjpeg"
        assert c.uvc_video_format == "yuy2"
        c.uvc_dshow_backend = "v1"
        assert uvc_engine.effective_video_format(c) == "yuy2"


class TestWebUvcEngineField:
    """The web client sets the engine through the derived `uvc_engine` schema field."""

    def test_schema_entry_exists_and_old_fields_stay_writable_aliases(self):
        schema = wcs._SCHEMA["capture"]
        assert schema["uvc_engine"]["choices"] == list(uvc_engine.ENGINES)
        assert "uvc_dshow_backend" in schema and "uvc_ffmpeg_enabled" in schema

    def test_read_and_write_go_through_the_shared_helper(self):
        c = NS(uvc_capture_method="dshow", uvc_dshow_backend="v1", uvc_ffmpeg_enabled=False)
        assert wcs.get_tab_settings(c, "capture")["uvc_engine"] == "opencv"
        res = wcs.apply_tab_settings(c, "capture", {"uvc_engine": "Native"})    # case-insensitive like every choice
        assert res["ok"] is True and (c.uvc_dshow_backend, c.uvc_ffmpeg_enabled) == ("v2", False)
        assert wcs.get_tab_settings(c, "capture")["uvc_engine"] == "native"
        wcs.apply_tab_settings(c, "capture", {"uvc_engine": "ffmpeg"})
        assert (c.uvc_dshow_backend, c.uvc_ffmpeg_enabled) == ("v1", True)

    def test_bad_value_leaves_config_untouched(self):
        c = NS(uvc_capture_method="dshow", uvc_dshow_backend="v2", uvc_ffmpeg_enabled=False)
        res = wcs.apply_tab_settings(c, "capture", {"uvc_engine": "bogus"})
        assert res["ok"] is False and c.uvc_dshow_backend == "v2"
