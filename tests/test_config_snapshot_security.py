"""Full-config snapshots must not carry credentials or remote-control settings.

Reproduction that motivated this (before the fix): exporting a snapshot wrote
``web_control_token`` into the file, and loading a shared snapshot switched the
LAN control server on with the file author's token and pointed
``uvc_ffmpeg_path`` at an arbitrary executable.
"""

import json
import os
from unittest.mock import patch

import pytest

from core.config_manager import (
    _SNAPSHOT_EXCLUDED_FIELDS,
    ConfigManager,
)

SECRET_FIELDS = ("web_control_enabled", "web_control_port", "web_control_token",
                 "preset_slots", "uvc_ffmpeg_path", "web_esp_enabled")


def _config():
    with patch("core.config._get_screen_size", return_value=(1920, 1080)):
        from core.config import Config
        return Config()


@pytest.fixture
def full_mgr(tmp_path):
    return ConfigManager(configs_dir=str(tmp_path / "configs"), aim_only=False)


@pytest.fixture
def aim_mgr(tmp_path):
    return ConfigManager(configs_dir=str(tmp_path / "presets"), aim_only=True)


def _write_snapshot(mgr, name, flat):
    path = os.path.join(mgr.configs_dir, f"{name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"name": name, "config": flat}, f)


def test_excluded_set_is_exactly_the_documented_fields():
    assert _SNAPSHOT_EXCLUDED_FIELDS == frozenset(SECRET_FIELDS)


def test_exported_snapshot_contains_no_secret_fields(full_mgr):
    cfg = _config()
    cfg.web_control_token = "super-secret-token"
    cfg.web_control_enabled = True
    cfg.uvc_ffmpeg_path = r"C:\tools\ffmpeg.exe"
    assert full_mgr.save_config(cfg, "mine")

    path = os.path.join(full_mgr.configs_dir, "mine.json")
    raw = open(path, encoding="utf-8").read()
    saved = json.loads(raw)["config"]
    assert "super-secret-token" not in raw
    for field in SECRET_FIELDS:
        assert field not in saved
    # ...while ordinary settings are still there.
    assert "fov_size" in saved and "model_path" in saved


def test_loading_a_hostile_snapshot_does_not_enable_remote_control(full_mgr):
    victim = _config()
    victim.web_control_enabled = False
    victim.web_control_token = "victim-token"
    victim.web_control_port = 8090
    victim.uvc_ffmpeg_path = ""

    _write_snapshot(full_mgr, "shared", {
        "fov_size": 321,                                  # legitimate setting
        "web_control_enabled": True,
        "web_control_token": "known-to-attacker",
        "web_control_port": 1234,
        "preset_slots": ["x", "", "", "", ""],
        "uvc_ffmpeg_path": r"C:\Users\Public\evil.exe",
    })
    assert full_mgr.load_config(victim, "shared") is True

    assert victim.fov_size == 321                          # still applied
    assert victim.web_control_enabled is False
    assert victim.web_control_token == "victim-token"
    assert victim.web_control_port == 8090
    assert victim.uvc_ffmpeg_path == ""
    assert victim.preset_slots != ["x", "", "", "", ""]


def test_legacy_snapshot_with_secrets_loads_without_overwriting_them(full_mgr):
    """Snapshots saved by older versions still contain the token."""
    cfg = _config()
    cfg.web_control_token = "current-token"
    _write_snapshot(full_mgr, "old", {"web_control_token": "stale-token", "fov_size": 250})
    assert full_mgr.load_config(cfg, "old") is True
    assert cfg.web_control_token == "current-token"
    assert cfg.fov_size == 250


def test_preview_does_not_report_or_apply_excluded_fields(full_mgr):
    cfg = _config()
    _write_snapshot(full_mgr, "p", {"web_control_token": "attacker", "web_control_enabled": True})
    changes = full_mgr.preview_config_changes(cfg, "p")
    assert changes == []
    assert cfg.web_control_token != "attacker"


def test_aim_preset_behaviour_unchanged(aim_mgr):
    cfg = _config()
    cfg.fov_size = 222
    assert aim_mgr.save_config(cfg, "mine")
    other = _config()
    other.fov_size = 100
    assert aim_mgr.load_config(other, "mine") is True
    assert other.fov_size == 222


def test_grouped_layout_cannot_reach_excluded_fields(full_mgr):
    """from_dict also reads the nested v2 layout; that must not bypass the
    exclusions either."""
    victim = _config()
    victim.web_control_enabled = False
    victim.web_control_token = "victim-token"
    victim.web_control_port = 8090
    victim.web_esp_enabled = False
    _write_snapshot(full_mgr, "nested", {
        "web_control": {"enabled": True, "token": "known-to-attacker", "port": 1234},
        "web_esp": {"enabled": True},
        "capture": {"uvc": {"ffmpeg_path": r"C:\Users\Public\evil.exe"}},
    })
    assert full_mgr.load_config(victim, "nested") is True
    assert victim.web_control_enabled is False
    assert victim.web_control_token == "victim-token"
    assert victim.web_control_port == 8090
    assert victim.web_esp_enabled is False
    assert victim.uvc_ffmpeg_path == ""


def test_snapshot_cannot_set_one_time_state(full_mgr):
    cfg = _config()
    cfg.disclaimer_agreed = False
    _write_snapshot(full_mgr, "s", {"disclaimer_agreed": True, "first_run_complete": True})
    assert full_mgr.load_config(cfg, "s") is True
    assert cfg.disclaimer_agreed is False


@pytest.mark.parametrize("bad", [
    r"C:\Users\Public\x.onnx",
    r"\\server\share\x.onnx",
    "//server/share/x.onnx",
    "/abs/x.onnx",
])
def test_snapshot_model_path_must_stay_inside_app(full_mgr, bad):
    cfg = _config()
    before_model, before_hud = cfg.model_path, cfg.hud_model_path
    _write_snapshot(full_mgr, "m", {"model_path": bad, "hud_model_path": bad, "fov_size": 260})
    assert full_mgr.load_config(cfg, "m") is True
    assert cfg.model_path == before_model
    assert cfg.hud_model_path == before_hud
    assert cfg.fov_size == 260
    assert full_mgr.preview_config_changes(_config(), "m") == full_mgr.preview_config_changes(
        _config(), "m")


def test_snapshot_relative_model_path_is_applied(full_mgr):
    cfg = _config()
    _write_snapshot(full_mgr, "r", {"model_path": "Model/other.onnx"})
    assert full_mgr.load_config(cfg, "r") is True
    assert cfg.model_path == "Model/other.onnx"


def test_snapshot_values_are_validated(full_mgr):
    cfg = _config()
    _write_snapshot(full_mgr, "v", {"detect_interval": 0, "max_queue_size": 0})
    assert full_mgr.load_config(cfg, "v") is True
    assert cfg.detect_interval == pytest.approx(0.001)
    assert cfg.max_queue_size >= 1


def test_preview_matches_what_a_load_does(full_mgr):
    _write_snapshot(full_mgr, "pv", {
        "model_path": r"C:\evil.onnx",
        "web_control": {"enabled": True},
        "fov_size": 270,
    })
    changes = full_mgr.preview_config_changes(_config(), "pv")
    assert changes and len(changes) == 1
    assert "evil" not in " ".join(changes)
