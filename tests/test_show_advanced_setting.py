"""The global "Show advanced settings" switch: persisted, migrated for existing users, never
carried by a snapshot or preset."""
import json
from unittest.mock import patch


from core import config as cfgmod
from core.config import _FIELD_MAP
from core.config_manager import ConfigManager, _AIM_PRESET_FIELDS, _SNAPSHOT_EXCLUDED_FIELDS


def new_config():
    with patch("core.config._get_screen_size", return_value=(1920, 1080)):
        return cfgmod.Config()


def load_file(tmp_path, data):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    c = new_config()
    assert cfgmod.load_config(c, str(path)) is True
    return c


def test_default_is_off_and_it_is_persisted_under_ui():
    c = new_config()
    assert c.show_advanced_settings is False
    assert _FIELD_MAP["show_advanced_settings"] == "ui.show_advanced_settings"
    c.show_advanced_settings = True
    assert c.to_dict()["ui"]["show_advanced_settings"] is True


def test_an_existing_config_file_from_before_the_switch_keeps_everything_visible(tmp_path):
    c = load_file(tmp_path, {"config_version": 3, "aim": {"fov_size": 250}})
    assert c.fov_size == 250 and c.show_advanced_settings is True


def test_a_legacy_flat_file_also_counts_as_existing(tmp_path):
    assert load_file(tmp_path, {"fov_size": 250}).show_advanced_settings is True


def test_an_explicit_off_survives_a_reload(tmp_path):
    c = load_file(tmp_path, {"config_version": 3, "ui": {"show_advanced_settings": False}})
    assert c.show_advanced_settings is False


def test_an_explicit_on_survives_a_reload(tmp_path):
    assert load_file(tmp_path, {"ui": {"show_advanced_settings": True}}).show_advanced_settings is True


def test_an_empty_file_is_a_fresh_install_not_an_existing_one(tmp_path):
    c = new_config()
    path = tmp_path / "config.json"
    path.write_text("{}", encoding="utf-8")
    cfgmod.load_config(c, str(path))
    assert c.show_advanced_settings is False


def test_from_dict_alone_never_flips_it():
    """Presets and snapshots go through from_dict: it must leave the user's own setting alone."""
    c = new_config()
    c.show_advanced_settings = False
    c.from_dict({"aim": {"fov_size": 250}})
    assert c.show_advanced_settings is False
    c.show_advanced_settings = True
    c.from_dict({"aim": {"fov_size": 250}})
    assert c.show_advanced_settings is True


def test_snapshots_and_presets_never_carry_it(tmp_path):
    assert "show_advanced_settings" in _SNAPSHOT_EXCLUDED_FIELDS
    assert "show_advanced_settings" not in _AIM_PRESET_FIELDS
    mgr = ConfigManager(configs_dir=str(tmp_path), aim_only=False)
    c = new_config()
    c.show_advanced_settings = True
    mgr.save_config(c, "snap")
    saved = (tmp_path / "snap.json").read_text(encoding="utf-8")
    assert "show_advanced_settings" not in saved
    c2 = new_config()
    c2.show_advanced_settings = False
    assert mgr.load_config(c2, "snap")
    assert c2.show_advanced_settings is False
    # a shared snapshot that does contain it must not flip the loader's view either
    (tmp_path / "evil.json").write_text(json.dumps(
        {"kind": "config", "config_version": 3, "name": "evil",
         "settings": {"ui": {"show_advanced_settings": True}}}), encoding="utf-8")
    assert mgr.load_config(c2, "evil")
    assert c2.show_advanced_settings is False
