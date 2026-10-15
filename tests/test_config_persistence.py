"""config.json durability: atomic saves and no silent loss of unreadable files."""

import glob
import json
import os
from unittest.mock import patch

import pytest


@pytest.fixture
def cfg():
    with patch("core.config._get_screen_size", return_value=(1920, 1080)):
        from core.config import Config
        return Config()


def _read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


class TestAtomicSave:
    def test_round_trip(self, cfg, tmp_path):
        from core.config import Config, load_config, save_config

        path = str(tmp_path / "config.json")
        cfg.fov_size = 321
        assert save_config(cfg, path) is True
        with patch("core.config._get_screen_size", return_value=(1920, 1080)):
            fresh = Config()
        assert load_config(fresh, path) is True
        assert fresh.fov_size == 321

    def test_serialization_failure_keeps_existing_file(self, cfg, tmp_path):
        """The old code opened with 'w' first, so a failure while serializing
        left a 0-byte config.json."""
        from core.config import save_config

        path = str(tmp_path / "config.json")
        with open(path, "w", encoding="utf-8") as f:
            f.write('{"aim": {"fov_size": 123}}')

        with patch.object(type(cfg), "to_dict", side_effect=TypeError("boom")):
            assert save_config(cfg, path) is False

        assert _read(path) == '{"aim": {"fov_size": 123}}'
        assert glob.glob(str(tmp_path / "*.tmp")) == []

    def test_failure_while_swapping_keeps_existing_file_and_cleans_up(self, cfg, tmp_path):
        from core.config import save_config

        path = str(tmp_path / "config.json")
        with open(path, "w", encoding="utf-8") as f:
            f.write('{"original": true}')

        with patch("core.config.os.replace", side_effect=PermissionError("locked")):
            assert save_config(cfg, path) is False

        assert json.loads(_read(path)) == {"original": True}
        assert glob.glob(str(tmp_path / "*.tmp")) == []

    def test_state_json_is_written_atomically_too(self, cfg, tmp_path):
        from core.config import save_state

        path = str(tmp_path / "state.json")
        with open(path, "w", encoding="utf-8") as f:
            f.write('{"disclaimer_agreed": true}')
        with patch("core.config.os.replace", side_effect=OSError("disk")):
            assert save_state(cfg, path) is False
        assert json.loads(_read(path)) == {"disclaimer_agreed": True}


class TestUnreadableConfigIsPreserved:
    def test_truncated_json_is_moved_aside_not_lost(self, cfg, tmp_path):
        from core.config import load_config, save_config

        path = str(tmp_path / "config.json")
        broken = '{"aim": {"fov_si'
        with open(path, "w", encoding="utf-8") as f:
            f.write(broken)

        assert load_config(cfg, path) is False
        backups = glob.glob(str(tmp_path / "config.json.corrupt-*"))
        assert len(backups) == 1 and _read(backups[0]) == broken
        assert not os.path.exists(path)

        # main.py saves on shutdown: that must not clobber the preserved copy.
        assert save_config(cfg, path) is True
        assert _read(backups[0]) == broken

    def test_non_object_top_level_does_not_crash(self, cfg, tmp_path):
        """A JSON list used to raise AttributeError out of load_config()."""
        from core.config import load_config

        path = str(tmp_path / "config.json")
        with open(path, "w", encoding="utf-8") as f:
            f.write("[1, 2, 3]")

        assert load_config(cfg, path) is False
        assert len(glob.glob(str(tmp_path / "config.json.corrupt-*"))) == 1

    def test_empty_file_is_not_quarantined(self, cfg, tmp_path):
        from core.config import load_config

        path = str(tmp_path / "config.json")
        open(path, "w").close()
        assert load_config(cfg, path) is False
        assert glob.glob(str(tmp_path / "config.json.corrupt-*")) == []

    def test_missing_file_still_just_uses_defaults(self, cfg, tmp_path):
        from core.config import load_config

        assert load_config(cfg, str(tmp_path / "nope.json")) is False
        assert os.listdir(tmp_path) == []

    def test_valid_file_is_never_quarantined(self, cfg, tmp_path):
        from core.config import load_config, save_config

        path = str(tmp_path / "config.json")
        save_config(cfg, path)
        assert load_config(cfg, path) is True
        assert glob.glob(str(tmp_path / "*.corrupt-*")) == []
