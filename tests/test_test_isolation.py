"""The suite must never touch files in the directory pytest was launched from."""

import os


def test_each_test_runs_in_an_empty_temp_dir(tmp_path):
    assert os.path.realpath(os.getcwd()) == os.path.realpath(str(tmp_path))
    assert os.listdir(os.getcwd()) == []


def test_default_config_path_cannot_reach_a_real_config(tmp_path):
    """save_config()'s default 'config.json' resolves inside the temp dir."""
    from unittest.mock import patch

    from core.config import Config, save_config

    with patch("core.config._get_screen_size", return_value=(1920, 1080)):
        cfg = Config()
    assert save_config(cfg) is True
    assert os.path.dirname(os.path.realpath("config.json")) == os.path.realpath(str(tmp_path))
