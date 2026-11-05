"""core/paths.py: user files live in <Axiom>/user/ and move there once."""
import os

from core import paths


def _write(path, text="x"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def test_user_dir_follows_env_and_is_created(tmp_path, monkeypatch):
    target = tmp_path / "somewhere" / "user"
    monkeypatch.setenv("AXIOM_USER_DIR", str(target))
    assert paths.user_dir() == str(target)
    assert target.is_dir()
    assert paths.user_path("config.json") == str(target / "config.json")


def test_default_user_dir_is_inside_project_root(monkeypatch):
    monkeypatch.delenv("AXIOM_USER_DIR", raising=False)
    monkeypatch.setattr(paths.os, "makedirs", lambda *a, **k: None)
    assert paths.user_dir() == os.path.join(paths.project_root(), "user")
    assert os.path.isdir(os.path.join(paths.project_root(), "src", "core"))


def test_migration_moves_files_and_renames_model_notes(tmp_path):
    root, dest = tmp_path / "root", tmp_path / "root" / "user"
    for name in ("config.json", "state.json", "language.json", "theme_colors.json",
                 "model_info.json", "config.json.corrupt-123"):
        _write(str(root / name), name)
    _write(str(root / "presets" / "Legit.json"), "legit")
    _write(str(root / "configs" / "full.json"), "full")
    _write(str(root / "game.json"), "shipped")

    moved = paths.migrate_root_user_files(str(root), str(dest))

    assert _read(str(dest / "config.json")) == "config.json"
    assert _read(str(dest / "model_notes.json")) == "model_info.json"
    assert _read(str(dest / "config.json.corrupt-123")) == "config.json.corrupt-123"
    assert _read(str(dest / "presets" / "Legit.json")) == "legit"
    assert _read(str(dest / "configs" / "full.json")) == "full"
    for gone in ("config.json", "state.json", "model_info.json", "presets", "configs"):
        assert not (root / gone).exists()
    assert (root / "game.json").exists()  # shipped data stays put
    assert len(moved) == 8


def test_migration_never_overwrites_and_merges_folders(tmp_path):
    root, dest = tmp_path / "root", tmp_path / "root" / "user"
    _write(str(root / "config.json"), "old")
    _write(str(dest / "config.json"), "new")
    _write(str(root / "presets" / "A.json"), "old A")
    _write(str(root / "presets" / "B.json"), "old B")
    _write(str(dest / "presets" / "A.json"), "new A")

    paths.migrate_root_user_files(str(root), str(dest))

    assert _read(str(dest / "config.json")) == "new"
    assert _read(str(root / "config.json")) == "old"  # left alone, not lost
    assert _read(str(dest / "presets" / "A.json")) == "new A"
    assert _read(str(dest / "presets" / "B.json")) == "old B"
    assert _read(str(root / "presets" / "A.json")) == "old A"


def test_migration_is_idempotent(tmp_path):
    root, dest = tmp_path / "root", tmp_path / "root" / "user"
    _write(str(root / "config.json"), "c")
    assert paths.migrate_root_user_files(str(root), str(dest))
    assert paths.migrate_root_user_files(str(root), str(dest)) == []
    assert _read(str(dest / "config.json")) == "c"


def test_save_config_defaults_to_user_dir(tmp_path, monkeypatch):
    from unittest.mock import patch
    target = tmp_path / "u"
    monkeypatch.setenv("AXIOM_USER_DIR", str(target))
    with patch("core.config._get_screen_size", return_value=(1920, 1080)):
        from core.config import Config, save_config
        assert save_config(Config()) is True
    assert (target / "config.json").is_file()
    assert (target / "state.json").is_file()
