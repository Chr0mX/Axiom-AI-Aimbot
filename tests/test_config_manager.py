# tests/test_config_manager.py
"""
ConfigManager 模組單元測試

測試範圍：
1. ConfigManager 初始化
2. save_config / load_config 參數配置存取
3. get_config_list 列表取得
4. delete_config / rename_config 檔案操作
5. export_config / import_config 匯入匯出
"""

import json
import os
import tempfile
import shutil
from unittest.mock import patch

import pytest


def _make_config():
    """建立一個 mock Config"""
    with patch("core.config._get_screen_size", return_value=(1920, 1080)):
        from core.config import Config
        return Config()


class TestConfigManagerInit:
    """測試 ConfigManager 初始化"""

    def test_init_creates_directory(self):
        from core.config_manager import ConfigManager
        with tempfile.TemporaryDirectory() as tmpdir:
            configs_dir = os.path.join(tmpdir, "test_configs")
            cm = ConfigManager(configs_dir)
            assert os.path.exists(configs_dir)

    def test_init_existing_directory(self):
        from core.config_manager import ConfigManager
        with tempfile.TemporaryDirectory() as tmpdir:
            cm = ConfigManager(tmpdir)
            assert os.path.exists(tmpdir)


class TestConfigManagerSaveLoad:
    """測試參數配置保存和載入"""

    def test_save_config_creates_file(self):
        from core.config_manager import ConfigManager
        with tempfile.TemporaryDirectory() as tmpdir:
            cm = ConfigManager(tmpdir)
            c = _make_config()
            result = cm.save_config(c, "test_profile")
            assert result is True
            assert os.path.exists(os.path.join(tmpdir, "test_profile.json"))

    def test_save_config_file_content(self):
        from core.config_manager import ConfigManager
        with tempfile.TemporaryDirectory() as tmpdir:
            cm = ConfigManager(tmpdir)
            c = _make_config()
            c.fov_size = 777
            c.status_panel_show_mouse_click = False
            c.ndi_source_name = "distoav-main"
            cm.save_config(c, "test_profile")
            filepath = os.path.join(tmpdir, "test_profile.json")
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            assert data['name'] == "test_profile"
            assert 'config' in data
            assert data['config']['fov_size'] == 777
            assert data['config']['status_panel_show_mouse_click'] is False
            assert data['config']['ndi_source_name'] == "distoav-main"

    def test_load_config_restores_values(self):
        from core.config_manager import ConfigManager
        with tempfile.TemporaryDirectory() as tmpdir:
            cm = ConfigManager(tmpdir)
            c = _make_config()
            c.fov_size = 888
            c.pid_kp_x = 0.5
            cm.save_config(c, "load_test")

            c2 = _make_config()
            result = cm.load_config(c2, "load_test")
            assert result is True
            assert c2.fov_size == 888
            assert c2.pid_kp_x == 0.5

    def test_load_config_nonexistent(self):
        from core.config_manager import ConfigManager
        with tempfile.TemporaryDirectory() as tmpdir:
            cm = ConfigManager(tmpdir)
            c = _make_config()
            result = cm.load_config(c, "nonexistent_profile")
            assert result is False


class TestConfigManagerList:
    """測試配置列表"""

    def test_empty_list(self):
        from core.config_manager import ConfigManager
        with tempfile.TemporaryDirectory() as tmpdir:
            cm = ConfigManager(tmpdir)
            assert cm.get_config_list() == []

    def test_list_after_save(self):
        from core.config_manager import ConfigManager
        with tempfile.TemporaryDirectory() as tmpdir:
            cm = ConfigManager(tmpdir)
            c = _make_config()
            cm.save_config(c, "profile_a")
            cm.save_config(c, "profile_b")
            configs = cm.get_config_list()
            assert "profile_a" in configs
            assert "profile_b" in configs

    def test_list_sorted(self):
        from core.config_manager import ConfigManager
        with tempfile.TemporaryDirectory() as tmpdir:
            cm = ConfigManager(tmpdir)
            c = _make_config()
            cm.save_config(c, "charlie")
            cm.save_config(c, "alpha")
            cm.save_config(c, "bravo")
            configs = cm.get_config_list()
            assert configs == ["alpha", "bravo", "charlie"]

    def test_list_ignores_non_json(self):
        from core.config_manager import ConfigManager
        with tempfile.TemporaryDirectory() as tmpdir:
            cm = ConfigManager(tmpdir)
            # 建立一個非 JSON 檔案
            with open(os.path.join(tmpdir, "notes.txt"), 'w') as f:
                f.write("not a config")
            assert cm.get_config_list() == []


class TestConfigManagerDelete:
    """測試配置刪除"""

    def test_delete_existing(self):
        from core.config_manager import ConfigManager
        with tempfile.TemporaryDirectory() as tmpdir:
            cm = ConfigManager(tmpdir)
            c = _make_config()
            cm.save_config(c, "to_delete")
            result = cm.delete_config("to_delete")
            assert result is True
            assert not os.path.exists(os.path.join(tmpdir, "to_delete.json"))

    def test_delete_nonexistent(self):
        from core.config_manager import ConfigManager
        with tempfile.TemporaryDirectory() as tmpdir:
            cm = ConfigManager(tmpdir)
            result = cm.delete_config("nonexistent")
            assert result is False


class TestConfigManagerRename:
    """測試配置重命名"""

    def test_rename_success(self):
        from core.config_manager import ConfigManager
        with tempfile.TemporaryDirectory() as tmpdir:
            cm = ConfigManager(tmpdir)
            c = _make_config()
            cm.save_config(c, "old_name")
            result = cm.rename_config("old_name", "new_name")
            assert result is True
            assert not os.path.exists(os.path.join(tmpdir, "old_name.json"))
            assert os.path.exists(os.path.join(tmpdir, "new_name.json"))
            # 檢查內部 name 欄位也更新了
            with open(os.path.join(tmpdir, "new_name.json"), 'r', encoding='utf-8') as f:
                data = json.load(f)
            assert data['name'] == "new_name"

    def test_rename_target_exists_fails(self):
        from core.config_manager import ConfigManager
        with tempfile.TemporaryDirectory() as tmpdir:
            cm = ConfigManager(tmpdir)
            c = _make_config()
            cm.save_config(c, "name_a")
            cm.save_config(c, "name_b")
            result = cm.rename_config("name_a", "name_b")
            assert result is False

    def test_rename_source_missing_fails(self):
        from core.config_manager import ConfigManager
        with tempfile.TemporaryDirectory() as tmpdir:
            cm = ConfigManager(tmpdir)
            result = cm.rename_config("nonexistent", "new_name")
            assert result is False


class TestConfigManagerExportImport:
    """測試配置匯出和匯入"""

    def test_export_success(self):
        from core.config_manager import ConfigManager
        with tempfile.TemporaryDirectory() as tmpdir:
            cm = ConfigManager(tmpdir)
            c = _make_config()
            cm.save_config(c, "to_export")
            export_path = os.path.join(tmpdir, "exported.json")
            result = cm.export_config("to_export", export_path)
            assert result is True
            assert os.path.exists(export_path)

    def test_export_nonexistent(self):
        from core.config_manager import ConfigManager
        with tempfile.TemporaryDirectory() as tmpdir:
            cm = ConfigManager(tmpdir)
            result = cm.export_config("nonexistent", os.path.join(tmpdir, "out.json"))
            assert result is False

    def test_import_success(self):
        from core.config_manager import ConfigManager
        with tempfile.TemporaryDirectory() as tmpdir:
            cm = ConfigManager(tmpdir)
            c = _make_config()
            cm.save_config(c, "original")
            
            # 匯出
            export_path = os.path.join(tmpdir, "exported.json")
            cm.export_config("original", export_path)
            
            # 刪除原始
            cm.delete_config("original")
            
            # 匯入
            result = cm.import_config(export_path)
            assert result is not None
            assert result in cm.get_config_list()

    def test_import_renames_duplicate(self):
        from core.config_manager import ConfigManager
        with tempfile.TemporaryDirectory() as tmpdir:
            cm = ConfigManager(tmpdir)
            c = _make_config()
            cm.save_config(c, "my_config")
            
            export_path = os.path.join(tmpdir, "exported.json")
            cm.export_config("my_config", export_path)
            
            # 再次匯入（名稱已存在）
            result = cm.import_config(export_path)
            assert result is not None
            assert result != "my_config"  # 應該被重命名
            assert result in cm.get_config_list()

    def test_import_nonexistent_file(self):
        from core.config_manager import ConfigManager
        with tempfile.TemporaryDirectory() as tmpdir:
            cm = ConfigManager(tmpdir)
            result = cm.import_config("/nonexistent/file.json")
            assert result is None
