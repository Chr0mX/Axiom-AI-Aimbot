# config_manager.py
"""參數配置管理模組 - 純業務邏輯，無 GUI 依賴"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import re
import shutil
from datetime import datetime
from typing import List, Optional, Dict, Any, TYPE_CHECKING

from .config import _FIELD_MAP

logger = logging.getLogger(__name__)

# Bundled built-in presets shipped with the app (seeded into the user config dir).
_BUILTIN_PRESETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'presets')

_INVALID_NAME_CHARS = re.compile(r'[\\/:*?"<>|]')


def _sanitize_config_name(name: str) -> str:
    """Strip path separators and other filesystem-unsafe characters from a
    preset name before it's interpolated into a file path.

    Names reach here from free-text GUI dialogs and from the 'name' field of
    an imported JSON file — both untrusted. Without this, a name like
    "../../whatever" would escape configs_dir via os.path.join(). basename()
    strips any directory components; the regex then strips characters that
    are invalid in Windows filenames (or which qfluentwidgets dialogs allow
    but the filesystem doesn't).
    """
    name = os.path.basename(str(name or '')).strip()
    name = _INVALID_NAME_CHARS.sub('_', name)
    return name.strip('. ')

if TYPE_CHECKING:
    from .config import Config


class ConfigManager:
    """參數配置管理器
    
    處理參數配置檔案的保存、載入、刪除、重命名、匯入匯出等操作。
    配置檔案以 JSON 格式儲存在指定目錄中。
    
    Attributes:
        configs_dir: 參數配置儲存目錄路徑
    """
    
    def __init__(self, configs_dir: str = "config") -> None:
        self.configs_dir = configs_dir
        self.ensure_configs_directory()
        
    def ensure_configs_directory(self) -> None:
        """確保參數配置目錄存在"""
        if not os.path.exists(self.configs_dir):
            os.makedirs(self.configs_dir)
        self._seed_builtin_presets()

    def _seed_builtin_presets(self) -> None:
        """Copy any bundled built-in presets into the user config dir.

        Only seeds a preset whose target file does not already exist, so user
        edits are never clobbered. A deleted built-in re-appears on next launch.
        """
        if not os.path.isdir(_BUILTIN_PRESETS_DIR):
            return
        for fn in os.listdir(_BUILTIN_PRESETS_DIR):
            if not fn.endswith('.json'):
                continue
            dst = os.path.join(self.configs_dir, fn)
            if os.path.exists(dst):
                continue
            try:
                shutil.copy2(os.path.join(_BUILTIN_PRESETS_DIR, fn), dst)
                logger.info("Seeded built-in preset: %s", fn[:-5])
            except OSError as e:
                logger.warning("Failed to seed built-in preset '%s': %s", fn, e)
            
    def get_config_list(self) -> List[str]:
        """獲取所有參數配置列表"""
        if not os.path.exists(self.configs_dir):
            return []
        
        configs = []
        for file in os.listdir(self.configs_dir):
            if file.endswith('.json'):
                config_name = file[:-5]  # 移除.json後綴
                configs.append(config_name)
        return sorted(configs)
    
    def save_config(self, config_instance: Config, config_name: str) -> bool:
        """保存當前配置為參數配置"""
        config_name = _sanitize_config_name(config_name)
        if not config_name:
            return False
        config_path = os.path.join(self.configs_dir, f"{config_name}.json")

        # 創建參數配置數據
        config_data = {
            'name': config_name,
            'created_time': datetime.now().isoformat(),
            'description': f"參數配置 - {config_name}",
            'config': self._get_config_data(config_instance)
        }
        
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, ensure_ascii=False, indent=2)
            return True
        except OSError as e:
            logger.error("Failed to save preset '%s': %s", config_name, e)
            return False
    
    def _get_config_data(self, config_instance: Config) -> Dict[str, Any]:
        """從配置實例獲取配置數據

        Derived from `_FIELD_MAP` — the same single source of truth that
        `Config.to_dict()`/`from_dict()` use for config.json — rather than a
        separately hand-maintained key list, so presets can no longer drift
        out of sync with newly added Config fields. Output stays a flat
        {attr_name: value} dict (the format existing preset files already
        use on disk); `ConfigManager.load_config()` restores it via
        `Config.from_dict()`, which reads flat keys as a fallback and simply
        skips any key that is absent, so older presets with fewer keys still
        load fine without clobbering current values.
        """
        data: Dict[str, Any] = {
            attr: getattr(config_instance, attr)
            for attr in _FIELD_MAP
            if hasattr(config_instance, attr)
        }

        # A couple of fields are intentionally excluded from _FIELD_MAP and
        # specially handled by Config.from_dict()/to_dict() instead (see
        # config.py) — include them here too, in the same shape from_dict()
        # expects, so presets round-trip them as well.
        data['crosshair_color_r'] = getattr(config_instance, 'crosshair_color_r', 255)
        data['crosshair_color_g'] = getattr(config_instance, 'crosshair_color_g', 255)
        data['crosshair_color_b'] = getattr(config_instance, 'crosshair_color_b', 255)
        if hasattr(config_instance, 'humanization'):
            data['humanization'] = dataclasses.asdict(config_instance.humanization)

        # model_input_size is auto-detected at runtime and deliberately not
        # in _FIELD_MAP, but was part of the old hand-picked preset list —
        # keep saving it for backward-compat visibility (from_dict() ignores
        # it since it's not in _FIELD_MAP, so it's informational only).
        if hasattr(config_instance, 'model_input_size'):
            data['model_input_size'] = config_instance.model_input_size

        return data
    
    def load_config(self, config_instance: Config, config_name: str) -> bool:
        """載入參數配置"""
        config_name = _sanitize_config_name(config_name)
        if not config_name:
            return False
        config_path = os.path.join(self.configs_dir, f"{config_name}.json")

        if not os.path.exists(config_path):
            return False
            
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                raw = json.load(f)

            # Preset files wrap settings under a 'config' key; support both wrapped and flat.
            config_data = raw.get('config', raw)
            config_instance.from_dict(config_data)
            return True
        except (OSError, json.JSONDecodeError) as e:
            logger.error("Failed to load preset '%s': %s", config_name, e)
            return False
    
    def delete_config(self, config_name: str) -> bool:
        """刪除參數配置"""
        config_name = _sanitize_config_name(config_name)
        if not config_name:
            return False
        config_path = os.path.join(self.configs_dir, f"{config_name}.json")

        if os.path.exists(config_path):
            try:
                os.remove(config_path)
                return True
            except OSError as e:
                logger.error("Failed to delete preset '%s': %s", config_name, e)
                return False
        return False
    
    def rename_config(self, old_name: str, new_name: str) -> bool:
        """重命名參數配置"""
        old_name = _sanitize_config_name(old_name)
        new_name = _sanitize_config_name(new_name)
        if not old_name or not new_name:
            return False
        old_path = os.path.join(self.configs_dir, f"{old_name}.json")
        new_path = os.path.join(self.configs_dir, f"{new_name}.json")

        if os.path.exists(old_path) and not os.path.exists(new_path):
            try:
                # 讀取舊文件並更新名稱
                with open(old_path, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                config_data['name'] = new_name
                
                # 寫入新文件
                with open(new_path, 'w', encoding='utf-8') as f:
                    json.dump(config_data, f, ensure_ascii=False, indent=2)
                
                # 刪除舊文件
                os.remove(old_path)
                return True
            except (OSError, json.JSONDecodeError) as e:
                logger.error("Failed to rename preset '%s' to '%s': %s", old_name, new_name, e)
                return False
        return False
    
    def export_config(self, config_name: str, export_path: str) -> bool:
        """匯出參數配置"""
        config_name = _sanitize_config_name(config_name)
        if not config_name:
            return False
        config_path = os.path.join(self.configs_dir, f"{config_name}.json")

        if os.path.exists(config_path):
            try:
                shutil.copy2(config_path, export_path)
                return True
            except OSError as e:
                logger.error("Failed to export preset '%s': %s", config_name, e)
                return False
        return False
    
    def import_config(self, import_path: str) -> Optional[str]:
        """
        匯入參數配置
        
        Returns:
            成功時返回參數名稱，失敗時返回 None
        """
        if not os.path.exists(import_path):
            return None
            
        try:
            # 讀取匯入的配置
            with open(import_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)

            # A JSON file whose top level isn't an object (a bare list, or a
            # string) is valid JSON and so survives json.load — but every
            # access below assumes a dict, and the AttributeError from
            # .get() would escape the OSError/JSONDecodeError handler and
            # propagate out of what is documented to return None on failure.
            if not isinstance(config_data, dict):
                logger.error(
                    "Preset import from '%s' failed: top-level JSON is %s, expected an object",
                    import_path, type(config_data).__name__,
                )
                return None
            
            # 獲取配置名稱 (untrusted — comes from the imported file's own content)
            config_name = _sanitize_config_name(config_data.get('name', 'imported_config'))
            if not config_name:
                config_name = 'imported_config'

            # 確保名稱唯一
            original_name = config_name
            counter = 1
            while os.path.exists(os.path.join(self.configs_dir, f"{config_name}.json")):
                config_name = f"{original_name}_{counter}"
                counter += 1
            
            # 更新名稱並保存
            config_data['name'] = config_name
            config_path = os.path.join(self.configs_dir, f"{config_name}.json")
            
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, ensure_ascii=False, indent=2)
            
            return config_name
        except (OSError, json.JSONDecodeError) as e:
            logger.error("Failed to import preset from '%s': %s", import_path, e)
            return None 
