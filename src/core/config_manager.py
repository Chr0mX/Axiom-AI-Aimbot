# config_manager.py
"""參數配置管理模組 - 純業務邏輯，無 GUI 依賴"""

from __future__ import annotations

import copy
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

# preview_config_changes() display helpers ─────────────────────────────────
# Purely cosmetic: groups a handful of attrs that commonly change together
# (e.g. all 6 PID gains) under one friendly label instead of listing every
# attr name individually. Order matters — more specific prefixes are
# checked first so e.g. 'aim_y_reduce_*' doesn't fall through to a generic
# 'aim_' bucket that doesn't exist here. Missing an attr from this table
# isn't a correctness issue, just a slightly less compact summary line —
# it falls back to its own prettified name.
_PRESET_DIFF_GROUPS: List[tuple] = [
    ('humanization.', 'Humanization'),
    ('pid_', 'PID'),
    ('aim_y_reduce_', 'Y-Recoil Suppression'),
    ('aim_y_vel_', 'Y-Recoil Suppression'),
    ('makcu_', 'MAKCU'),
    ('uvc_', 'UVC Capture'),
    ('ndi_', 'NDI Capture'),
    ('udp_', 'UDP Capture'),
    ('web_esp_', 'Web ESP'),
    ('hud_', 'HUD Weapon Detection'),
    ('xbox_', 'Xbox Controller'),
    ('arduino_', 'Arduino'),
    ('ddxoft_', 'ddxoft'),
    ('detect_semantic_', 'Semantic Filter'),
    ('target_priority_', 'Target Priority'),
    ('sticky_lock_', 'Sticky Lock'),
    ('fov_', 'FOV'),
    ('crosshair_color_', 'Crosshair Color'),
]


def _flatten_preset_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Expand the one nested value _get_config_data() produces
    ('humanization', a dataclasses.asdict() block) into pseudo dotted keys
    so its sub-fields diff individually like every other (already-flat)
    attr, instead of the whole block always reporting as one opaque change."""
    flat: Dict[str, Any] = {}
    for k, v in data.items():
        if k == 'humanization' and isinstance(v, dict):
            for hk, hv in v.items():
                flat[f'humanization.{hk}'] = hv
        else:
            flat[k] = v
    return flat


def _describe_changed_attrs(changed_attrs: List[str]) -> List[str]:
    """Turn a list of changed flat/pseudo-dotted attr names into a short,
    human-readable summary list — grouping attrs that share a known prefix
    (see _PRESET_DIFF_GROUPS) into one "N values in <group>" entry, and
    showing anything else by its own prettified name."""
    groups: Dict[str, List[str]] = {}
    ungrouped: List[str] = []
    for attr in changed_attrs:
        label = None
        for prefix, group_label in _PRESET_DIFF_GROUPS:
            if attr.startswith(prefix):
                label = group_label
                break
        if label:
            groups.setdefault(label, []).append(attr)
        else:
            ungrouped.append(attr)

    parts = []
    for label, attrs in groups.items():
        if len(attrs) == 1:
            parts.append(f"{label} ({attrs[0].split('.')[-1].replace('_', ' ')})")
        else:
            parts.append(f"{len(attrs)} values in {label}")
    for attr in ungrouped:
        parts.append(attr.split('.')[-1].replace('_', ' '))
    return parts


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

    def preview_config_changes(self, config_instance: Config, config_name: str) -> Optional[List[str]]:
        """Dry-run load_config(): report what applying `config_name` would
        actually change on `config_instance`, without mutating it — used to
        show a pre-load diff summary instead of silently overwriting.

        Simulates the load on a deep copy via the exact same
        Config.from_dict() path load_config() itself uses (so the preview
        can't drift from real load behavior), then diffs the two instances'
        _get_config_data() output — the same flat, attr-keyed dict
        save_config()/load_config() already read and write, so this needs
        no new preset format of its own.

        Returns a list of short, human-readable change descriptions (empty
        if the preset is identical to the current config), or None if the
        preset file couldn't be read — the same failure mode load_config()
        itself would hit, so callers should treat it the same way (e.g.
        fall through to calling load_config() and let its own error
        handling report the failure) rather than showing a bogus "no
        changes" summary for a preset that was never actually read.
        """
        config_name = _sanitize_config_name(config_name)
        if not config_name:
            return None
        config_path = os.path.join(self.configs_dir, f"{config_name}.json")
        if not os.path.exists(config_path):
            return None

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                raw = json.load(f)
            preset_data = raw.get('config', raw)
        except (OSError, json.JSONDecodeError) as e:
            logger.error("Failed to read preset '%s' for preview: %s", config_name, e)
            return None

        simulated = copy.deepcopy(config_instance)
        simulated.from_dict(preset_data)

        current_flat = _flatten_preset_data(self._get_config_data(config_instance))
        simulated_flat = _flatten_preset_data(self._get_config_data(simulated))

        _unset = object()
        changed_attrs = [
            attr for attr, new_val in simulated_flat.items()
            if current_flat.get(attr, _unset) != new_val
        ]
        return _describe_changed_attrs(changed_attrs)

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
