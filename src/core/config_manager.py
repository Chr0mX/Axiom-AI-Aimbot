# config_manager.py
"""Config/preset management module — pure business logic, no GUI dependency."""

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

# Bundled built-in presets shipped with the app (seeded into the user preset dir).
_BUILTIN_PRESETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'presets')

# Pre-rename location of the live/user-writable preset directory — this
# module used to default to project-root `config/`, which was a confusing
# name (it sounds like it should hold the real app config, config.json, but
# it never did; it only ever held presets). _migrate_legacy_config_dir()
# does a one-time copy of anything found here into the new `presets/`
# default so nobody's saved presets silently vanish across the rename.
_LEGACY_CONFIGS_DIR = "config"

_INVALID_NAME_CHARS = re.compile(r'[\\/:*?"<>|]')

# A preset is an *aim* preset, not a full config snapshot: every attr whose
# _FIELD_MAP path lives under the `aim.` or `tracking.` JSON prefix, plus
# the `humanization` dataclass block (handled separately below since it
# isn't a flat _FIELD_MAP entry at all — see Config.to_dict()/from_dict()).
# This is the one place that scope is defined; _get_config_data() (what a
# preset file is allowed to contain) and load_config() (what a preset file
# is allowed to apply, even if it contains more — an old-format or
# hand-edited/imported file included) both filter through this same set, so
# the two can't drift apart.
_AIM_PRESET_FIELDS = frozenset(
    attr for attr, path in _FIELD_MAP.items()
    if path.startswith('aim.') or path.startswith('tracking.')
)

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
    ('detect_semantic_', 'Semantic Filter'),
    ('target_priority_', 'Target Priority'),
    ('sticky_lock_', 'Sticky Lock'),
    ('fov_', 'FOV'),
    # These groups are only ever reachable through a *full* Config
    # ConfigManager (aim_only=False) — an aim preset can never contain a
    # non-aim field in the first place, so they were dead weight while this
    # table only served aim presets. Restored now that _describe_changed_attrs()
    # is shared by both scopes.
    ('makcu_', 'MAKCU'),
    ('uvc_', 'UVC Capture'),
    ('ndi_', 'NDI Capture'),
    ('udp_', 'UDP Capture'),
    ('web_esp_', 'Web ESP'),
    ('hud_', 'HUD Detection'),
    ('xbox_', 'Xbox Controller'),
    ('arduino_', 'Arduino'),
    ('ddxoft_', 'ddxoft'),
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


def _filter_to_aim_preset_fields(data: Dict[str, Any]) -> Dict[str, Any]:
    """Keeps only the keys load_config() is allowed to apply — every attr in
    `_AIM_PRESET_FIELDS` plus `'humanization'` — dropping anything else a
    preset file might still contain (an old-format file saved before
    presets were scoped to aim settings, or a hand-edited/imported one).
    This is what makes the aim-only invariant hold no matter how the file
    was produced, not just for freshly-saved ones."""
    return {k: v for k, v in data.items() if k in _AIM_PRESET_FIELDS or k == 'humanization'}


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
    """Config/preset manager.

    Handles save/load/delete/rename/import/export for named settings files —
    either aim-only *presets* or full *config* snapshots, selected by
    `aim_only`. Both scopes share this one class and every method below,
    since the only real difference between them is which fields
    `_get_config_data()`/`load_config()` are willing to touch; everything
    else (file I/O, name sanitizing, diff preview) is identical.

    Attributes:
        configs_dir: filesystem path to the config/preset storage directory
        aim_only: True (the default) = an *aim preset* — scoped to
            `_AIM_PRESET_FIELDS` plus `humanization`, matching the
            project-root `presets/` directory. False = a *full config*
            snapshot — every `_FIELD_MAP` field, matching a separate
            project-root `configs/` directory. The GUI's Configs page shows
            both as two distinct sections against two separate
            ConfigManager instances — see configs_page.py.
    """

    def __init__(self, configs_dir: str = "presets", aim_only: bool = True) -> None:
        self.configs_dir = configs_dir
        self.aim_only = aim_only
        self.ensure_configs_directory()

    def ensure_configs_directory(self) -> None:
        """Ensures the config/preset storage directory exists."""
        if not os.path.exists(self.configs_dir):
            os.makedirs(self.configs_dir)
            # Legacy-dir migration and built-in seeding are both aim-preset
            # concepts specifically — the old config/ directory historically
            # fed into what's now the aim-only presets/ dir, and the only
            # bundled built-in file that ever existed was itself an aim
            # preset. A full-config instance (aim_only=False) is a brand
            # new directory with nothing to migrate from and no bundled
            # example to seed, so both are skipped entirely for it.
            if self.aim_only:
                self._migrate_legacy_config_dir()
        if self.aim_only:
            self._seed_builtin_presets()

    def _migrate_legacy_config_dir(self) -> None:
        """One-time migration for anyone upgrading from before the preset
        directory was renamed from project-root `config/` to `presets/`
        (see _LEGACY_CONFIGS_DIR's own comment for why). Copies any preset
        `*.json` file found there into the new directory — never
        overwrites, never deletes or otherwise touches the old directory,
        so this is safe to run unconditionally and idempotent in spirit
        with _seed_builtin_presets()'s own copy-if-missing behavior.
        """
        if not os.path.isdir(_LEGACY_CONFIGS_DIR):
            return
        for fn in os.listdir(_LEGACY_CONFIGS_DIR):
            if not fn.endswith('.json'):
                continue
            dst = os.path.join(self.configs_dir, fn)
            if os.path.exists(dst):
                continue
            try:
                shutil.copy2(os.path.join(_LEGACY_CONFIGS_DIR, fn), dst)
                logger.info("Migrated preset from legacy config/ dir: %s", fn[:-5])
            except OSError as e:
                logger.warning("Failed to migrate legacy preset '%s': %s", fn, e)

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
        """Returns the list of all saved config/preset names."""
        if not os.path.exists(self.configs_dir):
            return []
        
        configs = []
        for file in os.listdir(self.configs_dir):
            if file.endswith('.json'):
                config_name = file[:-5]  # Strip the .json suffix
                configs.append(config_name)
        return sorted(configs)
    
    def save_config(self, config_instance: Config, config_name: str) -> bool:
        """Saves the current config as a named config/preset file."""
        config_name = _sanitize_config_name(config_name)
        if not config_name:
            return False
        config_path = os.path.join(self.configs_dir, f"{config_name}.json")

        # Build the config/preset data payload
        config_data = {
            'name': config_name,
            'created_time': datetime.now().isoformat(),
            'description': f"{'Preset' if self.aim_only else 'Config'} - {config_name}",
            'config': self._get_config_data(config_instance)
        }
        
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, ensure_ascii=False, indent=2)
            # The live config now *is* this named preset/config — mirrors a
            # typical "Save As" (see load_config()'s matching update, and
            # Config.active_preset_name/active_config_name's docstring).
            if self.aim_only:
                config_instance.active_preset_name = config_name
            else:
                config_instance.active_config_name = config_name
            return True
        except OSError as e:
            logger.error("Failed to save preset '%s': %s", config_name, e)
            return False
    
    def _get_config_data(self, config_instance: Config) -> Dict[str, Any]:
        """Reads the config/preset data from a Config instance.

        Scope depends on `self.aim_only`. When True (an *aim preset*), this
        is deliberately not a full config snapshot — it's scoped to
        `_AIM_PRESET_FIELDS` (every `_FIELD_MAP` attr under the `aim.`/
        `tracking.` JSON prefix) plus the `humanization` dataclass block,
        the same "aim settings, not every setting" boundary
        `config_manager.py`'s module docstring/comments describe. When
        False (a *full config* snapshot), every `_FIELD_MAP` attr is
        included instead — the complete settings surface, matching what
        `config.json` itself persists.

        Either way this is still derived from `_FIELD_MAP` — the same
        single source of truth `Config.to_dict()`/`from_dict()` use for
        config.json — rather than a separately hand-maintained key list, so
        neither scope can drift out of sync with newly added Config fields.
        Output stays a flat {attr_name: value} dict (the format existing
        preset/config files already use on disk); `load_config()` filters
        through this same scope again before applying it (so an
        old-format or hand-edited/imported aim-preset file that still
        carries non-aim keys can't leak them onto the live config), and
        `Config.from_dict()` itself simply skips any key that is absent, so
        older/smaller files still load fine without clobbering unrelated
        current values.
        """
        field_names = _AIM_PRESET_FIELDS if self.aim_only else _FIELD_MAP.keys()
        data: Dict[str, Any] = {
            attr: getattr(config_instance, attr)
            for attr in field_names
            if hasattr(config_instance, attr)
        }

        # humanization is aim-behavior-shaping (post-PID mouse-output
        # shaping — see ai_aiming.py) but isn't a flat _FIELD_MAP entry at
        # all; Config.to_dict()/from_dict() special-case it the same way.
        # Included in both scopes — it's aim-relevant either way, and a full
        # config snapshot should obviously carry it too.
        if hasattr(config_instance, 'humanization'):
            data['humanization'] = dataclasses.asdict(config_instance.humanization)

        return data

    def load_config(self, config_instance: Config, config_name: str) -> bool:
        """Loads a named config/preset file onto a Config instance."""
        config_name = _sanitize_config_name(config_name)
        if not config_name:
            return False
        config_path = os.path.join(self.configs_dir, f"{config_name}.json")

        if not os.path.exists(config_path):
            return False

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                raw = json.load(f)

            # Preset/config files wrap settings under a 'config' key; support both wrapped and flat.
            config_data = raw.get('config', raw)
            if self.aim_only:
                # Enforce the aim-only scope on the way in too, not just on
                # the way out via _get_config_data() — a preset file saved
                # before this scoping existed, or hand-edited/imported,
                # could still carry non-aim keys; this is what keeps
                # loading ANY preset file aim-only regardless of how it was
                # produced. A full-config instance applies everything the
                # file contains, unfiltered — that's the whole point of it.
                config_data = _filter_to_aim_preset_fields(config_data)
            config_instance.from_dict(config_data)
            # Record what's now active for the status panel (Config.
            # active_preset_name/active_config_name). A full-config load
            # (aim_only=False) touches every aim./tracking.* field too, so
            # it can silently override whatever preset previously set
            # them — clear active_preset_name so the status panel doesn't
            # keep crediting a preset whose settings may no longer be in
            # effect. An aim-preset load never touches non-aim fields, so
            # it has no equivalent reason to touch active_config_name.
            if self.aim_only:
                config_instance.active_preset_name = config_name
            else:
                config_instance.active_config_name = config_name
                config_instance.active_preset_name = ""
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
        """Deletes a named config/preset file."""
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
        """Renames a config/preset file."""
        old_name = _sanitize_config_name(old_name)
        new_name = _sanitize_config_name(new_name)
        if not old_name or not new_name:
            return False
        old_path = os.path.join(self.configs_dir, f"{old_name}.json")
        new_path = os.path.join(self.configs_dir, f"{new_name}.json")

        if os.path.exists(old_path) and not os.path.exists(new_path):
            try:
                # Read the old file and update its stored name
                with open(old_path, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                config_data['name'] = new_name

                # Write the new file
                with open(new_path, 'w', encoding='utf-8') as f:
                    json.dump(config_data, f, ensure_ascii=False, indent=2)

                # Remove the old file
                os.remove(old_path)
                return True
            except (OSError, json.JSONDecodeError) as e:
                logger.error("Failed to rename preset '%s' to '%s': %s", old_name, new_name, e)
                return False
        return False
    
    def export_config(self, config_name: str, export_path: str) -> bool:
        """Exports a config/preset file to an external path."""
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
        Imports a config/preset file from an external path.

        Returns:
            The config/preset name on success, or None on failure.
        """
        if not os.path.exists(import_path):
            return None

        try:
            # Read the imported file
            with open(import_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)

            # Derive the name to save under (untrusted — comes from the imported file's own content)
            config_name = _sanitize_config_name(config_data.get('name', 'imported_config'))
            if not config_name:
                config_name = 'imported_config'

            # Ensure the name is unique among existing files
            original_name = config_name
            counter = 1
            while os.path.exists(os.path.join(self.configs_dir, f"{config_name}.json")):
                config_name = f"{original_name}_{counter}"
                counter += 1

            # Update the stored name and save
            config_data['name'] = config_name
            config_path = os.path.join(self.configs_dir, f"{config_name}.json")
            
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, ensure_ascii=False, indent=2)
            
            return config_name
        except (OSError, json.JSONDecodeError) as e:
            logger.error("Failed to import preset from '%s': %s", import_path, e)
            return None 
