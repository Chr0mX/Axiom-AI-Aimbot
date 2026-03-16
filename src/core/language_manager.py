"""Language management module - Provides multi-language support for the GUI"""
from __future__ import annotations

import json
import os
import glob
from typing import Dict, List


class LanguageManager:
    """Language Manager - Handles loading and switching of translated text
    
    Provides acquisition of multi-language text, language switching, and persistent storage of language preferences.
    Automatically reads .json files from the language_data folder as language packs.
    """

    # Default language using filename (without extension)
    DEFAULT_LANGUAGE = "English_English"
    CONFIG_FILE = "config.json"
    LANGUAGE_DIR = "language_data"

    # Legacy code mapping table for migrating old settings
    LEGACY_MAPPING = {
        "zh_tw": "Chinese_中文",
        "en": "English_English", 
        "de": "German_Deutsch",
        "es": "Spanish_Español",
        "fr": "French_Français",
        "hi": "Hindi_हिन्दी",
        "ja": "Japanese_日本語",
        "ko": "Korean_한국어",
        "pt": "Portuguese_Português",
        "ru": "Russian_Русский"
    }

    def __init__(self) -> None:
        self.translations: Dict[str, Dict[str, str]] = {}
        self.current_language: str = self.DEFAULT_LANGUAGE
        
        # Get absolute path of language_data
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.language_dir_path = os.path.join(base_dir, self.LANGUAGE_DIR)
        
        # Load all language files
        self.load_all_languages()
        
        # Load user settings
        self.load_language_config()

    def load_all_languages(self) -> None:
        """Load all json language files from language_data folder"""
        self.translations.clear()
        
        if not os.path.exists(self.language_dir_path):
            os.makedirs(self.language_dir_path, exist_ok=True)
            return

        json_pattern = os.path.join(self.language_dir_path, "*.json")
        for file_path in glob.glob(json_pattern):
            filename = os.path.basename(file_path)
            lang_name = os.path.splitext(filename)[0]  # Remove .json
            
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        self.translations[lang_name] = data
            except Exception as e:
                print(f"Error loading language file {filename}: {e}")

        # Ensure there is at least a default language (avoiding crashes)
        if self.DEFAULT_LANGUAGE not in self.translations and self.translations:
            # If default language is not loaded but others exist, pick the first one as default
            self.DEFAULT_LANGUAGE = list(self.translations.keys())[0]

    def get_text(self, key: str, default: str = "") -> str:
        """Return translated text for the active language."""
        lang_table = self.translations.get(self.current_language, {})
        return lang_table.get(key, default or key)

    def set_language(self, language_name: str) -> bool:
        """Switch to a different language if available."""
        if language_name in self.translations:
            self.current_language = language_name
            self.save_language_config()
            return True
        return False

    def get_current_language(self) -> str:
        return self.current_language

    def get_available_languages(self) -> List[str]:
        # Return loaded languages (filenames without extension)
        return list(self.translations.keys())

    def save_language_config(self) -> None:
        try:
            # Read existing config.json content
            config_data = {}
            if os.path.exists(self.CONFIG_FILE):
                with open(self.CONFIG_FILE, "r", encoding="utf-8") as handle:
                    config_data = json.load(handle)
            
            # 更新 language 欄位
            config_data["language"] = self.current_language
            
            with open(self.CONFIG_FILE, "w", encoding="utf-8") as handle:
                json.dump(config_data, handle, ensure_ascii=False, indent=2)
        except Exception as exc:  # pragma: no cover
            print(f"Failed to save language config: {exc}")

    def load_language_config(self) -> None:
        try:
            if os.path.exists(self.CONFIG_FILE):
                with open(self.CONFIG_FILE, "r", encoding="utf-8") as handle:
                    config_data = json.load(handle)
                    stored_lang = config_data.get("language", self.DEFAULT_LANGUAGE)
                    
                    # 嘗試遷移舊代碼
                    if stored_lang in self.LEGACY_MAPPING:
                        stored_lang = self.LEGACY_MAPPING[stored_lang]
                    
                    if stored_lang in self.translations:
                        self.current_language = stored_lang
                    else:
                        # 如果找不到設定的語言，退回預設
                        self.current_language = self.DEFAULT_LANGUAGE
        except Exception as exc:  # pragma: no cover
            print(f"Failed to load language config: {exc}")
            self.current_language = self.DEFAULT_LANGUAGE


# 創建全局實例（必須在便捷函數之前）
language_manager = LanguageManager()


def get_text(key: str, default: str = "") -> str:
    """便捷函數：獲取翻譯文字"""
    return language_manager.get_text(key, default)


def set_language(language_code: str) -> bool:
    """便捷函數：設置語言"""
    return language_manager.set_language(language_code)
