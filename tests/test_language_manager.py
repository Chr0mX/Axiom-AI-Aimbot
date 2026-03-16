# tests/test_language_manager.py
"""
語言管理器單元測試

測試範圍：
1. LanguageManager 初始化
2. get_text - 翻譯文字取得
3. set_language - 語言切換
4. get_available_languages / get_current_language
5. load_all_languages - 語言檔載入
6. save_language_config / load_language_config - 語言設定持久化
7. LEGACY_MAPPING - 舊版代碼遷移
"""

import json
import os
import tempfile
from unittest.mock import patch, MagicMock

import pytest


def _make_language_manager(lang_data=None, config_data=None):
    """
    建立 LanguageManager 實例，使用臨時目錄
    lang_data: dict of {filename: {key: value}} 語言檔內容
    config_data: dict，模擬 config.json 內容
    """
    from core.language_manager import LanguageManager

    tmpdir = tempfile.mkdtemp()
    lang_dir = os.path.join(tmpdir, "language_data")
    os.makedirs(lang_dir, exist_ok=True)

    if lang_data:
        for name, data in lang_data.items():
            filepath = os.path.join(lang_dir, f"{name}.json")
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

    config_path = os.path.join(tmpdir, "config.json")
    if config_data:
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, ensure_ascii=False, indent=2)

    # Patch 路徑讓 LanguageManager 使用臨時目錄
    lm = LanguageManager.__new__(LanguageManager)
    lm.translations = {}
    lm.current_language = LanguageManager.DEFAULT_LANGUAGE
    lm.language_dir_path = lang_dir
    lm.CONFIG_FILE = config_path

    lm.load_all_languages()
    lm.load_language_config()

    return lm, tmpdir


# ============================================================
# 1. 初始化測試
# ============================================================

class TestLanguageManagerInit:
    """測試語言管理器初始化"""

    def test_init_with_languages(self):
        lm, tmpdir = _make_language_manager(
            lang_data={
                "English_English": {"hello": "Hello"},
                "Chinese_中文": {"hello": "你好"},
            }
        )
        assert len(lm.translations) == 2
        assert "English_English" in lm.translations
        assert "Chinese_中文" in lm.translations

    def test_init_empty_dir(self):
        lm, tmpdir = _make_language_manager()
        assert len(lm.translations) == 0


# ============================================================
# 2. get_text 測試
# ============================================================

class TestGetText:
    """測試翻譯文字取得"""

    def test_existing_key(self):
        lm, _ = _make_language_manager(
            lang_data={"English_English": {"greeting": "Hello World"}},
        )
        lm.current_language = "English_English"
        assert lm.get_text("greeting") == "Hello World"

    def test_missing_key_returns_key(self):
        lm, _ = _make_language_manager(
            lang_data={"English_English": {"greeting": "Hello"}},
        )
        lm.current_language = "English_English"
        assert lm.get_text("nonexistent_key") == "nonexistent_key"

    def test_missing_key_with_default(self):
        lm, _ = _make_language_manager(
            lang_data={"English_English": {}},
        )
        lm.current_language = "English_English"
        assert lm.get_text("missing", "fallback") == "fallback"

    def test_missing_language_returns_key(self):
        lm, _ = _make_language_manager()
        lm.current_language = "Nonexistent_Language"
        assert lm.get_text("key") == "key"


# ============================================================
# 3. set_language 測試
# ============================================================

class TestSetLanguage:
    """測試語言切換"""

    def test_switch_to_valid_language(self):
        lm, _ = _make_language_manager(
            lang_data={
                "English_English": {"hello": "Hello"},
                "Chinese_中文": {"hello": "你好"},
            }
        )
        result = lm.set_language("Chinese_中文")
        assert result is True
        assert lm.current_language == "Chinese_中文"

    def test_switch_to_invalid_language(self):
        lm, _ = _make_language_manager(
            lang_data={"English_English": {"hello": "Hello"}},
        )
        result = lm.set_language("Nonexistent")
        assert result is False
        assert lm.current_language != "Nonexistent"


# ============================================================
# 4. get_available_languages / get_current_language 測試
# ============================================================

class TestLanguageList:
    """測試語言列表功能"""

    def test_get_available_languages(self):
        lm, _ = _make_language_manager(
            lang_data={
                "English_English": {},
                "Chinese_中文": {},
                "Japanese_日本語": {},
            }
        )
        langs = lm.get_available_languages()
        assert len(langs) == 3
        assert "English_English" in langs

    def test_get_current_language(self):
        lm, _ = _make_language_manager(
            lang_data={"English_English": {}},
        )
        lm.current_language = "English_English"
        assert lm.get_current_language() == "English_English"


# ============================================================
# 5. 語言設定持久化測試
# ============================================================

class TestLanguageConfigPersistence:
    """測試語言設定的存取"""

    def test_save_and_load_language(self):
        lm, tmpdir = _make_language_manager(
            lang_data={
                "English_English": {},
                "Chinese_中文": {},
            }
        )
        lm.set_language("Chinese_中文")
        # save_language_config 已在 set_language 中被呼叫

        # 直接驗證檔案內容
        with open(lm.CONFIG_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        assert data['language'] == "Chinese_中文"

    def test_load_with_legacy_code(self):
        """舊版 language code 應被自動遷移"""
        lm, _ = _make_language_manager(
            lang_data={
                "English_English": {},
                "Chinese_中文": {},
            },
            config_data={"language": "zh_tw"},  # 舊版代碼
        )
        assert lm.current_language == "Chinese_中文"

    def test_load_nonexistent_language_fallback(self):
        """設定中的語言找不到時應回退到預設"""
        lm, _ = _make_language_manager(
            lang_data={"English_English": {}},
            config_data={"language": "Martian_火星文"},
        )
        assert lm.current_language == "English_English"


# ============================================================
# 6. LEGACY_MAPPING 測試
# ============================================================

class TestLegacyMapping:
    """測試舊版代碼遷移對照表"""

    def test_all_legacy_codes_mapped(self):
        from core.language_manager import LanguageManager
        expected = ["zh_tw", "en", "de", "es", "fr", "hi", "ja", "ko", "pt", "ru"]
        for code in expected:
            assert code in LanguageManager.LEGACY_MAPPING

    def test_legacy_to_new_mapping(self):
        from core.language_manager import LanguageManager
        assert LanguageManager.LEGACY_MAPPING["zh_tw"] == "Chinese_中文"
        assert LanguageManager.LEGACY_MAPPING["en"] == "English_English"
        assert LanguageManager.LEGACY_MAPPING["ja"] == "Japanese_日本語"
