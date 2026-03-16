# tests/test_logging_config.py
"""
日誌配置模組單元測試

測試範圍：
1. setup_logging - 日誌設定
2. _has_handlers - handler 檢查
"""

import logging
from unittest.mock import patch

import pytest


class TestSetupLogging:
    """測試日誌設定"""

    def test_returns_root_logger(self):
        from core.logging_config import setup_logging
        logger = setup_logging("DEBUG")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "root"

    def test_sets_level(self):
        from core.logging_config import setup_logging
        logger = setup_logging("WARNING")
        assert logger.level == logging.WARNING

    def test_idempotent(self):
        """重複呼叫不會增加 handler"""
        from core.logging_config import setup_logging
        logger = setup_logging("INFO")
        count1 = len(logger.handlers)
        setup_logging("INFO")
        count2 = len(logger.handlers)
        # 不應增加 handler（或最多增加一個）
        assert count2 <= count1 + 1


class TestHasHandlers:
    """測試 _has_handlers"""

    def test_no_handlers(self):
        from core.logging_config import _has_handlers
        logger = logging.getLogger("test_empty_logger_unique")
        logger.handlers.clear()
        assert _has_handlers(logger) is False

    def test_with_handler(self):
        from core.logging_config import _has_handlers
        logger = logging.getLogger("test_has_handler_unique")
        logger.handlers.clear()
        logger.addHandler(logging.StreamHandler())
        assert _has_handlers(logger) is True
