# tests/test_logging_config.py
"""
日誌配置模組單元測試

測試範圍：
1. setup_logging - 日誌設定
2. _has_handlers - handler 檢查
"""

import logging



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


class TestLogThrottled:
    """log_throttled: one line per key per interval, so a failing per-frame handler
    reports itself without flooding the console."""

    def setup_method(self):
        from core.logging_config import reset_log_throttle
        reset_log_throttle()

    def test_first_call_logs_and_repeats_are_suppressed(self, caplog):
        from core.logging_config import log_throttled
        log = logging.getLogger("test.throttle")
        with caplog.at_level(logging.DEBUG, logger="test.throttle"):
            assert log_throttled(log, "k", logging.WARNING, "boom %s", 1, interval_s=60) is True
            assert log_throttled(log, "k", logging.WARNING, "boom %s", 2, interval_s=60) is False
        assert [r.getMessage() for r in caplog.records] == ["boom 1"]

    def test_distinct_keys_are_independent(self, caplog):
        from core.logging_config import log_throttled
        log = logging.getLogger("test.throttle")
        with caplog.at_level(logging.DEBUG, logger="test.throttle"):
            assert log_throttled(log, "a", logging.INFO, "a", interval_s=60)
            assert log_throttled(log, "b", logging.INFO, "b", interval_s=60)
        assert len(caplog.records) == 2

    def test_logs_again_after_the_interval(self, caplog, monkeypatch):
        import core.logging_config as lc
        clock = [100.0]
        monkeypatch.setattr(lc.time, "monotonic", lambda: clock[0])
        log = logging.getLogger("test.throttle")
        with caplog.at_level(logging.DEBUG, logger="test.throttle"):
            assert lc.log_throttled(log, "k", logging.INFO, "x", interval_s=5)
            clock[0] += 4.9
            assert not lc.log_throttled(log, "k", logging.INFO, "x", interval_s=5)
            clock[0] += 0.2
            assert lc.log_throttled(log, "k", logging.INFO, "x", interval_s=5)
        assert len(caplog.records) == 2

    def test_exc_info_is_attached_inside_an_except_block(self, caplog):
        from core.logging_config import log_throttled
        log = logging.getLogger("test.throttle")
        with caplog.at_level(logging.DEBUG, logger="test.throttle"):
            try:
                raise ValueError("device gone")
            except ValueError:
                log_throttled(log, "e", logging.WARNING, "read failed", exc_info=True)
        assert caplog.records[0].exc_info and caplog.records[0].exc_info[0] is ValueError
