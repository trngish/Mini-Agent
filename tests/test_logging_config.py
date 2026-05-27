import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

from mini_agent.utils.logging_config import AgentLoggerAdapter, AgentLoggingConfig, StructuredFormatter, get_logger


class TestAgentLoggingConfig:
    def test_singleton(self):
        AgentLoggingConfig._instance = None
        instance1 = AgentLoggingConfig.get_instance()
        instance2 = AgentLoggingConfig.get_instance()
        assert instance1 is instance2
        AgentLoggingConfig._instance = None

    def test_get_logger(self):
        AgentLoggingConfig._instance = None
        config = AgentLoggingConfig.get_instance()
        with patch.object(logging, "FileHandler", return_value=MagicMock(level=logging.DEBUG)):
            logger = config.get_logger("test_module")
        assert isinstance(logger, AgentLoggerAdapter)
        AgentLoggingConfig._instance = None

    def test_get_agent_logger(self):
        AgentLoggingConfig._instance = None
        config = AgentLoggingConfig.get_instance()
        with patch.object(logging, "FileHandler", return_value=MagicMock(level=logging.DEBUG)):
            logger = config.get_agent_logger()
        assert isinstance(logger, AgentLoggerAdapter)
        AgentLoggingConfig._instance = None

    def test_get_llm_logger(self):
        AgentLoggingConfig._instance = None
        config = AgentLoggingConfig.get_instance()
        with patch.object(logging, "FileHandler", return_value=MagicMock(level=logging.DEBUG)):
            logger = config.get_llm_logger()
        assert isinstance(logger, AgentLoggerAdapter)
        AgentLoggingConfig._instance = None

    def test_get_tool_logger(self):
        AgentLoggingConfig._instance = None
        config = AgentLoggingConfig.get_instance()
        with patch.object(logging, "FileHandler", return_value=MagicMock(level=logging.DEBUG)):
            logger = config.get_tool_logger()
        assert isinstance(logger, AgentLoggerAdapter)
        AgentLoggingConfig._instance = None

    def test_configure_idempotent(self):
        AgentLoggingConfig._instance = None
        config = AgentLoggingConfig.get_instance()
        with patch.object(logging, "FileHandler", return_value=MagicMock(level=logging.DEBUG)):
            config.configure()
        assert config._configured is True
        config.configure()
        AgentLoggingConfig._instance = None

    def test_get_logger_auto_configures(self):
        AgentLoggingConfig._instance = None
        config = AgentLoggingConfig.get_instance()
        assert config._configured is False
        with patch.object(logging, "FileHandler", return_value=MagicMock(level=logging.DEBUG)):
            config.get_logger("auto_test")
        assert config._configured is True
        AgentLoggingConfig._instance = None

    def test_log_dir_created(self):
        AgentLoggingConfig._instance = None
        config = AgentLoggingConfig.get_instance()
        assert config.log_dir == Path.home() / ".mini-agent" / "log"
        AgentLoggingConfig._instance = None


class TestAgentLoggerAdapter:
    def test_process_with_extra(self):
        logger = logging.getLogger("test_adapter_extra")
        adapter = AgentLoggerAdapter(logger, {"tool": "bash", "step": 1})
        msg, kwargs = adapter.process("hello", {})
        assert "[tool=bash | step=1]" in msg
        assert "hello" in msg

    def test_process_without_extra(self):
        logger = logging.getLogger("test_adapter_no_extra")
        adapter = AgentLoggerAdapter(logger, {})
        msg, kwargs = adapter.process("hello", {})
        assert msg == "hello"


class TestStructuredFormatter:
    def test_format_with_timestamp_and_level(self):
        formatter = StructuredFormatter(include_timestamp=True, include_level=True)
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="test message",
            args=(),
            exc_info=None,
        )
        result = formatter.format(record)
        assert "INFO" in result
        assert "test message" in result

    def test_format_without_timestamp(self):
        formatter = StructuredFormatter(include_timestamp=False, include_level=True)
        record = logging.LogRecord(
            name="test",
            level=logging.WARNING,
            pathname="",
            lineno=0,
            msg="warn msg",
            args=(),
            exc_info=None,
        )
        result = formatter.format(record)
        assert "WARNING" in result
        assert "warn msg" in result

    def test_format_without_level(self):
        formatter = StructuredFormatter(include_timestamp=True, include_level=False)
        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="",
            lineno=0,
            msg="error msg",
            args=(),
            exc_info=None,
        )
        result = formatter.format(record)
        assert "ERROR" not in result
        assert "error msg" in result


class TestGetLogger:
    def test_returns_adapter(self):
        AgentLoggingConfig._instance = None
        with patch.object(logging, "FileHandler", return_value=MagicMock(level=logging.DEBUG)):
            logger = get_logger("test_convenience")
        assert isinstance(logger, AgentLoggerAdapter)
        AgentLoggingConfig._instance = None
