"""Structured logging module for the agent.

Provides consistent logging across the application with support for
multiple output handlers and log levels.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


class AgentLoggerAdapter(logging.LoggerAdapter):
    """Custom logger adapter with additional context."""

    def process(self, msg: str, kwargs: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """Add extra context to log messages."""
        if self.extra:
            context_str = " | ".join(f"{k}={v}" for k, v in self.extra.items())
            msg = f"[{context_str}] {msg}"
        return msg, kwargs


class StructuredFormatter(logging.Formatter):
    """Formatter that outputs structured log entries."""

    def __init__(self, include_timestamp: bool = True, include_level: bool = True):
        super().__init__()
        self.include_timestamp = include_timestamp
        self.include_level = include_level

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with structure."""
        parts = []

        if self.include_timestamp:
            dt = datetime.fromtimestamp(record.created)
            parts.append(dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3])

        if self.include_level:
            parts.append(f"[{record.levelname}]")

        parts.append(record.getMessage())

        if record.exc_info:
            parts.append(self.formatException(record.exc_info))

        return " | ".join(parts)


class AgentLoggingConfig:
    """Logging configuration for the agent."""

    _instance: "AgentLoggingConfig | None" = None

    def __init__(self):
        self.log_dir = Path.home() / ".mini-agent" / "log"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.console_level = logging.INFO
        self.file_level = logging.DEBUG
        self._configured = False

    @classmethod
    def get_instance(cls) -> "AgentLoggingConfig":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def configure(self) -> None:
        """Configure logging for the entire application."""
        if self._configured:
            return

        # Root logger configuration
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)

        # Console handler (INFO level)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.console_level)
        console_handler.setFormatter(
            StructuredFormatter(include_timestamp=False, include_level=True)
        )
        root_logger.addHandler(console_handler)

        # File handler (DEBUG level)
        timestamp = datetime.now().strftime("%Y%m%d")
        log_file = self.log_dir / f"mini-agent_{timestamp}.log"
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(self.file_level)
        file_handler.setFormatter(
            StructuredFormatter(include_timestamp=True, include_level=True)
        )
        root_logger.addHandler(file_handler)

        self._configured = True

    def get_logger(
        self,
        name: str,
        extra: dict[str, Any] | None = None,
    ) -> logging.LoggerAdapter:
        """Get a logger with the given name.

        Args:
            name: Logger name (typically __name__)
            extra: Extra context to add to all log messages

        Returns:
            Configured logger adapter
        """
        if not self._configured:
            self.configure()

        logger = logging.getLogger(name)
        return AgentLoggerAdapter(logger, extra or {})

    def get_agent_logger(self) -> logging.LoggerAdapter:
        """Get logger for agent module."""
        return self.get_logger("mini_agent.agent")

    def get_llm_logger(self) -> logging.LoggerAdapter:
        """Get logger for LLM module."""
        return self.get_logger("mini_agent.llm")

    def get_tool_logger(self) -> logging.LoggerAdapter:
        """Get logger for tools module."""
        return self.get_logger("mini_agent.tools")


# Convenience function
def get_logger(name: str, **kwargs: Any) -> logging.LoggerAdapter:
    """Get a logger for the given module.

    Usage:
        logger = get_logger(__name__)
        logger.info("Message")
    """
    return AgentLoggingConfig.get_instance().get_logger(name, kwargs)