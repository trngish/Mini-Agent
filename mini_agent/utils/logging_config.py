"""智能体的结构化日志模块。

提供跨应用程序的一致日志记录，支持多种输出处理器和日志级别。
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


class AgentLoggerAdapter(logging.LoggerAdapter):  # type: ignore[type-arg]
    """带有额外上下文信息的自定义日志适配器。"""

    def process(self, msg: str, kwargs: Any) -> tuple[str, Any]:
        """为日志消息添加额外上下文。"""
        if self.extra:
            context_str = " | ".join(f"{k}={v}" for k, v in self.extra.items())
            msg = f"[{context_str}] {msg}"
        return msg, kwargs


class StructuredFormatter(logging.Formatter):
    """输出结构化日志条目的格式化器。"""

    def __init__(self, include_timestamp: bool = True, include_level: bool = True):
        super().__init__()
        self.include_timestamp = include_timestamp
        self.include_level = include_level

    def format(self, record: logging.LogRecord) -> str:
        """格式化日志记录。"""
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
    """智能体的日志配置。"""

    _instance: AgentLoggingConfig | None = None

    def __init__(self) -> None:
        self.log_dir = Path.home() / ".mini-agent" / "log"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.console_level = logging.INFO
        self.file_level = logging.DEBUG
        self._configured = False

    @classmethod
    def get_instance(cls) -> AgentLoggingConfig:
        """获取单例实例。"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def configure(self) -> None:
        """为整个应用程序配置日志。"""
        if self._configured:
            return

        # 根日志器配置
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)

        # 控制台处理器（INFO 级别）
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.console_level)
        console_handler.setFormatter(StructuredFormatter(include_timestamp=False, include_level=True))
        root_logger.addHandler(console_handler)

        # 文件处理器（DEBUG 级别）
        timestamp = datetime.now().strftime("%Y%m%d")
        log_file = self.log_dir / f"mini-agent_{timestamp}.log"
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(self.file_level)
        file_handler.setFormatter(StructuredFormatter(include_timestamp=True, include_level=True))
        root_logger.addHandler(file_handler)

        self._configured = True

    def get_logger(
        self,
        name: str,
        extra: dict[str, Any] | None = None,
    ) -> AgentLoggerAdapter:
        """获取具有给定名称的日志器。

        Args:
            name: 日志器名称（通常为 __name__）
            extra: 要添加到所有日志消息的额外上下文

        Returns:
            配置好的日志适配器
        """
        if not self._configured:
            self.configure()

        logger = logging.getLogger(name)
        return AgentLoggerAdapter(logger, extra or {})

    def get_agent_logger(self) -> AgentLoggerAdapter:
        """获取智能体模块的日志器。"""
        return self.get_logger("mini_agent.agent")

    def get_llm_logger(self) -> AgentLoggerAdapter:
        """获取 LLM 模块的日志器。"""
        return self.get_logger("mini_agent.llm")

    def get_tool_logger(self) -> AgentLoggerAdapter:
        """获取工具模块的日志器。"""
        return self.get_logger("mini_agent.tools")


# 便捷函数
def get_logger(name: str, **kwargs: Any) -> AgentLoggerAdapter:
    """获取给定模块的日志器。

    用法:
        logger = get_logger(__name__)
        logger.info("Message")
    """
    return AgentLoggingConfig.get_instance().get_logger(name, kwargs)
