"""统一的工具执行错误处理。

提供标准化的异常类型和错误处理用于工具执行。
"""

from typing import Any


class ToolExecutionError(Exception):
    """工具执行失败的基类异常。

    Attributes:
        tool_name: 失败工具的名称
        arguments: 传递给工具的参数
        message: 人类可读的错误消息
        original_exception: 被抛出的原始异常
        recoverable: 此错误是否可以恢复
    """

    def __init__(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        message: str,
        original_exception: Exception | None = None,
        recoverable: bool = True,
    ):
        self.tool_name = tool_name
        self.arguments = arguments
        self.message = message
        self.original_exception = original_exception
        self.recoverable = recoverable
        super().__init__(self.message)

    def __str__(self) -> str:
        parts = [f"Tool '{self.tool_name}' failed: {self.message}"]
        if self.original_exception:
            parts.append(f"Original: {type(self.original_exception).__name__}: {self.original_exception}")
        return " | ".join(parts)


class ToolValidationError(ToolExecutionError):
    """工具输入验证失败的异常。"""

    def __init__(self, tool_name: str, arguments: dict[str, Any], message: str):
        super().__init__(
            tool_name=tool_name,
            arguments=arguments,
            message=message,
            recoverable=False,
        )


class ToolPermissionError(ToolExecutionError):
    """工具权限拒绝的异常。"""

    def __init__(self, tool_name: str, arguments: dict[str, Any], message: str):
        super().__init__(
            tool_name=tool_name,
            arguments=arguments,
            message=message,
            recoverable=False,
        )


class ToolResourceError(ToolExecutionError):
    """工具资源失败的异常（如文件未找到、网络错误等）。"""

    def __init__(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        message: str,
        original_exception: Exception | None = None,
    ):
        super().__init__(
            tool_name=tool_name,
            arguments=arguments,
            message=message,
            original_exception=original_exception,
            recoverable=True,
        )


def handle_tool_error(
    tool_name: str,
    arguments: dict[str, Any],
    exception: Exception,
) -> ToolExecutionError:
    """将异常分类并包装成适当的ToolExecutionError。

    Args:
        tool_name: 失败工具的名称
        arguments: 传递给工具的参数
        exception: 原始异常

    Returns:
        已分类的ToolExecutionError
    """
    if isinstance(exception, ToolExecutionError):
        return exception

    if isinstance(exception, (FileNotFoundError, OSError)):
        return ToolResourceError(
            tool_name=tool_name,
            arguments=arguments,
            message=str(exception),
            original_exception=exception,
        )

    if isinstance(exception, PermissionError):
        return ToolPermissionError(
            tool_name=tool_name,
            arguments=arguments,
            message=f"Permission denied: {exception}",
        )

    if isinstance(exception, ValueError):
        return ToolValidationError(
            tool_name=tool_name,
            arguments=arguments,
            message=f"Invalid input: {exception}",
        )

    # 默认：通用执行错误
    return ToolExecutionError(
        tool_name=tool_name,
        arguments=arguments,
        message=f"Execution failed: {type(exception).__name__}: {exception}",
        original_exception=exception,
    )
