"""Unified tool execution error handling.

Provides standardized exception types and error handling for tool execution.
"""

from typing import Any


class ToolExecutionError(Exception):
    """Base exception for tool execution failures.

    Attributes:
        tool_name: Name of the tool that failed
        arguments: Arguments passed to the tool
        message: Human-readable error message
        original_exception: Original exception that was raised
        recoverable: Whether this error can be recovered from
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
    """Exception for tool input validation failures."""

    def __init__(self, tool_name: str, arguments: dict[str, Any], message: str):
        super().__init__(
            tool_name=tool_name,
            arguments=arguments,
            message=message,
            recoverable=False,
        )


class ToolPermissionError(ToolExecutionError):
    """Exception for tool permission denials."""

    def __init__(self, tool_name: str, arguments: dict[str, Any], message: str):
        super().__init__(
            tool_name=tool_name,
            arguments=arguments,
            message=message,
            recoverable=False,
        )


class ToolResourceError(ToolExecutionError):
    """Exception for tool resource failures (file not found, network error, etc.)."""

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
    """Classify and wrap an exception into appropriate ToolExecutionError.

    Args:
        tool_name: Name of the tool that failed
        arguments: Arguments passed to the tool
        exception: Original exception

    Returns:
        Classified ToolExecutionError
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

    # Default: generic execution error
    return ToolExecutionError(
        tool_name=tool_name,
        arguments=arguments,
        message=f"Execution failed: {type(exception).__name__}: {exception}",
        original_exception=exception,
    )
