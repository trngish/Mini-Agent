"""Tool execution utilities for agent.

Provides reusable tool execution helpers including:
- Tool timeout calculators
- Result compressors
- Execution decorators
"""

import asyncio
from typing import Any

from ..tools.base import Tool, ToolResult
from ..schema import ToolCall

# Default timeouts by tool category (seconds)
DEFAULT_TOOL_TIMEOUTS = {
    "read_file": 10,
    "multi_read": 30,
    "write_file": 20,
    "edit_file": 20,
    "grep": 15,
    "multi_grep": 30,
    "find": 10,
    "tree": 10,
    "bash": 60,
    "multi_bash": 120,
    "git": 30,
    "delete_file": 10,
    "move_file": 10,
    "copy_file": 10,
    "workspace_context": 30,
    "deep_context": 60,
}

# MCP tools typically have longer timeouts
MCP_DEFAULT_TIMEOUT = 120


def get_tool_timeout(tool_name: str, default_timeout: float = 60.0) -> float:
    """Get timeout for a tool based on its name.

    Args:
        tool_name: Name of the tool
        default_timeout: Default timeout if not found in categories

    Returns:
        Timeout in seconds
    """
    # Check exact match first
    if tool_name in DEFAULT_TOOL_TIMEOUTS:
        return DEFAULT_TOOL_TIMEOUTS[tool_name]

    # Check prefix patterns (e.g., mcp_*)
    for category, timeout in DEFAULT_TOOL_TIMEOUTS.items():
        if category.endswith("*") and tool_name.startswith(category[:-1]):
            return timeout

    # MCP tools default to longer timeout
    if tool_name.startswith("mcp_"):
        return MCP_DEFAULT_TIMEOUT

    return default_timeout


def compress_tool_result(result: ToolResult, max_chars: int = 5000) -> ToolResult:
    """Compress tool result to reduce token usage.

    Args:
        result: Original tool result
        max_chars: Maximum characters to keep

    Returns:
        Compressed tool result (or original if small enough)
    """
    if len(result.content) <= max_chars:
        return result

    # Return truncated result
    return ToolResult(
        success=result.success,
        content=f"{result.content[:max_chars]}... [truncated {len(result.content) - max_chars} chars]",
        error=result.error,
    )


def is_transient_error(error: str) -> bool:
    """Check if an error is transient (worth retrying).

    Args:
        error: Error message

    Returns:
        True if error is transient
    """
    transient_patterns = [
        "timeout",
        "connection",
        "temporary",
        "unavailable",
        "rate limit",
        "too many requests",
    ]
    error_lower = error.lower()
    return any(pattern in error_lower for pattern in transient_patterns)


async def execute_with_timeout(
    tool: Tool,
    timeout: float,
    **kwargs
) -> ToolResult:
    """Execute a tool with timeout protection.

    Args:
        tool: Tool to execute
        timeout: Timeout in seconds
        **kwargs: Tool arguments

    Returns:
        Tool result or timeout error
    """
    try:
        async with asyncio.timeout(timeout):
            return await tool.execute(**kwargs)
    except TimeoutError:
        return ToolResult(
            success=False,
            content="",
            error=f"Tool execution timed out after {timeout}s",
        )


def should_compress_result(tool_name: str, result_size: int) -> bool:
    """Determine if result should be compressed for LLM consumption.

    Args:
        tool_name: Name of the tool
        result_size: Size of result in characters

    Returns:
        True if should compress
    """
    # Tools that produce large outputs that LLM doesn't need in full
    large_output_tools = {
        "grep", "multi_grep", "find", "tree", "workspace_context",
        "deep_context", "read_file", "multi_read",
    }

    if tool_name not in large_output_tools:
        return False

    # Compress if result is very large (> 50KB)
    return result_size > 50_000


# Re-export ToolCall for convenience
__all__ = [
    "get_tool_timeout",
    "compress_tool_result",
    "is_transient_error",
    "execute_with_timeout",
    "should_compress_result",
    "DEFAULT_TOOL_TIMEOUTS",
]