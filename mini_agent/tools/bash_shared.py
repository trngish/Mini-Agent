"""Shared utilities for bash tool implementations.

This module is kept for backwards compatibility. New code should use
mini_agent.utils.platform_utils.PlatformUtils instead.
"""

# Re-export from new location for backwards compatibility
from ..utils.platform_utils import (
    get_platform_shell_args as _get_platform_shell_args,
    get_subprocess_env as _get_subprocess_env,
    normalize_path_separators,
)


def get_platform_shell_args(platform_mode: str = "auto") -> tuple[str, list[str], str]:
    """Get platform-appropriate shell configuration.
    
    Args:
        platform_mode: "windows", "linux", or "auto" (auto-detect)
        
    Returns:
        Tuple of (shell_executable, shell_args, shell_name)
    """
    return _get_platform_shell_args(platform_mode)


def get_subprocess_env() -> dict[str, str]:
    """Get platform-appropriate environment variables for subprocess.
    
    On Windows, ensures proper encoding (UTF-8) for better compatibility.
    """
    return _get_subprocess_env()