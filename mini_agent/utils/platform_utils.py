"""Platform utilities for cross-platform compatibility.

Provides unified platform detection and shell configuration.
"""

import os
import platform
import sys
from typing import NamedTuple


class PlatformInfo(NamedTuple):
    """Platform information container."""
    is_windows: bool
    is_linux: bool
    is_macos: bool
    shell_name: str
    path_separator: str
    executable: str
    shell_args: list[str]


class PlatformUtils:
    """Unified platform utilities.
    
    Single source of truth for platform detection and configuration,
    replacing scattered platform checks across the codebase.
    """
    
    _cached_info: PlatformInfo | None = None
    
    @classmethod
    def get_info(cls, platform_mode: str = "auto") -> PlatformInfo:
        """Get platform information.
        
        Args:
            platform_mode: "windows", "linux", "auto" (auto-detect from OS)
            
        Returns:
            PlatformInfo with platform-specific settings
        """
        if cls._cached_info is not None:
            return cls._cached_info
        
        if platform_mode == "auto":
            system = platform.system()
        else:
            system = platform_mode.lower()
        
        is_windows = system == "Windows"
        is_linux = system == "Linux"
        is_macos = system == "Darwin"
        
        if is_windows:
            shell_name = "PowerShell"
            path_separator = "\\"
            executable = "powershell.exe"
            shell_args = ["-NoProfile", "-Command"]
        else:
            shell_name = "bash"
            path_separator = "/"
            executable = "/bin/bash"
            shell_args = ["-c"]
        
        cls._cached_info = PlatformInfo(
            is_windows=is_windows,
            is_linux=is_linux,
            is_macos=is_macos,
            shell_name=shell_name,
            path_separator=path_separator,
            executable=executable,
            shell_args=shell_args,
        )
        return cls._cached_info
    
    @classmethod
    def reset_cache(cls) -> None:
        """Reset cached platform info (mainly for testing)."""
        cls._cached_info = None
    
    @classmethod
    def is_windows(cls, platform_mode: str = "auto") -> bool:
        """Check if running on Windows."""
        return cls.get_info(platform_mode).is_windows
    
    @classmethod
    def is_linux(cls, platform_mode: str = "auto") -> bool:
        """Check if running on Linux."""
        return cls.get_info(platform_mode).is_linux
    
    @classmethod
    def is_macos(cls, platform_mode: str = "auto") -> bool:
        """Check if running on macOS."""
        return cls.get_info(platform_mode).is_macos
    
    @classmethod
    def get_shell_config(cls, platform_mode: str = "auto") -> tuple[str, list[str], str]:
        """Get shell configuration for subprocess.
        
        Args:
            platform_mode: Target platform mode
            
        Returns:
            Tuple of (executable, shell_args, shell_name)
        """
        info = cls.get_info(platform_mode)
        return info.executable, info.shell_args, info.shell_name
    
    @classmethod
    def get_path_separator(cls, platform_mode: str = "auto") -> str:
        """Get path separator for target platform."""
        return cls.get_info(platform_mode).path_separator
    
    @classmethod
    def get_subprocess_env(cls) -> dict[str, str]:
        """Get platform-appropriate environment for subprocess.
        
        Returns:
            Environment dict with proper encoding settings
        """
        env = dict(os.environ)
        
        if cls._cached_info and cls._cached_info.is_windows:
            # Ensure UTF-8 output on Windows
            env["PYTHONIOENCODING"] = "utf-8"
            # Set console mode for better Unicode support
            try:
                import ctypes
                kernel32 = ctypes.windll.kernel32
                kernel32.SetConsoleOutputCP(65001)
                kernel32.SetConsoleCP(65001)
            except Exception:
                pass
        
        return env


def get_platform_shell_args(platform_mode: str = "auto") -> tuple[str, list[str], str]:
    """Compatibility wrapper for bash_shared module.
    
    Args:
        platform_mode: Platform mode - "windows", "linux", or "auto"
        
    Returns:
        Tuple of (executable, shell_args, shell_name)
    """
    return PlatformUtils.get_shell_config(platform_mode)


def get_subprocess_env() -> dict[str, str]:
    """Compatibility wrapper for subprocess environment.
    
    Returns:
        Platform-appropriate environment dict
    """
    return PlatformUtils.get_subprocess_env()


def normalize_path_separators(path: str, platform_mode: str = "auto") -> str:
    """Normalize path separators for the target platform.
    
    Args:
        path: Path string that may contain mixed separators
        platform_mode: Target platform mode - "windows", "linux", or "auto"
        
    Returns:
        Normalized path string
    """
    info = PlatformUtils.get_info(platform_mode)
    if info.is_windows:
        return path.replace("/", "\\")
    return path  # Linux keeps forward slashes