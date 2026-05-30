"""平台工具，用于跨平台兼容性。

提供统一的平台检测、shell 配置，
以及平台间的命令转换。
"""

import os
import platform
from typing import NamedTuple


class PlatformInfo(NamedTuple):
    """平台信息容器。"""

    is_windows: bool
    is_linux: bool
    is_macos: bool
    shell_name: str
    path_separator: str
    executable: str
    shell_args: list[str]


class PlatformUtils:
    """统一的平台工具。

    平台检测和配置的单一数据源，
    替代代码库中分散的平台检查。
    """

    _cached_info: PlatformInfo | None = None

    @classmethod
    def get_info(cls, platform_mode: str = "auto") -> PlatformInfo:
        """获取平台信息。

        Args:
            platform_mode: "windows"、"linux"、"auto"（从操作系统自动检测）

        Returns:
            包含平台特定设置的 PlatformInfo
        """
        if cls._cached_info is not None:
            return cls._cached_info

        system = platform.system() if platform_mode == "auto" else platform_mode.lower()

        is_windows = system == "Windows"
        is_linux = system == "Linux"
        is_macos = system == "Darwin"

        if is_windows:
            shell_name = "PowerShell"
            path_separator = "\\"
            executable = cls._detect_powershell()
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
    def _detect_powershell(cls) -> str:
        """检测可用的最佳 PowerShell 可执行文件。

        优先使用 pwsh（PowerShell 7+）而非 powershell.exe（Windows PowerShell 5.1）。

        Returns:
            PowerShell 可执行文件路径
        """
        for candidate in ("pwsh.exe", "pwsh", "powershell.exe"):
            path = cls._which(candidate)
            if path:
                return path
        return "powershell.exe"

    @classmethod
    def _which(cls, name: str) -> str | None:
        """在 PATH 中查找可执行文件（跨平台）。

        Args:
            name: 要查找的可执行文件名

        Returns:
            如果找到则返回完整路径，否则返回 None
        """
        import shutil

        return shutil.which(name)

    @classmethod
    def reset_cache(cls) -> None:
        """重置缓存的平台信息（主要用于测试）。"""
        cls._cached_info = None

    @classmethod
    def is_windows(cls, platform_mode: str = "auto") -> bool:
        """检查是否在 Windows 上运行。"""
        return cls.get_info(platform_mode).is_windows

    @classmethod
    def is_linux(cls, platform_mode: str = "auto") -> bool:
        """检查是否在 Linux 上运行。"""
        return cls.get_info(platform_mode).is_linux

    @classmethod
    def is_macos(cls, platform_mode: str = "auto") -> bool:
        """检查是否在 macOS 上运行。"""
        return cls.get_info(platform_mode).is_macos

    @classmethod
    def get_shell_config(cls, platform_mode: str = "auto") -> tuple[str, list[str], str]:
        """获取子进程的 shell 配置。

        Args:
            platform_mode: 目标平台模式

        Returns:
            (executable, shell_args, shell_name) 元组
        """
        info = cls.get_info(platform_mode)
        return info.executable, info.shell_args, info.shell_name

    @classmethod
    def get_path_separator(cls, platform_mode: str = "auto") -> str:
        """获取目标平台的路径分隔符。"""
        return cls.get_info(platform_mode).path_separator

    @classmethod
    def get_subprocess_env(cls) -> dict[str, str]:
        """获取适合平台的子进程环境。

        Returns:
            带有正确编码设置的环境字典
        """
        env = dict(os.environ)

        if cls._cached_info and cls._cached_info.is_windows:
            env["PYTHONIOENCODING"] = "utf-8"
            try:
                import ctypes

                kernel32 = ctypes.windll.kernel32
                kernel32.SetConsoleOutputCP(65001)
                kernel32.SetConsoleCP(65001)
            except Exception:
                pass

        return env

    @classmethod
    def get_platform_command_tips(cls) -> list[str]:
        """获取当前平台的特定命令提示。

        Returns:
            常用命令转换提示字符串列表
        """
        if cls.get_info().is_windows:
            return [
                "Use 'Get-ChildItem' instead of 'ls'",
                "Use 'Get-Content' instead of 'cat'",
                "Use 'Select-Object -First N' instead of 'head -N'",
                "Use 'Select-Object -Last N' instead of 'tail -N'",
                "Use 'Select-String' instead of 'grep'",
                "Use 'Get-Command' instead of 'which'",
                "Use 'New-Item -ItemType File' instead of 'touch'",
                "Use 'Remove-Item -Recurse -Force' instead of 'rm -rf'",
                "Use 'Copy-Item -Recurse' instead of 'cp -r'",
                "Use 'Get-Location' instead of 'pwd'",
                "Use ';' instead of '&&' to chain commands",
                "Use '2>$null' instead of '2>/dev/null'",
                "Use '$env:VAR' instead of '$VAR' for env variables",
            ]
        else:
            return [
                "Use 'ls' instead of 'Get-ChildItem'",
                "Use 'cat' instead of 'Get-Content'",
                "Use 'head/tail' instead of 'Select-Object -First/-Last'",
                "Use 'grep' instead of 'Select-String'",
                "Use 'which' instead of 'Get-Command'",
                "Use '&&' instead of ';' to chain commands",
                "Use '2>/dev/null' instead of '2>$null'",
                "Use 'find . -type f' instead of 'dir /s /b'",
            ]


def get_platform_shell_args(platform_mode: str = "auto") -> tuple[str, list[str], str]:
    """bash_shared 模块的兼容性包装器。

    Args:
        platform_mode: 平台模式 - "windows"、"linux" 或 "auto"

    Returns:
        (executable, shell_args, shell_name) 元组
    """
    return PlatformUtils.get_shell_config(platform_mode)


def get_subprocess_env() -> dict[str, str]:
    """子进程环境的兼容性包装器。

    Returns:
        适合平台的環境字典
    """
    return PlatformUtils.get_subprocess_env()


def normalize_path_separators(path: str, platform_mode: str = "auto") -> str:
    """为目标平台规范化路径分隔符。

    Args:
        path: 可能包含混合分隔符的路径字符串
        platform_mode: 目标平台模式 - "windows"、"linux" 或 "auto"

    Returns:
        规范化后的路径字符串
    """
    info = PlatformUtils.get_info(platform_mode)
    if info.is_windows:
        return path.replace("/", "\\")
    return path
