"""Shell命令执行工具，支持后台进程管理。

支持bash（Unix/Linux/macOS）和PowerShell（Windows）。
平台模式可通过platform_mode参数配置或自动检测。

模块说明：
- bash_shared: 平台特定的shell配置工具
- bash_result: BashOutputResult结果类型
- bash_background: BackgroundShell和BackgroundShellManager
- bash_foreground: ForegroundExecutor
"""

import asyncio
import time
import uuid
from typing import Any

from ..utils.command_validator import (
    DangerLevel,
    assess_command_danger,
    detect_platform_mismatch,
    translate_command_for_platform,
)
from ..utils.platform_utils import PlatformUtils
from .base import Tool
from .bash_background import BackgroundShell, BackgroundShellManager
from .bash_foreground import ForegroundExecutor
from .bash_result import BashOutputResult
from .bash_shared import get_platform_shell_args, get_subprocess_env


class BashTool(Tool):
    """在前台或后台执行shell命令。

    根据platform_mode自动使用合适的shell：
    - Windows模式：PowerShell
    - Linux模式：bash
    """

    def __init__(self, workspace_dir: str | None = None, platform_mode: str = "auto", default_timeout: int = 120):
        """初始化BashTool，使用平台特定的shell。

        参数说明:
            workspace_dir: 命令执行的工作目录。
                         如果提供，所有命令将在此目录中运行。
                         如果为None，命令将在进程的当前工作目录中运行。
            platform_mode: 平台模式 - "windows"、"linux"或"auto"（自动检测操作系统）
            default_timeout: 前台命令的默认超时时间（秒）（默认：120）
        """
        self.workspace_dir = workspace_dir
        self.default_timeout = default_timeout

        # 使用统一的PlatformUtils进行平台检测
        self.is_windows = PlatformUtils.is_windows(platform_mode)
        self.shell_name = "PowerShell" if self.is_windows else "bash"

        # 将前台执行委托给专用模块
        self._foreground_executor = ForegroundExecutor(
            workspace_dir=workspace_dir,
            is_windows=self.is_windows,
            default_timeout=default_timeout,
        )

    @property
    def name(self) -> str:
        return "bash"

    @property
    def description(self) -> str:
        shell_examples = {
            "Windows": """Execute PowerShell commands in foreground or background.

For terminal operations like git, npm, docker, etc. DO NOT use for file operations - use specialized tools.

Parameters:
  - command (required): PowerShell command to execute
  - timeout (optional): Timeout in seconds (default: 120, max: 600) for foreground commands
  - run_in_background (optional): Set true for long-running commands (servers, etc.)

Tips:
  - Quote file paths with spaces: cd "My Documents"
  - Chain dependent commands with semicolon: git add . ; git commit -m "msg"
  - Use absolute paths instead of cd when possible
  - For background commands, monitor with bash_output and terminate with bash_kill

Examples:
  - git status
  - npm test
  - python -m http.server 8080 (with run_in_background=true)""",
            "Unix": """Execute bash commands in foreground or background.

For terminal operations like git, npm, docker, etc. DO NOT use for file operations - use specialized tools.

Parameters:
  - command (required): Bash command to execute
  - timeout (optional): Timeout in seconds (default: 120, max: 600) for foreground commands
  - run_in_background (optional): Set true for long-running commands (servers, etc.)

Tips:
  - Quote file paths with spaces: cd "My Documents"
  - Chain dependent commands with &&: git add . && git commit -m "msg"
  - Use absolute paths instead of cd when possible
  - For background commands, monitor with bash_output and terminate with bash_kill

Examples:
  - git status
  - npm test
  - python3 -m http.server 8080 (with run_in_background=true)""",
        }
        return shell_examples["Windows"] if self.is_windows else shell_examples["Unix"]

    @property
    def parameters(self) -> dict[str, Any]:
        cmd_desc = f"The {self.shell_name} command to execute. Quote file paths with spaces using double quotes."
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": cmd_desc,
                },
                "timeout": {
                    "type": "integer",
                    "description": (
                        "Optional: Timeout in seconds (default: 120, max: 600). Only applies to foreground commands."
                    ),
                    "default": 120,
                },
                "run_in_background": {
                    "type": "boolean",
                    "description": (
                        "Optional: Set to true to run the command in the background."
                        " Use this for long-running commands like servers."
                        " You can monitor output using bash_output tool."
                    ),
                    "default": False,
                },
            },
            "required": ["command"],
        }

    async def execute(
        self,
        command: str,
        timeout: int = 120,
        run_in_background: bool = False,
    ) -> BashOutputResult:
        """执行shell命令，支持可选的后台执行。

        参数说明:
            command: 要执行的shell命令
            timeout: 超时时间（秒）（默认：120，最大：600）
            run_in_background: 设置为true可在后台运行命令

        返回值:
            包含命令输出和状态的BashOutputResult
        """
        try:
            # 安全检查：检测危险命令
            danger_level, danger_reason = assess_command_danger(command)
            if danger_level == DangerLevel.BLOCKED:
                return BashOutputResult(
                    success=False,
                    error=f"命令因安全原因被阻止：{danger_reason}",
                    stdout="",
                    stderr=danger_reason or "被阻止的命令",
                    exit_code=-1,
                )

            # 平台兼容性检查
            platform_warning = detect_platform_mismatch(command, self.is_windows)

            # 根据平台模式获取shell配置
            shell_exe, shell_args, shell_name = get_platform_shell_args("windows" if self.is_windows else "linux")

            # 获取平台适当的环境变量
            env = get_subprocess_env()

            # 准备shell命令
            shell_cmd = self._build_shell_command(shell_exe, shell_args, command)

            if run_in_background:
                result = await self._execute_background(shell_cmd, command, env)
            else:
                result = await self._foreground_executor.execute(shell_cmd, command, env, timeout)

            # 检测到平台不匹配时添加警告
            if platform_warning:
                warning_section = f"\n[{result.exit_code}]" if result.exit_code else ""
                result.stdout = (
                    f"{platform_warning}{warning_section}\n\n{result.stdout}" if result.stdout else platform_warning
                )
                result.content = result.stdout
            return result

        except Exception as e:
            return BashOutputResult(
                success=False,
                error=str(e),
                stdout="",
                stderr=str(e),
                exit_code=-1,
            )

    async def _execute_background(
        self,
        shell_cmd: list[str] | str,
        original_command: str,
        env: dict[str, str],
    ) -> BashOutputResult:
        """以后台模式执行命令。

        委托给BackgroundShellManager进行进程生命周期管理。

        参数说明:
            shell_cmd: 平台特定的shell命令
            original_command: 原始命令字符串
            env: 子进程的环境变量

        返回值:
            表示后台命令已启动的BashOutputResult
        """
        bash_id = uuid.uuid4().hex[:12]

        if self.is_windows:
            process = await asyncio.create_subprocess_exec(
                *shell_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=self.workspace_dir,
                env=env,
            )
        else:
            process = await asyncio.create_subprocess_shell(
                shell_cmd,  # type: ignore[arg-type]
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=self.workspace_dir,
                env=env,
            )

        # 创建shell数据容器
        shell = BackgroundShell(
            bash_id=bash_id,
            command=original_command,
            process=process,
            start_time=time.time(),
        )

        # 注册到管理器并开始监控
        BackgroundShellManager.add(shell)
        BackgroundShellManager.start_monitor(bash_id)  # type: ignore[unused-coroutine]

        return BashOutputResult(
            success=True,
            stdout=f"后台命令已启动，ID：{bash_id}",
            stderr="",
            bash_id=bash_id,
            exit_code=0,
        )

    def _build_shell_command(
        self,
        shell_exe: str,
        shell_args: list[str],
        command: str,
    ) -> list[str] | str:
        """构建平台特定的shell命令。

        参数说明:
            shell_exe: Shell可执行文件路径
            shell_args: Shell参数
            command: 要执行的命令

        返回值:
            平台适当的命令结构
        """
        if self.is_windows:
            # 将bash风格的&&转换为PowerShell风格的;
            normalized = self._normalize_command(command)
            return [shell_exe] + shell_args + [normalized]
        else:
            return command

    def _normalize_command(self, command: str) -> str:
        """规范化Windows PowerShell命令。

        使用command_validator中的综合转换系统将bash风格语法转换为PowerShell兼容语法。

        参数说明:
            command: 原始bash风格命令

        返回值:
            PowerShell兼容命令
        """
        translated, _ = translate_command_for_platform(command, is_windows=True)

        import re

        translated = re.sub(r"\$([a-zA-Z_][a-zA-Z0-9_]*)", r"$env:\1", translated)
        translated = re.sub(r"\$\{([a-zA-Z_][a-zA-Z0-9_]*)\}", r"$env:\1", translated)

        translated = translated.replace("~", "$env:USERPROFILE")

        translated = re.sub(r"`([^`]+)`", r"$( \1 )", translated)

        if "&&" in translated:
            translated = translated.replace("&&", ";")
        if "||" in translated:
            translated = translated.replace("||", ";")

        return translated


class BashOutputTool(Tool):
    """从后台bash shell获取输出"""

    @property
    def name(self) -> str:
        return "bash_output"

    @property
    def description(self) -> str:
        return """从运行中或已完成的后台bash shell获取输出。

- 通过bash_id参数指定shell
- 仅返回自上次检查以来的新输出
- 返回stdout和stderr输出以及shell状态
- 支持可选的正则表达式过滤，仅显示匹配的行
- 当需要监控或检查长时间运行的shell输出时使用此工具
- Shell ID可通过带有run_in_background=true参数的bash工具获取

进程状态值：
  - "running"：仍在执行
  - "completed"：成功完成
  - "failed"：出错完成
  - "terminated"：已被终止
  - "error"：发生错误

示例：bash_output(bash_id="abc12345")"""

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "bash_id": {
                    "type": "string",
                    "description": (
                        "要获取输出的后台shell的ID。"
                        " 当使用run_in_background=true启动命令时返回shell ID。"
                    ),
                },
                "filter_str": {
                    "type": "string",
                    "description": (
                        "可选的正则表达式，用于过滤输出行。"
                        " 只有匹配此正则表达式的行才会包含在结果中。"
                        " 不匹配的行将无法再读取。"
                    ),
                },
            },
            "required": ["bash_id"],
        }

    async def execute(
        self,
        bash_id: str,
        filter_str: str | None = None,
    ) -> BashOutputResult:
        """从后台shell获取输出。

        参数说明:
            bash_id: 后台shell的唯一标识符
            filter_str: 可选的正则表达式模式，用于过滤输出行

        返回值:
            包含shell输出的BashOutputResult，包括stdout、stderr、状态和成功标志
        """
        try:
            # 从管理器获取后台shell
            bg_shell = BackgroundShellManager.get(bash_id)
            if not bg_shell:
                available_ids = BackgroundShellManager.get_available_ids()
                return BashOutputResult(
                    success=False,
                    error=f"未找到 shell：{bash_id}。可用 ID：{available_ids or '无'}",
                    stdout="",
                    stderr="",
                    exit_code=-1,
                )

            # 获取新输出
            new_lines = bg_shell.get_new_output(filter_pattern=filter_str)
            stdout = "\n".join(new_lines) if new_lines else ""

            return BashOutputResult(
                success=True,
                stdout=stdout,
                stderr="",  # 后台shell合并stdout/stderr
                exit_code=bg_shell.exit_code if bg_shell.exit_code is not None else 0,
                bash_id=bash_id,
            )

        except Exception as e:
            return BashOutputResult(
                success=False,
                error=f"获取 bash 输出失败：{str(e)}",
                stdout="",
                stderr=str(e),
                exit_code=-1,
            )


class BashKillTool(Tool):
    """终止运行中的后台bash shell"""

    @property
    def name(self) -> str:
        return "bash_kill"

    @property
    def description(self) -> str:
        return """根据ID终止运行中的后台bash shell。

- 通过bash_id参数指定要终止的shell
- 首先尝试优雅终止（SIGTERM），必要时强制终止（SIGKILL）
- 返回终止前的最终状态和剩余输出
- 清理与shell相关的所有资源
- 当需要终止长时间运行的shell时使用此工具
- Shell ID可通过带有run_in_background=true参数的bash工具获取

示例：bash_kill(bash_id="abc12345")"""

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "bash_id": {
                    "type": "string",
                    "description": (
                        "要终止的后台shell的ID。"
                        " 当使用run_in_background=true启动命令时返回shell ID。"
                    ),
                },
            },
            "required": ["bash_id"],
        }

    async def execute(self, bash_id: str) -> BashOutputResult:
        """终止后台shell进程。

        参数说明:
            bash_id: 要终止的后台shell的唯一标识符

        返回值:
            包含终止状态和剩余输出的BashOutputResult
        """
        try:
            # 终止前获取剩余输出
            bg_shell = BackgroundShellManager.get(bash_id)
            remaining_lines = bg_shell.get_new_output() if bg_shell else []

            # 通过管理器终止（处理所有清理工作）
            bg_shell = await BackgroundShellManager.terminate(bash_id)

            # 获取剩余输出
            stdout = "\n".join(remaining_lines) if remaining_lines else ""

            return BashOutputResult(
                success=True,
                stdout=stdout,
                stderr="",
                exit_code=bg_shell.exit_code if bg_shell.exit_code is not None else 0,
                bash_id=bash_id,
            )

        except ValueError as e:
            # Shell未找到
            available_ids = BackgroundShellManager.get_available_ids()
            return BashOutputResult(
                success=False,
                error=f"{str(e)}。可用 ID：{available_ids or '无'}",
                stdout="",
                stderr=str(e),
                exit_code=-1,
            )
        except Exception as e:
            return BashOutputResult(
                success=False,
                error=f"终止 bash shell 失败：{str(e)}",
                stdout="",
                stderr=str(e),
                exit_code=-1,
            )
