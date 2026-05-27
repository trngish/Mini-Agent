"""Shell command execution tool with background process management.

Supports both bash (Unix/Linux/macOS) and PowerShell (Windows).
Platform mode can be configured via platform_mode parameter or auto-detected.

Modules:
- bash_shared: Platform-specific shell configuration utilities
- bash_result: BashOutputResult result type
- bash_background: BackgroundShell and BackgroundShellManager
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
    """Execute shell commands in foreground or background.

    Automatically uses appropriate shell based on platform_mode:
    - Windows mode: PowerShell
    - Linux mode: bash
    """

    def __init__(self, workspace_dir: str | None = None, platform_mode: str = "auto", default_timeout: int = 120):
        """Initialize BashTool with platform-specific shell.

        Args:
            workspace_dir: Working directory for command execution.
                           If provided, all commands run in this directory.
                           If None, commands run in the process's cwd.
            platform_mode: Platform mode - "windows", "linux", or "auto" (auto-detect from OS)
            default_timeout: Default timeout in seconds for foreground commands (default: 120)
        """
        self.workspace_dir = workspace_dir
        self.default_timeout = default_timeout

        # Use unified PlatformUtils for platform detection
        self.is_windows = PlatformUtils.is_windows(platform_mode)
        self.shell_name = "PowerShell" if self.is_windows else "bash"

        # Delegate foreground execution to specialized module
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
        """Execute shell command with optional background execution.

        Args:
            command: The shell command to execute
            timeout: Timeout in seconds (default: 120, max: 600)
            run_in_background: Set true to run command in background

        Returns:
            BashOutputResult with command output and status
        """
        try:
            # Security: Check for dangerous commands
            danger_level, danger_reason = assess_command_danger(command)
            if danger_level == DangerLevel.BLOCKED:
                return BashOutputResult(
                    success=False,
                    error=f"Command blocked for safety: {danger_reason}",
                    stdout="",
                    stderr=danger_reason or "Blocked command",
                    exit_code=-1,
                )

            # Platform compatibility check
            platform_warning = detect_platform_mismatch(command, self.is_windows)

            # Get shell configuration based on platform mode
            shell_exe, shell_args, shell_name = get_platform_shell_args("windows" if self.is_windows else "linux")

            # Get platform-appropriate environment
            env = get_subprocess_env()

            # Prepare shell command
            shell_cmd = self._build_shell_command(shell_exe, shell_args, command)

            if run_in_background:
                result = await self._execute_background(shell_cmd, command, env)
            else:
                result = await self._foreground_executor.execute(shell_cmd, command, env, timeout)

            # Prepend platform warning if mismatch detected
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
        """Execute command in background mode.

        Delegates to BackgroundShellManager for process lifecycle management.

        Args:
            shell_cmd: Platform-specific shell command
            original_command: Original command string
            env: Environment variables for subprocess

        Returns:
            BashOutputResult indicating background command started
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

        # Create shell data container
        shell = BackgroundShell(
            bash_id=bash_id,
            command=original_command,
            process=process,
            start_time=time.time(),
        )

        # Register with manager and start monitoring
        BackgroundShellManager.add(shell)
        BackgroundShellManager.start_monitor(bash_id)  # type: ignore[unused-coroutine]

        return BashOutputResult(
            success=True,
            stdout=f"Background command started with ID: {bash_id}",
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
        """Build platform-specific shell command.

        Args:
            shell_exe: Shell executable path
            shell_args: Shell arguments
            command: Command to execute

        Returns:
            Platform-appropriate command structure
        """
        if self.is_windows:
            # Convert bash-style && to PowerShell-style ;
            normalized = self._normalize_command(command)
            return [shell_exe] + shell_args + [normalized]
        else:
            return command

    def _normalize_command(self, command: str) -> str:
        """Normalize command for Windows PowerShell.

        Converts bash-style syntax to PowerShell-compatible syntax using
        the comprehensive translation system from command_validator.

        Args:
            command: Original bash-style command

        Returns:
            PowerShell-compatible command
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
    """Retrieve output from background bash shells."""

    @property
    def name(self) -> str:
        return "bash_output"

    @property
    def description(self) -> str:
        return """Retrieves output from a running or completed background bash shell.

        - Takes a bash_id parameter identifying the shell
        - Always returns only new output since the last check
        - Returns stdout and stderr output along with shell status
        - Supports optional regex filtering to show only lines matching a pattern
        - Use this tool when you need to monitor or check the output of a long-running shell
        - Shell IDs can be found using the bash tool with run_in_background=true

        Process status values:
          - "running": Still executing
          - "completed": Finished successfully
          - "failed": Finished with error
          - "terminated": Was terminated
          - "error": Error occurred

        Example: bash_output(bash_id="abc12345")"""

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "bash_id": {
                    "type": "string",
                    "description": (
                        "The ID of the background shell to retrieve output from."
                        " Shell IDs are returned when starting a command with run_in_background=true."
                    ),
                },
                "filter_str": {
                    "type": "string",
                    "description": (
                        "Optional regular expression to filter the output lines."
                        " Only lines matching this regex will be included in the result."
                        " Any lines that do not match will no longer be available to read."
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
        """Retrieve output from background shell.

        Args:
            bash_id: The unique identifier of the background shell
            filter_str: Optional regex pattern to filter output lines

        Returns:
            BashOutputResult with shell output including stdout, stderr, status, and success flag
        """
        try:
            # Get background shell from manager
            bg_shell = BackgroundShellManager.get(bash_id)
            if not bg_shell:
                available_ids = BackgroundShellManager.get_available_ids()
                return BashOutputResult(
                    success=False,
                    error=f"Shell not found: {bash_id}. Available: {available_ids or 'none'}",
                    stdout="",
                    stderr="",
                    exit_code=-1,
                )

            # Get new output
            new_lines = bg_shell.get_new_output(filter_pattern=filter_str)
            stdout = "\n".join(new_lines) if new_lines else ""

            return BashOutputResult(
                success=True,
                stdout=stdout,
                stderr="",  # Background shells combine stdout/stderr
                exit_code=bg_shell.exit_code if bg_shell.exit_code is not None else 0,
                bash_id=bash_id,
            )

        except Exception as e:
            return BashOutputResult(
                success=False,
                error=f"Failed to get bash output: {str(e)}",
                stdout="",
                stderr=str(e),
                exit_code=-1,
            )


class BashKillTool(Tool):
    """Terminate a running background bash shell."""

    @property
    def name(self) -> str:
        return "bash_kill"

    @property
    def description(self) -> str:
        return """Kills a running background bash shell by its ID.

        - Takes a bash_id parameter identifying the shell to kill
        - Attempts graceful termination (SIGTERM) first, then forces (SIGKILL) if needed
        - Returns the final status and any remaining output before termination
        - Cleans up all resources associated with the shell
        - Use this tool when you need to terminate a long-running shell
        - Shell IDs can be found using the bash tool with run_in_background=true

        Example: bash_kill(bash_id="abc12345")"""

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "bash_id": {
                    "type": "string",
                    "description": (
                        "The ID of the background shell to terminate."
                        " Shell IDs are returned when starting a command with run_in_background=true."
                    ),
                },
            },
            "required": ["bash_id"],
        }

    async def execute(self, bash_id: str) -> BashOutputResult:
        """Terminate a background shell process.

        Args:
            bash_id: The unique identifier of the background shell to terminate

        Returns:
            BashOutputResult with termination status and remaining output
        """
        try:
            # Get remaining output before termination
            bg_shell = BackgroundShellManager.get(bash_id)
            remaining_lines = bg_shell.get_new_output() if bg_shell else []

            # Terminate through manager (handles all cleanup)
            bg_shell = await BackgroundShellManager.terminate(bash_id)

            # Get remaining output
            stdout = "\n".join(remaining_lines) if remaining_lines else ""

            return BashOutputResult(
                success=True,
                stdout=stdout,
                stderr="",
                exit_code=bg_shell.exit_code if bg_shell.exit_code is not None else 0,
                bash_id=bash_id,
            )

        except ValueError as e:
            # Shell not found
            available_ids = BackgroundShellManager.get_available_ids()
            return BashOutputResult(
                success=False,
                error=f"{str(e)}. Available: {available_ids or 'none'}",
                stdout="",
                stderr=str(e),
                exit_code=-1,
            )
        except Exception as e:
            return BashOutputResult(
                success=False,
                error=f"Failed to terminate bash shell: {str(e)}",
                stdout="",
                stderr=str(e),
                exit_code=-1,
            )
