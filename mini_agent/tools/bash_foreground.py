"""Foreground shell command execution.

Handles synchronous command execution with proper timeout and output handling.
"""

from __future__ import annotations

import asyncio

from .bash_result import BashOutputResult

# Constants
DEFAULT_TIMEOUT = 120
MAX_TIMEOUT = 600
FALLBACK_ENCODINGS = ("utf-8", "gbk", "cp1252")


class ForegroundExecutor:
    """Execute shell commands in foreground with timeout handling."""

    def __init__(
        self,
        workspace_dir: str | None = None,
        is_windows: bool = False,
        default_timeout: int = DEFAULT_TIMEOUT,
    ):
        self.workspace_dir = workspace_dir
        self.is_windows = is_windows
        self.default_timeout = default_timeout

    def validate_timeout(self, timeout: int) -> int:
        """Validate and normalize timeout value.

        Args:
            timeout: Raw timeout value

        Returns:
            Normalized timeout within bounds
        """
        if timeout > MAX_TIMEOUT:
            return MAX_TIMEOUT
        elif timeout < 1:
            return self.default_timeout
        return timeout

    async def execute(
        self,
        shell_cmd: list[str] | str,
        original_command: str,  # noqa: ARG002
        env: dict[str, str],
        timeout: int,
    ) -> BashOutputResult:
        """Execute command in foreground mode.

        Args:
            shell_cmd: Platform-specific shell command
            original_command: Original command string
            env: Environment variables for subprocess
            timeout: Timeout in seconds

        Returns:
            BashOutputResult with command output
        """
        timeout = self.validate_timeout(timeout)

        if self.is_windows:
            process = await asyncio.create_subprocess_exec(
                *shell_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.workspace_dir,
                env=env,
            )
        else:
            process = await asyncio.create_subprocess_shell(
                shell_cmd,  # type: ignore[arg-type]
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.workspace_dir,
                env=env,
            )

        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            process.kill()
            error_msg = f"Command timed out after {timeout} seconds"
            return BashOutputResult(
                success=False,
                error=error_msg,
                stdout="",
                stderr=error_msg,
                exit_code=-1,
            )

        # Decode output with fallback encodings
        stdout_text = self._decode_output(stdout)
        stderr_text = self._decode_output(stderr)

        # Create result
        is_success = process.returncode == 0
        result_error: str | None = None
        if not is_success:
            result_error = f"Command failed with exit code {process.returncode}"
            if stderr_text:
                result_error += f"\n{stderr_text.strip()}"

        return BashOutputResult(
            success=is_success,
            error=result_error,
            stdout=stdout_text,
            stderr=stderr_text,
            exit_code=process.returncode or 0,
        )

    def _decode_output(self, data: bytes) -> str:
        """Decode output bytes with fallback encodings.

        Args:
            data: Raw bytes from subprocess

        Returns:
            Decoded string with replacements for invalid chars
        """
        if not data:
            return ""

        # Try encodings in order of preference
        for encoding in FALLBACK_ENCODINGS:
            try:
                return data.decode(encoding, errors="strict")
            except UnicodeDecodeError:
                continue

        # Final fallback: replace errors
        return data.decode("utf-8", errors="replace")
