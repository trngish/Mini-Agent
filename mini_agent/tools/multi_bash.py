"""Multi-bash工具，用于在一次调用中执行多个命令"""

import asyncio
import json
from typing import Any

from ..utils.platform_utils import PlatformUtils
from .base import Tool, ToolResult


class MultiBashTool(Tool):
    """在一次工具调用中执行多个独立命令。

    按调用计费优化：将多个bash调用合并为一个。
    """

    def __init__(self, workspace_dir: str = ".", platform_mode: str = "auto"):
        self.workspace_dir = workspace_dir
        self.is_windows = PlatformUtils.is_windows(platform_mode)

    @property
    def name(self) -> str:
        return "multi_bash"

    @property
    def description(self) -> str:
        return (
            "Execute multiple independent shell commands simultaneously. "
            "Returns results for each command. Commands run in parallel. "
            "Use this instead of multiple bash calls to save API calls."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "commands": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "label": {"type": "string", "description": "Label for this command"},
                            "command": {"type": "string", "description": "Shell command to execute"},
                            "timeout": {"type": "integer", "description": "Timeout in seconds", "default": 60},
                        },
                        "required": ["command"],
                    },
                    "description": "List of commands to execute in parallel.",
                },
            },
            "required": ["commands"],
        }

    async def execute(self, commands: list[dict[str, Any]]) -> ToolResult:
        """并行执行多个命令"""
        commands = _ensure_list(commands)

        async def run_single(cmd: dict[str, Any]) -> str:
            label = cmd.get("label", f"cmd_{commands.index(cmd)}")
            command = cmd.get("command", "")
            timeout = cmd.get("timeout", 60)

            if not command:
                return f"[{label}]: No command specified"

            try:
                process = await asyncio.create_subprocess_shell(
                    command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=self.workspace_dir,
                )
                try:
                    stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
                except asyncio.TimeoutError:
                    process.kill()
                    return f"[{label}]: Timeout after {timeout}s"

                stdout_text = stdout.decode("utf-8", errors="replace").strip()
                stderr_text = stderr.decode("utf-8", errors="replace").strip()
                exit_code = process.returncode or 0

                if exit_code == 0:
                    output = stdout_text or "(no output)"
                    if len(output) > 3000:
                        output = output[:3000] + f"\n... (truncated, {len(output)} total chars)"
                    return f"SUCCESS [{label}] (exit={exit_code}):\n{output}"
                else:
                    error = stderr_text or stdout_text or f"exit code {exit_code}"
                    if len(error) > 1500:
                        error = error[:1500] + "..."
                    return f"ERROR [{label}] (exit={exit_code}):\n{error}"

            except Exception as e:
                return f"ERROR [{label}]: {type(e).__name__}: {str(e)}"

        tasks = [run_single(cmd) for cmd in commands]
        results = await asyncio.gather(*tasks)

        combined = "\n\n".join(results)
        success_count = sum(1 for r in results if r.startswith("SUCCESS"))
        fail_count = len(results) - success_count
        combined += f"\n\nSummary: {success_count} succeeded, {fail_count} failed"

        return ToolResult(
            success=fail_count == 0,
            content=combined,
            error="" if fail_count == 0 else f"{fail_count} command(s) failed",
        )


def _ensure_list(data: list[Any] | str | None) -> list[Any]:
    """确保输入是列表，必要时解析JSON字符串"""
    if data is None:
        return []
    if isinstance(data, list):
        return data
    if isinstance(data, str):
        try:
            parsed = json.loads(data)
            if isinstance(parsed, list):
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass
    return [data]
