"""Bash command execution result types."""

from pydantic import Field, model_validator

from .base import ToolResult


class BashOutputResult(ToolResult):
    """Bash command execution result with separated stdout and stderr.

    Inherits from ToolResult which provides:
    - success: bool
    - content: str (used for formatted output message, auto-generated from stdout/stderr)
    - error: str | None (used for error messages)
    """

    stdout: str = Field(description="The command's standard output")
    stderr: str = Field(description="The command's standard error output")
    exit_code: int = Field(description="The command's exit code")
    bash_id: str | None = Field(default=None, description="Shell process ID (only when run_in_background=True)")

    @model_validator(mode="after")
    def format_content(self) -> "BashOutputResult":
        """Auto-format content from stdout and stderr if content is empty."""
        output = ""
        if self.stdout:
            output += self.stdout
        if self.stderr:
            output += f"\n[stderr]:\n{self.stderr}"
        if self.bash_id:
            output += f"\n[bash_id]:\n{self.bash_id}"
        if self.exit_code is not None:
            output += f"\n[exit_code]:\n{self.exit_code}"

        if not output:
            output = "(no output)"

        self.content = output
        return self
