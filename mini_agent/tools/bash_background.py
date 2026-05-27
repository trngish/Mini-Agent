"""Background shell process management."""

from __future__ import annotations

import asyncio
import contextlib
import re
from typing import Any


class BackgroundShell:
    """Background shell data container.

    Pure data class that only stores state and output.
    IO operations are managed externally by BackgroundShellManager.
    """

    def __init__(self, bash_id: str, command: str, process: asyncio.subprocess.Process, start_time: float):
        self.bash_id = bash_id
        self.command = command
        self.process = process
        self.start_time = start_time
        self.output_lines: list[str] = []
        self.last_read_index = 0
        self.status = "running"
        self.exit_code: int | None = None

    def add_output(self, line: str) -> None:
        """Add new output line."""
        self.output_lines.append(line)

    def get_new_output(self, filter_pattern: str | None = None) -> list[str]:
        """Get new output since last check, optionally filtered by regex."""
        new_lines = self.output_lines[self.last_read_index :]
        self.last_read_index = len(self.output_lines)

        if filter_pattern:
            try:
                pattern = re.compile(filter_pattern)
                new_lines = [line for line in new_lines if pattern.search(line)]
            except re.error:
                # Invalid regex, return all lines
                pass

        return new_lines

    def update_status(self, is_alive: bool, exit_code: int | None = None) -> None:
        """Update process status."""
        if not is_alive:
            self.status = "completed" if exit_code == 0 else "failed"
            self.exit_code = exit_code
        else:
            self.status = "running"

    async def terminate(self) -> None:
        """Terminate the background process."""
        if self.process.returncode is None:
            self.process.terminate()
            try:
                await asyncio.wait_for(self.process.wait(), timeout=5)
            except asyncio.TimeoutError:
                self.process.kill()
        self.status = "terminated"
        self.exit_code = self.process.returncode


class BackgroundShellManager:
    """Manager for all background shell processes.

    Uses class-level storage with agent_id prefixes to avoid cross-agent pollution.
    Each agent should use a unique agent_id when adding shells.
    """

    _shells: dict[str, BackgroundShell] = {}
    _monitor_tasks: dict[str, asyncio.Task[Any]] = {}

    @classmethod
    def _prefix_key(cls, agent_id: str, bash_id: str) -> str:
        """Generate prefixed key for shell storage."""
        return f"{agent_id}:{bash_id}"

    @classmethod
    def add(cls, shell: BackgroundShell, agent_id: str = "default") -> None:
        """Add a background shell to management.

        Args:
            shell: The BackgroundShell to add
            agent_id: Unique identifier for the agent (to isolate storage)
        """
        key = cls._prefix_key(agent_id, shell.bash_id)
        cls._shells[key] = shell

    @classmethod
    def get(cls, bash_id: str, agent_id: str = "default") -> BackgroundShell | None:
        """Get a background shell by ID.

        Args:
            bash_id: The shell ID
            agent_id: Agent identifier used when adding the shell
        """
        key = cls._prefix_key(agent_id, bash_id)
        return cls._shells.get(key)

    @classmethod
    def get_available_ids(cls, agent_id: str = "default") -> list[str]:
        """Get all available bash IDs for an agent.

        Args:
            agent_id: Agent identifier to filter shells
        """
        prefix = f"{agent_id}:"
        return [k.replace(prefix, "") for k in cls._shells if k.startswith(prefix)]

    @classmethod
    def _remove(cls, bash_id: str, agent_id: str = "default") -> None:
        """Remove a background shell from management (internal use only)."""
        key = cls._prefix_key(agent_id, bash_id)
        if key in cls._shells:
            del cls._shells[key]

    @classmethod
    async def start_monitor(cls, bash_id: str, agent_id: str = "default") -> None:
        """Start monitoring a background shell's output."""
        shell = cls.get(bash_id, agent_id)
        if not shell:
            return

        async def monitor() -> None:
            try:
                process = shell.process
                # Continuously read output until process ends
                while process.returncode is None:
                    try:
                        if process.stdout:
                            line = await asyncio.wait_for(process.stdout.readline(), timeout=0.1)
                            if line:
                                decoded_line = line.decode("utf-8", errors="replace").rstrip("\n")
                                shell.add_output(decoded_line)
                            else:
                                break
                    except asyncio.TimeoutError:
                        await asyncio.sleep(0.1)
                        continue
                    except Exception:
                        await asyncio.sleep(0.1)
                        continue

                # Process ended, wait for exit code
                try:
                    returncode = await process.wait()
                except Exception:
                    returncode = -1

                shell.update_status(is_alive=False, exit_code=returncode)

            except Exception as e:
                key = cls._prefix_key(agent_id, bash_id)
                if key in cls._shells:
                    cls._shells[key].status = "error"
                    cls._shells[key].add_output(f"Monitor error: {str(e)}")
            finally:
                monitor_key = f"{agent_id}:monitor:{bash_id}"
                if monitor_key in cls._monitor_tasks:
                    del cls._monitor_tasks[monitor_key]

        task = asyncio.create_task(monitor())
        monitor_key = f"{agent_id}:monitor:{bash_id}"
        cls._monitor_tasks[monitor_key] = task

    @classmethod
    def _cancel_monitor(cls, bash_id: str, agent_id: str = "default") -> None:
        """Cancel and remove a monitoring task (internal use only)."""
        monitor_key = f"{agent_id}:monitor:{bash_id}"
        if monitor_key in cls._monitor_tasks:
            task = cls._monitor_tasks[monitor_key]
            if not task.done():
                task.cancel()

    @classmethod
    async def terminate(cls, bash_id: str, agent_id: str = "default") -> BackgroundShell:
        """Terminate a background shell and clean up all resources.

        Args:
            bash_id: The unique identifier of the background shell
            agent_id: Agent identifier used when adding the shell

        Returns:
            The terminated BackgroundShell object

        Raises:
            ValueError: If shell not found
        """
        shell = cls.get(bash_id, agent_id)
        if not shell:
            raise ValueError(f"Shell not found: {bash_id}")

        # Terminate the process
        await shell.terminate()

        # Clean up monitoring and remove from manager
        cls._cancel_monitor(bash_id, agent_id)
        cls._remove(bash_id, agent_id)

        return shell

    @classmethod
    async def cleanup_all(cls, agent_id: str = "default") -> list[str]:
        """Clean up all background shells for an agent.

        Args:
            agent_id: Agent identifier to clean up shells for

        Returns:
            List of terminated shell IDs
        """
        terminated_ids = cls.get_available_ids(agent_id)

        # Terminate all shells for this agent
        for bash_id in terminated_ids:
            with contextlib.suppress(Exception):
                await cls.terminate(bash_id, agent_id)

        return terminated_ids

    @classmethod
    def get_stats(cls, agent_id: str = "default") -> dict[str, Any]:
        """Get statistics about managed shells for an agent.

        Args:
            agent_id: Agent identifier to get stats for

        Returns:
            Dictionary with shell counts and status
        """
        prefix = f"{agent_id}:"
        agent_shells = {
            k: v for k, v in cls._shells.items() if k.startswith(prefix) and not k.startswith(f"{agent_id}:monitor:")
        }
        running = sum(1 for s in agent_shells.values() if s.status == "running")
        completed = sum(1 for s in agent_shells.values() if s.status == "completed")
        failed = sum(1 for s in agent_shells.values() if s.status == "failed")

        return {
            "total": len(agent_shells),
            "running": running,
            "completed": completed,
            "failed": failed,
        }
