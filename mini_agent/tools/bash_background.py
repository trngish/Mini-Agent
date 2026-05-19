"""Background shell process management."""

import asyncio
import re


class BackgroundShell:
    """Background shell data container.

    Pure data class that only stores state and output.
    IO operations are managed externally by BackgroundShellManager.
    """

    def __init__(self, bash_id: str, command: str, process: "asyncio.subprocess.Process", start_time: float):
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
    """Manager for all background shell processes."""

    _shells: dict[str, BackgroundShell] = {}
    _monitor_tasks: dict[str, asyncio.Task] = {}

    @classmethod
    def add(cls, shell: BackgroundShell) -> None:
        """Add a background shell to management."""
        cls._shells[shell.bash_id] = shell

    @classmethod
    def get(cls, bash_id: str) -> BackgroundShell | None:
        """Get a background shell by ID."""
        return cls._shells.get(bash_id)

    @classmethod
    def get_available_ids(cls) -> list[str]:
        """Get all available bash IDs."""
        return list(cls._shells.keys())

    @classmethod
    def _remove(cls, bash_id: str) -> None:
        """Remove a background shell from management (internal use only)."""
        if bash_id in cls._shells:
            del cls._shells[bash_id]

    @classmethod
    async def start_monitor(cls, bash_id: str) -> None:
        """Start monitoring a background shell's output."""
        shell = cls.get(bash_id)
        if not shell:
            return

        async def monitor():
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
                if bash_id in cls._shells:
                    cls._shells[bash_id].status = "error"
                    cls._shells[bash_id].add_output(f"Monitor error: {str(e)}")
            finally:
                if bash_id in cls._monitor_tasks:
                    del cls._monitor_tasks[bash_id]

        task = asyncio.create_task(monitor())
        cls._monitor_tasks[bash_id] = task

    @classmethod
    def _cancel_monitor(cls, bash_id: str) -> None:
        """Cancel and remove a monitoring task (internal use only)."""
        if bash_id in cls._monitor_tasks:
            task = cls._monitor_tasks[bash_id]
            if not task.done():
                task.cancel()
            del cls._monitor_tasks[bash_id]

    @classmethod
    async def terminate(cls, bash_id: str) -> BackgroundShell:
        """Terminate a background shell and clean up all resources.

        Args:
            bash_id: The unique identifier of the background shell

        Returns:
            The terminated BackgroundShell object

        Raises:
            ValueError: If shell not found
        """
        shell = cls.get(bash_id)
        if not shell:
            raise ValueError(f"Shell not found: {bash_id}")

        # Terminate the process
        await shell.terminate()

        # Clean up monitoring and remove from manager
        cls._cancel_monitor(bash_id)
        cls._remove(bash_id)

        return shell

    @classmethod
    async def cleanup_all(cls) -> list[str]:
        """Clean up all background shells and resources.
        
        Returns:
            List of terminated shell IDs
        """
        terminated_ids = list(cls._shells.keys())
        
        # Terminate all shells
        for bash_id in terminated_ids:
            try:
                await cls.terminate(bash_id)
            except Exception:
                # Ignore errors during cleanup
                pass
        
        return terminated_ids

    @classmethod
    def get_stats(cls) -> dict:
        """Get statistics about managed shells.
        
        Returns:
            Dictionary with shell counts and status
        """
        running = sum(1 for s in cls._shells.values() if s.status == "running")
        completed = sum(1 for s in cls._shells.values() if s.status == "completed")
        failed = sum(1 for s in cls._shells.values() if s.status == "failed")
        
        return {
            "total": len(cls._shells),
            "running": running,
            "completed": completed,
            "failed": failed,
            "monitor_tasks": len(cls._monitor_tasks),
        }