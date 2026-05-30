"""后台Shell进程管理，支持线程安全访问。"""

from __future__ import annotations

import asyncio
import contextlib
import re
from threading import Lock as ThreadLock
from typing import Any


class BackgroundShell:
    """后台Shell数据容器。

    纯数据类，仅存储状态和输出。
    IO操作由BackgroundShellManager外部管理。
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
        self._output_lock = ThreadLock()  # 保护 output_lines 的并发访问

    def add_output(self, line: str) -> None:
        """添加新的输出行（线程安全）。"""
        with self._output_lock:
            self.output_lines.append(line)

    def get_new_output(self, filter_pattern: str | None = None) -> list[str]:
        """获取自上次检查以来的新输出，可选择按正则表达式过滤（线程安全）。"""
        with self._output_lock:
            new_lines = self.output_lines[self.last_read_index :]
            self.last_read_index = len(self.output_lines)

            if filter_pattern:
                try:
                    pattern = re.compile(filter_pattern)
                    new_lines = [line for line in new_lines if pattern.search(line)]
                except re.error:
                    # 无效的正则表达式，返回所有行
                    pass

            return new_lines

    def update_status(self, is_alive: bool, exit_code: int | None = None) -> None:
        """更新进程状态。"""
        if not is_alive:
            self.status = "completed" if exit_code == 0 else "failed"
            self.exit_code = exit_code
        else:
            self.status = "running"

    async def terminate(self) -> None:
        """终止后台进程。"""
        if self.process.returncode is None:
            self.process.terminate()
            try:
                await asyncio.wait_for(self.process.wait(), timeout=5)
            except asyncio.TimeoutError:
                self.process.kill()
        self.status = "terminated"
        self.exit_code = self.process.returncode


class BackgroundShellManager:
    """所有后台Shell进程的线程安全管理器。

    使用类级别存储和agent_id前缀以避免跨agent污染。
    每个agent在添加shell时应使用唯一的agent_id。

    线程安全：在所有访问共享状态（_shells、_monitor_tasks）的方法上使用Lock进行并发访问保护。
    """

    _lock = ThreadLock()
    _shells: dict[str, BackgroundShell] = {}
    _monitor_tasks: dict[str, asyncio.Task[Any]] = {}

    @classmethod
    def _prefix_key(cls, agent_id: str, bash_id: str) -> str:
        """生成用于Shell存储的前缀键。"""
        return f"{agent_id}:{bash_id}"

    @classmethod
    def add(cls, shell: BackgroundShell, agent_id: str = "default") -> None:
        """添加后台Shell到管理器（线程安全）。

        Args:
            shell: 要添加的BackgroundShell
            agent_id: agent的唯一标识符（用于隔离存储）
        """
        with cls._lock:
            key = cls._prefix_key(agent_id, shell.bash_id)
            cls._shells[key] = shell

    @classmethod
    def get(cls, bash_id: str, agent_id: str = "default") -> BackgroundShell | None:
        """根据ID获取后台Shell（线程安全）。

        Args:
            bash_id: Shell ID
            agent_id: 添加shell时使用的agent标识符
        """
        with cls._lock:
            key = cls._prefix_key(agent_id, bash_id)
            return cls._shells.get(key)

    @classmethod
    def get_available_ids(cls, agent_id: str = "default") -> list[str]:
        """获取某个agent的所有可用bash ID（线程安全）。

        Args:
            agent_id: 用于过滤shell的agent标识符
        """
        with cls._lock:
            prefix = f"{agent_id}:"
            return [k.replace(prefix, "") for k in cls._shells if k.startswith(prefix)]

    @classmethod
    def _remove(cls, bash_id: str, agent_id: str = "default") -> None:
        """从管理器中移除后台Shell（仅供内部使用，线程安全）。"""
        with cls._lock:
            key = cls._prefix_key(agent_id, bash_id)
            if key in cls._shells:
                del cls._shells[key]

    @classmethod
    async def start_monitor(cls, bash_id: str, agent_id: str = "default") -> None:
        """开始监控后台Shell的输出。"""
        shell = cls.get(bash_id, agent_id)
        if not shell:
            return

        async def monitor() -> None:
            try:
                process = shell.process
                # 持续读取输出直到进程结束
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

                # 进程已结束，等待退出码
                try:
                    returncode = await process.wait()
                except Exception:
                    returncode = -1

                shell.update_status(is_alive=False, exit_code=returncode)

            except Exception as e:
                key = cls._prefix_key(agent_id, bash_id)
                with cls._lock:
                    if key in cls._shells:
                        cls._shells[key].status = "error"
                        cls._shells[key].add_output(f"Monitor error: {str(e)}")
            finally:
                monitor_key = f"{agent_id}:monitor:{bash_id}"
                with cls._lock:
                    if monitor_key in cls._monitor_tasks:
                        del cls._monitor_tasks[monitor_key]

        task = asyncio.create_task(monitor())
        with cls._lock:
            monitor_key = f"{agent_id}:monitor:{bash_id}"
            cls._monitor_tasks[monitor_key] = task

    @classmethod
    def _cancel_monitor(cls, bash_id: str, agent_id: str = "default") -> None:
        """取消并移除监控任务（仅供内部使用，线程安全）。"""
        monitor_key = f"{agent_id}:monitor:{bash_id}"
        with cls._lock:
            if monitor_key in cls._monitor_tasks:
                task = cls._monitor_tasks[monitor_key]
                if not task.done():
                    task.cancel()

    @classmethod
    async def terminate(cls, bash_id: str, agent_id: str = "default") -> BackgroundShell:
        """终止后台Shell并清理所有资源（线程安全）。

        Args:
            bash_id: 后台Shell的唯一标识符
            agent_id: 添加shell时使用的agent标识符

        Returns:
            已终止的BackgroundShell对象

        Raises:
            ValueError: 如果shell未找到
        """
        shell = cls.get(bash_id, agent_id)
        if not shell:
            raise ValueError(f"Shell not found: {bash_id}")

        # 终止进程
        await shell.terminate()

        # 清理监控并从管理器中移除
        cls._cancel_monitor(bash_id, agent_id)
        cls._remove(bash_id, agent_id)

        return shell

    @classmethod
    async def cleanup_all(cls, agent_id: str = "default") -> list[str]:
        """清理某个agent的所有后台Shell（线程安全）。

        Args:
            agent_id: 要清理shell的agent标识符

        Returns:
            已终止的shell ID列表
        """
        terminated_ids = cls.get_available_ids(agent_id)

        # 终止该agent的所有shell
        for bash_id in terminated_ids:
            with contextlib.suppress(Exception):
                await cls.terminate(bash_id, agent_id)

        return terminated_ids

    @classmethod
    def get_stats(cls, agent_id: str = "default") -> dict[str, Any]:
        """获取某个agent管理的Shell统计数据（线程安全）。

        Args:
            agent_id: 要获取统计数据的agent标识符

        Returns:
            包含Shell数量和状态的字典
        """
        with cls._lock:
            prefix = f"{agent_id}:"
            agent_shells = {
                k: v
                for k, v in cls._shells.items()
                if k.startswith(prefix) and not k.startswith(f"{agent_id}:monitor:")
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

    @classmethod
    def clear_all(cls) -> None:
        """清除所有shell和监控（线程安全）。用于测试。"""
        with cls._lock:
            cls._shells.clear()
            for task in cls._monitor_tasks.values():
                if not task.done():
                    task.cancel()
            cls._monitor_tasks.clear()
