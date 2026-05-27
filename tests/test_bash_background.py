"""Tests for bash_background module - BackgroundShell and BackgroundShellManager."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mini_agent.tools.bash_background import BackgroundShell, BackgroundShellManager


def _make_process(returncode=None, pid=1234):
    proc = MagicMock(spec=asyncio.subprocess.Process)
    proc.returncode = returncode
    proc.pid = pid
    proc.terminate = MagicMock()
    proc.kill = MagicMock()
    proc.wait = AsyncMock(return_value=returncode if returncode is not None else 0)
    proc.stdout = None
    return proc


def _make_shell(bash_id="sh1", command="echo hello", returncode=None):
    proc = _make_process(returncode=returncode)
    shell = BackgroundShell(
        bash_id=bash_id,
        command=command,
        process=proc,
        start_time=time.time(),
    )
    return shell, proc


@pytest.fixture(autouse=True)
def _reset_manager():
    BackgroundShellManager._shells.clear()
    BackgroundShellManager._monitor_tasks.clear()
    yield
    BackgroundShellManager._shells.clear()
    BackgroundShellManager._monitor_tasks.clear()


class TestBackgroundShell:
    def test_init_defaults(self):
        proc = _make_process()
        shell = BackgroundShell(
            bash_id="id1",
            command="ls",
            process=proc,
            start_time=100.0,
        )
        assert shell.bash_id == "id1"
        assert shell.command == "ls"
        assert shell.process is proc
        assert shell.start_time == 100.0
        assert shell.output_lines == []
        assert shell.last_read_index == 0
        assert shell.status == "running"
        assert shell.exit_code is None

    def test_add_output(self):
        shell, _ = _make_shell()
        shell.add_output("line 1")
        shell.add_output("line 2")
        assert shell.output_lines == ["line 1", "line 2"]

    def test_get_new_output_returns_incremental(self):
        shell, _ = _make_shell()
        shell.add_output("a")
        shell.add_output("b")
        result = shell.get_new_output()
        assert result == ["a", "b"]
        assert shell.last_read_index == 2

        shell.add_output("c")
        result = shell.get_new_output()
        assert result == ["c"]
        assert shell.last_read_index == 3

    def test_get_new_output_empty_when_no_new(self):
        shell, _ = _make_shell()
        shell.add_output("x")
        shell.get_new_output()
        result = shell.get_new_output()
        assert result == []

    def test_get_new_output_with_filter(self):
        shell, _ = _make_shell()
        shell.add_output("error: something failed")
        shell.add_output("info: all good")
        shell.add_output("error: another failure")
        result = shell.get_new_output(filter_pattern="error")
        assert len(result) == 2
        assert "error: something failed" in result
        assert "error: another failure" in result

    def test_get_new_output_filter_regex(self):
        shell, _ = _make_shell()
        shell.add_output("Line 1")
        shell.add_output("Line 2")
        shell.add_output("Line 10")
        result = shell.get_new_output(filter_pattern=r"Line \d$")
        assert "Line 1" in result
        assert "Line 2" in result
        assert "Line 10" not in result

    def test_get_new_output_invalid_regex_returns_all(self):
        shell, _ = _make_shell()
        shell.add_output("hello")
        shell.add_output("world")
        result = shell.get_new_output(filter_pattern="[invalid")
        assert result == ["hello", "world"]

    def test_update_status_alive(self):
        shell, _ = _make_shell()
        shell.update_status(is_alive=True)
        assert shell.status == "running"
        assert shell.exit_code is None

    def test_update_status_completed(self):
        shell, _ = _make_shell()
        shell.update_status(is_alive=False, exit_code=0)
        assert shell.status == "completed"
        assert shell.exit_code == 0

    def test_update_status_failed(self):
        shell, _ = _make_shell()
        shell.update_status(is_alive=False, exit_code=1)
        assert shell.status == "failed"
        assert shell.exit_code == 1

    def test_update_status_failed_negative_exit_code(self):
        shell, _ = _make_shell()
        shell.update_status(is_alive=False, exit_code=-15)
        assert shell.status == "failed"
        assert shell.exit_code == -15

    @pytest.mark.asyncio
    async def test_terminate_running_process(self):
        shell, proc = _make_shell(returncode=None)
        proc.wait = AsyncMock(return_value=0)

        async def fake_wait_with_returncode(*args, **kwargs):
            proc.returncode = 0
            return 0

        proc.wait = fake_wait_with_returncode
        await shell.terminate()
        proc.terminate.assert_called_once()
        assert shell.status == "terminated"
        assert shell.exit_code == 0

    @pytest.mark.asyncio
    async def test_terminate_already_ended_process(self):
        shell, proc = _make_shell(returncode=0)
        await shell.terminate()
        proc.terminate.assert_not_called()
        assert shell.status == "terminated"
        assert shell.exit_code == 0

    @pytest.mark.asyncio
    async def test_terminate_timeout_kills(self):
        shell, proc = _make_shell(returncode=None)
        proc.wait = AsyncMock(side_effect=asyncio.TimeoutError)
        await shell.terminate()
        proc.terminate.assert_called_once()
        proc.kill.assert_called_once()
        assert shell.status == "terminated"


class TestBackgroundShellManagerPrefixKey:
    def test_prefix_key_default(self):
        key = BackgroundShellManager._prefix_key("default", "sh1")
        assert key == "default:sh1"

    def test_prefix_key_custom_agent(self):
        key = BackgroundShellManager._prefix_key("agent-42", "sh1")
        assert key == "agent-42:sh1"


class TestBackgroundShellManagerAddGet:
    def test_add_and_get(self):
        shell, _ = _make_shell(bash_id="sh1")
        BackgroundShellManager.add(shell, agent_id="test")
        result = BackgroundShellManager.get("sh1", agent_id="test")
        assert result is shell

    def test_get_nonexistent(self):
        result = BackgroundShellManager.get("nope", agent_id="test")
        assert result is None

    def test_agent_isolation(self):
        shell_a, _ = _make_shell(bash_id="sh1")
        shell_b, _ = _make_shell(bash_id="sh1")
        BackgroundShellManager.add(shell_a, agent_id="agent-a")
        BackgroundShellManager.add(shell_b, agent_id="agent-b")
        assert BackgroundShellManager.get("sh1", agent_id="agent-a") is shell_a
        assert BackgroundShellManager.get("sh1", agent_id="agent-b") is shell_b

    def test_default_agent_id(self):
        shell, _ = _make_shell(bash_id="sh1")
        BackgroundShellManager.add(shell)
        result = BackgroundShellManager.get("sh1")
        assert result is shell


class TestBackgroundShellManagerGetAvailableIds:
    def test_empty(self):
        assert BackgroundShellManager.get_available_ids("test") == []

    def test_returns_ids_for_agent(self):
        for i in range(3):
            shell, _ = _make_shell(bash_id=f"sh{i}")
            BackgroundShellManager.add(shell, agent_id="test")
        ids = BackgroundShellManager.get_available_ids("test")
        assert sorted(ids) == ["sh0", "sh1", "sh2"]

    def test_filters_by_agent(self):
        shell_a, _ = _make_shell(bash_id="sh-a")
        shell_b, _ = _make_shell(bash_id="sh-b")
        BackgroundShellManager.add(shell_a, agent_id="agent-a")
        BackgroundShellManager.add(shell_b, agent_id="agent-b")
        assert BackgroundShellManager.get_available_ids("agent-a") == ["sh-a"]
        assert BackgroundShellManager.get_available_ids("agent-b") == ["sh-b"]


class TestBackgroundShellManagerRemove:
    def test_remove_existing(self):
        shell, _ = _make_shell(bash_id="sh1")
        BackgroundShellManager.add(shell, agent_id="test")
        BackgroundShellManager._remove("sh1", agent_id="test")
        assert BackgroundShellManager.get("sh1", agent_id="test") is None

    def test_remove_nonexistent_no_error(self):
        BackgroundShellManager._remove("nope", agent_id="test")


class TestBackgroundShellManagerTerminate:
    @pytest.mark.asyncio
    async def test_terminate_shell(self):
        shell, proc = _make_shell(bash_id="sh1", returncode=None)

        async def fake_wait_with_returncode(*args, **kwargs):
            proc.returncode = 0
            return 0

        proc.wait = fake_wait_with_returncode
        BackgroundShellManager.add(shell, agent_id="test")
        result = await BackgroundShellManager.terminate("sh1", agent_id="test")
        assert result is shell
        assert shell.status == "terminated"
        assert BackgroundShellManager.get("sh1", agent_id="test") is None

    @pytest.mark.asyncio
    async def test_terminate_not_found_raises(self):
        with pytest.raises(ValueError, match="Shell not found"):
            await BackgroundShellManager.terminate("nope", agent_id="test")

    @pytest.mark.asyncio
    async def test_terminate_cancels_monitor(self):
        shell, proc = _make_shell(bash_id="sh1", returncode=None)

        async def fake_wait_with_returncode(*args, **kwargs):
            proc.returncode = 0
            return 0

        proc.wait = fake_wait_with_returncode
        BackgroundShellManager.add(shell, agent_id="test")

        mock_task = MagicMock()
        mock_task.done.return_value = False
        BackgroundShellManager._monitor_tasks["test:monitor:sh1"] = mock_task

        await BackgroundShellManager.terminate("sh1", agent_id="test")
        mock_task.cancel.assert_called_once()


class TestBackgroundShellManagerCleanupAll:
    @pytest.mark.asyncio
    async def test_cleanup_all_terminates_and_removes(self):
        shells = []
        for i in range(3):
            shell, proc = _make_shell(bash_id=f"sh{i}", returncode=None)

            async def _make_wait(p):
                async def fw(*args, **kwargs):
                    p.returncode = 0
                    return 0

                return fw

            proc.wait = await _make_wait(proc)
            BackgroundShellManager.add(shell, agent_id="test")
            shells.append(shell)

        terminated_ids = await BackgroundShellManager.cleanup_all("test")
        assert sorted(terminated_ids) == ["sh0", "sh1", "sh2"]
        for shell in shells:
            assert shell.status == "terminated"
        assert BackgroundShellManager.get_available_ids("test") == []

    @pytest.mark.asyncio
    async def test_cleanup_all_does_not_affect_other_agents(self):
        shell_a, proc_a = _make_shell(bash_id="sh-a", returncode=None)

        async def fake_wait_a(*args, **kwargs):
            proc_a.returncode = 0
            return 0

        proc_a.wait = fake_wait_a
        shell_b, _ = _make_shell(bash_id="sh-b", returncode=None)
        BackgroundShellManager.add(shell_a, agent_id="agent-a")
        BackgroundShellManager.add(shell_b, agent_id="agent-b")

        await BackgroundShellManager.cleanup_all("agent-a")
        assert BackgroundShellManager.get("sh-b", agent_id="agent-b") is shell_b

    @pytest.mark.asyncio
    async def test_cleanup_all_empty(self):
        terminated_ids = await BackgroundShellManager.cleanup_all("test")
        assert terminated_ids == []

    @pytest.mark.asyncio
    async def test_cleanup_all_suppresses_termination_errors(self):
        shell, proc = _make_shell(bash_id="sh1", returncode=None)
        proc.wait = AsyncMock(side_effect=RuntimeError("boom"))
        BackgroundShellManager.add(shell, agent_id="test")

        terminated_ids = await BackgroundShellManager.cleanup_all("test")
        assert "sh1" in terminated_ids


class TestBackgroundShellManagerGetStats:
    def test_empty_stats(self):
        stats = BackgroundShellManager.get_stats("test")
        assert stats == {"total": 0, "running": 0, "completed": 0, "failed": 0}

    def test_mixed_statuses(self):
        shell_r, _ = _make_shell(bash_id="sh1")
        shell_c, _ = _make_shell(bash_id="sh2")
        shell_c.status = "completed"
        shell_c.exit_code = 0
        shell_f, _ = _make_shell(bash_id="sh3")
        shell_f.status = "failed"
        shell_f.exit_code = 1

        BackgroundShellManager.add(shell_r, agent_id="test")
        BackgroundShellManager.add(shell_c, agent_id="test")
        BackgroundShellManager.add(shell_f, agent_id="test")

        stats = BackgroundShellManager.get_stats("test")
        assert stats == {"total": 3, "running": 1, "completed": 1, "failed": 1}

    def test_stats_filters_by_agent(self):
        shell_a, _ = _make_shell(bash_id="sh-a")
        shell_b, _ = _make_shell(bash_id="sh-b")
        BackgroundShellManager.add(shell_a, agent_id="agent-a")
        BackgroundShellManager.add(shell_b, agent_id="agent-b")

        stats_a = BackgroundShellManager.get_stats("agent-a")
        assert stats_a["total"] == 1
        stats_b = BackgroundShellManager.get_stats("agent-b")
        assert stats_b["total"] == 1

    def test_stats_excludes_monitor_keys(self):
        shell, _ = _make_shell(bash_id="sh1")
        BackgroundShellManager.add(shell, agent_id="test")
        BackgroundShellManager._shells["test:monitor:sh1"] = shell

        stats = BackgroundShellManager.get_stats("test")
        assert stats["total"] == 1


class TestBackgroundShellManagerCancelMonitor:
    def test_cancel_monitor_cancels_running_task(self):
        mock_task = MagicMock()
        mock_task.done.return_value = False
        BackgroundShellManager._monitor_tasks["test:monitor:sh1"] = mock_task

        BackgroundShellManager._cancel_monitor("sh1", agent_id="test")
        mock_task.cancel.assert_called_once()

    def test_cancel_monitor_skips_done_task(self):
        mock_task = MagicMock()
        mock_task.done.return_value = True
        BackgroundShellManager._monitor_tasks["test:monitor:sh1"] = mock_task

        BackgroundShellManager._cancel_monitor("sh1", agent_id="test")
        mock_task.cancel.assert_not_called()

    def test_cancel_monitor_no_existing_task(self):
        BackgroundShellManager._cancel_monitor("nope", agent_id="test")


class TestBackgroundShellManagerStartMonitor:
    @pytest.mark.asyncio
    async def test_start_monitor_shell_not_found(self):
        await BackgroundShellManager.start_monitor("nope", agent_id="test")
        assert len(BackgroundShellManager._monitor_tasks) == 0

    @pytest.mark.asyncio
    async def test_start_monitor_creates_task(self):
        shell, proc = _make_shell(bash_id="sh1", returncode=None)

        async def fake_readline():
            proc.returncode = 0
            return b""

        stdout_mock = MagicMock()
        stdout_mock.readline = fake_readline
        proc.stdout = stdout_mock
        proc.wait = AsyncMock(return_value=0)

        BackgroundShellManager.add(shell, agent_id="test")
        await BackgroundShellManager.start_monitor("sh1", agent_id="test")

        monitor_key = "test:monitor:sh1"
        assert monitor_key in BackgroundShellManager._monitor_tasks

    @pytest.mark.asyncio
    async def test_start_monitor_reads_stdout_lines(self):
        shell, proc = _make_shell(bash_id="sh1", returncode=None)

        readline_calls = 0

        async def fake_readline():
            nonlocal readline_calls
            readline_calls += 1
            if readline_calls == 1:
                return b"hello world\n"
            if readline_calls == 2:
                return b"second line\n"
            proc.returncode = 0
            return b""

        stdout_mock = MagicMock()
        stdout_mock.readline = fake_readline
        proc.stdout = stdout_mock
        proc.wait = AsyncMock(return_value=0)

        BackgroundShellManager.add(shell, agent_id="test")
        await BackgroundShellManager.start_monitor("sh1", agent_id="test")

        for _ in range(20):
            await asyncio.sleep(0.05)
            if len(shell.output_lines) >= 2:
                break

        assert "hello world" in shell.output_lines
        assert "second line" in shell.output_lines

    @pytest.mark.asyncio
    async def test_start_monitor_updates_status_on_exit(self):
        shell, proc = _make_shell(bash_id="sh1", returncode=None)

        async def fake_readline():
            proc.returncode = 0
            return b""

        stdout_mock = MagicMock()
        stdout_mock.readline = fake_readline
        proc.stdout = stdout_mock
        proc.wait = AsyncMock(return_value=0)

        BackgroundShellManager.add(shell, agent_id="test")
        await BackgroundShellManager.start_monitor("sh1", agent_id="test")

        for _ in range(20):
            await asyncio.sleep(0.05)
            if shell.status != "running":
                break

        assert shell.status == "completed"
        assert shell.exit_code == 0

    @pytest.mark.asyncio
    async def test_start_monitor_handles_wait_exception(self):
        shell, proc = _make_shell(bash_id="sh1", returncode=None)

        async def fake_readline():
            proc.returncode = -1
            return b""

        stdout_mock = MagicMock()
        stdout_mock.readline = fake_readline
        proc.stdout = stdout_mock
        proc.wait = AsyncMock(side_effect=RuntimeError("wait failed"))

        BackgroundShellManager.add(shell, agent_id="test")
        await BackgroundShellManager.start_monitor("sh1", agent_id="test")

        for _ in range(20):
            await asyncio.sleep(0.05)
            if shell.status != "running":
                break

        assert shell.status == "failed"
        assert shell.exit_code == -1

    @pytest.mark.asyncio
    async def test_start_monitor_handles_readline_timeout(self):
        shell, proc = _make_shell(bash_id="sh1", returncode=None)

        readline_calls = 0

        async def fake_readline():
            nonlocal readline_calls
            readline_calls += 1
            if readline_calls <= 2:
                raise asyncio.TimeoutError
            proc.returncode = 0
            return b"final line\n"

        stdout_mock = MagicMock()
        stdout_mock.readline = fake_readline
        proc.stdout = stdout_mock
        proc.wait = AsyncMock(return_value=0)

        BackgroundShellManager.add(shell, agent_id="test")
        await BackgroundShellManager.start_monitor("sh1", agent_id="test")

        for _ in range(30):
            await asyncio.sleep(0.05)
            if shell.status != "running":
                break

        assert shell.status == "completed"

    @pytest.mark.asyncio
    async def test_start_monitor_handles_readline_exception(self):
        shell, proc = _make_shell(bash_id="sh1", returncode=None)

        readline_calls = 0

        async def fake_readline():
            nonlocal readline_calls
            readline_calls += 1
            if readline_calls == 1:
                raise OSError("read error")
            proc.returncode = 0
            return b""

        stdout_mock = MagicMock()
        stdout_mock.readline = fake_readline
        proc.stdout = stdout_mock
        proc.wait = AsyncMock(return_value=0)

        BackgroundShellManager.add(shell, agent_id="test")
        await BackgroundShellManager.start_monitor("sh1", agent_id="test")

        for _ in range(30):
            await asyncio.sleep(0.05)
            if shell.status != "running":
                break

        assert shell.status == "completed"

    @pytest.mark.asyncio
    async def test_start_monitor_cleans_up_task_on_completion(self):
        shell, proc = _make_shell(bash_id="sh1", returncode=None)

        async def fake_readline():
            proc.returncode = 0
            return b""

        stdout_mock = MagicMock()
        stdout_mock.readline = fake_readline
        proc.stdout = stdout_mock
        proc.wait = AsyncMock(return_value=0)

        BackgroundShellManager.add(shell, agent_id="test")
        await BackgroundShellManager.start_monitor("sh1", agent_id="test")

        monitor_key = "test:monitor:sh1"
        for _ in range(30):
            await asyncio.sleep(0.05)
            if monitor_key not in BackgroundShellManager._monitor_tasks:
                break

        assert monitor_key not in BackgroundShellManager._monitor_tasks

    @pytest.mark.asyncio
    async def test_start_monitor_sets_error_on_outer_exception(self):
        shell, proc = _make_shell(bash_id="sh1", returncode=None)
        proc.stdout = MagicMock()

        BackgroundShellManager.add(shell, agent_id="test")

        sleep_call_count = 0
        original_sleep = asyncio.sleep

        async def selective_sleep(delay):
            nonlocal sleep_call_count
            sleep_call_count += 1
            if sleep_call_count <= 5:
                raise RuntimeError("catastrophic")
            await original_sleep(delay)

        with (
            patch(
                "mini_agent.tools.bash_background.asyncio.wait_for",
                side_effect=RuntimeError("catastrophic"),
            ),
            patch(
                "mini_agent.tools.bash_background.asyncio.sleep",
                side_effect=selective_sleep,
            ),
        ):
            await BackgroundShellManager.start_monitor("sh1", agent_id="test")

            for _ in range(30):
                await original_sleep(0.05)
                if shell.status == "error":
                    break

        assert shell.status == "error"
        assert any("Monitor error" in line for line in shell.output_lines)

    @pytest.mark.asyncio
    async def test_start_monitor_with_no_stdout_exits_on_returncode(self):
        shell, proc = _make_shell(bash_id="sh1", returncode=0)
        proc.stdout = None
        proc.wait = AsyncMock(return_value=0)

        BackgroundShellManager.add(shell, agent_id="test")
        await BackgroundShellManager.start_monitor("sh1", agent_id="test")

        for _ in range(30):
            await asyncio.sleep(0.05)
            if shell.status != "running":
                break

        assert shell.status == "completed"
