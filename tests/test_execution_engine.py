import asyncio
from unittest.mock import MagicMock, patch

import pytest

from mini_agent.core.execution_engine import ExecutionEngine
from mini_agent.core.rate_limiter import RateLimiter
from mini_agent.schema import AgentMode, FunctionCall, ToolCall
from mini_agent.tools.base import Tool, ToolResult


class MockTool(Tool):
    def __init__(self, name: str, result: ToolResult | None = None):
        self._name = name
        self._result = result or ToolResult(success=True, content="mock result")

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return f"Mock {self._name} tool"

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {}}

    async def execute(self, **kwargs) -> ToolResult:
        return self._result


class FailingTool(Tool):
    def __init__(self, name: str):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return f"Failing {self._name} tool"

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {}}

    async def execute(self, **kwargs) -> ToolResult:
        raise RuntimeError("Tool execution failed")


def _make_engine(**overrides):
    defaults = dict(
        tools={},
        logger=MagicMock(),
        retry_handler=MagicMock(),
        metrics=MagicMock(),
        error_recovery=MagicMock(),
        write_tools=frozenset(),
    )
    defaults.update(overrides)
    return ExecutionEngine(**defaults)


class TestExecutionEngineInit:
    def test_init_with_tools(self):
        tools = {"read_file": MockTool("read_file")}
        engine = ExecutionEngine(
            tools=tools,
            logger=MagicMock(),
            retry_handler=MagicMock(),
            metrics=MagicMock(),
            error_recovery=MagicMock(),
            write_tools=frozenset(),
        )
        assert "read_file" in engine.tools

    def test_init_empty_tools(self):
        engine = ExecutionEngine(
            tools={},
            logger=MagicMock(),
            retry_handler=MagicMock(),
            metrics=MagicMock(),
            error_recovery=MagicMock(),
            write_tools=frozenset(),
        )
        assert len(engine.tools) == 0

    def test_init_with_rate_limiter(self):
        rl = MagicMock(spec=RateLimiter)
        engine = ExecutionEngine(
            tools={},
            logger=MagicMock(),
            retry_handler=MagicMock(),
            metrics=MagicMock(),
            error_recovery=MagicMock(),
            write_tools=frozenset(),
            rate_limiter=rl,
        )
        assert engine._rate_limiter is rl

    def test_init_without_rate_limiter(self):
        engine = _make_engine()
        assert engine._rate_limiter is None


class TestExecuteToolsSequential:
    @pytest.mark.asyncio
    async def test_single_tool_call(self):
        tool = MockTool("read_file")
        engine = _make_engine(tools={"read_file": tool})
        tool_call = ToolCall(
            id="tc-1",
            type="function",
            function=FunctionCall(name="read_file", arguments={}),
        )
        check_fn = MagicMock(return_value=True)
        results = await engine.execute_tools(
            tool_calls=[tool_call],
            max_concurrent=1,
            parallel_enabled=False,
            mode=AgentMode.YOLO,
            check_approved_fn=check_fn,
        )
        assert len(results) == 1
        tc, msg = results[0]
        assert msg.role == "tool"

    @pytest.mark.asyncio
    async def test_unknown_tool(self):
        engine = _make_engine()
        tool_call = ToolCall(
            id="tc-1",
            type="function",
            function=FunctionCall(name="nonexistent", arguments={}),
        )
        results = await engine.execute_tools(
            tool_calls=[tool_call],
            max_concurrent=1,
            parallel_enabled=False,
            mode=AgentMode.YOLO,
            check_approved_fn=MagicMock(return_value=True),
        )
        assert len(results) == 1
        _, msg = results[0]
        assert "Error" in msg.content or "Unknown" in msg.content

    @pytest.mark.asyncio
    async def test_plan_mode_blocks_write_tool(self):
        tool = MockTool("write_file")
        engine = _make_engine(tools={"write_file": tool}, write_tools=frozenset({"write_file"}))
        tool_call = ToolCall(
            id="tc-1",
            type="function",
            function=FunctionCall(name="write_file", arguments={}),
        )
        results = await engine.execute_tools(
            tool_calls=[tool_call],
            max_concurrent=1,
            parallel_enabled=False,
            mode=AgentMode.PLAN,
            check_approved_fn=MagicMock(return_value=True),
        )
        assert len(results) == 1
        _, msg = results[0]
        assert "plan" in msg.content.lower() or "blocked" in msg.content.lower()

    @pytest.mark.asyncio
    async def test_empty_tool_calls_returns_empty(self):
        engine = _make_engine()
        results = await engine.execute_tools(
            tool_calls=[],
            max_concurrent=1,
            parallel_enabled=False,
            mode=AgentMode.YOLO,
            check_approved_fn=MagicMock(return_value=True),
        )
        assert results == []

    @pytest.mark.asyncio
    async def test_parallel_enabled_single_tool_falls_to_sequential(self):
        tool = MockTool("read_file")
        engine = _make_engine(tools={"read_file": tool})
        tool_call = ToolCall(
            id="tc-1",
            type="function",
            function=FunctionCall(name="read_file", arguments={}),
        )
        results = await engine.execute_tools(
            tool_calls=[tool_call],
            max_concurrent=5,
            parallel_enabled=True,
            mode=AgentMode.YOLO,
            check_approved_fn=MagicMock(return_value=True),
        )
        assert len(results) == 1
        _, msg = results[0]
        assert msg.role == "tool"
        assert "mock result" in msg.content


class TestExecuteToolsParallel:
    @pytest.mark.asyncio
    async def test_parallel_execution(self):
        tools = {
            "read_file": MockTool("read_file"),
            "search": MockTool("search"),
        }
        engine = _make_engine(tools=tools)
        tool_calls = [
            ToolCall(id="tc-1", type="function", function=FunctionCall(name="read_file", arguments={})),
            ToolCall(id="tc-2", type="function", function=FunctionCall(name="search", arguments={})),
        ]
        results = await engine.execute_tools(
            tool_calls=tool_calls,
            max_concurrent=5,
            parallel_enabled=True,
            mode=AgentMode.YOLO,
            check_approved_fn=MagicMock(return_value=True),
        )
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_agent_mode_approval(self):
        tool = MockTool("read_file")
        engine = _make_engine(tools={"read_file": tool})
        tool_call = ToolCall(
            id="tc-1",
            type="function",
            function=FunctionCall(name="read_file", arguments={}),
        )
        check_fn = MagicMock(return_value=False)
        results = await engine.execute_tools(
            tool_calls=[tool_call],
            max_concurrent=1,
            parallel_enabled=False,
            mode=AgentMode.AGENT,
            check_approved_fn=check_fn,
        )
        assert len(results) == 1
        _, msg = results[0]
        assert (
            "denied" in msg.content.lower()
            or "rejected" in msg.content.lower()
            or "not approved" in msg.content.lower()
        )

    @pytest.mark.asyncio
    async def test_parallel_with_multiple_read_tools(self):
        tools = {
            "read_file": MockTool("read_file"),
            "grep": MockTool("grep"),
            "tree": MockTool("tree"),
        }
        engine = _make_engine(tools=tools)
        tool_calls = [
            ToolCall(id="tc-1", type="function", function=FunctionCall(name="read_file", arguments={})),
            ToolCall(id="tc-2", type="function", function=FunctionCall(name="grep", arguments={})),
            ToolCall(id="tc-3", type="function", function=FunctionCall(name="tree", arguments={})),
        ]
        results = await engine.execute_tools(
            tool_calls=tool_calls,
            max_concurrent=5,
            parallel_enabled=True,
            mode=AgentMode.YOLO,
            check_approved_fn=MagicMock(return_value=True),
        )
        assert len(results) == 3
        for _, msg in results:
            assert msg.role == "tool"

    @pytest.mark.asyncio
    async def test_parallel_exception_in_tool_captured(self):
        failing_tool = FailingTool("failing_tool")
        ok_tool = MockTool("ok_tool")
        engine = _make_engine(tools={"failing_tool": failing_tool, "ok_tool": ok_tool})
        retry_handler = engine._retry_handler
        retry_handler.get_max_retries.return_value = 1
        retry_handler.is_transient_error.return_value = False
        retry_handler.get_delay.return_value = 0

        tool_calls = [
            ToolCall(id="tc-1", type="function", function=FunctionCall(name="failing_tool", arguments={})),
            ToolCall(id="tc-2", type="function", function=FunctionCall(name="ok_tool", arguments={})),
        ]
        results = await engine.execute_tools(
            tool_calls=tool_calls,
            max_concurrent=5,
            parallel_enabled=True,
            mode=AgentMode.YOLO,
            check_approved_fn=MagicMock(return_value=True),
        )
        assert len(results) == 2
        failing_msgs = [msg for tc, msg in results if tc.function.name == "failing_tool"]
        ok_msgs = [msg for tc, msg in results if tc.function.name == "ok_tool"]
        assert len(failing_msgs) == 1
        assert "Error" in failing_msgs[0].content
        assert len(ok_msgs) == 1
        assert "mock result" in ok_msgs[0].content


class TestExecuteBatched:
    @pytest.mark.asyncio
    async def test_batched_with_write_conflict(self):
        tools = {
            "write_file": MockTool("write_file"),
            "edit_file": MockTool("edit_file"),
        }
        engine = _make_engine(tools=tools, write_tools=frozenset({"write_file", "edit_file"}))
        retry_handler = engine._retry_handler
        retry_handler.get_max_retries.return_value = 1
        retry_handler.is_transient_error.return_value = False
        retry_handler.get_delay.return_value = 0

        tool_calls = [
            ToolCall(
                id="tc-1",
                type="function",
                function=FunctionCall(name="write_file", arguments={"path": "/same.py"}),
            ),
            ToolCall(
                id="tc-2",
                type="function",
                function=FunctionCall(name="edit_file", arguments={"path": "/same.py"}),
            ),
        ]
        results = await engine.execute_tools(
            tool_calls=tool_calls,
            max_concurrent=5,
            parallel_enabled=True,
            mode=AgentMode.YOLO,
            check_approved_fn=MagicMock(return_value=True),
        )
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_batched_mixed_read_and_write(self):
        tools = {
            "read_file": MockTool("read_file"),
            "write_file": MockTool("write_file"),
        }
        engine = _make_engine(tools=tools, write_tools=frozenset({"write_file"}))
        retry_handler = engine._retry_handler
        retry_handler.get_max_retries.return_value = 1
        retry_handler.is_transient_error.return_value = False
        retry_handler.get_delay.return_value = 0

        tool_calls = [
            ToolCall(
                id="tc-1",
                type="function",
                function=FunctionCall(name="read_file", arguments={"path": "/a.py"}),
            ),
            ToolCall(
                id="tc-2",
                type="function",
                function=FunctionCall(name="write_file", arguments={"path": "/b.py"}),
            ),
        ]
        results = await engine.execute_tools(
            tool_calls=tool_calls,
            max_concurrent=5,
            parallel_enabled=True,
            mode=AgentMode.YOLO,
            check_approved_fn=MagicMock(return_value=True),
        )
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_batched_single_item_batch_uses_sequential(self):
        tools = {
            "write_file": MockTool("write_file"),
        }
        engine = _make_engine(tools=tools, write_tools=frozenset({"write_file"}))
        retry_handler = engine._retry_handler
        retry_handler.get_max_retries.return_value = 1
        retry_handler.is_transient_error.return_value = False
        retry_handler.get_delay.return_value = 0

        tool_calls = [
            ToolCall(
                id="tc-1",
                type="function",
                function=FunctionCall(name="write_file", arguments={"path": "/a.py"}),
            ),
        ]
        results = await engine.execute_tools(
            tool_calls=tool_calls,
            max_concurrent=5,
            parallel_enabled=True,
            mode=AgentMode.YOLO,
            check_approved_fn=MagicMock(return_value=True),
        )
        assert len(results) == 1
        _, msg = results[0]
        assert msg.role == "tool"


class TestExecuteSingleToolRateLimiter:
    @pytest.mark.asyncio
    async def test_rate_limiter_blocks_when_not_allowed(self):
        rl = MagicMock(spec=RateLimiter)
        rl.check_rate.return_value = (False, "Rate limit for 'bash' exceeded")
        rl.validate_input_length.return_value = (True, "")
        engine = _make_engine(tools={"bash": MockTool("bash")}, rate_limiter=rl)

        tool_call = ToolCall(
            id="tc-1",
            type="function",
            function=FunctionCall(name="bash", arguments={"cmd": "ls"}),
        )
        tc, msg = await engine._execute_single_tool(tool_call, AgentMode.YOLO, MagicMock(return_value=True))
        assert "Rate limit" in msg.content
        assert msg.tool_call_id == "tc-1"
        assert msg.name == "bash"
        rl.check_rate.assert_called_once_with("bash")

    @pytest.mark.asyncio
    async def test_rate_limiter_rejects_long_input(self):
        rl = MagicMock(spec=RateLimiter)
        rl.check_rate.return_value = (True, "")
        rl.validate_input_length.return_value = (
            False,
            "Input too long for 'bash.cmd': 5000 chars exceeds limit of 1000.",
        )
        engine = _make_engine(tools={"bash": MockTool("bash")}, rate_limiter=rl)

        tool_call = ToolCall(
            id="tc-2",
            type="function",
            function=FunctionCall(name="bash", arguments={"cmd": "a" * 5000}),
        )
        tc, msg = await engine._execute_single_tool(tool_call, AgentMode.YOLO, MagicMock(return_value=True))
        assert "too long" in msg.content.lower()
        assert msg.tool_call_id == "tc-2"
        rl.validate_input_length.assert_called_once_with("bash", {"cmd": "a" * 5000})

    @pytest.mark.asyncio
    async def test_rate_limiter_allows_when_ok(self):
        rl = MagicMock(spec=RateLimiter)
        rl.check_rate.return_value = (True, "")
        rl.validate_input_length.return_value = (True, "")
        tool = MockTool("bash")
        engine = _make_engine(tools={"bash": tool}, rate_limiter=rl)
        engine._retry_handler.get_max_retries.return_value = 1
        engine._retry_handler.is_transient_error.return_value = False
        engine._retry_handler.get_delay.return_value = 0

        tool_call = ToolCall(
            id="tc-1",
            type="function",
            function=FunctionCall(name="bash", arguments={"cmd": "ls"}),
        )
        tc, msg = await engine._execute_single_tool(tool_call, AgentMode.YOLO, MagicMock(return_value=True))
        assert "mock result" in msg.content

    @pytest.mark.asyncio
    async def test_no_rate_limiter_skips_check(self):
        engine = _make_engine(tools={"bash": MockTool("bash")})
        engine._retry_handler.get_max_retries.return_value = 1
        engine._retry_handler.is_transient_error.return_value = False
        engine._retry_handler.get_delay.return_value = 0

        tool_call = ToolCall(
            id="tc-1",
            type="function",
            function=FunctionCall(name="bash", arguments={"cmd": "ls"}),
        )
        tc, msg = await engine._execute_single_tool(tool_call, AgentMode.YOLO, MagicMock(return_value=True))
        assert "mock result" in msg.content


class TestExecuteSingleToolPlanMode:
    @pytest.mark.asyncio
    async def test_plan_mode_blocks_write_tool(self):
        tool = MockTool("write_file")
        engine = _make_engine(tools={"write_file": tool}, write_tools=frozenset({"write_file"}))
        tool_call = ToolCall(
            id="tc-1",
            type="function",
            function=FunctionCall(name="write_file", arguments={"path": "test.txt"}),
        )
        tc, msg = await engine._execute_single_tool(tool_call, AgentMode.PLAN, MagicMock(return_value=True))
        assert "PLAN" in msg.content or "plan" in msg.content.lower()
        assert "blocked" in msg.content.lower()

    @pytest.mark.asyncio
    async def test_plan_mode_allows_read_tool(self):
        tool = MockTool("read_file")
        engine = _make_engine(tools={"read_file": tool}, write_tools=frozenset({"write_file"}))
        engine._retry_handler.get_max_retries.return_value = 1
        engine._retry_handler.is_transient_error.return_value = False
        engine._retry_handler.get_delay.return_value = 0

        tool_call = ToolCall(
            id="tc-1",
            type="function",
            function=FunctionCall(name="read_file", arguments={"path": "test.txt"}),
        )
        tc, msg = await engine._execute_single_tool(tool_call, AgentMode.PLAN, MagicMock(return_value=True))
        assert "mock result" in msg.content

    @pytest.mark.asyncio
    async def test_plan_mode_blocks_bash(self):
        tool = MockTool("bash")
        engine = _make_engine(tools={"bash": tool}, write_tools=frozenset({"bash"}))
        tool_call = ToolCall(
            id="tc-1",
            type="function",
            function=FunctionCall(name="bash", arguments={"cmd": "rm -rf /"}),
        )
        tc, msg = await engine._execute_single_tool(tool_call, AgentMode.PLAN, MagicMock(return_value=True))
        assert "blocked" in msg.content.lower()


class TestExecuteSingleToolAgentMode:
    @pytest.mark.asyncio
    async def test_agent_mode_rejects_unapproved_tool(self):
        tool = MockTool("bash")
        engine = _make_engine(tools={"bash": tool})
        tool_call = ToolCall(
            id="tc-1",
            type="function",
            function=FunctionCall(name="bash", arguments={"cmd": "ls"}),
        )
        check_fn = MagicMock(return_value=False)
        tc, msg = await engine._execute_single_tool(tool_call, AgentMode.AGENT, check_fn)
        assert "rejected" in msg.content.lower()
        check_fn.assert_called_once_with("bash")

    @pytest.mark.asyncio
    async def test_agent_mode_allows_approved_tool(self):
        tool = MockTool("bash")
        engine = _make_engine(tools={"bash": tool})
        engine._retry_handler.get_max_retries.return_value = 1
        engine._retry_handler.is_transient_error.return_value = False
        engine._retry_handler.get_delay.return_value = 0

        tool_call = ToolCall(
            id="tc-1",
            type="function",
            function=FunctionCall(name="bash", arguments={"cmd": "ls"}),
        )
        check_fn = MagicMock(return_value=True)
        tc, msg = await engine._execute_single_tool(tool_call, AgentMode.AGENT, check_fn)
        assert "mock result" in msg.content

    @pytest.mark.asyncio
    async def test_yolo_mode_skips_approval(self):
        tool = MockTool("bash")
        engine = _make_engine(tools={"bash": tool})
        engine._retry_handler.get_max_retries.return_value = 1
        engine._retry_handler.is_transient_error.return_value = False
        engine._retry_handler.get_delay.return_value = 0

        tool_call = ToolCall(
            id="tc-1",
            type="function",
            function=FunctionCall(name="bash", arguments={"cmd": "ls"}),
        )
        check_fn = MagicMock(return_value=False)
        tc, msg = await engine._execute_single_tool(tool_call, AgentMode.YOLO, check_fn)
        assert "mock result" in msg.content
        check_fn.assert_not_called()


class TestExecuteSingleToolUnknownTool:
    @pytest.mark.asyncio
    async def test_unknown_tool_returns_error(self):
        engine = _make_engine()
        tool_call = ToolCall(
            id="tc-1",
            type="function",
            function=FunctionCall(name="nonexistent", arguments={}),
        )
        tc, msg = await engine._execute_single_tool(tool_call, AgentMode.YOLO, MagicMock(return_value=True))
        assert "Unknown tool" in msg.content
        assert msg.tool_call_id == "tc-1"
        assert msg.name == "nonexistent"

    @pytest.mark.asyncio
    async def test_unknown_tool_records_failure(self):
        engine = _make_engine()
        tool_call = ToolCall(
            id="tc-1",
            type="function",
            function=FunctionCall(name="nonexistent", arguments={}),
        )
        await engine._execute_single_tool(tool_call, AgentMode.YOLO, MagicMock(return_value=True))
        engine._error_recovery.record_failure.assert_called_once()
        engine._error_recovery.record_error.assert_called_once()


class TestExecuteSingleToolRetryLogic:
    @pytest.mark.asyncio
    async def test_transient_error_retries_and_succeeds(self):
        call_count = 0

        class TransientThenSuccessTool(Tool):
            @property
            def name(self) -> str:
                return "flaky"

            @property
            def description(self) -> str:
                return "Flaky tool"

            @property
            def parameters(self) -> dict:
                return {"type": "object", "properties": {}}

            async def execute(self, **kwargs) -> ToolResult:
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return ToolResult(success=False, content="", error="timeout: connection lost")
                return ToolResult(success=True, content="recovered")

        tool = TransientThenSuccessTool()
        engine = _make_engine(tools={"flaky": tool})
        engine._retry_handler.get_max_retries.return_value = 3
        engine._retry_handler.is_transient_error.return_value = True
        engine._retry_handler.get_delay.return_value = 0

        tool_call = ToolCall(
            id="tc-1",
            type="function",
            function=FunctionCall(name="flaky", arguments={}),
        )
        tc, msg = await engine._execute_single_tool(tool_call, AgentMode.YOLO, MagicMock(return_value=True))
        assert "recovered" in msg.content
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_non_transient_error_no_retry(self):
        call_count = 0

        class NonTransientTool(Tool):
            @property
            def name(self) -> str:
                return "bad_input"

            @property
            def description(self) -> str:
                return "Bad input tool"

            @property
            def parameters(self) -> dict:
                return {"type": "object", "properties": {}}

            async def execute(self, **kwargs) -> ToolResult:
                nonlocal call_count
                call_count += 1
                return ToolResult(success=False, content="", error="Invalid input: bad argument")

        tool = NonTransientTool()
        engine = _make_engine(tools={"bad_input": tool})
        engine._retry_handler.get_max_retries.return_value = 3
        engine._retry_handler.is_transient_error.return_value = False
        engine._retry_handler.get_delay.return_value = 0

        tool_call = ToolCall(
            id="tc-1",
            type="function",
            function=FunctionCall(name="bad_input", arguments={}),
        )
        tc, msg = await engine._execute_single_tool(tool_call, AgentMode.YOLO, MagicMock(return_value=True))
        assert "Invalid input" in msg.content
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_exception_with_rejected_stops_retry(self):
        call_count = 0

        class RejectedTool(Tool):
            @property
            def name(self) -> str:
                return "rejected_tool"

            @property
            def description(self) -> str:
                return "Rejected tool"

            @property
            def parameters(self) -> dict:
                return {"type": "object", "properties": {}}

            async def execute(self, **kwargs) -> ToolResult:
                nonlocal call_count
                call_count += 1
                raise PermissionError("Operation rejected by policy")

        tool = RejectedTool()
        engine = _make_engine(tools={"rejected_tool": tool})
        engine._retry_handler.get_max_retries.return_value = 3
        engine._retry_handler.is_transient_error.return_value = True
        engine._retry_handler.get_delay.return_value = 0

        tool_call = ToolCall(
            id="tc-1",
            type="function",
            function=FunctionCall(name="rejected_tool", arguments={}),
        )
        tc, msg = await engine._execute_single_tool(tool_call, AgentMode.YOLO, MagicMock(return_value=True))
        assert "Error" in msg.content
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_exception_with_blocked_stops_retry(self):
        call_count = 0

        class BlockedTool(Tool):
            @property
            def name(self) -> str:
                return "blocked_tool"

            @property
            def description(self) -> str:
                return "Blocked tool"

            @property
            def parameters(self) -> dict:
                return {"type": "object", "properties": {}}

            async def execute(self, **kwargs) -> ToolResult:
                nonlocal call_count
                call_count += 1
                raise RuntimeError("Operation blocked by security policy")

        tool = BlockedTool()
        engine = _make_engine(tools={"blocked_tool": tool})
        engine._retry_handler.get_max_retries.return_value = 3
        engine._retry_handler.is_transient_error.return_value = True
        engine._retry_handler.get_delay.return_value = 0

        tool_call = ToolCall(
            id="tc-1",
            type="function",
            function=FunctionCall(name="blocked_tool", arguments={}),
        )
        tc, msg = await engine._execute_single_tool(tool_call, AgentMode.YOLO, MagicMock(return_value=True))
        assert "Error" in msg.content
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_result_with_rejected_error_stops_retry(self):
        call_count = 0

        class RejectedResultTool(Tool):
            @property
            def name(self) -> str:
                return "rejected_result"

            @property
            def description(self) -> str:
                return "Rejected result tool"

            @property
            def parameters(self) -> dict:
                return {"type": "object", "properties": {}}

            async def execute(self, **kwargs) -> ToolResult:
                nonlocal call_count
                call_count += 1
                return ToolResult(success=False, content="", error="Request rejected by server")

        tool = RejectedResultTool()
        engine = _make_engine(tools={"rejected_result": tool})
        engine._retry_handler.get_max_retries.return_value = 3
        engine._retry_handler.is_transient_error.return_value = True
        engine._retry_handler.get_delay.return_value = 0

        tool_call = ToolCall(
            id="tc-1",
            type="function",
            function=FunctionCall(name="rejected_result", arguments={}),
        )
        tc, msg = await engine._execute_single_tool(tool_call, AgentMode.YOLO, MagicMock(return_value=True))
        assert "rejected" in msg.content.lower()
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_result_with_blocked_error_stops_retry(self):
        call_count = 0

        class BlockedResultTool(Tool):
            @property
            def name(self) -> str:
                return "blocked_result"

            @property
            def description(self) -> str:
                return "Blocked result tool"

            @property
            def parameters(self) -> dict:
                return {"type": "object", "properties": {}}

            async def execute(self, **kwargs) -> ToolResult:
                nonlocal call_count
                call_count += 1
                return ToolResult(success=False, content="", error="Operation blocked")

        tool = BlockedResultTool()
        engine = _make_engine(tools={"blocked_result": tool})
        engine._retry_handler.get_max_retries.return_value = 3
        engine._retry_handler.is_transient_error.return_value = True
        engine._retry_handler.get_delay.return_value = 0

        tool_call = ToolCall(
            id="tc-1",
            type="function",
            function=FunctionCall(name="blocked_result", arguments={}),
        )
        tc, msg = await engine._execute_single_tool(tool_call, AgentMode.YOLO, MagicMock(return_value=True))
        assert "blocked" in msg.content.lower()
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_exception_non_transient_no_retry(self):
        call_count = 0

        class ExceptionTool(Tool):
            @property
            def name(self) -> str:
                return "exception_tool"

            @property
            def description(self) -> str:
                return "Exception tool"

            @property
            def parameters(self) -> dict:
                return {"type": "object", "properties": {}}

            async def execute(self, **kwargs) -> ToolResult:
                nonlocal call_count
                call_count += 1
                raise ValueError("Invalid argument")

        tool = ExceptionTool()
        engine = _make_engine(tools={"exception_tool": tool})
        engine._retry_handler.get_max_retries.return_value = 3
        engine._retry_handler.is_transient_error.return_value = False
        engine._retry_handler.get_delay.return_value = 0

        tool_call = ToolCall(
            id="tc-1",
            type="function",
            function=FunctionCall(name="exception_tool", arguments={}),
        )
        tc, msg = await engine._execute_single_tool(tool_call, AgentMode.YOLO, MagicMock(return_value=True))
        assert "Error" in msg.content
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_max_retries_exhausted(self):
        call_count = 0

        class AlwaysTransientTool(Tool):
            @property
            def name(self) -> str:
                return "always_transient"

            @property
            def description(self) -> str:
                return "Always transient tool"

            @property
            def parameters(self) -> dict:
                return {"type": "object", "properties": {}}

            async def execute(self, **kwargs) -> ToolResult:
                nonlocal call_count
                call_count += 1
                return ToolResult(success=False, content="", error="timeout: operation timed out")

        tool = AlwaysTransientTool()
        engine = _make_engine(tools={"always_transient": tool})
        engine._retry_handler.get_max_retries.return_value = 3
        engine._retry_handler.is_transient_error.return_value = True
        engine._retry_handler.get_delay.return_value = 0

        tool_call = ToolCall(
            id="tc-1",
            type="function",
            function=FunctionCall(name="always_transient", arguments={}),
        )
        tc, msg = await engine._execute_single_tool(tool_call, AgentMode.YOLO, MagicMock(return_value=True))
        assert "timeout" in msg.content.lower()
        assert call_count == 3


class TestExecuteSingleToolErrorRecovery:
    @pytest.mark.asyncio
    async def test_success_records_success(self):
        tool = MockTool("read_file")
        engine = _make_engine(tools={"read_file": tool})
        engine._retry_handler.get_max_retries.return_value = 1
        engine._retry_handler.is_transient_error.return_value = False
        engine._retry_handler.get_delay.return_value = 0

        tool_call = ToolCall(
            id="tc-1",
            type="function",
            function=FunctionCall(name="read_file", arguments={}),
        )
        await engine._execute_single_tool(tool_call, AgentMode.YOLO, MagicMock(return_value=True))
        engine._error_recovery.record_success.assert_called_once()

    @pytest.mark.asyncio
    async def test_failure_records_failure_and_error(self):
        tool = MockTool("bad_tool", result=ToolResult(success=False, content="", error="Something went wrong"))
        engine = _make_engine(tools={"bad_tool": tool})
        engine._retry_handler.get_max_retries.return_value = 1
        engine._retry_handler.is_transient_error.return_value = False
        engine._retry_handler.get_delay.return_value = 0

        tool_call = ToolCall(
            id="tc-1",
            type="function",
            function=FunctionCall(name="bad_tool", arguments={}),
        )
        await engine._execute_single_tool(tool_call, AgentMode.YOLO, MagicMock(return_value=True))
        engine._error_recovery.record_failure.assert_called_once()
        engine._error_recovery.record_error.assert_called_once()

    @pytest.mark.asyncio
    async def test_metrics_record_duration(self):
        tool = MockTool("read_file")
        engine = _make_engine(tools={"read_file": tool})
        engine._retry_handler.get_max_retries.return_value = 1
        engine._retry_handler.is_transient_error.return_value = False
        engine._retry_handler.get_delay.return_value = 0

        tool_call = ToolCall(
            id="tc-1",
            type="function",
            function=FunctionCall(name="read_file", arguments={}),
        )
        await engine._execute_single_tool(tool_call, AgentMode.YOLO, MagicMock(return_value=True))
        engine._metrics.record_tool_duration.assert_called_once()
        args = engine._metrics.record_tool_duration.call_args
        assert args[0][0] == "read_file"
        assert isinstance(args[0][1], float)


class TestOptimizeToolCalls:
    def test_optimize_does_not_crash(self):
        engine = _make_engine()
        tool_calls = [
            ToolCall(id="tc-1", type="function", function=FunctionCall(name="read_file", arguments={"path": "a.txt"})),
        ]
        result = engine.optimize_tool_calls(tool_calls)
        assert isinstance(result, list)

    def test_multiple_multi_read_calls_dedup_paths(self):
        engine = _make_engine()
        tool_calls = [
            ToolCall(
                id="tc-1",
                type="function",
                function=FunctionCall(name="multi_read", arguments={"paths": ["/a.txt", "/b.txt"]}),
            ),
            ToolCall(
                id="tc-2",
                type="function",
                function=FunctionCall(name="multi_read", arguments={"paths": ["/b.txt", "/c.txt"]}),
            ),
        ]
        result = engine._optimize_tool_calls(tool_calls)
        multi_read_calls = [tc for tc in result if tc.function.name == "multi_read"]
        assert len(multi_read_calls) == 1
        paths = multi_read_calls[0].function.arguments["paths"]
        assert "/a.txt" in paths
        assert "/b.txt" in paths
        assert "/c.txt" in paths
        assert len(paths) == 3

    def test_single_multi_read_not_deduped(self):
        engine = _make_engine()
        tool_calls = [
            ToolCall(
                id="tc-1",
                type="function",
                function=FunctionCall(name="multi_read", arguments={"paths": ["/a.txt"]}),
            ),
        ]
        result = engine._optimize_tool_calls(tool_calls)
        assert len(result) == 1
        assert result[0].function.name == "multi_read"

    def test_mixed_multi_read_and_other_calls(self):
        engine = _make_engine()
        tool_calls = [
            ToolCall(
                id="tc-1",
                type="function",
                function=FunctionCall(name="multi_read", arguments={"paths": ["/a.txt"]}),
            ),
            ToolCall(
                id="tc-2",
                type="function",
                function=FunctionCall(name="multi_read", arguments={"paths": ["/b.txt"]}),
            ),
            ToolCall(
                id="tc-3",
                type="function",
                function=FunctionCall(name="grep", arguments={"pattern": "test"}),
            ),
        ]
        result = engine._optimize_tool_calls(tool_calls)
        multi_read_calls = [tc for tc in result if tc.function.name == "multi_read"]
        grep_calls = [tc for tc in result if tc.function.name == "grep"]
        assert len(multi_read_calls) == 1
        assert len(grep_calls) == 1
        paths = multi_read_calls[0].function.arguments["paths"]
        assert "/a.txt" in paths
        assert "/b.txt" in paths

    def test_no_multi_read_calls_unchanged(self):
        engine = _make_engine()
        tool_calls = [
            ToolCall(id="tc-1", type="function", function=FunctionCall(name="read_file", arguments={"path": "/a.txt"})),
            ToolCall(id="tc-2", type="function", function=FunctionCall(name="grep", arguments={"pattern": "test"})),
        ]
        result = engine._optimize_tool_calls(tool_calls)
        assert len(result) == 2
        assert result[0].function.name == "read_file"
        assert result[1].function.name == "grep"

    def test_duplicate_paths_across_multi_read_deduped(self):
        engine = _make_engine()
        tool_calls = [
            ToolCall(
                id="tc-1",
                type="function",
                function=FunctionCall(name="multi_read", arguments={"paths": ["/a.txt", "/b.txt"]}),
            ),
            ToolCall(
                id="tc-2",
                type="function",
                function=FunctionCall(name="multi_read", arguments={"paths": ["/a.txt", "/c.txt"]}),
            ),
            ToolCall(
                id="tc-3",
                type="function",
                function=FunctionCall(name="multi_read", arguments={"paths": ["/b.txt", "/d.txt"]}),
            ),
        ]
        result = engine._optimize_tool_calls(tool_calls)
        multi_read_calls = [tc for tc in result if tc.function.name == "multi_read"]
        assert len(multi_read_calls) == 1
        paths = multi_read_calls[0].function.arguments["paths"]
        assert len(paths) == 4
        assert "/a.txt" in paths
        assert "/b.txt" in paths
        assert "/c.txt" in paths
        assert "/d.txt" in paths

    def test_multi_read_with_non_list_paths(self):
        engine = _make_engine()
        tool_calls = [
            ToolCall(
                id="tc-1",
                type="function",
                function=FunctionCall(name="multi_read", arguments={"paths": "not_a_list"}),
            ),
            ToolCall(
                id="tc-2",
                type="function",
                function=FunctionCall(name="multi_read", arguments={"paths": ["/a.txt"]}),
            ),
        ]
        result = engine._optimize_tool_calls(tool_calls)
        multi_read_calls = [tc for tc in result if tc.function.name == "multi_read"]
        assert len(multi_read_calls) == 1
        paths = multi_read_calls[0].function.arguments["paths"]
        assert "/a.txt" in paths

    def test_multi_read_with_empty_paths(self):
        engine = _make_engine()
        tool_calls = [
            ToolCall(
                id="tc-1",
                type="function",
                function=FunctionCall(name="multi_read", arguments={"paths": []}),
            ),
            ToolCall(
                id="tc-2",
                type="function",
                function=FunctionCall(name="multi_read", arguments={"paths": []}),
            ),
        ]
        result = engine._optimize_tool_calls(tool_calls)
        assert len(result) == 2

    def test_public_optimize_tool_calls_delegates(self):
        engine = _make_engine()
        tool_calls = [
            ToolCall(
                id="tc-1",
                type="function",
                function=FunctionCall(name="multi_read", arguments={"paths": ["/a.txt"]}),
            ),
            ToolCall(
                id="tc-2",
                type="function",
                function=FunctionCall(name="multi_read", arguments={"paths": ["/a.txt", "/b.txt"]}),
            ),
        ]
        result = engine.optimize_tool_calls(tool_calls)
        multi_read_calls = [tc for tc in result if tc.function.name == "multi_read"]
        assert len(multi_read_calls) == 1

    def test_deduped_call_uses_first_call_id(self):
        engine = _make_engine()
        tool_calls = [
            ToolCall(
                id="first-id",
                type="function",
                function=FunctionCall(name="multi_read", arguments={"paths": ["/a.txt"]}),
            ),
            ToolCall(
                id="second-id",
                type="function",
                function=FunctionCall(name="multi_read", arguments={"paths": ["/b.txt"]}),
            ),
        ]
        result = engine._optimize_tool_calls(tool_calls)
        multi_read_calls = [tc for tc in result if tc.function.name == "multi_read"]
        assert len(multi_read_calls) == 1
        assert multi_read_calls[0].id == "first-id"


class TestFormatArguments:
    def test_short_arguments(self):
        engine = _make_engine()
        result = engine._format_arguments({"path": "/test.txt", "mode": "read"})
        assert "path" in result
        assert "/test.txt" in result
        assert "mode" in result
        assert "read" in result

    def test_long_value_truncated(self):
        engine = _make_engine()
        long_value = "x" * 300
        result = engine._format_arguments({"data": long_value})
        assert "data" in result
        assert "..." in result
        parsed_lines = [line.strip().rstrip(",").rstrip('"') for line in result.split("\n")]
        data_line = [ln for ln in parsed_lines if "data" in ln][0]
        assert "..." in data_line

    def test_empty_arguments(self):
        engine = _make_engine()
        result = engine._format_arguments({})
        assert result == "{}"

    def test_value_exactly_200_chars(self):
        engine = _make_engine()
        value = "x" * 200
        result = engine._format_arguments({"key": value})
        assert "..." not in result or "key" in result

    def test_value_201_chars_truncated(self):
        engine = _make_engine()
        value = "x" * 201
        result = engine._format_arguments({"key": value})
        assert "..." in result

    def test_non_string_values(self):
        engine = _make_engine()
        result = engine._format_arguments({"count": 42, "flag": True, "items": [1, 2, 3]})
        assert "count" in result
        assert "42" in result
        assert "flag" in result

    def test_unicode_arguments(self):
        engine = _make_engine()
        result = engine._format_arguments({"text": "你好世界"})
        assert "你好世界" in result


class TestPrintToolResult:
    def test_print_success_result(self, capsys):
        engine = _make_engine()
        result = ToolResult(success=True, content="Operation completed successfully")
        engine._print_tool_result(result)
        captured = capsys.readouterr()
        assert "✓" in captured.out
        assert "Operation completed successfully" in captured.out

    def test_print_error_result(self, capsys):
        engine = _make_engine()
        result = ToolResult(success=False, content="", error="File not found")
        engine._print_tool_result(result)
        captured = capsys.readouterr()
        assert "✗" in captured.out
        assert "File not found" in captured.out

    def test_print_long_success_result_truncated(self, capsys):
        engine = _make_engine()
        long_content = "x" * 400
        result = ToolResult(success=True, content=long_content)
        engine._print_tool_result(result)
        captured = capsys.readouterr()
        assert "✓" in captured.out
        assert "..." in captured.out

    def test_print_short_success_result_not_truncated(self, capsys):
        engine = _make_engine()
        short_content = "x" * 100
        result = ToolResult(success=True, content=short_content)
        engine._print_tool_result(result)
        captured = capsys.readouterr()
        assert "✓" in captured.out
        assert short_content in captured.out


class TestPrintToolCall:
    def test_print_tool_call(self, capsys):
        engine = _make_engine()
        engine._print_tool_call("read_file", {"path": "/test.txt"})
        captured = capsys.readouterr()
        assert "read_file" in captured.out
        assert "/test.txt" in captured.out

    def test_print_tool_call_empty_args(self, capsys):
        engine = _make_engine()
        engine._print_tool_call("list_files", {})
        captured = capsys.readouterr()
        assert "list_files" in captured.out


class TestOnToolResult:
    def test_on_tool_result_success(self):
        engine = _make_engine()
        result = ToolResult(success=True, content="file contents here")
        engine._on_tool_result("read_file", result, {"path": "/test.txt"})
        engine.logger.log_tool_result.assert_called_once_with(
            tool_name="read_file",
            arguments={"path": "/test.txt"},
            result_success=True,
            result_content="file contents here",
            result_error=None,
        )

    def test_on_tool_result_failure(self):
        engine = _make_engine()
        result = ToolResult(success=False, content="", error="File not found")
        engine._on_tool_result("read_file", result, {"path": "/missing.txt"})
        engine.logger.log_tool_result.assert_called_once_with(
            tool_name="read_file",
            arguments={"path": "/missing.txt"},
            result_success=False,
            result_content=None,
            result_error="File not found",
        )

    def test_on_tool_result_no_arguments(self):
        engine = _make_engine()
        result = ToolResult(success=True, content="ok")
        engine._on_tool_result("list_files", result)
        engine.logger.log_tool_result.assert_called_once_with(
            tool_name="list_files",
            arguments={},
            result_success=True,
            result_content="ok",
            result_error=None,
        )


class TestSubAgentSecurityPolicy:
    def test_yolo_policy_default(self):
        from mini_agent.schema import AgentMode
        from mini_agent.subagent import SubAgent

        mock_llm = MagicMock()
        mock_llm.model = "test"
        agent = SubAgent(llm_client=mock_llm, tools=[])
        assert agent.mode == AgentMode.YOLO
        assert agent._approve_write_only is False

    def test_approve_write_policy(self):
        from mini_agent.schema import AgentMode
        from mini_agent.subagent import SubAgent, SubAgentSecurityPolicy

        mock_llm = MagicMock()
        mock_llm.model = "test"
        agent = SubAgent(llm_client=mock_llm, tools=[], security_policy=SubAgentSecurityPolicy.APPROVE_WRITE)
        assert agent.mode == AgentMode.AGENT
        assert agent._approve_write_only is True

    def test_approve_all_policy(self):
        from mini_agent.schema import AgentMode
        from mini_agent.subagent import SubAgent, SubAgentSecurityPolicy

        mock_llm = MagicMock()
        mock_llm.model = "test"
        agent = SubAgent(llm_client=mock_llm, tools=[], security_policy=SubAgentSecurityPolicy.APPROVE_ALL)
        assert agent.mode == AgentMode.AGENT
        assert agent._approve_write_only is False

    @pytest.mark.asyncio
    async def test_write_tool_blocked_in_approve_write_mode(self):
        from mini_agent.subagent import SubAgent, SubAgentSecurityPolicy

        mock_llm = MagicMock()
        mock_llm.model = "test"
        write_tool = MockTool("write_file")
        agent = SubAgent(
            llm_client=mock_llm,
            tools=[write_tool],
            security_policy=SubAgentSecurityPolicy.APPROVE_WRITE,
        )

        tool_call = ToolCall(
            id="tc-1",
            type="function",
            function=FunctionCall(name="write_file", arguments={"path": "test.txt", "content": "hello"}),
        )
        _, msg = await agent._execute_single_tool(tool_call)
        assert "approval" in msg.content.lower() or "approve" in msg.content.lower()


class TestConcurrentToolExecution:
    @pytest.mark.asyncio
    async def test_concurrent_execution_no_race(self):
        execution_order = []

        class OrderTrackingTool(Tool):
            def __init__(self, name: str, delay: float = 0.01):
                self._name = name
                self._delay = delay

            @property
            def name(self) -> str:
                return self._name

            @property
            def description(self) -> str:
                return f"Order tracking {self._name}"

            @property
            def parameters(self) -> dict:
                return {"type": "object", "properties": {}}

            async def execute(self, **kwargs) -> ToolResult:
                execution_order.append(self._name)
                await asyncio.sleep(self._delay)
                execution_order.append(f"{self._name}_done")
                return ToolResult(success=True, content=f"{self._name} result")

        tools = {
            "tool_a": OrderTrackingTool("tool_a"),
            "tool_b": OrderTrackingTool("tool_b"),
        }
        engine = _make_engine(tools=tools)
        tool_calls = [
            ToolCall(id="tc-1", type="function", function=FunctionCall(name="tool_a", arguments={})),
            ToolCall(id="tc-2", type="function", function=FunctionCall(name="tool_b", arguments={})),
        ]
        results = await engine.execute_tools(
            tool_calls=tool_calls,
            max_concurrent=5,
            parallel_enabled=True,
            mode=AgentMode.YOLO,
            check_approved_fn=MagicMock(return_value=True),
        )
        assert len(results) == 2
        assert "tool_a" in execution_order
        assert "tool_b" in execution_order

    @pytest.mark.asyncio
    async def test_concurrency_limit_respected(self):
        max_concurrent_seen = 0
        current_concurrent = 0

        class ConcurrencyTrackingTool(Tool):
            def __init__(self, name: str, delay: float = 0.05):
                self._name = name
                self._delay = delay

            @property
            def name(self) -> str:
                return self._name

            @property
            def description(self) -> str:
                return f"Concurrency tracking {self._name}"

            @property
            def parameters(self) -> dict:
                return {"type": "object", "properties": {}}

            async def execute(self, **kwargs) -> ToolResult:
                nonlocal current_concurrent, max_concurrent_seen
                current_concurrent += 1
                max_concurrent_seen = max(max_concurrent_seen, current_concurrent)
                await asyncio.sleep(self._delay)
                current_concurrent -= 1
                return ToolResult(success=True, content=f"{self._name} result")

        tools = {f"tool_{i}": ConcurrencyTrackingTool(f"tool_{i}") for i in range(6)}
        engine = _make_engine(tools=tools)
        tool_calls = [
            ToolCall(id=f"tc-{i}", type="function", function=FunctionCall(name=f"tool_{i}", arguments={}))
            for i in range(6)
        ]
        results = await engine.execute_tools(
            tool_calls=tool_calls,
            max_concurrent=2,
            parallel_enabled=True,
            mode=AgentMode.YOLO,
            check_approved_fn=MagicMock(return_value=True),
        )
        assert len(results) == 6
        assert max_concurrent_seen <= 2


class TestExecuteToolsRouting:
    @pytest.mark.asyncio
    async def test_parallel_false_routes_to_sequential(self):
        tool = MockTool("read_file")
        engine = _make_engine(tools={"read_file": tool})
        engine._retry_handler.get_max_retries.return_value = 1
        engine._retry_handler.is_transient_error.return_value = False
        engine._retry_handler.get_delay.return_value = 0

        tool_calls = [
            ToolCall(id="tc-1", type="function", function=FunctionCall(name="read_file", arguments={})),
            ToolCall(id="tc-2", type="function", function=FunctionCall(name="read_file", arguments={})),
        ]
        with patch.object(engine, "_execute_sequential", wraps=engine._execute_sequential) as mock_seq:
            _ = await engine.execute_tools(
                tool_calls=tool_calls,
                max_concurrent=5,
                parallel_enabled=False,
                mode=AgentMode.YOLO,
                check_approved_fn=MagicMock(return_value=True),
            )
            mock_seq.assert_called_once()

    @pytest.mark.asyncio
    async def test_parallel_true_single_tool_routes_to_sequential(self):
        tool = MockTool("read_file")
        engine = _make_engine(tools={"read_file": tool})
        engine._retry_handler.get_max_retries.return_value = 1
        engine._retry_handler.is_transient_error.return_value = False
        engine._retry_handler.get_delay.return_value = 0

        tool_calls = [
            ToolCall(id="tc-1", type="function", function=FunctionCall(name="read_file", arguments={})),
        ]
        with patch.object(engine, "_execute_sequential", wraps=engine._execute_sequential) as mock_seq:
            _ = await engine.execute_tools(
                tool_calls=tool_calls,
                max_concurrent=5,
                parallel_enabled=True,
                mode=AgentMode.YOLO,
                check_approved_fn=MagicMock(return_value=True),
            )
            mock_seq.assert_called_once()

    @pytest.mark.asyncio
    async def test_parallel_true_can_parallelize_routes_to_parallel(self):
        tools = {"read_file": MockTool("read_file"), "grep": MockTool("grep")}
        engine = _make_engine(tools=tools)
        engine._retry_handler.get_max_retries.return_value = 1
        engine._retry_handler.is_transient_error.return_value = False
        engine._retry_handler.get_delay.return_value = 0

        tool_calls = [
            ToolCall(id="tc-1", type="function", function=FunctionCall(name="read_file", arguments={})),
            ToolCall(id="tc-2", type="function", function=FunctionCall(name="grep", arguments={})),
        ]
        with patch.object(engine, "_execute_parallel", wraps=engine._execute_parallel) as mock_par:
            _ = await engine.execute_tools(
                tool_calls=tool_calls,
                max_concurrent=5,
                parallel_enabled=True,
                mode=AgentMode.YOLO,
                check_approved_fn=MagicMock(return_value=True),
            )
            mock_par.assert_called_once()

    @pytest.mark.asyncio
    async def test_parallel_true_cannot_parallelize_routes_to_batched(self):
        tools = {"write_file": MockTool("write_file"), "edit_file": MockTool("edit_file")}
        engine = _make_engine(tools=tools, write_tools=frozenset({"write_file", "edit_file"}))
        engine._retry_handler.get_max_retries.return_value = 1
        engine._retry_handler.is_transient_error.return_value = False
        engine._retry_handler.get_delay.return_value = 0

        tool_calls = [
            ToolCall(
                id="tc-1",
                type="function",
                function=FunctionCall(name="write_file", arguments={"path": "/same.py"}),
            ),
            ToolCall(
                id="tc-2",
                type="function",
                function=FunctionCall(name="edit_file", arguments={"path": "/same.py"}),
            ),
        ]
        with patch.object(engine, "_execute_batched", wraps=engine._execute_batched) as mock_batch:
            _ = await engine.execute_tools(
                tool_calls=tool_calls,
                max_concurrent=5,
                parallel_enabled=True,
                mode=AgentMode.YOLO,
                check_approved_fn=MagicMock(return_value=True),
            )
            mock_batch.assert_called_once()


class TestExecuteSingleToolMessageContent:
    @pytest.mark.asyncio
    async def test_successful_tool_returns_content(self):
        tool = MockTool("read_file", result=ToolResult(success=True, content="file contents"))
        engine = _make_engine(tools={"read_file": tool})
        engine._retry_handler.get_max_retries.return_value = 1
        engine._retry_handler.is_transient_error.return_value = False
        engine._retry_handler.get_delay.return_value = 0

        tool_call = ToolCall(
            id="tc-1",
            type="function",
            function=FunctionCall(name="read_file", arguments={"path": "/test.txt"}),
        )
        tc, msg = await engine._execute_single_tool(tool_call, AgentMode.YOLO, MagicMock(return_value=True))
        assert msg.content == "file contents"
        assert msg.role == "tool"
        assert msg.tool_call_id == "tc-1"
        assert msg.name == "read_file"

    @pytest.mark.asyncio
    async def test_failed_tool_returns_error_prefix(self):
        tool = MockTool("bad_tool", result=ToolResult(success=False, content="", error="disk full"))
        engine = _make_engine(tools={"bad_tool": tool})
        engine._retry_handler.get_max_retries.return_value = 1
        engine._retry_handler.is_transient_error.return_value = False
        engine._retry_handler.get_delay.return_value = 0

        tool_call = ToolCall(
            id="tc-1",
            type="function",
            function=FunctionCall(name="bad_tool", arguments={}),
        )
        tc, msg = await engine._execute_single_tool(tool_call, AgentMode.YOLO, MagicMock(return_value=True))
        assert msg.content.startswith("Error:")
        assert "disk full" in msg.content
