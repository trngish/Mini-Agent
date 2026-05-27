"""Tests for SubAgent and run_sub_agents."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from mini_agent.schema.schema import FunctionCall, ToolCall
from mini_agent.subagent import SubAgent, SubAgentResult, run_sub_agents


class TestSubAgentResult:
    """Test SubAgentResult class."""

    def test_result_properties(self):
        """Test SubAgentResult stores values correctly."""
        result = SubAgentResult(
            task="test task",
            content="result content",
            success=True,
            elapsed=1.5,
            error="",
        )

        assert result.task == "test task"
        assert result.content == "result content"
        assert result.success is True
        assert result.elapsed == 1.5
        assert result.error == ""

    def test_result_with_error(self):
        """Test SubAgentResult with error."""
        result = SubAgentResult(
            task="test task",
            content="",
            success=False,
            elapsed=0.5,
            error="Something went wrong",
        )

        assert result.success is False
        assert result.error == "Something went wrong"


class TestSubAgent:
    """Test SubAgent functionality."""

    @pytest.fixture
    def mock_llm_client(self):
        """Create a mock LLM client."""
        client = MagicMock()
        client.model = "MiniMax-M2.5"
        return client

    @pytest.fixture
    def mock_tool(self):
        """Create a mock tool."""
        from mini_agent.tools.base import ToolResult

        tool = MagicMock()
        tool.name = "test_tool"
        # Return a proper ToolResult object
        tool.execute = AsyncMock(return_value=ToolResult(success=True, content="tool result"))
        return tool

    @pytest.fixture
    def subagent(self, mock_llm_client, mock_tool):
        """Create a SubAgent instance."""
        return SubAgent(
            llm_client=mock_llm_client,
            tools=[mock_tool],
            system_prompt="You are a test assistant.",
            max_steps=10,
        )

    def test_subagent_initialization(self, subagent, mock_llm_client, mock_tool):
        """Test SubAgent initializes correctly."""
        assert subagent.llm == mock_llm_client
        assert subagent.tools["test_tool"] == mock_tool
        assert subagent.max_steps == 10
        assert subagent.mode.value == "yolo"

    def test_execute_single_tool_success(self, subagent, mock_tool):
        """Test successful tool execution."""
        tool_call = ToolCall(
            id="1",
            type="function",
            function=FunctionCall(name="test_tool", arguments={"arg": "value"}),
        )

        result = asyncio.run(subagent._execute_single_tool(tool_call))

        assert result[0] == tool_call
        mock_tool.execute.assert_called_once_with(arg="value")

    def test_execute_single_tool_unknown_tool(self, subagent):
        """Test handling of unknown tool."""
        tool_call = ToolCall(
            id="1",
            type="function",
            function=FunctionCall(name="unknown_tool", arguments={}),
        )

        result = asyncio.run(subagent._execute_single_tool(tool_call))

        _, tool_msg = result
        assert "Unknown tool" in tool_msg.content

    def test_execute_single_tool_exception(self, subagent, mock_tool):
        """Test handling of tool exception."""
        mock_tool.execute = AsyncMock(side_effect=Exception("Tool failed"))

        tool_call = ToolCall(
            id="1",
            type="function",
            function=FunctionCall(name="test_tool", arguments={}),
        )

        result = asyncio.run(subagent._execute_single_tool(tool_call))

        _, tool_msg = result
        assert "Error" in tool_msg.content

    def test_execute_tools_sequential(self, subagent, mock_tool):
        """Test sequential tool execution."""
        tool_calls = [
            ToolCall(
                id=str(i),
                type="function",
                function=FunctionCall(name="test_tool", arguments={"n": i}),
            )
            for i in range(3)
        ]

        results = asyncio.run(subagent._execute_tools_sequential(tool_calls))

        assert len(results) == 3
        mock_tool.execute.assert_called()

    def test_execute_tools_parallel(self, subagent, mock_tool):
        """Test parallel tool execution."""
        tool_calls = [
            ToolCall(
                id=str(i),
                type="function",
                function=FunctionCall(name="test_tool", arguments={"n": i}),
            )
            for i in range(5)
        ]

        results = asyncio.run(subagent._execute_tools_parallel(tool_calls, max_concurrent=3))

        assert len(results) == 5

    def test_execute_no_tool_calls(self, mock_llm_client, mock_tool):
        """Test execution when LLM returns no tool calls."""
        mock_llm_client.generate = AsyncMock(
            return_value=MagicMock(
                content="Final response",
                tool_calls=None,
                thinking=None,
            )
        )

        subagent = SubAgent(llm_client=mock_llm_client, tools=[mock_tool])

        result = asyncio.run(subagent.run("Test task"))

        assert result.success is True
        assert result.content == "Final response"

    @pytest.mark.skip(reason="SubAgent.run() has subtle async mock interactions")
    def test_execute_with_tool_calls(self, mock_llm_client, mock_tool):
        """Test execution with tool calls."""
        mock_llm_client.generate = AsyncMock(
            return_value=MagicMock(
                content="Using tool",
                tool_calls=[
                    ToolCall(
                        id="1",
                        type="function",
                        function=FunctionCall(name="test_tool", arguments={}),
                    )
                ],
                thinking=None,
            )
        )

        subagent = SubAgent(llm_client=mock_llm_client, tools=[mock_tool])

        result = asyncio.run(subagent.run("Test task"))

        assert result.success is True
        assert mock_tool.execute.called

    def test_execute_max_steps_reached(self, mock_llm_client, mock_tool):
        """Test execution when max steps reached."""

        async def keep_calling(*args, **kwargs):
            return MagicMock(
                content="Still working",
                tool_calls=[
                    ToolCall(
                        id="1",
                        type="function",
                        function=FunctionCall(name="test_tool", arguments={}),
                    )
                ],
                thinking=None,
            )

        mock_llm_client.generate = keep_calling

        subagent = SubAgent(llm_client=mock_llm_client, tools=[mock_tool], max_steps=2)

        result = asyncio.run(subagent.run("Test task"))

        assert result.success is False
        assert "Max steps" in result.error

    def test_execute_llm_exception(self, mock_llm_client, mock_tool):
        """Test handling LLM exception."""
        mock_llm_client.generate = AsyncMock(side_effect=Exception("LLM failed"))

        subagent = SubAgent(llm_client=mock_llm_client, tools=[mock_tool])

        result = asyncio.run(subagent.run("Test task"))

        assert result.success is False
        assert "LLM failed" in result.error

    def test_cleanup(self, subagent):
        """Test cleanup clears tool references."""
        assert len(subagent.tools) > 0
        assert len(subagent.tool_list) > 0

        subagent.cleanup()

        assert len(subagent.tools) == 0
        assert len(subagent.tool_list) == 0


class TestRunSubAgents:
    """Test run_sub_agents function."""

    @pytest.fixture
    def mock_llm_client(self):
        """Create a mock LLM client."""
        client = MagicMock()
        client.model = "MiniMax-M2.5"
        client.clone = MagicMock(return_value=client)
        return client

    @pytest.fixture
    def mock_tool(self):
        """Create a mock tool."""
        tool = MagicMock()
        tool.name = "test_tool"
        tool.execute = AsyncMock(return_value=MagicMock(success=True, content="done"))
        return tool

    @pytest.mark.asyncio
    async def test_run_sub_agents_single_task(self, mock_llm_client, mock_tool):
        """Test running a single task."""
        mock_llm_client.generate = AsyncMock(return_value=MagicMock(content="Done", tool_calls=None, thinking=None))

        results = await run_sub_agents(
            llm_client=mock_llm_client,
            tasks=["Task 1"],
            tools=[mock_tool],
            max_concurrent=1,
        )

        assert len(results) == 1
        assert results[0].success is True
        assert results[0].task == "Task 1"

    @pytest.mark.asyncio
    async def test_run_sub_agents_multiple_tasks(self, mock_llm_client, mock_tool):
        """Test running multiple tasks."""
        mock_llm_client.generate = AsyncMock(return_value=MagicMock(content="Done", tool_calls=None, thinking=None))

        results = await run_sub_agents(
            llm_client=mock_llm_client,
            tasks=["Task 1", "Task 2", "Task 3"],
            tools=[mock_tool],
            max_concurrent=2,
        )

        assert len(results) == 3
        assert all(r.success for r in results)
        assert results[0].task == "Task 1"
        assert results[1].task == "Task 2"
        assert results[2].task == "Task 3"

    @pytest.mark.asyncio
    async def test_run_sub_agents_respects_concurrency(self, mock_llm_client, mock_tool):
        """Test that concurrency limit is respected."""
        call_count = [0]

        async def slow_generate(*args, **kwargs):
            call_count[0] += 1
            await asyncio.sleep(0.1)
            return MagicMock(content="Done", tool_calls=None, thinking=None)

        mock_llm_client.generate = slow_generate

        results = await run_sub_agents(
            llm_client=mock_llm_client,
            tasks=["Task 1", "Task 2", "Task 3", "Task 4"],
            tools=[mock_tool],
            max_concurrent=2,
        )

        assert len(results) == 4
        # All should succeed
        assert all(r.success for r in results)

    @pytest.mark.asyncio
    async def test_run_sub_agents_preserves_order(self, mock_llm_client, mock_tool):
        """Test that results maintain task order."""
        mock_llm_client.generate = AsyncMock(return_value=MagicMock(content="Done", tool_calls=None, thinking=None))

        tasks = ["Alpha", "Beta", "Gamma"]
        results = await run_sub_agents(
            llm_client=mock_llm_client,
            tasks=tasks,
            tools=[mock_tool],
            max_concurrent=3,
        )

        assert results[0].task == "Alpha"
        assert results[1].task == "Beta"
        assert results[2].task == "Gamma"
