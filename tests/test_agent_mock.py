"""Mock-based agent tests that don't require real API keys."""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from mini_agent.agent import Agent
from mini_agent.core.token_tracker import TokenTracker
from mini_agent.schema import (
    AgentMode,
    FunctionCall,
    LLMResponse,
    Message,
    TokenUsage,
    ToolCall,
)
from mini_agent.tools.base import Tool, ToolResult


class FakeTool(Tool):
    """A fake tool for testing that returns predefined results."""

    def __init__(self, name: str = "test_tool", fail: bool = False):
        self._name = name
        self._fail = fail

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return f"A test tool: {self._name}"

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {"input": {"type": "string", "description": "Test input"}},
            "required": ["input"],
        }

    async def execute(self, input: str = "") -> ToolResult:
        if self._fail:
            return ToolResult(success=False, content="", error="Intentional failure")
        return ToolResult(success=True, content=f"Processed: {input}")


@pytest.fixture
def mock_llm():
    """Create a mock LLM client."""
    llm = MagicMock()
    llm.model = "MiniMax-M2.7"
    llm.api_key = "test-key"
    llm.api_base = "https://api.test.com"
    llm.provider = "anthropic"
    return llm


@pytest.fixture
def temp_workspace():
    """Create a temporary workspace."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.mark.mock
class TestAgentInit:
    """Test Agent initialization."""

    def test_basic_init(self, mock_llm, temp_workspace):
        agent = Agent(
            llm_client=mock_llm,
            system_prompt="You are a test assistant.",
            tools=[],
            max_steps=5,
            workspace_dir=str(temp_workspace),
            mode=AgentMode.YOLO,
        )
        assert agent.max_steps == 5
        assert agent.mode == AgentMode.YOLO
        assert agent.api_call_count == 0
        assert len(agent.messages) == 1  # system prompt only
        assert agent.workspace_dir == temp_workspace

    def test_init_with_tools(self, mock_llm, temp_workspace):
        tools = [FakeTool("read"), FakeTool("write")]
        agent = Agent(
            llm_client=mock_llm,
            system_prompt="You are a test assistant.",
            tools=tools,
            max_steps=10,
            workspace_dir=str(temp_workspace),
        )
        assert "read" in agent.tools
        assert "write" in agent.tools
        assert len(agent.tool_list) == 2

    def test_init_different_modes(self, mock_llm, temp_workspace):
        for mode in AgentMode:
            agent = Agent(
                llm_client=mock_llm,
                system_prompt="You are a test assistant.",
                tools=[],
                max_steps=5,
                workspace_dir=str(temp_workspace),
                mode=mode,
            )
            assert agent.mode == mode
            assert mode.value in agent.system_prompt.lower() or "(default)" in agent.system_prompt

    def test_workspace_creation(self, mock_llm):
        with tempfile.TemporaryDirectory() as base:
            ws = Path(base) / "nested" / "workspace"
            Agent(
                llm_client=mock_llm,
                system_prompt="test",
                tools=[],
                workspace_dir=str(ws),
            )
            assert ws.exists()
            assert ws.is_dir()


@pytest.mark.mock
class TestAgentTokenTracking:
    """Test TokenTracker integration."""

    def test_token_tracker_init(self):
        tracker = TokenTracker()
        assert tracker.cached_count == 0
        assert tracker.cache_version == 0

    def test_token_tracker_estimate_empty(self):
        tracker = TokenTracker()
        count = tracker.estimate_tokens([])
        assert count == 0

    def test_token_tracker_basic_estimate(self):
        tracker = TokenTracker()
        messages = [
            Message(role="system", content="You are a helpful assistant."),
            Message(role="user", content="Hello!"),
        ]
        count = tracker.estimate_tokens(messages)
        assert count > 0
        # Second call should return cached value
        assert tracker.estimate_tokens(messages) == count

    def test_token_tracker_incremental(self):
        tracker = TokenTracker()
        msg1 = Message(role="user", content="Hello")
        count1 = tracker.estimate_tokens([msg1])

        msg2 = Message(role="assistant", content="Hi there!")
        count2 = tracker.estimate_tokens([msg1, msg2])
        assert count2 > count1

    def test_token_tracker_invalidate(self):
        tracker = TokenTracker()
        messages = [Message(role="user", content="Test message")]
        count = tracker.estimate_tokens(messages)
        tracker.invalidate_cache()
        # After invalidation, should recount from scratch
        assert tracker.estimate_tokens(messages) == count
        assert tracker.cached_count == count

    def test_token_tracker_fallback(self, monkeypatch):
        tracker = TokenTracker()
        # Force tiktoken to fail
        import mini_agent.utils.token_utils

        original = mini_agent.utils.token_utils.get_encoder
        mini_agent.utils.token_utils.get_encoder = MagicMock(side_effect=Exception("no tiktoken"))

        try:
            messages = [Message(role="user", content="Test message for fallback")]
            count = tracker.estimate_tokens(messages)
            assert count > 0  # Fallback should still produce a result
        finally:
            mini_agent.utils.token_utils.get_encoder = original

    def test_message_estimate_tokens(self, mock_llm, temp_workspace):
        agent = Agent(
            llm_client=mock_llm,
            system_prompt="You are a test assistant.",
            tools=[],
            workspace_dir=str(temp_workspace),
        )
        agent.add_user_message("Hello!")
        count = agent._context.estimate_tokens()
        assert count > 0


@pytest.mark.mock
class TestAgentBehavior:
    """Test agent behavior with mocked LLM."""

    async def _make_response(self, content: str = "", tool_calls: list | None = None):
        return LLMResponse(
            content=content,
            thinking=None,
            tool_calls=tool_calls,
            finish_reason="end_turn" if not tool_calls else "tool_use",
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )

    async def test_agent_text_response(self, mock_llm, temp_workspace):
        mock_llm.generate = AsyncMock(return_value=await self._make_response("Hello, world!"))
        agent = Agent(
            llm_client=mock_llm,
            system_prompt="You are a test assistant.",
            tools=[],
            max_steps=5,
            workspace_dir=str(temp_workspace),
        )
        agent.add_user_message("Say hello")
        result = await agent.run()
        assert result == "Hello, world!"
        assert agent.api_call_count == 1

    async def test_agent_tool_call_then_text(self, mock_llm, temp_workspace):
        tool = FakeTool("test_tool")
        tool_call = ToolCall(
            id="call_1",
            type="function",
            function=FunctionCall(name="test_tool", arguments={"input": "hello"}),
        )

        mock_llm.generate = AsyncMock()
        mock_llm.generate.side_effect = [
            await self._make_response("Using tool...", tool_calls=[tool_call]),
            await self._make_response("Task done!"),
        ]

        agent = Agent(
            llm_client=mock_llm,
            system_prompt="You are a test assistant.",
            tools=[tool],
            max_steps=5,
            workspace_dir=str(temp_workspace),
        )
        agent.add_user_message("Do something")
        result = await agent.run()
        assert result == "Task done!"
        assert agent.api_call_count == 2

    async def test_agent_tool_failure(self, mock_llm, temp_workspace):
        tool = FakeTool("failing_tool", fail=True)
        tool_call = ToolCall(
            id="call_fail",
            type="function",
            function=FunctionCall(name="failing_tool", arguments={"input": "test"}),
        )

        mock_llm.generate = AsyncMock()
        mock_llm.generate.side_effect = [
            await self._make_response("Trying...", tool_calls=[tool_call]),
            await self._make_response("Recovered from failure."),
        ]

        agent = Agent(
            llm_client=mock_llm,
            system_prompt="You are a test assistant.",
            tools=[tool],
            max_steps=5,
            workspace_dir=str(temp_workspace),
        )
        agent.add_user_message("Test failure recovery")
        result = await agent.run()
        assert result is not None
        assert agent.api_call_count == 2

    async def test_max_steps_reached(self, mock_llm, temp_workspace):
        class InfiniteTool(FakeTool):
            async def execute(self, input: str = "") -> ToolResult:
                return ToolResult(success=True, content="Still going...")

        tool = InfiniteTool("loop_tool")
        tool_call = ToolCall(
            id="call_loop",
            type="function",
            function=FunctionCall(name="loop_tool", arguments={"input": "continue"}),
        )

        mock_llm.generate = AsyncMock(return_value=await self._make_response("Next step...", tool_calls=[tool_call]))

        agent = Agent(
            llm_client=mock_llm,
            system_prompt="You are a test assistant.",
            tools=[tool],
            max_steps=3,
            workspace_dir=str(temp_workspace),
        )
        agent.add_user_message("Keep going")
        result = await agent.run()
        assert "couldn't be completed" in result or "steps" in result


@pytest.mark.mock
class TestAgentMode:
    """Test agent mode behavior."""

    def test_mode_switching(self, mock_llm, temp_workspace):
        agent = Agent(
            llm_client=mock_llm,
            system_prompt="test",
            tools=[],
            workspace_dir=str(temp_workspace),
            mode=AgentMode.AGENT,
        )
        assert agent.mode == AgentMode.AGENT
        agent.set_mode(AgentMode.YOLO)
        assert agent.mode == AgentMode.YOLO
        agent.set_mode(AgentMode.PLAN)
        assert agent.mode == AgentMode.PLAN

    def test_get_status(self, mock_llm, temp_workspace):
        agent = Agent(
            llm_client=mock_llm,
            system_prompt="test",
            tools=[],
            workspace_dir=str(temp_workspace),
        )
        status = agent.get_status()
        assert "token_usage" in status
        assert "token_limit" in status
        assert "api_call_count" in status
        assert "mode" in status
        assert status["mode"] == AgentMode.YOLO.value

    def test_get_status_report(self, mock_llm, temp_workspace):
        agent = Agent(
            llm_client=mock_llm,
            system_prompt="test",
            tools=[],
            workspace_dir=str(temp_workspace),
        )
        report = agent.get_status_report()
        assert "Agent Status Report" in report
        assert "Token usage" in report
        assert "Mode" in report


@pytest.mark.mock
class TestAgentSuggestions:
    """Test agent self-diagnosis."""

    def test_suggestions_no_errors(self, mock_llm, temp_workspace):
        agent = Agent(
            llm_client=mock_llm,
            system_prompt="test",
            tools=[],
            workspace_dir=str(temp_workspace),
        )
        suggestions = agent.get_suggestions()
        assert isinstance(suggestions, list)

    def test_error_patterns(self, mock_llm, temp_workspace):
        agent = Agent(
            llm_client=mock_llm,
            system_prompt="test",
            tools=[],
            workspace_dir=str(temp_workspace),
        )
        patterns = agent.get_error_patterns()
        assert "error_counts_by_tool" in patterns
        assert "recent_errors" in patterns
