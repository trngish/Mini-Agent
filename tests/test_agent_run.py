"""Comprehensive tests for Agent.run() method.

Covers all branches of the run loop:
- Normal completion (no tool calls)
- Tool execution and continuation
- Cancellation at step start
- Cancellation during tool result appending
- LLM error handling (retryable and non-retryable)
- Max steps reached
- Health issues display
- Thinking/text streaming callbacks
- Optimized tool calls (results < tool_calls)
- Auto-save on max steps
"""

import asyncio
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mini_agent.agent import Agent
from mini_agent.core.execution_engine import ExecutionEngine
from mini_agent.schema import AgentMode, FunctionCall, LLMResponse, Message, TokenUsage, ToolCall
from mini_agent.tools.base import Tool, ToolResult
from mini_agent.utils.error_handler import LLMError, LLMErrorType


class FakeTool(Tool):
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
            "properties": {"input": {"type": "string"}},
            "required": ["input"],
        }

    async def execute(self, input: str = "") -> ToolResult:
        if self._fail:
            return ToolResult(success=False, content="", error="Intentional failure")
        return ToolResult(success=True, content=f"Processed: {input}")


def _make_response(content: str = "", tool_calls: list | None = None, thinking: str | None = None):
    return LLMResponse(
        content=content,
        thinking=thinking,
        tool_calls=tool_calls,
        finish_reason="end_turn" if not tool_calls else "tool_use",
        usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


def _make_tool_call(name: str = "test_tool", call_id: str = "call_1", args: dict | None = None):
    return ToolCall(
        id=call_id,
        type="function",
        function=FunctionCall(name=name, arguments=args or {"input": "test"}),
    )


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.model = "test-model"
    llm.api_key = "test-key"
    llm.api_base = "https://api.test.com"
    llm.provider = "anthropic"
    llm.generate = AsyncMock()
    return llm


@pytest.fixture
def temp_workspace():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


def _make_agent(mock_llm, temp_workspace, tools=None, max_steps=5, mode=AgentMode.YOLO):
    return Agent(
        llm_client=mock_llm,
        system_prompt="You are a test assistant.",
        tools=tools or [],
        max_steps=max_steps,
        workspace_dir=temp_workspace,
        mode=mode,
    )


@pytest.mark.mock
class TestRunNormalCompletion:
    async def test_immediate_text_response(self, mock_llm, temp_workspace):
        mock_llm.generate = AsyncMock(return_value=_make_response("Done!"))
        agent = _make_agent(mock_llm, temp_workspace)
        agent.add_user_message("Say hi")
        result = await agent.run()
        assert result == "Done!"
        assert agent.api_call_count == 1

    async def test_response_with_thinking(self, mock_llm, temp_workspace):
        mock_llm.generate = AsyncMock(return_value=_make_response("Final answer", thinking="Let me think..."))
        agent = _make_agent(mock_llm, temp_workspace)
        agent.add_user_message("Think and answer")
        result = await agent.run()
        assert result == "Final answer"

    async def test_cancel_event_none_by_default(self, mock_llm, temp_workspace):
        mock_llm.generate = AsyncMock(return_value=_make_response("ok"))
        agent = _make_agent(mock_llm, temp_workspace)
        agent.add_user_message("hi")
        assert agent.cancel_event is None
        await agent.run()


@pytest.mark.mock
class TestRunToolExecution:
    async def test_single_tool_then_complete(self, mock_llm, temp_workspace):
        tool = FakeTool("test_tool")
        tc = _make_tool_call("test_tool")
        mock_llm.generate = AsyncMock(
            side_effect=[
                _make_response("Using tool...", tool_calls=[tc]),
                _make_response("All done!"),
            ]
        )
        agent = _make_agent(mock_llm, temp_workspace, tools=[tool])
        agent.add_user_message("Use the tool")
        result = await agent.run()
        assert result == "All done!"
        assert agent.api_call_count == 2

    async def test_multiple_tool_calls_sequential(self, mock_llm, temp_workspace):
        tool_a = FakeTool("tool_a")
        tool_b = FakeTool("tool_b")
        tc_a = _make_tool_call("tool_a", call_id="c1")
        tc_b = _make_tool_call("tool_b", call_id="c2")
        mock_llm.generate = AsyncMock(
            side_effect=[
                _make_response("Using tools...", tool_calls=[tc_a, tc_b]),
                _make_response("Finished!"),
            ]
        )
        agent = _make_agent(mock_llm, temp_workspace, tools=[tool_a, tool_b])
        agent.add_user_message("Use both tools")
        result = await agent.run()
        assert result == "Finished!"

    async def test_tool_failure_continues(self, mock_llm, temp_workspace):
        tool = FakeTool("failing_tool", fail=True)
        tc = _make_tool_call("failing_tool")
        mock_llm.generate = AsyncMock(
            side_effect=[
                _make_response("Trying...", tool_calls=[tc]),
                _make_response("Recovered!"),
            ]
        )
        agent = _make_agent(mock_llm, temp_workspace, tools=[tool])
        agent.add_user_message("Try failing tool")
        result = await agent.run()
        assert result == "Recovered!"

    async def test_multiple_steps_with_tools(self, mock_llm, temp_workspace):
        tool = FakeTool("test_tool")
        tc = _make_tool_call("test_tool")
        mock_llm.generate = AsyncMock(
            side_effect=[
                _make_response("Step 1", tool_calls=[tc]),
                _make_response("Step 2", tool_calls=[tc]),
                _make_response("Step 3 done"),
            ]
        )
        agent = _make_agent(mock_llm, temp_workspace, tools=[tool])
        agent.add_user_message("Multi-step task")
        result = await agent.run()
        assert result == "Step 3 done"
        assert agent.api_call_count == 3


@pytest.mark.mock
class TestRunCancellation:
    async def test_cancel_at_step_start(self, mock_llm, temp_workspace):
        mock_llm.generate = AsyncMock(return_value=_make_response("ok"))
        agent = _make_agent(mock_llm, temp_workspace)
        agent.add_user_message("hi")

        cancel_event = asyncio.Event()
        cancel_event.set()
        result = await agent.run(cancel_event=cancel_event)
        assert result == "Task cancelled by user."

    async def test_cancel_during_tool_results(self, mock_llm, temp_workspace):
        tool = FakeTool("test_tool")
        tc = _make_tool_call("test_tool")
        mock_llm.generate = AsyncMock(return_value=_make_response("Using tool", tool_calls=[tc]))

        agent = _make_agent(mock_llm, temp_workspace, tools=[tool])
        agent.add_user_message("hi")

        cancel_event = asyncio.Event()

        real_execute = ExecutionEngine.execute_tools

        async def execute_and_cancel(self_engine, *args, **kwargs):
            cancel_event.set()
            return await real_execute(self_engine, *args, **kwargs)

        with patch.object(ExecutionEngine, "execute_tools", execute_and_cancel):
            result = await agent.run(cancel_event=cancel_event)

        assert result == "Task cancelled by user."

    async def test_cancel_event_set_via_attribute(self, mock_llm, temp_workspace):
        mock_llm.generate = AsyncMock(return_value=_make_response("ok"))
        agent = _make_agent(mock_llm, temp_workspace)
        agent.add_user_message("hi")

        cancel_event = asyncio.Event()
        agent.cancel_event = cancel_event
        cancel_event.set()

        result = await agent.run()
        assert result == "Task cancelled by user."

    async def test_cancel_cleans_up_incomplete_messages(self, mock_llm, temp_workspace):
        tool = FakeTool("test_tool")
        tc = _make_tool_call("test_tool")
        mock_llm.generate = AsyncMock(return_value=_make_response("Using tool", tool_calls=[tc]))

        agent = _make_agent(mock_llm, temp_workspace, tools=[tool])
        agent.add_user_message("hi")

        cancel_event = asyncio.Event()

        real_execute = ExecutionEngine.execute_tools

        async def execute_and_cancel(self_engine, *args, **kwargs):
            cancel_event.set()
            return await real_execute(self_engine, *args, **kwargs)

        with patch.object(ExecutionEngine, "execute_tools", execute_and_cancel):
            await agent.run(cancel_event=cancel_event)

        for msg in agent.messages:
            assert msg.role != "assistant" or msg.tool_calls is None or len(agent.messages) <= 2


@pytest.mark.mock
class TestRunLLMError:
    async def test_llm_generic_error(self, mock_llm, temp_workspace):
        mock_llm.generate = AsyncMock(side_effect=RuntimeError("Something broke"))
        agent = _make_agent(mock_llm, temp_workspace)
        agent.add_user_message("hi")
        result = await agent.run()
        assert "LLM call failed" in result

    async def test_llm_rate_limit_error(self, mock_llm, temp_workspace):
        error = LLMError(
            message="Rate limited",
            error_type=LLMErrorType.RATE_LIMIT_ERROR,
            retry_after=5,
        )
        mock_llm.generate = AsyncMock(side_effect=error)
        agent = _make_agent(mock_llm, temp_workspace)
        agent.add_user_message("hi")
        result = await agent.run()
        assert "LLM call failed" in result

    async def test_llm_auth_error(self, mock_llm, temp_workspace):
        error = LLMError(
            message="Invalid API key",
            error_type=LLMErrorType.AUTHENTICATION_ERROR,
        )
        mock_llm.generate = AsyncMock(side_effect=error)
        agent = _make_agent(mock_llm, temp_workspace)
        agent.add_user_message("hi")
        result = await agent.run()
        assert "LLM call failed" in result

    async def test_llm_context_length_exceeded(self, mock_llm, temp_workspace):
        error = LLMError(
            message="Context too long",
            error_type=LLMErrorType.CONTEXT_LENGTH_EXCEEDED,
        )
        mock_llm.generate = AsyncMock(side_effect=error)
        agent = _make_agent(mock_llm, temp_workspace)
        agent.add_user_message("hi")
        result = await agent.run()
        assert "LLM call failed" in result


@pytest.mark.mock
class TestRunMaxSteps:
    async def test_max_steps_reached(self, mock_llm, temp_workspace):
        tool = FakeTool("loop_tool")
        tc = _make_tool_call("loop_tool")
        mock_llm.generate = AsyncMock(return_value=_make_response("Continuing...", tool_calls=[tc]))
        agent = _make_agent(mock_llm, temp_workspace, tools=[tool], max_steps=2)
        agent.add_user_message("Keep going")
        result = await agent.run()
        assert "couldn't be completed" in result
        assert "2 steps" in result

    async def test_max_steps_auto_save(self, mock_llm, temp_workspace):
        tool = FakeTool("loop_tool")
        tc = _make_tool_call("loop_tool")
        mock_llm.generate = AsyncMock(return_value=_make_response("Continuing...", tool_calls=[tc]))
        agent = _make_agent(mock_llm, temp_workspace, tools=[tool], max_steps=2)
        agent.auto_save = True
        agent.add_user_message("Keep going")

        with patch.object(agent._session_manager, "save", return_value="test_session_id") as mock_save:
            result = await agent.run()
            assert "couldn't be completed" in result
            mock_save.assert_called_once()

    async def test_max_steps_auto_save_failure(self, mock_llm, temp_workspace):
        tool = FakeTool("loop_tool")
        tc = _make_tool_call("loop_tool")
        mock_llm.generate = AsyncMock(return_value=_make_response("Continuing...", tool_calls=[tc]))
        agent = _make_agent(mock_llm, temp_workspace, tools=[tool], max_steps=2)
        agent.auto_save = True
        agent.add_user_message("Keep going")

        with patch.object(agent._session_manager, "save", side_effect=OSError("disk full")):
            result = await agent.run()
            assert "couldn't be completed" in result

    async def test_max_steps_auto_save_disabled(self, mock_llm, temp_workspace):
        tool = FakeTool("loop_tool")
        tc = _make_tool_call("loop_tool")
        mock_llm.generate = AsyncMock(return_value=_make_response("Continuing...", tool_calls=[tc]))
        agent = _make_agent(mock_llm, temp_workspace, tools=[tool], max_steps=2)
        agent.auto_save = False
        agent.add_user_message("Keep going")

        with patch.object(agent._session_manager, "save", return_value="sid") as mock_save:
            result = await agent.run()
            assert "couldn't be completed" in result
            mock_save.assert_not_called()


@pytest.mark.mock
class TestRunThinkingCallbacks:
    async def test_on_thinking_callback(self, mock_llm, temp_workspace):
        async def fake_generate(messages, tools, on_text=None, on_thinking=None):
            if on_thinking:
                on_thinking("hmm...")
            if on_text:
                on_text("answer")
            return _make_response("answer")

        mock_llm.generate = fake_generate
        agent = _make_agent(mock_llm, temp_workspace)
        agent.add_user_message("Think and answer")
        result = await agent.run()
        assert result == "answer"

    async def test_on_text_without_thinking(self, mock_llm, temp_workspace):
        async def fake_generate(messages, tools, on_text=None, on_thinking=None):
            if on_text:
                on_text("direct ")
                on_text("answer")
            return _make_response("direct answer")

        mock_llm.generate = fake_generate
        agent = _make_agent(mock_llm, temp_workspace)
        agent.add_user_message("Just answer")
        result = await agent.run()
        assert result == "direct answer"

    async def test_on_thinking_then_text(self, mock_llm, temp_workspace):
        async def fake_generate(messages, tools, on_text=None, on_thinking=None):
            if on_thinking:
                on_thinking("Let me think")
            if on_text:
                on_text("The answer is 42")
            return _make_response("The answer is 42", thinking="Let me think")

        mock_llm.generate = fake_generate
        agent = _make_agent(mock_llm, temp_workspace)
        agent.add_user_message("Think then answer")
        result = await agent.run()
        assert result == "The answer is 42"

    async def test_multiple_thinking_chunks(self, mock_llm, temp_workspace):
        async def fake_generate(messages, tools, on_text=None, on_thinking=None):
            if on_thinking:
                on_thinking("step 1...")
                on_thinking("step 2...")
            if on_text:
                on_text("final")
            return _make_response("final", thinking="step 1...step 2...")

        mock_llm.generate = fake_generate
        agent = _make_agent(mock_llm, temp_workspace)
        agent.add_user_message("Multi-think")
        result = await agent.run()
        assert result == "final"


@pytest.mark.mock
class TestRunHealthCheck:
    async def test_health_issues_displayed(self, mock_llm, temp_workspace):
        mock_llm.generate = AsyncMock(return_value=_make_response("ok"))
        agent = _make_agent(mock_llm, temp_workspace)
        agent.add_user_message("hi")

        with patch.object(agent._health_checker, "check") as mock_check:
            mock_issue = MagicMock()
            mock_issue.issues = ["High token usage", "Slow response"]
            mock_check.return_value = mock_issue
            result = await agent.run()

        assert result == "ok"

    async def test_no_health_issues(self, mock_llm, temp_workspace):
        mock_llm.generate = AsyncMock(return_value=_make_response("ok"))
        agent = _make_agent(mock_llm, temp_workspace)
        agent.add_user_message("hi")

        with patch.object(agent._health_checker, "check") as mock_check:
            mock_issue = MagicMock()
            mock_issue.issues = []
            mock_check.return_value = mock_issue
            result = await agent.run()

        assert result == "ok"


@pytest.mark.mock
class TestRunOptimizedToolCalls:
    async def test_fewer_results_than_tool_calls(self, mock_llm, temp_workspace):
        tool = FakeTool("test_tool")
        tc1 = _make_tool_call("test_tool", call_id="c1")
        tc2 = _make_tool_call("test_tool", call_id="c2")

        mock_llm.generate = AsyncMock(
            side_effect=[
                _make_response("Using tools", tool_calls=[tc1, tc2]),
                _make_response("Done!"),
            ]
        )
        agent = _make_agent(mock_llm, temp_workspace, tools=[tool])
        agent.add_user_message("Use tools")

        async def execute_fewer(*args, **kwargs):
            tool_msg = Message(
                role="tool",
                content="Result",
                tool_call_id="c1",
                name="test_tool",
            )
            return [(tc1, tool_msg)]

        with patch.object(agent._execution_engine, "execute_tools", side_effect=execute_fewer):
            result = await agent.run()

        assert result == "Done!"
        assistants_with_tc = [m for m in agent.messages if m.role == "assistant" and m.tool_calls is not None]
        assert len(assistants_with_tc) == 1
        assert len(assistants_with_tc[0].tool_calls) == 1
        assert assistants_with_tc[0].tool_calls[0].id == "c1"


@pytest.mark.mock
class TestRunMessageSummarization:
    async def test_summarize_called_each_step(self, mock_llm, temp_workspace):
        tool = FakeTool("test_tool")
        tc = _make_tool_call("test_tool")
        mock_llm.generate = AsyncMock(
            side_effect=[
                _make_response("Using tool", tool_calls=[tc]),
                _make_response("Done!"),
            ]
        )
        agent = _make_agent(mock_llm, temp_workspace, tools=[tool])
        agent.add_user_message("hi")

        with patch.object(agent, "_summarize_messages", new_callable=AsyncMock) as mock_sum:
            await agent.run()
            assert mock_sum.call_count >= 2

    async def test_cancel_event_passed_to_run(self, mock_llm, temp_workspace):
        mock_llm.generate = AsyncMock(return_value=_make_response("ok"))
        agent = _make_agent(mock_llm, temp_workspace)
        agent.add_user_message("hi")

        cancel_event = asyncio.Event()
        await agent.run(cancel_event=cancel_event)
        assert agent.cancel_event is cancel_event


@pytest.mark.mock
class TestRunStepRunnerIntegration:
    async def test_step_runner_process_response(self, mock_llm, temp_workspace):
        response = LLMResponse(
            content="ok",
            thinking=None,
            tool_calls=None,
            finish_reason="end_turn",
            usage=TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
        )
        mock_llm.generate = AsyncMock(return_value=response)
        agent = _make_agent(mock_llm, temp_workspace)
        agent.add_user_message("hi")
        await agent.run()
        assert agent.api_total_tokens == 150
        assert agent.api_call_count == 1

    async def test_step_runner_auto_save_on_completion(self, mock_llm, temp_workspace):
        mock_llm.generate = AsyncMock(return_value=_make_response("done"))
        agent = _make_agent(mock_llm, temp_workspace)
        agent.auto_save = True
        agent.add_user_message("hi")

        with patch.object(agent._session_manager, "save", return_value="save_id") as mock_save:
            await agent.run()
            mock_save.assert_called()

    async def test_step_timing_recorded(self, mock_llm, temp_workspace):
        mock_llm.generate = AsyncMock(return_value=_make_response("ok"))
        agent = _make_agent(mock_llm, temp_workspace)
        agent.add_user_message("hi")
        await agent.run()
        metrics = agent.get_performance_metrics()
        assert "step_metrics" in metrics


@pytest.mark.mock
class TestRunCheckCancelled:
    async def test_check_cancelled_returns_false_when_no_event(self, mock_llm, temp_workspace):
        agent = _make_agent(mock_llm, temp_workspace)
        assert agent._check_cancelled() is False

    async def test_check_cancelled_returns_false_when_not_set(self, mock_llm, temp_workspace):
        agent = _make_agent(mock_llm, temp_workspace)
        agent.cancel_event = asyncio.Event()
        assert agent._check_cancelled() is False

    async def test_check_cancelled_returns_true_when_set(self, mock_llm, temp_workspace):
        agent = _make_agent(mock_llm, temp_workspace)
        agent.cancel_event = asyncio.Event()
        agent.cancel_event.set()
        assert agent._check_cancelled() is True


@pytest.mark.mock
class TestRunCleanupIncompleteMessages:
    async def test_cleanup_removes_last_assistant_and_tool_messages(self, mock_llm, temp_workspace):
        agent = _make_agent(mock_llm, temp_workspace)
        initial_count = len(agent.messages)
        agent.messages.append(Message(role="assistant", content="partial", tool_calls=[_make_tool_call()]))
        agent.messages.append(Message(role="tool", content="result", tool_call_id="c1", name="test"))
        assert len(agent.messages) == initial_count + 2
        agent._cleanup_incomplete_messages()
        assert len(agent.messages) == initial_count

    async def test_cleanup_no_assistant_message(self, mock_llm, temp_workspace):
        agent = _make_agent(mock_llm, temp_workspace)
        initial_count = len(agent.messages)
        agent._cleanup_incomplete_messages()
        assert len(agent.messages) == initial_count

    async def test_cleanup_preserves_earlier_messages(self, mock_llm, temp_workspace):
        agent = _make_agent(mock_llm, temp_workspace)
        agent.messages.append(Message(role="user", content="keep me"))
        agent.messages.append(Message(role="assistant", content="partial"))
        agent._cleanup_incomplete_messages()
        roles = [m.role for m in agent.messages]
        assert "user" in roles
