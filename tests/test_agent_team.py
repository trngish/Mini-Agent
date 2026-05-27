"""Comprehensive unit tests for AgentTeam class.

Tests cover:
- __init__ with various configurations
- add_agent / add_critic / add_reviewer / add_planner / add_executor
- remove_agent (via agents property)
- broadcast (via message_bus)
- send_message (via message_bus)
- get_agent_status (via agents property and metrics)
- start/stop (via execute lifecycle)
- run_task (execute method with various scenarios)
"""

import asyncio
from collections import defaultdict
from unittest.mock import AsyncMock, MagicMock

import pytest

from mini_agent.schema import FunctionCall, LLMResponse, TokenUsage, ToolCall
from mini_agent.team.agent_team import AgentMember, AgentTeam, TeamResult
from mini_agent.team.message_bus import MessageBus, MessagePriority, MessageType
from mini_agent.team.roles import AgentRole
from mini_agent.tools.base import Tool, ToolResult


class FakeTool(Tool):
    """A fake tool for testing."""

    def __init__(self, name: str = "fake_tool", fail: bool = False):
        self._name = name
        self._fail = fail

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return f"Fake tool: {self._name}"

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {"input": {"type": "string"}},
            "required": ["input"],
        }

    async def execute(self, input: str = "") -> ToolResult:
        if self._fail:
            return ToolResult(success=False, content="", error="Tool failed")
        return ToolResult(success=True, content=f"Result for {input}")


@pytest.fixture
def mock_llm():
    """Create a mock LLM client with clone support."""
    llm = MagicMock()
    llm.model = "test-model"
    llm.api_key = "test-key"
    llm.api_base = "https://api.test.com"
    llm.generate = AsyncMock()

    def _make_clone():
        c = MagicMock()
        c.generate = AsyncMock()
        c.model = "test-model-cloned"
        return c

    llm.clone.side_effect = _make_clone
    return llm


@pytest.fixture
def tools():
    """Create a list of fake tools."""
    return [FakeTool("read_file"), FakeTool("write_file"), FakeTool("bash")]


@pytest.fixture
def team(mock_llm, tools):
    """Create a basic AgentTeam instance."""
    return AgentTeam(
        llm_client=mock_llm,
        tools=tools,
        system_prompt="You are a team member.",
        max_concurrent=2,
        enable_adversarial=False,
        critique_rounds=1,
    )


def _make_response(content: str = "", tool_calls: list | None = None) -> LLMResponse:
    return LLMResponse(
        content=content,
        thinking=None,
        tool_calls=tool_calls,
        finish_reason="end_turn" if not tool_calls else "tool_use",
        usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


class TestAgentTeamInit:
    """Test AgentTeam.__init__."""

    def test_default_init(self, mock_llm, tools):
        team = AgentTeam(llm_client=mock_llm, tools=tools)
        assert team.base_llm is mock_llm
        assert team.base_tools is tools
        assert team.system_prompt == "You are a helpful AI assistant on a software engineering team."
        assert team.max_concurrent == 3
        assert team.enable_adversarial is True
        assert team.critique_rounds == 2
        assert team.m27_config == {}
        assert team._agents == {}
        assert isinstance(team._bus, MessageBus)
        assert team._current_task == ""
        assert team._task_id == 0
        assert isinstance(team._metrics, defaultdict)

    def test_custom_init(self, mock_llm, tools):
        m27 = {"thinking_budget": 8192}
        team = AgentTeam(
            llm_client=mock_llm,
            tools=tools,
            system_prompt="Custom prompt",
            max_concurrent=5,
            enable_adversarial=False,
            critique_rounds=3,
            m27_config=m27,
        )
        assert team.system_prompt == "Custom prompt"
        assert team.max_concurrent == 5
        assert team.enable_adversarial is False
        assert team.critique_rounds == 3
        assert team.m27_config == m27

    def test_init_with_none_m27_config(self, mock_llm, tools):
        team = AgentTeam(llm_client=mock_llm, tools=tools, m27_config=None)
        assert team.m27_config == {}

    def test_len_empty(self, team):
        assert len(team) == 0

    def test_repr_empty(self, team):
        r = repr(team)
        assert r == "AgentTeam([])"


class TestAddAgent:
    """Test AgentTeam.add_agent and convenience methods."""

    def test_add_agent_basic(self, team):
        team.add_agent("planner", AgentRole.PLANNER)
        assert "planner" in team._agents
        assert team._agents["planner"].role == AgentRole.PLANNER
        assert len(team) == 1

    def test_add_agent_duplicate_name_raises(self, team):
        team.add_agent("planner", AgentRole.PLANNER)
        with pytest.raises(ValueError, match="already exists"):
            team.add_agent("planner", AgentRole.EXECUTOR)

    def test_add_agent_clones_llm(self, team):
        team.add_agent("exec", AgentRole.EXECUTOR)
        team.base_llm.clone.assert_called_once()

    def test_add_agent_without_clone_method(self, mock_llm, tools):
        del mock_llm.clone
        team = AgentTeam(llm_client=mock_llm, tools=tools)
        team.add_agent("exec", AgentRole.EXECUTOR)
        assert "exec" in team._agents
        assert team._agents["exec"].llm_client is mock_llm

    def test_add_agent_with_tool_filter(self, team):
        team.add_agent("exec", AgentRole.EXECUTOR, tool_names=["read_file", "bash"])
        agent = team._agents["exec"]
        tool_names = [t.name for t in agent.tools]
        assert "read_file" in tool_names
        assert "bash" in tool_names
        assert "write_file" not in tool_names

    def test_add_agent_with_empty_tool_filter(self, team):
        team.add_agent("exec", AgentRole.EXECUTOR, tool_names=["nonexistent"])
        agent = team._agents["exec"]
        assert agent.tools == []

    def test_add_agent_with_system_prompt_addition(self, team):
        team.add_agent("planner", AgentRole.PLANNER, system_prompt_addition="Focus on security.")
        agent = team._agents["planner"]
        assert "Focus on security." in agent.system_prompt

    def test_add_agent_custom_max_steps(self, team):
        team.add_agent("exec", AgentRole.EXECUTOR, max_steps=10)
        assert team._agents["exec"].max_steps == 10

    def test_add_agent_default_max_steps(self, team):
        team.add_agent("exec", AgentRole.EXECUTOR)
        assert team._agents["exec"].max_steps == 50

    def test_add_agent_system_prompt_includes_role_suffix(self, team):
        team.add_agent("critic", AgentRole.CRITIC)
        agent = team._agents["critic"]
        assert "Critic" in agent.system_prompt

    def test_add_agent_m27_config_propagated(self, mock_llm, tools):
        m27 = {"thinking_budget": 4096}
        team = AgentTeam(llm_client=mock_llm, tools=tools, m27_config=m27)
        team.add_agent("exec", AgentRole.EXECUTOR)
        assert team._agents["exec"].m27_config == m27

    def test_add_critic(self, team):
        team.add_critic()
        assert "critic" in team._agents
        assert team._agents["critic"].role == AgentRole.CRITIC
        assert team._agents["critic"].max_steps == 30

    def test_add_critic_custom_name(self, team):
        team.add_critic(name="security_critic", max_steps=20)
        assert "security_critic" in team._agents
        assert team._agents["security_critic"].role == AgentRole.CRITIC
        assert team._agents["security_critic"].max_steps == 20

    def test_add_reviewer(self, team):
        team.add_reviewer()
        assert "reviewer" in team._agents
        assert team._agents["reviewer"].role == AgentRole.REVIEWER
        assert team._agents["reviewer"].max_steps == 30

    def test_add_reviewer_custom_name(self, team):
        team.add_reviewer(name="code_reviewer", max_steps=15)
        assert "code_reviewer" in team._agents
        assert team._agents["code_reviewer"].max_steps == 15

    def test_add_planner(self, team):
        team.add_planner()
        assert "planner" in team._agents
        assert team._agents["planner"].role == AgentRole.PLANNER
        assert team._agents["planner"].max_steps == 40

    def test_add_planner_custom_name(self, team):
        team.add_planner(name="architect", max_steps=25)
        assert "architect" in team._agents
        assert team._agents["architect"].max_steps == 25

    def test_add_executor(self, team):
        team.add_executor()
        assert "executor" in team._agents
        assert team._agents["executor"].role == AgentRole.EXECUTOR
        assert team._agents["executor"].max_steps == 50

    def test_add_executor_custom_name(self, team):
        team.add_executor(name="coder", max_steps=100)
        assert "coder" in team._agents
        assert team._agents["coder"].max_steps == 100

    def test_add_multiple_agents(self, team):
        team.add_planner()
        team.add_executor()
        team.add_reviewer()
        team.add_critic()
        assert len(team) == 4

    def test_repr_with_agents(self, team):
        team.add_planner()
        team.add_executor()
        r = repr(team)
        assert "planner(planner)" in r
        assert "executor(executor)" in r


class TestRemoveAgent:
    """Test agent removal via the agents property and direct manipulation."""

    def test_agents_property_returns_copy(self, team):
        team.add_agent("exec", AgentRole.EXECUTOR)
        agents = team.agents
        agents["exec"] = None
        assert team._agents["exec"] is not None

    def test_remove_agent_by_deleting_from_internal_dict(self, team):
        team.add_agent("exec", AgentRole.EXECUTOR)
        team.add_agent("planner", AgentRole.PLANNER)
        assert len(team) == 2
        del team._agents["exec"]
        assert "exec" not in team._agents
        assert len(team) == 1

    def test_len_after_removal(self, team):
        team.add_agent("exec", AgentRole.EXECUTOR)
        team.add_agent("planner", AgentRole.PLANNER)
        del team._agents["exec"]
        assert len(team) == 1


class TestBroadcast:
    """Test broadcast messaging via the message bus."""

    def test_broadcast_via_message_bus(self, team):
        msg_id = team.message_bus.broadcast("coordinator", "New task available")
        assert msg_id is not None
        assert team.message_bus.has_messages("planner")
        assert team.message_bus.has_messages("executor")

    def test_broadcast_with_priority(self, team):
        team.add_agent("exec", AgentRole.EXECUTOR)
        msg_id = team.message_bus.broadcast(
            "coordinator",
            "Urgent update",
            priority=MessagePriority.CRITICAL,
        )
        assert msg_id is not None
        messages = team.message_bus.receive("exec")
        assert len(messages) >= 1
        assert messages[0].priority == MessagePriority.CRITICAL

    def test_broadcast_with_metadata(self, team):
        team.add_agent("exec", AgentRole.EXECUTOR)
        team.message_bus.broadcast(
            "coordinator",
            "Task update",
            metadata={"task_id": 42},
        )
        messages = team.message_bus.receive("exec")
        assert any(m.metadata.get("task_id") == 42 for m in messages)

    def test_multiple_broadcasts(self, team):
        team.add_agent("exec", AgentRole.EXECUTOR)
        team.message_bus.broadcast("coordinator", "First")
        team.message_bus.broadcast("coordinator", "Second")
        messages = team.message_bus.receive("exec")
        assert len(messages) == 2

    def test_broadcast_type_is_broadcast(self, team):
        team.add_agent("exec", AgentRole.EXECUTOR)
        team.message_bus.broadcast("coordinator", "Hello team")
        messages = team.message_bus.receive("exec")
        assert all(m.type == MessageType.BROADCAST for m in messages)


class TestSendMessage:
    """Test point-to-point messaging via the message bus."""

    def test_send_direct_message(self, team):
        team.add_agent("exec", AgentRole.EXECUTOR)
        team.add_agent("planner", AgentRole.PLANNER)
        msg_id = team.message_bus.send_task("planner", "exec", "Implement feature X")
        assert msg_id is not None
        messages = team.message_bus.receive("exec")
        assert len(messages) == 1
        assert messages[0].type == MessageType.TASK
        assert messages[0].content == "Implement feature X"
        assert messages[0].sender == "planner"

    def test_send_result_message(self, team):
        team.add_agent("exec", AgentRole.EXECUTOR)
        team.add_agent("planner", AgentRole.PLANNER)
        team.message_bus.send_result("exec", "planner", "Feature implemented", success=True)
        messages = team.message_bus.receive("planner")
        assert len(messages) == 1
        assert messages[0].type == MessageType.RESULT
        assert messages[0].metadata.get("success") is True

    def test_send_message_not_received_by_others(self, team):
        team.add_agent("exec", AgentRole.EXECUTOR)
        team.add_agent("planner", AgentRole.PLANNER)
        team.message_bus.send_task("planner", "exec", "Do something")
        exec_msgs = team.message_bus.receive("exec")
        planner_msgs = team.message_bus.receive("planner")
        assert len(exec_msgs) == 1
        assert len(planner_msgs) == 0

    def test_send_with_priority(self, team):
        team.add_agent("exec", AgentRole.EXECUTOR)
        team.message_bus.send_task("coordinator", "exec", "Low task", priority=MessagePriority.LOW)
        team.message_bus.send_task("coordinator", "exec", "High task", priority=MessagePriority.HIGH)
        messages = team.message_bus.receive("exec")
        assert messages[0].content == "High task"
        assert messages[1].content == "Low task"

    def test_peek_does_not_remove(self, team):
        team.add_agent("exec", AgentRole.EXECUTOR)
        team.message_bus.send_task("coordinator", "exec", "Task 1")
        peeked = team.message_bus.peek("exec")
        assert len(peeked) == 1
        messages = team.message_bus.receive("exec")
        assert len(messages) == 1

    def test_has_messages(self, team):
        team.add_agent("exec", AgentRole.EXECUTOR)
        assert not team.message_bus.has_messages("exec")
        team.message_bus.send_task("coordinator", "exec", "Task")
        assert team.message_bus.has_messages("exec")

    def test_clear_messages(self, team):
        team.add_agent("exec", AgentRole.EXECUTOR)
        team.message_bus.send_task("coordinator", "exec", "Task")
        count = team.message_bus.clear("exec")
        assert count == 1
        assert not team.message_bus.has_messages("exec")

    def test_clear_all_messages(self, team):
        team.add_agent("exec", AgentRole.EXECUTOR)
        team.add_agent("planner", AgentRole.PLANNER)
        team.message_bus.send_task("coordinator", "exec", "Task 1")
        team.message_bus.send_task("coordinator", "planner", "Task 2")
        count = team.message_bus.clear("*")
        assert count == 2


class TestGetAgentStatus:
    """Test agent status retrieval."""

    def test_agents_property_empty(self, team):
        assert team.agents == {}

    def test_agents_property_with_members(self, team):
        team.add_agent("exec", AgentRole.EXECUTOR)
        team.add_agent("planner", AgentRole.PLANNER)
        agents = team.agents
        assert len(agents) == 2
        assert "exec" in agents
        assert "planner" in agents

    def test_agent_member_attributes(self, team):
        team.add_agent("exec", AgentRole.EXECUTOR, max_steps=10)
        agent = team.agents["exec"]
        assert agent.name == "exec"
        assert agent.role == AgentRole.EXECUTOR
        assert agent.is_active is True
        assert agent.max_steps == 10

    def test_agent_member_tools_dict(self, team):
        team.add_agent("exec", AgentRole.EXECUTOR)
        agent = team.agents["exec"]
        assert "read_file" in agent.tools_dict
        assert "write_file" in agent.tools_dict
        assert "bash" in agent.tools_dict

    def test_agent_member_get_system_prompt(self, team):
        team.add_agent("exec", AgentRole.EXECUTOR)
        agent = team.agents["exec"]
        prompt = agent.get_system_prompt()
        assert "Executor" in prompt

    def test_get_metrics_empty(self, team):
        metrics = team.get_metrics()
        assert metrics == {}

    def test_message_bus_property(self, team):
        bus = team.message_bus
        assert isinstance(bus, MessageBus)


class TestExecute:
    """Test AgentTeam.execute (run_task)."""

    @pytest.mark.asyncio
    async def test_execute_no_agents(self, team):
        result = await team.execute("Do something")
        assert result.success is False
        assert result.error == "No agents available to execute task"
        assert result.content == ""
        assert result.agent_results == {}

    @pytest.mark.asyncio
    async def test_execute_with_planner_only(self, team):
        team.add_planner()
        team._agents["planner"].llm_client.generate = AsyncMock(
            return_value=_make_response("Plan: Step 1, Step 2, Step 3")
        )
        result = await team.execute("Create a plan")
        assert result.success is True
        assert "planner" in result.agent_results
        assert result.iterations >= 1
        assert result.elapsed > 0

    @pytest.mark.asyncio
    async def test_execute_with_executor_only(self, team):
        team.add_executor()
        team._agents["executor"].llm_client.generate = AsyncMock(return_value=_make_response("Implementation complete"))
        result = await team.execute("Write code")
        assert result.success is True
        assert "executor" in result.agent_results

    @pytest.mark.asyncio
    async def test_execute_with_planner_and_executor(self, team):
        team.add_planner()
        team.add_executor()
        team._agents["planner"].llm_client.generate = AsyncMock(return_value=_make_response("Plan created"))
        team._agents["executor"].llm_client.generate = AsyncMock(return_value=_make_response("Code written"))
        result = await team.execute("Build feature")
        assert result.success is True
        assert "planner" in result.agent_results
        assert "executor" in result.agent_results
        assert result.iterations >= 2

    @pytest.mark.asyncio
    async def test_execute_with_initial_roles_filter(self, team):
        team.add_planner()
        team.add_executor()
        team.add_reviewer()
        team._agents["planner"].llm_client.generate = AsyncMock(return_value=_make_response("Plan"))
        team._agents["executor"].llm_client.generate = AsyncMock(return_value=_make_response("Done"))
        result = await team.execute(
            "Build feature",
            initial_roles=[AgentRole.PLANNER],
        )
        assert result.success is True
        assert "planner" in result.agent_results
        assert "executor" not in result.agent_results

    @pytest.mark.asyncio
    async def test_execute_with_reviewer_and_critic(self, team):
        team.add_reviewer()
        team.add_critic()
        team._agents["reviewer"].llm_client.generate = AsyncMock(return_value=_make_response("Review looks good"))
        team._agents["critic"].llm_client.generate = AsyncMock(
            return_value=_make_response("Critique: consider edge cases")
        )
        result = await team.execute("Review this code")
        assert result.success is True
        assert "reviewer" in result.agent_results

    @pytest.mark.asyncio
    async def test_execute_adversarial_enabled(self, mock_llm, tools):
        team = AgentTeam(
            llm_client=mock_llm,
            tools=tools,
            enable_adversarial=True,
            critique_rounds=2,
        )
        team.add_planner()
        team.add_critic()
        team._agents["planner"].llm_client.generate = AsyncMock(return_value=_make_response("Plan done"))
        team._agents["critic"].llm_client.generate = AsyncMock(return_value=_make_response("Critique done"))
        result = await team.execute("Complex task")
        assert result.success is True
        assert result.iterations >= 3

    @pytest.mark.asyncio
    async def test_execute_adversarial_no_critics(self, mock_llm, tools):
        team = AgentTeam(
            llm_client=mock_llm,
            tools=tools,
            enable_adversarial=True,
            critique_rounds=2,
        )
        team.add_planner()
        team._agents["planner"].llm_client.generate = AsyncMock(return_value=_make_response("Plan done"))
        result = await team.execute("Task without critics")
        assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_with_coordinator_synthesis(self, team):
        team.add_agent("coordinator", AgentRole.COORDINATOR)
        team.add_planner()
        team._agents["coordinator"].llm_client.generate = AsyncMock(
            return_value=_make_response("Synthesized final result")
        )
        team._agents["planner"].llm_client.generate = AsyncMock(return_value=_make_response("Plan analysis"))
        result = await team.execute("Complex task")
        assert result.success is True
        assert "Synthesized final result" in result.content

    @pytest.mark.asyncio
    async def test_execute_without_coordinator_simple_synthesis(self, team):
        team.add_planner()
        team._agents["planner"].llm_client.generate = AsyncMock(return_value=_make_response("Plan analysis"))
        result = await team.execute("Simple task")
        assert result.success is True
        assert "Team Results" in result.content

    @pytest.mark.asyncio
    async def test_execute_agent_failure_graceful(self, team):
        team.add_planner()
        team._agents["planner"].llm_client.generate = AsyncMock(side_effect=RuntimeError("LLM unavailable"))
        result = await team.execute("Failing task")
        assert "Error" in result.agent_results.get("planner", "")
        assert "LLM unavailable" in result.agent_results["planner"]

    @pytest.mark.asyncio
    async def test_execute_updates_task_state(self, team):
        team.add_planner()
        team._agents["planner"].llm_client.generate = AsyncMock(return_value=_make_response("Done"))
        assert team._current_task == ""
        assert team._task_id == 0
        await team.execute("New task")
        assert team._current_task == "New task"
        assert team._task_id == 1

    @pytest.mark.asyncio
    async def test_execute_task_id_increments(self, team):
        team.add_planner()
        team._agents["planner"].llm_client.generate = AsyncMock(return_value=_make_response("Done"))
        await team.execute("Task 1")
        await team.execute("Task 2")
        assert team._task_id == 2

    @pytest.mark.asyncio
    async def test_execute_with_timeout(self, team):
        team.add_planner()

        async def slow_generate(*args, **kwargs):
            await asyncio.sleep(10)
            return _make_response("Too slow")

        team._agents["planner"].llm_client.generate = AsyncMock(side_effect=slow_generate)
        result = await team.execute("Slow task", timeout=1)
        assert result.success is False
        assert "timed out" in result.error.lower()

    @pytest.mark.asyncio
    async def test_execute_result_has_elapsed(self, team):
        team.add_planner()
        team._agents["planner"].llm_client.generate = AsyncMock(return_value=_make_response("Done"))
        result = await team.execute("Task")
        assert result.elapsed > 0

    @pytest.mark.asyncio
    async def test_execute_consensus_on_success(self, team):
        team.add_planner()
        team._agents["planner"].llm_client.generate = AsyncMock(return_value=_make_response("Done"))
        result = await team.execute("Task")
        assert result.consensus is True

    @pytest.mark.asyncio
    async def test_execute_no_consensus_on_failure(self, team):
        result = await team.execute("Task with no agents")
        assert result.consensus is False


class TestRunSingleAgent:
    """Test _run_single_agent method."""

    @pytest.mark.asyncio
    async def test_run_single_agent_text_response(self, team):
        team.add_agent("exec", AgentRole.EXECUTOR)
        agent = team._agents["exec"]
        agent.llm_client.generate = AsyncMock(return_value=_make_response("Task complete"))
        result = await team._run_single_agent(agent, "Do something")
        assert result == "Task complete"

    @pytest.mark.asyncio
    async def test_run_single_agent_with_tool_calls(self, team):
        team.add_agent("exec", AgentRole.EXECUTOR)
        agent = team._agents["exec"]
        tool_call = ToolCall(
            id="tc_1",
            type="function",
            function=FunctionCall(name="read_file", arguments={"input": "test.py"}),
        )
        agent.llm_client.generate = AsyncMock(
            side_effect=[
                _make_response("Using tool...", tool_calls=[tool_call]),
                _make_response("Final answer after tool use"),
            ]
        )
        result = await team._run_single_agent(agent, "Read a file")
        assert result == "Final answer after tool use"

    @pytest.mark.asyncio
    async def test_run_single_agent_max_steps(self, team):
        team.add_agent("exec", AgentRole.EXECUTOR, max_steps=2)
        agent = team._agents["exec"]
        tool_call = ToolCall(
            id="tc_loop",
            type="function",
            function=FunctionCall(name="read_file", arguments={"input": "loop"}),
        )
        agent.llm_client.generate = AsyncMock(return_value=_make_response("Still working...", tool_calls=[tool_call]))
        result = await team._run_single_agent(agent, "Loop task")
        assert result == "Max steps reached without completion"

    @pytest.mark.asyncio
    async def test_run_single_agent_empty_content(self, team):
        team.add_agent("exec", AgentRole.EXECUTOR)
        agent = team._agents["exec"]
        agent.llm_client.generate = AsyncMock(return_value=_make_response(""))
        result = await team._run_single_agent(agent, "Empty task")
        assert result == ""


class TestExecuteToolsForAgent:
    """Test _execute_tools_for_agent method."""

    @pytest.mark.asyncio
    async def test_execute_known_tool(self, team):
        team.add_agent("exec", AgentRole.EXECUTOR)
        agent = team._agents["exec"]
        tc = ToolCall(
            id="tc_1",
            type="function",
            function=FunctionCall(name="read_file", arguments={"input": "test.py"}),
        )
        results = await team._execute_tools_for_agent(agent, [tc])
        assert len(results) == 1
        _, tool_msg = results[0]
        assert tool_msg.role == "tool"
        assert "Result for test.py" in tool_msg.content

    @pytest.mark.asyncio
    async def test_execute_unknown_tool(self, team):
        team.add_agent("exec", AgentRole.EXECUTOR)
        agent = team._agents["exec"]
        tc = ToolCall(
            id="tc_unk",
            type="function",
            function=FunctionCall(name="nonexistent_tool", arguments={}),
        )
        results = await team._execute_tools_for_agent(agent, [tc])
        assert len(results) == 1
        _, tool_msg = results[0]
        assert "Unknown tool" in tool_msg.content

    @pytest.mark.asyncio
    async def test_execute_failing_tool(self, team):
        fail_tool = FakeTool("fail_tool", fail=True)
        team.base_tools.append(fail_tool)
        team.add_agent("exec", AgentRole.EXECUTOR)
        agent = team._agents["exec"]
        tc = ToolCall(
            id="tc_fail",
            type="function",
            function=FunctionCall(name="fail_tool", arguments={"input": "test"}),
        )
        results = await team._execute_tools_for_agent(agent, [tc])
        assert len(results) == 1
        _, tool_msg = results[0]
        assert "Error" in tool_msg.content

    @pytest.mark.asyncio
    async def test_execute_multiple_tool_calls(self, team):
        team.add_agent("exec", AgentRole.EXECUTOR)
        agent = team._agents["exec"]
        tc1 = ToolCall(
            id="tc_1",
            type="function",
            function=FunctionCall(name="read_file", arguments={"input": "a.py"}),
        )
        tc2 = ToolCall(
            id="tc_2",
            type="function",
            function=FunctionCall(name="bash", arguments={"input": "ls"}),
        )
        results = await team._execute_tools_for_agent(agent, [tc1, tc2])
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_tool_message_has_correct_call_id(self, team):
        team.add_agent("exec", AgentRole.EXECUTOR)
        agent = team._agents["exec"]
        tc = ToolCall(
            id="tc_special_id",
            type="function",
            function=FunctionCall(name="read_file", arguments={"input": "test"}),
        )
        results = await team._execute_tools_for_agent(agent, [tc])
        _, tool_msg = results[0]
        assert tool_msg.tool_call_id == "tc_special_id"


class TestRunAgentsParallel:
    """Test _run_agents_parallel method."""

    @pytest.mark.asyncio
    async def test_parallel_execution(self, team):
        team.add_planner()
        team.add_reviewer()
        team._agents["planner"].llm_client.generate = AsyncMock(return_value=_make_response("Plan result"))
        team._agents["reviewer"].llm_client.generate = AsyncMock(return_value=_make_response("Review result"))
        agents = {n: a for n, a in team._agents.items()}
        results = await team._run_agents_parallel(agents, "Analyze this")
        assert results["planner"] == "Plan result"
        assert results["reviewer"] == "Review result"

    @pytest.mark.asyncio
    async def test_parallel_with_agent_error(self, team):
        team.add_planner()
        team.add_reviewer()
        team._agents["planner"].llm_client.generate = AsyncMock(side_effect=RuntimeError("Planner crashed"))
        team._agents["reviewer"].llm_client.generate = AsyncMock(return_value=_make_response("Review ok"))
        agents = {n: a for n, a in team._agents.items()}
        results = await team._run_agents_parallel(agents, "Analyze this")
        assert "Error" in results["planner"]
        assert results["reviewer"] == "Review ok"

    @pytest.mark.asyncio
    async def test_parallel_empty_agents(self, team):
        results = await team._run_agents_parallel({}, "Empty task")
        assert results == {}

    @pytest.mark.asyncio
    async def test_parallel_respects_max_concurrent(self, mock_llm, tools):
        team = AgentTeam(
            llm_client=mock_llm,
            tools=tools,
            max_concurrent=1,
        )
        team.add_agent("a1", AgentRole.PLANNER)
        team.add_agent("a2", AgentRole.REVIEWER)
        team.add_agent("a3", AgentRole.RESEARCHER)
        for agent in team._agents.values():
            agent.llm_client.generate = AsyncMock(return_value=_make_response("Done"))
        agents = {n: a for n, a in team._agents.items()}
        results = await team._run_agents_parallel(agents, "Task")
        assert len(results) == 3


class TestBuildContextForExecutor:
    """Test _build_context_for_executor method."""

    def test_build_context_empty(self, team):
        context = team._build_context_for_executor({})
        assert "Previous Analysis" in context

    def test_build_context_with_results(self, team):
        results = {
            "planner": "Step 1: Design. Step 2: Implement.",
            "reviewer": "Looks good overall.",
        }
        context = team._build_context_for_executor(results)
        assert "Planner Analysis" in context
        assert "Reviewer Analysis" in context
        assert "Step 1" in context

    def test_build_context_truncates_long_content(self, team):
        long_content = "x" * 1000
        results = {"planner": long_content}
        context = team._build_context_for_executor(results)
        assert len(context) < len(long_content) + 100


class TestRunCritiqueRound:
    """Test _run_critique_round method."""

    @pytest.mark.asyncio
    async def test_critique_round_with_critics(self, team):
        team.add_critic()
        team._agents["critic"].llm_client.generate = AsyncMock(return_value=_make_response("Found potential issues"))
        active_agents = {n: a for n, a in team._agents.items()}
        results = await team._run_critique_round(active_agents, "Build feature", {"planner": "Plan"}, 0)
        assert "critic_critique_0" in results

    @pytest.mark.asyncio
    async def test_critique_round_no_critics(self, team):
        team.add_planner()
        active_agents = {n: a for n, a in team._agents.items()}
        results = await team._run_critique_round(active_agents, "Task", {"planner": "Plan"}, 0)
        assert results == {}

    @pytest.mark.asyncio
    async def test_critique_round_number_in_key(self, team):
        team.add_critic()
        team._agents["critic"].llm_client.generate = AsyncMock(return_value=_make_response("Round 2 critique"))
        active_agents = {n: a for n, a in team._agents.items()}
        results = await team._run_critique_round(active_agents, "Task", {}, 1)
        assert "critic_critique_1" in results


class TestSynthesizeResults:
    """Test _synthesize_results method."""

    @pytest.mark.asyncio
    async def test_synthesize_with_coordinator(self, team):
        team.add_agent("coordinator", AgentRole.COORDINATOR)
        team._agents["coordinator"].llm_client.generate = AsyncMock(
            return_value=_make_response("Final synthesized answer")
        )
        result = await team._synthesize_results("Task", {"planner": "Plan"})
        assert result == "Final synthesized answer"

    @pytest.mark.asyncio
    async def test_synthesize_without_coordinator(self, team):
        team.add_planner()
        result = await team._synthesize_results("Task", {"planner": "Plan", "reviewer": "Review"})
        assert "Team Results" in result
        assert "Planner" in result
        assert "Reviewer" in result

    @pytest.mark.asyncio
    async def test_synthesize_empty_results(self, team):
        team.add_planner()
        result = await team._synthesize_results("Task", {})
        assert "Team Results" in result


class TestTeamResult:
    """Test TeamResult dataclass."""

    def test_team_result_defaults(self):
        result = TeamResult(success=True, content="Done", agent_results={"a": "b"})
        assert result.consensus is False
        assert result.iterations == 0
        assert result.elapsed == 0.0
        assert result.error == ""

    def test_team_result_failure(self):
        result = TeamResult(
            success=False,
            content="",
            agent_results={},
            error="Something failed",
        )
        assert result.success is False
        assert result.error == "Something failed"


class TestAgentMember:
    """Test AgentMember dataclass."""

    def test_agent_member_init(self, mock_llm, tools):
        member = AgentMember(
            name="test_agent",
            role=AgentRole.EXECUTOR,
            llm_client=mock_llm,
            tools=tools,
            system_prompt="Test prompt",
        )
        assert member.name == "test_agent"
        assert member.role == AgentRole.EXECUTOR
        assert member.is_active is True
        assert member.max_steps == 50
        assert member.m27_config is None

    def test_agent_member_tools_dict(self, mock_llm, tools):
        member = AgentMember(
            name="test_agent",
            role=AgentRole.EXECUTOR,
            llm_client=mock_llm,
            tools=tools,
            system_prompt="Test",
        )
        assert "read_file" in member.tools_dict
        assert "write_file" in member.tools_dict
        assert "bash" in member.tools_dict

    def test_agent_member_get_system_prompt(self, mock_llm, tools):
        member = AgentMember(
            name="test_agent",
            role=AgentRole.PLANNER,
            llm_client=mock_llm,
            tools=tools,
            system_prompt="Base prompt",
        )
        prompt = member.get_system_prompt()
        assert "Base prompt" in prompt
        assert "Planner" in prompt

    def test_agent_member_empty_tools(self, mock_llm):
        member = AgentMember(
            name="test_agent",
            role=AgentRole.EXECUTOR,
            llm_client=mock_llm,
            tools=[],
            system_prompt="Test",
        )
        assert member.tools_dict == {}
