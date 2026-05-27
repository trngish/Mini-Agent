from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mini_agent.tools.team_dispatch_tool import TeamDispatchTool


@pytest.fixture
def mock_llm_client():
    client = MagicMock()
    client.model = "test-model"
    client.api_key = "test-key"
    return client


@pytest.fixture
def mock_tools():
    return []


@pytest.fixture
def tool(mock_llm_client, mock_tools):
    return TeamDispatchTool(
        llm_client=mock_llm_client,
        tools=mock_tools,
        system_prompt="Test prompt",
        m27_config={"key": "value"},
    )


class TestTeamDispatchToolInit:
    def test_name(self, tool):
        assert tool.name == "team_dispatch"

    def test_description(self, tool):
        assert isinstance(tool.description, str)
        assert len(tool.description) > 0
        assert "decompose" in tool.description
        assert "review" in tool.description

    def test_parameters(self, tool):
        params = tool.parameters
        assert params["type"] == "object"
        assert "task" in params["properties"]
        assert "mode" in params["properties"]
        assert params["properties"]["mode"]["enum"] == ["decompose", "review"]
        assert params["properties"]["mode"]["default"] == "decompose"
        assert "max_rounds" in params["properties"]
        assert "timeout_per_agent" in params["properties"]
        assert "task" in params["required"]

    def test_default_system_prompt(self, mock_llm_client, mock_tools):
        tool = TeamDispatchTool(llm_client=mock_llm_client, tools=mock_tools)
        assert tool._system_prompt == "You are a helpful AI assistant on a software engineering team."

    def test_default_m27_config(self, mock_llm_client, mock_tools):
        tool = TeamDispatchTool(llm_client=mock_llm_client, tools=mock_tools)
        assert tool._m27_config == {}

    def test_custom_m27_config(self, mock_llm_client, mock_tools):
        config = {"thinking_budget": 10000}
        tool = TeamDispatchTool(llm_client=mock_llm_client, tools=mock_tools, m27_config=config)
        assert tool._m27_config == config


class TestTeamDispatchToolExecuteDecompose:
    @pytest.mark.asyncio
    @patch("mini_agent.tools.team_dispatch_tool.AgentTeam")
    async def test_decompose_success(self, MockAgentTeam, tool):
        mock_team = MagicMock()
        mock_team.execute = AsyncMock(
            return_value=MagicMock(
                success=True,
                content="Task completed",
                agent_results={},
            )
        )
        MockAgentTeam.return_value = mock_team

        result = await tool.execute(task="Build a web app", mode="decompose")

        assert result.success is True
        assert "Task completed" in result.content
        mock_team.add_planner.assert_called_once_with("planner", max_steps=40)
        mock_team.add_executor.assert_called_once_with("executor", max_steps=50)
        mock_team.add_reviewer.assert_called_once_with("reviewer", max_steps=30)
        mock_team.execute.assert_awaited_once()

    @pytest.mark.asyncio
    @patch("mini_agent.tools.team_dispatch_tool.AgentTeam")
    async def test_decompose_failure(self, MockAgentTeam, tool):
        mock_team = MagicMock()
        mock_team.execute = AsyncMock(
            return_value=MagicMock(
                success=False,
                content="Partial work",
                error="Something went wrong",
            )
        )
        MockAgentTeam.return_value = mock_team

        result = await tool.execute(task="Build a web app", mode="decompose")

        assert result.success is False
        assert result.error == "Something went wrong"

    @pytest.mark.asyncio
    @patch("mini_agent.tools.team_dispatch_tool.AgentTeam")
    async def test_decompose_default_mode(self, MockAgentTeam, tool):
        mock_team = MagicMock()
        mock_team.execute = AsyncMock(
            return_value=MagicMock(
                success=True,
                content="Done",
                agent_results={},
            )
        )
        MockAgentTeam.return_value = mock_team

        result = await tool.execute(task="Some task")

        assert result.success is True
        mock_team.add_planner.assert_called_once()


class TestTeamDispatchToolExecuteReview:
    @pytest.mark.asyncio
    @patch("mini_agent.tools.team_dispatch_tool.AgentTeam")
    async def test_review_success(self, MockAgentTeam, tool):
        mock_team = MagicMock()
        mock_team.execute = AsyncMock(
            return_value=MagicMock(
                success=True,
                content="Final synthesis",
                agent_results={
                    "reviewer": "Looks good overall",
                    "critic1": "Consider edge cases",
                    "critic2": "Needs more tests",
                },
            )
        )
        MockAgentTeam.return_value = mock_team

        result = await tool.execute(task="Review this design", mode="review")

        assert result.success is True
        assert "Review Summary" in result.content
        assert "Final Synthesis" in result.content
        assert "reviewer" in result.content
        assert "critic1" in result.content
        assert "critic2" in result.content
        mock_team.add_reviewer.assert_called_once_with("reviewer", max_steps=30)
        mock_team.add_critic.assert_any_call("critic1", max_steps=30)
        mock_team.add_critic.assert_any_call("critic2", max_steps=30)

    @pytest.mark.asyncio
    @patch("mini_agent.tools.team_dispatch_tool.AgentTeam")
    async def test_review_failure(self, MockAgentTeam, tool):
        mock_team = MagicMock()
        mock_team.execute = AsyncMock(
            return_value=MagicMock(
                success=False,
                content="",
                agent_results={"reviewer": "Bad design"},
                error="Consensus not reached",
            )
        )
        MockAgentTeam.return_value = mock_team

        result = await tool.execute(task="Review this design", mode="review")

        assert result.success is False
        assert result.error == "Consensus not reached"

    @pytest.mark.asyncio
    @patch("mini_agent.tools.team_dispatch_tool.AgentTeam")
    async def test_review_with_custom_max_rounds(self, MockAgentTeam, tool):
        mock_team = MagicMock()
        mock_team.execute = AsyncMock(
            return_value=MagicMock(
                success=True,
                content="Synthesized",
                agent_results={},
            )
        )
        MockAgentTeam.return_value = mock_team

        result = await tool.execute(task="Review code", mode="review", max_rounds=5)

        assert result.success is True
        MockAgentTeam.assert_called_once()
        call_kwargs = MockAgentTeam.call_args
        assert call_kwargs.kwargs.get("critique_rounds", call_kwargs[1].get("critique_rounds")) == 5


class TestTeamDispatchToolInvalidMode:
    @pytest.mark.asyncio
    @patch("mini_agent.tools.team_dispatch_tool.AgentTeam")
    async def test_unknown_mode(self, MockAgentTeam, tool):
        mock_team = MagicMock()
        MockAgentTeam.return_value = mock_team

        result = await tool.execute(task="Do something", mode="unknown")

        assert result.success is False
        assert "Unknown mode" in result.error
        assert "unknown" in result.error


class TestTeamDispatchToolTimeout:
    @pytest.mark.asyncio
    @patch("mini_agent.tools.team_dispatch_tool.AgentTeam")
    async def test_timeout_handling(self, MockAgentTeam, tool):
        mock_team = MagicMock()

        async def slow_execute(*args, **kwargs):
            await asyncio.sleep(10)
            return MagicMock(success=True, content="Done", agent_results={})

        mock_team.execute = AsyncMock(side_effect=slow_execute)
        MockAgentTeam.return_value = mock_team

        with patch("mini_agent.tools.team_dispatch_tool.asyncio.timeout") as mock_timeout:
            mock_timeout.return_value.__aenter__ = AsyncMock(side_effect=asyncio.TimeoutError())
            mock_timeout.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await tool.execute(task="Slow task", timeout_per_agent=1)

        assert result.success is False
        assert "timed out" in result.error

    @pytest.mark.asyncio
    @patch("mini_agent.tools.team_dispatch_tool.AgentTeam")
    async def test_overall_timeout_calculation(self, MockAgentTeam, tool):
        mock_team = MagicMock()
        mock_team.execute = AsyncMock(
            return_value=MagicMock(
                success=True,
                content="Done",
                agent_results={},
            )
        )
        MockAgentTeam.return_value = mock_team

        await tool.execute(task="Test", timeout_per_agent=200)

        MockAgentTeam.assert_called_once()


class TestTeamDispatchToolErrorHandling:
    @pytest.mark.asyncio
    @patch("mini_agent.tools.team_dispatch_tool.AgentTeam")
    async def test_agent_team_raises_exception(self, MockAgentTeam, tool):
        MockAgentTeam.side_effect = RuntimeError("Failed to create team")

        result = await tool.execute(task="Test task")

        assert result.success is False
        assert "Team dispatch failed" in result.error
        assert "Failed to create team" in result.error

    @pytest.mark.asyncio
    @patch("mini_agent.tools.team_dispatch_tool.AgentTeam")
    async def test_agent_team_execute_raises_exception(self, MockAgentTeam, tool):
        mock_team = MagicMock()
        mock_team.execute = AsyncMock(side_effect=RuntimeError("Agent crashed"))
        mock_team.add_planner = MagicMock()
        mock_team.add_executor = MagicMock()
        mock_team.add_reviewer = MagicMock()
        MockAgentTeam.return_value = mock_team

        result = await tool.execute(task="Test task", mode="decompose")

        assert result.success is False
        assert "Team dispatch failed" in result.error
        assert "Agent crashed" in result.error

    @pytest.mark.asyncio
    @patch("mini_agent.tools.team_dispatch_tool.AgentTeam")
    async def test_agent_team_execute_returns_no_error(self, MockAgentTeam, tool):
        mock_team = MagicMock()
        mock_team.execute = AsyncMock(
            return_value=MagicMock(
                success=False,
                content="Partial",
                error="",
                agent_results={},
            )
        )
        MockAgentTeam.return_value = mock_team

        result = await tool.execute(task="Test task", mode="decompose")

        assert result.success is False
        assert result.error == "Task failed"
