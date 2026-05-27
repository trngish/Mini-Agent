"""Agent Team - Multi-agent collaboration with role boundaries and adversarial reasoning.

This module provides a team of agents that can work together on complex tasks,
with support for:
- Role-boundary isolation (coordinator, planner, executor, reviewer, critic)
- Parallel independent reasoning
- Adversarial reasoning (multiple critics challenge solutions)
- Result synthesis and consensus building
"""

import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from ..llm import LLMClient
from ..schema import Message
from ..tools.base import Tool
from .message_bus import MessageBus
from .roles import AgentRole, RoleConfig

logger = logging.getLogger(__name__)


@dataclass
class AgentMember:
    """A member of the agent team.

    Attributes:
        name: Unique identifier for this agent
        role: The role this agent plays
        llm_client: LLM client for this agent
        tools: Tools available to this agent
        system_prompt: Base system prompt
        max_steps: Maximum steps for execution
        m27_config: M2.7 configuration
        is_active: Whether this agent is currently active
    """

    name: str
    role: AgentRole
    llm_client: LLMClient
    tools: list[Tool]
    system_prompt: str
    max_steps: int = 50
    m27_config: dict[str, Any] | None = None
    is_active: bool = True
    _message_buffer: list[Message] = field(default_factory=list)
    _tools_dict: dict[str, Tool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Post-initialization setup."""
        self._tools_dict = {tool.name: tool for tool in self.tools}

    @property
    def tools_dict(self) -> dict[str, Tool]:
        """Get tools as a dict for quick lookup."""
        return self._tools_dict

    def get_system_prompt(self) -> str:
        """Get the full system prompt including role instructions."""
        role_config = RoleConfig(role=self.role)
        return role_config.get_system_prompt(self.system_prompt)


@dataclass
class TeamResult:
    """Result from a team execution.

    Attributes:
        success: Whether the team completed the task successfully
        content: Final synthesized content
        agent_results: Results from each agent
        consensus: Whether consensus was reached (if applicable)
        iterations: Number of reasoning iterations
        elapsed: Total execution time in seconds
        error: Error message if failed
    """

    success: bool
    content: str
    agent_results: dict[str, str]
    consensus: bool = False
    iterations: int = 0
    elapsed: float = 0.0
    error: str = ""


class AgentTeam:
    """Multi-agent team with role boundaries and adversarial reasoning.

    This class orchestrates a team of agents, each with a specific role,
    to collaborate on complex tasks. Key features:

    1. Role Assignment: Each agent has a clear role with defined boundaries
    2. Parallel Execution: Independent agents work in parallel
    3. Adversarial Reasoning: Critics challenge solutions to strengthen them
    4. Result Synthesis: Coordinator synthesizes results into final solution

    Example:
        team = AgentTeam(llm_client=client, tools=tools)

        team.add_agent("planner", AgentRole.PLANNER)
        team.add_agent("executor", AgentRole.EXECUTOR)
        team.add_agent("reviewer", AgentRole.REVIEWER)
        team.add_agent("critic", AgentRole.CRITIC)

        result = await team.execute("Implement a calculator in Python")
    """

    def __init__(
        self,
        llm_client: LLMClient,
        tools: list[Tool],
        system_prompt: str = "You are a helpful AI assistant on a software engineering team.",
        max_concurrent: int = 3,
        enable_adversarial: bool = True,
        critique_rounds: int = 2,
        m27_config: dict[str, Any] | None = None,
    ):
        """Initialize the agent team.

        Args:
            llm_client: LLM client for agents (will be cloned for each agent)
            tools: Tools available to team members
            system_prompt: Base system prompt for all agents
            max_concurrent: Maximum concurrent agent executions
            enable_adversarial: Whether to enable adversarial reasoning
            critique_rounds: Number of critique rounds in adversarial reasoning
            m27_config: M2.7 specific configuration
        """
        self.base_llm = llm_client
        self.base_tools = tools
        self.system_prompt = system_prompt
        self.max_concurrent = max_concurrent
        self.enable_adversarial = enable_adversarial
        self.critique_rounds = critique_rounds
        self.m27_config = m27_config or {}

        # Team members: name -> AgentMember
        self._agents: dict[str, AgentMember] = {}
        # Message bus for inter-agent communication
        self._bus = MessageBus()
        # Track task state
        self._current_task: str = ""
        self._task_id: int = 0
        # Metrics
        self._metrics: dict[str, Any] = defaultdict(int)

    def add_agent(
        self,
        name: str,
        role: AgentRole,
        system_prompt_addition: str = "",
        max_steps: int = 50,
        tool_names: list[str] | None = None,
    ) -> None:
        """Add an agent to the team.

        Args:
            name: Unique name for this agent
            role: Role this agent will play
            system_prompt_addition: Additional system prompt instructions
            max_steps: Maximum steps for this agent
            tool_names: Specific tools to grant (None = all tools)
        """
        if name in self._agents:
            raise ValueError(f"Agent with name '{name}' already exists")

        # Clone LLM client for this agent
        agent_llm = self.base_llm.clone() if hasattr(self.base_llm, "clone") else self.base_llm

        # Filter tools if specified
        tools = self.base_tools
        if tool_names:
            tools = [t for t in tools if t.name in tool_names]

        # Build system prompt with role instructions
        role_config = RoleConfig(
            role=role,
            system_prompt_addition=system_prompt_addition,
            max_steps=max_steps,
            m27_config=self.m27_config,
        )
        full_system_prompt = role_config.get_system_prompt(self.system_prompt)

        self._agents[name] = AgentMember(
            name=name,
            role=role,
            llm_client=agent_llm,
            tools=tools,
            system_prompt=full_system_prompt,
            max_steps=max_steps,
            m27_config=self.m27_config,
        )
        logger.info("Added agent '%s' with role '%s'", name, role.value)

    def add_critic(self, name: str = "critic", max_steps: int = 30) -> None:
        """Add a critic agent (convenience method).

        Args:
            name: Name for the critic agent
            max_steps: Maximum steps for critique
        """
        self.add_agent(
            name=name,
            role=AgentRole.CRITIC,
            max_steps=max_steps,
        )

    def add_reviewer(self, name: str = "reviewer", max_steps: int = 30) -> None:
        """Add a reviewer agent (convenience method).

        Args:
            name: Name for the reviewer agent
            max_steps: Maximum steps for review
        """
        self.add_agent(
            name=name,
            role=AgentRole.REVIEWER,
            max_steps=max_steps,
        )

    def add_planner(self, name: str = "planner", max_steps: int = 40) -> None:
        """Add a planner agent (convenience method).

        Args:
            name: Name for the planner agent
            max_steps: Maximum steps for planning
        """
        self.add_agent(
            name=name,
            role=AgentRole.PLANNER,
            max_steps=max_steps,
        )

    def add_executor(self, name: str = "executor", max_steps: int = 50) -> None:
        """Add an executor agent (convenience method).

        Args:
            name: Name for the executor agent
            max_steps: Maximum steps for execution
        """
        self.add_agent(
            name=name,
            role=AgentRole.EXECUTOR,
            max_steps=max_steps,
        )

    async def execute(
        self,
        task: str,
        initial_roles: list[AgentRole] | None = None,
        _wait_for_complete: bool = True,  # noqa: ARG002
        timeout: int | None = None,
    ) -> TeamResult:
        """Execute a task with the team.

        This method coordinates the team's effort to complete a task.
        The execution flow depends on the roles and configuration:

        1. If roles specified: Use those agents in sequence/parallel
        2. Default flow:
           a. Coordinator decomposes task
           b. Planner creates execution plan
           c. Executor implements
           d. Reviewer reviews
           e. If adversarial: Critics challenge
           f. Coordinator synthesizes final result

        Args:
            task: The task to execute
            initial_roles: Specific roles to use (None = auto-detect)
            wait_for_complete: Whether to wait for all agents (True) or run async (False)

        Returns:
            TeamResult with execution results
        """
        from time import perf_counter

        start = perf_counter()

        self._current_task = task
        self._task_id += 1

        logger.info("Team executing task: %s", task[:100])

        # Determine which roles to use
        if initial_roles:
            role_names = {r.value for r in initial_roles}
            active_agents = {name: agent for name, agent in self._agents.items() if agent.role.value in role_names}
        else:
            active_agents = self._agents.copy()

        if not active_agents:
            return TeamResult(
                success=False,
                content="",
                agent_results={},
                error="No agents available to execute task",
                elapsed=perf_counter() - start,
            )

        agent_results: dict[str, str] = {}
        iterations = 0

        async def run_with_timeout() -> tuple[str, str | None]:
            nonlocal iterations
            try:
                # Phase 1: Parallel reasoning by independent agents
                reasoning_tasks = [
                    name
                    for name, agent in active_agents.items()
                    if agent.role in (AgentRole.PLANNER, AgentRole.RESEARCHER, AgentRole.CRITIC, AgentRole.REVIEWER)
                ]

                if reasoning_tasks:
                    # Run reasoning agents in parallel
                    results = await self._run_agents_parallel(
                        {name: active_agents[name] for name in reasoning_tasks},
                        f"Task: {task}\n\nAnalyze and provide your reasoning on this task.",
                    )
                    agent_results.update(results)
                    iterations += 1

                # Phase 2: Executor implements based on reasoning
                executor_names = [name for name, agent in active_agents.items() if agent.role == AgentRole.EXECUTOR]
                if executor_names:
                    # Provide context from reasoning agents to executor
                    context = self._build_context_for_executor(agent_results)
                    executor_results = await self._run_agents_parallel(
                        {name: active_agents[name] for name in executor_names},
                        f"{context}\n\nTask: {task}\n\nExecute this task.",
                    )
                    agent_results.update(executor_results)
                    iterations += 1

                # Phase 3: Adversarial reasoning (if enabled)
                if self.enable_adversarial:
                    for round_num in range(self.critique_rounds):
                        critique_results = await self._run_critique_round(active_agents, task, agent_results, round_num)
                        agent_results.update(critique_results)
                        iterations += 1

                # Phase 4: Synthesis
                synthesis = await self._synthesize_results(task, agent_results)
                return synthesis, None

            except asyncio.TimeoutError:
                return "", "Execution timed out"
            except Exception as e:
                logger.error("Team execution failed: %s", e)
                return "", str(e)

        if timeout:
            try:
                synthesis, error = await asyncio.wait_for(run_with_timeout(), timeout=timeout)
            except asyncio.TimeoutError:
                synthesis, error = "", "Execution timed out"
        else:
            synthesis, error = await run_with_timeout()

        elapsed = perf_counter() - start

        if error:
            return TeamResult(
                success=False,
                content=synthesis if synthesis else "",
                agent_results=agent_results,
                error=error,
                elapsed=elapsed,
            )

        return TeamResult(
            success=True,
            content=synthesis,
            agent_results=agent_results,
            consensus=True,
            iterations=iterations,
            elapsed=elapsed,
        )

    async def _run_agents_parallel(
        self,
        agents: dict[str, AgentMember],
        task_prompt: str,
    ) -> dict[str, str]:
        """Run multiple agents in parallel.

        Args:
            agents: Dict of name -> AgentMember
            task_prompt: Task prompt for all agents

        Returns:
            Dict of agent name -> result content
        """
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def run_one(name: str, agent: AgentMember) -> tuple[str, str]:
            async with semaphore:
                try:
                    result = await self._run_single_agent(agent, task_prompt)
                    return (name, result)
                except Exception as e:
                    logger.error("Agent '%s' failed: %s", name, e)
                    return (name, f"Error: {str(e)}")

        tasks = [run_one(name, agent) for name, agent in agents.items()]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        output = {}
        for result in results:
            if isinstance(result, tuple):
                name, content = result
                output[name] = content
            elif isinstance(result, Exception):
                logger.error("Task failed with exception: %s", result)

        return output

    async def _run_single_agent(
        self,
        agent: AgentMember,
        task_prompt: str,
    ) -> str:
        """Run a single agent to completion.

        Args:
            agent: The agent to run
            task_prompt: Task prompt

        Returns:
            Agent's final response
        """
        messages = [
            Message(role="system", content=agent.system_prompt),
            Message(role="user", content=task_prompt),
        ]

        for _step in range(agent.max_steps):
            response = await agent.llm_client.generate(
                messages=messages,
                tools=agent.tools,
            )

            if response.tool_calls:
                # Execute tools
                results = await self._execute_tools_for_agent(agent, response.tool_calls)

                # Add assistant message
                messages.append(
                    Message(
                        role="assistant",
                        content=response.content or "",
                        tool_calls=response.tool_calls,
                    )
                )

                # Add tool results
                for _, tool_msg in results:
                    messages.append(tool_msg)
                continue

            # No tool calls - return the content
            return response.content or ""

        return "Max steps reached without completion"

    async def _execute_tools_for_agent(
        self,
        agent: AgentMember,
        tool_calls: list[Any],
    ) -> list[tuple[Any, ...]]:
        """Execute tool calls for an agent.

        Args:
            agent: The agent executing tools
            tool_calls: List of tool calls

        Returns:
            List of (tool_call, tool_message) tuples
        """
        results = []
        for tc in tool_calls:
            tool = agent.tools_dict.get(tc.function.name)
            if not tool:
                results.append(
                    (
                        tc,
                        Message(
                            role="tool",
                            content=f"Error: Unknown tool: {tc.function.name}",
                            tool_call_id=tc.id,
                            name=tc.function.name,
                        ),
                    )
                )
                continue

            try:
                result = await tool.execute(**tc.function.arguments)
            except Exception as e:
                from ..tools.base import ToolResult

                result = ToolResult(success=False, content="", error=str(e))

            results.append(
                (
                    tc,
                    Message(
                        role="tool",
                        content=result.content if result.success else f"Error: {result.error}",
                        tool_call_id=tc.id,
                        name=tc.function.name,
                    ),
                )
            )

        return results

    def _build_context_for_executor(
        self,
        agent_results: dict[str, str],
    ) -> str:
        """Build context string for executor from reasoning results."""
        context_parts = ["## Previous Analysis\n"]

        for name, content in agent_results.items():
            context_parts.append(f"\n### {name.title()} Analysis:\n{content[:500]}")

        return "\n".join(context_parts)

    async def _run_critique_round(
        self,
        active_agents: dict[str, AgentMember],
        task: str,
        current_results: dict[str, str],
        round_num: int,
    ) -> dict[str, str]:
        """Run a round of adversarial critique.

        Args:
            active_agents: All active agents
            task: The original task
            current_results: Current results from other agents
            round_num: Current critique round number

        Returns:
            Dict of critique results
        """
        critique_agents = {name: agent for name, agent in active_agents.items() if agent.role == AgentRole.CRITIC}

        if not critique_agents:
            return {}

        # Build context with current results
        context = self._build_context_for_executor(current_results)
        critique_prompt = f"""{context}

## Task to Critique
{task}

## Critique Round {round_num + 1}
Analyze the current approach and identify:
1. Potential failure modes or risks
2. Assumptions that might be wrong
3. Edge cases not addressed
4. Alternative approaches worth considering

Be specific and constructive. Challenge overconfidence.
"""

        results = await self._run_agents_parallel(critique_agents, critique_prompt)
        return {f"{name}_critique_{round_num}": content for name, content in results.items()}

    async def _synthesize_results(
        self,
        task: str,
        agent_results: dict[str, str],
    ) -> str:
        """Synthesize results from all agents into final output.

        Args:
            task: The original task
            agent_results: Results from all agents

        Returns:
            Synthesized final content
        """
        # Find coordinator or use first agent
        coordinator = next((agent for agent in self._agents.values() if agent.role == AgentRole.COORDINATOR), None)

        if not coordinator:
            # Simple synthesis: combine all results
            synthesis_parts = [f"# Team Results for: {task}\n\n"]
            for name, content in agent_results.items():
                synthesis_parts.append(f"## {name.title()}\n{content}\n\n")
            return "\n".join(synthesis_parts)

        # Use coordinator to synthesize
        synthesis_prompt = f"""## Original Task
{task}

## Team Member Results
{self._build_context_for_executor(agent_results)}

## Synthesis Task
Based on the analysis and work from your team members, provide a comprehensive final answer.
Synthesize different perspectives into a coherent solution.
Address any conflicts or disagreements raised by critics.
"""

        return await self._run_single_agent(coordinator, synthesis_prompt)

    @property
    def agents(self) -> dict[str, AgentMember]:
        """Get all agents in the team."""
        return self._agents.copy()

    @property
    def message_bus(self) -> MessageBus:
        """Get the team's message bus."""
        return self._bus

    def get_metrics(self) -> dict[str, Any]:
        """Get team execution metrics."""
        return dict(self._metrics)

    def __len__(self) -> int:
        """Get number of agents in team."""
        return len(self._agents)

    def __repr__(self) -> str:
        """String representation of team."""
        agent_summary = [f"{name}({agent.role.value})" for name, agent in self._agents.items()]
        return f"AgentTeam({agent_summary})"
