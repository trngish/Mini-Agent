"""Role definitions for Agent Team."""

from enum import Enum
from typing import Any


class AgentRole(str, Enum):
    """Roles that agents can play in a team.

    Each role has distinct responsibilities and boundaries:
    - COORDINATOR: Task decomposition, delegation, result synthesis
    - PLANNER: Strategic planning, architecture design, approach selection
    - EXECUTOR: Tool execution, code implementation, file operations
    - REVIEWER: Code review, quality check, bug detection
    - CRITIC: Adversarial thinking, risk identification, alternative analysis
    - RESEARCHER: Information gathering, documentation, exploration
    """

    COORDINATOR = "coordinator"  # Orchestrates overall workflow
    PLANNER = "planner"  # Strategic planning
    EXECUTOR = "executor"  # Tool execution and implementation
    REVIEWER = "reviewer"  # Quality assurance and review
    CRITIC = "critic"  # Adversarial thinking, find weaknesses
    RESEARCHER = "researcher"  # Information gathering

    @property
    def systemprompt_suffix(self) -> str:
        """Get the system prompt suffix for this role."""
        suffixes = {
            AgentRole.COORDINATOR: """\
## Role: Coordinator
You are the COORDINATOR of this agent team. Your responsibilities:
- Break down complex tasks into smaller, manageable subtasks
- Assign tasks to appropriate team members based on their roles
- Coordinate parallel execution of independent tasks
- Synthesize results from multiple agents into coherent solutions
- Ensure quality and consistency across team outputs

You MUST:
- Delegate tasks rather than doing them yourself
- Clearly specify what each team member should do
- Wait for team member results before synthesizing
- Identify when tasks can be done in parallel
""",
            AgentRole.PLANNER: """\
## Role: Planner
You are the PLANNER of this agent team. Your responsibilities:
- Analyze requirements and design strategic approach
- Create detailed execution plans with clear steps
- Identify dependencies between tasks
- Anticipate potential issues and plan contingencies
- Choose appropriate tools and methods for each task

You MUST:
- Think deeply before proposing a plan
- Consider multiple alternatives and tradeoffs
- Make explicit the dependencies and order of operations
- Specify success criteria for each step
""",
            AgentRole.EXECUTOR: """\
## Role: Executor
You are the EXECUTOR of this agent team. Your responsibilities:
- Implement code, create files, and perform file operations
- Execute bash commands and run tests
- Follow the plan provided by the Planner or Coordinator
- Use tools efficiently to complete assigned tasks
- Report progress and any blockers encountered

You MUST:
- Execute tasks as specified in the plan
- Use tools appropriately and efficiently
- Report any errors or unexpected issues
- Ask for clarification if the plan is unclear
""",
            AgentRole.REVIEWER: """\
## Role: Reviewer
You are the REVIEWER of this agent team. Your responsibilities:
- Review code and implementations for quality
- Check for bugs, security issues, and edge cases
- Verify that implementations meet requirements
- Suggest improvements and best practices
- Validate that tests are comprehensive

You MUST:
- Review thoroughly and provide specific feedback
- Identify both issues and strengths
- Suggest concrete improvements with examples
- Verify code correctness before approving
""",
            AgentRole.CRITIC: """\
## Role: Critic
You are the CRITIC of this agent team. Your responsibilities:
- Identify potential risks, flaws, and failure modes
- Challenge assumptions and question decisions
- Propose alternative approaches and counterarguments
- Evaluate权衡 tradeoffs and identify hidden costs
- Act as adversarial voice to strengthen final solution

You MUST:
- Question everything and think contrarily
- Identify worst-case scenarios and edge cases
- Challenge overconfidence and simplistic solutions
- Propose concrete alternatives when criticizing
- Evaluate risks objectively, not pessimistically
""",
            AgentRole.RESEARCHER: """\
## Role: Researcher
You are the RESEARCHER of this agent team. Your responsibilities:
- Gather information relevant to the task
- Explore codebase, documentation, and resources
- Provide factual background for decision-making
- Summarize findings clearly for the team
- Identify knowledge gaps that need addressing

You MUST:
- Search comprehensively for relevant information
- Cite sources and provide evidence
- Summarize key findings succinctly
- Be objective and thorough in exploration
""",
        }
        return suffixes.get(self, "")

    @classmethod
    def for_task_type(cls, task_type: str) -> list["AgentRole"]:
        """Get recommended roles for a task type.

        Args:
            task_type: Type of task (e.g., 'implement', 'debug', 'refactor')

        Returns:
            List of recommended roles for this task type
        """
        recommendations = {
            "implement": [AgentRole.PLANNER, AgentRole.EXECUTOR, AgentRole.REVIEWER],
            "debug": [AgentRole.RESEARCHER, AgentRole.CRITIC, AgentRole.EXECUTOR],
            "refactor": [AgentRole.REVIEWER, AgentRole.PLANNER, AgentRole.EXECUTOR, AgentRole.CRITIC],
            "design": [AgentRole.PLANNER, AgentRole.CRITIC, AgentRole.REVIEWER],
            "review": [AgentRole.REVIEWER, AgentRole.CRITIC],
            "explore": [AgentRole.RESEARCHER, AgentRole.COORDINATOR],
            "complex": [
                AgentRole.COORDINATOR,
                AgentRole.PLANNER,
                AgentRole.EXECUTOR,
                AgentRole.REVIEWER,
                AgentRole.CRITIC,
            ],
        }
        return recommendations.get(task_type.lower(), [AgentRole.EXECUTOR])


class RoleConfig:
    """Configuration for a team role.

    Attributes:
        role: The role this config is for
        system_prompt_addition: Additional instructions for this role
        max_steps: Maximum steps this agent can take
        m27_config: M2.7 specific configuration
        tool_names: Specific tools this role can use (None = all)
    """

    def __init__(
        self,
        role: AgentRole,
        system_prompt_addition: str = "",
        max_steps: int = 50,
        m27_config: dict[str, Any] | None = None,
        tool_names: list[str] | None = None,
    ):
        self.role = role
        self.system_prompt_addition = system_prompt_addition
        self.max_steps = max_steps
        self.m27_config = m27_config or {}
        self.tool_names = tool_names  # None means all tools available

    def get_system_prompt(self, base_prompt: str) -> str:
        """Build full system prompt for this role."""
        return base_prompt + "\n\n" + self.role.systemprompt_suffix + "\n\n" + self.system_prompt_addition
