"""Example: Agent Team with Multi-Agent Collaboration

This example demonstrates:
- Creating a team with multiple agents
- Role-boundary isolation (planner, executor, reviewer, critic)
- Parallel independent reasoning
- Adversarial reasoning with multiple critics
- Result synthesis

Run with:
    python examples/05_agent_team.py
"""

import asyncio
import tempfile
from pathlib import Path

from mini_agent import LLMClient, AgentTeam, AgentRole
from mini_agent.config import Config
from mini_agent.tools import ReadTool, WriteTool, EditTool, BashTool


async def demo_agent_team():
    """Demo: Multi-agent team collaboration."""
    print("\n" + "=" * 60)
    print("Agent Team - Multi-Agent Collaboration Demo")
    print("=" * 60)

    # Load configuration
    config_path = Path("mini_agent/config/config.yaml")
    if not config_path.exists():
        print("❌ config.yaml not found. Please run:")
        print("   cp mini_agent/config/config-example.yaml mini_agent/config/config.yaml")
        return

    config = Config.from_yaml(config_path)

    # Check API key
    if not config.llm.api_key or config.llm.api_key.startswith("YOUR_"):
        print("❌ API key not configured in config.yaml")
        return

    # Create workspace
    with tempfile.TemporaryDirectory() as workspace_dir:
        print(f"📁 Workspace: {workspace_dir}")

        # Initialize LLM
        llm_client = LLMClient(
            api_key=config.llm.api_key,
            api_base=config.llm.api_base,
            model=config.llm.model,
        )

        # Initialize tools
        tools = [
            ReadTool(workspace_dir=workspace_dir),
            WriteTool(workspace_dir=workspace_dir),
            EditTool(workspace_dir=workspace_dir),
            BashTool(),
        ]
        print("✓ Loaded 4 basic tools")

        # Create team with diverse roles
        team = AgentTeam(
            llm_client=llm_client,
            tools=tools,
            system_prompt="""You are part of a software engineering team.
Your role is defined by your responsibilities below.
Work collaboratively with other team members to deliver high-quality solutions.""",
            max_concurrent=3,
            enable_adversarial=True,
            critique_rounds=2,
            m27_config={"thinking_budget_tokens": 16384},
        )

        # Add team members with different roles
        team.add_planner("planner")
        print("✓ Added planner agent")

        team.add_executor("executor")
        print("✓ Added executor agent")

        team.add_reviewer("reviewer")
        print("✓ Added reviewer agent")

        team.add_critic("critic1")
        team.add_critic("critic2")  # Multiple critics for stronger adversarial reasoning
        print("✓ Added 2 critic agents (adversarial reasoning)")

        print(f"\n🤖 Team created with {len(team)} agents")
        print(f"   Roles: {[f'{name}({agent.role.value})' for name, agent in team.agents.items()]}")
        print("\n" + "-" * 60)

        # Task that benefits from team collaboration
        task = """
Create a Python module called 'string_utils.py' with the following functionality:

1. A function `reverse_string(s: str) -> str` - reverses a string
2. A function `is_palindrome(s: str) -> bool` - checks if string is palindrome
3. A function `count_vowels(s: str) -> int` - counts vowel letters
4. A function `anagram_groups(strings: list[str]) -> list[list[str]]` - groups anagrams

Each function should have proper docstrings and type hints.
Write comprehensive tests using pytest.
"""

        print("📋 Task:")
        print(task)
        print("\n" + "=" * 60)
        print("🤖 Team is working...\n")

        # Execute with team
        result = await team.execute(task)

        print("\n" + "=" * 60)
        if result.success:
            print("✅ Team completed successfully!")
            print("=" * 60)
            print(f"\n📊 Metrics:")
            print(f"   - Iterations: {result.iterations}")
            print(f"   - Elapsed time: {result.elapsed:.2f}s")
            print(f"   - Consensus reached: {result.consensus}")
            print(f"\n🤖 Agent Results:")
            for name, content in result.agent_results.items():
                preview = content[:200] + "..." if len(content) > 200 else content
                print(f"\n   [{name}]:\n   {preview}")

            print("\n" + "=" * 60)
            print("📄 Final Synthesized Output:")
            print("=" * 60)
            print(result.content)
        else:
            print(f"❌ Team execution failed: {result.error}")

        # Show created files
        print("\n" + "=" * 60)
        print("📁 Created files in workspace:")
        print("=" * 60)

        workspace = Path(workspace_dir)
        for file in workspace.glob("*.py"):
            print(f"\n📄 {file.name}:")
            print("-" * 60)
            content = file.read_text()
            lines = content.split("\n")
            if len(lines) > 30:
                print("\n".join(lines[:30]))
                print(f"... ({len(lines) - 30} more lines)")
            else:
                print(content)
            print("-" * 60)


async def demo_simple_team():
    """Demo: Simple team with just executor and critic."""
    print("\n" + "=" * 60)
    print("Simple Agent Team Demo")
    print("=" * 60)

    config_path = Path("mini_agent/config/config.yaml")
    if not config_path.exists():
        print("❌ config.yaml not found")
        return

    config = Config.from_yaml(config_path)

    if not config.llm.api_key or config.llm.api_key.startswith("YOUR_"):
        print("❌ API key not configured")
        return

    with tempfile.TemporaryDirectory() as workspace_dir:
        llm_client = LLMClient(
            api_key=config.llm.api_key,
            api_base=config.llm.api_base,
            model=config.llm.model,
        )

        tools = [
            WriteTool(workspace_dir=workspace_dir),
            ReadTool(workspace_dir=workspace_dir),
            BashTool(),
        ]

        # Simple team: executor + critic
        team = AgentTeam(
            llm_client=llm_client,
            tools=tools,
            enable_adversarial=True,
            critique_rounds=1,
        )

        team.add_executor("builder")
        team.add_critic("critic")

        task = "Create a file 'hello.txt' with 'Hello, World!' inside it."

        print(f"📋 Task: {task}\n")

        result = await team.execute(task)

        if result.success:
            print("✅ Success!")
            print(f"📄 Result:\n{result.content}")
        else:
            print(f"❌ Failed: {result.error}")


async def main():
    """Run all demos."""
    print("=" * 60)
    print("Agent Team Examples")
    print("=" * 60)

    await demo_simple_team()
    print("\n" * 2)
    await demo_agent_team()

    print("\n" + "=" * 60)
    print("All demos completed! ✅")
    print("=" * 60)
    print("\n💡 Agent Team Features:")
    print("   - Role-boundary isolation (planner, executor, reviewer, critic)")
    print("   - Parallel independent reasoning")
    print("   - Adversarial reasoning (multiple critics)")
    print("   - Result synthesis")
    print("\n💡 Next step: Customize your own team with specific roles!")


if __name__ == "__main__":
    asyncio.run(main())