from unittest.mock import MagicMock

from mini_agent.utils.m27_optimization import M27AgentTeams, M27ContextManager, M27PromptOptimizer, M27ToolOptimizer


class TestM27PromptOptimizer:
    def test_optimize_system_prompt(self):
        result = M27PromptOptimizer.optimize_system_prompt("Be helpful", "/workspace")
        assert "M2.7" in result
        assert "/workspace" in result

    def test_optimize_system_prompt_already_m27(self):
        prompt = "You are M2.7 powered agent"
        result = M27PromptOptimizer.optimize_system_prompt(prompt, "/workspace")
        assert result == prompt

    def test_optimize_system_prompt_already_extended_thinking(self):
        prompt = "Use extended thinking for complex tasks"
        result = M27PromptOptimizer.optimize_system_prompt(prompt, "/workspace")
        assert result == prompt

    def test_optimize_empty_prompt(self):
        result = M27PromptOptimizer.optimize_system_prompt("", "/workspace")
        assert "M2.7" in result

    def test_get_thinking_block(self):
        block = M27PromptOptimizer.get_thinking_block("my reasoning")
        assert block["type"] == "thinking"
        assert block["thinking"] == "my reasoning"

    def test_extract_thinking_from_response_with_thinking_attr(self):
        resp = MagicMock()
        resp.thinking = "deep thoughts"
        result = M27PromptOptimizer.extract_thinking_from_response(resp)
        assert result == "deep thoughts"

    def test_extract_thinking_from_response_no_thinking(self):
        resp = MagicMock(spec=[])
        result = M27PromptOptimizer.extract_thinking_from_response(resp)
        assert result is None

    def test_extract_thinking_from_content_blocks_dict(self):
        resp = MagicMock(spec=[])
        resp.content = [{"type": "thinking", "thinking": "block thought"}]
        result = M27PromptOptimizer.extract_thinking_from_response(resp)
        assert result == "block thought"

    def test_extract_thinking_from_content_blocks_object(self):
        block = MagicMock()
        block.type = "thinking"
        block.thinking = "object thought"
        resp = MagicMock(spec=[])
        resp.content = [block]
        result = M27PromptOptimizer.extract_thinking_from_response(resp)
        assert result == "object thought"


class TestM27ContextManager:
    def test_init(self):
        cm = M27ContextManager(token_limit=800_000)
        assert cm.token_limit == 800_000

    def test_should_summarize_below(self):
        cm = M27ContextManager(token_limit=800_000)
        assert cm.should_summarize(100_000) is False

    def test_should_summarize_above(self):
        cm = M27ContextManager(token_limit=800_000)
        assert cm.should_summarize(900_000) is True

    def test_should_summarize_api_tokens_above(self):
        cm = M27ContextManager(token_limit=800_000)
        cm.api_total_tokens = 900_000
        assert cm.should_summarize(100_000) is True

    def test_get_optimized_summarization_prompt(self):
        cm = M27ContextManager()
        prompt = cm.get_optimized_summarization_prompt([])
        assert "summar" in prompt.lower()

    def test_calculate_optimal_batch_size_small(self):
        cm = M27ContextManager()
        assert cm.calculate_optimal_batch_size(50) == 50

    def test_calculate_optimal_batch_size_medium(self):
        cm = M27ContextManager()
        assert cm.calculate_optimal_batch_size(200) == 100

    def test_calculate_optimal_batch_size_large(self):
        cm = M27ContextManager()
        assert cm.calculate_optimal_batch_size(1000) == 200

    def test_calculate_optimal_batch_size_very_large(self):
        cm = M27ContextManager()
        assert cm.calculate_optimal_batch_size(5000) == 500


class TestM27ToolOptimizer:
    def test_can_parallelize_read_tools(self):
        assert M27ToolOptimizer.can_parallelize(["bash", "read"]) is True

    def test_cannot_parallelize_with_write(self):
        assert M27ToolOptimizer.can_parallelize(["bash", "write"]) is False

    def test_cannot_parallelize_with_edit(self):
        assert M27ToolOptimizer.can_parallelize(["read", "edit"]) is False

    def test_cannot_parallelize_with_git(self):
        assert M27ToolOptimizer.can_parallelize(["bash", "git"]) is False

    def test_can_parallelize_grep_find(self):
        assert M27ToolOptimizer.can_parallelize(["grep", "find"]) is True

    def test_optimize_tool_schema_adds_description(self):
        schema = {"name": "my_tool"}
        result = M27ToolOptimizer.optimize_tool_schema(schema)
        assert "description" in result

    def test_optimize_tool_schema_preserves_description(self):
        schema = {"name": "my_tool", "description": "existing"}
        result = M27ToolOptimizer.optimize_tool_schema(schema)
        assert result["description"] == "existing"


class TestM27AgentTeams:
    def test_create_team_prompt(self):
        prompt = M27AgentTeams.create_team_prompt(["planner", "executor"], "Build app")
        assert "planner" in prompt
        assert "executor" in prompt
        assert "Build app" in prompt

    def test_supported_roles(self):
        assert "planner" in M27AgentTeams.SUPPORTED_ROLES
        assert "executor" in M27AgentTeams.SUPPORTED_ROLES
        assert "reviewer" in M27AgentTeams.SUPPORTED_ROLES
        assert "coordinator" in M27AgentTeams.SUPPORTED_ROLES

    def test_supported_protocols(self):
        assert "sequential" in M27AgentTeams.SUPPORTED_PROTOCOLS
        assert "parallel" in M27AgentTeams.SUPPORTED_PROTOCOLS
        assert "hierarchical" in M27AgentTeams.SUPPORTED_PROTOCOLS
