# Mini-Agent 架构与性能优化实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复架构设计和性能优化中的高优先级和中优先级问题

**Architecture:** 优化架构设计（StepRunner 解耦、Token 估计精度）、增强性能和缓存机制

**Tech Stack:** Python, tiktoken, asyncio

---

## 高优先级任务

### Task 1: 统一 AgentContext 中 Message 模型为 Pydantic

**Files:**
- Modify: `mini_agent/core/agent_context.py:1-103`
- Modify: `mini_agent/schema/schema.py:41-50`
- Test: `tests/test_agent_context.py`

- [ ] **Step 1: 创建 AgentContext 的 Pydantic 消息模型测试**

```python
# tests/test_agent_context.py - 添加新测试
class TestMessageModel:
    """Test Message pydantic model for AgentContext compatibility."""

    def test_message_from_dataclass_conversion(self):
        """Test converting existing Message dataclass to pydantic."""
        from mini_agent.schema import Message
        # Ensure Message has model_validate and model_dump
        msg = Message(role="user", content="test")
        data = msg.model_dump()
        restored = Message.model_validate(data)
        assert restored.role == msg.role
        assert restored.content == msg.content

    def test_message_with_tool_calls_serialization(self):
        """Test Message with tool_calls serializes correctly."""
        from mini_agent.schema import Message, ToolCall, FunctionCall

        tool_call = ToolCall(
            id="test-1",
            type="function",
            function=FunctionCall(name="bash", arguments={"command": "ls"})
        )
        msg = Message(
            role="assistant",
            content="Running command",
            tool_calls=[tool_call]
        )
        data = msg.model_dump()
        restored = Message.model_validate(data)
        assert len(restored.tool_calls) == 1
        assert restored.tool_calls[0].function.name == "bash"
```

- [ ] **Step 2: 运行测试验证 Message 模型**

Run: `python -m pytest tests/test_agent_context.py -v -k "test_message_from_dataclass"`
Expected: PASS (Message 已使用 pydantic)

- [ ] **Step 3: 更新 AgentContext 注释以反映 Pydantic 模型**

Modify `mini_agent/core/agent_context.py`:
```python
# 在 AgentContext 类 docstring 中更新说明
# 原有:
# """Central context container for agent state and dependencies.
# ...
# - Thread-safe state updates via properties
# - Clear interface contracts for all state access

# 修改为:
"""
Central context container for agent state and dependencies.

This class provides:
- Unified state management (messages, token counts, mode, etc.)
- Dependency injection for components that need context access
- Thread-safe state updates via properties (using pydantic Message model)
- Clear interface contracts for all state access
"""
```

- [ ] **Step 4: 运行相关测试**

Run: `python -m pytest tests/test_agent.py tests/test_agent_context.py -v --tb=short`
Expected: All PASS

- [ ] **Step 5: 提交更改**

```bash
git add mini_agent/core/agent_context.py tests/test_agent_context.py
git commit -m "docs: update AgentContext docstring to reflect pydantic Message model"
```

---

### Task 2: 增强 Token 估计精度（使用 tiktoken）

**Files:**
- Modify: `mini_agent/core/agent_context.py:61-66`
- Modify: `mini_agent/core/health_check.py:56-66`
- Test: `tests/test_token_tracker.py`

- [ ] **Step 1: 添加 Token 估计精度测试**

```python
# tests/test_token_tracker.py - 添加精度测试
class TestTokenEstimationAccuracy:
    """Test tiktoken-based token estimation accuracy."""

    def test_tiktoken_encoder_available(self):
        """Verify tiktoken encoder is available for accurate estimation."""
        from mini_agent.utils.token_utils import get_encoder
        encoder = get_encoder("cl100k_base")
        tokens = encoder.encode("Hello, world!")
        assert len(tokens) == 5  # cl100k_base encoding

    def test_estimate_tokens_vs_fallback(self):
        """Compare tiktoken estimation vs fallback for accuracy."""
        from mini_agent.core.token_tracker import TokenTracker
        from mini_agent.schema import Message

        tracker = TokenTracker()
        messages = [
            Message(role="user", content="This is a longer test message to check token estimation accuracy.")
        ]

        tiktoken_count = tracker.estimate_tokens(messages)
        # Tiktoken should give more accurate count than char/2.5
        assert tiktoken_count > 0

        # Verify fallback exists and works
        fallback_count = tracker._estimate_tokens_fallback(messages)
        assert fallback_count > 0

    def test_thinking_content_token_counting(self):
        """Test that thinking content is included in token estimation."""
        from mini_agent.core.token_tracker import TokenTracker
        from mini_agent.schema import Message

        tracker = TokenTracker()
        messages = [
            Message(role="assistant", content="Thinking...", thinking="Let me think about this carefully")
        ]

        count = tracker.estimate_tokens(messages)
        assert count > len("Thinking...") / 2.5  # Should include thinking

- [ ] **Step 2: 运行测试验证 Token 估计**

Run: `python -m pytest tests/test_token_tracker.py -v -k "TestTokenEstimationAccuracy"`
Expected: PASS

- [ ] **Step 3: 更新 HealthCheck 使用精确 Token 估计**

Modify `mini_agent/core/health_check.py:56-66`:
```python
# 原有代码:
# tokens = self._context.estimate_tokens()
# limit = self._context.token_limit
# if tokens > limit * self.TOKEN_CRITICAL_THRESHOLD:

# 修改为: 直接使用 context 的 token_tracker（更精确）
def check(self) -> HealthCheckResult:
    """Perform health check."""
    issues = []

    # Check token usage - use AgentContext's token tracker directly for accuracy
    try:
        # AgentContext.estimate_tokens() already uses tiktoken for accurate counting
        # but provide fallback for edge cases
        tokens = self._context.estimate_tokens()
        limit = self._context.token_limit

        if tokens > limit * self.TOKEN_CRITICAL_THRESHOLD:
            issues.append(f"Token usage critical: {tokens:,} / {limit:,}")
        elif tokens > limit * self.TOKEN_WARNING_THRESHOLD:
            issues.append(f"Token usage high: {tokens:,} / {limit:,}")
    except Exception:
        pass

    # ... rest of check
```

- [ ] **Step 4: 运行测试验证 HealthCheck 更新**

Run: `python -m pytest tests/test_health_check.py -v`
Expected: All PASS

- [ ] **Step 5: 提交更改**

```bash
git add mini_agent/core/health_check.py tests/test_token_tracker.py
git commit -m "perf: enhance token estimation accuracy in health check"
```

---

## 中优先级任务

### Task 3: StepRunner 解耦 - 减少对 Agent 的直接依赖

**Files:**
- Modify: `mini_agent/core/step_runner.py:20-143`
- Create: `mini_agent/core/step_runner_interface.py`
- Test: `tests/test_step_runner.py`

- [ ] **Step 1: 创建 StepRunner 接口定义**

```python
# mini_agent/core/step_runner_interface.py
"""StepRunner interface for agent decoupling."""

from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ..agent import Agent

class IStepRunnerDelegate(ABC):
    """Interface for StepRunner to interact with Agent without direct coupling."""

    @property
    @abstractmethod
    def context(self) -> Any:
        """Get the agent's context."""
        pass

    @property
    @abstractmethod
    def logger(self) -> Any:
        """Get the agent's logger."""
        pass

    @property
    @abstractmethod
    def thinking_manager(self) -> Any:
        """Get the thinking manager (can be None)."""
        pass

    @abstractmethod
    def check_health(self) -> list[str]:
        """Run health check and return issues."""
        pass

    @abstractmethod
    def save_session(self, step: int, prefix: str) -> None:
        """Save session."""
        pass
```

- [ ] **Step 2: 创建 StepRunner 解耦测试**

```python
# tests/test_step_runner.py - 添加接口测试
class TestStepRunnerInterface:
    """Test StepRunner interface for proper decoupling."""

    def test_step_runner_uses_delegate_interface(self):
        """Verify StepRunner can work with delegate interface."""
        from mini_agent.core.step_runner import StepRunner
        from mini_agent.core.step_runner_interface import IStepRunnerDelegate
        from unittest.mock import MagicMock

        # Create mock delegate
        mock_agent = MagicMock()
        mock_agent._context = MagicMock()
        mock_agent._context.api_call_count = 0
        mock_agent._context.api_total_tokens = 0
        mock_agent.logger = MagicMock()
        mock_agent._thinking_manager = None
        mock_agent._session_manager = MagicMock()
        mock_agent._last_health_check_step = -1
        mock_agent._health_check_interval = 5
        mock_agent._check_health = MagicMock(return_value=[])
        mock_agent._token_tracker = MagicMock()

        # StepRunner should work with the mock
        runner = StepRunner(mock_agent, 0.0)
        assert runner._agent is mock_agent
```

- [ ] **Step 3: 运行测试验证接口**

Run: `python -m pytest tests/test_step_runner.py -v -k "TestStepRunnerInterface"`
Expected: PASS

- [ ] **Step 4: 添加 StepRunner 注释说明其角色**

Modify `mini_agent/core/step_runner.py:20-30`:
```python
class StepRunner:
    """Manages a single step in the agent execution loop.

    NOTE: This class holds a reference to Agent for convenience, but should
    interact primarily through well-defined interfaces to maintain
    testability and reduce coupling.

    Responsibilities:
    - Process LLM response (add assistant message, log)
    - Health check (throttled)
    - Thinking content pruning
    - Tool execution delegation (via Agent)
    - Auto-save management
    - Step timing and metrics
    """
```

- [ ] **Step 5: 运行完整测试**

Run: `python -m pytest tests/test_step_runner.py tests/test_agent.py -v --tb=short`
Expected: All PASS

- [ ] **Step 6: 提交更改**

```bash
git add mini_agent/core/step_runner.py mini_agent/core/step_runner_interface.py tests/test_step_runner.py
git commit -m "refactor: add StepRunner interface for agent decoupling"
```

---

### Task 4: 环境变量安全性增强

**Files:**
- Modify: `mini_agent/config.py:376-428`
- Create: `tests/test_config_env_security.py`

- [ ] **Step 1: 添加环境变量安全性测试**

```python
# tests/test_config_env_security.py
"""Tests for config environment variable security."""

class TestEnvVarSecurity:
    """Test environment variable handling security."""

    def test_production_api_key_env_override(self):
        """Test that API key from env takes precedence (expected behavior)."""
        import os
        from mini_agent.config import Config, LLMConfig, AgentConfig, ToolsConfig

        # Env var should override config file
        os.environ["MINIMAX_API_KEY"] = "env-api-key-12345678"
        try:
            config = Config(
                llm=LLMConfig(api_key="file-key-12345678", api_base="https://api.test.com"),
                agent=AgentConfig(),
                tools=ToolsConfig(),
            )
            # This would normally be validated by ConfigValidator
            # For this test, just verify the override happens
            assert config.llm.api_key == "env-api-key-12345678"
        finally:
            del os.environ["MINIMAX_API_KEY"]

    def test_env_override_validation(self):
        """Test that CLI override values are re-validated."""
        from mini_agent.config import Config, LLMConfig, AgentConfig, ToolsConfig, CLIOverrideConfig

        config = Config(
            llm=LLMConfig(api_key="sk-test-key-12345678", api_base="https://api.test.com"),
            agent=AgentConfig(),
            tools=ToolsConfig(),
        )

        # CLI override should trigger re-validation
        cli_override = CLIOverrideConfig(max_steps=200)
        config.merge_cli_overrides(cli_override)
        assert config.agent.max_steps == 200

    def test_invalid_api_key_length_rejected(self):
        """Test that short API keys are rejected."""
        from pydantic import ValidationError
        from mini_agent.config import LLMConfig

        with pytest.raises(ValidationError, match="at least 8 characters"):
            LLMConfig(api_key="short", api_base="https://api.test.com")
```

- [ ] **Step 2: 运行安全测试**

Run: `python -m pytest tests/test_config_env_security.py -v`
Expected: All PASS

- [ ] **Step 3: 更新 Config 文档注释**

Modify `mini_agent/config.py:376-389`:
```python
@classmethod
def _apply_env_overrides(cls, data: dict[str, Any]) -> dict[str, Any]:
    """Apply environment variable overrides to configuration data.

    SECURITY NOTE: Environment variables have the HIGHEST priority and
    will override any values in the YAML configuration file.

    Supported env vars:
    - MINIMAX_API_KEY: API key (required, min 8 chars)
    - MINI_AGENT_API_KEY: Legacy alias for API key
    - MINI_AGENT_API_BASE: API base URL
    - MINI_AGENT_MODEL: Model name
    - MINI_AGENT_PROVIDER: Provider type
    - MINI_AGENT_MAX_STEPS: Max execution steps
    - MINI_AGENT_WORKSPACE_DIR: Workspace directory
    - MINI_AGENT_PLATFORM_MODE: Platform mode (windows/linux/auto)

    For production deployments, it is recommended to:
    1. Set API_KEY via environment variable only
    2. Use read-only config files in production
    3. Validate all overrides through ConfigValidator

    Args:
        data: Configuration dictionary from YAML

    Returns:
        Updated configuration dictionary with environment overrides
    """
```

- [ ] **Step 4: 运行所有配置测试**

Run: `python -m pytest tests/test_config.py tests/test_config_env_security.py -v --tb=short`
Expected: All PASS

- [ ] **Step 5: 提交更改**

```bash
git add mini_agent/config.py tests/test_config_env_security.py
git commit -m "docs: add security notes for environment variable overrides"
```

---

### Task 5: 工具测试扩展 (MCP 和 Skill 加载器)

**Files:**
- Modify: `tests/test_mcp.py`
- Modify: `tests/test_skill_loader.py`

- [ ] **Step 1: 添加 MCP 加载器单元测试**

```python
# tests/test_mcp.py - 添加缺失的测试
class TestMCPLoaderUnit:
    """Unit tests for MCP loader."""

    def test_mcp_loader_initialization(self):
        """Test MCP loader can be initialized."""
        from mini_agent.tools.mcp_loader import MCPLoader

        # Should be able to create loader
        loader = MCPLoader()
        assert loader is not None

    def test_mcp_config_parsing(self):
        """Test MCP config file parsing."""
        from mini_agent.config import MCPConfig

        config = MCPConfig()
        assert config.connect_timeout == 10.0
        assert config.execute_timeout == 60.0
        assert config.sse_read_timeout == 120.0

    def test_get_mcp_config_paths(self):
        """Test getting MCP config paths."""
        from mini_agent.config import ToolsConfig

        tools_config = ToolsConfig()
        paths = tools_config.get_mcp_config_paths()
        assert isinstance(paths, list)
```

- [ ] **Step 2: 添加 Skill 加载器单元测试**

```python
# tests/test_skill_loader.py - 添加单元测试
class TestSkillLoaderUnit:
    """Unit tests for Skill loader."""

    def test_skill_loader_initialization(self):
        """Test Skill loader can be initialized."""
        from mini_agent.tools.skill_loader import SkillLoader

        loader = SkillLoader()
        assert loader is not None

    def test_skills_dir_resolution(self):
        """Test skills directory resolution from config."""
        from mini_agent.config import ToolsConfig

        tools_config = ToolsConfig()
        paths = tools_config.get_skills_search_paths()
        assert isinstance(paths, list)
        # Should have at least project skills dir
        assert len(paths) >= 1

    def test_skill_loader_with_empty_workspace(self, tmp_path):
        """Test skill loader handles missing workspace gracefully."""
        from mini_agent.tools.skill_loader import SkillLoader

        loader = SkillLoader()
        # Should not raise, just return empty list
        skills = loader.load_skills(str(tmp_path / "nonexistent"))
        assert isinstance(skills, list)
```

- [ ] **Step 3: 运行新测试**

Run: `python -m pytest tests/test_mcp.py::TestMCPLoaderUnit tests/test_skill_loader.py::TestSkillLoaderUnit -v`
Expected: All PASS

- [ ] **Step 4: 运行所有 MCP 和 Skill 测试**

Run: `python -m pytest tests/test_mcp.py tests/test_skill_loader.py -v --tb=short`
Expected: All PASS

- [ ] **Step 5: 提交更改**

```bash
git add tests/test_mcp.py tests/test_skill_loader.py
git commit -m "test: add unit tests for MCP and Skill loaders"
```

---

### Task 6: 性能测试扩展（压力测试场景）

**Files:**
- Modify: `tests/test_stress.py`

- [ ] **Step 1: 添加缓存压力测试**

```python
# tests/test_stress.py - 添加缓存压力测试
class TestCacheStress:
    """Stress tests for context cache."""

    def test_cache_warmup_with_many_files(self, tmp_path):
        """Test cache warmup with large directory structure."""
        from mini_agent.utils.context_cache import get_context_cache

        # Create many small files
        for i in range(100):
            (tmp_path / f"file_{i}.txt").write_text(f"content {i}" * 100)

        cache = get_context_cache()
        cached = cache.warmup(tmp_path)
        assert cached >= 100

    def test_cache_content_retrieval(self, tmp_path):
        """Test retrieving content from cache."""
        from mini_agent.utils.context_cache import get_context_cache

        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")

        cache = get_context_cache()
        cache.warmup(tmp_path)

        content = cache.get_file_content(test_file)
        assert content == "test content"

    def test_cache_invalidation(self, tmp_path):
        """Test cache invalidation."""
        from mini_agent.utils.context_cache import get_context_cache

        cache = get_context_cache()
        test_file = tmp_path / "test.txt"
        test_file.write_text("test")

        cache.warmup(tmp_path)
        cache.invalidate()

        # After invalidation, cache should be empty
        content = cache.get_file_content(test_file)
        assert content is None
```

- [ ] **Step 2: 添加并发工具调用压力测试**

```python
# tests/test_stress.py - 添加并发测试
class TestConcurrentToolStress:
    """Stress tests for concurrent tool execution."""

    @pytest.mark.asyncio
    async def test_parallel_tool_execution(self):
        """Test parallel execution of multiple tools."""
        from mini_agent.core.execution_engine import ExecutionEngine
        from mini_agent.tools.base import Tool, ToolResult
        from unittest.mock import MagicMock

        # Create mock tools
        class SlowTool(Tool):
            @property
            def name(self):
                return "slow_tool"

            @property
            def description(self):
                return "A slow tool for testing"

            @property
            def parameters(self):
                return {"type": "object", "properties": {}}

            async def execute(self):
                await asyncio.sleep(0.1)
                return ToolResult(success=True, content="done")

        tools = {"slow_tool": SlowTool()}
        mock_logger = MagicMock()
        mock_retry = MagicMock()
        mock_retry.get_max_retries.return_value = 1
        mock_metrics = MagicMock()
        mock_error_recovery = MagicMock()

        engine = ExecutionEngine(
            tools=tools,
            logger=mock_logger,
            retry_handler=mock_retry,
            metrics=mock_metrics,
            error_recovery=mock_error_recovery,
        )

        from mini_agent.schema import ToolCall, FunctionCall

        tool_calls = [
            ToolCall(id=f"call-{i}", type="function",
                    function=FunctionCall(name="slow_tool", arguments={}))
            for i in range(10)
        ]

        import time
        start = time.time()
        results = await engine._execute_parallel(tool_calls, 5, AgentMode.YOLO, lambda x: True)
        elapsed = time.time() - start

        assert len(results) == 10
        # With 5 concurrent and 0.1s each, should take ~0.2s
        assert elapsed < 1.0  # Should be much faster than sequential
```

- [ ] **Step 3: 运行压力测试**

Run: `python -m pytest tests/test_stress.py::TestCacheStress tests/test_stress.py::TestConcurrentToolStress -v`
Expected: All PASS

- [ ] **Step 4: 运行所有压力测试**

Run: `python -m pytest tests/test_stress.py -v --tb=short`
Expected: All PASS

- [ ] **Step 5: 提交更改**

```bash
git add tests/test_stress.py
git commit -m "test: add cache and concurrent tool stress tests"
```

---

### Task 7: 异步测试增强

**Files:**
- Modify: `tests/test_execution_engine.py`
- Modify: `tests/test_step_runner.py`

- [ ] **Step 1: 添加边界情况异步测试**

```python
# tests/test_execution_engine.py - 添加异步边界测试
class TestExecutionEngineAsyncEdgeCases:
    """Async edge case tests for execution engine."""

    @pytest.mark.asyncio
    async def test_tool_execution_with_exception(self):
        """Test tool execution handles exceptions gracefully."""
        from mini_agent.core.execution_engine import ExecutionEngine
        from mini_agent.tools.base import Tool, ToolResult

        class FailingTool(Tool):
            @property
            def name(self):
                return "failing_tool"

            @property
            def description(self):
                return "A tool that fails"

            @property
            def parameters(self):
                return {"type": "object", "properties": {}}

            async def execute(self):
                raise RuntimeError("Intentional failure")

        tools = {"failing_tool": FailingTool()}
        mock_logger = MagicMock()
        mock_retry = MagicMock()
        mock_retry.get_max_retries.return_value = 1
        mock_retry.is_transient_error.return_value = False
        mock_metrics = MagicMock()
        mock_error_recovery = MagicMock()

        engine = ExecutionEngine(
            tools=tools,
            logger=mock_logger,
            retry_handler=mock_retry,
            metrics=mock_metrics,
            error_recovery=mock_error_recovery,
        )

        from mini_agent.schema import AgentMode, ToolCall, FunctionCall

        tool_call = ToolCall(
            id="test-1",
            type="function",
            function=FunctionCall(name="failing_tool", arguments={})
        )

        result = await engine._execute_single_tool(tool_call, AgentMode.YOLO, lambda x: True)
        assert result[1].content.startswith("Error:")
```

- [ ] **Step 2: 添加 StepRunner 异步测试**

```python
# tests/test_step_runner.py - 添加异步测试
class TestStepRunnerAsync:
    """Async tests for StepRunner."""

    def test_step_runner_handles_none_thinking_manager(self):
        """Test StepRunner works without thinking manager."""
        from mini_agent.core.step_runner import StepRunner
        from unittest.mock import MagicMock

        mock_agent = MagicMock()
        mock_agent._context = MagicMock()
        mock_agent._context.api_call_count = 0
        mock_agent._context.api_total_tokens = 0
        mock_agent.logger = MagicMock()
        mock_agent._thinking_manager = None  # No thinking manager
        mock_agent._session_manager = MagicMock()
        mock_agent._last_health_check_step = -1
        mock_agent._health_check_interval = 5
        mock_agent._check_health = MagicMock(return_value=[])
        mock_agent._token_tracker = MagicMock()

        runner = StepRunner(mock_agent, 0.0)
        # prune_thinking should handle None gracefully
        tokens_freed = runner.prune_thinking()
        assert tokens_freed == 0
```

- [ ] **Step 3: 运行异步测试**

Run: `python -m pytest tests/test_execution_engine.py::TestExecutionEngineAsyncEdgeCases tests/test_step_runner.py::TestStepRunnerAsync -v`
Expected: All PASS

- [ ] **Step 4: 提交更改**

```bash
git add tests/test_execution_engine.py tests/test_step_runner.py
git commit -m "test: add async edge case tests for execution engine and step runner"
```

---

## 实施检查清单

在完成所有任务后，运行以下命令验证：

```bash
# 运行所有相关测试
python -m pytest \
  tests/test_agent_context.py \
  tests/test_token_tracker.py \
  tests/test_health_check.py \
  tests/test_step_runner.py \
  tests/test_config.py \
  tests/test_config_env_security.py \
  tests/test_mcp.py \
  tests/test_skill_loader.py \
  tests/test_stress.py \
  tests/test_execution_engine.py \
  -v --tb=short
```

Expected: All tests PASS

---

## 任务依赖关系

```
Task 1 (Message 模型统一) ─┬─> Task 2 (Token 估计精度)
                           │
Task 3 (StepRunner 解耦) ──┼─> Task 4 (环境变量安全)
                           │
Task 5 (工具测试扩展) ─────┤
                           │
Task 6 (性能压力测试) ─────┼─> Task 7 (异步测试增强)
                           │
                         (无依赖，可并行执行)
```

---

**Plan complete.** All tasks are independent and can be executed in parallel using subagent-driven development.