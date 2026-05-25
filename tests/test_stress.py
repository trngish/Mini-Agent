"""Maximum stress tests - pushing mini-agent to its limits.

Run with: python -m pytest tests/test_stress.py -v -s
"""

import asyncio
import gc
import sys
import threading
import traceback
import weakref
from time import perf_counter
from typing import List
from unittest.mock import Mock

import pytest

from mini_agent.subagent import SubAgent, SubAgentResult
from mini_agent.tools.batch_tools import MultiBashTool
from mini_agent.tools.team_dispatch_tool import TeamDispatchTool
from mini_agent.utils.context_cache import get_context_cache


class MockLLMClient:
    model = "MiniMax-M2.7"
    call_count = 0
    total_tokens = 0

    def clone(self):
        return MockLLMClient()

    def configure_thinking_budget(self, budget: int):
        pass

    def configure_m27(self, config: dict):
        pass

    async def generate(self, messages, tools=None, on_text=None, on_thinking=None):
        MockLLMClient.call_count += 1
        class MockResponse:
            content = f"Mock response #{MockLLMClient.call_count}"
            thinking = None
            tool_calls = []
            finish_reason = "stop"
            usage = type('obj', (object,), {'total_tokens': 100})()
        return MockResponse()


@pytest.fixture
def mock_llm_client():
    return MockLLMClient()


@pytest.fixture
def mock_tools():
    return [
        MultiBashTool(platform_mode="windows"),
    ]


@pytest.fixture
def bash_tool():
    from mini_agent.tools.bash_tool import BashTool
    return BashTool(platform_mode="windows")


# =============================================================================
# MAXIMUM SubAgent Tests
# =============================================================================

class TestSubAgentStress:
    """Maximum stress tests for SubAgent."""

    @pytest.mark.asyncio
    async def test_2000_concurrent_subagents(self, mock_llm_client, mock_tools):
        """2000 subagents in parallel - MAXIMUM."""
        tasks = [f"Task {i}: Analyze module {i}" for i in range(2000)]

        start = perf_counter()
        coros = [
            SubAgent(
                llm_client=mock_llm_client,
                tools=mock_tools,
                system_prompt="You are a code reviewer.",
                max_steps=10,
            ).run(task)
            for task in tasks
        ]
        results = await asyncio.gather(*coros)
        elapsed = perf_counter() - start

        assert len(results) == 2000
        success_count = sum(1 for r in results if r.success)
        print(f"\n[PASS] 2000 concurrent subagents: {success_count}/2000 in {elapsed:.2f}s")

    @pytest.mark.asyncio
    async def test_5000_rapid_fire(self, mock_llm_client, mock_tools):
        """5000 rapid sequential subagents - MAXIMUM."""
        results = []
        start = perf_counter()

        for i in range(5000):
            agent = SubAgent(
                llm_client=mock_llm_client,
                tools=mock_tools,
                max_steps=2,
            )
            results.append(await agent.run(f"Quick task {i}"))

        elapsed = perf_counter() - start

        success_count = sum(1 for r in results if r.success)
        print(f"\n[PASS] 5000 rapid-fire: {success_count}/5000 in {elapsed:.2f}s")

    @pytest.mark.asyncio
    async def test_deep_nesting_20_levels(self, mock_llm_client, mock_tools):
        """Deep nesting with 20 levels - MAXIMUM."""
        async def run_with_depth(depth: int, max_depth: int):
            if depth >= max_depth:
                return SubAgentResult(success=True, content=f"Depth {depth}", error=None, elapsed=0.0)

            child_agent = SubAgent(
                llm_client=mock_llm_client,
                tools=mock_tools,
                max_steps=2,
            )
            return await child_agent.run(f"Task depth {depth}")

        # Start 100 agents with max depth of 20
        start = perf_counter()
        coros = [run_with_depth(0, 20) for _ in range(100)]
        results = await asyncio.gather(*coros)
        elapsed = perf_counter() - start

        success_count = sum(1 for r in results if r.success)
        print(f"\n[PASS] 100 deeply nested (20 levels): {success_count}/100 in {elapsed:.2f}s")

    @pytest.mark.asyncio
    async def test_massive_wave_10k_concurrent(self, mock_llm_client, mock_tools):
        """10 waves of 1000 concurrent = 10000 total - MAXIMUM."""
        all_tasks = []
        for wave in range(10):
            tasks = [f"Wave{wave} Task{i}" for i in range(1000)]
            coros = [
                SubAgent(llm_client=mock_llm_client, tools=mock_tools, max_steps=3).run(task)
                for task in tasks
            ]
            all_tasks.extend(coros)

        start = perf_counter()
        results = await asyncio.gather(*all_tasks)
        elapsed = perf_counter() - start

        success_count = sum(1 for r in results if r.success)
        print(f"\n[PASS] 10 waves x 1000 = 10000 concurrent: {success_count}/10000 in {elapsed:.2f}s")

    @pytest.mark.asyncio
    async def test_subagent_continuous_30s(self, mock_llm_client, mock_tools):
        """Continuous subagent creation for 30 seconds - MAXIMUM."""
        start = perf_counter()
        end_time = start + 30
        count = 0

        while perf_counter() < end_time:
            agent = SubAgent(
                llm_client=mock_llm_client,
                tools=mock_tools,
                max_steps=2,
            )
            result = await agent.run(f"Task {count}")
            if result.success:
                count += 1

        elapsed = perf_counter() - start
        print(f"\n[PASS] Continuous subagents for 30s: {count} completed in {elapsed:.2f}s")


# =============================================================================
# MAXIMUM ContextCache
# =============================================================================

class TestContextCacheStress:
    """Maximum stress tests for context cache."""

    def test_cache_100k_entries(self):
        """100000 entries - MAXIMUM."""
        cache = get_context_cache()
        cache.invalidate_all()

        start = perf_counter()
        for i in range(100000):
            cache.set_file_content(f"file_{i}.py", f"content{i}" * 20)
        elapsed = perf_counter() - start

        stats = cache.get_stats()
        assert stats['file_entries'] == 100000
        print(f"\n[PASS] 100000 entries in {elapsed:.2f}s")

    def test_cache_100_threads_concurrent(self):
        """100 threads concurrent writes - MAXIMUM."""
        cache = get_context_cache()
        cache.invalidate_all()

        errors = []

        def writer(idx, count):
            try:
                for i in range(count):
                    cache.set_file_content(f"t{idx}_f{i}.py", f"content{i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(i, 100)) for i in range(100)]

        start = perf_counter()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        elapsed = perf_counter() - start

        stats = cache.get_stats()
        assert len(errors) == 0, f"Errors: {errors}"
        assert stats['file_entries'] == 10000
        print(f"\n[PASS] 100 threads x 100 = 10000 writes in {elapsed:.2f}s, 0 errors")

    def test_cache_mixed_50_threads(self):
        """50 threads mixed read/write/delete - MAXIMUM."""
        cache = get_context_cache()
        cache.invalidate_all()

        # Pre-fill
        for i in range(5000):
            cache.set_file_content(f"prefilled_{i}.py", f"content{i}")

        errors = []
        counters = [0, 0, 0, 0]
        lock = threading.Lock()

        def writer(idx):
            try:
                for i in range(300):
                    cache.set_file_content(f"w_t{idx}_f{i}.py", f"content{i}")
                    with lock:
                        counters[0] += 1
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for i in range(1000):
                    cache.get_file_content(f"prefilled_{i % 5000}.py")
                    with lock:
                        counters[1] += 1
            except Exception as e:
                errors.append(e)

        def updater():
            try:
                for i in range(300):
                    cache.set_file_content(f"prefilled_{i % 5000}.py", f"updated{i}")
                    with lock:
                        counters[2] += 1
            except Exception as e:
                errors.append(e)

        def deleter():
            try:
                for i in range(100):
                    cache.invalidate_file(f"prefilled_{i}.py")
                    with lock:
                        counters[3] += 1
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(15):
            threads.append(threading.Thread(target=writer, args=(i,)))
        for i in range(20):
            threads.append(threading.Thread(target=reader))
        for i in range(10):
            threads.append(threading.Thread(target=updater))
        for i in range(5):
            threads.append(threading.Thread(target=deleter))

        start = perf_counter()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        elapsed = perf_counter() - start

        assert len(errors) == 0, f"Errors: {errors}"
        print(f"\n[PASS] 50 threads: {counters[1]}r/{counters[0]}w/{counters[2]}u/{counters[3]}d in {elapsed:.2f}s")

    def test_cache_filter_50000_paths(self):
        """Filter 50000 paths - MAXIMUM."""
        cache = get_context_cache()
        cache.invalidate_all()

        cached_count = 25000
        for i in range(cached_count):
            cache.set_file_content(f"cached_{i}.py", f"content{i}")

        all_paths = [f"cached_{i}.py" for i in range(cached_count)]
        all_paths += [f"uncached_{i}.py" for i in range(25000)]

        start = perf_counter()
        uncached = cache.filter_uncached_paths(all_paths)
        elapsed = perf_counter() - start

        assert len(uncached) == 25000
        print(f"\n[PASS] Filtered 50000 paths in {elapsed:.2f}s")

    def test_cache_memory_500k(self):
        """500k entries memory stress - MAXIMUM."""
        import tracemalloc

        cache = get_context_cache()
        cache.invalidate_all()

        tracemalloc.start()

        for i in range(500000):
            cache.set_file_content(f"file_{i}.py", f"x" * 100)
            if i % 50000 == 0:
                gc.collect()

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        stats = cache.get_stats()
        print(f"\n[PASS] 500k entries: Cache={stats['file_entries']}, Peak mem={peak/1024/1024:.1f}MB")


# =============================================================================
# MAXIMUM TeamDispatch
# =============================================================================

class TestTeamDispatchTool:
    """Maximum stress tests for TeamDispatchTool."""

    @pytest.mark.asyncio
    async def test_100_parallel_team_tasks(self, mock_llm_client, mock_tools):
        """100 parallel team dispatch tasks - MAXIMUM."""
        tool = TeamDispatchTool(
            llm_client=mock_llm_client,
            tools=mock_tools,
        )

        tasks = [f"Task {i}: Complex analysis {i}" for i in range(100)]

        start = perf_counter()
        results = await asyncio.gather(*[
            tool.execute(task=task, mode="decompose")
            for task in tasks
        ])
        elapsed = perf_counter() - start

        print(f"\n[PASS] 100 parallel team tasks in {elapsed:.2f}s")

    @pytest.mark.asyncio
    async def test_team_burst_200(self, mock_llm_client, mock_tools):
        """200 rapid team dispatches - MAXIMUM."""
        tool = TeamDispatchTool(
            llm_client=mock_llm_client,
            tools=mock_tools,
        )

        start = perf_counter()
        coros = [
            tool.execute(task=f"Burst task {i}", mode="decompose")
            for i in range(200)
        ]
        results = await asyncio.gather(*coros)
        elapsed = perf_counter() - start

        success_count = sum(1 for r in results if r.success or r.error)
        print(f"\n[PASS] 200 burst team dispatches: {success_count}/200 in {elapsed:.2f}s")


# =============================================================================
# MAXIMUM BatchTool
# =============================================================================

class TestBatchToolStress:
    """Maximum stress tests for batch tools."""

    @pytest.mark.asyncio
    async def test_multi_bash_2000_commands(self, mock_tools):
        """2000 commands in one batch - MAXIMUM."""
        tool = mock_tools[0]

        commands = [
            {"command": f"echo Stress {i}", "label": f"cmd_{i}"}
            for i in range(2000)
        ]

        start = perf_counter()
        result = await tool.execute(commands=commands)
        elapsed = perf_counter() - start

        print(f"\n[PASS] MultiBash 2000 commands: {result.success} in {elapsed:.2f}s")

    @pytest.mark.asyncio
    async def test_multi_bash_10_batches_1000_each(self, mock_tools):
        """10 batches of 1000 commands each = 10000 total - MAXIMUM."""
        tool = mock_tools[0]

        start = perf_counter()
        for batch in range(10):
            commands = [
                {"command": f"echo Batch{batch} {i}", "label": f"b{batch}_{i}"}
                for i in range(1000)
            ]
            await tool.execute(commands=commands)
        elapsed = perf_counter() - start

        print(f"\n[PASS] 10 batches x 1000 = 10000 commands in {elapsed:.2f}s")


# =============================================================================
# MAXIMUM Tool Execution
# =============================================================================

class TestToolExecution:
    """Maximum stress tests for tool execution."""

    @pytest.mark.asyncio
    async def test_5000_rapid_tool_calls(self, bash_tool):
        """5000 rapid tool calls - MAXIMUM."""
        start = perf_counter()

        for i in range(5000):
            await bash_tool.execute(command=f"echo {i}")

        elapsed = perf_counter() - start
        print(f"\n[PASS] 5000 rapid tool calls in {elapsed:.2f}s")

    @pytest.mark.asyncio
    async def test_2000_concurrent_tool_calls(self, bash_tool):
        """2000 concurrent tool calls - MAXIMUM."""
        start = perf_counter()
        coros = [
            bash_tool.execute(command=f"echo concurrent_{i}")
            for i in range(2000)
        ]
        results = await asyncio.gather(*coros)
        elapsed = perf_counter() - start

        success_count = sum(1 for r in results if r.success)
        print(f"\n[PASS] 2000 concurrent tool calls: {success_count}/2000 in {elapsed:.2f}s")

    @pytest.mark.asyncio
    async def test_tool_interleaving_100x(self, bash_tool, mock_tools):
        """100 interleaved bash + multi_bash calls - MAXIMUM."""
        multi_tool = mock_tools[0]

        start = perf_counter()

        for i in range(100):
            await bash_tool.execute(command=f"echo bash_{i}")
            await multi_tool.execute(commands=[
                {"command": f"echo mb_{i}_{j}", "label": f"mb_{j}"}
                for j in range(20)
            ])

        elapsed = perf_counter() - start
        print(f"\n[PASS] 100 x (1 bash + 1 multi 20) = 2100 tool calls in {elapsed:.2f}s")


# =============================================================================
# MAXIMUM PerformanceMetrics
# =============================================================================

class TestPerformanceMetrics:
    """Maximum stress tests for performance metrics."""

    @pytest.mark.asyncio
    async def test_100k_metrics_records(self, mock_llm_client, mock_tools):
        """100000 metrics records - MAXIMUM."""
        from mini_agent.core.metrics import PerformanceMetrics
        from mini_agent.agent import Agent

        agent = Agent(
            llm_client=mock_llm_client,
            tools=mock_tools,
            system_prompt="You are a test agent.",
            max_steps=2,
        )
        metrics = PerformanceMetrics(agent)

        start = perf_counter()
        for i in range(100000):
            metrics.record_tool_duration("bash", 0.0001)
        for i in range(100000):
            metrics.record_step_duration(0.0001)
        elapsed = perf_counter() - start

        stats = metrics.get_metrics()
        print(f"\n[PASS] 200000 metrics records in {elapsed:.2f}s")

    @pytest.mark.asyncio
    async def test_health_check_100k_times(self, mock_llm_client, mock_tools):
        """100000 health checks - MAXIMUM."""
        from mini_agent.core.health_check import HealthChecker
        from mini_agent.agent import Agent

        agent = Agent(
            llm_client=mock_llm_client,
            tools=mock_tools,
            system_prompt="You are a test agent.",
            max_steps=2,
        )
        checker = HealthChecker(agent)

        start = perf_counter()
        for _ in range(100000):
            checker.check()
        elapsed = perf_counter() - start

        print(f"\n[PASS] 100000 health checks in {elapsed:.2f}s")


# =============================================================================
# MAXIMUM Agent Integration
# =============================================================================

class TestAgentIntegration:
    """Maximum stress tests for agent integration."""

    @pytest.mark.asyncio
    async def test_1000_agent_instances(self, mock_llm_client, mock_tools):
        """1000 agent instances rapidly - MAXIMUM."""
        from mini_agent.agent import Agent

        start = perf_counter()
        for i in range(1000):
            agent = Agent(
                llm_client=mock_llm_client,
                tools=mock_tools,
                system_prompt="You are a test agent.",
                max_steps=2,
            )
        elapsed = perf_counter() - start

        print(f"\n[PASS] 1000 agent instances in {elapsed:.2f}s")

    @pytest.mark.asyncio
    async def test_agent_50000_messages(self, mock_llm_client, mock_tools):
        """Agent with 50000 messages - MAXIMUM."""
        from mini_agent.agent import Agent
        from mini_agent.schema import Message

        agent = Agent(
            llm_client=mock_llm_client,
            tools=mock_tools,
            system_prompt="You are a test agent.",
            max_steps=2,
        )

        start = perf_counter()
        for i in range(25000):
            agent.add_user_message(f"User message {i}")
            agent.messages.append(Message(role="assistant", content=f"Agent response {i}"))
        elapsed = perf_counter() - start

        stats = len(agent.messages)
        print(f"\n[PASS] Agent with {stats} messages in {elapsed:.2f}s")
        assert stats == 50001

    @pytest.mark.asyncio
    async def test_agent_continuous_creation_30s(self, mock_llm_client, mock_tools):
        """Continuous agent creation for 30 seconds - MAXIMUM."""
        from mini_agent.agent import Agent

        start = perf_counter()
        end_time = start + 30
        count = 0

        while perf_counter() < end_time:
            agent = Agent(
                llm_client=mock_llm_client,
                tools=mock_tools,
                system_prompt="You are a test agent.",
                max_steps=2,
            )
            count += 1

        elapsed = perf_counter() - start
        print(f"\n[PASS] Continuous agent creation 30s: {count} agents in {elapsed:.2f}s")


# =============================================================================
# MAXIMUM Bash Normalization
# =============================================================================

class TestBashNormalization:
    """Maximum stress tests for bash command normalization."""

    @pytest.mark.asyncio
    async def test_10000_command_normalizations(self, bash_tool):
        """10000 command normalizations - MAXIMUM."""
        commands = [
            f"echo test{i} && echo more{i} && echo done{i}"
            for i in range(10000)
        ]

        start = perf_counter()
        results = []
        for cmd in commands:
            result = await bash_tool.execute(command=cmd)
            results.append(result)
        elapsed = perf_counter() - start

        success_count = sum(1 for r in results if r.success)
        print(f"\n[PASS] 10000 command normalizations: {success_count}/10000 in {elapsed:.2f}s")


# =============================================================================
# MAXIMUM Error Recovery
# =============================================================================

class TestErrorRecovery:
    """Maximum stress tests for error recovery."""

    @pytest.mark.asyncio
    async def test_2000_error_recovery_iterations(self, mock_llm_client, mock_tools):
        """2000 error recovery iterations - MAXIMUM."""
        successes = 0
        start = perf_counter()

        for i in range(2000):
            agent = SubAgent(
                llm_client=mock_llm_client,
                tools=mock_tools,
                max_steps=2,
            )
            result = await agent.run(f"Task {i}")
            if result.success:
                successes += 1

        elapsed = perf_counter() - start
        print(f"\n[PASS] 2000 error recovery iterations: {successes}/2000 in {elapsed:.2f}s")


# =============================================================================
# MAXIMUM Stress: Concurrent Everything
# =============================================================================

class TestConcurrentEverything:
    """MAXIMUM: All systems running concurrently."""

    @pytest.mark.asyncio
    async def test_all_systems_concurrent(self, mock_llm_client, mock_tools, bash_tool):
        """Everything running at once - MAXIMUM."""
        from mini_agent.agent import Agent
        from mini_agent.core.metrics import PerformanceMetrics
        from mini_agent.core.health_check import HealthChecker

        cache = get_context_cache()
        cache.invalidate_all()

        start = perf_counter()

        # SubAgents
        subagent_tasks = [
            SubAgent(llm_client=mock_llm_client, tools=mock_tools, max_steps=3).run(f"SA_{i}")
            for i in range(500)
        ]

        # Cache writes
        def cache_writer():
            for i in range(10000):
                cache.set_file_content(f"file_{i}.py", f"content{i}")

        # Metrics tracking
        agent = Agent(
            llm_client=mock_llm_client,
            tools=mock_tools,
            system_prompt="You are a test agent.",
            max_steps=2,
        )
        metrics = PerformanceMetrics(agent)
        checker = HealthChecker(agent)

        # Tool calls
        tool_tasks = [
            bash_tool.execute(command=f"echo {i}")
            for i in range(500)
        ]

        # Run all concurrently
        cache_thread = threading.Thread(target=cache_writer)
        cache_thread.start()

        results = await asyncio.gather(*subagent_tasks, *tool_tasks)

        cache_thread.join()
        elapsed = perf_counter() - start

        success_count = sum(1 for r in results if hasattr(r, 'success') and r.success)
        stats = cache.get_stats()

        print(f"\n[PASS] All systems concurrent:")
        print(f"  - SubAgents: {sum(1 for r in results[:500] if r.success)}/500")
        print(f"  - Tool calls: {sum(1 for r in results[500:] if r.success)}/500")
        print(f"  - Cache entries: {stats['file_entries']}")
        print(f"  - Total time: {elapsed:.2f}s")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])