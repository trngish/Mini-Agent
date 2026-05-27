from __future__ import annotations

import time

from mini_agent.core.rate_limiter import RateLimiter


class TestRateLimiterInit:
    def test_default_limits(self):
        rl = RateLimiter()
        assert rl._global_limit == 100
        assert rl._global_window == 60
        assert rl._per_tool_limit == 30
        assert rl._per_tool_window == 60
        assert rl._input_max_length == 1_000_000

    def test_custom_limits(self):
        rl = RateLimiter(
            global_limit=50,
            global_window=30,
            per_tool_limit=10,
            per_tool_window=30,
            input_max_length=500_000,
        )
        assert rl._global_limit == 50
        assert rl._global_window == 30
        assert rl._per_tool_limit == 10
        assert rl._input_max_length == 500_000


class TestCheckRate:
    def test_allows_within_limit(self):
        rl = RateLimiter(per_tool_limit=5, global_limit=10)
        allowed, msg = rl.check_rate("bash")
        assert allowed is True
        assert msg == ""

    def test_blocks_per_tool_limit(self):
        rl = RateLimiter(per_tool_limit=3, per_tool_window=60, global_limit=100)
        for _ in range(3):
            rl.check_rate("bash")
        allowed, msg = rl.check_rate("bash")
        assert allowed is False
        assert "bash" in msg
        assert "exceeded" in msg

    def test_different_tools_independent(self):
        rl = RateLimiter(per_tool_limit=2, global_limit=100)
        rl.check_rate("bash")
        rl.check_rate("bash")
        allowed, _ = rl.check_rate("read")
        assert allowed is True

    def test_blocks_global_limit(self):
        rl = RateLimiter(global_limit=3, global_window=60, per_tool_limit=100)
        for i in range(3):
            rl.check_rate(f"tool_{i}")
        allowed, msg = rl.check_rate("tool_4")
        assert allowed is False
        assert "Global rate limit" in msg

    def test_window_expiry(self):
        rl = RateLimiter(per_tool_limit=2, per_tool_window=1, global_limit=100)
        rl.check_rate("bash")
        rl.check_rate("bash")
        allowed, _ = rl.check_rate("bash")
        assert allowed is False
        time.sleep(1.1)
        allowed, _ = rl.check_rate("bash")
        assert allowed is True

    def test_retry_after_hint(self):
        rl = RateLimiter(per_tool_limit=1, per_tool_window=60, global_limit=100)
        rl.check_rate("bash")
        allowed, msg = rl.check_rate("bash")
        assert allowed is False
        assert "Retry after" in msg


class TestValidateInputLength:
    def test_valid_short_input(self):
        rl = RateLimiter(input_max_length=100)
        valid, msg = rl.validate_input_length("bash", {"cmd": "ls"})
        assert valid is True
        assert msg == ""

    def test_rejects_long_input(self):
        rl = RateLimiter(input_max_length=10)
        valid, msg = rl.validate_input_length("bash", {"cmd": "a" * 100})
        assert valid is False
        assert "too long" in msg
        assert "bash.cmd" in msg

    def test_non_string_values_pass(self):
        rl = RateLimiter(input_max_length=10)
        valid, _ = rl.validate_input_length("tool", {"count": 99999})
        assert valid is True

    def test_empty_arguments_pass(self):
        rl = RateLimiter(input_max_length=10)
        valid, _ = rl.validate_input_length("tool", {})
        assert valid is True

    def test_multiple_args_first_too_long(self):
        rl = RateLimiter(input_max_length=5)
        valid, msg = rl.validate_input_length("tool", {"a": "123456", "b": "ok"})
        assert valid is False
        assert "tool.a" in msg

    def test_multiple_args_second_too_long(self):
        rl = RateLimiter(input_max_length=5)
        valid, msg = rl.validate_input_length("tool", {"a": "ok", "b": "123456"})
        assert valid is False
        assert "tool.b" in msg


class TestReset:
    def test_reset_clears_counters(self):
        rl = RateLimiter(per_tool_limit=2, global_limit=100)
        rl.check_rate("bash")
        rl.check_rate("bash")
        allowed, _ = rl.check_rate("bash")
        assert allowed is False
        rl.reset()
        allowed, _ = rl.check_rate("bash")
        assert allowed is True

    def test_reset_clears_all_tools(self):
        rl = RateLimiter(per_tool_limit=1, global_limit=100)
        rl.check_rate("bash")
        rl.check_rate("read")
        rl.reset()
        assert rl.check_rate("bash")[0] is True
        assert rl.check_rate("read")[0] is True


class TestThreadSafety:
    def test_concurrent_access(self):
        import threading

        rl = RateLimiter(global_limit=1000, per_tool_limit=500)
        errors: list[Exception] = []

        def worker():
            try:
                for _ in range(50):
                    rl.check_rate("bash")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
