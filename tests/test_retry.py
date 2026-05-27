"""Tests for retry.py retry mechanism."""

import asyncio

import pytest

from mini_agent.retry import (
    RetryConfig,
    RetryExhaustedError,
    _get_retryable_exceptions,
    async_retry,
)


class TestRetryConfig:
    """Test RetryConfig class."""

    def test_default_values(self):
        """Test default retry config values."""
        config = RetryConfig()
        assert config.enabled is True
        assert config.max_retries == 3
        assert config.initial_delay == 1.0
        assert config.max_delay == 60.0
        assert config.exponential_base == 2.0

    def test_custom_values(self):
        """Test custom retry config values."""
        config = RetryConfig(
            enabled=False,
            max_retries=5,
            initial_delay=2.0,
            max_delay=120.0,
            exponential_base=3.0,
        )
        assert config.max_retries == 5
        assert config.initial_delay == 2.0
        assert config.max_delay == 120.0
        assert config.exponential_base == 3.0

    def test_calculate_delay(self):
        """Test exponential backoff delay calculation."""
        config = RetryConfig(initial_delay=1.0, exponential_base=2.0, max_delay=60.0)

        # delay = initial_delay * (exponential_base ^ attempt)
        assert config.calculate_delay(0) == 1.0
        assert config.calculate_delay(1) == 2.0
        assert config.calculate_delay(2) == 4.0
        assert config.calculate_delay(3) == 8.0

    def test_calculate_delay_respects_max(self):
        """Test delay calculation respects max_delay cap."""
        config = RetryConfig(initial_delay=1.0, exponential_base=2.0, max_delay=10.0)

        # Should cap at max_delay
        assert config.calculate_delay(10) == 10.0
        assert config.calculate_delay(100) == 10.0


class TestRetryExhaustedError:
    """Test RetryExhaustedError exception."""

    def test_exception_properties(self):
        """Test exception stores last exception and attempts."""
        original_error = ValueError("Test error")
        error = RetryExhaustedError(original_error, 3)

        assert error.last_exception == original_error
        assert error.attempts == 3
        assert "Retry failed after 3 attempts" in str(error)
        assert "Test error" in str(error)


class TestGetRetryableExceptions:
    """Test _get_retryable_exceptions function."""

    def test_returns_tuple(self):
        """Test function returns a tuple of exception types."""
        result = _get_retryable_exceptions()
        assert isinstance(result, tuple)

    def test_excludes_non_retryable(self):
        """Test that fatal exceptions are excluded."""
        result = _get_retryable_exceptions()

        # These should be excluded
        assert asyncio.CancelledError not in result
        assert KeyboardInterrupt not in result
        assert SystemExit not in result

    def test_cached(self):
        """Test that result is cached."""
        result1 = _get_retryable_exceptions()
        result2 = _get_retryable_exceptions()
        assert result1 is result2


class TestAsyncRetry:
    """Test async_retry decorator."""

    @pytest.mark.asyncio
    async def test_successful_call_no_retries(self):
        """Test successful function call doesn't trigger retries."""
        config = RetryConfig(max_retries=3)
        call_count = [0]

        @async_retry(config)
        async def successful_func():
            call_count[0] += 1
            return "success"

        result = await successful_func()

        assert result == "success"
        assert call_count[0] == 1

    @pytest.mark.asyncio
    async def test_retries_on_exception(self):
        """Test that function is retried on exception."""
        config = RetryConfig(max_retries=3, initial_delay=0.01)
        call_count = [0]

        @async_retry(config)
        async def flaky_func():
            call_count[0] += 1
            if call_count[0] < 3:
                raise ValueError("Transient error")
            return "success after retries"

        result = await flaky_func()

        assert result == "success after retries"
        assert call_count[0] == 3

    @pytest.mark.asyncio
    async def test_exhausts_retries(self):
        """Test that retries are exhausted after max_attempts."""
        config = RetryConfig(max_retries=2, initial_delay=0.01)
        call_count = [0]

        @async_retry(config)
        async def always_fails():
            call_count[0] += 1
            raise ValueError("Always fails")

        with pytest.raises(RetryExhaustedError) as exc_info:
            await always_fails()

        assert exc_info.value.attempts == 3  # Initial + 2 retries
        assert call_count[0] == 3

    @pytest.mark.asyncio
    async def test_non_retryable_exception_not_retried(self):
        """Test that fatal exceptions are not retried."""
        config = RetryConfig(max_retries=3, initial_delay=0.01)
        call_count = [0]

        @async_retry(config)
        async def keyboard_interrupt_func():
            call_count[0] += 1
            raise KeyboardInterrupt("Interrupted")

        with pytest.raises(KeyboardInterrupt):
            await keyboard_interrupt_func()

        # Should not have retried
        assert call_count[0] == 1

    @pytest.mark.asyncio
    async def test_on_retry_callback(self):
        """Test that on_retry callback is called."""
        config = RetryConfig(max_retries=2, initial_delay=0.01)
        callback_calls = []

        def on_retry(exc, attempt):
            callback_calls.append((exc, attempt))

        @async_retry(config, on_retry=on_retry)
        async def flaky_with_callback():
            raise ValueError("Error")

        with pytest.raises(RetryExhaustedError):
            await flaky_with_callback()

        assert len(callback_calls) == 2
        assert callback_calls[0][1] == 1  # First retry attempt
        assert callback_calls[1][1] == 2  # Second retry attempt

    @pytest.mark.asyncio
    async def test_exponential_backoff_timing(self):
        """Test that exponential backoff is applied."""
        config = RetryConfig(max_retries=2, initial_delay=0.1, max_delay=60.0)
        call_times = []

        async def record_call_time():
            call_times.append(asyncio.get_event_loop().time())
            if len(call_times) < 3:
                raise ValueError("Error")
            return "success"

        @async_retry(config)
        async def func_with_timing():
            return await record_call_time()

        await func_with_timing()

        # Check delays between calls
        delay1 = call_times[1] - call_times[0]
        delay2 = call_times[2] - call_times[1]

        # First retry: 0.1 * 2^0 = 0.1s
        # Second retry: 0.1 * 2^1 = 0.2s
        # We allow some tolerance for test execution time
        assert delay1 >= 0.08  # Allow some tolerance
        assert delay2 >= 0.18

    @pytest.mark.asyncio
    async def test_default_config(self):
        """Test decorator works with default config."""
        call_count = [0]

        @async_retry()
        async def default_retry_func():
            call_count[0] += 1
            if call_count[0] < 2:
                raise ValueError("Retry")
            return "success"

        result = await default_retry_func()
        assert result == "success"
        assert call_count[0] == 2

    @pytest.mark.asyncio
    async def test_preserves_function_name(self):
        """Test that decorator preserves function metadata."""

        @async_retry()
        async def my_function():
            return "result"

        assert my_function.__name__ == "my_function"

    @pytest.mark.asyncio
    async def test_with_args_and_kwargs(self):
        """Test that function args and kwargs are passed through."""
        config = RetryConfig(max_retries=1, initial_delay=0.01)
        received_args = []
        received_kwargs = []

        @async_retry(config)
        async def func_with_args(arg1, arg2, kwarg1=None):
            received_args.append(arg1)
            received_args.append(arg2)
            received_kwargs.append(kwarg1)
            if len(received_args) < 3:
                raise ValueError("Retry")
            return "done"

        result = await func_with_args("a", "b", kwarg1="c")

        assert result == "done"
        assert "a" in received_args
        assert "b" in received_args
        assert "c" in received_kwargs


class TestRetryDecoratorEdgeCases:
    """Test edge cases for retry decorator."""

    @pytest.mark.asyncio
    async def test_successful_after_previous_failure(self):
        """Test successful call after failures in previous calls."""
        config = RetryConfig(max_retries=1, initial_delay=0.01)
        call_count = [0]

        @async_retry(config)
        async def eventually_successful():
            call_count[0] += 1
            return "success"

        # First call succeeds immediately
        result1 = await eventually_successful()
        assert result1 == "success"
        assert call_count[0] == 1

        # Call again - should also succeed immediately
        result2 = await eventually_successful()
        assert result2 == "success"
        assert call_count[0] == 2

    @pytest.mark.asyncio
    async def test_multiple_exception_types(self):
        """Test retry with multiple exception types."""
        config = RetryConfig(max_retries=3, initial_delay=0.01)
        call_count = [0]

        @async_retry(config)
        async def multi_exception_func():
            call_count[0] += 1
            if call_count[0] == 1:
                raise ValueError("Value error")
            elif call_count[0] == 2:
                raise TypeError("Type error")
            return "success"

        result = await multi_exception_func()
        assert result == "success"
        assert call_count[0] == 3
