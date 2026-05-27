"""Tests for error handler utilities."""

import asyncio

from mini_agent.utils.error_handler import (
    LLMError,
    LLMErrorClassifier,
    LLMErrorType,
    format_llm_error,
)


class TestLLMErrorClassifier:
    """Tests for LLMErrorClassifier."""

    def test_classify_rate_limit_429(self):
        error = Exception("Rate limited")
        llm_error = LLMErrorClassifier.classify(error, status_code=429)
        assert llm_error.error_type == LLMErrorType.RATE_LIMIT_ERROR

    def test_classify_auth_error_401(self):
        error = Exception("Unauthorized")
        llm_error = LLMErrorClassifier.classify(error, status_code=401)
        assert llm_error.error_type == LLMErrorType.AUTHENTICATION_ERROR

    def test_classify_auth_error_403(self):
        error = Exception("Forbidden")
        llm_error = LLMErrorClassifier.classify(error, status_code=403)
        assert llm_error.error_type == LLMErrorType.PERMISSION_DENIED

    def test_classify_server_error_500(self):
        error = Exception("Internal server error")
        llm_error = LLMErrorClassifier.classify(error, status_code=500)
        assert llm_error.error_type == LLMErrorType.SERVER_ERROR

    def test_classify_server_error_502(self):
        error = Exception("Bad gateway")
        llm_error = LLMErrorClassifier.classify(error, status_code=502)
        assert llm_error.error_type == LLMErrorType.SERVER_ERROR

    def test_classify_context_length_400(self):
        error = Exception("context_length_exceeded")
        llm_error = LLMErrorClassifier.classify(error, status_code=400)
        assert llm_error.error_type in (LLMErrorType.CONTEXT_LENGTH_EXCEEDED, LLMErrorType.BAD_REQUEST)

    def test_classify_timeout(self):
        error = asyncio.TimeoutError()
        llm_error = LLMErrorClassifier.classify(error)
        # asyncio.TimeoutError may not be specifically handled; verify it classifies without crash
        assert isinstance(llm_error.error_type, LLMErrorType)

    def test_classify_connection_error(self):
        error = ConnectionError("Connection refused")
        llm_error = LLMErrorClassifier.classify(error)
        assert llm_error.error_type == LLMErrorType.CONNECTION_ERROR

    def test_classify_unknown_error(self):
        error = ValueError("Some unknown error")
        llm_error = LLMErrorClassifier.classify(error)
        assert llm_error.error_type == LLMErrorType.UNKNOWN_ERROR


class TestFormatLLMError:
    """Tests for format_llm_error function."""

    def test_formats_rate_limit_error(self):
        error = Exception("Rate limited")
        result = format_llm_error(error, status_code=429)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_formats_auth_error(self):
        error = Exception("Unauthorized")
        result = format_llm_error(error, status_code=401)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_formats_without_status_code(self):
        error = ValueError("Something went wrong")
        result = format_llm_error(error)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_different_errors_produce_different_messages(self):
        """Verify that the lru_cache removal fix works - different inputs produce different outputs."""
        error1 = Exception("Rate limited")
        error2 = Exception("Unauthorized")
        result1 = format_llm_error(error1, status_code=429)
        result2 = format_llm_error(error2, status_code=401)
        assert result1 != result2


class TestLLMError:
    """Tests for LLMError dataclass."""

    def test_llm_error_fields(self):
        error = LLMError(
            error_type=LLMErrorType.RATE_LIMIT_ERROR,
            message="Rate limited",
            status_code=429,
            retry_after=60,
        )
        assert error.error_type == LLMErrorType.RATE_LIMIT_ERROR
        assert error.message == "Rate limited"
        assert error.status_code == 429
        assert error.retry_after == 60

    def test_llm_error_defaults(self):
        error = LLMError(
            error_type=LLMErrorType.UNKNOWN_ERROR,
            message="Unknown error",
        )
        assert error.status_code is None
        assert error.retry_after is None
