"""Tests for ContextCache."""

import time

import pytest

from mini_agent.utils.context_cache import (
    ContextCache,
    get_cache_hit_rate,
    get_context_cache,
    record_cache_access,
)


class TestContextCache:
    """Test ContextCache functionality."""

    @pytest.fixture
    def cache(self):
        """Create a fresh ContextCache instance."""
        return ContextCache(max_memory_mb=10.0)

    def test_set_and_get_file_content(self, cache):
        """Test basic file content caching."""
        cache.set_file_content("/test.py", "print('hello')")
        result = cache.get_file_content("/test.py")

        assert result == "print('hello')"

    def test_get_file_content_not_cached(self, cache):
        """Test getting content that hasn't been cached."""
        result = cache.get_file_content("/nonexistent.py")
        assert result is None

    def test_get_file_content_expired(self, cache):
        """Test that expired content returns None."""
        cache.set_file_content("/test.py", "content", ttl=1)  # 1 second TTL
        time.sleep(1.1)  # Wait for expiration

        result = cache.get_file_content("/test.py")
        assert result is None

    def test_set_file_content_updates_existing(self, cache):
        """Test updating existing cached content."""
        cache.set_file_content("/test.py", "version 1")
        cache.set_file_content("/test.py", "version 2")

        result = cache.get_file_content("/test.py")
        assert result == "version 2"

    def test_invalidate_file(self, cache):
        """Test file cache invalidation."""
        cache.set_file_content("/test.py", "content")
        cache.invalidate_file("/test.py")

        result = cache.get_file_content("/test.py")
        assert result is None

    def test_set_and_get_tree(self, cache):
        """Test tree output caching."""
        cache.set_tree("/project", 2, "directory structure")
        result = cache.get_tree("/project", 2)

        assert result == "directory structure"

    def test_set_tree_different_depths(self, cache):
        """Test tree caching with different depths."""
        cache.set_tree("/project", 1, "depth 1")
        cache.set_tree("/project", 2, "depth 2")

        assert cache.get_tree("/project", 1) == "depth 1"
        assert cache.get_tree("/project", 2) == "depth 2"

    def test_invalidate_tree(self, cache):
        """Test tree cache invalidation."""
        cache.set_tree("/project", 2, "structure")
        cache.invalidate_tree("/project", 2)

        result = cache.get_tree("/project", 2)
        assert result is None

    def test_set_and_get_grep_result(self, cache):
        """Test grep result caching."""
        cache.set_grep_result("test", "/project", "*.py", True, ["file1.py", "file2.py"])
        result = cache.get_grep_result("test", "/project", "*.py", True)

        assert result == ["file1.py", "file2.py"]

    def test_grep_cache_key_includes_all_params(self, cache):
        """Test that grep cache key includes all search parameters."""
        cache.set_grep_result("test", "/project", "*.py", True, ["result1"])
        cache.set_grep_result("test", "/project", "*.py", False, ["result2"])

        # Different case sensitivity should produce different results
        result_true = cache.get_grep_result("test", "/project", "*.py", True)
        result_false = cache.get_grep_result("test", "/project", "*.py", False)

        assert result_true == ["result1"]
        assert result_false == ["result2"]

    def test_invalidate_grep(self, cache):
        """Test grep cache invalidation."""
        cache.set_grep_result("test", "/project", "*.py", True, ["file.py"])
        cache.invalidate_grep("test", "/project")

        result = cache.get_grep_result("test", "/project", "*.py", True)
        assert result is None

    def test_invalidate_grep_wildcard(self, cache):
        """Test grep invalidation with wildcard."""
        cache.set_grep_result("test1", "/project", "*.py", True, ["file1.py"])
        cache.set_grep_result("test2", "/project", "*.py", True, ["file2.py"])
        cache.set_grep_result("other", "/other", "*.py", True, ["file3.py"])

        cache.invalidate_grep("*", "/project")

        assert cache.get_grep_result("test1", "/project", "*.py", True) is None
        assert cache.get_grep_result("test2", "/project", "*.py", True) is None
        # Other path should still be cached
        assert cache.get_grep_result("other", "/other", "*.py", True) is not None

    def test_invalidate_all(self, cache):
        """Test clearing all caches."""
        cache.set_file_content("/test.py", "content")
        cache.set_tree("/project", 2, "structure")
        cache.set_grep_result("test", "/project", "*.py", True, ["file.py"])

        cache.invalidate_all()

        assert cache.get_file_content("/test.py") is None
        assert cache.get_tree("/project", 2) is None
        assert cache.get_grep_result("test", "/project", "*.py", True) is None

    def test_get_stats(self, cache):
        """Test cache statistics."""
        cache.set_file_content("/a.py", "content a")
        cache.set_file_content("/b.py", "content b")
        cache.set_tree("/project", 2, "structure")

        stats = cache.get_stats()

        assert stats["file_entries"] == 2
        assert stats["tree_entries"] == 1
        assert stats["grep_entries"] == 0

    def test_filter_uncached_paths(self, cache):
        """Test filtering paths that are not in cache."""
        cache.set_file_content("/cached.py", "content")

        paths = ["/cached.py", "/uncached1.py", "/uncached2.py"]
        result = cache.filter_uncached_paths(paths)

        assert "/cached.py" not in result
        assert "/uncached1.py" in result
        assert "/uncached2.py" in result

    def test_filter_uncached_paths_expired(self, cache):
        """Test filtering with expired cache entries."""
        cache.set_file_content("/expired.py", "content", ttl=1)
        time.sleep(1.1)

        paths = ["/expired.py"]
        result = cache.filter_uncached_paths(paths)

        # Expired entries should be treated as uncached
        assert "/expired.py" in result

    def test_lru_eviction_file_cache(self, cache):
        """Test that LRU eviction works for file cache."""
        for i in range(ContextCache.DEFAULT_MAX_FILE_ENTRIES + 10):
            cache.set_file_content(f"/file{i}.py", f"content {i}")

        stats = cache.get_stats()
        assert stats["file_entries"] == ContextCache.DEFAULT_MAX_FILE_ENTRIES

    def test_lru_eviction_tree_cache(self, cache):
        """Test that LRU eviction works for tree cache."""
        for i in range(ContextCache.DEFAULT_MAX_TREE_ENTRIES + 5):
            cache.set_tree(f"/project{i}", 2, f"structure {i}")

        stats = cache.get_stats()
        assert stats["tree_entries"] == ContextCache.DEFAULT_MAX_TREE_ENTRIES

    def test_lru_eviction_grep_cache(self, cache):
        """Test that LRU eviction works for grep cache."""
        for i in range(ContextCache.DEFAULT_MAX_GREP_ENTRIES + 10):
            cache.set_grep_result(f"pattern{i}", "/project", "*.py", True, [f"file{i}.py"])

        stats = cache.get_stats()
        assert stats["grep_entries"] == ContextCache.DEFAULT_MAX_GREP_ENTRIES

    def test_access_count_updates(self, cache):
        """Test that access count is tracked."""
        cache.set_file_content("/test.py", "content")
        cache.get_file_content("/test.py")
        cache.get_file_content("/test.py")

        # Verify cache entry exists and was accessed
        content = cache.get_file_content("/test.py")
        assert content == "content"


class TestCacheHitRate:
    """Test cache hit rate tracking."""

    def test_record_cache_access_hit(self):
        """Test recording cache hit."""
        record_cache_access(True)
        # Just verify it doesn't raise

    def test_record_cache_access_miss(self):
        """Test recording cache miss."""
        record_cache_access(False)
        # Just verify it doesn't raise

    def test_get_cache_hit_rate_no_data(self):
        """Test hit rate with no data."""
        # Reset global counters before testing
        import mini_agent.utils.context_cache as ctx_cache

        ctx_cache._cache_hits = 0
        ctx_cache._cache_misses = 0

        rate = get_cache_hit_rate()
        assert rate == 0.0


class TestGetContextCache:
    """Test get_context_cache singleton."""

    def test_returns_singleton(self):
        """Test that get_context_cache returns the same instance."""
        cache1 = get_context_cache()
        cache2 = get_context_cache()

        assert cache1 is cache2

    def test_singleton_persists(self):
        """Test that the singleton persists across calls."""
        cache1 = get_context_cache()
        cache1.set_file_content("/test.py", "content")

        cache2 = get_context_cache()
        assert cache2.get_file_content("/test.py") == "content"


class TestContextCacheConstants:
    """Test ContextCache constants."""

    def test_default_ttl(self):
        """Test DEFAULT_TTL constant."""
        assert ContextCache.DEFAULT_TTL == 300

    def test_max_entries(self):
        """Test MAX entries constants."""
        assert ContextCache.DEFAULT_MAX_FILE_ENTRIES == 200
        assert ContextCache.DEFAULT_MAX_TREE_ENTRIES == 20
        assert ContextCache.DEFAULT_MAX_GREP_ENTRIES == 50
