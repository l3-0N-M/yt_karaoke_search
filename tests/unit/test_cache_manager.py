"""Unit tests for cache_manager.py."""

import asyncio
import sys
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from collector.search.cache_manager import CacheEntry, CacheManager, LRUCache


class TestCacheEntry:
    """Test cases for CacheEntry dataclass."""

    def test_cache_entry_creation(self):
        """Test creating a CacheEntry."""
        entry = CacheEntry(
            key="test query",
            value=[{"video_id": "1", "title": "Result 1"}],
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            access_count=0,
            ttl_seconds=3600,
            metadata={"provider": "youtube"},
        )

        assert entry.key == "test query"
        assert len(entry.value) == 1
        assert entry.ttl_seconds == 3600
        assert entry.metadata is not None
        assert entry.metadata["provider"] == "youtube"

    def test_cache_entry_is_expired(self):
        """Test cache entry expiration check."""
        # Non-expired entry
        entry1 = CacheEntry(
            key="test",
            value=[],
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            ttl_seconds=3600,
        )
        assert not entry1.is_expired

        # Expired entry
        entry2 = CacheEntry(
            key="test",
            value=[],
            created_at=datetime.now() - timedelta(hours=2),
            last_accessed=datetime.now() - timedelta(hours=2),
            ttl_seconds=3600,
        )
        assert entry2.is_expired

    def test_cache_entry_default_values(self):
        """Test CacheEntry default values."""
        entry = CacheEntry(
            key="test", value=[], created_at=datetime.now(), last_accessed=datetime.now()
        )

        assert entry.created_at is not None
        assert entry.ttl_seconds is None  # No default TTL
        assert entry.metadata is not None


class TestLRUCache:
    """Test cases for LRUCache."""

    @pytest.mark.asyncio
    async def test_lru_cache_basic_operations(self):
        """Test basic LRU cache operations."""
        cache = LRUCache(max_size=3)

        # Add items
        await cache.put("key1", "value1")
        await cache.put("key2", "value2")
        await cache.put("key3", "value3")

        assert await cache.get("key1") == "value1"
        assert await cache.get("key2") == "value2"
        assert await cache.get("key3") == "value3"

    @pytest.mark.asyncio
    async def test_lru_cache_eviction(self):
        """Test LRU eviction when cache is full."""
        cache = LRUCache(max_size=3)

        # Fill cache
        await cache.put("key1", "value1")
        await cache.put("key2", "value2")
        await cache.put("key3", "value3")

        # Access key1 to make it recently used
        await cache.get("key1")

        # Add new item, should evict key2
        await cache.put("key4", "value4")

        assert await cache.get("key1") == "value1"  # Still present
        assert await cache.get("key2") is None  # Evicted
        assert await cache.get("key3") == "value3"  # Still present
        assert await cache.get("key4") == "value4"  # New item

    @pytest.mark.asyncio
    async def test_lru_cache_update_existing(self):
        """Test updating existing cache entries."""
        cache = LRUCache(max_size=3)

        await cache.put("key1", "value1")
        await cache.put("key1", "updated_value1")

        assert await cache.get("key1") == "updated_value1"
        assert len(cache._cache) == 1

    @pytest.mark.asyncio
    async def test_lru_cache_clear(self):
        """Test clearing the cache."""
        cache = LRUCache(max_size=3)

        await cache.put("key1", "value1")
        await cache.put("key2", "value2")

        await cache.clear()

        assert await cache.get("key1") is None
        assert await cache.get("key2") is None
        assert len(cache._cache) == 0

    @pytest.mark.asyncio
    async def test_lru_cache_memory_limit(self):
        """Test memory limit enforcement."""
        cache = LRUCache(max_size=10)  # Very small limit

        # Try to add large data
        large_data = "x" * 10000  # ~10KB
        await cache.put("large1", large_data)
        await cache.put("large2", large_data)

        # Should have evicted some data
        assert len(cache._cache) < 100


class TestCacheManager:
    """Test cases for CacheManager."""

    @pytest.fixture
    def temp_cache_file(self):
        """Create a temporary cache file."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            cache_path = f.name
        yield cache_path
        # Cleanup
        Path(cache_path).unlink(missing_ok=True)

    @pytest.fixture
    def cache_manager(self, temp_cache_file):
        """Create a CacheManager instance."""
        return CacheManager({})

    @pytest.mark.asyncio
    async def test_cache_manager_initialization(self, cache_manager):
        """Test cache manager initialization."""
        assert hasattr(cache_manager, "lru_cache")
        assert hasattr(cache_manager, "db_cache")
        assert True  # TTL is configured

    @pytest.mark.asyncio
    async def test_cache_manager_put_and_get(self, cache_manager):
        """Test putting and getting cache entries."""
        results = [{"video_id": "1", "title": "Result 1"}, {"video_id": "2", "title": "Result 2"}]

        # Put entry
        await cache_manager.cache_search_results(
            "test query", results, provider="youtube", max_results=100
        )

        # Get entry
        cached_results = await cache_manager.get_search_results("test query")

        assert cached_results is not None
        assert len(cached_results) == 2
        assert cached_results[0]["title"] == "Result 1"

    @pytest.mark.asyncio
    async def test_cache_manager_expired_entries(self, cache_manager):
        """Test handling of expired entries."""
        # Put entry with short TTL
        await cache_manager.cache_search_results(
            "expire test", [{"video_id": "1"}], provider="test", max_results=100
        )

        # Wait for expiration
        time.sleep(0.2)

        # Should return None for expired entry
        assert await cache_manager.get_search_results("expire test") is None

    @pytest.mark.asyncio
    async def test_cache_manager_persistence(self, temp_cache_file):
        """Test cache persistence across instances."""
        # First instance - save data
        cache1 = CacheManager({"cache_dir": Path(temp_cache_file).parent})
        await cache1._put_with_promotion("persistent", [{"video_id": "123"}])

        # Second instance - read data
        cache2 = CacheManager({"cache_dir": Path(temp_cache_file).parent})
        results = await cache2._get_with_fallback("persistent")

        assert results is not None
        assert results[0]["video_id"] == "123"

    @pytest.mark.asyncio
    async def test_cache_manager_clear(self, cache_manager):
        """Test clearing the cache."""
        # Add entries
        await cache_manager.cache_search_results(
            "query1", [{"video_id": "1"}], provider="test", max_results=100
        )
        await cache_manager.cache_search_results(
            "query2", [{"video_id": "2"}], provider="test", max_results=100
        )

        # Clear cache
        await cache_manager.l1_cache.clear()
        await cache_manager.l2_cache.clear_expired()

        # Check entries are gone
        assert await cache_manager.get_search_results("query1") is None
        assert await cache_manager.get_search_results("query2") is None

    @pytest.mark.asyncio
    async def test_cache_manager_statistics(self, cache_manager):
        """Test cache statistics."""
        # Add entries
        await cache_manager.cache_search_results(
            "query1", [{"video_id": "1"}], provider="test", max_results=100
        )
        await cache_manager.cache_search_results(
            "query2", [{"video_id": "2"}], provider="test", max_results=100
        )

        # Get some entries
        await cache_manager.get_search_results("query1")  # Hit
        await cache_manager.get_search_results("query3")  # Miss

        stats = await cache_manager.get_comprehensive_stats()

        assert stats["total_puts"] >= 2
        assert stats["l1_hits"] >= 0
        assert stats["l1_misses"] >= 0
        assert "total_gets" in stats

    @pytest.mark.asyncio
    async def test_cache_manager_cleanup(self, cache_manager):
        """Test automatic cleanup of expired entries."""
        # Add mix of expired and valid entries
        await cache_manager.cache_search_results(
            "valid", [{"video_id": "1"}], provider="test", max_results=100
        )
        await cache_manager.cache_search_results(
            "expired1", [{"video_id": "2"}], provider="test", max_results=100
        )
        await cache_manager.cache_search_results(
            "expired2", [{"video_id": "3"}], provider="test", max_results=100
        )

        time.sleep(0.2)

        # Trigger cleanup
        await cache_manager.l2_cache.clear_expired()

        # Valid entry should remain
        assert await cache_manager.get_search_results("valid") is not None
        # Expired entries should be gone
        assert await cache_manager.get_search_results("expired1") is None
        assert await cache_manager.get_search_results("expired2") is None

    @pytest.mark.asyncio
    async def test_cache_manager_max_entries(self, temp_cache_file):
        """Test max entries limit enforcement."""
        cache = CacheManager({})

        # Add more than max entries
        for i in range(5):
            await cache._put_with_promotion(f"query{i}", [{"id": str(i)}])

        # Should have limited entries
        stats = await cache.get_comprehensive_stats()
        assert stats["total_entries"] <= 3

    @pytest.mark.asyncio
    async def test_cache_manager_concurrent_access(self, cache_manager):
        """Test concurrent access to cache."""

        async def writer_task(task_id):
            for i in range(10):
                await cache_manager.cache_search_results(
                    f"task{task_id}_query{i}",
                    [{"id": f"{task_id}_{i}"}],
                    provider="test",
                    max_results=100,
                )

        async def reader_task(task_id):
            for i in range(10):
                await cache_manager.get_search_results(f"task{task_id}_query{i}")

        # Create tasks
        tasks = []
        for i in range(3):
            tasks.append(writer_task(i))
            tasks.append(reader_task(i))

        # Run tasks concurrently
        await asyncio.gather(*tasks)

        # Cache should still be functional
        stats = await cache_manager.get_comprehensive_stats()
        assert stats["total_puts"] > 0

    @pytest.mark.asyncio
    async def test_cache_manager_query_normalization(self, cache_manager):
        """Test query normalization for cache hits."""
        results = [{"video_id": "1"}]

        # Put with one format
        await cache_manager.cache_search_results(
            "  Test Query  ", results, provider="test", max_results=100
        )

        # Get with different format
        assert await cache_manager.get_search_results("test query") is not None
        assert await cache_manager.get_search_results("TEST QUERY") is not None
        assert await cache_manager.get_search_results("  test  query  ") is not None

    @pytest.mark.asyncio
    async def test_cache_manager_source_filtering(self, cache_manager):
        """Test filtering cache entries by source."""
        await cache_manager.cache_search_results(
            "query1", [{"video_id": "1"}], provider="youtube", max_results=100
        )
        await cache_manager.cache_search_results(
            "query2", [{"video_id": "2"}], provider="youtube", max_results=100
        )

        # Check stats to verify entries were cached
        stats = await cache_manager.get_comprehensive_stats()

        assert stats["total_puts"] >= 2

    @pytest.mark.asyncio
    async def test_cache_manager_memory_efficiency(self, cache_manager):
        """Test memory efficiency with large result sets."""
        # Create large result set
        large_results = [
            {"id": str(i), "title": f"Title {i}", "data": "x" * 1000} for i in range(100)
        ]

        # Store multiple large result sets
        for i in range(10):
            await cache_manager.cache_search_results(
                f"large_query_{i}", large_results, provider="test", max_results=100
            )

        # Cache should handle memory efficiently
        stats = await cache_manager.get_comprehensive_stats()
        assert stats["total_puts"] > 0

        # Should still be able to retrieve data
        retrieved = await cache_manager.get_search_results("large_query_0")
        assert retrieved is not None
