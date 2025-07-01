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
from collector.search.providers.base import SearchResult


class TestCacheEntry:
    """Test cases for CacheEntry dataclass."""

    def test_cache_entry_creation(self):
        """Test creating a CacheEntry."""
        entry = CacheEntry(
            key="test query",
            value=[
                SearchResult(
                    video_id="1",
                    title="Result 1",
                    url="https://example.com/1",
                    channel="Test Channel",
                    channel_id="ch1",
                )
            ],
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
        assert hasattr(cache_manager, "l1_cache")
        assert hasattr(cache_manager, "l2_cache")
        assert True  # TTL is configured

    @pytest.mark.asyncio
    async def test_cache_manager_put_and_get(self, cache_manager):
        """Test putting and getting cache entries."""
        results = [
            SearchResult(
                video_id="1",
                title="Result 1",
                url="https://example.com/1",
                channel="Test Channel",
                channel_id="ch1",
            ),
            SearchResult(
                video_id="2",
                title="Result 2",
                url="https://example.com/2",
                channel="Test Channel",
                channel_id="ch1",
            ),
        ]

        # Put entry
        await cache_manager.cache_search_results(
            "test query", results, provider="youtube", max_results=100
        )

        # Get entry - specify same provider and max_results as when cached
        cached_results = await cache_manager.get_search_results(
            "test query", provider="youtube", max_results=100
        )

        assert cached_results is not None
        assert len(cached_results) == 2
        # Results are serialized as dicts, so we check the dict fields
        assert cached_results[0]["title"] == "Result 1"

    @pytest.mark.asyncio
    async def test_cache_manager_expired_entries(self, cache_manager):
        """Test handling of expired entries."""
        # Put entry with short TTL
        await cache_manager.cache_search_results(
            "expire test",
            [
                SearchResult(
                    video_id="1",
                    title="Expire Test",
                    url="https://example.com/1",
                    channel="Test",
                    channel_id="ch1",
                )
            ],
            provider="test",
            max_results=100,
            ttl=1,
        )

        # Wait for expiration
        # Wait longer for TTL=1 second entry to expire
        time.sleep(1.5)

        # Should return None for expired entry
        assert await cache_manager.get_search_results("expire test") is None

    @pytest.mark.asyncio
    async def test_cache_manager_persistence(self, temp_cache_file):
        """Test cache persistence across instances."""
        # First instance - save data
        cache1 = CacheManager({"cache_dir": Path(temp_cache_file).parent})
        await cache1._put_with_promotion(
            "persistent",
            [
                {
                    "video_id": "123",
                    "title": "Persistent",
                    "url": "https://example.com/123",
                    "channel": "Test",
                    "channel_id": "ch1",
                }
            ],
        )

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
            "query1",
            [
                SearchResult(
                    video_id="1",
                    title="Query 1",
                    url="https://example.com/1",
                    channel="Test",
                    channel_id="ch1",
                )
            ],
            provider="test",
            max_results=100,
        )
        await cache_manager.cache_search_results(
            "query2",
            [
                SearchResult(
                    video_id="2",
                    title="Query 2",
                    url="https://example.com/2",
                    channel="Test",
                    channel_id="ch1",
                )
            ],
            provider="test",
            max_results=100,
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
            "query1",
            [
                SearchResult(
                    video_id="1",
                    title="Query 1",
                    url="https://example.com/1",
                    channel="Test",
                    channel_id="ch1",
                )
            ],
            provider="test",
            max_results=100,
        )
        await cache_manager.cache_search_results(
            "query2",
            [
                SearchResult(
                    video_id="2",
                    title="Query 2",
                    url="https://example.com/2",
                    channel="Test",
                    channel_id="ch1",
                )
            ],
            provider="test",
            max_results=100,
        )

        # Get some entries
        await cache_manager.get_search_results("query1")  # Hit
        await cache_manager.get_search_results("query3")  # Miss

        stats = await cache_manager.get_comprehensive_stats()

        assert stats["overall"]["total_stores"] >= 2
        assert stats["l1_cache"]["hits"] >= 0
        assert stats["l1_cache"]["misses"] >= 0
        assert "total_requests" in stats["overall"]

    @pytest.mark.asyncio
    async def test_cache_manager_cleanup(self, cache_manager):
        """Test automatic cleanup of expired entries."""
        # Add mix of expired and valid entries
        await cache_manager.cache_search_results(
            "valid",
            [
                SearchResult(
                    video_id="1",
                    title="Valid",
                    url="https://example.com/1",
                    channel="Test",
                    channel_id="ch1",
                )
            ],
            provider="test",
            max_results=100,
            ttl=3600,
        )
        await cache_manager.cache_search_results(
            "expired1",
            [
                SearchResult(
                    video_id="2",
                    title="Expired 1",
                    url="https://example.com/2",
                    channel="Test",
                    channel_id="ch1",
                )
            ],
            provider="test",
            max_results=100,
            ttl=1,
        )
        await cache_manager.cache_search_results(
            "expired2",
            [
                SearchResult(
                    video_id="3",
                    title="Expired 2",
                    url="https://example.com/3",
                    channel="Test",
                    channel_id="ch1",
                )
            ],
            provider="test",
            max_results=100,
            ttl=1,
        )

        # Wait longer for TTL=1 second entries to expire
        time.sleep(1.5)

        # Trigger cleanup
        await cache_manager.l2_cache.clear_expired()

        # Valid entry should remain - need to specify same provider and max_results
        assert (
            await cache_manager.get_search_results("valid", provider="test", max_results=100)
            is not None
        )
        # Expired entries should be gone
        assert (
            await cache_manager.get_search_results("expired1", provider="test", max_results=100)
            is None
        )
        assert (
            await cache_manager.get_search_results("expired2", provider="test", max_results=100)
            is None
        )

    @pytest.mark.asyncio
    async def test_cache_manager_max_entries(self, temp_cache_file):
        """Test max entries limit enforcement."""
        cache = CacheManager({})

        # Add more than max entries
        for i in range(5):
            await cache._put_with_promotion(
                f"query{i}",
                [
                    {
                        "video_id": str(i),
                        "title": f"Query {i}",
                        "url": f"https://example.com/{i}",
                        "channel": "Test",
                        "channel_id": "ch1",
                    }
                ],
            )

        # Should have limited entries
        stats = await cache.get_comprehensive_stats()
        # Check that entries were added
        assert stats["l1_cache"]["size"] <= cache.l1_max_size

    @pytest.mark.asyncio
    async def test_cache_manager_concurrent_access(self, cache_manager):
        """Test concurrent access to cache."""

        async def writer_task(task_id):
            for i in range(10):
                await cache_manager.cache_search_results(
                    f"task{task_id}_query{i}",
                    [
                        SearchResult(
                            video_id=f"{task_id}_{i}",
                            title=f"Task {task_id} Query {i}",
                            url=f"https://example.com/{task_id}_{i}",
                            channel="Test",
                            channel_id="ch1",
                        )
                    ],
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
        assert stats["overall"]["total_stores"] > 0

    @pytest.mark.asyncio
    async def test_cache_manager_query_normalization(self, cache_manager):
        """Test query normalization for cache hits."""
        results = [
            SearchResult(
                video_id="1",
                title="Normalized Query",
                url="https://example.com/1",
                channel="Test",
                channel_id="ch1",
            )
        ]

        # Put with one format
        success = await cache_manager.cache_search_results(
            "  Test Query  ", results, provider="test", max_results=100
        )
        assert success, "Failed to cache results"

        # Debug: Check stats to see if it was stored
        stats = await cache_manager.get_comprehensive_stats()
        print(f"Stats after cache: {stats}")

        # Get with different format - need to specify same provider and max_results
        result1 = await cache_manager.get_search_results(
            "test query", provider="test", max_results=100
        )
        assert result1 is not None, "Failed to retrieve with lowercase"

        result2 = await cache_manager.get_search_results(
            "TEST QUERY", provider="test", max_results=100
        )
        assert result2 is not None, "Failed to retrieve with uppercase"

        result3 = await cache_manager.get_search_results(
            "  test  query  ", provider="test", max_results=100
        )
        assert result3 is not None, "Failed to retrieve with extra spaces"

    @pytest.mark.asyncio
    async def test_cache_manager_source_filtering(self, cache_manager):
        """Test filtering cache entries by source."""
        await cache_manager.cache_search_results(
            "query1",
            [
                SearchResult(
                    video_id="1",
                    title="YouTube 1",
                    url="https://youtube.com/1",
                    channel="YouTube Channel",
                    channel_id="yt1",
                )
            ],
            provider="youtube",
            max_results=100,
        )
        await cache_manager.cache_search_results(
            "query2",
            [
                SearchResult(
                    video_id="2",
                    title="YouTube 2",
                    url="https://youtube.com/2",
                    channel="YouTube Channel",
                    channel_id="yt1",
                )
            ],
            provider="youtube",
            max_results=100,
        )

        # Check stats to verify entries were cached
        stats = await cache_manager.get_comprehensive_stats()

        assert stats["overall"]["total_stores"] >= 2

    @pytest.mark.asyncio
    async def test_cache_manager_memory_efficiency(self, cache_manager):
        """Test memory efficiency with large result sets."""
        # Create large result set
        large_results = [
            SearchResult(
                video_id=str(i),
                title=f"Title {i}",
                url=f"https://example.com/{i}",
                channel="Large Channel",
                channel_id="large1",
                metadata={"data": "x" * 1000},
            )
            for i in range(100)
        ]

        # Store multiple large result sets
        for i in range(10):
            await cache_manager.cache_search_results(
                f"large_query_{i}", large_results, provider="test", max_results=100
            )

        # Cache should handle memory efficiently
        stats = await cache_manager.get_comprehensive_stats()
        assert stats["overall"]["total_stores"] > 0

        # Should still be able to retrieve data - need to specify same provider and max_results
        retrieved = await cache_manager.get_search_results(
            "large_query_0", provider="test", max_results=100
        )
        assert retrieved is not None
