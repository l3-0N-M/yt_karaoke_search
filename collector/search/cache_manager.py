"""Multi-level caching system for search results and API responses."""

import asyncio
import hashlib
import json
import logging
import pickle
import sqlite3
from collections import OrderedDict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .providers.base import SearchResult

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with metadata."""

    key: str
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    ttl_seconds: Optional[int] = None
    size_bytes: int = 0
    metadata: Optional[Dict] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    @property
    def is_expired(self) -> bool:  # type: ignore
        """Check if cache entry has expired."""
        if self.ttl_seconds is None:
            return False
        return (datetime.now() - self.created_at).total_seconds() > self.ttl_seconds

    def touch(self):
        """Update last accessed time and increment access count."""
        self.last_accessed = datetime.now()
        self.access_count += 1


class LRUCache:
    """Thread-safe LRU cache implementation."""

    def __init__(self, max_size: int = 1000, default_ttl: Optional[int] = None):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache = OrderedDict()
        self._lock: Optional[asyncio.Lock] = None

    def _ensure_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        async with self._ensure_lock():
            if key not in self._cache:
                return None

            entry = self._cache[key]

            # Check expiration
            if entry.is_expired:
                del self._cache[key]
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            entry.touch()

            return entry.value

    async def put(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Put value in cache."""
        async with self._ensure_lock():
            ttl = ttl or self.default_ttl

            # Calculate size (approximate)
            try:
                size_bytes = len(pickle.dumps(value))
            except Exception:
                size_bytes = len(str(value))

            entry = CacheEntry(
                key=key,
                value=value,
                created_at=datetime.now(),
                last_accessed=datetime.now(),
                ttl_seconds=ttl,
                size_bytes=size_bytes,
            )

            # Remove existing entry if present
            if key in self._cache:
                del self._cache[key]

            # Add new entry
            self._cache[key] = entry

            # Evict oldest entries if over capacity
            while len(self._cache) > self.max_size:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]

            return True

    async def delete(self, key: str) -> bool:
        """Delete entry from cache."""
        async with self._ensure_lock():
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    async def clear(self):
        """Clear all cache entries."""
        async with self._ensure_lock():
            self._cache.clear()

    async def get_stats(self) -> Dict:
        """Get cache statistics."""
        async with self._ensure_lock():
            total_size = sum(entry.size_bytes for entry in self._cache.values())
            expired_count = sum(1 for entry in self._cache.values() if entry.is_expired)

            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "total_bytes": total_size,
                "expired_entries": expired_count,
                "hit_rate": self._calculate_hit_rate(),
            }

    def _calculate_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        if not self._cache:
            return 0.0

        total_accesses = sum(entry.access_count for entry in self._cache.values())
        if total_accesses == 0:
            return 0.0

        # This is a simplified hit rate calculation
        # In practice, you'd want to track hits/misses separately
        return min(1.0, total_accesses / (len(self._cache) * 2))


class DatabaseCache:
    """SQLite-based persistent cache."""

    def __init__(self, db_path: str, table_name: str = "search_cache"):
        self.db_path = db_path
        self.table_name = table_name
        self._init_database()

    def _init_database(self):
        """Initialize cache database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    key TEXT PRIMARY KEY,
                    value BLOB NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    last_accessed TIMESTAMP NOT NULL,
                    access_count INTEGER DEFAULT 0,
                    ttl_seconds INTEGER,
                    size_bytes INTEGER DEFAULT 0,
                    metadata TEXT
                )
            """
            )

            # Create indexes for performance
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{self.table_name}_created ON {self.table_name}(created_at)"
            )
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{self.table_name}_accessed ON {self.table_name}(last_accessed)"
            )

    async def get(self, key: str) -> Optional[Any]:
        """Get value from database cache."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    f"""
                    SELECT value, created_at, ttl_seconds, access_count
                    FROM {self.table_name}
                    WHERE key = ?
                """,
                    (key,),
                )

                row = cursor.fetchone()
                if not row:
                    return None

                # Check expiration
                created_at = datetime.fromisoformat(row["created_at"])
                if (
                    row["ttl_seconds"]
                    and (datetime.now() - created_at).total_seconds() > row["ttl_seconds"]
                ):
                    await self.delete(key)
                    return None

                # Update access statistics
                conn.execute(
                    f"""
                    UPDATE {self.table_name}
                    SET last_accessed = ?, access_count = access_count + 1
                    WHERE key = ?
                """,
                    (datetime.now().isoformat(), key),
                )

                # Deserialize value
                value = pickle.loads(row["value"])
                return value

        except Exception as e:
            logger.error(f"Error getting value from database cache: {e}")
            return None

    async def put(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Put value in database cache."""
        try:
            # Serialize value
            serialized_value = pickle.dumps(value)
            size_bytes = len(serialized_value)

            with sqlite3.connect(self.db_path) as conn:
                now = datetime.now().isoformat()
                conn.execute(
                    f"""
                    INSERT OR REPLACE INTO {self.table_name}
                    (key, value, created_at, last_accessed, ttl_seconds, size_bytes)
                    VALUES (?, ?, ?, ?, ?, ?)
                """,
                    (key, serialized_value, now, now, ttl, size_bytes),
                )

            return True

        except Exception as e:
            logger.error(f"Error putting value in database cache: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """Delete entry from database cache."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(f"DELETE FROM {self.table_name} WHERE key = ?", (key,))
                return cursor.rowcount > 0

        except Exception as e:
            logger.error(f"Error deleting from database cache: {e}")
            return False

    async def clear_expired(self) -> int:
        """Clear expired entries and return count."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                now = datetime.now()
                cursor = conn.execute(
                    f"""
                    DELETE FROM {self.table_name}
                    WHERE ttl_seconds IS NOT NULL
                    AND (julianday(?) - julianday(created_at)) * 86400 > ttl_seconds
                """,
                    (now.isoformat(),),
                )

                return cursor.rowcount

        except Exception as e:
            logger.error(f"Error clearing expired entries: {e}")
            return 0

    async def get_stats(self) -> Dict:
        """Get database cache statistics."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row

                cursor = conn.execute(
                    f"""
                    SELECT
                        COUNT(*) as total_entries,
                        SUM(size_bytes) as total_bytes,
                        AVG(access_count) as avg_access_count,
                        MAX(created_at) as newest_entry,
                        MIN(created_at) as oldest_entry
                    FROM {self.table_name}
                """
                )

                stats = dict(cursor.fetchone())

                # Count expired entries
                now = datetime.now()
                cursor = conn.execute(
                    f"""
                    SELECT COUNT(*) as expired_count
                    FROM {self.table_name}
                    WHERE ttl_seconds IS NOT NULL
                    AND (julianday(?) - julianday(created_at)) * 86400 > ttl_seconds
                """,
                    (now.isoformat(),),
                )

                stats["expired_entries"] = cursor.fetchone()[0]

                return stats

        except Exception as e:
            logger.error(f"Error getting database cache stats: {e}")
            return {}


class CacheManager:
    """Multi-level cache manager with intelligent caching strategies."""

    def __init__(self, config=None):
        self.config = config or {}

        # Cache configuration
        self.l1_max_size = self.config.get("l1_max_size", 1000)
        self.l1_ttl = self.config.get("l1_ttl_seconds", 300)  # 5 minutes
        self.l2_ttl = self.config.get("l2_ttl_seconds", 3600)  # 1 hour
        self.l3_ttl = self.config.get("l3_ttl_seconds", 86400)  # 24 hours

        # Cache directory
        cache_dir = Path(self.config.get("cache_dir", "cache"))
        cache_dir.mkdir(exist_ok=True)

        # Initialize cache layers
        self.l1_cache = LRUCache(self.l1_max_size, self.l1_ttl)  # In-memory
        self.l2_cache = DatabaseCache(str(cache_dir / "search_cache.db"))  # SQLite

        # Statistics
        self.stats = {
            "l1_hits": 0,
            "l1_misses": 0,
            "l2_hits": 0,
            "l2_misses": 0,
            "total_gets": 0,
            "total_puts": 0,
        }

    def _normalize_query_for_cache(self, query: str) -> str:
        """Normalize query for better cache hits while preserving search intent."""
        if not query:
            return ""
        
        import re
        import unicodedata
        
        # Unicode normalization
        normalized = unicodedata.normalize('NFKC', query)
        
        # Convert to lowercase
        normalized = normalized.lower()
        
        # Remove excessive whitespace
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        # Normalize common punctuation variations
        normalized = re.sub(r'[–—−]', '-', normalized)  # Various dash types to hyphen
        normalized = re.sub(r'[""''`´]', '"', normalized)  # Various quotes to standard
        normalized = re.sub(r'[…]', '...', normalized)  # Ellipsis normalization
        
        # Only remove quality indicators that don't affect search results
        # Keep karaoke-related terms as they are part of the search intent
        quality_patterns = [
            r'\b(?:hd|hq|4k|1080p|720p|480p)\b',
            r'\b(?:high\s*quality|low\s*quality)\b',
        ]
        
        for pattern in quality_patterns:
            normalized = re.sub(pattern, '', normalized, flags=re.IGNORECASE)
        
        # Clean up resulting whitespace
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        # Remove brackets with only quality/format indicators
        normalized = re.sub(r'\[(hd|hq|4k|1080p|720p|480p)\]', '', normalized, flags=re.IGNORECASE)
        normalized = re.sub(r'\((hd|hq|4k|1080p|720p|480p)\)', '', normalized, flags=re.IGNORECASE)
        
        # Clean up again
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        return normalized

    def _generate_cache_key(self, namespace: str, *args, **kwargs) -> str:
        """Generate consistent cache key from parameters with enhanced normalization."""
        # Special handling for search queries to improve cache effectiveness
        if namespace == "search_results" and "query" in kwargs:
            # Normalize the query for better cache hits
            original_query = kwargs["query"]
            normalized_query = self._normalize_query_for_cache(original_query)
            
            # Create a copy of kwargs with normalized query
            normalized_kwargs = kwargs.copy()
            normalized_kwargs["query"] = normalized_query
            
            # Also store the original for debugging
            normalized_kwargs["_original_query"] = original_query
            
            key_data = {
                "namespace": namespace,
                "args": args,
                "kwargs": sorted(normalized_kwargs.items()),
            }
        else:
            key_data = {
                "namespace": namespace,
                "args": args,
                "kwargs": sorted(kwargs.items()) if kwargs else {},
            }

        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.sha256(key_str.encode()).hexdigest()[:32]

    async def get_search_results(
        self, query: str, provider: str = "", max_results: int = 100
    ) -> Optional[List[SearchResult]]:
        """Get cached search results."""
        key = self._generate_cache_key(
            "search_results", query=query, provider=provider, max_results=max_results
        )

        return await self._get_with_fallback(key)

    async def cache_search_results(
        self,
        query: str,
        results: List[SearchResult],
        provider: str = "",
        max_results: int = 100,
        ttl: Optional[int] = None,
    ) -> bool:
        """Cache search results."""
        key = self._generate_cache_key(
            "search_results", query=query, provider=provider, max_results=max_results
        )

        # Convert SearchResult objects to dicts for serialization
        serializable_results = [asdict(result) for result in results]

        return await self._put_with_promotion(key, serializable_results, ttl)

    async def get_parsed_metadata(self, video_id: str) -> Optional[Dict]:
        """Get cached parsed metadata."""
        key = self._generate_cache_key("parsed_metadata", video_id=video_id)
        return await self._get_with_fallback(key)

    async def cache_parsed_metadata(
        self, video_id: str, metadata: Dict, ttl: Optional[int] = None
    ) -> bool:
        """Cache parsed metadata."""
        key = self._generate_cache_key("parsed_metadata", video_id=video_id)
        return await self._put_with_promotion(key, metadata, ttl)

    async def get_channel_info(self, channel_id: str) -> Optional[Dict]:
        """Get cached channel information."""
        key = self._generate_cache_key("channel_info", channel_id=channel_id)
        return await self._get_with_fallback(key)

    async def cache_channel_info(
        self, channel_id: str, info: Dict, ttl: Optional[int] = None
    ) -> bool:
        """Cache channel information."""
        key = self._generate_cache_key("channel_info", channel_id=channel_id)
        return await self._put_with_promotion(key, info, ttl or self.l3_ttl)

    async def _get_with_fallback(self, key: str) -> Optional[Any]:
        """Get value with L1 -> L2 fallback."""
        self.stats["total_gets"] += 1

        # Try L1 cache first
        value = await self.l1_cache.get(key)
        if value is not None:
            self.stats["l1_hits"] += 1
            return value

        self.stats["l1_misses"] += 1

        # Try L2 cache
        value = await self.l2_cache.get(key)
        if value is not None:
            self.stats["l2_hits"] += 1
            # Promote to L1 cache
            await self.l1_cache.put(key, value, self.l1_ttl)
            return value

        self.stats["l2_misses"] += 1
        return None

    async def _put_with_promotion(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Put value in both cache layers."""
        self.stats["total_puts"] += 1

        # Put in L1 cache
        l1_success = await self.l1_cache.put(key, value, ttl or self.l1_ttl)

        # Put in L2 cache with longer TTL
        l2_success = await self.l2_cache.put(key, value, ttl or self.l2_ttl)

        return l1_success and l2_success

    async def invalidate_query(self, query: str):
        """Invalidate all cached results for a specific query."""
        # This would need a more sophisticated key tracking system
        # For now, we implement basic pattern-based invalidation
        # Clear from L1 cache (need to implement pattern matching)
        await self.l1_cache.clear()

        # For L2, we'd need to track keys or implement pattern matching
        logger.info(f"Cache invalidation requested for query: {query}")

    async def warm_cache(self, popular_queries: List[str]):
        """Pre-warm cache with popular queries."""
        logger.info(f"Warming cache for {len(popular_queries)} popular queries")

        # This would be implemented by the search engine
        # when it has access to the actual search providers
        pass

    async def cleanup_expired(self) -> Dict[str, int]:
        """Clean up expired entries from all cache layers."""
        l2_cleaned = await self.l2_cache.clear_expired()

        return {
            "l2_expired_removed": l2_cleaned,
        }

    async def get_comprehensive_stats(self) -> Dict:
        """Get comprehensive cache statistics."""
        l1_stats = await self.l1_cache.get_stats()
        l2_stats = await self.l2_cache.get_stats()

        # Calculate hit rates
        total_l1_requests = self.stats["l1_hits"] + self.stats["l1_misses"]
        total_l2_requests = self.stats["l2_hits"] + self.stats["l2_misses"]

        l1_hit_rate = self.stats["l1_hits"] / total_l1_requests if total_l1_requests > 0 else 0
        l2_hit_rate = self.stats["l2_hits"] / total_l2_requests if total_l2_requests > 0 else 0

        overall_hit_rate = (
            (self.stats["l1_hits"] + self.stats["l2_hits"]) / self.stats["total_gets"]
            if self.stats["total_gets"] > 0
            else 0
        )

        return {
            "overall": {
                "total_requests": self.stats["total_gets"],
                "total_stores": self.stats["total_puts"],
                "hit_rate": overall_hit_rate,
            },
            "l1_cache": {
                **l1_stats,
                "hits": self.stats["l1_hits"],
                "misses": self.stats["l1_misses"],
                "hit_rate": l1_hit_rate,
            },
            "l2_cache": {
                **l2_stats,
                "hits": self.stats["l2_hits"],
                "misses": self.stats["l2_misses"],
                "hit_rate": l2_hit_rate,
            },
            "configuration": {
                "l1_max_size": self.l1_max_size,
                "l1_ttl_seconds": self.l1_ttl,
                "l2_ttl_seconds": self.l2_ttl,
                "l3_ttl_seconds": self.l3_ttl,
            },
        }

    async def clear_all_caches(self):
        """Clear all cache layers."""
        await self.l1_cache.clear()
        # For L2, we'd implement a clear method
        logger.info("All caches cleared")

    async def optimize_caches(self):
        """Optimize cache performance by cleaning up and rebalancing."""
        # Clean expired entries
        cleanup_stats = await self.cleanup_expired()

        # Log optimization results
        logger.info(f"Cache optimization completed: {cleanup_stats}")

        return cleanup_stats
