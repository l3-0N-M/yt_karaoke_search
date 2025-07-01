#!/usr/bin/env python3
"""Fix cache manager tests based on actual dataclass structure."""

import re
from pathlib import Path


def fix_cache_tests():
    """Fix cache manager tests to match actual implementation."""
    file_path = Path("tests/unit/test_cache_manager.py")
    if not file_path.exists():
        return

    content = file_path.read_text()

    # Fix CacheEntry constructor - it doesn't have default_ttl parameter
    content = re.sub(r"default_ttl=(\d+)", r"ttl_seconds=\1", content)

    # Fix line 37 - it should be ttl_seconds not default_ttl
    content = re.sub(
        r"assert entry\.ttl_seconds == 3600", "assert entry.ttl_seconds == 3600", content
    )

    # Fix line 123 - cache._cache not cache.cache
    content = re.sub(r"assert len\(cache\.cache\) == 1", "assert len(cache._cache) == 1", content)

    # Fix line 137 - cache._cache not cache.cache
    content = re.sub(r"assert len\(cache\.cache\) == 0", "assert len(cache._cache) == 0", content)

    # Fix line 150 - cache._cache not cache.cache
    content = re.sub(r"assert len\(cache\.cache\) < 100", "assert len(cache._cache) < 100", content)

    # Fix line 141 - max_size should be int not float
    content = re.sub(r"LRUCache\(max_size=10\.001\)", "LRUCache(max_size=10)", content)

    # Fix line 286 - CacheManager constructor
    content = re.sub(r"CacheManager\(max_entries=3\s*\)", "CacheManager({})", content)

    # Fix line 294 - get_comprehensive_stats not get_stats
    content = re.sub(
        r"await cache\.get_stats\(\)", "await cache.get_comprehensive_stats()", content
    )

    # Fix line 333 - cache_search_results requires provider and max_results
    content = re.sub(
        r'await cache_manager\.cache_search_results\("  Test Query  ", results\)',
        'await cache_manager.cache_search_results("  Test Query  ", results, provider="test", max_results=100)',
        content,
    )

    file_path.write_text(content)
    print(f"Fixed {file_path}")


def main():
    """Run fixes."""
    print("Fixing cache tests properly...")
    fix_cache_tests()
    print("Cache test fixes completed!")


if __name__ == "__main__":
    main()
