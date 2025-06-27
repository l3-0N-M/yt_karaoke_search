#!/usr/bin/env python3
"""Example script demonstrating the enhanced search capabilities."""

import asyncio
import logging
import sys
from pathlib import Path

# Add the collector directory to the path
sys.path.insert(0, str(Path(__file__).parent / "collector"))

from collector.config import load_config
from collector.db import DatabaseManager
from collector.enhanced_search import MultiStrategySearchEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def demonstrate_enhanced_search():
    """Demonstrate the enhanced search capabilities."""

    print("🎤 Enhanced Karaoke Search Engine Demo")
    print("=" * 50)

    try:
        # Load configuration
        config = load_config("config.yaml")

        # Initialize database
        db_manager = DatabaseManager(config.database)

        # Initialize enhanced search engine
        search_engine = MultiStrategySearchEngine(config.search, config.scraping, db_manager)

        print(f"✅ Initialized search engine with {len(search_engine.providers)} providers")

        # Demo queries to test
        demo_queries = [
            "Bohemian Rhapsody Queen",
            "Hotel California Eagles",
            "Sweet Caroline Neil Diamond",
            "Don't Stop Believin Journey",
            "Billie Jean Michael Jackson",
        ]

        print("\n🔍 Testing Multi-Strategy Search")
        print("-" * 30)

        for query in demo_queries:
            print(f"\nSearching for: '{query}'")

            try:
                # Perform enhanced search
                results = await search_engine.search_videos(
                    query, max_results=10, use_cache=True, enable_fallback=True
                )

                print(f"  📊 Found {len(results)} results")

                # Show top 3 results with scoring details
                for i, result in enumerate(results[:3], 1):
                    print(f"  {i}. {result.title[:60]}...")
                    print(f"     Channel: {result.channel}")
                    print(
                        f"     Scores: Relevance={result.relevance_score:.2f}, "
                        f"Final={getattr(result, 'final_score', 0.0):.2f}"
                    )

            except Exception as e:
                print(f"  ❌ Search failed: {e}")

        # Demonstrate fallback strategy search
        print("\n🔄 Testing Fallback Strategy Search")
        print("-" * 35)

        difficult_query = "My Heart Will Go On"
        print(f"\nTesting difficult query: '{difficult_query}'")

        try:
            results = await search_engine.search_with_fallback_strategies(
                difficult_query, max_results=5
            )

            print(f"  📊 Fallback strategies found {len(results)} results")

            for i, result in enumerate(results[:2], 1):
                print(f"  {i}. {result.title[:60]}...")
                print(f"     Provider: {result.provider}")

        except Exception as e:
            print(f"  ❌ Fallback search failed: {e}")

        # Show comprehensive statistics
        print("\n📈 Search Engine Statistics")
        print("-" * 25)

        try:
            stats = await search_engine.get_comprehensive_statistics()

            # Search engine stats
            engine_stats = stats.get("search_engine", {})
            print(f"  Total searches: {engine_stats.get('total_searches', 0)}")
            print(f"  Cache hits: {engine_stats.get('cache_hits', 0)}")
            print(f"  Fallback used: {engine_stats.get('fallback_used', 0)}")
            print(f"  Avg response time: {engine_stats.get('average_response_time', 0):.2f}s")

            # Provider stats
            print("  Provider usage:")
            for provider, usage in engine_stats.get("provider_usage", {}).items():
                print(f"    {provider}: {usage} searches")

            # Cache stats
            cache_stats = stats.get("cache", {})
            overall_cache = cache_stats.get("overall", {})
            print(f"  Cache hit rate: {overall_cache.get('hit_rate', 0):.1%}")

        except Exception as e:
            print(f"  ❌ Failed to get statistics: {e}")

        # Demonstrate cache warming
        print("\n🔥 Cache Warming Demo")
        print("-" * 20)

        popular_queries = [
            "Yesterday Beatles karaoke",
            "Sweet Child O Mine Guns N Roses",
            "Wonderwall Oasis",
        ]

        try:
            await search_engine.warm_cache_for_popular_queries(popular_queries)
            print(f"  ✅ Warmed cache for {len(popular_queries)} popular queries")

        except Exception as e:
            print(f"  ❌ Cache warming failed: {e}")

        # Performance optimization
        print("\n⚡ Performance Optimization")
        print("-" * 25)

        try:
            optimization_stats = await search_engine.optimize_performance()
            print(f"  ✅ Optimization completed: {optimization_stats}")

        except Exception as e:
            print(f"  ❌ Optimization failed: {e}")

        print("\n🎉 Demo completed successfully!")
        print("\nKey Features Demonstrated:")
        print("  ✓ Multi-provider search (YouTube, Bing, DuckDuckGo)")
        print("  ✓ Intelligent result ranking")
        print("  ✓ Multi-level caching")
        print("  ✓ Fuzzy matching capabilities")
        print("  ✓ Fallback search strategies")
        print("  ✓ Performance monitoring")
        print("  ✓ Cache warming and optimization")

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        logger.exception("Demo failed with exception")


async def test_individual_components():
    """Test individual components separately."""

    print("\n🔧 Component Testing")
    print("=" * 20)

    # Test fuzzy matcher
    print("\n1. Testing Fuzzy Matcher")
    try:
        from collector.search.fuzzy_matcher import FuzzyMatcher

        fuzzy_matcher = FuzzyMatcher()

        # Test similarity calculation
        similarity = fuzzy_matcher.calculate_similarity("Bohemian Rhapsody", "Bohemian Rapsody")
        print(f"   Similarity 'Bohemian Rhapsody' vs 'Bohemian Rapsody': {similarity:.2f}")

        # Test best match finding
        candidates = ["Queen", "The Beatles", "Led Zeppelin", "Pink Floyd"]
        best_match = fuzzy_matcher.find_best_match("Quen", candidates, "artist")

        if best_match:
            print(f"   Best match for 'Quen': {best_match.matched} (score: {best_match.score:.2f})")
        else:
            print("   No match found for 'Quen'")

        print("   ✅ Fuzzy matcher working")

    except Exception as e:
        print(f"   ❌ Fuzzy matcher failed: {e}")

    # Test result ranker
    print("\n2. Testing Result Ranker")
    try:
        from collector.search.providers.base import SearchResult
        from collector.search.result_ranker import ResultRanker

        ranker = ResultRanker()

        # Create test results
        test_results = [
            SearchResult(
                video_id="test1",
                url="https://youtube.com/watch?v=test1",
                title="Bohemian Rhapsody - Queen - Karaoke Version",
                channel="KaraokeChannel",
                channel_id="karaoke123",
                view_count=1000000,
                provider="youtube",
            ),
            SearchResult(
                video_id="test2",
                url="https://youtube.com/watch?v=test2",
                title="Queen Bohemian Rhapsody Instrumental",
                channel="InstrumentalMusic",
                channel_id="instr456",
                view_count=500000,
                provider="youtube",
            ),
        ]

        # Rank results
        ranked_results = ranker.rank_results(test_results, "Bohemian Rhapsody Queen")

        print(f"   Ranked {len(ranked_results)} test results")
        for i, ranked_result in enumerate(ranked_results, 1):
            print(f"   {i}. Final score: {ranked_result.final_score:.2f}")

        print("   ✅ Result ranker working")

    except Exception as e:
        print(f"   ❌ Result ranker failed: {e}")

    # Test cache manager
    print("\n3. Testing Cache Manager")
    try:
        from collector.search.cache_manager import CacheManager

        cache_manager = CacheManager()

        # Test basic caching
        test_data = {"test": "data", "number": 42}
        await cache_manager.cache_parsed_metadata("test_video", test_data)

        retrieved_data = await cache_manager.get_parsed_metadata("test_video")

        if retrieved_data == test_data:
            print("   ✅ Cache manager working")
        else:
            print("   ❌ Cache manager data mismatch")

        # Get cache stats
        stats = await cache_manager.get_comprehensive_stats()
        print(f"   Cache stats: {stats.get('overall', {}).get('total_stores', 0)} items stored")

    except Exception as e:
        print(f"   ❌ Cache manager failed: {e}")


if __name__ == "__main__":
    print("Starting Enhanced Karaoke Search Demo...")

    # Run the main demo
    asyncio.run(demonstrate_enhanced_search())

    # Run component tests
    asyncio.run(test_individual_components())

    print("\nDemo complete! 🎤🎵")
